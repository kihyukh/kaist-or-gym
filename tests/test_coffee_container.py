"""Ensure a fresh volume is prepared before container privileges are dropped."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def entrypoint():
    path = Path(__file__).parents[1] / "deploy/coffee-web/entrypoint.py"
    spec = importlib.util.spec_from_file_location("coffee_web_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_prepares_mount_and_drops_privileges_before_start(entrypoint, monkeypatch, tmp_path):
    data = tmp_path / "data"
    calls = []
    monkeypatch.setenv("COFFEE_DATA_DIR", str(data))
    monkeypatch.setattr(entrypoint.os, "umask", lambda mask: calls.append(("umask", mask)))
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        entrypoint.pwd,
        "getpwnam",
        lambda name: SimpleNamespace(
            pw_name=name, pw_uid=10001, pw_gid=10001, pw_dir="/home/coffee"
        ),
    )
    for method in ("chown", "setgroups", "setgid", "setuid", "execvp"):
        monkeypatch.setattr(
            entrypoint.os, method, lambda *args, method=method: calls.append((method, *args))
        )
    # launch deliberately changes these values after dropping to the coffee user.
    monkeypatch.setenv("HOME", "/root")
    monkeypatch.setenv("USER", "root")

    entrypoint.launch(["coffee-pouring-web"])

    assert data.is_dir()
    assert data.stat().st_mode & 0o777 == 0o700
    assert calls == [
        ("umask", 0o077),
        ("chown", data, 10001, 10001),
        ("setgroups", []),
        ("setgid", 10001),
        ("setuid", 10001),
        ("execvp", "coffee-pouring-web", ["coffee-pouring-web"]),
    ]
    assert entrypoint.os.environ["HOME"] == "/home/coffee"
    assert entrypoint.os.environ["USER"] == "coffee"


def test_nonroot_container_uses_existing_access(entrypoint, monkeypatch, tmp_path):
    data = tmp_path / "data"
    calls = []
    monkeypatch.setenv("COFFEE_DATA_DIR", str(data))
    monkeypatch.setattr(entrypoint.os, "umask", lambda _: None)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 10001)
    for method in ("chown", "setgroups", "setgid", "setuid"):
        monkeypatch.setattr(entrypoint.os, method, lambda *args: pytest.fail("changed identity"))
    monkeypatch.setattr(entrypoint.os, "execvp", lambda *args: calls.append(args))

    entrypoint.launch(["coffee-pouring-web"])

    assert data.is_dir()
    assert calls == [("coffee-pouring-web", ["coffee-pouring-web"])]


def test_failed_privilege_drop_never_starts_server(entrypoint, monkeypatch, tmp_path):
    monkeypatch.setenv("COFFEE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(entrypoint.os, "umask", lambda _: None)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        entrypoint.pwd,
        "getpwnam",
        lambda _: SimpleNamespace(pw_uid=10001, pw_gid=10001),
    )
    monkeypatch.setattr(entrypoint.os, "chown", lambda *args: None)
    monkeypatch.setattr(entrypoint.os, "setgroups", lambda *args: None)
    monkeypatch.setattr(entrypoint.os, "setgid", lambda *args: None)

    def fail_to_drop(_):
        raise PermissionError("cannot drop privileges")

    monkeypatch.setattr(entrypoint.os, "setuid", fail_to_drop)
    monkeypatch.setattr(entrypoint.os, "execvp", lambda *args: pytest.fail("started as root"))
    with pytest.raises(PermissionError):
        entrypoint.launch(["coffee-pouring-web"])
