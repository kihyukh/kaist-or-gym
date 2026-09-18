"""Prepare the persistent mount before dropping container privileges."""

import os
import pwd
import sys
from pathlib import Path


def launch(argv: list[str]) -> None:
    """Run the server with private files and a writable persistent directory."""
    os.umask(0o077)
    data_dir = Path(os.environ.get("COFFEE_DATA_DIR", "/data"))
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.geteuid() == 0:
        account = pwd.getpwnam("coffee")
        # A runtime mount hides the directory ownership set in the image build.
        # Files within the directory are always created by the coffee account.
        os.chown(data_dir, account.pw_uid, account.pw_gid)
        data_dir.chmod(0o700)
        os.setgroups([])
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
        os.environ.update(HOME=account.pw_dir, USER=account.pw_name)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    launch(sys.argv[1:] or ["coffee-pouring-web"])
