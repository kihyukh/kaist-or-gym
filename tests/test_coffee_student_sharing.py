"""Public class discovery and saving to the intended instructor classroom."""

import base64
import json
import shutil
import subprocess
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from kaist_rl_lab.apps.coffee_browser import CONTROLLER_JAVASCRIPT
from kaist_rl_lab.apps.coffee_browser_runtime import BrowserRuntime
from kaist_rl_lab.apps.coffee_web import create_app
from kaist_rl_lab.apps.coffee_web_assets import STUDENT_BOOTSTRAP
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT


@pytest.fixture
def classroom_clients(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("Student site")
    (static / "instructor.html").write_text("Instructor site")
    password = "sharing-test-password-only"
    app = create_app(
        data_dir=tmp_path / "data", static_dir=static, build_assets=False,
        password=password, session_secret="sharing-test-signing-secret-only",
        public_base_url="https://coffee.example",
    )
    headers = {"Origin": "https://coffee.example"}
    with TestClient(app, base_url="https://coffee.example", headers=headers) as instructor, \
            TestClient(app, base_url="https://coffee.example", headers=headers) as student:
        assert instructor.post("/api/instructor/login", json={"password": password}).status_code == 200
        yield instructor, student


def create_class(instructor, name, required=True):
    response = instructor.post("/api/instructor/sessions", json={
        "name": name, "participant_required": required,
    })
    assert response.status_code == 200, response.text
    classroom = response.json()
    token = parse_qs(urlsplit(classroom["join_url"]).query)["class"][0]
    return classroom, token


def close_class(instructor, classroom):
    response = instructor.post(f"/api/instructor/sessions/{classroom['id']}/close")
    assert response.status_code == 200
    assert response.json()["open"] is False


def attempt(participant="20260001"):
    runtime = BrowserRuntime(seed=17)
    try:
        runtime.session.set_motor(0, 1)
        runtime.session.toggle_pause()
        for _ in range(3):
            runtime.session.advance()
        return runtime.session.save_demonstration(participant).read_bytes()
    finally:
        runtime.session.close()


def save_archive(student, token, data):
    return student.post("/api/submissions", params={"class": token}, content=data,
                        headers={"Content-Type": "application/octet-stream"})


def test_default_discovery_requires_exactly_one_open_class(classroom_clients):
    instructor, student = classroom_clients
    response = student.get("/api/session/default")
    assert response.status_code == 200
    assert response.json() == {"session": None, "reason": "no_open_class"}
    assert response.headers["cache-control"] == "no-store"
    first, first_token = create_class(instructor, "First class")
    second, second_token = create_class(instructor, "Second class")
    response = student.get("/api/session/default")
    assert response.json() == {"session": None, "reason": "multiple_open_classes"}
    assert first_token not in response.text and second_token not in response.text
    close_class(instructor, first)
    assert student.get("/api/session/default").json() == {"session": {
        "id": second["id"], "name": "Second class", "open": True,
        "participant_required": True, "join_token": second_token,
    }}
    close_class(instructor, second)
    assert student.get("/api/session/default").json() == {
        "session": None, "reason": "no_open_class",
    }


def test_default_shared_archive_stays_in_resolved_class_and_receipt_is_idempotent(classroom_clients):
    instructor, student = classroom_clients
    classroom, token = create_class(instructor, "Current seminar")
    resolved = student.get("/api/session/default").json()["session"]
    assert resolved["id"] == classroom["id"] and resolved["join_token"] == token
    explicit = student.get("/api/session", params={"class": token})
    assert explicit.json() == {"id": classroom["id"], "name": "Current seminar",
                               "open": True, "participant_required": True}
    # Another class opening after page load must never redirect the saved attempt.
    other, _ = create_class(instructor, "Next seminar")
    archive = attempt()
    response = save_archive(student, resolved["join_token"], archive)
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt["status"] == "saved" and receipt["duplicate"] is False
    assert save_archive(student, token, archive).json() == {**receipt, "duplicate": True}
    rows = instructor.get("/api/instructor/submissions", params={"session": classroom["id"]}).json()
    assert len(rows) == 1
    assert rows[0]["episode_id"] == receipt["episode_id"]
    assert rows[0]["participant"] == "20260001" and rows[0]["steps"] == 3
    assert instructor.get("/api/instructor/submissions", params={"session": other["id"]}).json() == []
    endpoint = f"/api/instructor/submissions/{receipt['episode_id']}"
    assert instructor.get(endpoint + "/download").content == archive
    for url in ["/api/instructor/sessions", "/api/instructor/submissions?session=" + classroom["id"],
                endpoint + "/download", endpoint + "/replay", endpoint + "/replay-stream"]:
        assert student.get(url).status_code == 401
    close_class(instructor, classroom)
    assert save_archive(student, token, archive).json() == {**receipt, "duplicate": True}


def test_closed_or_invalid_explicit_link_never_saves_to_the_current_default(classroom_clients):
    instructor, student = classroom_clients
    old, old_token = create_class(instructor, "Closed class")
    close_class(instructor, old)
    current, current_token = create_class(instructor, "Open class")
    assert student.get("/api/session/default").json()["session"]["join_token"] == current_token
    explicit = student.get("/api/session", params={"class": old_token})
    assert explicit.json()["open"] is False
    assert "join_token" not in explicit.json()
    assert student.get("/api/session", params={"class": "invalid-token"}).status_code == 404
    archive = attempt()
    closed = save_archive(student, old_token, archive)
    assert closed.status_code == 409 and "closed" in closed.json()["detail"].lower()
    assert save_archive(student, "invalid-token", archive).status_code == 409
    assert student.post("/api/submissions", content=archive,
                        headers={"Content-Type": "application/octet-stream"}).status_code == 422
    for classroom in [old, current]:
        assert instructor.get("/api/instructor/submissions", params={"session": classroom["id"]}).json() == []


@pytest.mark.parametrize("required", [True, False])
def test_default_class_preserves_the_instructors_participant_requirement(classroom_clients, required):
    instructor, student = classroom_clients
    _, token = create_class(instructor, "Participant policy", required)
    resolved = student.get("/api/session/default").json()["session"]
    assert resolved["participant_required"] is required
    response = save_archive(student, token, attempt(""))
    assert response.status_code == (409 if required else 200)


BOOTSTRAP_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const input=JSON.parse(fs.readFileSync(0,'utf8')),requests=[],replacements=[],nodes=new Map();
function node(selector){
  if(!nodes.has(selector))nodes.set(selector,{textContent:'',hidden:false,required:false,disabled:false,
    setAttribute(name,value){this[name]=value;},querySelector:node});
  return nodes.get(selector);
}
const location=new URL(input.url);
const history={replaceState(state,title,url){replacements.push(String(url));}};
const context=vm.createContext({URL,URLSearchParams,AbortSignal,location,history,
  window:{history,location},document:{querySelector:node},
  fetch:async(path)=>{
    requests.push(String(path));const response=input.responses[requests.length-1];
    assert.ok(response,'Unexpected class lookup '+path);assert.equal(path,response.path);
    if(response.error){const error=new Error(response.error);error.name=response.error_name||'Error';throw error;}
    return {ok:(response.status??200)<400,status:response.status??200,json:async()=>response.body};
  },console});
(async()=>{
  await vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
  console.log(JSON.stringify({config:context.captured??null,requests,replacements,
    label:node('#class-name').textContent,status:node('.coffee-status').textContent,
    participantRequired:node('.coffee-participant').required,
    saveNote:node('.save-note').textContent,heading:node('#save-heading').textContent}));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""


def run_bootstrap(tmp_path, url, responses):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for student routing regression tests")
    assert CANVAS_JAVASCRIPT in STUDENT_BOOTSTRAP
    assert CONTROLLER_JAVASCRIPT in STUDENT_BOOTSTRAP
    source = STUDENT_BOOTSTRAP.replace(CANVAS_JAVASCRIPT, "").replace(
        CONTROLLER_JAVASCRIPT, "globalThis.captured=JSON.parse(JSON.stringify(props.value));",
    )
    script = tmp_path / "bootstrap.js"
    script.write_text(source)
    harness = tmp_path / "bootstrap.cjs"
    harness.write_text(BOOTSTRAP_HARNESS)
    result = subprocess.run([node, str(harness), str(script)],
                            input=json.dumps({"url": url, "responses": responses}),
                            text=True, capture_output=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def public_class(**extra):
    return {"id": "current", "name": "Today’s seminar", "open": True,
            "participant_required": True, **extra}


def test_bare_home_page_pins_discovered_class_and_configures_shared_save(tmp_path):
    url = "https://coffee.example/?source=lecture#play"
    result = run_bootstrap(tmp_path, url, [{"path": "/api/session/default", "body": {
        "session": public_class(join_token="class_token-2026"),
    }}])
    assert result["config"]["collecting"] is True
    assert result["config"]["upload_url"] == "/api/submissions?class=class_token-2026"
    assert result["config"]["save_label"] == "Save & share"
    assert result["config"]["saved_label"] == "Shared ✓"
    assert result["participantRequired"] is True
    assert result["label"] == "Today’s seminar"
    assert len(result["replacements"]) == 1
    pinned = urlsplit(result["replacements"][0])
    assert parse_qs(pinned.query)["class"] == ["class_token-2026"]
    assert parse_qs(pinned.query)["source"] == ["lecture"]
    assert pinned.fragment == "play"


@pytest.mark.parametrize("reason", ["no_open_class", "multiple_open_classes"])
def test_no_unique_default_has_no_silent_upload_destination(tmp_path, reason):
    result = run_bootstrap(tmp_path, "https://coffee.example/", [{
        "path": "/api/session/default", "body": {"session": None, "reason": reason},
    }])
    assert result["replacements"] == []
    assert result["participantRequired"] is False
    if reason == "multiple_open_classes":
        assert result["config"] is None
        assert "QR code" in result["label"]
    else:
        assert result["config"]["collecting"] is False
        assert "upload_url" not in result["config"]
        assert "practice" in result["label"].lower()


def test_explicit_practice_does_not_discover_or_share_with_an_open_class(tmp_path):
    result = run_bootstrap(tmp_path, "https://coffee.example/?practice=1", [])
    assert result["requests"] == [] and result["replacements"] == []
    assert result["config"]["collecting"] is False
    assert "upload_url" not in result["config"]
    assert "practice" in result["label"].lower()
    assert result["config"]["save_label"] == "Save trajectory"


@pytest.mark.parametrize("opened", [True, False])
def test_explicit_class_link_does_not_fall_back_to_a_default_class(tmp_path, opened):
    result = run_bootstrap(tmp_path, "https://coffee.example/?class=old_token", [{
        "path": "/api/session?class=old_token", "body": public_class(open=opened),
    }])
    assert result["requests"] == ["/api/session?class=old_token"]
    assert result["replacements"] == []
    assert result["config"]["collecting"] is opened
    if opened:
        assert result["config"]["upload_url"] == "/api/submissions?class=old_token"
    else:
        assert "upload_url" not in result["config"]
        assert "closed" in result["label"].lower()


def test_explicit_class_takes_priority_over_practice_and_empty_class_is_invalid(tmp_path):
    result = run_bootstrap(tmp_path, "https://coffee.example/?class=chosen&practice=1", [{
        "path": "/api/session?class=chosen", "body": public_class(),
    }])
    assert result["config"]["collecting"] is True
    assert result["config"]["upload_url"] == "/api/submissions?class=chosen"
    result = run_bootstrap(tmp_path, "https://coffee.example/?class=&practice=1", [])
    assert result["config"] is None
    assert result["requests"] == [] and result["replacements"] == []
    assert "not valid" in result["label"]


@pytest.mark.parametrize("body", [None, {}, {"session": {}}, {"session": public_class()},
                                  {"session": public_class(join_token="")}])
def test_malformed_default_response_never_starts_or_pins_a_class(tmp_path, body):
    result = run_bootstrap(tmp_path, "https://coffee.example/", [{
        "path": "/api/session/default", "body": body,
    }])
    assert result["config"] is None
    assert result["replacements"] == []
    assert result["status"]


@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("failure", ["http", "network", "timeout"])
def test_failed_class_lookup_does_not_start_an_unshared_demo(tmp_path, explicit, failure):
    path = "/api/session?class=missing" if explicit else "/api/session/default"
    response = {"path": path, "status": 404, "body": {"detail": "Not found"}}
    if failure != "http":
        response = {"path": path, "error": "Network unavailable",
                    "error_name": "TimeoutError" if failure == "timeout" else "TypeError"}
    result = run_bootstrap(tmp_path, "https://coffee.example/" + ("?class=missing" if explicit else ""), [response])
    assert result["config"] is None
    assert result["requests"] == [path] and result["replacements"] == []
    assert result["status"]


def test_main_page_save_automatically_posts_exact_recording_and_waits_for_shared_receipt(
    tmp_path, classroom_clients,
):
    from test_coffee_web_browser import NODE_HARNESS

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for automatic shared-save regression tests")
    instructor, student = classroom_clients
    classroom, token = create_class(instructor, "Default classroom")
    route = run_bootstrap(tmp_path, "https://coffee.example/", [{
        "path": "/api/session/default", "body": student.get("/api/session/default").json(),
    }])
    runtime = BrowserRuntime(seed=17)
    try:
        initial = json.loads(runtime.snapshot())
        generation = initial["snapshot"]["playback"]["generation"]
        for sequence, paused in [(1, True), (2, False)]:
            runtime.dispatch(json.dumps({
                "kind": "motor" if paused else "pause", "sequence": sequence,
                "motors": [1, 0, 0, 0, 0, 0], "paused": paused, "generation": generation,
            }))
        moving = json.loads(runtime.dispatch('{"kind":"tick"}'))
        saved = json.loads(runtime.dispatch('{"kind":"save","participant":"20260001"}'))
    finally:
        runtime.session.close()
    data = base64.b64decode(saved["archive"])
    response = save_archive(student, token, data)
    assert response.status_code == 200, response.text
    fixtures = {"initial": initial, "moving": moving, "saved": saved,
                "config": route["config"], "receipt": response.json()}
    source = tmp_path / "controller.js"
    source.write_text(CANVAS_JAVASCRIPT + CONTROLLER_JAVASCRIPT)
    # Reuse the browser's worker/network harness, with the actual main-page
    # configuration. Keep the real controller and server receipt unchanged.
    harness = NODE_HARNESS.replace(
        "vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);",
        "context.props.value=fixtures.config;\n"
        "vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);",
    )
    runner = tmp_path / "shared-save.cjs"
    runner.write_text(harness + r"""
(async()=>{
assert.equal(node('[data-command="save"]').textContent,'Save & share');
begin();node('.coffee-participant').value='20260001';clickSave();
assert.equal(sent.at(-1).kind,'save');assert.equal(requests.length,0);
receive(fixtures.saved);assert.equal(requests.length,1,'Saving must automatically upload, without another click');
assert.equal(requests[0].url,fixtures.config.upload_url);
assert.equal(requests[0].options.method,'POST');
assert.equal(requests[0].options.headers['Content-Type'],'application/octet-stream');
assert.deepEqual(Array.from(requests[0].options.body),Array.from(Buffer.from(fixtures.saved.archive,'base64')));
assert.notEqual(node('[data-command="save"]').textContent,'Shared ✓','Sharing needs a server receipt');
assert.equal(node('[data-command="save"]').disabled,true);
reply(0,true,fixtures.receipt);await settle();
assert.equal(node('[data-command="save"]').textContent,'Shared ✓');
assert.ok(node('.coffee-submission').textContent.includes(fixtures.receipt.receipt));
assert.equal(node('.coffee-download').hidden,false);assert.equal(read('dirty'),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")
    result = subprocess.run([node, str(runner), str(source)], input=json.dumps(fixtures),
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = instructor.get("/api/instructor/submissions", params={"session": classroom["id"]}).json()
    assert [row["episode_id"] for row in rows] == [response.json()["episode_id"]]
    assert instructor.get(f"/api/instructor/submissions/{rows[0]['episode_id']}/download").content == data
