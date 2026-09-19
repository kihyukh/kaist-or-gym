"""Website/controller checks without loading the Pyodide CDN.

Node tests run with the normal test suite. The rendered website tests additionally
use Playwright and an installed Chromium/Chrome browser.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kaist_rl_lab.apps.coffee_browser import CONTROLLER_JAVASCRIPT
from kaist_rl_lab.apps.coffee_browser_runtime import BrowserRuntime
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT


@pytest.fixture(scope="module")
def recording():
    runtime = BrowserRuntime()
    try:
        initial = json.loads(runtime.snapshot())
        for sequence, paused in [(1, True), (2, False)]:
            runtime.dispatch(json.dumps({
                "kind": "motor" if paused else "pause", "sequence": sequence,
                "motors": [1, 0, 0, 0, 0, 0], "paused": paused,
                "generation": initial["snapshot"]["playback"]["generation"],
            }))
        moving = json.loads(runtime.dispatch('{"kind":"tick"}'))
        saved = json.loads(runtime.dispatch('{"kind":"save","participant":"20260001"}'))
        return {"initial": initial, "moving": moving, "saved": saved}
    finally:
        runtime.session.close()


NODE_HARNESS = r"""
const vm=require('node:vm'), fs=require('node:fs'), assert=require('node:assert/strict');
const fixtures=JSON.parse(fs.readFileSync(0,'utf8'));
const sent=[], requests=[], nodeListeners=new Map(), docListeners=new Map(), winListeners=new Map();
const nodes=new Map();
function node(key) {
  if(key==='.coffee-canvas')return null;
  if(!nodes.has(key)) {
    const command=key.match(/data-command="([^"]+)"/);
    const value={textContent:'', value:'', hidden:true, disabled:false,
      dataset:command ? {command:command[1]} : {},
      classList:{contains:()=>false}, closest(){return this;},
      addEventListener(type,fn){nodeListeners.set(key+':'+type,fn);},
      setCustomValidity(message){this.validationMessage=message;},
      reportValidity(){this.reported=true;}, focus(){this.focused=true;}};
    nodes.set(key,value);
  }
  return nodes.get(key);
}
const buttons=['pause','reset','stop','save'].map(kind=>node('[data-command="'+kind+'"]'));
const element={querySelector:node,
  querySelectorAll:selector=>selector==='[data-command]' ? buttons : [],
  addEventListener:(type,fn)=>nodeListeners.set('element:'+type,fn), contains:()=>true,
  isConnected:true};
const document={hidden:false,body:{},addEventListener:(type,fn)=>docListeners.set(type,fn),
  removeEventListener:(type)=>docListeners.delete(type)};
const window={addEventListener:(type,fn)=>winListeners.set(type,fn),
  removeEventListener:(type)=>winListeners.delete(type),confirm:()=>false};
const context=vm.createContext({element,document,window,
  props:{value:{worker_url:'/coffee-worker.js',bundle_url:'/coffee-bundle.zip',
    collecting:true,participant_required:true,upload_url:'/api/submissions?class=test-class'}},
  watch(){},trigger(){throw new Error('The website must use HTTP rather than a notebook bridge.');},
  Blob:class {},URL:{createObjectURL:()=>'/backup.npz',revokeObjectURL(){}},
  Worker:class {postMessage(value){sent.push(JSON.parse(JSON.stringify(value)));} terminate(){}},
  MutationObserver:class{observe(){} disconnect(){}},
  requestAnimationFrame:fn=>fn(),setTimeout:()=>1,clearTimeout(){},
  atob,Uint8Array,AbortController,
  fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject})),
});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const evaluate=source=>vm.runInContext(source,context);
const read=name=>JSON.parse(evaluate('JSON.stringify('+name+')'));
function receive(value){context.message=value;evaluate('worker.onmessage({data:message})');}
function clickSave(){const button=node('[data-command="save"]');
  nodeListeners.get('element:click')({target:button});}
function begin(){
  receive(fixtures.initial);
  evaluate('sendControl("motor",0,1);sendControl("pause")');
  receive(fixtures.moving);
}
async function settle(){await new Promise(resolve=>setImmediate(resolve));}
function reply(index,ok,result){requests[index].resolve({ok,json:async()=>result});}
"""


def run_controller(tmp_path, recording, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for controller regression tests")
    controller = tmp_path / "controller.js"
    controller.write_text(CANVAS_JAVASCRIPT + CONTROLLER_JAVASCRIPT)
    runner = tmp_path / "runner.cjs"
    runner.write_text(NODE_HARNESS + "\n(async()=>{\n" + assertions +
                      "\n})().catch(error=>{console.error(error);process.exitCode=1;});")
    result = subprocess.run(
        [node, str(runner), str(controller)], input=json.dumps(recording),
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("event", ["blur", "pagehide", "visibilitychange"])
@pytest.mark.parametrize("running", [False, True])
def test_backgrounding_stops_every_motor_and_pauses(tmp_path, recording, event, running):
    run_controller(tmp_path, recording, """
receive(fixtures.initial);
evaluate('sendControl("motor",0,1);sendControl("motor",5,-1)');
if (RUNNING) evaluate('sendControl("pause")');
const previousSequence=sent.at(-1).sequence;
if (EVENT==='visibilitychange') {document.hidden=true;docListeners.get(EVENT)();}
else winListeners.get(EVENT)();
const command=sent.at(-1);
assert.equal(command.kind,'pause');
assert.equal(command.paused,true);
assert.deepEqual(command.motors,[0,0,0,0,0,0]);
assert.equal(command.sequence,previousSequence+1);
assert.equal(command.generation,fixtures.initial.snapshot.playback.generation);
assert.deepEqual(read('desiredMotors'),[0,0,0,0,0,0]);
assert.equal(read('desiredPaused'),true);
// A delayed pre-background frame cannot re-select a moving motor.
receive(fixtures.moving);
assert.deepEqual(read('desiredMotors'),[0,0,0,0,0,0]);
assert.equal(read('desiredPaused'),true);
""".replace("RUNNING", str(running).lower()).replace("EVENT", json.dumps(event)))


def test_student_id_must_be_present_before_finishing_recording(tmp_path, recording):
    run_controller(tmp_path, recording, """
receive(fixtures.initial);
const participant=node('.coffee-participant');
participant.value='  \t ';
clickSave();
assert.equal(sent.filter(command=>command.kind==='save').length,0);
assert.equal(read('saving'),false);
assert.equal(participant.disabled,false);
assert.equal(participant.reported,true);assert.equal(participant.focused,true);
assert.match(participant.validationMessage,/student ID/);
participant.value='  20260001  ';
nodeListeners.get('.coffee-participant:input')();
assert.equal(participant.validationMessage,'');
clickSave();
assert.equal(sent.at(-1).kind,'save');
assert.equal(sent.at(-1).participant,'20260001');
assert.equal(participant.disabled,true);
assert.equal(read('saving'),true);
// A worker save error lets the student correct their details and try again.
receive({error:'No recording yet',command:'save'});
assert.equal(participant.disabled,false);
assert.equal(read('saving'),false);
""")


def test_deadline_countdown_and_timeout_leave_submission_available(tmp_path, recording):
    runtime = BrowserRuntime(seed=33)
    try:
        runtime.session.paused = False
        for _ in range(1920):
            runtime.session.advance()
        timeout = json.loads(runtime.snapshot())
        saved = json.loads(runtime.dispatch('{"kind":"save","participant":"20260001"}'))
    finally:
        runtime.session.close()
    run_controller(tmp_path, {**recording, "timeout": timeout, "timeout_saved": saved}, r"""
receive(fixtures.initial);assert.match(node('.coffee-time').textContent,/60.0 s left/);
receive(fixtures.timeout);
assert.match(node('.coffee-time').textContent,/60.00 s simulated.*0.0 s left/);
assert.match(node('.coffee-status').textContent,/60-second limit reached.*incomplete/);
assert.equal(node('[data-command="pause"]').disabled,true);
assert.equal(node('[data-command="save"]').disabled,false);
assert.equal(node('[data-command="reset"]').disabled,false);
assert.equal(read('dirty'),true);
const count=sent.length;evaluate('sendControl("motor",0,1)');assert.equal(sent.length,count);
node('.coffee-participant').value='20260001';clickSave();
assert.equal(sent.at(-1).kind,'save');receive(fixtures.timeout_saved);
assert.equal(requests.length,1);assert.equal(node('.coffee-download').hidden,false);
""")


def test_failed_upload_retries_same_archive_and_requires_matching_receipt(tmp_path, recording):
    run_controller(tmp_path, recording, """
begin();
node('.coffee-participant').value='20260001';
clickSave();receive(fixtures.saved);
assert.equal(requests.length,1);
assert.equal(requests[0].url,'/api/submissions?class=test-class');
assert.equal(requests[0].options.method,'POST');
assert.equal(node('[data-command="save"]').disabled,true);
const bytes=Buffer.from(fixtures.saved.archive,'base64');
assert.deepEqual(Buffer.from(requests[0].options.body),bytes);
reply(0,false,{detail:'Temporarily unavailable'});await settle();
assert.equal(read('dirty'),true);
assert.equal(read('archive'),fixtures.saved.archive);
assert.equal(node('.coffee-download').hidden,false);
assert.equal(node('[data-command="save"]').disabled,false);
assert.match(node('.coffee-submission').textContent,/Temporarily unavailable.*retry/);
clickSave();
assert.equal(sent.filter(command=>command.kind==='save').length,1);
assert.deepEqual(Buffer.from(requests[1].options.body),bytes);
reply(1,true,{status:'saved',episode_id:'some-other-attempt',receipt:'wrong'});await settle();
assert.equal(read('dirty'),true);
assert.match(node('.coffee-submission').textContent,/receipt did not match/);
clickSave();
assert.deepEqual(Buffer.from(requests[2].options.body),bytes);
reply(2,true,{status:'pending',episode_id:fixtures.saved.episode_id,receipt:'not-saved'});await settle();
assert.equal(read('dirty'),true);
assert.match(node('.coffee-submission').textContent,/receipt did not match/);
clickSave();
assert.deepEqual(Buffer.from(requests[3].options.body),bytes);
reply(3,true,{status:'saved',episode_id:fixtures.saved.episode_id,receipt:'verified-receipt'});await settle();
assert.equal(read('dirty'),false);
assert.equal(read('archive'),fixtures.saved.archive);
assert.equal(node('[data-command="save"]').textContent,'Submitted ✓');
assert.match(node('.coffee-submission').textContent,/Receipt: verified-receipt/);
assert.equal(sent.filter(command=>command.kind==='save').length,1);
""")


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("COFFEE_BROWSER_EXECUTABLE")
    executable = executable or shutil.which("chromium") or shutil.which("google-chrome")
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not executable and mac_chrome.exists():
        executable = str(mac_chrome)
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(executable_path=executable, headless=True)
        except playwright.Error as exc:
            pytest.skip(f"A Chromium browser is required for rendered website checks: {exc}")
        yield instance
        instance.close()


@pytest.fixture(scope="module")
def website(tmp_path_factory):
    from kaist_rl_lab.apps.coffee_web_assets import build_static_site

    return build_static_site(tmp_path_factory.mktemp("coffee-site"))


def open_student_page(browser, website, recording, width, height, upload=None):
    """Serve the actual exported files while replacing only the physics worker."""
    page = browser.new_page(viewport={"width": width, "height": height}, has_touch=True)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script("""
window.workerMessages=[];
window.Worker=class {
  constructor(url,options) {window.workerUrl=url;this.state=INITIAL;}
  postMessage(command) {
    window.workerMessages.push(command);
    if(command.kind==='save') {
      this.state=SAVED;
    } else if(command.kind!=='init') {
      this.state.snapshot.playback.input_sequence=command.sequence;
      this.state.snapshot.playback.motors=command.motors;
      this.state.snapshot.playback.paused=command.paused;
      this.state.snapshot.playback.revision++;
    }
    const frame=structuredClone(this.state);
    setTimeout(()=>this.onmessage({data:frame}),0);
  }
  terminate(){}
};
""".replace("INITIAL", json.dumps(recording["initial"]))
       .replace("SAVED", json.dumps(recording["saved"])))

    def serve(route):
        from urllib.parse import urlparse

        path = urlparse(route.request.url).path
        if path == "/api/session":
            route.fulfill(json={"name": "Fall 2026 · RL Lab", "open": True,
                                "participant_required": True})
            return
        if path == "/api/submissions" and upload is not None:
            upload(route)
            return
        file = website / ("index.html" if path == "/" else path.lstrip("/"))
        if not file.is_file():
            route.abort()
            return
        types = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}
        route.fulfill(path=file, content_type=types.get(file.suffix, "application/octet-stream"))

    page.route("**/*", serve)
    page.goto("http://coffee.test/?class=test-class")
    try:
        page.wait_for_function("document.querySelector('[data-command=pause]').disabled === false",
                               timeout=5000)
    except Exception:
        page.close()
        assert not errors, errors
        raise
    return page, errors


@pytest.mark.parametrize("width,height", [(360, 780), (390, 844), (430, 932), (740, 360)])
def test_exported_student_page_is_usable_on_phones(browser, website, recording, width, height):
    page, errors = open_student_page(browser, website, recording, width, height)
    try:
        assert page.locator("#class-name").inner_text() == "Fall 2026 · RL Lab"
        assert page.locator(".collection-limit-note").is_visible()
        assert "60 seconds of simulation time" in page.locator(".collection-limit-note").inner_text()
        assert page.locator(".coffee-participant").get_attribute("required") is not None
        assert page.evaluate("window.workerUrl") == "/coffee-worker.js"
        assert page.evaluate("window.workerMessages[0].bundle_url") == \
            "http://coffee.test/coffee-bundle.zip"
        dimensions = page.evaluate("""() => ({
          pageWidth:document.documentElement.scrollWidth,
          targets:Array.from(document.querySelectorAll('button'),button=>{
            const r=button.getBoundingClientRect();return {width:r.width,height:r.height,
              left:r.left,right:r.right};
          })
        })""")
        assert dimensions["pageWidth"] <= width
        assert len(dimensions["targets"]) == 22
        for target in dimensions["targets"]:
            assert target["width"] >= 44
            assert target["height"] >= 44
            assert target["left"] >= 0
            assert target["right"] <= width
        if width < height:
            last_joint = page.locator('[data-joint-index="5"]').bounding_box()
            assert last_joint["y"] + last_joint["height"] <= height
        # A real touchscreen click reaches the shared controller and changes the
        # selected command, with no notebook/kernel interaction.
        direction = page.locator('[data-joint-index="0"] [data-direction="1"]')
        direction.tap()
        page.wait_for_function("window.workerMessages.some(message => message.kind==='motor')")
        assert page.evaluate("window.workerMessages.filter(m=>m.kind==='motor').at(-1).motors") == \
            [1, 0, 0, 0, 0, 0]
        assert direction.get_attribute("aria-pressed") == "true"
        page.evaluate("window.dispatchEvent(new Event('pagehide'))")
        assert page.evaluate("window.workerMessages.at(-1).motors") == [0] * 6
        assert page.evaluate("window.workerMessages.at(-1).paused")
        assert not errors
    finally:
        page.close()


def test_practice_page_keeps_the_collection_limit_notice(browser, website, recording):
    page, errors = open_student_page(browser, website, recording, 390, 844)
    try:
        page.goto("http://coffee.test/")
        page.wait_for_function("document.querySelector('#save-heading').textContent === 'Finish & save'")
        notice = page.locator(".collection-limit-note")
        assert notice.is_visible()
        assert "60 seconds of simulation time" in notice.inner_text()
        assert "Pausing also pauses the countdown" in notice.inner_text()
        assert "No recording is sent" in page.locator(".save-note").inner_text()
        assert not errors
    finally:
        page.close()


def test_exported_page_requires_id_and_recovers_failed_submission(browser, website, recording):
    requests = []

    def upload(route):
        requests.append(route.request.post_data_buffer)
        if len(requests) == 1:
            route.fulfill(status=503, json={"detail": "Try again shortly"})
        else:
            route.fulfill(json={"status": "saved", "episode_id": recording["saved"]["episode_id"],
                                "receipt": "mobile-receipt"})

    page, errors = open_student_page(browser, website, recording, 390, 844, upload)
    try:
        submit = page.locator('[data-command="save"]')
        submit.click()
        assert page.evaluate("window.workerMessages.filter(m=>m.kind==='save').length") == 0
        assert not requests
        assert page.locator(".coffee-participant").evaluate("input => !input.validity.valid")
        page.locator(".coffee-participant").fill("20260001")
        submit.click()
        page.wait_for_function(
            "document.querySelector('.coffee-submission').textContent.includes('Try again shortly')"
        )
        assert page.locator(".coffee-download").is_visible()
        assert page.locator(".coffee-participant").is_disabled()
        assert submit.is_enabled()
        submit.click()
        page.wait_for_function(
            "document.querySelector('.coffee-submission').textContent.includes('mobile-receipt')"
        )
        assert len(requests) == 2
        assert requests[0] == requests[1]
        assert requests[0]
        assert page.evaluate("window.workerMessages.filter(m=>m.kind==='save').length") == 1
        assert submit.inner_text() == "Submitted ✓"
        assert not errors
    finally:
        page.close()
