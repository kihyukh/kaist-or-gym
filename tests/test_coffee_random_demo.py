"""Random-demo controller checks using real snapshots and a mocked browser worker."""

import json
import shutil
import subprocess

import pytest

from kaist_rl_lab.apps.coffee_random_demo import RANDOM_DEMO_JAVASCRIPT
from kaist_rl_lab.apps.coffee_random_runtime import RandomAgentRuntime


@pytest.fixture(scope="module")
def random_snapshots():
    runtime = RandomAgentRuntime()
    try:
        initial = json.loads(runtime.snapshot())
        running = json.loads(runtime.dispatch('{"kind":"random-start","seed":3}'))
        paused = json.loads(runtime.dispatch('{"kind":"random-pause","paused":true}'))
        reset = json.loads(runtime.dispatch('{"kind":"random-reset"}'))
        return {"initial": initial, "running": running, "paused": paused, "reset": reset}
    finally:
        runtime.session.close()


NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const fixtures=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes=new Map(),docListeners=new Map(),winListeners=new Map(),workers=[];
class Node {
  constructor(){
    this.children=[];this.listeners={};this.hidden=true;this.disabled=false;
    this.textContent='';this.open=false;this.classList={toggle(){}};
  }
  set innerHTML(value){throw Error('Status and result text must never be parsed as HTML');}
  addEventListener(event,callback){this.listeners[event]=callback;}
  append(child){child.parent=this;this.children.push(child);}
  prepend(child){child.parent=this;this.children.unshift(child);}
  replaceChildren(){this.children=[];}
  get lastElementChild(){return this.children.at(-1);}
  remove(){this.parent.children=this.parent.children.filter(child=>child!==this);}
}
function node(selector){
  if(selector==='.coffee-canvas')return null;
  if(!nodes.has(selector))nodes.set(selector,new Node());
  return nodes.get(selector);
}
const element={querySelector:node,querySelectorAll:()=>[]};
const document={hidden:false,createElement:()=>new Node(),
  addEventListener:(event,callback)=>docListeners.set(event,callback)};
const window={addEventListener:(event,callback)=>winListeners.set(event,callback)};
let nextWorkerError=null;
class Worker {
  constructor(url,options){
    if(nextWorkerError){const error=nextWorkerError;nextWorkerError=null;throw error;}
    this.url=url;this.options=options;this.messages=[];this.terminated=false;workers.push(this);
  }
  postMessage(message){this.messages.push(JSON.parse(JSON.stringify(message)));}
  terminate(){this.terminated=true;}
  receive(message){this.onmessage({data:JSON.parse(JSON.stringify(message))});}
}
const context=vm.createContext({element,document,window,Worker,URL,
  location:{href:'https://coffee.test/instructor'},
  fetch(){throw Error('Random trials must not upload or request instructor data');},console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const demo=vm.runInContext('createRandomDemo(element)',context);
const click=selector=>node(selector).listeners.click();
const latest=()=>workers.at(-1);
function start(){
  demo.setEnabled(true);click('#random-start');
  latest().receive(fixtures.initial);latest().receive(fixtures.running);
}
function background(event){
  if(event==='visibilitychange'){document.hidden=true;docListeners.get(event)();}
  else winListeners.get(event)();
}
function completed(number){
  const message=JSON.parse(JSON.stringify(fixtures.paused));
  message.episode_id='completed-'+number;
  Object.assign(message.random_agent,{attempt:number,done:true,outcome:'time_limit',elapsed_seconds:30});
  Object.assign(message.snapshot.state.liquid,{fill_l:0.123,spill_l:0.045});
  message.snapshot.playback.running=false;
  return message;
}
"""


def run_demo(tmp_path, snapshots, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for random-demo controller regression tests")
    controller = tmp_path / "random-demo.js"
    controller.write_text(RANDOM_DEMO_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(NODE_HARNESS + "\n" + assertions)
    result = subprocess.run(
        [node, str(runner), str(controller)], input=json.dumps(snapshots),
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_loading_and_start_pause_resume_reset(tmp_path, random_snapshots):
    run_demo(tmp_path, random_snapshots, r"""
demo.setEnabled(false);click('#random-start');assert.equal(workers.length,0);
demo.setEnabled(true);assert.equal(workers.length,0);
click('#random-start');assert.equal(workers.length,1);
const worker=latest();assert.equal(worker.url,'/coffee-worker.js');
assert.equal(worker.options.type,'module');
assert.deepEqual(worker.messages,[{kind:'init',mode:'random',
  bundle_url:'https://coffee.test/coffee-bundle.zip'}]);
assert.equal(node('#random-start').disabled,true);
worker.receive({loading:'Loading NumPy…'});
assert.match(node('#random-status').textContent,/Loading NumPy/);
worker.receive(fixtures.initial);
assert.deepEqual(worker.messages.at(-1),{kind:'random-start'});
assert.equal(node('#random-start').disabled,true);
worker.receive(fixtures.running);
assert.equal(node('#random-scene').hidden,false);
assert.equal(node('#random-start').disabled,false);
assert.equal(node('#random-pause').disabled,false);
assert.equal(node('#random-trial').textContent,'1');
assert.equal(node('#random-decisions').textContent,'1');
assert.match(node('#random-status').textContent,/Running/);
click('#random-pause');assert.deepEqual(worker.messages.at(-1),{kind:'random-pause',paused:true});
worker.receive(fixtures.paused);assert.equal(node('#random-pause').textContent,'Resume');
click('#random-pause');assert.deepEqual(worker.messages.at(-1),{kind:'random-pause',paused:false});
worker.receive(fixtures.running);assert.equal(node('#random-pause').textContent,'Pause');
click('#random-reset');assert.deepEqual(worker.messages.at(-1),{kind:'random-reset'});
worker.receive(fixtures.reset);
assert.equal(node('#random-trial').textContent,'0');
assert.equal(node('#random-pause').disabled,true);
assert.equal(node('#random-start').textContent,'Start random trial');
assert.equal(node('#random-history').hidden,true);
""")


@pytest.mark.parametrize("event", ["visibilitychange", "blur", "pagehide"])
def test_backgrounding_pauses_a_running_trial(tmp_path, random_snapshots, event):
    run_demo(tmp_path, random_snapshots, """
start();background(EVENT);
assert.deepEqual(latest().messages.at(-1),{kind:'random-pause',paused:true});
latest().receive(fixtures.paused);
assert.equal(node('#random-pause').textContent,'Resume');
""".replace("EVENT", json.dumps(event)))


@pytest.mark.parametrize("event", ["visibilitychange", "blur", "pagehide"])
def test_backgrounding_cancels_start_during_runtime_loading(tmp_path, random_snapshots, event):
    run_demo(tmp_path, random_snapshots, """
demo.setEnabled(true);click('#random-start');background(EVENT);
latest().receive(fixtures.initial);
assert.equal(latest().messages.filter(message=>message.kind==='random-start').length,0);
assert.equal(node('#random-pause').disabled,true);
document.hidden=false;
click('#random-start');latest().receive(fixtures.running);
assert.equal(latest().messages.filter(message=>message.kind==='random-start').length,1);
assert.match(node('#random-status').textContent,/Running/);
""".replace("EVENT", json.dumps(event)))


@pytest.mark.parametrize("event", ["visibilitychange", "blur", "pagehide"])
def test_backgrounding_cancels_in_flight_start(tmp_path, random_snapshots, event):
    run_demo(tmp_path, random_snapshots, """
demo.setEnabled(true);click('#random-start');latest().receive(fixtures.initial);
assert.equal(latest().messages.at(-1).kind,'random-start');
background(EVENT);
// The worker has started, but its first frame arrives after the page loses focus.
latest().receive(fixtures.running);
assert.deepEqual(latest().messages.at(-1),{kind:'random-pause',paused:true});
latest().receive(fixtures.paused);
assert.equal(node('#random-pause').textContent,'Resume');
""".replace("EVENT", json.dumps(event)))


@pytest.mark.parametrize("running", [False, True])
def test_logout_terminates_worker_and_ignores_late_events(tmp_path, random_snapshots, running):
    run_demo(tmp_path, random_snapshots, """
demo.setEnabled(true);click('#random-start');
if(RUNNING){latest().receive(fixtures.initial);latest().receive(fixtures.running);}
const old=latest();demo.setEnabled(false);
assert.equal(old.terminated,true);assert.equal(node('#random-scene').hidden,true);
assert.equal(node('#random-start').disabled,true);
const status=node('#random-status').textContent;
old.receive(fixtures.running);old.receive(completed(1));old.onerror();
assert.equal(node('#random-status').textContent,status);
assert.equal(node('#random-scene').hidden,true);
assert.equal(node('#random-results').children.length,0);
demo.setEnabled(true);assert.equal(workers.length,1);
click('#random-start');const current=latest();assert.notEqual(current,old);
old.receive({error:'Stale worker error'});old.onerror();
assert.equal(current.terminated,false);assert.equal(node('#random-start').disabled,true);
current.receive(fixtures.initial);current.receive(fixtures.running);
assert.match(node('#random-status').textContent,/Running/);
""".replace("RUNNING", "true" if running else "false"))


def test_results_are_deduplicated_bounded_and_cleared_by_reset(tmp_path, random_snapshots):
    run_demo(tmp_path, random_snapshots, r"""
start();const worker=latest();
worker.receive(completed(1));worker.receive(completed(1));
assert.equal(node('#random-results').children.length,1);
assert.deepEqual(node('#random-results').children[0].children.map(cell=>cell.textContent),
  ['1','123 mL','45 mL','30.0 s','Time limit reached']);
assert.equal(node('#random-pause').disabled,true);
assert.equal(node('#random-start').disabled,false);
for(let number=2;number<=12;number++)worker.receive(completed(number));
assert.equal(node('#random-results').children.length,10);
assert.equal(node('#random-count').textContent,'(12)');
assert.equal(node('#random-results').children[0].children[0].textContent,'12');
assert.equal(node('#random-results').lastElementChild.children[0].textContent,'3');
assert.equal(node('#random-history').hidden,false);
click('#random-reset');worker.receive(fixtures.reset);
assert.equal(node('#random-results').children.length,0);
assert.equal(node('#random-count').textContent,'');
assert.equal(node('#random-history').hidden,true);
assert.equal(node('#random-history').open,false);
""")


@pytest.mark.parametrize("failure", ["initialization", "message", "worker"])
def test_runtime_errors_allow_a_clean_retry(tmp_path, random_snapshots, failure):
    run_demo(tmp_path, random_snapshots, """
demo.setEnabled(true);
if(FAILURE==='initialization')nextWorkerError=new Error('Worker unavailable');
click('#random-start');
if(FAILURE==='message')latest().receive({error:'<img src=x onerror=alert(1)>'});
if(FAILURE==='worker')latest().onerror();
assert.equal(node('#random-start').disabled,false);
assert.equal(node('#random-pause').disabled,true);
assert.match(node('#random-status').textContent,/try again/);
if(workers.length)assert.equal(latest().terminated,true);
click('#random-start');const worker=latest();
assert.equal(worker.terminated,false);
assert.deepEqual(worker.messages.map(message=>message.kind),['init']);
worker.receive(fixtures.initial);worker.receive(fixtures.running);
assert.match(node('#random-status').textContent,/Running/);
""".replace("FAILURE", json.dumps(failure)))
