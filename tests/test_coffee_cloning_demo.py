"""Instructor cloning UI: scoped training, real worker protocol, and stale replies."""

import json
import shutil
import subprocess

import pytest

from kaist_rl_lab.apps.coffee_cloning import train_behavior_cloning
from kaist_rl_lab.apps.coffee_cloning_demo import CLONING_DEMO_JAVASCRIPT
from kaist_rl_lab.apps.coffee_cloning_runtime import CloningAgentRuntime
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.apps.coffee_expert import load_examples


@pytest.fixture(scope="module")
def cloning_snapshots():
    demonstrations = []
    for archive in load_examples()[:2]:
        arrays, metadata = read_demonstration(archive)
        demonstrations.append(({key: value[:4] for key, value in arrays.items()}, metadata))
    model = train_behavior_cloning(demonstrations)
    runtime = CloningAgentRuntime()
    try:
        initial = json.loads(runtime.snapshot())
        loaded = json.loads(runtime.dispatch(json.dumps({"kind": "cloning-load", "model": model})))
        running = json.loads(runtime.dispatch('{"kind":"cloning-start"}'))
        paused = json.loads(runtime.dispatch('{"kind":"cloning-pause","paused":true}'))
        reset = json.loads(runtime.dispatch('{"kind":"cloning-reset"}'))
        return {
            "initial": initial,
            "loaded": loaded,
            "running": running,
            "paused": paused,
            "reset": reset,
            "model": model,
            "result": {
                "model": model,
                "source_label": "Class A <student text>",
                "selection": {
                    "skipped_unsuccessful": 2,
                    "skipped_invalid": 1,
                    "skipped_incompatible": 3,
                    "capped_trajectories": 4,
                },
            },
        }
    finally:
        runtime.session.close()


NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const fixtures=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes=new Map(),docListeners=new Map(),winListeners=new Map(),workers=[],requests=[];
let beforeRunCount=0,nextWorkerError=null;
class Node {
  constructor(){
    this.children=[];this.listeners={};this.hidden=true;this.disabled=false;
    this.textContent='';this.checked=false;this.value='';this.classList={toggle(){}};
  }
  set innerHTML(value){throw Error('Dynamic dataset and error text must never be parsed as HTML');}
  addEventListener(event,callback){this.listeners[event]=callback;}
}
function node(selector){
  if(selector==='.coffee-canvas')return null;
  if(!nodes.has(selector))nodes.set(selector,new Node());
  return nodes.get(selector);
}
node('#cloning-source').value='students';node('#cloning-successful').checked=true;
node('#cloning-speed').value='4';
const element={querySelector:node,querySelectorAll:()=>[]};
const document={hidden:false,createElement:()=>new Node(),
  addEventListener:(event,callback)=>docListeners.set(event,callback)};
const window={addEventListener:(event,callback)=>winListeners.set(event,callback)};
class Worker {
  constructor(url,options){
    if(nextWorkerError){const error=nextWorkerError;nextWorkerError=null;throw error;}
    this.url=url;this.options=options;this.messages=[];this.terminated=false;workers.push(this);
  }
  postMessage(message){this.messages.push(JSON.parse(JSON.stringify(message)));}
  terminate(){this.terminated=true;}
  receive(message){this.onmessage({data:JSON.parse(JSON.stringify(message))});}
}
function post(path,body){
  assert.equal(path,'/api/instructor/cloning/train','No trial or trajectory may be uploaded');
  return new Promise((resolve,reject)=>{
    requests.push({path,body:JSON.parse(JSON.stringify(body)),resolve,reject});
  });
}
const context=vm.createContext({element,document,window,Worker,URL,post,
  beforeRun:()=>beforeRunCount++,location:{href:'https://coffee.test/instructor'},
  fetch(){throw Error('Cloned trials must not upload trajectories');},console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const demo=vm.runInContext('createCloningDemo(element,post,beforeRun)',context);
const click=selector=>node(selector).listeners.click();
const change=selector=>node(selector).listeners.change();
const latest=()=>workers.at(-1);
const kinds=worker=>worker.messages.map(message=>message.kind);
const classA={id:'class-a',name:'Class A',total:3,successful:2};
const classB={id:'class-b',name:'Class B',total:4,successful:1};
function enable(){demo.setEnabled(true);demo.setContext(classA);}
async function train(result=fixtures.result){
  const completed=click('#cloning-train');requests.at(-1).resolve(result);await completed;
}
async function start(){
  enable();await train();click('#cloning-run');
  latest().receive(fixtures.initial);latest().receive(fixtures.loaded);latest().receive(fixtures.running);
}
function background(event){
  if(event==='visibilitychange'){document.hidden=true;docListeners.get(event)();}
  else winListeners.get(event)();
}
async function main(){
"""


def run_demo(tmp_path, snapshots, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for cloning-demo controller regression tests")
    controller = tmp_path / "cloning-demo.js"
    controller.write_text(CLONING_DEMO_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(
        NODE_HARNESS
        + assertions
        + "\n}\nmain().catch(error=>{console.error(error);process.exit(1)});"
    )
    result = subprocess.run(
        [node, str(runner), str(controller)],
        input=json.dumps(snapshots),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_available_student_data_and_generated_source_controls(tmp_path, cloning_snapshots):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
demo.setEnabled(false);click('#cloning-train');click('#cloning-run');
assert.equal(requests.length,0);assert.equal(workers.length,0);
demo.setEnabled(true);assert.equal(node('#cloning-train').disabled,true);
demo.setContext({...classA,total:2,successful:0});
assert.equal(node('#cloning-train').disabled,true);
node('#cloning-successful').checked=false;change('#cloning-successful');
assert.equal(node('#cloning-train').disabled,false);
await train();assert.deepEqual(requests.at(-1).body,
  {source:'students',session_id:'class-a',successful_only:false});
demo.setContext({...classA,total:0,successful:0});
assert.equal(node('#cloning-train').disabled,true);
node('#cloning-source').value='examples';change('#cloning-source');
assert.equal(node('#cloning-successful').disabled,true);
assert.equal(node('#cloning-train').disabled,false);
assert.equal(node('#cloning-run').disabled,true);
assert.match(node('#cloning-dataset').textContent,/never counted as student submissions/);
// A selected class must not leak into the generated-examples request.
await train();
assert.deepEqual(requests.at(-1).body,
  {source:'examples',session_id:null,successful_only:false});
assert.equal(workers.length,0);assert.equal(node('#cloning-run').disabled,false);
""",
    )


def test_training_scope_summary_and_no_automatic_trial(tmp_path, cloning_snapshots):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();const completed=click('#cloning-train');
assert.equal(beforeRunCount,1);assert.equal(node('#cloning-train').disabled,true);
assert.equal(node('#cloning-run').disabled,true);await click('#cloning-train');
assert.equal(requests.length,1);
assert.deepEqual(requests[0].body,{source:'students',session_id:'class-a',successful_only:true});
requests[0].resolve(fixtures.result);await completed;
assert.equal(node('#cloning-training-result').hidden,false);
assert.match(node('#cloning-training-summary').textContent,/Class A <student text>/);
assert.match(node('#cloning-selection').textContent,/2 unsuccessful attempts excluded/);
assert.match(node('#cloning-selection').textContent,/4 invalid or incompatible attempts excluded/);
assert.match(node('#cloning-selection').textContent,/4 additional attempts omitted/);
assert.match(node('#cloning-validation').textContent,/Held-out action error/);
assert.match(node('#cloning-validation').textContent,/not a pouring success rate/);
assert.equal(node('#cloning-run').disabled,false);assert.equal(workers.length,0);
demo.setContext({...classA,total:10,successful:5});
assert.equal(node('#cloning-training-result').hidden,false);
assert.equal(node('#cloning-run').disabled,false);
""",
    )


def test_real_worker_loading_start_pause_resume_reset_protocol(tmp_path, cloning_snapshots):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();await train();click('#cloning-run');const worker=latest();
assert.equal(beforeRunCount,2);assert.equal(worker.url,'/coffee-worker.js');
assert.equal(worker.options.type,'module');
assert.deepEqual(worker.messages,[{kind:'init',mode:'cloning',
  bundle_url:'https://coffee.test/coffee-bundle.zip'}]);
assert.equal(node('#cloning-run').disabled,true);
worker.receive({loading:'Loading NumPy…'});assert.match(node('#cloning-status').textContent,/NumPy/);
worker.receive(fixtures.initial);
assert.deepEqual(worker.messages.at(-1),{kind:'cloning-load',model:fixtures.model});
worker.receive(fixtures.loaded);assert.deepEqual(worker.messages.at(-1),{kind:'cloning-start',speed:4});
worker.receive(fixtures.running);assert.equal(node('#cloning-scene').hidden,false);
assert.equal(node('#cloning-trial').textContent,'1');
assert.equal(node('#cloning-pause').disabled,false);
assert.match(node('#cloning-status').textContent,/Running the cloned policy/);
click('#cloning-pause');assert.deepEqual(worker.messages.at(-1),{kind:'cloning-pause',paused:true});
worker.receive(fixtures.paused);assert.equal(node('#cloning-pause').textContent,'Resume policy');
click('#cloning-pause');assert.deepEqual(worker.messages.at(-1),{kind:'cloning-pause',paused:false});
assert.equal(beforeRunCount,3);worker.receive(fixtures.running);
click('#cloning-reset');assert.deepEqual(worker.messages.at(-1),{kind:'cloning-reset'});
worker.receive(fixtures.reset);assert.equal(node('#cloning-trial').textContent,'0');
assert.equal(node('#cloning-pause').disabled,true);
assert.equal(node('#cloning-run').disabled,false);
click('#cloning-run');assert.equal(workers.length,1);
assert.equal(worker.messages.at(-1).kind,'cloning-start');
assert.equal(requests.length,1);
""",
    )


def test_cloning_speed_control_defaults_to_four_and_updates_live_pacing(tmp_path, cloning_snapshots):
    from kaist_rl_lab.apps.coffee_cloning_demo import CLONING_DEMO_HTML

    assert '<option value="4" selected>4×</option>' in CLONING_DEMO_HTML
    run_demo(tmp_path, cloning_snapshots, r"""
await start();const worker=latest();
assert.match(node('#cloning-status').textContent,/4× playback speed/);
node('#cloning-speed').value='8';change('#cloning-speed');
assert.deepEqual(worker.messages.at(-1),{kind:'cloning-speed',speed:8});
assert.equal(node('#cloning-speed').disabled,true);
const faster=JSON.parse(JSON.stringify(fixtures.running));faster.cloning_agent.playback_speed=8;
worker.receive(faster);assert.equal(node('#cloning-speed').disabled,false);
assert.match(node('#cloning-status').textContent,/8× playback speed/);
click('#cloning-run');assert.deepEqual(worker.messages.at(-1),{kind:'cloning-start',speed:8});
""")


@pytest.mark.parametrize("change_type", ["source", "class", "filter", "logout"])
def test_stale_training_reply_cannot_restore_previous_data(
    tmp_path, cloning_snapshots, change_type
):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();const completed=click('#cloning-train');
if(CHANGE==='source'){node('#cloning-source').value='examples';change('#cloning-source');}
if(CHANGE==='class')demo.setContext(classB);
if(CHANGE==='filter'){node('#cloning-successful').checked=false;change('#cloning-successful');}
if(CHANGE==='logout')demo.setEnabled(false);
const status=node('#cloning-status').textContent;
requests[0].resolve(fixtures.result);await completed;
assert.equal(node('#cloning-training-result').hidden,true);
assert.equal(node('#cloning-training-summary').textContent,'');
assert.equal(node('#cloning-validation').textContent,'');
assert.equal(node('#cloning-selection').textContent,'');
assert.equal(node('#cloning-run').disabled,true);
assert.equal(node('#cloning-status').textContent,status);
assert.equal(workers.length,0);
if(CHANGE==='logout'){enable();}
await train();assert.equal(node('#cloning-run').disabled,false);
if(CHANGE==='class')assert.equal(requests.at(-1).body.session_id,'class-b');
if(CHANGE==='source')assert.equal(requests.at(-1).body.source,'examples');
""".replace("CHANGE", json.dumps(change_type)),
    )


def test_earlier_training_failure_cannot_override_new_training(tmp_path, cloning_snapshots):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();const old=click('#cloning-train');demo.setContext(classB);
const current=click('#cloning-train');requests[0].reject(new Error('Old request failed'));
await old;assert.equal(node('#cloning-train').disabled,true);
assert.match(node('#cloning-status').textContent,/Reading demonstrations/);
requests[1].resolve(fixtures.result);await current;
assert.equal(node('#cloning-run').disabled,false);
assert.match(node('#cloning-status').textContent,/Policy trained/);
""",
    )


@pytest.mark.parametrize("event", ["visibilitychange", "blur", "pagehide"])
@pytest.mark.parametrize("stage", ["initializing", "loading-model", "starting", "running"])
def test_backgrounding_prevents_late_unattended_start(tmp_path, cloning_snapshots, event, stage):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();await train();click('#cloning-run');const worker=latest();
if(STAGE!=='initializing')worker.receive(fixtures.initial);
if(['starting','running'].includes(STAGE))worker.receive(fixtures.loaded);
if(STAGE==='running')worker.receive(fixtures.running);
background(EVENT);
if(STAGE==='initializing'){worker.receive(fixtures.initial);worker.receive(fixtures.loaded);}
if(STAGE==='loading-model')worker.receive(fixtures.loaded);
if(['initializing','loading-model'].includes(STAGE)){
  assert.equal(kinds(worker).filter(kind=>kind==='cloning-start').length,0);
  assert.equal(node('#cloning-pause').disabled,true);
} else {
  // A worker that already received start may deliver its first running frame late.
  worker.receive(fixtures.running);
  assert.deepEqual(worker.messages.at(-1),{kind:'cloning-pause',paused:true});
  worker.receive(fixtures.paused);
  assert.equal(node('#cloning-pause').textContent,'Resume policy');
}
document.hidden=false;click('#cloning-run');
assert.equal(worker.messages.at(-1).kind,'cloning-start');
worker.receive(fixtures.running);assert.match(node('#cloning-status').textContent,/Running/);
""".replace("EVENT", json.dumps(event)).replace("STAGE", json.dumps(stage)),
    )


@pytest.mark.parametrize("change_type", ["source", "class", "filter", "logout"])
@pytest.mark.parametrize("running", [False, True])
def test_changing_training_context_terminates_worker_and_erases_policy(
    tmp_path,
    cloning_snapshots,
    change_type,
    running,
):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();await train();click('#cloning-run');
if(RUNNING){latest().receive(fixtures.initial);latest().receive(fixtures.loaded);latest().receive(fixtures.running);}
const old=latest();
if(CHANGE==='source'){node('#cloning-source').value='examples';change('#cloning-source');}
if(CHANGE==='class')demo.setContext(classB);
if(CHANGE==='filter'){node('#cloning-successful').checked=false;change('#cloning-successful');}
if(CHANGE==='logout')demo.setEnabled(false);
assert.equal(old.terminated,true);assert.equal(node('#cloning-scene').hidden,true);
assert.equal(node('#cloning-training-result').hidden,true);
assert.equal(node('#cloning-training-summary').textContent,'');
assert.equal(node('#cloning-run').disabled,true);
const status=node('#cloning-status').textContent;
old.receive(fixtures.running);old.receive({error:'Stale worker error'});old.onerror();
assert.equal(node('#cloning-status').textContent,status);
assert.equal(node('#cloning-scene').hidden,true);
if(CHANGE==='logout')enable();
await train();click('#cloning-run');const current=latest();
assert.notEqual(current,old);old.receive(fixtures.running);old.onerror();
assert.equal(current.terminated,false);assert.equal(node('#cloning-run').disabled,true);
current.receive(fixtures.initial);current.receive(fixtures.loaded);current.receive(fixtures.running);
assert.match(node('#cloning-status').textContent,/Running/);
""".replace("CHANGE", json.dumps(change_type)).replace("RUNNING", json.dumps(running)),
    )


@pytest.mark.parametrize("failure", ["request", "invalid-result", "invalid-metrics"])
def test_training_error_allows_retry_without_previous_model(tmp_path, cloning_snapshots, failure):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
await start();const worker=latest();const completed=click('#cloning-train');
assert.equal(worker.terminated,true);assert.equal(node('#cloning-scene').hidden,true);
if(FAILURE==='request')requests.at(-1).reject(new Error('<img src=x onerror=alert(1)>'));
else if(FAILURE==='invalid-metrics')requests.at(-1).resolve({model:{metrics:{}}});
else requests.at(-1).resolve({model:{}});
await completed;assert.equal(node('#cloning-train').disabled,false);
assert.equal(node('#cloning-run').disabled,true);assert.equal(node('#cloning-training-result').hidden,true);
await train();assert.equal(node('#cloning-run').disabled,false);
""".replace("FAILURE", json.dumps(failure)),
    )


@pytest.mark.parametrize("failure", ["constructor", "message", "worker", "model-load"])
def test_worker_failure_preserves_training_and_allows_clean_retry(
    tmp_path, cloning_snapshots, failure
):
    run_demo(
        tmp_path,
        cloning_snapshots,
        r"""
enable();await train();
if(FAILURE==='constructor')nextWorkerError=new Error('Worker unavailable');
click('#cloning-run');
if(FAILURE==='message')latest().receive({error:'<img src=x onerror=alert(1)>'});
if(FAILURE==='worker')latest().onerror();
if(FAILURE==='model-load'){latest().receive(fixtures.initial);latest().receive(fixtures.initial);}
assert.equal(node('#cloning-run').disabled,false);assert.equal(node('#cloning-pause').disabled,true);
assert.equal(node('#cloning-training-result').hidden,false);
assert.match(node('#cloning-status').textContent,/retry/);
if(workers.length)assert.equal(latest().terminated,true);
click('#cloning-run');const worker=latest();assert.equal(worker.terminated,false);
assert.deepEqual(kinds(worker),['init']);worker.receive(fixtures.initial);
worker.receive(fixtures.loaded);worker.receive(fixtures.running);
assert.match(node('#cloning-status').textContent,/Running/);assert.equal(requests.length,1);
""".replace("FAILURE", json.dumps(failure)),
    )
