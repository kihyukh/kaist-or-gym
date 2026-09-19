"""Browser controller lifecycle with actual runtime snapshots and worker isolation."""

import json
import re
import shutil
import subprocess
from copy import deepcopy

import pytest

from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M
from kaist_rl_lab.apps.coffee_cloning import FEATURE_INDICES, FEATURE_SCALES
from kaist_rl_lab.apps.coffee_cloning_demo import CLONING_DEMO_JAVASCRIPT
from kaist_rl_lab.apps.coffee_finetuning_demo import FINETUNING_HTML, FINETUNING_JAVASCRIPT
from kaist_rl_lab.apps.coffee_finetuning_runtime import FineTuningRuntime


@pytest.fixture(scope="module")
def finetuning_snapshots():
    model = {
        "arm_base_distance_m": ARM_BASE_DISTANCE_M,
        "schema_version": 1, "algorithm": "nearest_neighbor",
        "feature_indices": list(FEATURE_INDICES), "feature_scales": list(FEATURE_SCALES),
        "states": [[0.0] * 15], "actions": [[0.0] * 6],
    }
    runtime = FineTuningRuntime()

    def call(kind, **kwargs):
        return json.loads(runtime.dispatch(json.dumps({"kind": kind, **kwargs})))

    try:
        frames = {"model": model, "initial": call("snapshot")}
        frames["loaded"] = call("ft-load", model=model)
        frames["training"] = call("ft-train", episodes=4)
        frames["paused"] = call("ft-pause", paused=True)
        frames["stopped_early"] = call("ft-stop")
        frames["running"] = call("ft-run", policy="base")
        frames["rollout_paused"] = call("ft-pause", paused=True)
        frames["reset"] = call("ft-reset")
        # Controller rendering fixtures retain the runtime/core schema. Their
        # illustrative returns deliberately include negative rewards.
        completed = deepcopy(frames["paused"])
        state = completed["finetuning"]
        state.update({
            "training_active": False, "training_running": False,
            "training_paused": False, "best_available": True, "has_result": True,
        })
        state["progress"].update({"phase": "complete", "episode": 4, "completed_episodes": 4})
        baseline = {
            "return": -30.25, "raw_return": 12345.0, "fill_ml": 702.0, "spill_ml": 0.05,
            "seconds": 29.3, "success": True, "outcome": "success",
        }
        best = {**baseline, "return": -25.125, "seconds": 28.4, "episode": 2}
        state["result"].update({
            "phase": "complete", "done": True, "baseline": baseline, "best": best,
            "best_episode": 2, "improved": True, "completed_episodes": 4,
            "history": [{
                "episode": 1, "training": {**baseline, "return": -34.0},
                "evaluation": best,
                "update": {"mean_kl": 0.001, "mean_change_bound": 0.02, "actor_change": 0.01},
            }],
        })
        frames["completed"] = completed
        return frames
    finally:
        runtime.close()


NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const fixtures=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes=new Map(),docListeners=new Map(),winListeners=new Map(),workers=[];
let beforeRunCount=0,nextWorkerError=null;
class Node {
  constructor(tag='div'){
    this.tagName=tag;this.children=[];this.listeners={};this.attributes={};this.dataset={};
    this.hidden=true;this.disabled=false;this.checked=false;this.value='';this._text='';
    this.classes=new Set();this.classList={toggle:(name,on)=>{
      if(on)this.classes.add(name);else this.classes.delete(name);
    },contains:name=>this.classes.has(name)};
  }
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  set innerHTML(value){throw Error('Dynamic text must never be parsed as HTML');}
  setAttribute(key,value){
    this.attributes[key]=String(value);
    if(key==='class')this.classes=new Set(String(value).split(/\s+/));
    if(key.startsWith('data-'))this.dataset[key.slice(5).replace(/-([a-z])/g,(_,char)=>char.toUpperCase())]=String(value);
  }
  getAttribute(key){return this.attributes[key]??null;}
  removeAttribute(key){delete this.attributes[key];}
  addEventListener(event,callback){this.listeners[event]=callback;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this._text='';this.children=[...children];}
  get options(){return this.children;}
  focus(){this.focused=true;}
}
function node(selector){
  if(selector==='.coffee-canvas')return null;
  if(!nodes.has(selector)){
    const item=new Node();if(selector==='#ft-learning-viz')item.querySelector=node;
    nodes.set(selector,item);
  }
  return nodes.get(selector);
}
node('#ft-episodes').value='10';node('#ft-strategy').value='policy_search';node('#ft-speed').value='0';node('#ft-playback-speed').value='4';
const stages=['baseline','training','update','evaluation'].map(stage=>{const item=node('#ft-stage-'+stage);item.setAttribute('data-ft-stage',stage);return item;});
const element={querySelector:node,querySelectorAll:selector=>selector==='[data-ft-stage]'?stages:[]};
const document={hidden:false,createElement:tag=>new Node(tag),createElementNS:(_,tag)=>new Node(tag),
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
const context=vm.createContext({element,document,window,Worker,URL,
  beforeRun:()=>beforeRunCount++,location:{href:'https://coffee.test/instructor'},
  fetch(){throw Error('Fine-tuning must not request or upload classroom trajectories');},console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const demo=vm.runInContext('createFineTuningDemo(element,beforeRun)',context);
const click=selector=>node(selector).listeners.click();
const latest=()=>workers.at(-1);
const kinds=worker=>worker.messages.map(message=>message.kind);
const clone=value=>JSON.parse(JSON.stringify(value));
const cells=selector=>node(selector).children.map(row=>row.children.map(cell=>cell.textContent));
function enable(){demo.setEnabled(true);demo.setModel(fixtures.model);}
function start(){
  enable();click('#ft-train');latest().receive(fixtures.initial);
  latest().receive(fixtures.loaded);latest().receive(fixtures.training);
}
function background(event){
  if(event==='visibilitychange'){document.hidden=true;docListeners.get(event)();}
  else winListeners.get(event)();
}
"""


def run_demo(tmp_path, snapshots, assertions, *, include_cloning=False):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for fine-tuning controller regression tests")
    controller = tmp_path / "finetuning-demo.js"
    controller.write_text(
        FINETUNING_JAVASCRIPT + ("\n" + CLONING_DEMO_JAVASCRIPT if include_cloning else ""),
    )
    runner = tmp_path / "check.cjs"
    runner.write_text(NODE_HARNESS + "\n" + assertions)
    result = subprocess.run(
        [node, str(runner), str(controller)], input=json.dumps(snapshots),
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_initialization_load_then_start_without_network(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
demo.setEnabled(false);click('#ft-train');click('#ft-run-base');assert.equal(workers.length,0);
demo.setEnabled(true);assert.equal(node('#ft-train').disabled,true);
demo.setModel(fixtures.model);assert.equal(workers.length,0);
assert.equal(node('#ft-train').disabled,false);assert.equal(node('#ft-run-base').disabled,false);
assert.equal(node('#ft-run-best').disabled,true);
click('#ft-train');const worker=latest();assert.equal(beforeRunCount,1);
assert.equal(worker.url,'/coffee-worker.js');assert.equal(worker.options.type,'module');
assert.deepEqual(worker.messages,[{kind:'init',mode:'finetuning',bundle_url:'https://coffee.test/coffee-bundle.zip'}]);
click('#ft-train');assert.equal(workers.length,1);assert.equal(beforeRunCount,1);
worker.receive({loading:'Loading NumPy…'});assert.match(node('#ft-status').textContent,/NumPy/);
worker.receive(fixtures.initial);assert.deepEqual(worker.messages.at(-1),{kind:'ft-load',model:fixtures.model});
worker.receive(fixtures.loaded);assert.deepEqual(worker.messages.at(-1),{kind:'ft-train',episodes:10,strategy:'policy_search',speed:0});
worker.receive(fixtures.training);assert.equal(node('#ft-scene').hidden,false);
assert.equal(node('#ft-train').disabled,true);assert.equal(node('#ft-episodes').disabled,true);
assert.equal(node('#ft-pause').disabled,false);assert.equal(node('#ft-stop').disabled,false);
assert.equal(node('#ft-run-best').disabled,true);assert.match(node('#ft-status').textContent,/Checking original clone/);
assert.equal(worker.messages.some(message=>message.kind==='ft-step'||message.kind==='save'),false);
""")


def test_ten_iteration_default_and_time_objective_are_explained_visibly():
    options = re.search(r'<select id="ft-episodes">(.*?)</select>', FINETUNING_HTML, re.DOTALL).group(1)
    assert re.findall(r'<option value="(\d+)"', options) == ["10", "25", "50", "100"]
    assert '<option value="10" selected>' in options
    assert '<option value="policy_search" selected>' in FINETUNING_HTML
    assert '<option value="ppo">' in FINETUNING_HTML
    playback = re.search(r'<select id="ft-playback-speed">(.*?)</select>', FINETUNING_HTML, re.DOTALL).group(1)
    assert '<option value="4" selected>' in playback
    explanation = FINETUNING_HTML.split('id="ft-reward-explanation"', 1)[1].split('</section>', 1)[0]
    assert 'hidden' not in explanation.split('>', 1)[0]
    assert 'Earlier reward counts more.' in explanation
    assert 'RL time score' in explanation
    assert 'undiscounted recorded reward' in explanation
    assert 'otherwise −100' in explanation
    assert '− 100 × final error' in explanation
    assert 'γΦₜ₊₁ − Φₜ' in explanation
    assert 'Φ = 0 at every terminal state' in explanation
    assert '1 point' in explanation
    assert '− 0.008dt' not in explanation
    assert 'γ = 0.99<sup>1/32</sup>' in explanation


def test_training_pause_resume_and_live_counters(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
start();const worker=latest();const training=clone(fixtures.training);
training.finetuning.result.strategy='ppo';training.finetuning.result.algorithm='bounded_speed_ppo_actor_critic';
Object.assign(training.finetuning.progress,{phase:'training',episode:3,completed_episodes:2,elapsed_seconds:12.5,reward:-14.25});
worker.receive(training);assert.match(node('#ft-status').textContent,/Exploring nearby actions/);
assert.match(node('#ft-progress').textContent,/Completed 2 \/ 4/);
assert.equal(node('#ft-time').textContent,'12.5 s');assert.equal(node('#ft-reward').textContent,'-14.250');
click('#ft-pause');assert.deepEqual(worker.messages.at(-1),{kind:'ft-pause',paused:true});
worker.receive(fixtures.paused);assert.equal(node('#ft-pause').textContent,'Resume');
click('#ft-pause');assert.deepEqual(worker.messages.at(-1),{kind:'ft-pause',paused:false});
assert.equal(beforeRunCount,2);worker.receive(fixtures.training);
const evaluation=clone(training);evaluation.finetuning.progress.phase='evaluation';
worker.receive(evaluation);assert.match(node('#ft-status').textContent,/Testing the current policy/);
""")


def test_method_choice_is_sent_and_locked_while_training(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
enable();node('#ft-strategy').value='ppo';node('#ft-strategy').listeners.change();
assert.match(node('#ft-method-description').textContent,/critic.*predict future score/i);
node('#ft-episodes').value='25';click('#ft-train');const worker=latest();
assert.equal(node('#ft-strategy').disabled,true);
worker.receive(fixtures.initial);worker.receive(fixtures.loaded);
assert.deepEqual(worker.messages.at(-1),{kind:'ft-train',episodes:25,strategy:'ppo',speed:0});
worker.receive(fixtures.training);assert.equal(node('#ft-strategy').disabled,true);
worker.receive(fixtures.completed);assert.equal(node('#ft-strategy').disabled,false);
node('#ft-strategy').value='policy_search';node('#ft-strategy').listeners.change();
assert.match(node('#ft-method-description').textContent,/paired faster\/slower/);
click('#ft-train');assert.equal(worker.messages.at(-1).strategy,'policy_search');
const search=clone(fixtures.training);search.finetuning.result.strategy='policy_search';
search.finetuning.progress.phase='training';worker.receive(search);
assert.match(node('#ft-status').textContent,/Testing a candidate speed/);
""")


@pytest.mark.parametrize("event", ["visibilitychange", "blur", "pagehide"])
@pytest.mark.parametrize("stage", ["initializing", "loading-model", "starting", "training"])
def test_backgrounding_prevents_late_training_start(tmp_path, finetuning_snapshots, event, stage):
    run_demo(tmp_path, finetuning_snapshots, r"""
enable();click('#ft-train');const worker=latest();
if(STAGE!=='initializing')worker.receive(fixtures.initial);
if(['starting','training'].includes(STAGE))worker.receive(fixtures.loaded);
if(STAGE==='training')worker.receive(fixtures.training);
background(EVENT);
if(STAGE==='initializing'){worker.receive(fixtures.initial);worker.receive(fixtures.loaded);}
if(STAGE==='loading-model')worker.receive(fixtures.loaded);
if(['initializing','loading-model'].includes(STAGE)){
  assert.equal(kinds(worker).filter(kind=>kind==='ft-train').length,0);
  assert.equal(node('#ft-pause').disabled,true);
}else{
  worker.receive(fixtures.training);
  assert.deepEqual(worker.messages.at(-1),{kind:'ft-pause',paused:true});
  worker.receive(fixtures.paused);assert.equal(node('#ft-pause').textContent,'Resume');
}
""".replace("EVENT", json.dumps(event)).replace("STAGE", json.dumps(stage)))


@pytest.mark.parametrize("change", ["clear", "replace", "logout"])
@pytest.mark.parametrize("initialized", [False, True])
def test_model_changes_close_workers_and_ignore_stale_replies(
    tmp_path, finetuning_snapshots, change, initialized,
):
    run_demo(tmp_path, finetuning_snapshots, r"""
enable();click('#ft-train');const old=latest();
if(INITIALIZED){old.receive(fixtures.initial);old.receive(fixtures.loaded);old.receive(fixtures.training);}
if(CHANGE==='clear')demo.setModel(null);
if(CHANGE==='replace')demo.setModel({...fixtures.model,changed:true});
if(CHANGE==='logout')demo.setEnabled(false);
assert.equal(old.terminated,true);assert.equal(node('#ft-scene').hidden,true);
assert.equal(node('#ft-results').hidden,true);assert.equal(node('#ft-history').hidden,true);
assert.equal(node('#ft-learning-viz').hidden,true);assert.equal(node('#ft-learning-chart').children.length,0);
assert.equal(node('#ft-run-best').disabled,true);
const status=node('#ft-status').textContent;
old.receive(fixtures.training);old.receive(fixtures.completed);old.receive({error:'Stale failure'});old.onerror();
assert.equal(node('#ft-status').textContent,status);assert.equal(node('#ft-scene').hidden,true);
enable();click('#ft-train');const current=latest();assert.notEqual(current,old);
old.receive(fixtures.completed);old.onerror();assert.equal(current.terminated,false);
current.receive(fixtures.initial);current.receive(fixtures.loaded);current.receive(fixtures.training);
assert.equal(node('#ft-scene').hidden,false);
""".replace("CHANGE", json.dumps(change)).replace("INITIALIZED", json.dumps(initialized)))


def test_comparison_and_history_use_actual_core_metrics(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
start();latest().receive(fixtures.completed);
assert.equal(node('#ft-results').hidden,false);assert.equal(node('#ft-history').hidden,false);
assert.deepEqual(cells('#ft-comparison'),[
  ['Original clone','-30.250','702.0 mL','2.0 mL','0.1 mL','29.3 s','Success'],
  ['Best evaluated policy','-25.125','702.0 mL','2.0 mL','0.1 mL','28.4 s','Success']]);
assert.match(node('#ft-improvement').textContent,/increased by 5.125/);
assert.match(node('#ft-improvement').textContent,/iteration 2/);
const history=cells('#ft-history-rows');assert.equal(history.length,1);
assert.deepEqual(history[0].slice(0,3),['1','-34.000','-25.125']);
assert.equal(node('#ft-run-best').disabled,false);
const retained=clone(fixtures.completed);retained.finetuning.result.best={...retained.finetuning.result.baseline,episode:0};
retained.finetuning.result.improved=false;retained.finetuning.result.best_episode=0;
latest().receive(retained);assert.match(node('#ft-improvement').textContent,/original cloned policy is retained/);
""")


def test_optional_exploration_summary_uses_observed_perturbations(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
start();const frame=clone(fixtures.completed);
frame.finetuning.result.history[0].update.exploration={
  speed_min:.851,speed_max:1.146,std:.08,noise_std:.64,slower_decisions:57,faster_decisions:61};
latest().receive(frame);
assert.equal(node('#ft-exploration').hidden,false);
assert.match(node('#ft-exploration').textContent,/85.1–114.6% of the clone/);
assert.match(node('#ft-exploration').textContent,/57 slower \/ 61 faster adjustments than the policy being explored/);
latest().receive(fixtures.completed);assert.equal(node('#ft-exploration').hidden,true);
latest().receive(frame);demo.setModel(null);
assert.equal(node('#ft-exploration').hidden,true);assert.equal(node('#ft-exploration').textContent,'');
""")


def test_paired_search_summary_uses_shared_pair_center(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
start();const frame=clone(fixtures.completed);
frame.finetuning.result.strategy='policy_search';
frame.finetuning.result.history[0].update={
  accepted:false,actor_change:0,candidate_speed:.9,pair_center:1,
  exploration:{kind:'paired_parameter',speed_min:.9,speed_max:.9,slower_decisions:1,faster_decisions:0}};
latest().receive(frame);
assert.equal(node('#ft-exploration').textContent,'Candidate: 90.0% of cloned speed · pair centered at 100.0%.');
assert.doesNotMatch(node('#ft-exploration').textContent,/policy being explored|adjustments/);
assert.equal(cells('#ft-history-rows')[0][3],'No change');
""")


def test_stop_early_never_enables_unevaluated_policy_and_new_training_clears_results(
    tmp_path, finetuning_snapshots,
):
    run_demo(tmp_path, finetuning_snapshots, r"""
start();const worker=latest();click('#ft-stop');assert.deepEqual(worker.messages.at(-1),{kind:'ft-stop'});
worker.receive(fixtures.stopped_early);assert.equal(node('#ft-run-best').disabled,true);
assert.equal(node('#ft-train').disabled,false);assert.equal(node('#ft-run-base').disabled,false);
click('#ft-train');worker.receive(fixtures.training);worker.receive(fixtures.completed);
assert.equal(node('#ft-run-best').disabled,false);assert.equal(node('#ft-history').hidden,false);
click('#ft-train');worker.receive(fixtures.training);
assert.equal(node('#ft-results').hidden,true);assert.equal(node('#ft-history').hidden,true);
assert.equal(node('#ft-run-best').disabled,true);
""")


def test_watching_base_policy_requires_no_rl_training_and_pause_reset_are_scoped(
    tmp_path, finetuning_snapshots,
):
    run_demo(tmp_path, finetuning_snapshots, r"""
enable();click('#ft-run-base');const worker=latest();worker.receive(fixtures.initial);worker.receive(fixtures.loaded);
assert.deepEqual(worker.messages.at(-1),{kind:'ft-run',policy:'base',speed:4});
assert.equal(kinds(worker).includes('ft-train'),false);worker.receive(fixtures.running);
assert.match(node('#ft-status').textContent,/without exploration noise/);
click('#ft-pause');worker.receive(fixtures.rollout_paused);assert.equal(node('#ft-pause').textContent,'Resume');
click('#ft-pause');assert.equal(beforeRunCount,2);worker.receive(fixtures.running);
click('#ft-reset');assert.deepEqual(worker.messages.at(-1),{kind:'ft-reset'});worker.receive(fixtures.reset);
assert.equal(node('#ft-pause').disabled,true);assert.equal(node('#ft-run-base').disabled,false);
""")


@pytest.mark.parametrize("failure", ["constructor", "message", "worker", "model-load"])
def test_worker_failure_retains_bc_and_allows_retry(tmp_path, finetuning_snapshots, failure):
    run_demo(tmp_path, finetuning_snapshots, r"""
enable();if(FAILURE==='constructor')nextWorkerError=new Error('Worker unavailable');
click('#ft-train');
if(FAILURE==='message')latest().receive({error:'<img src=x onerror=alert(1)>'});
if(FAILURE==='worker')latest().onerror();
if(FAILURE==='model-load'){latest().receive(fixtures.initial);latest().receive(fixtures.initial);}
assert.equal(node('#ft-train').disabled,false);assert.equal(node('#ft-run-base').disabled,false);
assert.equal(node('#ft-run-best').disabled,true);assert.equal(node('#ft-pause').disabled,true);
assert.match(node('#ft-status').textContent,/try again/);if(workers.length)assert.equal(latest().terminated,true);
click('#ft-train');const worker=latest();assert.equal(worker.terminated,false);
worker.receive(fixtures.initial);worker.receive(fixtures.loaded);worker.receive(fixtures.training);
assert.equal(node('#ft-scene').hidden,false);
""".replace("FAILURE", json.dumps(failure)))


def test_cloning_callback_loads_and_invalidates_the_actual_finetuning_controller(
    tmp_path, finetuning_snapshots,
):
    run_demo(tmp_path, finetuning_snapshots, r"""
(async()=>{
  const requests=[];context.post=()=>new Promise(resolve=>requests.push(resolve));
  context.onModel=value=>demo.setModel(value);
  const cloning=vm.runInContext('createCloningDemo(element,post,()=>{},onModel)',context);
  node('#cloning-source').value='students';node('#cloning-successful').checked=true;
  demo.setEnabled(true);cloning.setEnabled(true);
  cloning.setContext({id:'class-a',name:'Class A',total:1,successful:1});
  assert.equal(node('#ft-train').disabled,true);
  const model={...fixtures.model,metrics:{demonstrations:1,training_samples:1,total_steps:1,heldout_action_mae:null}};
  let finished=click('#cloning-train');requests.at(-1)({model,source_label:'Class A'});await finished;
  assert.equal(node('#ft-train').disabled,false);assert.equal(workers.length,0);
  click('#ft-train');const worker=latest();worker.receive(fixtures.initial);
  worker.receive(fixtures.loaded);worker.receive(fixtures.training);
  cloning.setContext({id:'class-b',name:'Class B',total:1,successful:1});
  assert.equal(worker.terminated,true);assert.equal(node('#ft-train').disabled,true);
  finished=click('#cloning-train');
  cloning.setContext({id:'class-c',name:'Class C',total:1,successful:1});
  requests.at(-1)({model,source_label:'Stale class B'});await finished;
  assert.equal(node('#ft-train').disabled,true);assert.equal(node('#ft-scene').hidden,true);
})().catch(error=>{console.error(error);process.exit(1)});
""", include_cloning=True)



def test_completed_learning_chart_survives_watching_and_display_reset(tmp_path, finetuning_snapshots):
    run_demo(tmp_path, finetuning_snapshots, r"""
start();const worker=latest();worker.receive(fixtures.completed);
assert.equal(node('#ft-learning-viz').hidden,false);
const finishedResult=clone(fixtures.completed.finetuning.result);
const completedProgress=clone(fixtures.completed.finetuning.progress);
click('#ft-run-best');
const watching=clone(fixtures.running);
watching.finetuning.result=finishedResult;watching.finetuning.progress=completedProgress;
watching.finetuning.has_result=true;watching.finetuning.best_available=true;
watching.finetuning.rollout.policy='best';worker.receive(watching);
assert.equal(node('#ft-learning-viz').hidden,false);assert.equal(node('#ft-results').hidden,false);
assert.match(node('#ft-showing').textContent,/Best evaluated policy/);
click('#ft-reset');
const reset=clone(fixtures.reset);
reset.finetuning.result=finishedResult;reset.finetuning.progress=completedProgress;
reset.finetuning.has_result=true;reset.finetuning.best_available=true;worker.receive(reset);
assert.equal(node('#ft-learning-viz').hidden,false);assert.equal(node('#ft-results').hidden,false);
assert.doesNotMatch(node('#ft-showing').textContent,/Last evaluated policy/);
assert.match(node('#ft-showing').textContent,/reset|ready|starting pose/i);
""")


def test_default_fastest_and_speed_changes_use_runtime_controls_without_retraining(
    tmp_path, finetuning_snapshots,
):
    run_demo(tmp_path, finetuning_snapshots, r"""
demo.setEnabled(false);assert.equal(node('#ft-speed').disabled,true);
enable();assert.equal(node('#ft-speed').value,'0');assert.equal(node('#ft-speed').disabled,false);
node('#ft-speed').value='4';node('#ft-speed').listeners.change();assert.equal(workers.length,0);
click('#ft-train');const worker=latest();assert.equal(node('#ft-speed').disabled,true);
worker.receive(fixtures.initial);worker.receive(fixtures.loaded);
assert.deepEqual(worker.messages.at(-1),{kind:'ft-train',episodes:10,strategy:'policy_search',speed:4});
const training=clone(fixtures.training);training.finetuning.playback_speed=4;worker.receive(training);
assert.equal(node('#ft-speed').disabled,false);assert.match(node('#ft-progress').textContent,/4× requested/);
const starts=kinds(worker).filter(kind=>kind==='ft-train').length;
node('#ft-speed').value='8';node('#ft-speed').listeners.change();
assert.deepEqual(worker.messages.at(-1),{kind:'ft-speed',speed:8});
assert.equal(node('#ft-speed').disabled,true);assert.equal(workers.length,1);
training.finetuning.playback_speed=8;worker.receive(training);
assert.equal(kinds(worker).filter(kind=>kind==='ft-train').length,starts);
assert.equal(node('#ft-speed').disabled,false);assert.match(node('#ft-progress').textContent,/8× requested/);
node('#ft-speed').value='0';node('#ft-speed').listeners.change();
assert.deepEqual(worker.messages.at(-1),{kind:'ft-speed',speed:0});
training.finetuning.playback_speed=0;worker.receive(training);
assert.match(node('#ft-progress').textContent,/fastest available/);
worker.receive(fixtures.completed);node('#ft-speed').value='8';
click('#ft-run-best');assert.deepEqual(worker.messages.at(-1),{kind:'ft-run',policy:'best',speed:4});
worker.receive(fixtures.running);node('#ft-playback-speed').value='8';node('#ft-playback-speed').listeners.change();
assert.deepEqual(worker.messages.at(-1),{kind:'ft-speed',speed:8});
worker.receive(fixtures.running);const before=worker.messages.length;
node('#ft-speed').value='0';node('#ft-speed').listeners.change();assert.equal(worker.messages.length,before);
worker.receive(fixtures.completed);click('#ft-train');assert.equal(worker.messages.at(-1).speed,0);
""")
