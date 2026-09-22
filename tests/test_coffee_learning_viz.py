"""Learning chart uses real evaluation results and supports accessible inspection."""

import json
import shutil
import subprocess

import pytest

from kaist_rl_lab.apps.coffee_learning_viz import LEARNING_VIZ_JAVASCRIPT

NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const fixture=JSON.parse(fs.readFileSync(0,'utf8'));
const nodes=new Map();
class Node {
  constructor(tag='div') {
    this.tagName=tag;this.children=[];this.listeners={};this.attributes={};
    this.hidden=false;this.disabled=false;this.checked=false;this.value='';this.parentNode=null;
    this._text='';this.classes=new Set();this.dataset={};
    this.classList={toggle:(name,force)=>{
      const on=force===undefined?!this.classes.has(name):force;
      if(on)this.classes.add(name);else this.classes.delete(name);return on;
    },add:name=>this.classes.add(name),remove:name=>this.classes.delete(name),contains:name=>this.classes.has(name)};
  }
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  set innerHTML(value){throw Error('Untrusted chart data must not be parsed as HTML');}
  setAttribute(key,value){
    this.attributes[key]=String(value);
    if(key==='class')this.classes=new Set(String(value).split(/\s+/));
    if(key.startsWith('data-'))this.dataset[key.slice(5).replace(/-([a-z])/g,(_,char)=>char.toUpperCase())]=String(value);
  }
  getAttribute(key){return this.attributes[key]??null;}
  removeAttribute(key){delete this.attributes[key];}
  addEventListener(event,callback){(this.listeners[event]??=[]).push(callback);}
  append(...children){for(const child of children){this.children.push(child);if(typeof child==='object')child.parentNode=this;}}
  appendChild(child){this.append(child);return child;}
  replaceChildren(...children){this._text='';this.children=[];this.append(...children);}
  querySelector(selector){return this.querySelectorAll(selector)[0]??null;}
  querySelectorAll(selector){return all(this).filter(child=>child!==this&&matches(child,selector));}
  get options(){return this.children;}
  focus(){this.focused=true;}
}
function matches(item,selector){
  if(selector.startsWith('#'))return item.getAttribute('id')===selector.slice(1);
  if(selector.startsWith('.'))return item.classes.has(selector.slice(1));
  const attr=selector.match(/^\[([^=\]]+)(?:=["']?([^"'\]]+)["']?)?\]$/);
  if(attr)return attr[2]===undefined?item.getAttribute(attr[1])!==null:item.getAttribute(attr[1])===attr[2];
  return item.tagName===selector;
}
function all(item){return [item,...item.children.flatMap(child=>typeof child==='object'?all(child):[])];}
function node(selector){
  if(!nodes.has(selector)){
    const item=new Node(selector.includes('chart')?'svg':'div');
    if(selector.startsWith('#'))item.setAttribute('id',selector.slice(1));
    if(selector==='#ft-learning-viz')item.querySelector=node;
    nodes.set(selector,item);
  }
  return nodes.get(selector);
}
const stages=['baseline','training','update','evaluation'].map(stage=>{
  const item=node('#ft-stage-'+stage);item.setAttribute('data-ft-stage',stage);return item;
});
const element={hidden:true,querySelector:node,querySelectorAll:selector=>selector==='[data-ft-stage]'?stages:[]};
const document={createElement:tag=>new Node(tag),createElementNS:(_,tag)=>new Node(tag)};
const context=vm.createContext({document,console,
  fetch(){throw Error('Learning visualization must not use a network request');}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
context.element=element;
const viz=vm.runInContext('createLearningVisualization(element)',context);
const clone=value=>JSON.parse(JSON.stringify(value));
const svg=()=>node('#ft-learning-chart');
const descendants=()=>all(svg());
const points=series=>descendants().filter(item=>item.tagName==='circle'&&item.getAttribute('data-series')===series);
const path=series=>descendants().find(item=>item.classes.has('ft-chart-series-'+series));
const numeric=(item,attribute)=>Number(item.getAttribute(attribute));
const selected=()=>node('#ft-chart-trial').value;
function fire(item,event,extra={}){
  const value={target:item,key:'',preventDefault(){this.prevented=true;},...extra};
  for(const callback of item.listeners[event]??[])callback(value);return value;
}
function assertFiniteSvg(){
  for(const item of descendants())for(const [key,value]of Object.entries(item.attributes)){
    assert.doesNotMatch(value,/NaN|Infinity|undefined/,'Invalid '+key+' on '+item.tagName);
  }
}
"""


@pytest.fixture
def learning_state():
    def metrics(reward):
        return {"return": reward, "fill_ml": 700.0, "spill_ml": 0.0,
                "seconds": 29.0, "success": True, "outcome": "success"}

    history = []
    for episode, training, evaluation in [(1, -4, -2), (2, -5, 0), (3, -2, -3), (4, -1, 1)]:
        history.append({
            "episode": episode, "training": metrics(training), "evaluation": metrics(evaluation),
            "update": {"mean_kl": .001, "mean_change_bound": .02, "actor_change": .01},
        })
    result = {
        "phase": "complete", "done": True, "episode": 4, "episodes": 4,
        "baseline": metrics(-1), "best": {**metrics(1), "episode": 4}, "history": history,
        "latest_update": history[-1]["update"], "speed_bound": .15,
    }
    return {
        "result": result,
        "progress": {"phase": "complete", "episode": 4, "episodes": 4, "completed_episodes": 4},
        "training_active": False, "training_paused": False, "paused": True,
        "rollout": {"active": False},
    }


def run_viz(tmp_path, state, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for learning chart regression tests")
    controller = tmp_path / "learning-viz.js"
    controller.write_text(LEARNING_VIZ_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(NODE_HARNESS + "\n" + assertions)
    result = subprocess.run(
        [node, str(runner), str(controller)], input=json.dumps(state),
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_chart_separates_exploration_evaluation_and_best_checkpoint(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
node('#ft-chart-candidates').checked=true;
viz.render(fixture);
assert.equal(points('exploration').length,4);
assert.equal(points('evaluation').length,4);
assert.equal(Number(svg().getAttribute('data-x-max')),4);
const evaluations=points('evaluation');
// Screen y increases downward; the third evaluation really deteriorated.
assert.ok(numeric(evaluations[1],'cy')<numeric(evaluations[0],'cy'));
assert.ok(numeric(evaluations[2],'cy')>numeric(evaluations[1],'cy'));
assert.ok(numeric(evaluations[3],'cy')<numeric(evaluations[2],'cy'));
assert.notEqual(path('exploration').getAttribute('d'),path('evaluation').getAttribute('d'));
assert.notEqual(path('best').getAttribute('d'),path('evaluation').getAttribute('d'));
assert.ok(descendants().some(item=>item.classes.has('ft-chart-baseline')));
assertFiniteSvg();
""")


def test_candidate_toggle_keeps_failure_evidence_and_inspection_available(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
const state=clone(fixture);state.result.history[0].training={return:-100,success:false};
assert.equal(node('#ft-chart-candidates').checked,true);viz.render(state);
assert.equal(points('exploration').length,4);assert.equal(points('evaluation').length,4);
assert.ok(numeric(svg(),'data-y-min')<-100,'Default chart includes every measured candidate');
assert.match(node('#ft-candidate-summary').textContent,/4 completed.*1 unsuccessful.*shown on chart/);
node('#ft-chart-trial').value='1';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/Exploration \(noisy\).*score -100.000000/);
node('#ft-chart-candidates').checked=true;fire(node('#ft-chart-candidates'),'change');
assert.equal(points('exploration').length,4);assert.ok(numeric(svg(),'data-y-min')<-100);
assert.match(node('#ft-candidate-summary').textContent,/1 unsuccessful.*shown on chart/);
assert.equal(selected(),'1');
node('#ft-chart-candidates').checked=false;fire(node('#ft-chart-candidates'),'change');
assert.equal(points('exploration').length,0);assert.equal(selected(),'1');
assert.match(node('#ft-chart-inspector').textContent,/-100.000/);
node('#ft-chart-candidates').checked=true;fire(node('#ft-chart-candidates'),'change');
viz.reset();assert.equal(node('#ft-chart-candidates').checked,true);
assert.equal(node('#ft-candidate-summary').textContent,'');assertFiniteSvg();
""")


def test_policy_search_counts_accepted_updates_and_inspects_rejected_candidates(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
node('#ft-chart-candidates').checked=true;
const state=clone(fixture);state.result.strategy='policy_search';
state.result.algorithm='bounded_paired_policy_search';
state.training_active=true;state.paused=false;state.result.done=false;
state.progress.phase='training';state.progress.episode=4;
state.result.history.forEach((row,index)=>{row.update={accepted:index===1,actor_change:.02};});
viz.render(state);
assert.match(node('#ft-learning-status').textContent,/Exploring a candidate policy/);
assert.match(node('#ft-learning-updates').textContent,/Policy updates: 1 applied \/ 4 attempted/);
assert.equal(points('exploration').length,4);assert.equal(points('evaluation').length,4);
node('#ft-chart-trial').value='1';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/Candidate \(deterministic\)/);
assert.match(node('#ft-legend-candidate-text').textContent,/deterministic/);
assert.match(node('#ft-legend-evaluation-text').textContent,/Retained policy/);
assert.match(node('#ft-chart-inspector').textContent,/Policy unchanged/);
node('#ft-chart-trial').value='2';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/Policy update applied/);
assertFiniteSvg();
""")


def test_partial_trial_does_not_invent_evaluation_or_future_trials(tmp_path, learning_state):
    state = learning_state
    state["result"]["history"] = state["result"]["history"][:2]
    state["result"]["history"][-1]["evaluation"] = None
    state["result"]["best"] = {**state["result"]["baseline"], "episode": 0}
    state["result"]["phase"] = "evaluation"
    state["result"]["done"] = False
    state["progress"].update({"phase": "evaluation", "episode": 2, "completed_episodes": 1})
    state["training_active"] = True
    run_viz(tmp_path, state, r"""
node('#ft-chart-candidates').checked=true;
viz.render(fixture);
assert.equal(points('exploration').length,2);assert.equal(points('evaluation').length,1);
assert.deepEqual(points('evaluation').map(item=>item.getAttribute('data-trial')),['1']);
const trials=node('#ft-chart-trial').children.map(item=>String(item.value));
assert.deepEqual(trials,['0','1','2']);
node('#ft-chart-trial').value='2';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/pending|not yet|not evaluated|waiting|—/i);
assertFiniteSvg();
""")


@pytest.mark.parametrize("reward", [-28.5, 0, 27.9])
def test_constant_rewards_and_optional_zero_axis_remain_finite(tmp_path, learning_state, reward):
    state = learning_state
    for value in [state["result"]["baseline"], state["result"]["best"]]:
        value["return"] = reward
    for trial in state["result"]["history"]:
        trial["training"]["return"] = reward
        trial["evaluation"]["return"] = reward
    run_viz(tmp_path, state, r"""
viz.render(fixture);assertFiniteSvg();
const low=Number(svg().getAttribute('data-y-min')),high=Number(svg().getAttribute('data-y-max'));
assert.ok(high>low);assert.ok(low<=fixture.result.baseline.return&&high>=fixture.result.baseline.return);
node('#ft-chart-zero').checked=true;fire(node('#ft-chart-zero'),'change');
assert.ok(Number(svg().getAttribute('data-y-min'))<=0);
assert.ok(Number(svg().getAttribute('data-y-max'))>=0);assertFiniteSvg();
""")


def test_mouse_keyboard_and_select_inspection_use_the_same_trial(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
node('#ft-chart-candidates').checked=true;
viz.render(fixture);assert.equal(String(selected()),'4');
fire(points('evaluation')[0],'click');assert.equal(String(selected()),'1');
assert.match(node('#ft-chart-inspector').textContent,/1/);
let point=points('exploration').find(item=>item.getAttribute('data-trial')==='3');
assert.equal(point.getAttribute('tabindex'),'0');
assert.ok(point.getAttribute('aria-label'));
fire(point,'keydown',{key:'Enter'});assert.equal(String(selected()),'3');
point=points('evaluation').find(item=>item.getAttribute('data-trial')==='2');
fire(point,'keydown',{key:' '});assert.equal(String(selected()),'2');
node('#ft-chart-trial').value='0';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/clone|baseline/i);
viz.render(fixture);assert.equal(String(selected()),'0');
""")


def test_best_curve_retains_baseline_when_all_updates_are_worse(tmp_path, learning_state):
    state = learning_state
    state["result"]["baseline"]["return"] = 5
    state["result"]["best"] = {**state["result"]["baseline"], "episode": 0}
    run_viz(tmp_path, state, r"""
viz.render(fixture);
const best=path('best').getAttribute('d');
// A flat best-so-far curve must not follow any worse updated policy.
const start=best.match(/^M\s+\S+\s+(\S+)/);
const ys=[Number(start[1]),...Array.from(best.matchAll(/V\s+(\S+)/g),match=>Number(match[1]))];
assert.ok(ys.length>=2);assert.equal(new Set(ys).size,1);
node('#ft-chart-trial').value='0';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/Original clone · Accuracy \/ speed score 5/);
assertFiniteSvg();
""")


def test_reset_clears_selection_and_normal_playback_retains_learning_results(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
viz.render(fixture);fire(points('evaluation')[0],'click');
const before=path('evaluation').getAttribute('d');
const watching=clone(fixture);watching.rollout={active:true,policy:'best'};watching.paused=false;
viz.render(watching);assert.equal(path('evaluation').getAttribute('d'),before);assert.equal(String(selected()),'1');
viz.reset();assert.equal(node('#ft-learning-viz').hidden,true);assert.equal(svg().children.length,0);
assert.equal(node('#ft-chart-trial').disabled,true);
assert.equal(node('#ft-chart-trial').children[0].value,'');
viz.render(fixture);assert.equal(String(selected()),'4');
viz.render({result:null});assert.equal(node('#ft-learning-viz').hidden,true);
""")


def test_empty_and_nonfinite_records_are_not_plotted(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
node('#ft-chart-candidates').checked=true;
viz.render(null);assert.equal(node('#ft-learning-viz').hidden,true);
viz.render({result:{baseline:null,history:[]}});assertFiniteSvg();
const state=clone(fixture);
state.result.history[0].training.return=NaN;
state.result.history[1].evaluation.return=Infinity;
state.result.history[2].training.return=null;
state.result.history[3].evaluation.return=undefined;
viz.render(state);assert.equal(points('exploration').length,2);assert.equal(points('evaluation').length,2);
assertFiniteSvg();
""")


def test_baseline_only_stop_keeps_the_original_result_available(tmp_path, learning_state):
    state = learning_state
    state["result"]["history"] = []
    state["result"]["baseline"]["return"] = 0
    state["result"]["best"] = {**state["result"]["baseline"], "episode": 0}
    state["result"]["done"] = False
    state["result"]["phase"] = "training"
    state["progress"].update({"phase": "training", "episode": 1, "completed_episodes": 0})
    state["training_stopped"] = True
    run_viz(tmp_path, state, r"""
viz.render(fixture);assert.equal(node('#ft-learning-viz').hidden,false);
assert.equal(points('baseline').length,1);assert.equal(points('evaluation').length,0);
assert.equal(points('exploration').length,0);assert.equal(selected(),'0');
assert.match(node('#ft-learning-status').textContent,/stopped.*0 iterations/i);
assert.match(node('#ft-chart-inspector').textContent,/Original clone · Accuracy \/ speed score 0.000/);
assertFiniteSvg();
""")


def test_live_phase_changes_leave_completed_points_and_user_selection_stable(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
const state=clone(fixture);state.training_active=true;state.paused=false;
state.result.done=false;state.result.phase='training';state.progress.phase='training';
viz.render(state);fire(points('evaluation')[0],'click');
const firstPoint=points('evaluation')[0];
assert.equal(node('#ft-stage-training').getAttribute('data-active'),'true');
assert.equal(node('#ft-stage-training').getAttribute('aria-current'),'step');
state.progress.elapsed_seconds=9.5;state.progress.reward=123;
viz.render(state);
assert.equal(points('evaluation')[0],firstPoint);assert.equal(selected(),'1');
assert.equal(points('evaluation').length,4);
state.training_paused=true;state.paused=true;viz.render(state);
assert.match(node('#ft-learning-status').textContent,/Paused/);
state.training_paused=false;state.paused=false;state.progress.phase='evaluation';viz.render(state);
assert.equal(node('#ft-stage-training').getAttribute('aria-current'),null);
assert.equal(node('#ft-stage-evaluation').getAttribute('aria-current'),'step');
assert.equal(points('evaluation')[0],firstPoint);assertFiniteSvg();
""")


@pytest.mark.parametrize("width,ticks", [(360, [0, 20, 40, 60, 80, 100]), (680, list(range(0, 101, 10)))])
def test_hundred_iterations_keep_current_evaluation_distinct_and_missing_results_absent(
    tmp_path, learning_state, width, ticks,
):
    run_viz(tmp_path, learning_state, r"""
node('#ft-chart-candidates').checked=true;
const state=clone(fixture),history=[];
for(let iteration=1;iteration<=100;iteration++)history.push({
  episode:iteration,
  training:{return:-10+iteration/100,raw_return:10000+iteration},
  evaluation:iteration===100?null:{return:10+(iteration%7)/10,raw_return:20000+iteration},
  update:{actor_change:.01}
});
Object.assign(state.result,{history,episodes:100,episode:100,phase:'evaluation',done:false,
  baseline:{return:9,raw_return:9999},best:{return:10.6,raw_return:20006,episode:6}});
Object.assign(state.progress,{episodes:100,episode:100,phase:'evaluation',completed_episodes:99});
state.training_active=true;state.paused=false;svg().clientWidth=WIDTH;
viz.render(state);
assert.equal(numeric(svg(),'data-x-max'),100);
assert.equal(points('exploration').length,100);assert.equal(points('evaluation').length,99);
assert.equal(node('#ft-chart-trial').children.length,101);assert.equal(selected(),'100');
assert.equal(node('#ft-chart-trial').children.at(-1).textContent,'Iteration 100');
assert.equal(points('evaluation').at(-1).getAttribute('data-trial'),'99');
assert.ok(numeric(svg(),'data-y-max')<20,'Raw, undiscounted returns must not enter this chart');
assert.equal(numeric(points('evaluation')[6],'data-value'),10,'Show a worsening current policy, not best so far');
assert.notEqual(path('evaluation').getAttribute('d'),path('best').getAttribute('d'));
assert.match(node('#ft-chart-inspector').textContent,/Current policy \(no noise\).*score —/);
assert.match(node('#ft-learning-status').textContent,/iteration 100 \/ 100/);
const ticks=descendants().filter(item=>item.getAttribute('data-axis')==='trial').map(item=>Number(item.textContent));
assert.deepEqual(ticks,TICKS);
assert.ok(descendants().some(item=>item.textContent==='Accuracy / speed score'));
const earlier=points('evaluation')[0];viz.render(state);
assert.equal(points('evaluation')[0],earlier,'Unchanged history should not rebuild 100 iterations');
state.result.history[99].evaluation={return:10.2,raw_return:20100};viz.render(state);
assert.equal(points('evaluation').length,100);assert.equal(selected(),'100');
assert.match(node('#ft-chart-inspector').textContent,/Current policy \(no noise\).*score 10.200000/);
assertFiniteSvg();
""".replace("WIDTH", str(width)).replace("TICKS", json.dumps(ticks)))


def test_inspector_keeps_fine_measurements_and_rejected_candidate_deltas(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
const state=clone(fixture);state.result.strategy='policy_search';state.result.history=state.result.history.slice(0,2);
const first={return:779.1234567,fill_ml:699.12345678,seconds:32.03125,success:true};
state.result.baseline={return:51.0000004,fill_ml:672.0123456,seconds:36.0625,success:true};
state.result.history[0]={episode:1,training:first,evaluation:first,update:{accepted:true}};
state.result.history[1]={episode:2,training:{...first,return:779.1234547,fill_ml:699.12342345,seconds:32.0625},
  evaluation:first,update:{accepted:false,candidate_gains:[1.123456789,1.0000023]}};
const original=JSON.stringify(state);viz.render(state);
assert.equal(points('exploration').length,2);assert.equal(points('evaluation').length,2);
assert.notEqual(numeric(points('exploration')[0],'data-value'),numeric(points('exploration')[1],'data-value'));
assert.equal(numeric(points('evaluation')[0],'cy'),numeric(points('evaluation')[1],'cy'),'Rejected candidate retains the same policy');
const inspector=node('#ft-chart-inspector').textContent;
assert.match(inspector,/Candidate \(deterministic\).*score 779.123455/);
assert.match(inspector,/Retained policy \(no noise\).*score 779.123457/);
assert.match(inspector,/Original clone.*score 51.000000/);
assert.match(inspector,/Cup 699.1234 mL.*\|error\| 0.8766 mL.*32.06250 s/);
assert.match(inspector,/candidate -0.000002 · retained 0.000000/);
assert.match(inspector,/time Δ 0.00000 s/);
assert.match(inspector,/approach\/pour 1.123457× · return 1.000002×/);
assert.match(inspector,/Policy unchanged/);
node('#ft-chart-trial').value='1';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/candidate \+728.123456 · retained \+728.123456/);
assert.match(node('#ft-chart-inspector').textContent,/time Δ -4.03125 s/);
assert.equal(JSON.stringify(state),original,'Inspection must not round or mutate recorded values');assertFiniteSvg();
""")


def test_error_and_duration_plot_actual_measurements_not_best_score_surrogates(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
const state=clone(fixture);state.result.history=state.result.history.slice(0,3);
state.result.baseline.fill_ml=672;state.result.baseline.seconds=36.0625;
state.result.history[0].training.fill_ml=703.456789;state.result.history[0].evaluation.fill_ml=701.234567;
state.result.history[1].training.fill_ml=699.876543;state.result.history[1].evaluation={return:null,fill_ml:700,seconds:10};
state.result.history[2].training.fill_ml=null;state.result.history[2].evaluation=null;
state.result.history[0].evaluation.seconds=32.03125;
viz.render(state);node('#ft-chart-metric').value='error';fire(node('#ft-chart-metric'),'change');
assert.equal(points('exploration').length,2);assert.equal(points('evaluation').length,1);
assert.equal(numeric(points('exploration')[0],'data-value'),Math.abs(700-703.456789));
assert.equal(numeric(points('evaluation')[0],'data-value'),Math.abs(700-701.234567));
assert.equal(numeric(points('baseline')[0],'data-value'),28);
assert.equal(path('best'),undefined);assert.equal(node('#ft-legend-best').hidden,true);
assert.match(node('#ft-chart-direction').textContent,/Lower is better.*check success/);
assert.match(node('#ft-chart-view-note').textContent,/Best-score guide.*only.*score/);
assert.match(points('evaluation')[0].getAttribute('aria-label'),/Absolute target error \(mL\) 1.2346/);
node('#ft-chart-metric').value='time';fire(node('#ft-chart-metric'),'change');
assert.equal(points('evaluation').length,1);assert.equal(numeric(points('evaluation')[0],'data-value'),32.03125);
assert.equal(numeric(points('baseline')[0],'data-value'),36.0625);
assert.match(points('evaluation')[0].getAttribute('aria-label'),/Duration \(s\) 32.03125/);
assert.equal(path('best'),undefined);assertFiniteSvg();
""")


def test_recent_detail_rescales_real_points_and_explicitly_omits_old_guides(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
const state=clone(fixture);state.result.strategy='policy_search';state.result.baseline.return=51;
state.result.history=Array.from({length:20},(_,index)=>({episode:index+1,
  training:{return:index<10?-100:779+(index-10)*.000003},
  evaluation:{return:index<10?10000:779+(index-10)*.000002},update:{accepted:false}}));
state.result.episodes=100;viz.render(state);
assert.equal(numeric(svg(),'data-x-max'),20,'Pending iterations must not squeeze completed measurements');
assert.ok(numeric(svg(),'data-y-max')>10000);
node('#ft-chart-focus').value='recent';fire(node('#ft-chart-focus'),'change');
assert.deepEqual(points('exploration').map(item=>numeric(item,'data-trial')),Array.from({length:10},(_,index)=>index+11));
assert.equal(points('evaluation').length,10);assert.equal(points('baseline').length,0);assert.equal(path('best'),undefined);
assert.ok(!descendants().some(item=>item.classes.has('ft-chart-baseline')),'Omitted baseline must not be drawn offscreen');
assert.equal(numeric(svg(),'data-x-min'),11);assert.equal(numeric(svg(),'data-x-max'),20);
assert.ok(numeric(svg(),'data-y-min')>778.999);assert.ok(numeric(svg(),'data-y-max')<779.001);
assert.match(node('#ft-chart-view-note').textContent,/last 10 completed trials.*clone and best-score guides omitted/);
const ticks=descendants().filter(item=>item.getAttribute('data-axis')==='reward').map(item=>item.textContent);
assert.equal(new Set(ticks).size,ticks.length,'Fine axes must show distinct actual numeric tick values');
node('#ft-chart-trial').value='1';fire(node('#ft-chart-trial'),'change');
assert.match(node('#ft-chart-inspector').textContent,/score 10000.000000/,'Earlier results remain inspectable');
node('#ft-chart-focus').value='recent5';fire(node('#ft-chart-focus'),'change');
assert.equal(points('exploration').length,5);assert.equal(numeric(svg(),'data-x-min'),16);
assert.match(node('#ft-chart-view-note').textContent,/last 5 completed trials/);
node('#ft-chart-zero').checked=true;fire(node('#ft-chart-zero'),'change');
assert.ok(numeric(svg(),'data-y-min')<=0);assert.equal(points('baseline').length,0);
node('#ft-chart-focus').value='all';fire(node('#ft-chart-focus'),'change');
assert.equal(points('exploration').length,20);assert.equal(points('baseline').length,1);assert.ok(path('best'));
assert.equal(selected(),'1');assertFiniteSvg();
""")


def test_ppo_candidate_labels_do_not_claim_deterministic_evaluation(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
fixture.result.strategy='ppo';fixture.result.algorithm='bounded_speed_ppo_actor_critic';viz.render(fixture);
assert.match(node('#ft-legend-candidate-text').textContent,/exploration · noisy/);
assert.match(node('#ft-legend-evaluation-text').textContent,/Current policy · no noise/);
assert.match(node('#ft-chart-explanation').textContent,/Orange measures noisy exploration/);
assert.doesNotMatch(node('#ft-chart-explanation').textContent,/deterministic candidate/);
assert.match(points('exploration')[0].getAttribute('aria-label'),/noisy exploration/);
assert.match(node('#ft-chart-inspector').textContent,/Exploration \(noisy\)/);
assert.equal(points('exploration').length,4);assertFiniteSvg();
""")


def test_selector_changes_cache_and_reset_restore_fine_view_defaults(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
viz.render(fixture);node('#ft-chart-metric').value='error';fire(node('#ft-chart-metric'),'change');
assert.equal(svg().getAttribute('data-metric'),'error');
node('#ft-chart-focus').value='recent5';fire(node('#ft-chart-focus'),'change');
const point=points('evaluation')[0];viz.render(fixture);assert.equal(points('evaluation')[0],point);
node('#ft-chart-candidates').checked=false;fire(node('#ft-chart-candidates'),'change');
viz.reset();assert.equal(node('#ft-chart-metric').value,'score');assert.equal(node('#ft-chart-focus').value,'all');
assert.equal(node('#ft-chart-candidates').checked,true);
viz.render(fixture);assert.equal(points('exploration').length,4);assert.equal(points('baseline').length,1);
assert.equal(svg().getAttribute('data-metric'),'score');assertFiniteSvg();
""")


def test_sub_micro_score_changes_remain_visible_and_never_claim_zero_delta(tmp_path, learning_state):
    run_viz(tmp_path, learning_state, r"""
const state=clone(fixture);state.result.strategy='policy_search';state.result.history=state.result.history.slice(0,2);
state.result.baseline.return=779;
state.result.history[0].training.return=779.00000001;state.result.history[0].evaluation.return=779.00000001;
state.result.history[1].training.return=779.00000002;state.result.history[1].evaluation.return=779.00000002;
svg().clientWidth=320;viz.render(state);node('#ft-chart-focus').value='recent5';fire(node('#ft-chart-focus'),'change');
const inspector=node('#ft-chart-inspector').textContent;
assert.match(inspector,/candidate \+[^ ]+e-8 · retained \+[^ ]+e-8/);
assert.doesNotMatch(inspector,/candidate \+0.000000/);
const ticks=descendants().filter(item=>item.getAttribute('data-axis')==='reward');
assert.equal(new Set(ticks.map(item=>item.textContent)).size,5,'A tiny real range needs distinct numeric ticks');
assert.ok(numeric(points('evaluation')[1],'cy')<numeric(points('evaluation')[0],'cy'));
assert.equal(points('evaluation')[1].children[0].textContent,'Accuracy / speed score: 779.00000002');
assert.ok(all(node('#ft-chart-inspector')).some(item=>item.getAttribute('title')==='Recorded score: 779.00000002'));
assertFiniteSvg();
""")
