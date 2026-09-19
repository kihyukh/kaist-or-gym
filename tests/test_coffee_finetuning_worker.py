"""The shared worker yields between RL chunks and cancels work when paused."""

import base64
import json
import shutil
import subprocess
from io import BytesIO
from zipfile import ZipFile

import pytest

from kaist_rl_lab.apps.coffee_browser import WORKER_JAVASCRIPT, browser_bundle

NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const timers=new Map(),messages=[],commands=[];
let nextTimer=0,now=100,currentCommand=null,failNext=false,stepCost=0,advancePhysics=false;
let frame={snapshot:{playback:{paused:true,running:true}},finetuning:{training_running:false}};
const clone=value=>JSON.parse(JSON.stringify(value));
const fakePython={
  globals:{set(name,value){assert.equal(name,'_coffee_command');currentCommand=JSON.parse(value);}},
  runPython(code){
    assert.equal(code,'coffee_runtime.dispatch(_coffee_command)');commands.push(clone(currentCommand));
    if(failNext){failNext=false;throw Error('Simulated physics failure');}
    if(advancePhysics&&['ft-step','tick'].includes(currentCommand.kind)){
      const steps=currentCommand.max_steps||1;now+=steps*stepCost;
      if(currentCommand.kind==='ft-step')frame.finetuning.progress.total_steps+=steps;
      else if(frame.cloning_agent)frame.cloning_agent.step+=steps;
      else if(frame.finetuning)frame.finetuning.rollout.elapsed_seconds+=steps/32;
    }
    return JSON.stringify(frame);
  }
};
const self={};
const context=vm.createContext({self,fakePython,postMessage:value=>messages.push(clone(value)),
  setTimeout(callback,delay){const id=++nextTimer;timers.set(id,{callback,delay});return id;},
  clearTimeout:id=>timers.delete(id),performance:{now:()=>now},console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
vm.runInContext('python=fakePython',context);
async function send(kind,nextFrame,extra={}){
  if(nextFrame)frame=clone(nextFrame);
  await self.onmessage({data:{kind,...extra}});
}
function step(){
  assert.equal(timers.size,1,'Only one scheduled chunk or simulation tick is allowed');
  const [id,timer]=timers.entries().next().value;timers.delete(id);timer.callback();
}
const pending=()=>[...timers.values()];
const idle={snapshot:{playback:{paused:true,running:true}},finetuning:{training_running:false}};
const training={snapshot:{playback:{paused:false,running:true}},finetuning:{training_running:true}};
const paused={snapshot:{playback:{paused:true,running:true}},finetuning:{training_running:false}};
const running={snapshot:{playback:{paused:false,running:true}},finetuning:{training_running:false}};
const complete={snapshot:{playback:{paused:true,running:false}},finetuning:{training_running:false}};
async function main(){
"""


def run_worker(tmp_path, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for worker scheduler regression tests")
    worker = tmp_path / "coffee-worker.js"
    worker.write_text(WORKER_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(
        NODE_HARNESS + assertions
        + "\n}\nmain().catch(error=>{console.error(error);process.exit(1)});"
    )
    result = subprocess.run(
        [node, str(runner), str(worker)], text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_schedules_one_cooperative_chunk_at_a_time(tmp_path):
    run_worker(tmp_path, r"""
await send('ft-load',idle);assert.equal(timers.size,0);
await send('ft-train',training,{episodes:4});assert.equal(timers.size,1);
assert.equal(pending()[0].delay,0);
for(let index=0;index<3;index++){
  step();assert.equal(commands.at(-1).kind,'ft-step');
  assert.equal(timers.size,1);assert.equal(pending()[0].delay,0);
}
// Loading/result frames never cause an autonomous save or a normal-speed tick.
assert.equal(commands.some(command=>command.kind==='save'||command.kind==='tick'),false);
frame=complete;step();assert.equal(timers.size,0);
assert.equal(messages.at(-1).snapshot.playback.paused,true);
""")


@pytest.mark.parametrize("kind", ["ft-pause", "ft-stop", "ft-reset", "ft-load"])
def test_pause_stop_reset_and_model_change_cancel_the_pending_chunk(tmp_path, kind):
    run_worker(tmp_path, r"""
await send('ft-train',training);assert.equal(timers.size,1);
await send(KIND,paused,{paused:true});assert.equal(timers.size,0);
const count=commands.length;
assert.equal(commands.filter(command=>command.kind==='ft-step').length,0);
await send('snapshot',paused);assert.equal(timers.size,0);assert.equal(commands.length,count+1);
await send('ft-pause',training,{paused:false});assert.equal(timers.size,1);
step();assert.equal(commands.at(-1).kind,'ft-step');assert.equal(timers.size,1);
""".replace("KIND", json.dumps(kind)))


def test_comparison_uses_32hz_ticks_without_background_catchup(tmp_path):
    run_worker(tmp_path, r"""
await send('ft-run',running,{policy:'base'});assert.equal(timers.size,1);
step();assert.equal(commands.at(-1).kind,'tick');assert.equal(pending()[0].delay,1000/32);
now=10000;step();assert.equal(commands.at(-1).kind,'tick');
assert.equal(timers.size,1);assert.equal(pending()[0].delay,1000/32);
await send('ft-pause',paused,{paused:true});assert.equal(timers.size,0);
await send('ft-pause',running,{paused:false});assert.equal(timers.size,1);
step();assert.equal(commands.at(-1).kind,'tick');
frame=complete;step();assert.equal(timers.size,0);
assert.equal(commands.some(command=>command.kind==='ft-step'),false);
""")


def test_restarting_or_switching_mode_replaces_pending_timer(tmp_path):
    run_worker(tmp_path, r"""
await send('ft-train',training);const first=[...timers.keys()][0];
await send('ft-run',running,{policy:'best'});assert.equal(timers.size,1);
assert.notEqual([...timers.keys()][0],first);step();assert.equal(commands.at(-1).kind,'tick');
await send('ft-train',training);assert.equal(timers.size,1);step();assert.equal(commands.at(-1).kind,'ft-step');
await send('ft-reset',idle);assert.equal(timers.size,0);
""")


def test_runtime_failure_stops_scheduling_and_reports_error(tmp_path):
    run_worker(tmp_path, r"""
await send('ft-train',training);failNext=true;step();
assert.equal(timers.size,0);assert.match(messages.at(-1).error,/Simulated physics failure/);
vm.runInContext('schedule()',context);assert.equal(timers.size,0);
""")


def test_browser_bundle_contains_numpy_finetuning_runtime():
    with ZipFile(BytesIO(base64.b64decode(browser_bundle()))) as archive:
        names = set(archive.namelist())
        assert "kaist_rl_lab/apps/coffee_finetuning.py" in names
        assert "kaist_rl_lab/apps/coffee_finetuning_runtime.py" in names
    assert "FineTuningRuntime" in WORKER_JAVASCRIPT
    assert "data.mode === 'finetuning'" in WORKER_JAVASCRIPT



@pytest.mark.parametrize("speed", [4, 8])
@pytest.mark.parametrize("kind", ["ft-train", "ft-run"])
def test_accelerated_finetuning_batches_real_steps_at_requested_pace(tmp_path, speed, kind):
    run_worker(tmp_path, r"""
const current=clone(KIND==='ft-train'?training:running);
Object.assign(current.finetuning,{playback_speed:SPEED,progress:{total_steps:0},rollout:{elapsed_seconds:0}});
advancePhysics=true;stepCost=2;
await send(KIND,current);step();
assert.equal(commands.at(-1).kind,KIND==='ft-train'?'ft-step':'tick');
assert.equal(commands.at(-1).max_steps,SPEED);
assert.equal(pending()[0].delay,1000/32-SPEED*stepCost);
now+=pending()[0].delay;step();
assert.equal(commands.at(-1).max_steps,SPEED);
const advanced=KIND==='ft-train'?messages.at(-1).finetuning.progress.total_steps:
  messages.at(-1).finetuning.rollout.elapsed_seconds*32;
assert.equal(advanced,SPEED*2);
// A delayed foreground callback does not replay missed wall-clock intervals.
now=10000;step();assert.equal(commands.at(-1).max_steps,SPEED);
assert.equal(pending()[0].delay,1000/32-SPEED*stepCost);
await send('ft-pause',paused,{paused:true});assert.equal(timers.size,0);
""".replace("SPEED", str(speed)).replace("KIND", json.dumps(kind)))


def test_fastest_mode_adapts_chunk_size_and_yields_between_bounded_batches(tmp_path):
    run_worker(tmp_path, r"""
const current=clone(training);
Object.assign(current.finetuning,{playback_speed:0,progress:{total_steps:0},rollout:{elapsed_seconds:0}});
advancePhysics=true;stepCost=.5;
await send('ft-train',current);
for(let count=0;count<8;count++){
  step();assert.equal(pending()[0].delay,0);
  assert.ok(commands.at(-1).max_steps>=1&&commands.at(-1).max_steps<=32);
}
assert.equal(commands.at(-1).max_steps,32);
const chunks=commands.filter(command=>command.kind==='ft-step');
assert.equal(messages.at(-1).finetuning.progress.total_steps,chunks.reduce((total,item)=>total+item.max_steps,0));
await send('ft-pause',paused,{paused:true});assert.equal(timers.size,0);
""")


def test_expensive_steps_reduce_work_per_chunk_to_keep_controls_responsive(tmp_path):
    run_worker(tmp_path, r"""
const current=clone(training);
Object.assign(current.finetuning,{playback_speed:0,progress:{total_steps:0},rollout:{elapsed_seconds:0}});
advancePhysics=true;stepCost=12;
await send('ft-train',current);step();const initial=commands.at(-1).max_steps;
for(let count=0;count<4;count++)step();
assert.ok(commands.at(-1).max_steps<initial);
assert.ok(commands.at(-1).max_steps*stepCost<=50);
assert.equal(timers.size,1);assert.equal(pending()[0].delay,0);
await send('ft-stop',paused);assert.equal(timers.size,0);
""")


def test_speed_changes_replace_pending_timer_and_leave_student_ticks_at_one_step(tmp_path):
    run_worker(tmp_path, r"""
const current=clone(running);current.finetuning.playback_speed=4;
await send('ft-run',current);step();assert.equal(commands.at(-1).max_steps,4);
const old=[...timers.keys()][0];current.finetuning.playback_speed=0;
await send('ft-speed',current,{speed:0});
assert.equal(timers.size,1);assert.notEqual([...timers.keys()][0],old);assert.equal(pending()[0].delay,0);
step();assert.ok(commands.at(-1).max_steps>4);
await send('pause',{snapshot:{playback:{paused:true,running:true}}});assert.equal(timers.size,0);
await send('reset',{snapshot:{playback:{paused:false,running:true}}});step();
assert.deepEqual(commands.at(-1),{kind:'tick'});assert.equal(pending()[0].delay,1000/32);
""")


def test_cloning_defaults_to_four_real_steps_and_paces_snapshots_at_32hz(tmp_path):
    run_worker(tmp_path, r"""
const cloning={snapshot:{playback:{paused:false,running:true}},cloning_agent:{step:0}};
advancePhysics=true;stepCost=2;
await send('cloning-start',cloning);step();
assert.deepEqual(commands.at(-1),{kind:'tick',max_steps:4});
assert.equal(messages.at(-1).cloning_agent.step,4);
assert.equal(pending()[0].delay,1000/32-8);
now=10000;step();assert.equal(commands.at(-1).max_steps,4);
assert.equal(pending()[0].delay,1000/32-8);
const previous=[...timers.keys()][0];frame.cloning_agent.playback_speed=8;
await send('cloning-speed',null,{speed:8});
assert.notEqual([...timers.keys()][0],previous);step();assert.equal(commands.at(-1).max_steps,8);
frame.cloning_agent.playback_speed=1;await send('cloning-speed',null,{speed:1});
step();assert.equal(commands.at(-1).max_steps,1);
frame.snapshot.playback.paused=true;await send('cloning-pause',null,{paused:true});
assert.equal(timers.size,0);
""")


def test_cloning_batches_adapt_and_student_random_controls_remain_one_step(tmp_path):
    run_worker(tmp_path, r"""
advancePhysics=true;stepCost=20;
const cloning={snapshot:{playback:{paused:false,running:true}},cloning_agent:{step:0,playback_speed:4}};
await send('cloning-start',cloning);step();const initial=commands.at(-1).max_steps;
for(let index=0;index<6;index++)step();
assert.ok(commands.at(-1).max_steps<initial);
assert.ok(commands.at(-1).max_steps*stepCost<=50);
frame.snapshot.playback.running=false;frame.snapshot.playback.paused=true;step();
assert.equal(timers.size,0);
advancePhysics=false;
for(const kind of ['reset','random-start']){
  await send(kind,{snapshot:{playback:{paused:false,running:true}}});step();
  assert.deepEqual(commands.at(-1),{kind:'tick'});assert.equal(pending()[0].delay,1000/32);
}
""")
