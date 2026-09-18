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
let nextTimer=0,now=100,currentCommand=null,failNext=false;
let frame={snapshot:{playback:{paused:true,running:true}},finetuning:{training_running:false}};
const clone=value=>JSON.parse(JSON.stringify(value));
const fakePython={
  globals:{set(name,value){assert.equal(name,'_coffee_command');currentCommand=JSON.parse(value);}},
  runPython(code){
    assert.equal(code,'coffee_runtime.dispatch(_coffee_command)');commands.push(clone(currentCommand));
    if(failNext){failNext=false;throw Error('Simulated physics failure');}
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
