"""Regression tests for one simulation/recording timeline, without prediction."""
import base64
import json
import shutil
import subprocess
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_browser import (
    CONTROLLER_JAVASCRIPT,
    browser_bundle,
)
from kaist_rl_lab.apps.coffee_browser_runtime import BROWSER_DT, BrowserRuntime
from kaist_rl_lab.apps.coffee_demonstrations import read_demonstration
from kaist_rl_lab.envs import CoffeePouringEnv
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT


def call(runtime, kind, **kwargs):
    return json.loads(runtime.dispatch(json.dumps({"kind": kind, **kwargs})))


def control(runtime, sequence, motors, paused=False, kind="motor", generation=1):
    return call(runtime, kind, sequence=sequence, motors=motors, paused=paused,
                generation=generation)


def test_authoritative_motion_hold_reversal_pause_and_reset():
    runtime = BrowserRuntime()
    try:
        env = runtime.session.env
        initial = env.joint_angles.copy()
        control(runtime, 1, [1, 0, 0, 0, 0, 0], paused=True)
        call(runtime, "tick")
        np.testing.assert_array_equal(env.joint_angles, initial)
        control(runtime, 2, [1, 0, 0, 0, 0, 0])
        previous = initial[0]
        for _ in range(60):
            frame = call(runtime, "tick")["snapshot"]
            actual = frame["state"]["joint_angles_rad"][0]
            assert actual >= previous
            previous = actual
        held = env.joint_angles.copy()
        control(runtime, 3, [0] * 6)
        for _ in range(10):
            call(runtime, "tick")
            np.testing.assert_array_equal(env.joint_angles, held)
        control(runtime, 4, [-1, 0, 0, 0, 0, 0])
        for _ in range(60):
            call(runtime, "tick")
            assert env.joint_angles[0] <= previous
            previous = env.joint_angles[0]
        control(runtime, 6, [0] * 6, paused=True)
        control(runtime, 5, [1] * 6)
        step = env.elapsed_steps
        call(runtime, "tick")
        assert env.elapsed_steps == step
        assert runtime.session.paused
        np.testing.assert_array_equal(runtime.session.motors, 0)
        control(runtime, 7, [0] * 6, kind="reset")
        assert runtime.session.env.dt == BROWSER_DT
        assert runtime.session.env.elapsed_steps == 0
        assert runtime.session.generation == 2
        assert not runtime.session.paused
        control(runtime, 8, [1] * 6, generation=1)
        np.testing.assert_array_equal(runtime.session.motors, 0)
    finally:
        runtime.session.close()


def test_recording_replays_the_displayed_python_physics_exactly():
    runtime = BrowserRuntime()
    reference = CoffeePouringEnv(dt=BROWSER_DT, horizon=None)
    reference.reset(seed=7001, options={"target_fill": 0.7})
    rng = np.random.default_rng(88)
    rendered = []
    try:
        for sequence in range(1, 41):
            action = rng.integers(-1, 2, size=6).tolist()
            control(runtime, sequence, action)
            for _ in range(4):
                snapshot = call(runtime, "tick")["snapshot"]
                rendered.append(snapshot["state"]["joint_angles_rad"])
        saved = call(runtime, "save", participant="replay-check")
        arrays, metadata = read_demonstration(base64.b64decode(saved["archive"]))
        assert metadata["dt"] == BROWSER_DT
        assert metadata["physics_substep"] == 1/64
        assert len(arrays["actions"]) == len(rendered) == 160
        for i, action in enumerate(arrays["actions"]):
            obs, reward, _, _, _ = reference.step(action)
            np.testing.assert_array_equal(reference.joint_angles, rendered[i])
            np.testing.assert_array_equal(obs, arrays["next_observations"][i])
            assert reward == pytest.approx(arrays["rewards"][i], abs=1e-6)
        assert arrays["truncated"][-1]
        assert call(runtime, "save", participant="changed")["archive"] == saved["archive"]
        call(runtime, "tick")
        assert len(runtime.session.trajectory) == 160
    finally:
        runtime.session.close()
        reference.close()


def test_bundle_contains_installed_physics_source():
    with ZipFile(BytesIO(base64.b64decode(browser_bundle()))) as archive:
        source = archive.read("kaist_rl_lab/envs/coffee_pouring.py").decode()
        assert "class CoffeePouringEnv" in source
        assert "class BrowserRuntime" in archive.read(
            "kaist_rl_lab/apps/coffee_browser_runtime.py"
        ).decode()
        assert all(name.startswith("kaist_rl_lab/") and name.endswith(".py")
                   for name in archive.namelist())


def test_browser_never_extrapolates_or_rewinds_on_delayed_messages(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser controller tests")
    runtime = BrowserRuntime()
    initial = json.loads(runtime.snapshot())
    control(runtime, 2, [1, 0, 0, 0, 0, 0])
    moving = call(runtime, "tick")
    held = control(runtime, 3, [0] * 6)
    reset = control(runtime, 4, [0] * 6, kind="reset")
    runtime.session.close()
    script = tmp_path / "canvas.js"
    script.write_text(CANVAS_JAVASCRIPT + CONTROLLER_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text('''
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const f=JSON.parse(fs.readFileSync(0,'utf8')),sent=[];
const nodes=new Map();
function node(key) {if(key==='.coffee-canvas')return null;
 if(!nodes.has(key))nodes.set(key,{textContent:'',value:'',addEventListener(){}});return nodes.get(key);}
const context=vm.createContext({
 element:{querySelector:node,querySelectorAll:()=>[],addEventListener(){},isConnected:true},
 document:{addEventListener(){},body:{}}, window:{addEventListener(){}},
 props:{value:{bundle:'',worker:'',collecting:true}}, watch(){},trigger(){},
 Blob:class {}, URL:{createObjectURL:()=>'',revokeObjectURL(){}},
 Worker:class {postMessage(value){sent.push(value)}}, MutationObserver:class{observe(){}},
 requestAnimationFrame:fn=>fn(),setTimeout:()=>1,clearTimeout(){},
});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const evaluate=s=>vm.runInContext(s,context);
function receive(data){context.data=data;evaluate('worker.onmessage({data})');}
const state=()=>JSON.parse(evaluate('JSON.stringify(displayState)'));
receive(f.initial);
assert.equal(state().paused,true);
evaluate('sendControl("motor",0,1);sendControl("pause")');
assert.equal(sent.at(-1).sequence,2);
// A late paused update cannot move the pose or undo selected controls.
receive(f.initial);assert.equal(evaluate('desiredPaused'),false);
assert.deepEqual(state().q,f.initial.snapshot.state.joint_angles_rad);
receive(f.moving);
assert.deepEqual(state().q,f.moving.snapshot.state.joint_angles_rad);
evaluate('sendControl("motor",0,0)');receive(f.held);
const q=state().q;
receive(f.moving);assert.deepEqual(state().q,q);
evaluate('sendControl("reset")');receive(f.reset);
receive(f.held);assert.equal(state().generation,2);assert.equal(state().step,0);
''')
    subprocess.run([node, str(runner), str(script)], input=json.dumps({
        "initial": initial, "moving": moving, "held": held, "reset": reset,
    }), text=True, capture_output=True, check=True, timeout=30)
