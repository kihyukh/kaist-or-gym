"""Exercise the actual browser controller with delayed/out-of-order snapshots."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from kaist_rl_lab.apps.coffee_pouring_app import InteractiveSession
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT


def run_browser_script(tmp_path, script, fixture):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser-controller tests")
    source = tmp_path / "canvas.js"
    source.write_text(CANVAS_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text('''
const vm = require('node:vm'), fs = require('node:fs'), assert = require('node:assert/strict');
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));
let time = 0, sent = [];
const context = vm.createContext({
  performance: {now: () => time},
  window: {matchMedia: () => ({matches: false}), requestAnimationFrame: () => 1},
  element: {addEventListener() {}, querySelector() {return null;}, isConnected: true},
  props: {value: fixture.initial}, watch() {}, trigger: (kind, data) => sent.push(data),
});
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), context);
const evaluate = code => vm.runInContext(code, context);
const state = () => JSON.parse(evaluate('JSON.stringify(stateAt(performance.now()))'));
function snapshot(value) { context.props.value=value; evaluate('ingestSnapshot()'); }
''' + script)
    result = subprocess.run(
        [node, str(runner), str(source)], input=json.dumps(fixture), text=True,
        capture_output=True, timeout=30, check=True,
    )
    return result.stdout


def test_immediate_controls_survive_late_snapshots_and_stop_on_disconnect(tmp_path):
    session = InteractiveSession(7001, 700, start_paused=True)
    initial = session.animation_snapshot()
    try:
        run_browser_script(tmp_path, '''
const initialQ=state().q;
evaluate('sendControl("motor",0,1)'); time=16;
assert.deepEqual(state().q,initialQ, 'planning while paused must not move');
evaluate('sendControl("pause")'); time=32;
assert(state().q[0]>initialQ[0], 'motion starts on the next display frame without an acknowledgement');
assert.equal(sent.at(-1).sequence,2);
const late=structuredClone(fixture.initial);
late.playback.revision=10; late.playback.input_sequence=1;
snapshot(late);
assert.equal(state().motors[0],1);
assert.equal(state().paused,false, 'old paused snapshot cannot undo optimistic resume');
time=48; evaluate('sendControl("motor",0,0)'); const held=state().q;
time=100; assert.deepEqual(state().q,held, 'Hold has no local coasting');
evaluate('sendControl("motor",0,-1)'); time=116;
assert(state().q[0]<held[0], 'reversal starts without waiting for network');
assert.deepEqual(Array.from(sent.at(-1).motors),[-1,0,0,0,0,0]);
// Predict at most one second when a response is lost, then return to authority.
time=1200; state(); time=1400;
assert.deepEqual(state().q,initialQ);
assert.equal(state().paused,true);
''', {"initial": initial})
    finally:
        session.close()


def test_preview_respects_rigid_geometry_limits_and_contacts(tmp_path):
    session = InteractiveSession(7001, 700)
    rng = np.random.default_rng(52)
    cases = []
    try:
        for _ in range(100):
            motors = rng.choice([-1, 0, 1], size=6)
            session.motors[:] = motors
            session.advance()
            cases.append({"q": session.env.joint_angles.tolist(), "motors": motors.tolist()})
        output = run_browser_script(tmp_path, '''
const results = fixture.cases.map(({q,motors}) => {
  context.q=q; context.motors=motors;
  return evaluate('predictPose(q,motors,.2,targetState.geometry)');
});
process.stdout.write(JSON.stringify(results));
''', {"initial": session.animation_snapshot(), "cases": cases})
        for case, predicted in zip(cases, json.loads(output), strict=True):
            q = np.array(predicted)
            assert np.all(q >= session.env.joint_low) and np.all(q <= session.env.joint_high)
            assert not session.env._cross_robot_collision(q)
            for name, joints in [("cup", q[:3]), ("pot", q[3:])]:
                assert session.env._arm_table_clearance(name, joints) >= -1e-9
            # A fixed angular speed bounds every visual preview.
            assert np.all(np.abs(q - case["q"]) <= session.env.max_joint_speeds * .2 + 1e-12)
    finally:
        session.close()
