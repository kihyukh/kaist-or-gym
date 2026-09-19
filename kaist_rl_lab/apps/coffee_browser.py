"""Shared browser simulation for notebook and standalone website frontends."""

import base64
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_CSS, CANVAS_HTML, CANVAS_JAVASCRIPT

TOOLBAR_HTML = """
<div id="coffee-toolbar" class="coffee-toolbar">
  <div class="coffee-time" aria-live="off">Preparing the simulation…</div>
  <button type="button" disabled data-command="pause" class="coffee-primary">Resume time</button>
  <button type="button" disabled data-command="reset">Reset + start</button>
  <button type="button" disabled data-command="stop">Stop all motors</button>
</div>
<p class="coffee-status" role="status">Loading the simulation for the first time…</p>
"""
SAVE_HTML = """
<details class="coffee-save"><summary>Save your demonstration</summary>
  <p>Saving ends this attempt. Submit before resetting. Keep this tab open until you have a receipt or download.</p>
  <label>Participant code (optional) <input class="coffee-participant" maxlength="64" /></label>
  <button type="button" disabled data-command="save">Save trajectory</button>
  <a class="coffee-download" hidden>Download trajectory (.npz)</a>
  <p class="coffee-submission" role="status"></p>
</details>
"""
BROWSER_CSS = CANVAS_CSS + """
.coffee-toolbar {display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:12px;}
.coffee-time {flex:1; min-width:160px; font-size:14px; color:#334155;}
.coffee-toolbar button, .coffee-save button {border:1px solid #ccd5dd; border-radius:8px;
  background:#fff; color:#1d2b3a; padding:10px 16px; cursor:pointer; font-weight:600;}
.coffee-toolbar .coffee-primary {background:#224a67; color:white; border-color:#224a67;}
.coffee-toolbar button[data-command="stop"] {color:#a63d2b;}
.coffee-toolbar button:disabled, .coffee-save button:disabled {opacity:.5; cursor:wait;}
.coffee-status {font-size:14px; color:#52656f; min-height:20px; margin:8px 0;}
.coffee-save {margin-top:14px; border:1px solid #dce3e8; border-radius:8px; padding:14px;}
.coffee-save summary {cursor:pointer; font-weight:600;}
.coffee-save p {margin:10px 0; font-size:14px;}
.coffee-save label {display:block; margin:10px 0;}
.coffee-save input {display:block; border:1px solid #ccd5dd; border-radius:6px; padding:8px;}
.coffee-download {margin-left:12px; text-decoration:underline; color:#224a67;}
.coffee-toolbar button, .coffee-save button, .coffee-save input {min-height:44px; box-sizing:border-box;}
.coffee-save input {font-size:16px; width:100%; max-width:360px;}
.coffee-toolbar button {touch-action:manipulation;}
@media(max-width:600px) {
  .coffee-toolbar {gap:6px; margin-bottom:6px;}
  .coffee-time {flex-basis:100%; font-size:12px;}
  .coffee-toolbar button {flex:1; padding:8px 6px; font-size:12px;}
  .coffee-save {padding:12px;}
  .coffee-download {display:block; margin:12px 0 0;}
  .coffee-download[hidden] {display:none;}
}
"""

WORKER_JAVASCRIPT = r"""
let python, timer = null, deadline = 0, paused = true, running = true, training = false;
const interval = 1000 / 32;
const PYODIDE_BASE = 'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/';
// Keep the browser's existing Gymnasium version. Loading these verified wheels
// with Pyodide downloads dependencies during VM startup, without running pip.
const COFFEE_PACKAGES = {
  'gymnasium': {
    name:'gymnasium',version:'1.2.3',imports:['gymnasium'],
    file_name:'https://files.pythonhosted.org/packages/56/d3/ea5f088e3638dbab12e5c20d6559d5b3bdaeaa1f2af74e526e6815836285/gymnasium-1.2.3-py3-none-any.whl',
    sha256:'e6314bba8f549c7fdcc8677f7cd786b64908af6e79b57ddaa5ce1825bffb5373',
    depends:['numpy','cloudpickle','typing-extensions','farama-notifications'],
    install_dir:'site',package_type:'package',unvendored_tests:false,
  },
  'farama-notifications': {
    name:'farama-notifications',version:'0.0.6',imports:['farama_notifications'],
    file_name:'https://files.pythonhosted.org/packages/7c/f0/21f81892e4ed10f4ec3ef2e7cf8635fb76e7c0907c55d0da66be50094760/farama_notifications-0.0.6-py3-none-any.whl',
    sha256:'f84839188efa1ce5bb361c2a84881b2dc2c0d0d7fb661ff00421820170930935',
    depends:[],install_dir:'site',package_type:'package',unvendored_tests:false,
  },
};
async function loadCoffeePython() {
  const [module,response] = await Promise.all([
    import(PYODIDE_BASE+'pyodide.mjs'),fetch(PYODIDE_BASE+'pyodide-lock.json'),
  ]);
  if (!response.ok) throw new Error('Could not load the simulation libraries.');
  const lock = await response.json();
  Object.assign(lock.packages,COFFEE_PACKAGES);
  return module.loadPyodide({indexURL:PYODIDE_BASE,packageBaseUrl:PYODIDE_BASE,
    lockFileContents:lock,packages:['numpy','gymnasium']});
}
async function loadCoffeeSource(data) {
  if (!data.bundle_url) return Uint8Array.from(atob(data.bundle),c=>c.charCodeAt(0));
  const response = await fetch(data.bundle_url);
  if (!response.ok) throw new Error('Could not load the simulation source.');
  return new Uint8Array(await response.arrayBuffer());
}
function emit(result) {
  const data = JSON.parse(result), p = data.snapshot.playback;
  paused = p.paused; running = p.running;
  training = !!data.finetuning?.training_running;
  postMessage(data);
}
function dispatch(command) {
  python.globals.set('_coffee_command', JSON.stringify(command));
  emit(python.runPython('coffee_runtime.dispatch(_coffee_command)'));
}
function schedule() {
  if (timer !== null || (!training && (paused || !running))) return;
  timer = setTimeout(() => {
    timer = null;
    try {
      const wasTraining = training;
      dispatch({kind:wasTraining ? 'ft-step' : 'tick'});
      if (!wasTraining) {
        deadline += interval;
        // A slow/background tab never queues a burst of catch-up steps.
        if (deadline < performance.now() - interval) deadline = performance.now() + interval;
      }
      schedule();
    } catch (error) { paused=true; training=false; postMessage({error:String(error)}); }
  }, training ? 0 : Math.max(0, deadline - performance.now()));
}
self.onmessage = async ({data}) => {
  try {
    if (data.kind === 'init') {
      postMessage({loading:'Loading the simulation and its libraries…'});
      const [runtime,bytes] = await Promise.all([loadCoffeePython(),loadCoffeeSource(data)]);
      python = runtime;
      postMessage({loading:'Preparing your coffee station…'});
      python.unpackArchive(bytes, 'zip', {extractDir:'/home/pyodide'});
      python.runPython(data.mode === 'random'
        ? 'from kaist_rl_lab.apps.coffee_random_runtime import RandomAgentRuntime\ncoffee_runtime = RandomAgentRuntime()'
        : data.mode === 'cloning'
        ? 'from kaist_rl_lab.apps.coffee_cloning_runtime import CloningAgentRuntime\ncoffee_runtime = CloningAgentRuntime()'
        : data.mode === 'finetuning'
        ? 'from kaist_rl_lab.apps.coffee_finetuning_runtime import FineTuningRuntime\ncoffee_runtime = FineTuningRuntime()'
        : 'from kaist_rl_lab.apps.coffee_browser_runtime import BrowserRuntime\ncoffee_runtime = BrowserRuntime()');
      dispatch({kind:'snapshot'});
      return;
    }
    if (!python) return;
    const wasPaused = paused;
    dispatch(data);
    const restarting = ['reset','random-start','random-reset','cloning-load','cloning-start','cloning-reset',
      'ft-load','ft-train','ft-run','ft-reset','ft-stop'].includes(data.kind);
    if ((!training && (paused || !running)) || restarting) {
      clearTimeout(timer); timer = null;
    }
    if ((wasPaused && !paused) || restarting) deadline = performance.now();
    schedule();
  } catch (error) { postMessage({error:String(error), command:data.kind}); }
};
"""

CONTROLLER_JAVASCRIPT = r"""
const LOGICAL_WIDTH=960, LOGICAL_HEIGHT=560;
function clamp(value, low, high) { return Math.max(low, Math.min(high,value)); }
let canvasRef=null, resizeObserver=null, displayState=null;
let sequence=0, desiredMotors=Array(6).fill(0), desiredPaused=true;
let ready=false, resetting=false, saving=false, dirty=false;
let episodeId=null, archive=null, downloadUrl=null, uploadTimer=null, uploadSequence=0;
const $ = selector => element.querySelector(selector);
const status = message => { $('.coffee-status').textContent=message; };
const config = typeof props.value==='string' ? JSON.parse(props.value) : props.value;
const workerUrl=config.worker_url || URL.createObjectURL(new Blob([config.worker], {type:'text/javascript'}));
const worker=new Worker(workerUrl, {type:'module'});
if (!config.worker_url) URL.revokeObjectURL(workerUrl);
const saveButton=$('[data-command="save"]');
saveButton.textContent=config.collecting ? 'Submit trajectory' : 'Save trajectory';
let uploading=false;
function controls() {
  if (!displayState) return;
  updateJointControls({...displayState, motors:desiredMotors, running:ready && !resetting && !saving && displayState.running});
  $('[data-command="pause"]').textContent=desiredPaused ? 'Resume time' : 'Pause time';
  element.querySelectorAll('[data-command]').forEach(button=>{
    button.disabled=!ready || resetting || saving || uploading ||
      (!displayState.running && !['reset','save'].includes(button.dataset.command));
  });
}
function sendControl(kind, index, direction) {
  if (!ready || resetting || saving || uploading || (!displayState.running && kind!=='reset')) return;
  if (kind==='motor') desiredMotors[index]=direction;
  if (kind==='stop' || kind==='reset') desiredMotors.fill(0);
  if (kind==='pause') desiredPaused=!desiredPaused;
  if (kind==='reset') { desiredPaused=false; resetting=true; }
  sequence++;
  controls();
  worker.postMessage({kind, motors:desiredMotors.slice(), paused:desiredPaused,
    sequence, generation:displayState.generation});
}
async function submitArchive() {
  if (!config.collecting) {
    $('.coffee-submission').textContent='Attempt saved. Download the trajectory below.';
    return;
  }
  $('.coffee-submission').textContent='Submitting your saved trajectory…';
  if (config.upload_url) {
    if (uploading) return;
    uploading=true; controls();
    const thisEpisode=episodeId, thisArchive=archive;
    const abort=new AbortController();
    const timeout=setTimeout(()=>abort.abort(),60000);
    try {
      const response=await fetch(config.upload_url, {method:'POST',
        headers:{'Content-Type':'application/octet-stream'},
        body:Uint8Array.from(atob(thisArchive),c=>c.charCodeAt(0)), signal:abort.signal});
      const result=await response.json();
      if (!response.ok) throw new Error(typeof result.detail==='string' ? result.detail : 'Upload was not accepted.');
      if (episodeId!==thisEpisode || result.episode_id!==thisEpisode || result.status!=='saved')
        throw new Error('The server receipt did not match this attempt.');
      dirty=false;
      $('.coffee-submission').textContent='Submitted to your instructor. Receipt: '+result.receipt;
      saveButton.textContent='Submitted ✓';
    } catch(error) {
      $('.coffee-submission').textContent=(error.name==='AbortError' ? 'Upload timed out.' : error.message)+
        ' Your attempt is kept in this tab. Tap Submit trajectory to retry, or download a backup.';
      saveButton.textContent='Submit trajectory';
    } finally {clearTimeout(timeout); uploading=false; controls();}
    return;
  }
  clearTimeout(uploadTimer);
  uploadTimer=setTimeout(()=>{
    $('.coffee-submission').textContent='Upload was not confirmed. Retry Submit trajectory or download your backup.';
  },65000);
  uploadSequence++;
  trigger('click', {archive, episode_id:episodeId, request_id:uploadSequence});
}
function receiveUpload() {
  let value;
  try { value=typeof props.value==='string' ? JSON.parse(props.value) : props.value; }
  catch { return; }
  if (!value?.submission || value.episode_id!==episodeId || value.request_id!==uploadSequence) return;
  clearTimeout(uploadTimer);
  $('.coffee-submission').textContent=value.submission;
  if (value.confirmed) dirty=false;
}
worker.onmessage=({data})=>{
  if (data.loading) {status(data.loading+' The first visit can take a little longer.');return;}
  if (data.error) {
    saving=false;
    if(data.command==='save') $('.coffee-participant').disabled=false;
    status(data.command==='save' ? 'No recording saved: '+data.error :
      'Simulation stopped. Please reload the demo. '+data.error);
    if (data.command!=='save') { ready=false; worker.terminate(); }
    controls(); return;
  }
  const raw=data.snapshot;
  if (!raw || raw.schema_version!==4) return;
  const next=normalizedSnapshot(raw);
  // FIFO worker messages share one timeline. Pending inputs only update button
  // feedback; no pose is invented and no old frame can undo a newer command.
  if (next.inputSequence < sequence && !data.archive) return;
  if (displayState && (next.generation<displayState.generation ||
    (next.generation===displayState.generation && next.revision<displayState.revision))) return;
  const newEpisode=episodeId!==data.episode_id;
  if (newEpisode) {
    archive=null; dirty=false; $('.coffee-download').hidden=true;
    $('.coffee-submission').textContent=''; clearTimeout(uploadTimer);
    if(downloadUrl) URL.revokeObjectURL(downloadUrl);
    saveButton.textContent=config.collecting ? 'Submit trajectory' : 'Save trajectory';
    $('.coffee-participant').disabled=false;
  }
  episodeId=data.episode_id; displayState=next; ready=true; resetting=false;
  desiredMotors=next.motors.slice(); desiredPaused=next.paused;
  if (next.running && next.step>0) dirty=true;
  $('.coffee-time').textContent=(next.running ? (next.paused?'Paused':'Running'):'Stopped')+
    ' · Step '+next.step+' · '+next.elapsedTime.toFixed(2)+' s simulated';
  status(next.paused ? 'Time is paused. Set your joint commands, then resume when ready.' :
    next.running ? 'Joint commands stay selected until changed.' : 'Attempt ended. Save your trajectory or reset to start again.');
  if (data.archive) {
    saving=false; archive=data.archive;
    if(downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl=URL.createObjectURL(new Blob([Uint8Array.from(atob(archive),c=>c.charCodeAt(0))],{type:'application/octet-stream'}));
    const link=$('.coffee-download'); link.href=downloadUrl;
    link.download='coffee_'+episodeId+'.npz'; link.hidden=false;
    submitArchive();
  }
  // Rendering and recording use this exact state. No extrapolation or rewind.
  requestAnimationFrame(()=>{ if(displayState===next) {drawFrame(next); controls();} });
  controls();
};
worker.onerror=event=>{ready=false; status('Could not start the simulation. Reload the demo. '+event.message); controls();};
element.addEventListener('click',event=>{
  const button=event.target.closest('button');
  if (!button || !element.contains(button) || button.disabled) return;
  if(button.classList.contains('coffee-joint-button')) {
    sendControl('motor',Number(button.closest('[data-joint-index]').dataset.jointIndex),Number(button.dataset.direction));
  } else if(button.dataset.command==='save') {
    if(archive) {submitArchive();return;}
    const participant=$('.coffee-participant');
    if(config.participant_required && !participant.value.trim()) {
      participant.setCustomValidity('Enter your student ID before submitting.');
      participant.reportValidity(); participant.focus(); return;
    }
    participant.setCustomValidity?.('');
    saving=true; controls();
    participant.disabled=true;
    worker.postMessage({kind:'save',participant:participant.value.trim()});
  } else if(button.dataset.command) {
    if (button.dataset.command==='reset' && dirty &&
        !window.confirm('This attempt has not been submitted or downloaded. Discard it and start again?')) return;
    sendControl(button.dataset.command);
  }
});
$('.coffee-participant').addEventListener('input',()=>$('.coffee-participant').setCustomValidity?.(''));
$('.coffee-download').addEventListener('click',()=>{dirty=false;});
function pauseSafely() {
  if (!ready || resetting || saving || uploading || !displayState.running) return;
  desiredMotors.fill(0); desiredPaused=true; sequence++; controls();
  worker.postMessage({kind:'pause',motors:desiredMotors.slice(),paused:true,
    sequence,generation:displayState.generation});
}
function visibility() { if(document.hidden) pauseSafely(); }
$('.coffee-participant').addEventListener('focus',pauseSafely);
function beforeUnload(event) { if(dirty) {event.preventDefault();event.returnValue='';} }
document.addEventListener('visibilitychange',visibility);
window.addEventListener('beforeunload',beforeUnload);
window.addEventListener('blur',pauseSafely);
window.addEventListener('pagehide',pauseSafely);
const cleanup=new MutationObserver(()=>{
  if(element.isConnected)return;
  worker.terminate(); cleanup.disconnect(); resizeObserver?.disconnect();
  document.removeEventListener('visibilitychange',visibility);
  window.removeEventListener('beforeunload',beforeUnload);
  window.removeEventListener('blur',pauseSafely);
  window.removeEventListener('pagehide',pauseSafely);
  clearTimeout(uploadTimer); if(downloadUrl)URL.revokeObjectURL(downloadUrl);
});
cleanup.observe(document.body,{childList:true,subtree:true});
watch('value',receiveUpload);
worker.postMessage({kind:'init',bundle:config.bundle,bundle_url:config.bundle_url});
"""


def browser_bundle() -> str:
    """Ship installed source so browser and Colab always use the same release."""
    root = Path(__file__).resolve().parents[1]
    data = BytesIO()
    with ZipFile(data, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*.py")):
            archive.write(path, "kaist_rl_lab/" + str(path.relative_to(root)))
    return base64.b64encode(data.getvalue()).decode("ascii")


def build_browser_app(*, collector_url=None, lecture_code=None):
    import gradio as gr

    from kaist_rl_lab.apps.coffee_demonstrations import MAX_ARCHIVE_BYTES, read_demonstration

    if bool(collector_url) != bool(lecture_code):
        raise ValueError("Provide both the collector URL and lecture code, or leave both empty.")

    def upload(event: gr.EventData):
        from kaist_rl_lab.apps.coffee_demonstrations import submit_demonstration

        if not collector_url:
            raise gr.Error("This demo is configured for downloads only.")
        encoded = getattr(event, "archive", None)
        if not isinstance(encoded, str) or len(encoded) > 4 * MAX_ARCHIVE_BYTES // 3 + 4:
            raise gr.Error("Invalid trajectory archive.")
        try:
            data = base64.b64decode(encoded, validate=True)
            _, metadata = read_demonstration(data)
        except (ValueError, KeyError, TypeError) as exc:
            raise gr.Error("Invalid trajectory archive.") from exc
        request_id = getattr(event, "request_id", None)
        if type(request_id) is not int or request_id < 1:
            raise gr.Error("Invalid submission request.")
        # Distinct retries must produce a value change even for identical receipts,
        # otherwise Gradio's value watcher leaves the UI at 'Submitting…'.
        result = {"episode_id": metadata["episode_id"], "confirmed": False,
                  "request_id": request_id}
        try:
            receipt = submit_demonstration(collector_url, lecture_code, data)
            result.update(confirmed=True, submission=(
                f"Submitted {receipt['transitions']} transitions to your instructor. "
                f"Receipt: {receipt['episode_id']}"
            ))
        except Exception:  # noqa: BLE001 - the browser retains the immutable download.
            result["submission"] = (
                "Upload was not confirmed. Retry Submit trajectory before resetting, "
                "or download your backup."
            )
        return json.dumps(result)

    with gr.Blocks(title="KAIST OR Gym — Coffee Pouring") as demo:
        with gr.Column(min_width=0, elem_id="coffee-demo"):
            gr.Markdown("Guide the pot toward the cup and tilt to pour. "
                        "Use the joint controls below the scene; several joints can move together.")
            frame = gr.HTML(
                value=json.dumps({"bundle": browser_bundle(), "worker": WORKER_JAVASCRIPT,
                                  "collecting": bool(collector_url)}),
                html_template=TOOLBAR_HTML + CANVAS_HTML + SAVE_HTML,
                css_template=BROWSER_CSS,
                js_on_load=CANVAS_JAVASCRIPT + CONTROLLER_JAVASCRIPT,
                apply_default_css=False, container=False,
            )
        frame.click(upload, outputs=frame, queue=False, show_progress="hidden",
                    trigger_mode="once")
    return demo
