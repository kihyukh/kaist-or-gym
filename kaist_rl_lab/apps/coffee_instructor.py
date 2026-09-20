"""Private, framework-free instructor interface for the coffee class website."""

from kaist_rl_lab.apps.coffee_cloning_demo import (
    CLONING_DEMO_CSS,
    CLONING_DEMO_HTML,
    CLONING_DEMO_JAVASCRIPT,
)
from kaist_rl_lab.apps.coffee_finetuning_demo import (
    FINETUNING_CSS,
    FINETUNING_HTML,
    FINETUNING_JAVASCRIPT,
)
from kaist_rl_lab.apps.coffee_random_demo import (
    RANDOM_DEMO_CSS,
    RANDOM_DEMO_HTML,
    RANDOM_DEMO_JAVASCRIPT,
)
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_CSS, CANVAS_JAVASCRIPT

INSTRUCTOR_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="robots" content="noindex, nofollow">
  <title>Coffee pouring · Instructor</title>
  <link rel="stylesheet" href="/instructor.css">
  <script defer src="/instructor.js"></script>
</head>
<body>
  <main id="instructor">
    <header class="page-header">
      <div><p class="eyebrow">KAIST OR GYM</p><h1>Coffee pouring <span>Instructor</span></h1></div>
      <button id="logout" class="quiet" type="button" hidden>Sign out</button>
    </header>
    <p id="page-status" role="status" aria-live="polite">Checking instructor access…</p>
    <section id="login-panel" class="panel login-panel" hidden>
      <h2>Instructor sign in</h2>
      <p>Review the class's demonstrations here. Students use the class QR code and do not need this password.</p>
      <form id="login-form">
        <label for="password">Instructor password</label>
        <input id="password" type="password" autocomplete="current-password" required>
        <button class="primary" type="submit">Sign in</button>
      </form>
    </section>
    <div id="dashboard" hidden>
      __RANDOM_DEMO__
      <div class="class-grid">
        <section class="panel class-settings">
          <h2>Your classes</h2>
          <div class="field-row">
            <div class="grow"><label for="session-select">Class session</label><select id="session-select"></select></div>
            <button id="refresh" type="button">Refresh</button>
          </div>
          <details id="create-details">
            <summary>Create a class session</summary>
            <form id="create-form">
              <label for="class-name">Class name</label>
              <input id="class-name" name="name" maxlength="100" required placeholder="Fall 2026 · Freshman seminar">
              <label class="checkbox"><input id="participant-required" type="checkbox" checked> Require a student ID before submission</label>
              <button type="submit" class="primary">Create class link</button>
            </form>
          </details>
          <p class="hint">Each class gets its own link. Submissions are visible only after instructor sign in.</p>
        </section>
        <section id="share-panel" class="panel share-panel" hidden>
          <div class="share-copy">
            <div class="heading-row"><h2 id="session-name"></h2><span id="session-state" class="badge"></span></div>
            <p>Show this QR code on the lecture screen. It opens the demo directly on students' phones.</p>
            <label for="join-url">Student class link</label>
            <input id="join-url" readonly aria-label="Student class link">
            <div class="button-row"><button id="copy-link" type="button">Copy link</button><a id="open-demo" class="button" target="_blank" rel="noopener">Open student view</a></div>
            <p id="session-policy" class="hint"></p>
            <button id="close-session" class="danger quiet" type="button">Close submissions</button>
          </div>
          <a id="qr-download" title="Open QR code" target="_blank" rel="noopener"><img id="class-qr" width="220" height="220" alt="QR code for the student class link"></a>
        </section>
      </div>
      <section id="examples-panel" class="panel" aria-labelledby="examples-title">
        <div class="heading-row">
          <div><p class="eyebrow">INSPECT DEMONSTRATIONS</p><h2 id="examples-title">Generated practice demonstrations</h2><p id="example-count" class="hint"></p></div>
          <button id="refresh-examples" type="button">Refresh examples</button>
        </div>
        <p class="hint">These generated recordings reach the goal but deliberately use slow approaches and begin returning the pot a little early, finishing around 672–686 mL. They meet the existing success tolerance, but leave room to improve accuracy and efficiency. Choose Replay to inspect them before training.</p>
        <p id="examples-status" role="status" class="hint">Loading generated examples…</p>
        <div class="table-scroll"><table>
          <thead><tr><th>Example</th><th>Total reward</th><th>Cup / spill</th><th>Result</th><th>Duration</th><th>Trajectory</th></tr></thead>
          <tbody id="example-rows"></tbody>
        </table></div>
        <p class="hint">Total reward is the sum of the recording's step rewards. Generated examples are separate from student submissions.</p>
      </section>
      <section id="submissions-panel" class="panel" hidden>
        <div class="heading-row">
          <div><h2>Submitted demonstrations</h2><p id="submission-count" class="hint"></p></div>
          <label class="checkbox"><input id="auto-refresh" type="checkbox" checked> Refresh every 5 seconds</label>
        </div>
        <label class="filter-label" for="participant-filter">Find a participant</label>
        <input id="participant-filter" type="search" placeholder="Student ID or participant code">
        <div class="table-scroll"><table>
          <thead><tr><th>Participant</th><th>Received</th><th>Total reward</th><th>Cup / spill</th><th>Result</th><th>Duration</th><th>Trajectory</th></tr></thead>
          <tbody id="submission-rows"></tbody>
        </table></div>
        <p id="empty-submissions" class="empty-state">Waiting for the first submission. Keep this page open during the activity.</p>
        <p class="hint">Total reward sums the recorded step rewards, including recorded penalties and success bonuses. A dash means the reward is unavailable.</p>
      </section>
      <section id="replay-panel" class="panel" hidden aria-labelledby="replay-title">
        <div class="heading-row"><div><h2 id="replay-title">Trajectory replay</h2><p id="replay-summary" class="hint"></p></div><button id="close-replay" type="button" class="quiet">Close replay</button></div>
        <p id="replay-status" role="status"></p>
        <div class="coffee-stage replay-stage">
          <div class="coffee-canvas-wrap">
            <canvas class="coffee-canvas" role="img" aria-label="Replay of the selected coffee pouring trajectory"></canvas>
            <div class="coffee-scene-stats" aria-label="Recorded coffee amounts">
              <span><span class="coffee-stat-label">Cup / target</span><strong data-coffee-stat="fill">—</strong></span>
              <span><span class="coffee-stat-label">Spilled</span><strong data-coffee-stat="spill">—</strong></span>
              <span><span class="coffee-stat-label">In the pot</span><strong data-coffee-stat="remaining">—</strong></span>
            </div>
          </div>
        </div>
        <div class="playback-controls">
          <button id="play-replay" type="button" class="primary" disabled>Play</button>
          <output id="replay-time" for="replay-position">0.0 / 0.0 s</output>
          <span class="replay-reward-label">Reward so far <output id="replay-reward" for="replay-position">—</output></span>
          <label for="playback-speed">Speed</label><select id="playback-speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
          <a id="replay-download" class="button">Download .npz</a>
        </div>
        <label class="sr-only" for="replay-position">Replay position</label>
        <input id="replay-position" type="range" min="0" max="0" value="0" step="1" disabled>
        <p class="hint">Replay begins loading from the first frame while the original environment verifies the remaining recorded actions. Scrub through loaded frames; download the full trajectory for analysis.</p>
      </section>
      __CLONING_DEMO__
      __FINETUNING__
    </div>
    <footer>Instructor access is separate from the student class link.</footer>
  </main>
</body>
</html>
""".replace("__RANDOM_DEMO__", RANDOM_DEMO_HTML).replace("__CLONING_DEMO__", CLONING_DEMO_HTML).replace("__FINETUNING__", FINETUNING_HTML)

INSTRUCTOR_CSS = CANVAS_CSS + """
:root {color-scheme:light;--ink:#1d2b3a;--muted:#52656f;--navy:#224a67;--teal:#2b7a78;--line:#d8e0e3;}
* {box-sizing:border-box;}
[hidden] {display:none !important;}
body {margin:0;background:#eef2f4;color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;}
#instructor {max-width:1440px;margin:auto;padding:28px max(20px,env(safe-area-inset-left)) 24px;}
.page-header,.heading-row,.field-row,.button-row {display:flex;align-items:center;justify-content:space-between;gap:14px;}
.page-header {margin-bottom:12px;}
.eyebrow {margin:0 0 4px;color:var(--teal);font-size:12px;font-weight:800;letter-spacing:.13em;}
h1 {font-size:30px;line-height:1.2;margin:0;}
h1 span {font-size:15px;display:inline-block;color:var(--muted);font-weight:500;margin-left:10px;}
h2 {font-size:20px;line-height:1.3;margin:0 0 12px;}
p {margin:10px 0;}
#page-status {min-height:24px;margin:12px 0 16px;color:var(--muted);}
#page-status.error,#replay-status.error,#examples-status.error {color:#a03929;}
.panel {background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:20px;box-shadow:0 2px 8px #1d2b3a05;min-width:0;}
.login-panel {max-width:520px;margin:40px auto;}
label {display:block;font-size:14px;font-weight:600;margin:10px 0 6px;}
input,select,button,.button {font:inherit;min-height:44px;border-radius:8px;}
input,select {color:var(--ink);background:white;border:1px solid #aab8be;padding:9px 12px;width:100%;}
input:focus-visible,select:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible {outline:3px solid #bd8b2980;outline-offset:2px;}
button,.button {display:inline-flex;align-items:center;justify-content:center;border:1px solid #aab8be;background:white;color:var(--navy);padding:9px 14px;text-decoration:none;font-size:14px;font-weight:650;cursor:pointer;touch-action:manipulation;white-space:nowrap;}
button:hover,.button:hover {background:#eff5f6;}
button:disabled {opacity:.5;cursor:wait;}
button.primary {background:var(--navy);color:white;border-color:var(--navy);}
button.primary:hover {background:#183c56;}
button.quiet {background:transparent;}
button.danger {color:#a03929;border-color:#c9aea8;}
form button {margin-top:14px;}
.checkbox {display:flex;align-items:center;gap:8px;font-size:13px;line-height:1.4;font-weight:500;}
.checkbox input {width:18px;height:18px;min-height:18px;flex:0 0 auto;accent-color:var(--teal);}
.class-grid {display:grid;grid-template-columns:minmax(270px,.8fr) minmax(440px,1.7fr);gap:20px;align-items:start;}
.field-row {align-items:end;}
.grow {flex:1;min-width:0;}
.class-settings .field-row {gap:8px;}
.class-settings label {margin-top:0;}
details {margin-top:20px;border-top:1px solid var(--line);padding-top:15px;}
summary {cursor:pointer;font-size:14px;font-weight:650;min-height:44px;display:list-item;align-content:center;}
.create-form {margin-top:12px;}
.hint {font-size:13px;color:var(--muted);line-height:1.5;}
.share-panel {display:flex;align-items:center;gap:24px;}
.share-copy {min-width:0;flex:1;}
.share-copy p {font-size:14px;}
.share-copy .heading-row {gap:8px;align-items:start;}
.share-copy h2 {overflow-wrap:anywhere;}
.share-panel img {display:block;width:180px;height:180px;max-width:100%;border:1px solid var(--line);border-radius:6px;}
#join-url {font-size:14px;}
.button-row {justify-content:flex-start;gap:8px;margin-top:10px;flex-wrap:wrap;}
.badge {display:inline-block;border-radius:99px;background:#e8f3ed;color:#286446;padding:3px 9px;font-size:12px;font-weight:700;white-space:nowrap;}
.badge.closed {background:#f1eceb;color:#72534b;}
.filter-label {font-size:13px;}
#participant-filter {max-width:330px;margin-bottom:16px;}
.table-scroll {width:100%;overflow-x:auto;}
table {border-collapse:collapse;width:100%;font-size:14px;text-align:left;}
th {color:var(--muted);font-weight:600;font-size:12px;white-space:nowrap;}
th,td {padding:12px 10px;border-bottom:1px solid #e4e9ec;}
th:first-child,td:first-child {padding-left:0;}
td {font-variant-numeric:tabular-nums;}
td.participant {font-weight:650;min-width:120px;max-width:220px;overflow-wrap:anywhere;}
td.nowrap {white-space:nowrap;}
td.actions {display:flex;gap:8px;}
td .button,td button {font-size:12px;padding:7px 10px;}
.empty-state {padding:22px 0;color:var(--muted);text-align:center;}
.replay-stage {max-width:960px;margin:16px auto 0;display:block;}
.replay-stage .coffee-canvas {width:100%;max-width:960px;}
.playback-controls {display:flex;gap:12px;align-items:center;flex-wrap:wrap;max-width:960px;margin:14px auto 0;}
.playback-controls label {margin:0 0 0 auto;}
#playback-speed {width:80px;}
#replay-time,.replay-reward-label {font-size:14px;font-variant-numeric:tabular-nums;}
#replay-reward {font-weight:700;}
#replay-position {display:block;width:100%;max-width:960px;padding:0;margin:8px auto;accent-color:var(--teal);cursor:pointer;}
#replay-panel>.hint {text-align:center;}
footer {font-size:12px;color:var(--muted);text-align:center;margin-top:20px;}
.sr-only {position:absolute;width:1px;height:1px;margin:-1px;padding:0;border:0;clip:rect(0,0,0,0);overflow:hidden;}
@media(max-width:1150px) {.class-grid {grid-template-columns:1fr 1.7fr;}.share-panel {gap:16px;}.share-panel img {width:145px;height:145px;}.share-copy .heading-row {display:block;}.share-copy h2 {margin-bottom:6px;}}
@media(max-width:900px) {.class-grid {display:block;}.share-panel img {width:200px;height:200px;}.share-copy .heading-row {display:flex;}h1 {font-size:26px;}}
@media(max-width:580px) {#instructor {padding:20px 12px;}.panel {padding:16px;}.share-panel {display:block;}.share-panel img {margin:16px auto 0;width:220px;height:220px;}h1 span {display:block;margin:6px 0 0;}.heading-row {align-items:start;flex-wrap:wrap;}.heading-row h2 {margin-bottom:4px;}.playback-controls {gap:8px;}.playback-controls label {margin-left:0;}#replay-download {width:100%;}.field-row button {padding-inline:10px;}}
""" + RANDOM_DEMO_CSS + CLONING_DEMO_CSS + FINETUNING_CSS

INSTRUCTOR_JAVASCRIPT = r"""
'use strict';
const element = document.querySelector('#instructor');
const LOGICAL_WIDTH=960, LOGICAL_HEIGHT=560;
const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
let canvasRef=null, resizeObserver=null, displayState=null;
const $=selector=>element.querySelector(selector);
const randomDemo=createRandomDemo($('#random-panel'));
const fineTuningDemo=createFineTuningDemo($('#finetuning-panel'),
  ()=>{randomDemo.pause();cloningDemo.pause();stopPlayback();});
const cloningDemo=createCloningDemo($('#cloning-panel'),post,
  ()=>{randomDemo.pause();fineTuningDemo.pause();stopPlayback();},
  model=>fineTuningDemo.setModel(model));
let sessions=[], activeSession=null, submissions=[], authenticated=false, authEpoch=0;
let examples=[], examplesLoaded=false, exampleRequest=0;
let listRequest=0, replayRequest=0, refreshBusy=false, refreshError='', submissionLoad=null, replayFrames=[];
let frameIndex=0, playing=false, playStarted=0, playOrigin=0, animation=null;
let replayAbort=null,replayComplete=false,replayDuration=0,replayExpectedFrames=0;
function setStatus(message,error=false) {
  $('#page-status').textContent=message;
  $('#page-status').classList.toggle('error',error);
}
class ApiError extends Error {
  constructor(message,status) {super(message);this.status=status;}
}
async function api(path,options={}) {
  const response=await fetch(path,{credentials:'same-origin',...options,
    headers:{...(options.body?{'Content-Type':'application/json'}:{}),...options.headers}});
  let data;
  try {data=await response.json();} catch {data={};}
  if(!response.ok) {
    if(response.status===401) showLogin();
    throw new ApiError(typeof data.detail==='string'?data.detail:
      (typeof data.error==='string'?data.error:'The request failed. Please try again.'),response.status);
  }
  return data;
}
function post(path,body={}) {return api(path,{method:'POST',body:JSON.stringify(body)});}
async function readReplayStream(path,signal,onEvent) {
  const response=await fetch(path,{credentials:'same-origin',signal});
  if(!response.ok) {
    let data={};try{data=await response.json();}catch{}
    if(response.status===401)showLogin();
    throw new ApiError(typeof data.detail==='string'?data.detail:'Could not load this recording.',response.status);
  }
  if(!response.body?.getReader)throw new Error('This browser cannot stream the recording. Try a recent browser.');
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
  try {
    while(true) {
      const {value,done}=await reader.read();
      buffer+=decoder.decode(value,{stream:!done});
      let newline;
      while((newline=buffer.indexOf('\n'))>=0) {
        const line=buffer.slice(0,newline);buffer=buffer.slice(newline+1);
        if(line.trim())onEvent(JSON.parse(line));
      }
      if(done){if(buffer.trim())onEvent(JSON.parse(buffer));break;}
    }
  } finally {reader.releaseLock();}
}
function cancelReplayRequest() {
  replayRequest++;if(replayAbort)replayAbort.abort();replayAbort=null;stopPlayback();
}
function showLogin() {
  randomDemo.setEnabled(false);
  cloningDemo.setEnabled(false);
  fineTuningDemo.setEnabled(false);
  authenticated=false;authEpoch++;listRequest++;exampleRequest++;cancelReplayRequest();
  $('#dashboard').hidden=true;$('#logout').hidden=true;$('#login-panel').hidden=false;
  sessions=[];submissions=[];examples=[];examplesLoaded=false;activeSession=null;replayFrames=[];displayState=null;
  $('#submission-rows').replaceChildren();$('#example-rows').replaceChildren();$('#session-select').replaceChildren();
  $('#example-count').textContent='';$('#examples-status').textContent='';$('#refresh-examples').disabled=false;
  $('#replay-reward').textContent='—';$('#replay-time').textContent='0.0 / 0.0 s';
  $('#join-url').value='';$('#class-qr').removeAttribute('src');
  ['#open-demo','#qr-download','#replay-download'].forEach(selector=>$(selector).removeAttribute('href'));
  $('#session-name').textContent='';$('#replay-title').textContent='Trajectory replay';$('#replay-summary').textContent='';
  const canvas=$('#replay-panel .coffee-canvas');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
  $('#share-panel').hidden=true;$('#submissions-panel').hidden=true;$('#replay-panel').hidden=true;
}
function showDashboard() {
  randomDemo.setEnabled(true);
  cloningDemo.setEnabled(true);
  fineTuningDemo.setEnabled(true);
  authenticated=true;$('#login-panel').hidden=true;$('#dashboard').hidden=false;$('#logout').hidden=false;
}
function idPath(value) {return encodeURIComponent(String(value));}
function localTime(value) {
  const date=new Date(value);
  return Number.isNaN(date.valueOf())?'—':date.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
function number(value,digits=0) {return value!==null&&value!==undefined&&Number.isFinite(Number(value))?Number(value).toFixed(digits):'—';}
function duration(value) {return number(value,1)+' s';}
function safeJoinURL(value) {
  try {const url=new URL(value,location.origin);return url.origin===location.origin?url.href:'';} catch {return '';}
}
function selectedSession() {return sessions.find(session=>session.id===activeSession);}
function renderSession() {
  const session=selectedSession();
  cloningDemo.setContext({id:session?.id||null,name:session?.name||'',total:0,successful:0});
  $('#share-panel').hidden=!session;$('#submissions-panel').hidden=!session;
  if(!session) return;
  $('#session-name').textContent=session.name;
  $('#session-state').textContent=session.open?'Open':'Closed';
  $('#session-state').classList.toggle('closed',!session.open);
  const joinURL=safeJoinURL(session.join_url);
  $('#join-url').value=joinURL;$('#open-demo').href=joinURL||'#';
  $('#copy-link').disabled=!joinURL;
  const qrURL='/api/instructor/sessions/'+idPath(session.id)+'/qr.svg';
  $('#class-qr').src=qrURL;$('#qr-download').href=qrURL;
  $('#close-session').hidden=!session.open;
  $('#session-policy').textContent=(session.participant_required?'Student ID is required. ':'Participant code is optional. ')+
    (session.open?'Submissions are open.':'Submissions are closed; existing demonstrations remain available.');
  if (['localhost','127.0.0.1','[::1]'].includes(location.hostname))
    $('#session-policy').textContent+=' Local preview: this QR works only on this computer. Use the deployed website for class.';
}
async function loadSessions(preferred) {
  const epoch=authEpoch;
  const data=await api('/api/instructor/sessions');
  if(epoch!==authEpoch)return;
  sessions=data;
  if(!Array.isArray(sessions)) throw new Error('Could not read the class list.');
  showDashboard();
  activeSession=sessions.some(item=>item.id===(preferred||activeSession))?(preferred||activeSession):(sessions[0]?.id||null);
  $('#session-select').replaceChildren();
  sessions.forEach(session=>{
    const option=document.createElement('option');option.value=session.id;
    option.textContent=session.name+(session.open?'':' · Closed');$('#session-select').append(option);
  });
  $('#session-select').value=activeSession||'';$('#session-select').disabled=!sessions.length;
  $('#create-details').open=!sessions.length;
  renderSession();
  await Promise.all([examplesLoaded?Promise.resolve():loadExamples(),
    activeSession?loadSubmissions():Promise.resolve()]);
  if(epoch!==authEpoch||!authenticated)return;
  setStatus(sessions.length?'Class links open the demo directly. Keep the instructor password private.':'Create a class session to get a student link and QR code.');
}
async function loadSubmissions() {
  if(!activeSession||!authenticated) return;
  const session=activeSession,epoch=authEpoch;
  if(submissionLoad?.session===session&&submissionLoad.epoch===epoch)return submissionLoad.promise;
  const request=++listRequest,pending={session,epoch,promise:null},controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),15000);
  pending.promise=(async()=>{
    try {
      const data=await api('/api/instructor/submissions?session='+idPath(session),{signal:controller.signal});
      if(request!==listRequest||session!==activeSession||epoch!==authEpoch||!authenticated) return;
      if(!Array.isArray(data)) throw new Error('Could not read submissions.');
      submissions=data.slice().sort((a,b)=>String(b.received_at).localeCompare(String(a.received_at)));
      renderSubmissions();
    } catch(error) {
      if(controller.signal.aborted) {
        if(request!==listRequest||session!==activeSession||epoch!==authEpoch||!authenticated)return;
        throw new Error('Refreshing submissions took too long. It will retry automatically.');
      }
      throw error;
    }
  })().finally(()=>{clearTimeout(timeout);if(submissionLoad===pending)submissionLoad=null;});
  submissionLoad=pending;
  return pending.promise;
}
async function refreshSubmissions() {
  if(!authenticated||!activeSession||document.hidden||!$('#auto-refresh').checked||refreshBusy)return;
  const session=activeSession,epoch=authEpoch;
  refreshBusy=true;
  try {
    await loadSubmissions();
    if(session===activeSession&&epoch===authEpoch&&authenticated) {
      if(refreshError&&$('#page-status').textContent===refreshError)setStatus('Student submissions are up to date.');
      refreshError='';
    }
  } catch(error) {
    if((session===activeSession&&epoch===authEpoch)||(error.status===401&&!authenticated)) {
      refreshError=error.status===401?'Your sign in has expired. Sign in again.':error.message;
      setStatus(refreshError,true);
    }
  } finally {refreshBusy=false;}
}
function cell(row,text,className='') {
  const td=document.createElement('td');td.textContent=text;td.className=className;row.append(td);return td;
}
function trajectoryPath(item,source='students') {
  return source==='examples'?'/api/instructor/examples/'+idPath(item.example_id):
    '/api/instructor/submissions/'+idPath(item.episode_id);
}
function trajectoryActions(row,item,source='students') {
  const actions=cell(row,'','actions');
  const replay=document.createElement('button');replay.type='button';replay.textContent='Replay';
  replay.addEventListener('click',()=>openReplay(item,source));actions.append(replay);
  const download=document.createElement('a');download.className='button';download.textContent='Download';
  download.href=trajectoryPath(item,source)+'/download';download.download='';actions.append(download);
}
function trajectoryOutcome(row,item) {
  const outcome=cell(row,'');const badge=document.createElement('span');badge.className='badge'+(item.success?'':' closed');
  badge.textContent=item.success?'Success':item.termination_reason==='time_limit'?'Time limit':'Attempt';outcome.append(badge);
}
async function loadExamples() {
  if(!authenticated)return;
  const request=++exampleRequest,epoch=authEpoch;
  $('#refresh-examples').disabled=true;$('#examples-status').textContent='Loading generated examples…';
  $('#examples-status').classList.remove('error');
  try {
    const data=await api('/api/instructor/examples');
    if(request!==exampleRequest||epoch!==authEpoch||!authenticated)return;
    if(!Array.isArray(data))throw new Error('Could not read the generated examples.');
    examples=data;examplesLoaded=true;
    const rows=document.createDocumentFragment();
    examples.forEach(item=>{
      const row=document.createElement('tr');cell(row,item.label,'participant');
      cell(row,number(item.total_reward,3),'nowrap');
      cell(row,number(item.fill_ml,1)+' / '+number(item.spill_ml,1)+' mL','nowrap');
      trajectoryOutcome(row,item);cell(row,duration(item.duration_seconds),'nowrap');
      trajectoryActions(row,item,'examples');rows.append(row);
    });
    $('#example-rows').replaceChildren(rows);
    $('#example-count').textContent=examples.length+' prepared trajectories';
    $('#examples-status').textContent=examples.length?'':'No generated examples are available.';
  } catch(error) {
    if(request!==exampleRequest||epoch!==authEpoch||!authenticated)return;
    $('#examples-status').textContent=error.message;$('#examples-status').classList.add('error');
  } finally {
    if(request===exampleRequest&&epoch===authEpoch)$('#refresh-examples').disabled=false;
  }
}
function renderSubmissions() {
  const session=selectedSession();
  cloningDemo.setContext({id:session?.id||null,name:session?.name||'',total:submissions.length,
    successful:submissions.filter(item=>item.success).length});
  const filter=$('#participant-filter').value.trim().toLocaleLowerCase();
  const filtered=submissions.filter(item=>String(item.participant||'Anonymous').toLocaleLowerCase().includes(filter));
  $('#submission-count').textContent=submissions.length+' demonstration'+(submissions.length===1?'':'s')+' · '+
    new Set(submissions.map(item=>item.participant).filter(Boolean)).size+' identified participant(s)'+
    (filter?' · '+filtered.length+' shown':'');
  const rows=document.createDocumentFragment();
  filtered.forEach(item=>{
    const row=document.createElement('tr');
    cell(row,item.participant||'Anonymous','participant');cell(row,localTime(item.received_at),'nowrap');
    cell(row,number(item.total_reward,3),'nowrap');
    cell(row,number(item.fill_ml)+' / '+number(item.spill_ml)+' mL','nowrap');
    trajectoryOutcome(row,item);
    cell(row,duration(item.duration_seconds),'nowrap');trajectoryActions(row,item);
    rows.append(row);
  });
  $('#submission-rows').replaceChildren(rows);$('#empty-submissions').hidden=filtered.length>0;
  $('#empty-submissions').textContent=filter?'No participants match this search.':'Waiting for the first submission. Keep this page open during the activity.';
}
function stopPlayback() {
  playing=false;if(animation!==null)cancelAnimationFrame(animation);animation=null;
  $('#play-replay').textContent=replayComplete&&frameIndex===replayFrames.length-1&&replayFrames.length>1?'Replay':'Play';
}
function renderReplayFrame(index) {
  if(!replayFrames.length)return;
  frameIndex=clamp(index,0,replayFrames.length-1);
  const frame=replayFrames[frameIndex];
  displayState={...normalizedSnapshot(frame.snapshot),running:true,paused:false,transitioning:true};
  drawFrame(displayState);$('#replay-position').value=String(frameIndex);
  const end=replayDuration||replayFrames[replayFrames.length-1].time;
  $('#replay-time').textContent=number(frame.time,1)+' / '+number(end,1)+' s';
  $('#replay-reward').textContent=number(frame.cumulative_reward,3);
  $('#replay-position').setAttribute('aria-valuetext',number(frame.time,1)+' seconds');
}
function tickPlayback(now) {
  if(!playing)return;
  const time=playOrigin+(now-playStarted)/1000*Number($('#playback-speed').value);
  let index=frameIndex;
  while(index+1<replayFrames.length&&replayFrames[index+1].time<=time)index++;
  if(index!==frameIndex)renderReplayFrame(index);
  if(time>=replayFrames[replayFrames.length-1].time){
    renderReplayFrame(replayFrames.length-1);
    if(replayComplete){stopPlayback();return;}
    // Freeze the clock at the verified buffer edge; resume from that point
    // when another frame arrives rather than skipping unseen movement.
    playOrigin=replayFrames[replayFrames.length-1].time;playStarted=now;
    $('#replay-status').textContent='Loading the next verified frames…';
  }
  animation=requestAnimationFrame(tickPlayback);
}
function startPlayback() {
  if(!replayFrames.length)return;
  randomDemo.pause();
  cloningDemo.pause();
  fineTuningDemo.pause();
  if(replayComplete&&frameIndex===replayFrames.length-1)renderReplayFrame(0);
  playing=true;playStarted=performance.now();playOrigin=replayFrames[frameIndex].time;
  $('#play-replay').textContent='Pause';animation=requestAnimationFrame(tickPlayback);
}
async function openReplay(item,source='students') {
  randomDemo.pause();
  cloningDemo.pause();
  fineTuningDemo.pause();
  cancelReplayRequest();const request=replayRequest,controller=new AbortController();replayAbort=controller;
  replayFrames=[];frameIndex=0;displayState=null;
  replayComplete=false;replayDuration=Number(item.duration_seconds)||0;replayExpectedFrames=0;
  $('#play-replay').textContent='Play';
  const label=source==='examples'?item.label:(item.participant||'Anonymous');
  const provenance=source==='examples'?'Generated practice demonstration':'Student submission · '+localTime(item.received_at);
  const summary=provenance+' · '+number(item.steps)+' recorded steps · '+duration(item.duration_seconds);
  $('#replay-panel').hidden=false;$('#replay-title').textContent='Replay · '+label;
  $('#replay-summary').textContent=summary+' · total reward '+number(item.total_reward,3);
  $('#replay-reward').textContent='—';$('#replay-time').textContent='0.0 / '+duration(item.duration_seconds);
  $('#replay-status').textContent='Loading the first frame…';$('#replay-status').classList.remove('error');
  $('#play-replay').disabled=true;$('#replay-position').disabled=true;
  $('#replay-download').href=trajectoryPath(item,source)+'/download';$('#replay-download').download='';
  const canvas=$('#replay-panel .coffee-canvas');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
  element.querySelectorAll('#replay-panel [data-coffee-stat]').forEach(stat=>{stat.textContent='—';});
  $('#replay-panel').scrollIntoView({behavior:'smooth',block:'start'});
  try {
    await readReplayStream(trajectoryPath(item,source)+'/replay-stream',controller.signal,data=>{
      if(request!==replayRequest||!authenticated)return;
      if(data.kind==='error')throw new Error(data.detail||'The recording could not be verified.');
      if(data.kind==='start') {
        replayFrames=[data.frame];replayDuration=data.duration_seconds;replayExpectedFrames=data.frame_count;
        $('#replay-summary').textContent=summary+' · total reward '+number(data.total_reward,3);
        $('#play-replay').disabled=false;$('#replay-position').disabled=false;
        renderReplayFrame(0);
      } else if(data.kind==='frames'&&Array.isArray(data.frames))replayFrames.push(...data.frames);
      else if(data.kind==='complete')replayComplete=true;
      if(!replayFrames.length)throw new Error('No replay frames were available.');
      $('#replay-position').max=String(replayFrames.length-1);
      $('#replay-status').textContent=replayComplete?
        'Ready · '+replayFrames.length+' verified frames. The download contains every recorded step.':
        'Ready to play · '+replayFrames.length+' / '+replayExpectedFrames+' frames loaded. The rest loads while you watch.';
    });
    if(request!==replayRequest||!authenticated)return;
    if(!replayComplete)throw new Error('The connection ended before the recording finished loading. Select Replay to retry.');
  } catch(error) {
    if(request!==replayRequest||controller.signal.aborted)return;
    controller.abort();stopPlayback();replayFrames=[];replayComplete=false;displayState=null;
    $('#play-replay').disabled=true;$('#replay-position').disabled=true;
    canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
    $('#replay-status').textContent=error.message;$('#replay-status').classList.add('error');
  } finally {if(replayAbort===controller)replayAbort=null;}
}
$('#login-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=event.submitter||$('#login-form button');button.disabled=true;
  try {await post('/api/instructor/login',{password:$('#password').value});$('#password').value='';await loadSessions();}
  catch(error){setStatus(error.message,true);}finally{button.disabled=false;}
});
$('#logout').addEventListener('click',async()=>{
  try {await post('/api/instructor/logout');showLogin();setStatus('Signed out.');}
  catch(error){setStatus(error.message,true);}
});
$('#create-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=event.submitter||$('#create-form button');button.disabled=true;
  try {
    const session=await post('/api/instructor/sessions',{name:$('#class-name').value.trim(),participant_required:$('#participant-required').checked});
    $('#class-name').value='';$('#create-details').open=false;await loadSessions(session.id);setStatus('Class created. Share its QR code or student link.');
  }catch(error){setStatus(error.message,true);}finally{button.disabled=false;}
});
$('#session-select').addEventListener('change',async()=>{
  activeSession=$('#session-select').value;submissions=[];renderSession();renderSubmissions();
  cancelReplayRequest();$('#replay-panel').hidden=true;
  try{await loadSubmissions();}catch(error){setStatus(error.message,true);}
});
$('#close-session').addEventListener('click',async()=>{
  const session=selectedSession();if(!session)return;
  if(!window.confirm('Close submissions for '+session.name+'? Existing demonstrations will remain available.'))return;
  $('#close-session').disabled=true;
  try {await post('/api/instructor/sessions/'+idPath(session.id)+'/close');await loadSessions(session.id);setStatus('Submissions closed for this class.');}
  catch(error){setStatus(error.message,true);}finally{$('#close-session').disabled=false;}
});
$('#copy-link').addEventListener('click',async()=>{
  try {await navigator.clipboard.writeText($('#join-url').value);setStatus('Student class link copied.');}
  catch {$('#join-url').focus();$('#join-url').select();setStatus('Select and copy the student link above.');}
});
$('#refresh').addEventListener('click',async()=>{
  $('#refresh').disabled=true;
  try{await loadSessions();}catch(error){setStatus(error.message,true);}finally{$('#refresh').disabled=false;}
});
$('#participant-filter').addEventListener('input',renderSubmissions);
$('#refresh-examples').addEventListener('click',loadExamples);
$('#random-start').addEventListener('click',()=>{stopPlayback();cloningDemo.pause();fineTuningDemo.pause();});
$('#random-pause').addEventListener('click',()=>{stopPlayback();cloningDemo.pause();fineTuningDemo.pause();});
$('#play-replay').addEventListener('click',()=>playing?stopPlayback():startPlayback());
$('#replay-position').addEventListener('input',event=>{stopPlayback();renderReplayFrame(Number(event.target.value));});
$('#playback-speed').addEventListener('change',()=>{if(playing){stopPlayback();startPlayback();}});
$('#close-replay').addEventListener('click',()=>{cancelReplayRequest();$('#replay-panel').hidden=true;});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();else refreshSubmissions();});
window.addEventListener('focus',refreshSubmissions);
window.addEventListener('pagehide',()=>cancelReplayRequest());
setInterval(refreshSubmissions,5000);
loadSessions().catch(error=>setStatus(error.status===401?'Sign in to manage classes and review demonstrations.':error.message,error.status!==401));
""" + CANVAS_JAVASCRIPT.replace(
    'element.querySelector(', 'element.querySelector("#replay-panel " + '
).replace(
    'element.querySelectorAll(', 'element.querySelectorAll("#replay-panel " + '
) + RANDOM_DEMO_JAVASCRIPT + CLONING_DEMO_JAVASCRIPT + FINETUNING_JAVASCRIPT
