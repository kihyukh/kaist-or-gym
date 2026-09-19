"""Instructor-only controls for a browser-local random-policy demonstration."""

from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT

RANDOM_DEMO_HTML = """
      <section id="random-panel" class="panel" aria-labelledby="random-title">
        <div class="heading-row">
          <div><p class="eyebrow">LIVE EXPERIMENT</p><h2 id="random-title">Random agent · no learning</h2></div>
          <span class="badge random-policy">Policy updates: 0</span>
        </div>
        <p>What happens if the robot just tries random moves?</p>
        <p class="hint">Each of the six joints independently chooses clockwise, hold, or counter-clockwise
          with equal probability. It holds these commands for a random duration of up to <b>1 second</b>,
          then chooses again. Each trial starts from a random classroom pose and lasts up to <b>30 seconds</b>.</p>
        <div class="button-row">
          <button id="random-start" class="primary" type="button">Start random trial</button>
          <button id="random-pause" type="button" disabled>Pause</button>
          <button id="random-reset" type="button" disabled>Reset demo</button>
        </div>
        <p id="random-status" role="status">Ready when you are. The simulation loads on first use.</p>
        <div id="random-scene" hidden>
          <div class="random-readout" aria-live="off">
            <span>Trial <strong id="random-trial">0</strong></span>
            <span>Simulated time <strong id="random-time">0.0 / 30.0 s</strong></span>
            <span>Random choice <strong id="random-decisions">0</strong></span>
            <span>Hold duration <strong id="random-hold">—</strong></span>
          </div>
          <div class="random-layout">
            <div class="coffee-stage replay-stage">
              <div class="coffee-canvas-wrap">
                <canvas class="coffee-canvas" role="img" aria-label="Live random-agent coffee pouring simulation"></canvas>
                <div class="coffee-scene-stats" aria-label="Random agent coffee amounts">
                  <span><span class="coffee-stat-label">Cup / target</span><strong data-coffee-stat="fill">—</strong></span>
                  <span><span class="coffee-stat-label">Spilled</span><strong data-coffee-stat="spill">—</strong></span>
                  <span><span class="coffee-stat-label">In the pot</span><strong data-coffee-stat="remaining">—</strong></span>
                </div>
              </div>
            </div>
            <div class="random-commands">
              <h3>Current commands</h3>
              <p class="hint">Shoulder · elbow · wrist</p>
              <p><b>Cup arm</b><output id="random-cup">—</output></p>
              <p><b>Pot arm</b><output id="random-pot">—</output></p>
              <p class="hint">↺ counter-clockwise<br>■ hold<br>↻ clockwise</p>
              <p class="random-feedback">Feedback → <b>ignored</b><br>Next move → <b>random again</b></p>
            </div>
          </div>
        </div>
        <p class="random-lesson"><b>Experience alone is not learning.</b> This agent never uses the outcome
          to change its choices. A lucky pour can happen by chance. Learning requires using feedback to
          improve the policy.</p>
        <details id="random-history" hidden>
          <summary>Completed trials <span id="random-count"></span></summary>
          <p class="hint">Latest 10 trials in this page only. New random trials use the same unchanged policy.
            These runs are not submitted as student demonstrations.</p>
          <div class="table-scroll"><table><thead><tr><th>Trial</th><th>Cup</th><th>Spilled</th><th>Duration</th><th>Outcome</th></tr></thead>
            <tbody id="random-results"></tbody></table></div>
        </details>
      </section>
"""

RANDOM_DEMO_CSS = """
.random-policy {background:#fff1d5;color:#75551c;}
#random-status {color:var(--muted);font-size:14px;min-height:21px;}
#random-status.error {color:#a03929;}
.random-readout {display:flex;gap:12px 24px;flex-wrap:wrap;padding:12px 0;font-size:13px;font-variant-numeric:tabular-nums;}
.random-readout strong {display:block;font-size:17px;color:var(--navy);}
.random-layout {display:grid;grid-template-columns:minmax(0,1fr) 200px;gap:20px;align-items:center;}
.random-layout .replay-stage {width:100%;margin:0;}
.random-commands {padding:16px;border-radius:10px;background:#f2f6f6;font-size:14px;}
.random-commands h3 {font-size:16px;margin:0;}
.random-commands output {display:block;font-size:26px;letter-spacing:10px;white-space:nowrap;}
.random-feedback {border-top:1px solid var(--line);padding-top:12px;}
.random-lesson {border-left:4px solid #bd8b29;background:#fff8e9;padding:12px 16px;font-size:14px;}
@media(max-width:700px) {.random-layout {grid-template-columns:1fr;}.random-commands {display:grid;grid-template-columns:1fr 1fr;gap:0 16px;}.random-commands h3,.random-feedback {grid-column:1/-1;}.random-commands>.hint {display:none;}.random-readout {gap:10px 18px;}.random-readout strong {font-size:15px;}}
"""

RANDOM_DEMO_JAVASCRIPT = r"""
function createRandomDemo(element) {
  const $=selector=>element.querySelector(selector);
  const LOGICAL_WIDTH=960, LOGICAL_HEIGHT=560;
  const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
  let canvasRef=null, resizeObserver=null, displayState=null;
  let worker=null, enabled=false, loading=false, pendingStart=false, pendingCommand=false;
  let pauseRequested=false;
  let state=null, completed=0;
  const seen=new Set();
  function status(message,error=false) {
    $('#random-status').textContent=message;$('#random-status').classList.toggle('error',error);
  }
  function controls() {
    $('#random-start').disabled=!enabled||loading||pendingCommand;
    $('#random-start').textContent=loading?'Loading simulation…':state?.attempt?'New random trial':'Start random trial';
    $('#random-pause').disabled=!enabled||loading||pendingCommand||!state?.attempt||state.done;
    $('#random-pause').textContent=displayState?.paused?'Resume':'Pause';
    $('#random-reset').disabled=!enabled||loading||pendingCommand||!state;
  }
  function send(command) {
    if(!worker||!enabled)return;
    pendingCommand=true;controls();worker.postMessage(command);
  }
  function terminate() {
    if(worker)worker.terminate();worker=null;
    loading=false;pendingStart=false;pendingCommand=false;
  }
  function clearResults() {
    seen.clear();completed=0;$('#random-results').replaceChildren();
    $('#random-history').hidden=true;$('#random-history').open=false;$('#random-count').textContent='';
  }
  function fail(message) {
    terminate();state=null;displayState=null;controls();
    status('The simulation stopped. Click Start random trial to try again. '+message,true);
  }
  function receive(data) {
    if(data.loading){status(data.loading+' The first visit can take a little longer.');return;}
    if(data.error){fail(data.error);return;}
    if(!data.snapshot||!data.random_agent)return;
    loading=false;pendingCommand=false;
    state=data.random_agent;displayState=normalizedSnapshot(data.snapshot);
    // A hidden page must never start a trial after a delayed runtime load.
    if(document.hidden)pendingStart=false;
    if(pendingStart){pendingStart=false;send({kind:'random-start'});return;}
    $('#random-scene').hidden=false;
    drawFrame(displayState);
    $('#random-trial').textContent=String(state.attempt);
    $('#random-time').textContent=state.elapsed_seconds.toFixed(1)+' / '+state.limit_seconds.toFixed(1)+' s';
    $('#random-decisions').textContent=String(state.decisions);
    $('#random-hold').textContent=state.decisions?state.hold_seconds.toFixed(3)+' s':'—';
    const outcome={success:'Success',spill_or_overflow:'Too much spill or overflow',time_limit:'Time limit reached'}[state.outcome]||'Finished';
    const symbols=['↻','■','↺'];
    const commands=motors=>motors.map(value=>symbols[value+1]).join(' ');
    $('#random-cup').textContent=commands(displayState.motors.slice(0,3));
    $('#random-pot').textContent=commands(displayState.motors.slice(3));
    status(state.done?'Trial complete · '+outcome+'. Start another trial to try the same unchanged policy.':
      !state.attempt?'Ready. Start a random trial.':displayState.paused?'Paused. Resume to continue this trial.':
      'Running · controls are chosen without using the coffee amounts or rewards.');
    if(state.done&&!seen.has(data.episode_id)) {
      seen.add(data.episode_id);completed++;
      const row=document.createElement('tr');
      [state.attempt,Math.round(displayState.fill*1000)+' mL',Math.round(displayState.spill*1000)+' mL',
        state.elapsed_seconds.toFixed(1)+' s',outcome].forEach(value=>{
        const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);
      });
      $('#random-results').prepend(row);
      while($('#random-results').children.length>10)$('#random-results').lastElementChild.remove();
      $('#random-history').hidden=false;$('#random-history').open=true;
      $('#random-count').textContent='('+completed+')';
    }
    if((document.hidden||pauseRequested)&&state.attempt&&!state.done&&!displayState.paused) {
      send({kind:'random-pause',paused:true});return;
    }
    controls();
  }
  function start() {
    if(!enabled||loading||pendingCommand)return;
    pauseRequested=false;
    if(worker){send({kind:'random-start'});return;}
    clearResults();loading=true;pendingStart=true;controls();status('Loading the Python runtime…');
    try {
      const current=new Worker('/coffee-worker.js',{type:'module'});worker=current;
      current.onmessage=({data})=>{if(worker===current&&enabled)receive(data);};
      current.onerror=()=>{if(worker===current)fail('Check your connection and try again.');};
      current.postMessage({kind:'init',mode:'random',bundle_url:new URL('/coffee-bundle.zip',location.href).href});
    } catch(error){fail(error.message);}
  }
  function pause() {
    pendingStart=false;pauseRequested=true;
    if(worker&&!loading)send({kind:'random-pause',paused:true});
  }
  $('#random-start').addEventListener('click',start);
  $('#random-pause').addEventListener('click',()=>{
    pauseRequested=!displayState?.paused;send({kind:'random-pause',paused:pauseRequested});
  });
  $('#random-reset').addEventListener('click',()=>{clearResults();send({kind:'random-reset'});});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  window.addEventListener('blur',pause);
  window.addEventListener('pagehide',pause);
  return {pause,setEnabled(value) {
    enabled=value;
    if(!value) {
      terminate();state=null;displayState=null;clearResults();
      if(resizeObserver)resizeObserver.disconnect();resizeObserver=null;canvasRef=null;
      $('#random-scene').hidden=true;
      status('Ready when you are. The simulation loads on first use.');
    }
    controls();
  }};
  __CANVAS_JS__
}
""".replace("__CANVAS_JS__", CANVAS_JAVASCRIPT)
