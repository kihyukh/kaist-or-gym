"""Instructor workflow for fitting and running a policy from demonstrations."""

from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT

CLONING_DEMO_HTML = """
      <section id="cloning-panel" class="panel" aria-labelledby="cloning-title">
        <div class="heading-row"><div><p class="eyebrow">LEARN FROM DEMONSTRATIONS</p>
          <h2 id="cloning-title">Behavior cloning</h2></div><span class="badge">Imitation · no reward learning</span></div>
        <p>Can the robot copy a good example?</p>
        <p class="hint">This simple policy finds the most similar recorded state and copies its action.
          It chooses again at every simulation step. Training uses state–action pairs, without optimizing rewards.</p>
        <div class="cloning-training">
          <div><label for="cloning-source">Demonstrations to learn from</label>
            <select id="cloning-source"><option value="students">Selected class's student submissions</option>
              <option value="examples">Generated successful examples</option></select></div>
          <label class="checkbox"><input id="cloning-successful" type="checkbox" checked> Use successful attempts only</label>
          <button id="cloning-train" type="button" class="primary" disabled>Train policy</button>
        </div>
        <p id="cloning-dataset" class="hint">Select a class to use its student demonstrations, or choose generated examples.</p>
        <p id="cloning-status" role="status" aria-live="polite">Train a policy, then watch it try to pour 700 mL.</p>
        <div id="cloning-training-result" class="cloning-report" hidden>
          <h3 id="cloning-model-title">Nearest-neighbor behavior cloning</h3>
          <p id="cloning-training-summary"></p><p id="cloning-selection" class="hint"></p><p id="cloning-validation" class="hint"></p>
          <p class="hint">The trained policy stays in this tab. New submissions require retraining.</p>
        </div>
        <div class="button-row">
          <button id="cloning-run" class="primary" type="button" disabled>Run cloned policy</button>
          <button id="cloning-pause" type="button" disabled>Pause policy</button>
          <button id="cloning-reset" type="button" disabled>Reset policy trial</button>
        </div>
        <div id="cloning-scene" hidden>
          <div class="random-readout" aria-live="off">
            <span>Policy <strong>Nearest-neighbor imitation</strong></span>
            <span>Trial <strong id="cloning-trial">0</strong></span>
            <span>Simulated time <strong id="cloning-time">0.0 s</strong></span>
            <span>Outcome <strong id="cloning-outcome">Ready</strong></span>
          </div>
          <div class="coffee-stage replay-stage"><div class="coffee-canvas-wrap">
            <canvas class="coffee-canvas" role="img" aria-label="Live behavior-cloned coffee pouring policy"></canvas>
            <div class="coffee-scene-stats" aria-label="Cloned policy coffee amounts">
              <span><span class="coffee-stat-label">Cup / target</span><strong data-coffee-stat="fill">—</strong></span>
              <span><span class="coffee-stat-label">Spilled</span><strong data-coffee-stat="spill">—</strong></span>
              <span><span class="coffee-stat-label">In the pot</span><strong data-coffee-stat="remaining">—</strong></span>
            </div>
          </div></div>
          <p class="hint">This is a live policy acting on the current state, not a recording replay.
            Every trial uses the classroom starting pose. Success here does not test new starting poses.</p>
        </div>
      </section>
"""

CLONING_DEMO_CSS = """
.cloning-training {display:flex;gap:16px;align-items:end;flex-wrap:wrap;}
.cloning-training>div {flex:1;min-width:240px;}
.cloning-training>.checkbox {align-self:center;padding-top:20px;}
#cloning-status {color:var(--navy);font-size:14px;min-height:21px;}
#cloning-status.error {color:#a03929;}
.cloning-report {background:#edf5f1;border-left:4px solid var(--teal);padding:12px 16px;margin:16px 0;}
.cloning-report h3 {font-size:15px;margin:0;}
.cloning-report p {font-size:14px;}
.cloning-report .hint {font-size:12px;}
@media(max-width:580px) {.cloning-training {display:block;}.cloning-training>div {min-width:0;}.cloning-training>.checkbox {padding-top:0;}#cloning-train {width:100%;}}
"""

CLONING_DEMO_JAVASCRIPT = r"""
function createCloningDemo(element,post,beforeRun,onModel=()=>{}) {
  const $=selector=>element.querySelector(selector);
  const LOGICAL_WIDTH=960, LOGICAL_HEIGHT=560;
  const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
  let canvasRef=null,resizeObserver=null,displayState=null;
  let enabled=false,context={id:null,name:'',total:0,successful:0};
  let model=null,worker=null,state=null,epoch=0,training=false,loading=false;
  let pendingStart=false,pendingCommand=false,pauseRequested=false,modelSent=false;
  function status(message,error=false) {
    $('#cloning-status').textContent=message;$('#cloning-status').classList.toggle('error',error);
  }
  function dataset() {
    const examples=$('#cloning-source').value==='examples';
    $('#cloning-successful').disabled=examples;
    $('#cloning-dataset').textContent=examples?
      'Generated examples are separate from student submissions. They use the same physics, starting pose, and 700 mL target.':
      context.id?context.name+' · '+context.total+' submitted attempts · '+context.successful+' marked successful.':
      'Select a class to use its student demonstrations, or choose generated examples.';
  }
  function controls() {
    const examples=$('#cloning-source').value==='examples';
    const available=examples||(context.id&&($('#cloning-successful').checked?context.successful:context.total)>0);
    $('#cloning-train').disabled=!enabled||training||!available;
    $('#cloning-train').textContent=training?'Training policy…':model?'Retrain policy':'Train policy';
    $('#cloning-run').disabled=!enabled||training||loading||pendingCommand||!model;
    $('#cloning-run').textContent=loading?'Loading policy simulation…':'Run cloned policy';
    $('#cloning-pause').disabled=!enabled||training||loading||pendingCommand||!state?.attempt||state.done;
    $('#cloning-pause').textContent=displayState?.paused?'Resume policy':'Pause policy';
    $('#cloning-reset').disabled=!enabled||training||loading||pendingCommand||!state;
  }
  function stopWorker() {
    if(worker)worker.terminate();worker=null;
    loading=false;pendingStart=false;pendingCommand=false;modelSent=false;
    state=null;displayState=null;$('#cloning-scene').hidden=true;
  }
  function clearPolicy() {
    epoch++;training=false;model=null;stopWorker();onModel(null);
    $('#cloning-training-result').hidden=true;
    $('#cloning-training-summary').textContent='';$('#cloning-validation').textContent='';$('#cloning-selection').textContent='';
    status('Train a policy, then watch it try to pour 700 mL.');controls();
  }
  async function train() {
    if(!enabled||training)return;
    beforeRun();clearPolicy();training=true;const request=epoch;
    controls();status('Reading demonstrations and fitting the state–action policy…');
    try {
      const source=$('#cloning-source').value;
      const result=await post('/api/instructor/cloning/train',{
        source,session_id:source==='students'?context.id:null,
        successful_only:$('#cloning-successful').checked,
      });
      if(!enabled||request!==epoch)return;
      if(!result.model?.metrics)throw new Error('Training did not return a usable policy.');
      model=result.model;
      const metrics=model.metrics;
      $('#cloning-training-result').hidden=false;
      $('#cloning-training-summary').textContent=result.source_label+' · '+metrics.demonstrations+
        ' demonstrations · '+metrics.training_samples.toLocaleString()+' state–action examples';
      const selection=result.selection||{};
      const exclusions=[];
      if(selection.skipped_unsuccessful)exclusions.push(selection.skipped_unsuccessful+' unsuccessful attempts excluded');
      if(selection.skipped_invalid||selection.skipped_incompatible)
        exclusions.push((selection.skipped_invalid+selection.skipped_incompatible)+' invalid or incompatible attempts excluded');
      if(selection.capped_trajectories)exclusions.push(selection.capped_trajectories+' additional attempts omitted by the demo limit');
      if(metrics.total_steps>50000)exclusions.push('Training pairs sampled evenly across trajectories (50,000 maximum)');
      $('#cloning-selection').textContent=exclusions.join(' · ');
      const error=metrics.heldout_action_mae;
      $('#cloning-validation').textContent=Number.isFinite(error)?
        'Held-out action error: '+error.toFixed(3)+' (motor commands range from −1 to +1). '+
        metrics.validation_trajectories+' whole '+(metrics.validation_trajectories===1?'trajectory':'trajectories')+
        ' held out for this check; the final policy fits all selected examples. '+
        'Action agreement is not a pouring success rate.':
        'No held-out action check: at least two demonstrations are needed. Test the policy in the simulation below.';
      status('Policy trained. Click Run cloned policy to test it in the simulation.');
      onModel(model);
    } catch(error) {if(enabled&&request===epoch){
      model=null;$('#cloning-training-result').hidden=true;status(error.message,true);
    }}
    finally {if(request===epoch){training=false;controls();}}
  }
  function send(command) {
    if(!worker||!enabled)return;
    pendingCommand=true;controls();worker.postMessage(command);
  }
  function fail(message) {
    stopWorker();controls();status('Policy simulation stopped. Click Run cloned policy to retry. '+message,true);
  }
  function receive(data) {
    if(data.loading){status(data.loading+' The first visit can take a little longer.');return;}
    if(data.error){fail(data.error);return;}
    if(!data.snapshot||!data.cloning_agent)return;
    loading=false;pendingCommand=false;state=data.cloning_agent;
    displayState=normalizedSnapshot(data.snapshot);
    if(!modelSent) {
      modelSent=true;send({kind:'cloning-load',model});return;
    }
    if(!state.model_loaded){fail('The trained policy could not be loaded.');return;}
    if(document.hidden||pauseRequested)pendingStart=false;
    if(pendingStart){pendingStart=false;send({kind:'cloning-start'});return;}
    $('#cloning-scene').hidden=false;drawFrame(displayState);
    $('#cloning-trial').textContent=String(state.attempt);
    $('#cloning-time').textContent=state.elapsed_seconds.toFixed(1)+' / '+state.limit_seconds.toFixed(1)+' s';
    const outcome={success:'Success',spill_or_overflow:'Too much spill or overflow',time_limit:'Time limit reached'}[state.outcome]||'Finished';
    $('#cloning-outcome').textContent=state.done?outcome:!state.attempt?'Ready':displayState.paused?'Paused':'Running';
    status(state.done?'Trial complete · '+outcome+' · '+Math.round(displayState.fill*1000)+' mL in the cup, '+
      Math.round(displayState.spill*1000)+' mL spilled.':!state.attempt?'Policy loaded. Click Run cloned policy.':
      displayState.paused?'Policy paused. Resume to continue.':'Running the cloned policy on the current state.');
    if((document.hidden||pauseRequested)&&state.attempt&&!state.done&&!displayState.paused){
      send({kind:'cloning-pause',paused:true});return;
    }
    controls();
  }
  function run() {
    if(!enabled||!model||training||loading||pendingCommand)return;
    beforeRun();pauseRequested=false;
    if(worker){send({kind:'cloning-start'});return;}
    pendingStart=true;loading=true;modelSent=false;controls();status('Loading the policy simulation…');
    try {
      const current=new Worker('/coffee-worker.js',{type:'module'});worker=current;
      current.onmessage=({data})=>{if(worker===current&&enabled)receive(data);};
      current.onerror=()=>{if(worker===current)fail('Check your connection and try again.');};
      current.postMessage({kind:'init',mode:'cloning',bundle_url:new URL('/coffee-bundle.zip',location.href).href});
    } catch(error){fail(error.message);}
  }
  function pause() {
    pauseRequested=true;pendingStart=false;
    if(worker&&!loading&&state?.model_loaded)send({kind:'cloning-pause',paused:true});
  }
  $('#cloning-train').addEventListener('click',train);
  $('#cloning-run').addEventListener('click',run);
  $('#cloning-pause').addEventListener('click',()=>{
    if(displayState?.paused)beforeRun();
    pauseRequested=!displayState?.paused;send({kind:'cloning-pause',paused:pauseRequested});
  });
  $('#cloning-reset').addEventListener('click',()=>send({kind:'cloning-reset'}));
  $('#cloning-source').addEventListener('change',()=>{clearPolicy();dataset();controls();});
  $('#cloning-successful').addEventListener('change',()=>{clearPolicy();controls();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  window.addEventListener('blur',pause);window.addEventListener('pagehide',pause);
  return {pause,setEnabled(value) {
    enabled=value;if(!value){context={id:null,name:'',total:0,successful:0};clearPolicy();
      if(resizeObserver)resizeObserver.disconnect();resizeObserver=null;canvasRef=null;}
    dataset();controls();
  },setContext(value) {
    const changed=value.id!==context.id;context=value;
    if(changed)clearPolicy();dataset();controls();
  }};
  __CANVAS_JS__
}
""".replace("__CANVAS_JS__", CANVAS_JAVASCRIPT)
