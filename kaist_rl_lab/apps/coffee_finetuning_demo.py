"""Instructor controls for bounded actor–critic refinement of a cloned policy."""

from kaist_rl_lab.apps.coffee_learning_viz import (
    LEARNING_VIZ_CSS,
    LEARNING_VIZ_HTML,
    LEARNING_VIZ_JAVASCRIPT,
)
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT

FINETUNING_HTML = """
      <section id="finetuning-panel" class="panel" aria-labelledby="finetuning-title">
        <div class="heading-row"><div><p class="eyebrow">IMPROVE WITH REWARD</p>
          <h2 id="finetuning-title">Fine-tune with actor–critic</h2></div>
          <span class="badge">Small changes · reward feedback</span></div>
        <p>Start with imitation, explore nearby actions, and learn which adjustments help.</p>
        <p class="hint">The actor makes small speed adjustments to the cloned policy. The critic learns
          to predict future reward. PPO limits policy updates, and motor commands stay within
          <b>15% of the cloned command at the current state</b>. This limits action changes;
          it does not guarantee that every new trajectory stays close or improves.</p>
        <div class="ft-training-controls">
          <div><label for="ft-episodes">Learning iterations</label>
            <select id="ft-episodes"><option value="25">25 · quick demonstration</option>
              <option value="50">50 · more practice</option><option value="100" selected>100 · full experiment</option></select></div>
          <div><label for="ft-speed">Training &amp; playback speed</label>
            <select id="ft-speed"><option value="4">4×</option><option value="8">8×</option>
              <option value="0" selected>Fastest available</option></select></div>
          <button id="ft-train" type="button" class="primary" disabled>Start fine-tuning</button>
          <button id="ft-pause" type="button" disabled>Pause</button>
          <button id="ft-stop" type="button" disabled>Stop training</button>
        </div>
        <p class="hint">Each iteration explores once, updates the policy, then evaluates the current policy
          without exploration noise from the same fixed starting pose.</p>
        <p id="ft-status" role="status" aria-live="polite">Train a behavior-cloning policy above to begin.</p>
        <p id="ft-progress" class="hint"></p>
        <p id="ft-exploration" class="hint" hidden></p>
        <div class="ft-visual-grid">
          __LEARNING_VIZ__
        <div id="ft-scene" hidden>
          <div class="random-readout" aria-live="off">
            <span>Showing <strong id="ft-showing">—</strong></span>
            <span>Simulated time <strong id="ft-time">0.0 s</strong></span>
            <span>Discounted return so far <strong id="ft-reward">0.000</strong></span>
          </div>
          <div class="coffee-stage replay-stage"><div class="coffee-canvas-wrap">
            <canvas class="coffee-canvas" role="img" aria-label="Actor–critic coffee pouring training and evaluation"></canvas>
            <div class="coffee-scene-stats" aria-label="Fine-tuned policy coffee amounts">
              <span><span class="coffee-stat-label">Cup / target</span><strong data-coffee-stat="fill">—</strong></span>
              <span><span class="coffee-stat-label">Spilled</span><strong data-coffee-stat="spill">—</strong></span>
              <span><span class="coffee-stat-label">In the pot</span><strong data-coffee-stat="remaining">—</strong></span>
            </div>
          </div></div>
          <p class="hint">The scene shows the current rollout. The chart adds its discounted return after the rollout finishes.</p>
        </div>
        </div>
        <div id="ft-results" hidden>
          <h3>Fixed starting pose · no exploration noise in evaluation</h3>
          <div class="table-scroll"><table>
            <thead><tr><th>Policy</th><th>Discounted return ↑</th><th>Cup</th><th>Spilled</th><th>Time</th><th>Result</th></tr></thead>
            <tbody id="ft-comparison"></tbody></table></div>
          <p id="ft-improvement" class="ft-improvement"></p>
          <p class="hint">The best policy has the highest evaluated discounted return, including the original clone.
            Exploration includes random actions. The evaluation curve tests every updated policy without that noise.</p>
        </div>
        <div class="button-row">
          <button id="ft-run-base" type="button" disabled>Watch original clone</button>
          <button id="ft-run-best" type="button" class="primary" disabled>Watch best policy</button>
          <button id="ft-reset" type="button" disabled>Reset displayed trial</button>
        </div>
        <details id="ft-history" hidden><summary>Discounted return across learning iterations</summary>
          <div class="table-scroll"><table>
            <thead><tr><th>Iteration</th><th>Exploration return</th><th>Current policy return · no noise</th><th>Policy update</th></tr></thead>
            <tbody id="ft-history-rows"></tbody></table></div>
        </details>
        <section id="ft-reward-explanation" class="ft-reward-explanation" aria-labelledby="ft-reward-title">
          <h3 id="ft-reward-title">Reward used for fine-tuning</h3>
          <p>Move closer to <b>700 mL</b>, avoid spilling, keep the cup upright, and use less time and motor effort.
            Reaching the goal earns a success bonus. Overshooting the target loses accuracy reward.</p>
          <p><b>Earlier reward counts more.</b> A reward earned one simulated second later receives 99% of its
            earlier weight. This makes earlier success more valuable while preserving the costs of spills and poor control.
            The chart and policy comparisons show this <b>discounted return</b>.</p>
          <details class="ft-reward-formula"><summary>Exact reward and discount formula</summary>
            <p>Every physics step lasts <b>dt = 1/32 second</b>. Let <b>e</b> be the absolute distance from
              0.700 litres in the cup, <b>Δspill</b> the litres spilled in this step, <b>a₁ … a₆</b> the six
              controls between −1 and 1, and <b>θ</b> the cup angle in radians.</p>
            <p class="ft-equation">rₜ = 20(eₜ − eₜ₊₁) − 40Δspill − 0.024dt ∑ᵢ₌₁⁶ aᵢ² − 0.032dt |θ| − 0.008dt</p>
            <p>At the final step, also add:</p>
            <p class="ft-equation">(15 if successful, otherwise 0) − 10 × final error − 14 × total spill</p>
            <p>Errors and spill volumes are in litres. The final penalty is in addition to the step penalties.</p>
            <p><b>Success requires all five conditions:</b> the cup is within 40 mL of the 700 mL target,
              total spill ≤20 mL, flow rate ≤8 mL/s, cup tilt ≤8°, and pot tilt ≤12°.
              Each rollout has a 60-second time limit.</p>
            <p class="ft-equation">G = ∑ₜ₌₀ᵀ⁻¹ γᵗ rₜ, with γ = 0.99<sup>1/32</sup></p>
            <p>All terms, including the final bonus and penalties, use the same discount. The existing trajectory
              library lists undiscounted total reward; fine-tuning compares discounted return.</p>
          </details>
        </section>
        <details><summary>What is being learned?</summary>
          <p class="hint">The starting behavior-cloning policy stays fixed. The learned actor adjusts its speed using the
            current state; it does not invent a new direction of motion. Exploration samples a new random speed
            adjustment every 0.25 simulated seconds. Initially, about 95% of speed multipliers fall between 0.87 and 1.13;
            all stay between 0.85 and 1.15. The critic estimates discounted returns, and the actor uses those estimates
            to improve expected discounted return.</p>
          <p class="hint">Reward combines pouring accuracy, spilled coffee, vessel stability, elapsed time, and motor effort.
            Faster is not always better. Exploration, evaluation, and policy playback all reuse one fixed starting
            pose, with a 60-second time limit. Demonstrations use a wider range of random starting poses.</p>
          <p class="hint">Choose 4×, 8×, or the fastest speed your device can run. Speed changes how quickly the simulation
            is shown; every physics step and policy decision is still computed. Switching away pauses it.
            Stop keeps the best completed evaluation. Changing the cloning policy or reloading clears this experiment.</p>
        </details>
      </section>
""".replace("__LEARNING_VIZ__", LEARNING_VIZ_HTML)

FINETUNING_CSS = LEARNING_VIZ_CSS + """
.ft-visual-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:20px;align-items:start;margin:16px 0;}
.ft-visual-grid>* {min-width:0;}
.ft-visual-grid .replay-stage {margin-top:10px;}
.ft-visual-grid #ft-scene {padding:14px;background:#f8fafb;border:1px solid var(--line);border-radius:10px;}
.ft-visual-grid .random-readout {margin-top:0;gap:8px 16px;}
.ft-training-controls {display:flex;align-items:end;gap:12px;flex-wrap:wrap;}
.ft-training-controls>div {min-width:220px;flex:1;}
#ft-status {color:var(--navy);min-height:21px;font-size:14px;}
#ft-status.error {color:#a03929;}
#ft-results {margin:18px 0;padding:16px;background:#f0f6f3;border:1px solid #d5e5dc;border-radius:10px;}
#ft-results h3 {font-size:16px;margin:0 0 8px;}
.ft-improvement {font-weight:700;color:var(--teal);}
.ft-reward-explanation {margin:18px 0;padding:16px;background:#f5f8fb;border:1px solid var(--line);border-radius:10px;font-size:13px;}
.ft-reward-explanation h3 {font-size:16px;margin:0 0 8px;}
.ft-reward-explanation p {line-height:1.6;margin:8px 0;}
.ft-reward-formula {margin-top:10px;}
.ft-reward-formula summary {font-weight:700;cursor:pointer;}
.ft-equation {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;overflow-x:auto;padding:8px;background:#fff;border-radius:6px;}
#ft-comparison td,#ft-history-rows td {white-space:nowrap;}
@media(max-width:580px) {.ft-training-controls>div {flex-basis:100%;min-width:0;}#ft-train {flex-basis:100%;}#ft-results {padding:10px;}}
"""

FINETUNING_JAVASCRIPT = r"""
function createFineTuningDemo(element,beforeRun) {
  const $=selector=>element.querySelector(selector);
  const learningViz=createLearningVisualization(element);
  const LOGICAL_WIDTH=960,LOGICAL_HEIGHT=560;
  const clamp=(value,low,high)=>Math.max(low,Math.min(high,value));
  let canvasRef=null,resizeObserver=null,displayState=null;
  let enabled=false,model=null,worker=null,state=null,loading=false,modelSent=false;
  let pendingCommand=false,pendingAction=null,pauseRequested=false,renderedResult='';
  const number=(value,digits=3)=>Number.isFinite(value)?value.toFixed(digits):'—';
  function status(message,error=false) {
    $('#ft-status').textContent=message;$('#ft-status').classList.toggle('error',error);
  }
  function controls() {
    const busy=loading||pendingCommand,training=!!state?.training_active;
    $('#ft-train').disabled=!enabled||!model||busy||training;
    $('#ft-train').textContent=loading?'Loading simulation…':state?.has_result?'Start a new experiment':'Start fine-tuning';
    $('#ft-episodes').disabled=busy||training;
    $('#ft-speed').disabled=!enabled||!model||busy;
    $('#ft-pause').disabled=!enabled||busy||!(training||state?.rollout?.active);
    $('#ft-pause').textContent=state?.paused?'Resume':'Pause';
    $('#ft-stop').disabled=!enabled||loading||!training;
    $('#ft-run-base').disabled=!enabled||!model||busy||training;
    $('#ft-run-best').disabled=!enabled||!state?.best_available||busy||training;
    $('#ft-reset').disabled=!enabled||busy||training||!state;
  }
  function stopWorker() {
    if(worker)worker.terminate();worker=null;
    loading=false;modelSent=false;pendingCommand=false;pendingAction=null;state=null;displayState=null;
  }
  function clearExperiment() {
    stopWorker();renderedResult='';learningViz.reset();
    $('#ft-results').hidden=true;$('#ft-history').hidden=true;$('#ft-scene').hidden=true;
    $('#ft-comparison').replaceChildren();$('#ft-history-rows').replaceChildren();
    $('#ft-progress').textContent='';$('#ft-improvement').textContent='';
    $('#ft-exploration').hidden=true;$('#ft-exploration').textContent='';
    status(model?'Cloned policy ready. Start fine-tuning or watch the original clone.':'Train a behavior-cloning policy above to begin.');
    controls();
  }
  function send(command) {
    if(!worker||!enabled)return;
    pendingCommand=true;controls();worker.postMessage(command);
  }
  function row(parent,values) {
    const tr=document.createElement('tr');
    values.forEach(value=>{const td=document.createElement('td');td.textContent=String(value);tr.append(td);});
    parent.append(tr);
  }
  function renderResults(result) {
    const signature=JSON.stringify(result&&[result.baseline,result.best,result.history]);
    if(signature===renderedResult)return;renderedResult=signature;
    $('#ft-comparison').replaceChildren();$('#ft-history-rows').replaceChildren();
    $('#ft-results').hidden=!result?.baseline;
    $('#ft-history').hidden=!(result?.history?.length);
    $('#ft-exploration').hidden=true;$('#ft-exploration').textContent='';
    if(!result?.baseline)return;
    [['Original clone',result.baseline],['Best evaluated policy',result.best]].forEach(([label,value])=>{
      if(value)row($('#ft-comparison'),[label,number(value.return),number(value.fill_ml,1)+' mL',
        number(value.spill_ml,1)+' mL',number(value.seconds,1)+' s',value.success?'Success':'Attempt']);
    });
    const delta=(result.best?.return??result.baseline.return)-result.baseline.return;
    $('#ft-improvement').textContent=delta>1e-8?'Best discounted return increased by '+number(delta)+
      ' · checkpoint after iteration '+result.best.episode+'.':
      'No better checkpoint yet. The original cloned policy is retained.';
    const history=result.history||[];
    $('#ft-history').hidden=!history.length;
    const exploration=history.at(-1)?.update?.exploration;
    if(Number.isFinite(exploration?.speed_min)&&Number.isFinite(exploration?.speed_max)) {
      $('#ft-exploration').hidden=false;
      $('#ft-exploration').textContent='Latest exploration · speed multipliers '+number(100*exploration.speed_min,1)+
        '–'+number(100*exploration.speed_max,1)+'% of the clone'+
        (Number.isInteger(exploration.slower_decisions)&&Number.isInteger(exploration.faster_decisions)?
          ' · '+exploration.slower_decisions+' slower / '+exploration.faster_decisions+
          ' faster adjustments than the policy being explored.':'.');
    }
    history.forEach(item=>row($('#ft-history-rows'),[item.episode,number(item.training.return),
      number(item.evaluation?.return),item.update.actor_change>0?'Applied':'No change']));
  }
  function fail(message) {
    if(state)learningViz.render({...state,training_active:false,training_stopped:true,paused:true,
      rollout:{...state.rollout,active:false,done:false}});
    stopWorker();$('#ft-scene').hidden=true;controls();
    status('The experiment stopped. Start a new experiment to try again. '+message,true);
  }
  function receive(data) {
    if(data.loading){status(data.loading);return;}
    if(data.error){fail(data.error);return;}
    if(!data.snapshot||!data.finetuning)return;
    loading=false;pendingCommand=false;state=data.finetuning;
    displayState=normalizedSnapshot(data.snapshot);
    if(!modelSent){modelSent=true;send({kind:'ft-load',model});return;}
    if(!state.model_loaded){fail('The cloned policy could not be loaded.');return;}
    if(document.hidden||pauseRequested)pendingAction=null;
    if(pendingAction){const action=pendingAction;pendingAction=null;send(action);return;}
    $('#ft-scene').hidden=false;drawFrame(displayState);
    renderResults(state.result);learningViz.render(state);
    const progress=state.progress||{},rollout=state.rollout||{};
    const phase={baseline:'Checking original clone',training:'Exploring nearby actions',evaluation:'Testing the updated policy',complete:'Training complete'}[progress.phase]||'Ready';
    const training=state.training_active;
    $('#ft-showing').textContent=training?phase:rollout.active||rollout.done?
      (rollout.policy==='best'?'Best evaluated policy':'Original clone'):
      rollout.elapsed_seconds===0?'Starting pose':
      progress.phase==='complete'?'Last evaluated policy':state.has_result?'Stopped experiment':'Ready';
    $('#ft-time').textContent=number(training?progress.elapsed_seconds:rollout.elapsed_seconds,1)+' s';
    $('#ft-reward').textContent=number(training?progress.reward:rollout.reward);
    $('#ft-progress').textContent=state.has_result||training?
      'Completed '+(progress.completed_episodes||0)+' / '+(progress.episodes||0)+' learning iterations · '+
      (training?(state.playback_speed?state.playback_speed+'× requested':'fastest available')+' · fixed starting pose':'experiment stopped or completed'):'';
    const outcome={success:'Success',spill_or_overflow:'Too much spill or overflow',time_limit:'Time limit reached'}[rollout.outcome]||'Finished';
    if(training)status((state.paused?'Paused · ':'')+phase+(progress.episode?' · iteration '+progress.episode+' / '+progress.episodes:''));
    else if(rollout.done)status(outcome+' · discounted return '+number(rollout.reward)+' · '+
      Math.round(displayState.fill*1000)+' mL in the cup, '+Math.round(displayState.spill*1000)+' mL spilled.');
    else if(rollout.active)status(state.paused?'Policy paused. Resume to continue.':'Watching the policy without exploration noise.');
    else if(state.has_result)status('Experiment ready to compare. Watch the original clone and the best evaluated policy.');
    else status('Cloned policy loaded. Start fine-tuning when ready.');
    if((document.hidden||pauseRequested)&&!state.paused&&(training||rollout.active)){
      send({kind:'ft-pause',paused:true});return;
    }
    controls();
  }
  function startAction(command) {
    if(!enabled||!model||loading||pendingCommand)return;
    command={...command,speed:Number($('#ft-speed').value)};
    if(command.kind==='ft-train')learningViz.reset();
    beforeRun();pauseRequested=false;
    if(worker){send(command);return;}
    pendingAction=command;loading=true;modelSent=false;controls();status('Loading the fine-tuning simulation…');
    try {
      const current=new Worker('/coffee-worker.js',{type:'module'});worker=current;
      current.onmessage=({data})=>{if(worker===current&&enabled)receive(data);};
      current.onerror=()=>{if(worker===current)fail('Check your connection and try again.');};
      current.postMessage({kind:'init',mode:'finetuning',bundle_url:new URL('/coffee-bundle.zip',location.href).href});
    } catch(error){fail(error.message);}
  }
  function pause() {
    pauseRequested=true;pendingAction=null;
    if(worker&&!loading&&state?.model_loaded)send({kind:'ft-pause',paused:true});
  }
  $('#ft-train').addEventListener('click',()=>startAction({kind:'ft-train',episodes:Number($('#ft-episodes').value)}));
  $('#ft-run-base').addEventListener('click',()=>startAction({kind:'ft-run',policy:'base'}));
  $('#ft-run-best').addEventListener('click',()=>startAction({kind:'ft-run',policy:'best'}));
  $('#ft-pause').addEventListener('click',()=>{
    if(state?.paused)beforeRun();pauseRequested=!state?.paused;send({kind:'ft-pause',paused:pauseRequested});
  });
  $('#ft-stop').addEventListener('click',()=>{pendingAction=null;send({kind:'ft-stop'});});
  $('#ft-reset').addEventListener('click',()=>send({kind:'ft-reset'}));
  $('#ft-speed').addEventListener('change',()=>{
    if(worker&&!loading&&state?.model_loaded)send({kind:'ft-speed',speed:Number($('#ft-speed').value)});
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  window.addEventListener('blur',pause);window.addEventListener('pagehide',pause);
  return {pause,setModel(value){model=value;clearExperiment();},setEnabled(value){
    enabled=value;if(!value){model=null;clearExperiment();
      if(resizeObserver)resizeObserver.disconnect();resizeObserver=null;canvasRef=null;}
    controls();
  }};
  __CANVAS_JS__
}
""".replace("__CANVAS_JS__", CANVAS_JAVASCRIPT) + LEARNING_VIZ_JAVASCRIPT
