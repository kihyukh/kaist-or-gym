"""Instructor controls for reward-based refinement of a cloned policy."""

from kaist_rl_lab.apps.coffee_classroom import ARM_BASE_DISTANCE_M
from kaist_rl_lab.apps.coffee_learning_viz import (
    LEARNING_VIZ_CSS,
    LEARNING_VIZ_HTML,
    LEARNING_VIZ_JAVASCRIPT,
)
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_JAVASCRIPT

FINETUNING_HTML = """
      <section id="finetuning-panel" class="panel" aria-labelledby="finetuning-title">
        <div class="heading-row"><div><p class="eyebrow">IMPROVE WITH REWARD</p>
          <h2 id="finetuning-title">Fine-tune with reinforcement learning</h2></div>
          <span class="badge">700 mL · ±5 mL precision goal</span></div>
        <p>Start with imitation, try nearby speed adjustments, and use reward to choose what to keep.</p>
        <div class="ft-training-controls">
          <div><label for="ft-strategy">Learning method</label>
            <select id="ft-strategy"><option value="policy_search" selected>Policy search · classroom demo</option>
              <option value="ppo">Actor–critic · PPO</option></select></div>
          <div><label for="ft-episodes">Learning iterations</label>
            <select id="ft-episodes"><option value="10" selected>10 · quick demonstration</option>
              <option value="25">25 · more practice</option><option value="50">50 · longer run</option>
              <option value="100">100 · full experiment</option></select></div>
          <div><label for="ft-speed">Training speed</label>
            <select id="ft-speed"><option value="4">4×</option><option value="8">8×</option>
              <option value="0" selected>Fastest available</option></select></div>
          <button id="ft-train" type="button" class="primary" disabled>Start fine-tuning</button>
          <button id="ft-pause" type="button" disabled>Pause</button>
          <button id="ft-stop" type="button" disabled>Stop training</button>
        </div>
        <p id="ft-method-description" class="hint">Policy search tests paired faster/slower settings for approaching/pouring and
          returning the pot, keeping a candidate only when its score improves.</p>
        <p class="hint">Each iteration completes one exploratory or candidate trial, decides how to update the policy,
          then evaluates the current policy without exploration noise from the same fixed starting pose.
          The arm bases are <b>__ARM_SPACING__ m apart</b>; demonstrations and policy trials use this same spacing.</p>
        <p id="ft-status" role="status" aria-live="polite">Train a behavior-cloning policy above to begin.</p>
        <p id="ft-progress" class="hint"></p>
        <p id="ft-exploration" class="hint" hidden></p>
        <div class="ft-visual-grid">
          __LEARNING_VIZ__
        <div id="ft-scene" hidden>
          <div class="random-readout" aria-live="off">
            <span>Showing <strong id="ft-showing">—</strong></span>
            <span>Simulated time <strong id="ft-time">0.0 s</strong></span>
            <span>Accuracy / speed score so far <strong id="ft-reward">0.000</strong></span>
          </div>
          <div class="coffee-stage replay-stage"><div class="coffee-canvas-wrap">
            <canvas class="coffee-canvas" role="img" aria-label="Reinforcement learning coffee pouring trials and evaluation"></canvas>
            <div class="coffee-scene-stats" aria-label="Fine-tuned policy coffee amounts">
              <span><span class="coffee-stat-label">Cup / target</span><strong data-coffee-stat="fill">—</strong></span>
              <span><span class="coffee-stat-label">Spilled</span><strong data-coffee-stat="spill">—</strong></span>
              <span><span class="coffee-stat-label">In the pot</span><strong data-coffee-stat="remaining">—</strong></span>
            </div>
          </div></div>
          <p class="hint">The scene shows the current rollout. The chart adds its Accuracy / speed score after the rollout finishes.</p>
        </div>
        </div>
        <div id="ft-results" hidden>
          <h3>Fixed starting pose · no exploration noise in evaluation</h3>
          <div class="table-scroll"><table>
            <thead><tr><th>Policy</th><th>Accuracy / speed score ↑</th><th>Cup</th><th>Target error ↓</th><th>Within ±5 mL</th><th>Spilled</th><th>Time</th><th>Result</th></tr></thead>
            <tbody id="ft-comparison"></tbody></table></div>
          <p id="ft-improvement" class="ft-improvement"></p>
          <p class="hint">The best policy has the highest evaluated Accuracy / speed score, including the original clone.
            Candidate or exploratory trials are shown separately from the current policy's evaluation without noise.
            Aim for <b>exactly 700 mL</b>; <b>within ±5 mL</b> is the precision goal.
            The original environment's wider ±40 mL completion tolerance is not the precision goal.</p>
        </div>
        <div class="ft-playback-controls">
          <div><label for="ft-playback-speed">Policy playback speed</label>
            <select id="ft-playback-speed"><option value="4" selected>4×</option><option value="8">8×</option>
              <option value="0">Fastest available</option></select></div>
          <button id="ft-run-base" type="button" disabled>Watch original clone</button>
          <button id="ft-run-best" type="button" class="primary" disabled>Watch best policy</button>
          <button id="ft-reset" type="button" disabled>Reset displayed trial</button>
        </div>
        <details id="ft-history" hidden><summary>Measurements for every tested policy</summary>
          <p class="hint">Every completed candidate and separate evaluation is listed, including rejected candidates.
            Scores show six decimals, liquid amounts four, and time five. These are simulation measurements;
            policy selection uses the full stored values.</p>
          <div class="table-scroll"><table>
            <thead><tr><th>Iteration</th><th>Policy / trial</th><th>Score ↑</th><th>Cup (mL)</th><th>Target error (mL) ↓</th><th>Time (s)</th><th>Spill (mL)</th><th>Result</th><th>Policy update</th></tr></thead>
            <tbody id="ft-history-rows"></tbody></table></div>
        </details>
        <section id="ft-reward-explanation" class="ft-reward-explanation" aria-labelledby="ft-reward-title">
          <h3 id="ft-reward-title">Reward used for fine-tuning</h3>
          <p><b>Accuracy comes first: aim for exactly 700 mL, within ±5 mL.</b> A successful pour earns a precision
            bonus that peaks at 1,000 points exactly at 700 mL and is still at least 606.5 points within ±5 mL.
            A quick pour far from the target cannot make up for missing that precision.</p>
          <p>Success also earns 100 points; failure or timeout loses 100 points and earns no precision bonus.
            Each simulated second costs 1 point. Target error, spilling, motor effort, and cup tilt also cost points.</p>
          <p><b>Earlier reward counts more.</b> Reward one simulated second later receives 99% of its earlier weight.
            The chart and policy comparisons use this discounted <b>Accuracy / speed score</b>, with the same scoring rule
            for the original clone and every new policy. The trajectory library keeps its original
            <b>undiscounted recorded reward</b>; those values use a different reward function.</p>
          <details class="ft-reward-formula"><summary>Exact reward and discount formula</summary>
            <p>Every physics step lasts <b>dt = 1/32 second</b>. Let <b>e</b> be the absolute distance from
              0.700 litres in the cup, <b>Δspill</b> the litres spilled in this step, <b>a₁ … a₆</b> the six
              controls between −1 and 1, and <b>θ</b> the cup angle in radians.</p>
            <p class="ft-equation">rₜ = −dt − 40Δspill − 0.024dt ∑ᵢ₌₁⁶ aᵢ² − 0.032dt |θ| + γΦₜ₊₁ − Φₜ</p>
            <p>At the final step, also add:</p>
            <p class="ft-equation">(100 + A(e) if successful, otherwise −100) − 100 × final error − 14 × total spill</p>
            <p class="ft-equation">A(e) = 1000 exp[−½(e / 0.005)²]</p>
            <p>Here e is in litres: 0.005 L is 5 mL. The precision bonus is awarded only on success;
              it is largest at zero error and decreases continuously as the pour misses 700 mL.</p>
            <p class="ft-equation">Φₜ = −20eₜ; Φ = 0 at every terminal state, including timeout</p>
            <p>The Φ term provides feedback while filling. Its discounted total is the same 14 points for
              every trial starting with an empty cup, so it does not change which completed policy scores best.
              Errors and spill volumes are in litres; final penalties are additional to step penalties.</p>
            <p><b>Success requires all five conditions:</b> the cup is within 40 mL of the 700 mL target,
              total spill ≤20 mL, flow rate ≤8 mL/s, cup tilt ≤8°, and pot tilt ≤12°.
              Each rollout has a 60-second time limit. A fast failed attempt still scores below a slow success.</p>
            <p class="ft-equation">G = ∑ₜ₌₀ᵀ⁻¹ γᵗ rₜ, with γ = 0.99<sup>1/32</sup></p>
            <p>All terms use the same discount. Even a successful pour within ±5 mL that takes the full
              60 seconds outranks a successful pour at least 10 mL off target, however fast it finishes.
              Speed still matters among accurate pours. Animation speed does not affect the score.</p>
          </details>
        </section>
        <details><summary>What is being learned?</summary>
          <p class="hint">The behavior-cloning policy stays fixed. Both methods adjust its speed and continue to choose
            controls from the current state. Zero commands remain zero and motion directions stay those of the clone.
            They do not call the demonstration-generating controller.</p>
          <p class="hint"><b>Policy search</b> learns two speed settings: one for approaching and pouring,
            another for returning the pot. It tries paired faster/slower adjustments to one setting at a time,
            shrinking the search range when a pair fails to improve. Both multipliers stay between 0.7 and 1.4
            times the clone's commands. Paired candidates share the same starting settings even if the first
            is accepted. Controls are always clipped to −1…1; liquid measurements and the 700 mL target stay real.</p>
          <p class="hint"><b>Actor–critic (PPO)</b> explores speed changes during a trial. A critic estimates future
            Accuracy / speed scores, and the actor uses those estimates to update its state-dependent speed policy.
            Speed multipliers stay between 0.5 and 1.5 times the clone; exploration is resampled every
            0.5 simulated seconds. An explored trajectory or policy update can still be worse.</p>
          <p class="hint">Both methods use the same Accuracy / speed score, fixed starting pose, and 60-second limit.
            Demonstrations use a wider range of random starting poses. Improvement is not guaranteed.</p>
          <p class="hint">Training defaults to the fastest available speed; policy playback defaults to 4×.
            Both controls also offer 8×. Speed changes how quickly the simulation is shown; every physics step and policy decision is still computed. Switching away pauses it.
            Stop keeps the best completed evaluation. Changing the cloning policy or reloading clears this experiment.</p>
        </details>
      </section>
""".replace("__LEARNING_VIZ__", LEARNING_VIZ_HTML).replace("__ARM_SPACING__", f"{ARM_BASE_DISTANCE_M:.2f}")

FINETUNING_CSS = LEARNING_VIZ_CSS + """
.ft-visual-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));gap:20px;align-items:start;margin:16px 0;}
.ft-visual-grid>* {min-width:0;}
.ft-visual-grid .replay-stage {margin-top:10px;}
.ft-visual-grid #ft-scene {padding:14px;background:#f8fafb;border:1px solid var(--line);border-radius:10px;}
.ft-visual-grid .random-readout {margin-top:0;gap:8px 16px;}
.ft-playback-controls {display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin-top:16px;}
.ft-playback-controls>div {min-width:180px;}
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
  const number=(value,digits=6)=>Number.isFinite(value)?value.toFixed(digits):'—';
  const targetError=value=>Number.isFinite(value?.fill_ml)?Math.abs(value.fill_ml-700):null;
  function status(message,error=false) {
    $('#ft-status').textContent=message;$('#ft-status').classList.toggle('error',error);
  }
  function controls() {
    const busy=loading||pendingCommand,training=!!state?.training_active;
    $('#ft-train').disabled=!enabled||!model||busy||training;
    $('#ft-train').textContent=loading?'Loading simulation…':state?.has_result?'Start a new experiment':'Start fine-tuning';
    $('#ft-episodes').disabled=busy||training;
    $('#ft-strategy').disabled=busy||training;
    $('#ft-speed').disabled=!enabled||!model||busy;
    $('#ft-playback-speed').disabled=!enabled||!model||busy||training;
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
    const signature=JSON.stringify(result&&[result.baseline,result.best,result.history,result.strategy,result.algorithm]);
    if(signature===renderedResult)return;renderedResult=signature;
    $('#ft-comparison').replaceChildren();$('#ft-history-rows').replaceChildren();
    $('#ft-results').hidden=!result?.baseline;
    $('#ft-history').hidden=!(result?.history?.length);
    $('#ft-exploration').hidden=true;$('#ft-exploration').textContent='';
    if(!result?.baseline)return;
    const history=Array.isArray(result.history)?result.history:[];
    const latestEvaluation=[...history].reverse().find(item=>item.evaluation)?.evaluation;
    [['Original clone',result.baseline],['Latest evaluated policy',latestEvaluation],['Best evaluated policy',result.best]].forEach(([label,value])=>{
      if(value)row($('#ft-comparison'),[label,number(value.return),number(value.fill_ml,4)+' mL',
        number(targetError(value),4)+' mL',
        Number.isFinite(value.fill_ml)?(Math.abs(value.fill_ml-700)<=5?'Yes':'No'):'—',number(value.spill_ml,4)+' mL',
        number(value.seconds,5)+' s',value.success?'Success':'Attempt']);
    });
    const delta=(result.best?.return??result.baseline.return)-result.baseline.return;
    const deltaText=delta>0&&delta<0.000001?delta.toExponential(6):number(delta);
    $('#ft-improvement').textContent=delta>0?'Best accuracy / speed score increased by '+deltaText+
      ' · checkpoint after iteration '+result.best.episode+'.':
      'No better checkpoint yet. The original cloned policy is retained.';
    $('#ft-history').hidden=!history.length;
    const latestUpdate=history.at(-1)?.update,exploration=latestUpdate?.exploration;
    if(Number.isFinite(exploration?.speed_min)&&Number.isFinite(exploration?.speed_max)) {
      $('#ft-exploration').hidden=false;
      const pair=value=>Array.isArray(value)&&value.length===2&&value.every(Number.isFinite);
      if(exploration.kind==='paired_phase_parameter'&&pair(latestUpdate.candidate_gains)) {
        const gains=latestUpdate.candidate_gains,center=latestUpdate.pair_center;
        const coordinate=latestUpdate.coordinate==='return'?'return speed':'approach/pour speed';
        $('#ft-exploration').textContent='Candidate: approach/pour '+number(100*gains[0],1)+
          '% · return '+number(100*gains[1],1)+'% of cloned speed · exploring '+coordinate+
          (pair(center)?' · pair center '+number(100*center[0],1)+'% / '+number(100*center[1],1)+'%':'')+'.';
      } else if(exploration.kind==='paired_parameter') {
        const candidate=Number.isFinite(latestUpdate.candidate_speed)?latestUpdate.candidate_speed:exploration.speed_min;
        $('#ft-exploration').textContent='Candidate: '+number(100*candidate,1)+'% of cloned speed'+
          (Number.isFinite(latestUpdate.pair_center)?' · pair centered at '+number(100*latestUpdate.pair_center,1)+'%':'')+'.';
      } else {
        $('#ft-exploration').textContent='Latest exploration · speed multipliers '+number(100*exploration.speed_min,1)+
          '–'+number(100*exploration.speed_max,1)+'% of the clone'+
          (Number.isInteger(exploration.slower_decisions)&&Number.isInteger(exploration.faster_decisions)?
            ' · '+exploration.slower_decisions+' slower / '+exploration.faster_decisions+
            ' faster adjustments than the policy being explored.':'.');
      }
    }
    const measurementRow=(iteration,label,value,update)=>{
      if(!Number.isFinite(value?.return))return;
      const outcome={success:'Success',time_limit:'Time limit',spill_or_overflow:'Spill / overflow'}[value.outcome]||
        (value.success?'Success':'Attempt');
      row($('#ft-history-rows'),[iteration,label,number(value.return),number(value.fill_ml,4),
        number(targetError(value),4),number(value.seconds,5),number(value.spill_ml,4),outcome,update]);
    };
    measurementRow(0,'Original clone · no noise',result.baseline,'—');
    const search=result.strategy==='policy_search'||result.algorithm==='bounded_paired_policy_search';
    history.forEach(item=>{
      const applied=item.update?.accepted??(item.update?.actor_change>0);
      measurementRow(item.episode,search?'Candidate policy · no action noise':'Exploratory trial · with noise',
        item.training,applied?'Applied':'No change');
      measurementRow(item.episode,search?'Retained policy · no noise':'Current policy · no noise',item.evaluation,'—');
    });
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
    const search=state.result?.strategy==='policy_search'||state.result?.algorithm==='bounded_paired_policy_search';
    const phase={baseline:'Checking original clone',training:search?'Testing a candidate speed':'Exploring nearby actions',evaluation:'Testing the current policy',complete:'Training complete'}[progress.phase]||'Ready';
    const training=state.training_active;
    $('#ft-showing').textContent=training?phase:rollout.active||rollout.done?
      (rollout.policy==='best'?'Best evaluated policy':'Original clone'):
      rollout.elapsed_seconds===0?'Starting pose':
      progress.phase==='complete'?'Last evaluated policy':state.has_result?'Stopped experiment':'Ready';
    $('#ft-time').textContent=number(training?progress.elapsed_seconds:rollout.elapsed_seconds,5)+' s';
    $('#ft-reward').textContent=number(training?progress.reward:rollout.reward);
    $('#ft-progress').textContent=state.has_result||training?
      'Completed '+(progress.completed_episodes||0)+' / '+(progress.episodes||0)+' learning iterations · '+
      (training?(state.playback_speed?state.playback_speed+'× requested':'fastest available')+' · fixed starting pose':'experiment stopped or completed'):'';
    const outcome={success:'Success',spill_or_overflow:'Too much spill or overflow',time_limit:'Time limit reached'}[rollout.outcome]||'Finished';
    if(training)status((state.paused?'Paused · ':'')+phase+(progress.episode?' · iteration '+progress.episode+' / '+progress.episodes:''));
    else if(rollout.done)status(outcome+' · Accuracy / speed score '+number(rollout.reward)+' · '+
      number(displayState.fill*1000,4)+' mL in the cup, '+number(displayState.spill*1000,4)+' mL spilled · '+
      (Math.abs(displayState.fill*1000-700)<=5?'within ±5 mL precision goal.':'outside ±5 mL precision goal.'));
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
    command={...command,speed:Number($(command.kind==='ft-run'?'#ft-playback-speed':'#ft-speed').value)};
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
  $('#ft-train').addEventListener('click',()=>startAction({kind:'ft-train',episodes:Number($('#ft-episodes').value),strategy:$('#ft-strategy').value}));
  $('#ft-strategy').addEventListener('change',()=>{
    $('#ft-method-description').textContent=$('#ft-strategy').value==='ppo'?
      'Actor–critic explores speed changes during a trial. A critic learns to predict future score and guides the policy update.':
      'Policy search tests paired faster/slower settings for approaching/pouring and returning the pot, keeping a candidate only when its score improves.';
  });
  $('#ft-run-base').addEventListener('click',()=>startAction({kind:'ft-run',policy:'base'}));
  $('#ft-run-best').addEventListener('click',()=>startAction({kind:'ft-run',policy:'best'}));
  $('#ft-pause').addEventListener('click',()=>{
    if(state?.paused)beforeRun();pauseRequested=!state?.paused;send({kind:'ft-pause',paused:pauseRequested});
  });
  $('#ft-stop').addEventListener('click',()=>{pendingAction=null;send({kind:'ft-stop'});});
  $('#ft-reset').addEventListener('click',()=>send({kind:'ft-reset'}));
  $('#ft-speed').addEventListener('change',()=>{
    if(worker&&!loading&&state?.model_loaded&&state.training_active)send({kind:'ft-speed',speed:Number($('#ft-speed').value)});
  });
  $('#ft-playback-speed').addEventListener('change',()=>{
    if(worker&&!loading&&state?.model_loaded&&!state.training_active&&state.rollout?.active)
      send({kind:'ft-speed',speed:Number($('#ft-playback-speed').value)});
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
