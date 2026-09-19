"""A dependency-free view of the instructor's actual fine-tuning results."""

LEARNING_VIZ_HTML = """
        <section id="ft-learning-viz" class="ft-learning-viz" aria-labelledby="ft-learning-title" hidden>
          <div class="ft-learning-heading"><h3 id="ft-learning-title">Watch learning unfold</h3>
            <span class="hint">Higher score is better</span></div>
          <ol class="ft-learning-flow" aria-label="Learning cycle">
            <li id="ft-stage-baseline"><span>1</span>Check clone</li>
            <li id="ft-stage-training"><span>2</span>Explore</li>
            <li id="ft-stage-update"><span>3</span>Learn from reward</li>
            <li id="ft-stage-evaluation"><span>4</span>Evaluate</li>
          </ol>
          <p id="ft-learning-status" class="ft-learning-status" role="status" aria-live="polite"></p>
          <p id="ft-learning-updates" class="hint ft-learning-updates"></p>
          <div class="ft-chart-legend" aria-label="Chart legend">
            <label class="ft-chart-candidates-label"><input id="ft-chart-candidates" type="checkbox">
              <i class="ft-legend-exploration"></i>Show candidate / exploration scores</label>
            <span><i class="ft-legend-evaluation"></i>Current policy · no noise</span>
            <span><i class="ft-legend-best"></i>Best so far</span>
            <span><i class="ft-legend-baseline"></i>Original clone</span>
          </div>
          <p id="ft-candidate-summary" class="hint ft-candidate-summary"></p>
          <svg id="ft-learning-chart" viewBox="0 0 680 255" role="group"
            aria-labelledby="ft-chart-title ft-chart-description"></svg>
          <div class="ft-chart-controls"><label class="ft-chart-zero-label">
            <input id="ft-chart-zero" type="checkbox">Include zero on score axis</label>
            <span class="hint">Axis fits the visible series.</span></div>
          <div class="ft-chart-inspect-heading"><label for="ft-chart-trial">Inspect completed results</label>
            <select id="ft-chart-trial" disabled><option value="">No results yet</option></select></div>
          <div id="ft-chart-inspector" class="ft-chart-inspector" aria-live="polite"></div>
          <p class="hint ft-chart-explanation">After every policy update, evaluation measures the current policy from the
            fixed starting pose without exploration noise. This curve may rise or fall; best so far keeps the strongest
            evaluated checkpoint. Scores strongly reward 700 mL accuracy, especially within ±5 mL, while also valuing speed. Points appear after each rollout finishes.</p>
        </section>
"""

LEARNING_VIZ_CSS = """
.ft-learning-viz {min-width:0;padding:16px;background:#f8fafb;border:1px solid #d7e1e6;border-radius:12px;}
.ft-learning-heading {display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap;}
.ft-learning-heading h3 {font-size:17px;margin:0;}
.ft-learning-heading>.hint {font-size:12px;}
.ft-learning-flow {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px;list-style:none;padding:0;margin:13px 0 8px;}
.ft-learning-flow li {font-size:11px;padding:7px 4px;border:1px solid #dbe4e9;border-radius:6px;color:#536574;text-align:center;background:#fff;}
.ft-learning-flow li span {font-weight:800;margin-right:4px;}
.ft-learning-flow li[data-active="true"] {background:#e4f1f4;border-color:#27869b;color:#103f4b;font-weight:700;}
.ft-learning-flow li[data-complete="true"] span {color:#147653;}
.ft-learning-status {font-weight:700;font-size:13px;margin:9px 0 4px;color:#17394f;min-height:19px;}
.ft-learning-updates {font-size:12px;margin:0 0 11px;}
.ft-chart-legend {display:flex;gap:7px 13px;flex-wrap:wrap;font-size:11px;color:#3c5262;}
.ft-chart-legend span,.ft-chart-candidates-label {display:inline-flex;align-items:center;gap:5px;}
.ft-chart-candidates-label {margin:0!important;font-size:inherit!important;font-weight:400!important;}
.ft-chart-candidates-label input {width:16px!important;height:16px;min-height:16px;margin:0;accent-color:#ac6400;}
.ft-candidate-summary {font-size:11px;line-height:1.45;margin:7px 0 0;}
.ft-chart-legend i {width:20px;height:0;border-top:2px solid;display:inline-block;}
.ft-chart-legend .ft-legend-exploration {border-color:#ac6400;border-top-style:dashed;}
.ft-chart-legend .ft-legend-evaluation {border-color:#153b57;}
.ft-chart-legend .ft-legend-best {border-color:#12815c;border-top-width:3px;}
.ft-chart-legend .ft-legend-baseline {border-color:#7c8995;border-top-style:dashed;}
#ft-learning-chart {display:block;width:100%;height:auto;overflow:visible;margin-top:5px;}
#ft-learning-chart text {font-family:inherit;fill:#435767;font-size:12px;}
#ft-learning-chart .ft-chart-grid {stroke:#e0e7ec;stroke-width:1;}
#ft-learning-chart .ft-chart-axis {stroke:#a2b2bd;stroke-width:1;}
#ft-learning-chart .ft-chart-baseline {stroke:#7c8995;stroke-width:1.5;stroke-dasharray:6 5;}
#ft-learning-chart .ft-chart-series-exploration {stroke:#ac6400;stroke-dasharray:5 4;stroke-width:2;fill:none;}
#ft-learning-chart .ft-chart-series-evaluation {stroke:#153b57;stroke-width:2;fill:none;}
#ft-learning-chart .ft-chart-series-best {stroke:#12815c;stroke-width:3;fill:none;}
#ft-learning-chart .ft-chart-point {cursor:pointer;stroke-width:2;}
#ft-learning-chart .ft-chart-point:focus {outline:none;stroke:#0077b6;stroke-width:4;}
.ft-chart-controls {display:flex;align-items:center;justify-content:space-between;gap:5px;flex-wrap:wrap;margin:0 0 12px;}
.ft-chart-controls .hint {font-size:11px;}
.ft-chart-zero-label {display:flex!important;align-items:center;gap:6px;font-size:12px!important;margin:0!important;font-weight:400!important;}
.ft-chart-zero-label input {width:16px!important;height:16px;min-height:16px;margin:0;accent-color:#17394f;}
.ft-chart-inspect-heading {display:flex;align-items:center;justify-content:space-between;gap:8px;}
.ft-chart-inspect-heading label {font-size:12px;margin:0;}
.ft-chart-inspect-heading select {font-size:12px;max-width:190px;width:auto;padding:6px 9px;min-height:44px;}
.ft-chart-inspector {font-size:12px;min-height:63px;margin-top:7px;padding:9px 10px;background:#fff;border:1px solid #e0e7ec;border-radius:7px;}
.ft-chart-inspector p {margin:0;line-height:1.55;}
.ft-chart-inspector .ft-inspector-values {font-variant-numeric:tabular-nums;font-weight:700;color:#16394f;}
.ft-chart-inspector .ft-inspector-update {color:#526675;margin-top:3px;}
.ft-chart-explanation {font-size:11px;margin:8px 0 0;}
@media(max-width:580px) {.ft-learning-viz {padding:12px;}.ft-learning-flow li {font-size:10px;padding:7px 2px;}.ft-learning-flow li span {margin-right:2px;}.ft-chart-legend {gap:6px 9px;font-size:10px;}}
"""

LEARNING_VIZ_JAVASCRIPT = r"""
function createLearningVisualization(element) {
  const root=element.querySelector('#ft-learning-viz')||element;
  const $=selector=>root.querySelector(selector);
  const svg=$('#ft-learning-chart'),select=$('#ft-chart-trial'),zero=$('#ft-chart-zero'),candidates=$('#ft-chart-candidates');
  const NS='http://www.w3.org/2000/svg';
  const finite=value=>typeof value==='number'&&Number.isFinite(value);
  const reward=metrics=>finite(metrics?.return)?metrics.return:null;
  const number=value=>finite(value)?value.toFixed(3):'—';
  const changed=update=>update?.accepted===true||(update?.accepted!==false&&finite(update?.actor_change)&&update.actor_change>0);
  let state=null,selected=null,followLatest=true,lastSignature='';
  function text(node,value) {if(node.textContent!==value)node.textContent=value;}
  function svgNode(tag,attributes={},content=null) {
    const node=document.createElementNS(NS,tag);
    Object.entries(attributes).forEach(([name,value])=>node.setAttribute(name,String(value)));
    if(content!==null)node.textContent=content;
    svg.append(node);return node;
  }
  function rows(result) {
    const byTrial=new Map();
    for(const row of Array.isArray(result?.history)?result.history:[]) {
      if(!Number.isInteger(row?.episode)||row.episode<1)continue;
      if(reward(row.training)===null&&reward(row.evaluation)===null)continue;
      byTrial.set(row.episode,row);
    }
    return [...byTrial.values()].sort((a,b)=>a.episode-b.episode);
  }
  function phaseView(current,result,history) {
    const progress=current.progress||{},phase=progress.phase||result.phase;
    const count=history.filter(row=>reward(row.evaluation)!==null).length;
    const applied=history.filter(row=>changed(row.update)).length;
    const attempted=history.filter(row=>row.update&&typeof row.update==='object').length;
    const episode=Number.isInteger(progress.episode)?progress.episode:result.episode;
    const total=Number.isInteger(progress.episodes)?progress.episodes:result.episodes;
    const search=result.strategy==='policy_search'||result.algorithm==='bounded_paired_policy_search';
    const phaseLabel={baseline:'Checking the original clone',training:search?'Testing a candidate speed':'Exploring nearby actions',evaluation:'Evaluating without noise'}[phase]||'Experiment ready';
    let label;
    if(current.training_active)label=(current.training_paused||current.paused?'Paused · ':'')+phaseLabel+
      (phase!=='baseline'&&Number.isInteger(episode)?' · iteration '+episode+(Number.isInteger(total)?' / '+total:''):'');
    else if(result.done||phase==='complete')label='Experiment complete · '+count+' iteration'+(count===1?'':'s')+' evaluated';
    else label='Experiment stopped · '+count+' iteration'+(count===1?'':'s')+' evaluated';
    text($('#ft-learning-status'),label);
    text($('#ft-learning-updates'),'Policy updates: '+applied+' applied / '+attempted+' attempted · best evaluated policy retained');
    for(const key of ['baseline','training','update','evaluation']) {
      const node=$('#ft-stage-'+key);
      node.setAttribute('data-active',String(!!current.training_active&&phase===key));
      node.setAttribute('data-complete',String(key==='baseline'?reward(result.baseline)!==null:
        key==='training'?history.length>0:key==='update'?attempted>0:count>0));
      if(current.training_active&&phase===key)node.setAttribute('aria-current','step');
      else node.removeAttribute('aria-current');
    }
  }
  function inspect(result,history) {
    const holder=$('#ft-chart-inspector');holder.replaceChildren();
    const values=document.createElement('p');values.className='ft-inspector-values';
    const update=document.createElement('p');update.className='ft-inspector-update';
    const baseline=reward(result.baseline);
    if(selected===null) {
      values.textContent='Waiting for the first completed rollout.';
      update.textContent='Live movement is shown beside this chart.';
    } else if(selected===0) {
      values.textContent='Original clone · Accuracy / speed score '+number(baseline);
      update.textContent='Before fine-tuning · tested without exploration noise.';
    } else {
      const item=history.find(row=>row.episode===selected);
      let best=baseline;
      for(const row of history)if(row.episode<=selected&&reward(row.evaluation)!==null)
        best=best===null?reward(row.evaluation):Math.max(best,reward(row.evaluation));
      values.textContent='Accuracy / speed scores · Candidate / exploration '+number(reward(item?.training))+
        ' · Current policy (no noise) '+number(reward(item?.evaluation))+' · Best '+number(best);
      update.textContent=(item?.update?(changed(item.update)?'Policy update applied. ':'Policy unchanged. '):'')+
        (reward(item?.evaluation)===null?'Evaluation has not finished.':
          baseline!==null&&reward(item.evaluation)>baseline?'Evaluation exceeds the original clone.':'Original clone remains part of the comparison.');
    }
    holder.append(values,update);
  }
  function choose(trial) {selected=trial;followLatest=false;lastSignature='';render(state);}
  function plot(result,history,width) {
    const baseline=reward(result.baseline);
    const trials=(baseline!==null?[0]:[]).concat(history.map(row=>row.episode));
    if(followLatest||!trials.includes(selected))selected=trials.length?trials[trials.length-1]:null;
    select.replaceChildren();
    for(const trial of trials) {
      const option=document.createElement('option');option.value=String(trial);
      option.textContent=trial===0?'Original clone':'Iteration '+trial;
      select.append(option);
    }
    if(!trials.length) {
      const option=document.createElement('option');option.value='';option.textContent='No results yet';select.append(option);
    }
    select.disabled=!trials.length;select.value=selected===null?'':String(selected);
    inspect(result,history);
    const exploration=history.filter(row=>reward(row.training)!==null).map(row=>({trial:row.episode,value:reward(row.training)}));
    const evaluation=history.filter(row=>reward(row.evaluation)!==null).map(row=>({trial:row.episode,value:reward(row.evaluation)}));
    const best=[];let bestReward=baseline;
    if(baseline!==null)best.push({trial:0,value:baseline});
    for(const point of evaluation) {
      bestReward=bestReward===null?point.value:Math.max(bestReward,point.value);
      best.push({trial:point.trial,value:bestReward});
    }
    const shownExploration=candidates.checked?exploration:[];
    const candidateRows=history.filter(row=>reward(row.training)!==null);
    const knownOutcomes=candidateRows.filter(row=>typeof row.training.success==='boolean');
    const failures=knownOutcomes.filter(row=>!row.training.success).length;
    text($('#ft-candidate-summary'),'Candidate / exploration trials: '+candidateRows.length+' completed'+
      (knownOutcomes.length?' · '+failures+' unsuccessful':'')+' · '+
      (candidates.checked?'shown on chart.':'not plotted; available in the inspector and history.'));
    const observed=[...shownExploration,...evaluation,...best].map(point=>point.value);
    let low=observed.length?Math.min(...observed):0,high=observed.length?Math.max(...observed):1;
    if(zero.checked){low=Math.min(low,0);high=Math.max(high,0);}
    const padding=high===low?Math.max(.02,Math.abs(high)*.02):(high-low)*.12;
    low-=padding;high+=padding;
    const total=Number.isInteger(result.episodes)&&result.episodes>0?result.episodes:1;
    const maxTrial=Math.max(total,...trials,1),left=62,right=width-18,top=18,bottom=211;
    const x=trial=>left+trial/maxTrial*(right-left),y=value=>bottom-(value-low)/(high-low)*(bottom-top);
    svg.replaceChildren();svg.setAttribute('viewBox','0 0 '+width+' 255');
    svg.setAttribute('data-y-min',String(low));svg.setAttribute('data-y-max',String(high));
    svg.setAttribute('data-x-max',String(maxTrial));
    svgNode('title',{id:'ft-chart-title'},'Accuracy / speed score by learning iteration');
    svgNode('desc',{id:'ft-chart-description'},'Completed candidate or exploratory Accuracy / speed scores, the current policy evaluated without noise after each update, the original clone, and the best evaluated score so far. Select a point or use the iteration menu to inspect results.');
    const digits=high-low<.05?4:high-low<1?3:high-low<10?2:1;
    for(let index=0;index<=4;index++) {
      const value=low+(high-low)*index/4,py=y(value);
      svgNode('line',{x1:left,y1:py,x2:right,y2:py,class:'ft-chart-grid'});
      svgNode('text',{x:left-8,y:py+4,'text-anchor':'end','data-axis':'reward','data-value':value},value.toFixed(digits));
    }
    svgNode('line',{x1:left,y1:bottom,x2:right,y2:bottom,class:'ft-chart-axis'});
    const tickLimit=width<440?6:12;
    const tickStep=[1,2,5,10,20,25,50,100].find(value=>value>=maxTrial/tickLimit)||Math.ceil(maxTrial/tickLimit);
    for(let trial=0;trial<=maxTrial;trial++) {
      if(trial!==maxTrial&&trial%tickStep!==0)continue;
      if(trial!==maxTrial&&maxTrial-trial<tickStep/2)continue;
      svgNode('text',{x:x(trial),y:bottom+18,'text-anchor':'middle','data-axis':'trial'},String(trial));
    }
    svgNode('text',{x:left,y:12},'Accuracy / speed score');
    svgNode('text',{x:(left+right)/2,y:251,'text-anchor':'middle'},width<430?'Iteration (0 = clone)':'Learning iteration (0 = original clone)');
    if(baseline!==null)svgNode('line',{x1:left,y1:y(baseline),x2:right,y2:y(baseline),class:'ft-chart-baseline','data-value':baseline});
    const path=(points,step=false)=>points.map((point,index)=>index===0?'M '+x(point.trial)+' '+y(point.value):
      step?'H '+x(point.trial)+' V '+y(point.value):'L '+x(point.trial)+' '+y(point.value)).join(' ');
    for(const [series,points] of [['exploration',shownExploration],['evaluation',evaluation],['best',best]]) {
      if(points.length)svgNode('path',{d:path(points,series==='best'),class:'ft-chart-series-'+series,'data-series':series});
    }
    const pointSets=[['exploration',shownExploration],['evaluation',evaluation]];
    if(baseline!==null)pointSets.push(['baseline',[{trial:0,value:baseline}]]);
    for(const [series,points] of pointSets)for(const point of points) {
      const active=point.trial===selected;
      const node=svgNode('circle',{cx:x(point.trial),cy:y(point.value),r:active?6:4,
        class:'ft-chart-point','data-trial':point.trial,'data-series':series,'data-value':point.value,
        fill:series==='exploration'?'#fff':'#153b57',stroke:series==='exploration'?'#ac6400':'#153b57',
        tabindex:'0',role:'button','aria-pressed':String(active),
        'aria-label':(point.trial===0?'Original clone':'Iteration '+point.trial+', '+
          (series==='evaluation'?'current policy without noise':series))+' Accuracy / speed score '+number(point.value)});
      node.addEventListener('click',()=>choose(point.trial));
      node.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose(point.trial);select.focus();}});
    }
    if(!observed.length) {
      svgNode('text',{x:(left+right)/2,y:115,'text-anchor':'middle'},'Waiting for the original clone');
      svgNode('text',{x:(left+right)/2,y:133,'text-anchor':'middle'},'to finish its evaluation.');
    }
  }
  function render(current) {
    state=current;
    if(!current?.result){root.hidden=true;return;}
    root.hidden=false;
    const result=current.result,history=rows(result);
    phaseView(current,result,history);
    const width=Math.max(300,Math.min(900,svg.clientWidth||680));
    const signature=JSON.stringify([result.baseline,history,result.episodes,result.strategy,result.algorithm,zero.checked,candidates.checked,selected,followLatest,width]);
    if(signature===lastSignature)return;
    plot(result,history,width);
    lastSignature=JSON.stringify([result.baseline,history,result.episodes,result.strategy,result.algorithm,zero.checked,candidates.checked,selected,followLatest,width]);
  }
  function reset() {
    state=null;selected=null;followLatest=true;lastSignature='';zero.checked=false;candidates.checked=false;
    root.hidden=true;svg.replaceChildren();select.replaceChildren();select.disabled=true;
    const option=document.createElement('option');option.value='';option.textContent='No results yet';select.append(option);
    $('#ft-chart-inspector').replaceChildren();text($('#ft-learning-status'),'');text($('#ft-learning-updates'),'');text($('#ft-candidate-summary'),'');
  }
  select.addEventListener('change',()=>{const trial=Number(select.value);if(select.value!==''&&Number.isInteger(trial))choose(trial);});
  zero.addEventListener('change',()=>{lastSignature='';render(state);});
  candidates.addEventListener('change',()=>{lastSignature='';render(state);});
  const resize=()=>{if(state?.result)render(state);};
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(svg);
  else if(typeof window!=='undefined'&&window.addEventListener)window.addEventListener('resize',resize);
  return {render,reset};
}
"""
