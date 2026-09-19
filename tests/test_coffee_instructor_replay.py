"""Private instructor libraries, literal text, and shared trajectory playback."""

import shutil
import subprocess

import pytest

from kaist_rl_lab.apps.coffee_instructor import INSTRUCTOR_JAVASCRIPT

NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
class Node {
  constructor(){this.children=[];this.listeners={};this.hidden=false;this.value='';this.textContent='';this.classList={toggle(){},remove(){},add(){}};}
  set innerHTML(value){throw Error('Untrusted text must never be parsed as HTML');}
  addEventListener(name,callback){this.listeners[name]=callback;}
  append(...children){this.children.push(...children.flatMap(child=>child.fragment?child.children:[child]));}
  replaceChildren(...children){this.children=[];this.append(...children);}
  setAttribute(key,value){this[key]=value;}
  removeAttribute(key){delete this[key];}
  getContext(){return {clearRect(){}};}
  querySelector(selector){return get(selector);}
  scrollIntoView(){}
}
const nodes=new Map(),requests=[],held=new Set(),pending=new Map(),animations=new Map(),feeds=new Map();
function streamFeed(){
  const queue=[];let waiter=null,aborted=false;
  const push=value=>{if(waiter){const next=waiter;waiter=null;next.resolve(value);}else queue.push(value);};
  return {push:event=>push({value:new TextEncoder().encode(JSON.stringify(event)+'\n'),done:false}),finish:()=>push({done:true}),
    connect(signal){signal.addEventListener('abort',()=>{aborted=true;if(waiter){waiter.reject(Error('Aborted'));waiter=null;}});},
    body:{getReader(){return {read(){if(aborted)return Promise.reject(Error('Aborted'));
      return queue.length?Promise.resolve(queue.shift()):new Promise((resolve,reject)=>{waiter={resolve,reject};});},releaseLock(){}};}}};
}
let nextAnimation=0,now=0,authenticated=false,availableSessions;
const get=selector=>{if(!nodes.has(selector))nodes.set(selector,new Node());return nodes.get(selector);};
get('#playback-speed').value='1';
const element={querySelector:get,querySelectorAll:()=>[]};
const document={querySelector:()=>element,createElement:()=>new Node(),
  createDocumentFragment:()=>Object.assign(new Node(),{fragment:true}),addEventListener(){},hidden:false};
const session={id:'test-class',name:'A class',join_url:'https://coffee.test/join?token=test',open:true,participant_required:true};
const otherSession={...session,id:'other-class',name:'B class'};
availableSessions=[session,otherSession];
const row={episode_id:'id/with?punctuation',participant:'<img src=x onerror=alert(1)>',
  received_at:'2026-09-18T00:00:00Z',steps:64,success:false,fill_ml:300,spill_ml:20,duration_seconds:2,total_reward:-2.25};
const studentRows=[row,{...row,episode_id:'zero',participant:'Zero reward',total_reward:0},
  {...row,episode_id:'missing',participant:'Missing reward',total_reward:null}];
const example={example_id:'example/with?punctuation',episode_id:'generated-id',label:'<img src=x onerror=alert(2)>',
  steps:64,success:true,fill_ml:700,spill_ml:0,duration_seconds:2,total_reward:27.625};
const exampleRows=[example,{...example,example_id:'zero',label:'Zero example',total_reward:0},
  {...example,example_id:'missing',label:'Missing example',total_reward:null}];
const frames=[0,1,2].map((time,index)=>({time,snapshot:{marker:index},cumulative_reward:[0,1.25,-2.25][index]}));
const replayData={frames,total_reward:-2.25,steps:64};
const streamEvents=data=>[
  {kind:'start',frame:data.frames[0],frame_count:data.frames.length,duration_seconds:data.frames.at(-1).time,total_reward:data.total_reward},
  {kind:'frames',frames:data.frames.slice(1)}, {kind:'complete'}];
const response=(status,data)=>({ok:status<400,status,json:async()=>data,
  body:data.frames?{getReader(){let index=0;const events=streamEvents(data);return {
    async read(){return index<events.length?{value:new TextEncoder().encode(JSON.stringify(events[index++])+'\n'),done:false}:{done:true};},releaseLock(){}};}}:null});
const fetch=async(path,options)=>{
  requests.push({path,options});
  assert.equal(options.credentials,'same-origin');
  if(path==='/api/instructor/login'){authenticated=true;return response(200,{authenticated:true});}
  if(path==='/api/instructor/logout'){authenticated=false;return response(200,{authenticated:false});}
  if(!authenticated)return response(401,{detail:'Sign in required'});
  if(feeds.has(path)){const feed=feeds.get(path);feed.connect(options.signal);return {ok:true,status:200,body:feed.body};}
  if(held.has(path))return new Promise(resolve=>{
    const queue=pending.get(path)||[];queue.push(resolve);pending.set(path,queue);
  });
  if(path==='/api/instructor/sessions')return response(200,availableSessions);
  if(path==='/api/instructor/examples')return response(200,exampleRows);
  if(path==='/api/instructor/submissions?session=test-class')return response(200,studentRows);
  if(path==='/api/instructor/submissions?session=other-class')return response(200,[]);
  if(/^\/api\/instructor\/(examples|submissions)\/[^/]+\/replay-stream$/.test(path))return response(200,replayData);
  throw Error('Unexpected request: '+path);
};
const context=vm.createContext({document,fetch,URL,location:{origin:'https://coffee.test'},
  window:{addEventListener(){},confirm:()=>true},navigator:{},setInterval(){},AbortController,TextDecoder,
  requestAnimationFrame(callback){const id=++nextAnimation;animations.set(id,callback);return id;},
  cancelAnimationFrame(id){animations.delete(id);},performance:{now:()=>now},console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const evaluate=code=>vm.runInContext(code,context);
// Physics and canvas geometry have separate tests; these tests exercise the UI's
// choice of recording, display of feedback, and playback/authentication state.
evaluate('normalizedSnapshot=snapshot=>snapshot;drawFrame=()=>{};');
const pump=()=>new Promise(resolve=>setImmediate(resolve));
const click=selector=>get(selector).listeners.click();
const replayPath=(item,source)=>'/api/instructor/'+source+'/'+encodeURIComponent(source==='examples'?item.example_id:item.episode_id)+'/replay-stream';
const resolveHeld=(path,data,status=200)=>{
  assert.ok(pending.get(path)?.length,'Expected a pending request for '+path);
  pending.get(path).shift()(response(status,data));
};
const findRow=(selector,label)=>get(selector).children.find(item=>item.children[0].textContent===label);
const replayButton=row=>row.children.at(-1).children[0];
async function login(){
  await pump();get('#password').value='test-only-password';
  await get('#login-form').listeners.submit({preventDefault(){},submitter:new Node()});
  await pump();
}
async function main(){
"""


def run_instructor(tmp_path, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the instructor browser controller check")
    source = tmp_path / "instructor.js"
    source.write_text(INSTRUCTOR_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(
        NODE_HARNESS
        + assertions
        + "\n}\nmain().then(()=>console.log('complete')).catch(error=>{console.error(error);process.exitCode=1;});"
    )
    result = subprocess.run(
        [node, str(runner), str(source)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "complete", "The controller test did not finish."


def test_generated_examples_available_without_a_class_and_rewards_remain_distinct(tmp_path):
    run_instructor(
        tmp_path,
        r"""
availableSessions=[];await login();
assert.equal(get('#examples-panel').hidden,false);
assert.equal(get('#submissions-panel').hidden,true);
assert.equal(get('#example-rows').children.length,3);
assert.equal(requests.filter(item=>item.path.startsWith('/api/instructor/submissions?')).length,0);
const values=label=>findRow('#example-rows',label).children.map(cell=>cell.textContent);
assert.ok(values(example.label).some(value=>value==='27.625'));
assert.ok(values('Zero example').some(value=>value==='0.000'));
assert.ok(values('Missing example').some(value=>value==='—'));
await replayButton(findRow('#example-rows',example.label)).listeners.click();
assert.equal(requests.at(-1).path,replayPath(example,'examples'));
assert.equal(get('#replay-panel').hidden,false);
assert.equal(get('#play-replay').disabled,false);
assert.match(get('#replay-summary').textContent,/Generated example/);
assert.ok(get('#replay-title').textContent.includes(example.label));
assert.equal(get('#replay-download').href,'/api/instructor/examples/example%2Fwith%3Fpunctuation/download');
assert.equal(evaluate('playing'),false,'Selecting a recording prepares it without starting playback');
""",
    )


def test_student_selection_rewards_and_shared_replay_controls(tmp_path):
    run_instructor(
        tmp_path,
        r"""
await login();
assert.equal(findRow('#submission-rows',row.participant).children[2].textContent,'-2.250');
assert.equal(findRow('#submission-rows','Zero reward').children[2].textContent,'0.000');
assert.equal(findRow('#submission-rows','Missing reward').children[2].textContent,'—');
await replayButton(findRow('#submission-rows',row.participant)).listeners.click();
assert.equal(requests.at(-1).path,replayPath(row,'submissions'));
assert.match(get('#replay-summary').textContent,/Student submission/);
assert.ok(get('#replay-title').textContent.includes(row.participant));
assert.ok(get('#replay-summary').textContent.includes('-2.250'));
assert.ok(get('#replay-reward').textContent.includes('0.000'));
assert.equal(get('#replay-position').max,'2');
click('#play-replay');assert.equal(evaluate('playing'),true);assert.equal(get('#play-replay').textContent,'Pause');
now=1000;evaluate('tickPlayback(1000)');assert.equal(get('#replay-position').value,'1');
assert.ok(get('#replay-reward').textContent.includes('1.250'));
click('#play-replay');assert.equal(evaluate('playing'),false);
get('#replay-position').listeners.input({target:{value:'2'}});
assert.equal(get('#replay-time').textContent,'2.0 / 2.0 s');
assert.ok(get('#replay-reward').textContent.includes('-2.250'));
click('#play-replay');assert.equal(get('#replay-position').value,'0');
get('#playback-speed').value='2';get('#playback-speed').listeners.change();
now=2000;evaluate('tickPlayback(2000)');assert.equal(evaluate('playing'),false);
assert.equal(get('#replay-position').value,'2');
assert.equal(get('#play-replay').textContent,'Replay');
await replayButton(findRow('#example-rows',example.label)).listeners.click();
assert.equal(get('#play-replay').textContent,'Play','A newly selected trajectory starts at its first frame');
assert.equal(get('#replay-position').value,'0');
""",
    )


def test_switching_trajectories_rejects_an_older_replay_response(tmp_path):
    run_instructor(
        tmp_path,
        r"""
await login();const path=replayPath(row,'submissions');held.add(path);
const stale=replayButton(findRow('#submission-rows',row.participant)).listeners.click();await pump();
await replayButton(findRow('#example-rows',example.label)).listeners.click();
assert.equal(get('#replay-position').max,'2');
resolveHeld(path,{frames:[...frames,...frames,...frames]});await stale;
assert.equal(get('#replay-position').max,'2');
assert.ok(get('#replay-title').textContent.includes(example.label));
assert.equal(get('#replay-download').href,'/api/instructor/examples/example%2Fwith%3Fpunctuation/download');
""",
    )


def test_class_change_invalidates_pending_student_list_and_replay(tmp_path):
    run_instructor(
        tmp_path,
        r"""
await login();const listPath='/api/instructor/submissions?session=test-class';
const recordingPath=replayPath(row,'submissions');held.add(listPath);held.add(recordingPath);
const staleList=evaluate('loadSubmissions()');
const staleReplay=replayButton(findRow('#submission-rows',row.participant)).listeners.click();await pump();
get('#session-select').value='other-class';await get('#session-select').listeners.change();
resolveHeld(listPath,studentRows);resolveHeld(recordingPath,replayData);await Promise.all([staleList,staleReplay]);
assert.equal(get('#submission-rows').children.length,0);assert.equal(get('#replay-panel').hidden,true);
assert.equal(evaluate('replayFrames.length'),0);assert.equal(evaluate('playing'),false);
assert.equal(get('#example-rows').children.length,3,'Generated examples are independent of class selection');
""",
    )


def test_logout_invalidates_generated_list_and_generated_replay(tmp_path):
    run_instructor(
        tmp_path,
        r"""
await login();const listPath='/api/instructor/examples',recordingPath=replayPath(example,'examples');
held.add(listPath);held.add(recordingPath);
const staleList=click('#refresh-examples');
const staleReplay=replayButton(findRow('#example-rows',example.label)).listeners.click();await pump();
await click('#logout');resolveHeld(listPath,exampleRows);resolveHeld(recordingPath,replayData);
await Promise.all([staleList,staleReplay]);
assert.equal(get('#dashboard').hidden,true);assert.equal(get('#example-rows').children.length,0);
assert.equal(get('#replay-panel').hidden,true);assert.equal(evaluate('replayFrames.length'),0);
assert.equal(get('#replay-download').href,undefined);assert.equal(evaluate('playing'),false);
""",
    )


def test_stream_allows_playback_before_completion_and_buffers_without_skipping(tmp_path):
    run_instructor(tmp_path, r"""
await login();const path=replayPath(row,'submissions'),feed=streamFeed();feeds.set(path,feed);
const loading=replayButton(findRow('#submission-rows',row.participant)).listeners.click();await pump();
assert.equal(get('#play-replay').disabled,true);
feed.push(streamEvents(replayData)[0]);await pump();
assert.equal(get('#play-replay').disabled,false,'First frame is usable before the rest is generated');
assert.equal(evaluate('replayFrames.length'),1);assert.equal(evaluate('replayComplete'),false);
assert.equal(get('#replay-time').textContent,'0.0 / 2.0 s');
click('#play-replay');now=500;evaluate('tickPlayback(500)');
assert.equal(evaluate('playing'),true);assert.equal(get('#replay-position').value,'0');
feed.push({kind:'frames',frames:[frames[1]]});await pump();
now=1500;evaluate('tickPlayback(1500)');
assert.equal(get('#replay-position').value,'1');assert.equal(evaluate('playing'),true);
feed.push({kind:'frames',frames:[frames[2]]});feed.push({kind:'complete'});feed.finish();await loading;
now=2500;evaluate('tickPlayback(2500)');
assert.equal(get('#replay-position').value,'2');assert.equal(evaluate('playing'),false);
assert.equal(get('#play-replay').textContent,'Replay');
""")


def test_closing_partial_stream_aborts_the_request_and_rejects_late_frames(tmp_path):
    run_instructor(tmp_path, r"""
await login();const path=replayPath(example,'examples'),feed=streamFeed();feeds.set(path,feed);
const loading=replayButton(findRow('#example-rows',example.label)).listeners.click();await pump();
feed.push(streamEvents(replayData)[0]);await pump();
const signal=requests.at(-1).options.signal;
click('#close-replay');await loading;
assert.equal(signal.aborted,true);assert.equal(get('#replay-panel').hidden,true);
assert.equal(evaluate('playing'),false);
feed.push({kind:'frames',frames:[frames[1]]});await pump();
assert.equal(evaluate('replayFrames.length'),1);
""")


def test_stream_error_stops_and_disables_partial_replay(tmp_path):
    run_instructor(tmp_path, r"""
await login();const path=replayPath(row,'submissions'),feed=streamFeed();feeds.set(path,feed);
const loading=replayButton(findRow('#submission-rows',row.participant)).listeners.click();await pump();
feed.push(streamEvents(replayData)[0]);await pump();click('#play-replay');
feed.push({kind:'error',detail:'The recording does not match replayed physics.'});await loading;
assert.equal(evaluate('playing'),false);assert.equal(evaluate('replayFrames.length'),0);
assert.equal(get('#play-replay').disabled,true);
assert.match(get('#replay-status').textContent,/physics/);
assert.equal(requests.at(-1).options.signal.aborted,true);
""")
