"""Safety checks for the browser-only instructor interface."""

import json
import shutil
import subprocess

import pytest

from kaist_rl_lab.apps.coffee_instructor import INSTRUCTOR_JAVASCRIPT


def test_instructor_keeps_private_data_out_after_logout_and_renders_literal_text(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the instructor browser controller check")
    source = tmp_path / "instructor.js"
    source.write_text(INSTRUCTOR_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
class Node {
  constructor(){this.children=[];this.listeners={};this.hidden=false;this.value='';this.textContent='';this.classList={toggle(){},remove(){}};}
  set innerHTML(value){throw Error('Untrusted text must never be parsed as HTML');}
  addEventListener(name,callback){this.listeners[name]=callback;}
  append(...children){this.children.push(...children.flatMap(child=>child.fragment?child.children:[child]));}
  replaceChildren(...children){this.children=[];this.append(...children);}
  setAttribute(key,value){this[key]=value;}
  removeAttribute(key){delete this[key];}
  getContext(){return {clearRect(){}};}
  querySelector(selector){return get(selector);}
}
const nodes=new Map();
const get=selector=>{if(!nodes.has(selector))nodes.set(selector,new Node());return nodes.get(selector);};
const element={querySelector:get,querySelectorAll:()=>[]};
const document={querySelector:()=>element,createElement:()=>new Node(),
  createDocumentFragment:()=>Object.assign(new Node(),{fragment:true}),addEventListener(){},hidden:false};
const session={id:'test-class',name:'A class',join_url:'https://coffee.test/join?token=test',open:true,participant_required:true};
const row={episode_id:'id/with?punctuation',participant:'<img src=x onerror=alert(1)>',
  received_at:'2026-09-18T00:00:00Z',steps:64,success:false,fill_ml:300,spill_ml:20,duration_seconds:2};
let authenticated=false,pendingResolve=null,holdSessions=false;
const response=(status,data)=>Promise.resolve({ok:status<400,status,json:async()=>data});
const fetch=async(path,options)=>{
  if(path==='/api/instructor/login'){authenticated=true;return response(200,{authenticated:true});}
  if(path==='/api/instructor/logout'){authenticated=false;return response(200,{authenticated:false});}
  if(!authenticated)return response(401,{detail:'Sign in required'});
  if(path==='/api/instructor/sessions'){
    if(holdSessions)return new Promise(resolve=>{pendingResolve=resolve;});
    return response(200,[session]);
  }
  if(path==='/api/instructor/examples')return response(200,[]);
  if(path.startsWith('/api/instructor/submissions?'))return response(200,[row]);
  throw Error('Unexpected request: '+path);
};
const context=vm.createContext({document,fetch,URL,location:{origin:'https://coffee.test'},
  window:{addEventListener(){},confirm:()=>true},navigator:{},setInterval(){},setTimeout,clearTimeout,AbortController,
  requestAnimationFrame(){return 1;},cancelAnimationFrame(){},performance:{now:()=>0},console});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const evaluate=code=>vm.runInContext(code,context);
const pump=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
  await pump();assert.equal(get('#login-panel').hidden,false);assert.equal(get('#dashboard').hidden,true);
  get('#password').value='test-only-password';
  await get('#login-form').listeners.submit({preventDefault(){},submitter:new Node()});
  assert.equal(get('#dashboard').hidden,false);assert.equal(get('#password').value,'');
  const rows=get('#submission-rows').children;assert.equal(rows.length,1);
  assert.equal(rows[0].children[0].textContent,row.participant);
  assert.equal(rows[0].children[6].children[1].href,
    '/api/instructor/submissions/id%2Fwith%3Fpunctuation/download');
  assert.equal(evaluate("safeJoinURL('https://outside.test/phishing')"),'');
  assert.equal(evaluate("safeJoinURL('javascript:alert(1)')"),'');
  // An in-flight refresh must not restore private content after sign-out.
  holdSessions=true;const stale=evaluate('loadSessions()');await pump();
  await get('#logout').listeners.click();
  pendingResolve(await response(200,[session]));await stale;
  assert.equal(get('#dashboard').hidden,true);assert.equal(get('#login-panel').hidden,false);
  assert.equal(get('#submission-rows').children.length,0);assert.equal(get('#join-url').value,'');
  assert.equal(get('#open-demo').href,undefined);assert.equal(get('#class-qr').src,undefined);
  console.log(JSON.stringify({passed:true}));
})().catch(error=>{console.error(error);process.exitCode=1;});
""")
    result = subprocess.run(
        [node, str(runner), str(source)], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == {"passed": True}
