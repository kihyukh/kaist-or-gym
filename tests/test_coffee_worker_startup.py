"""Start the real worker bootstrap with controlled download and VM boundaries."""

import json
import shutil
import subprocess

import pytest

from kaist_rl_lab.apps.coffee_browser import WORKER_JAVASCRIPT

NODE_HARNESS = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const messages=[],requests=[],configs=[],actions=[];
const clone=value=>JSON.parse(JSON.stringify(value));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject};};
const moduleRequest=deferred(),lockRequest=deferred(),sourceRequest=deferred(),runtimeRequest=deferred();
const lock={info:{version:'0.29.3'},packages:{numpy:{version:'2.2.5'},cloudpickle:{version:'3.1.1'},'typing-extensions':{version:'4.15.0'}}};
let command=null;
const fakePython={
  globals:{set(name,value){assert.equal(name,'_coffee_command');command=JSON.parse(value);}},
  unpackArchive(bytes,format,options){actions.push(['unpack',Array.from(bytes),format,clone(options)]);},
  runPython(code){
    actions.push(['python',code]);
    if(code==='coffee_runtime.dispatch(_coffee_command)')return JSON.stringify({snapshot:{playback:{paused:true,running:true}},command});
  },
  loadPackage(){throw Error('Packages should download during bootstrap, not afterward');},
  runPythonAsync(){throw Error('Worker startup must not run a package installer');},
};
const pyodideModule={loadPyodide:options=>{configs.push(options);return runtimeRequest.promise;}};
const self={};
const context=vm.createContext({self,console,
  mockImport(){return moduleRequest.promise;},
  fetch(url){requests.push(url);return url.endsWith('pyodide-lock.json')?lockRequest.promise:sourceRequest.promise;},
  postMessage:value=>messages.push(clone(value)),
  atob:value=>Buffer.from(value,'base64').toString('binary'),
  performance:{now:()=>0},
  setTimeout(){throw Error('An idle initial snapshot must not start simulation ticks');},clearTimeout(){},
});
const source=fs.readFileSync(process.argv[2],'utf8').replace("import(PYODIDE_BASE+'pyodide.mjs')","mockImport()");
vm.runInContext(source,context);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function releaseLibraries(){
  moduleRequest.resolve(pyodideModule);
  lockRequest.resolve({ok:true,json:async()=>clone(lock)});
}
const releaseSource=()=>sourceRequest.resolve({ok:true,arrayBuffer:async()=>new Uint8Array([1,2,3]).buffer});
async function main(){
"""


def run_startup(tmp_path, assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for worker startup regression tests")
    worker = tmp_path / "coffee-worker.js"
    worker.write_text(WORKER_JAVASCRIPT)
    runner = tmp_path / "check.cjs"
    runner.write_text(NODE_HARNESS + assertions + "\n}\nmain().catch(error=>{console.error(error);process.exit(1)});")
    result = subprocess.run(
        [node, str(runner), str(worker)], text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode,runtime", [
    ("student", "BrowserRuntime"), ("random", "RandomAgentRuntime"),
    ("cloning", "CloningAgentRuntime"), ("finetuning", "FineTuningRuntime"),
])
def test_packages_and_source_download_during_vm_bootstrap(tmp_path, mode, runtime):
    run_startup(tmp_path, r"""
const task=self.onmessage({data:{kind:'init',mode:MODE,bundle_url:'https://coffee.test/coffee-bundle.zip'}});
await flush();
assert.deepEqual(requests,[
  'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide-lock.json',
  'https://coffee.test/coffee-bundle.zip']);
assert.equal(configs.length,0);assert.equal(actions.length,0);
releaseLibraries();await flush();assert.equal(configs.length,1);
const options=clone(configs[0]);
assert.deepEqual(options.packages,['numpy','gymnasium']);
assert.equal(options.indexURL,'https://cdn.jsdelivr.net/pyodide/v0.29.3/full/');
assert.equal(options.packageBaseUrl,options.indexURL);
assert.equal(options.lockFileContents.packages.numpy.version,'2.2.5');
const gym=options.lockFileContents.packages.gymnasium;
assert.equal(gym.version,'1.2.3');assert.equal(gym.install_dir,'site');
assert.deepEqual(gym.depends,['numpy','cloudpickle','typing-extensions','farama-notifications']);
for(const name of ['gymnasium','farama-notifications']){
  const package=options.lockFileContents.packages[name];
  assert.match(package.file_name,/^https:\/\/files\.pythonhosted\.org\/packages\/.*\.whl$/);
  assert.match(package.sha256,/^[a-f0-9]{64}$/);
}
runtimeRequest.resolve(fakePython);await flush();
assert.equal(actions.length,0,'Do not import Python code before its source is downloaded');
releaseSource();await task;
assert.deepEqual(actions[0],['unpack',[1,2,3],'zip',{extractDir:'/home/pyodide'}]);
assert.match(actions[1][1],new RegExp('coffee_runtime = '+RUNTIME+'\\(\\)'));
assert.deepEqual(messages.at(-1).command,{kind:'snapshot'});
assert.equal(messages.some(message=>message.error),false);
""".replace("MODE", json.dumps(mode)).replace("RUNTIME", json.dumps(runtime)))


def test_embedded_notebook_bundle_needs_no_source_network_request(tmp_path):
    run_startup(tmp_path, r"""
const task=self.onmessage({data:{kind:'init',bundle:'AQID'}});
releaseLibraries();runtimeRequest.resolve(fakePython);await task;
assert.equal(requests.length,1);assert.match(requests[0],/pyodide-lock.json$/);
assert.deepEqual(actions[0][1],[1,2,3]);
assert.deepEqual(messages.at(-1).command,{kind:'snapshot'});
""")


@pytest.mark.parametrize("failure", ["lock", "source", "runtime"])
def test_bootstrap_failure_does_not_emit_a_ready_snapshot(tmp_path, failure):
    run_startup(tmp_path, r"""
const task=self.onmessage({data:{kind:'init',bundle_url:'https://coffee.test/coffee-bundle.zip'}});
moduleRequest.resolve(pyodideModule);
if(FAILURE==='lock')lockRequest.resolve({ok:false});
else lockRequest.resolve({ok:true,json:async()=>clone(lock)});
if(FAILURE==='source')sourceRequest.resolve({ok:false});else releaseSource();
if(FAILURE==='runtime'){
  await flush();runtimeRequest.reject(Error('VM failed'));
}else runtimeRequest.resolve(fakePython);
await task;
assert.equal(messages.at(-1).command,'init');assert.ok(messages.at(-1).error);
assert.equal(messages.some(message=>message.snapshot),false);assert.equal(actions.length,0);
""".replace("FAILURE", json.dumps(failure)))
