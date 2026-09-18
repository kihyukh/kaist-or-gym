"""Build the standalone website using the same browser physics and controls."""

import base64
from pathlib import Path

from kaist_rl_lab.apps.coffee_browser import (
    BROWSER_CSS,
    CONTROLLER_JAVASCRIPT,
    TOOLBAR_HTML,
    WORKER_JAVASCRIPT,
    browser_bundle,
)
from kaist_rl_lab.envs.coffee_pouring_canvas import CANVAS_HTML, CANVAS_JAVASCRIPT

STUDENT_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#123c51">
<title>Coffee pouring · KAIST RL Lab</title>
<link rel="stylesheet" href="/styles.css"><script src="/student.js" defer></script>
</head><body>
<main id="coffee-demo">
  <header class="site-header"><p class="site-eyebrow">KAIST RL LAB · LEARN BY DOING</p>
    <h1>Can you pour 700 mL?</h1><p id="class-name">Checking your class link…</p>
  </header>
  <details class="coffee-help"><summary>How to play</summary>
    <p>Move the cup under the pot, then tilt the pot to pour. Each arrow keeps a joint moving.
    Tap its square button to hold that joint, or <b>Stop all motors</b> to hold both arms.</p>
    <p><b>Hold still does not stop the coffee.</b> Use <b>Pause time</b> to think.
    Fill the cup near 700 mL with little spill, then return both vessels upright.</p>
    <p>Tap <b>Submit trajectory</b> when you finish. It ends this attempt and sends the moves
    to your instructor. Wait for your receipt before closing this tab.</p>
  </details>
  <noscript>This simulation needs JavaScript enabled in your browser.</noscript>
  __TOOLBAR__
  __CANVAS__
  <section class="coffee-save" aria-labelledby="save-heading">
    <h2 id="save-heading">Finish &amp; submit</h2>
    <label><span id="participant-label">Student ID</span>
      <input class="coffee-participant" maxlength="64" autocomplete="off" autocapitalize="off"
        spellcheck="false" aria-describedby="participant-note" /></label>
    <p id="participant-note">Your ID and recording are visible to the instructor.</p>
    <button type="button" disabled data-command="save">Submit trajectory</button>
    <p class="coffee-submission" role="status" aria-live="polite"></p>
    <a class="coffee-download" hidden>Download a backup (.npz)</a>
    <p class="save-note">Submitting ends this attempt. You can reset afterward to try again.</p>
  </section>
  <footer>Runs on your phone. Switching away pauses time and stops the motors.</footer>
</main></body></html>
""".replace("__TOOLBAR__", TOOLBAR_HTML).replace("__CANVAS__", CANVAS_HTML)

STUDENT_CSS = BROWSER_CSS + """
:root {font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:#17374a; background:#f3f6f8; color-scheme:light;}
body {margin:0; padding:20px max(12px,env(safe-area-inset-right)) 20px max(12px,env(safe-area-inset-left));}
* {box-sizing:border-box;}
main {max-width:980px; margin:auto;}
.site-header {margin-bottom:12px;}
.site-eyebrow {font-size:11px; font-weight:750; letter-spacing:1.5px; color:#467385; margin:0 0 6px;}
h1 {font-size:clamp(24px,4vw,36px); line-height:1.1; margin:0 0 8px; letter-spacing:-.5px;}
#class-name {font-size:14px; margin:0; color:#516b7c; overflow-wrap:anywhere;}
.coffee-help {font-size:14px; margin:10px 0 14px;}
.coffee-help summary {cursor:pointer; padding:6px 0; font-weight:650;}
.coffee-help p {max-width:740px; line-height:1.55;}
.coffee-status {line-height:1.4;}
.coffee-save {background:white;}
.coffee-save h2 {font-size:18px; margin:0 0 8px;}
.coffee-save label {font-size:14px; font-weight:650;}
.coffee-save input {margin-top:6px;}
.coffee-save button {background:#123c51; color:white; min-width:190px; font-size:16px;}
.coffee-submission {overflow-wrap:anywhere; font-weight:650;}
.coffee-save .save-note, #participant-note {font-size:12px; color:#52656f; font-weight:400;}
footer {font-size:11px; color:#607586; padding:16px 0; text-align:center;}
button:focus-visible, summary:focus-visible, a:focus-visible, input:focus-visible {outline:3px solid #0099aa; outline-offset:3px;}
@media(max-width:600px) {
  body {padding-top:12px;}
  .site-eyebrow {font-size:10px;}
  .site-header {margin-bottom:4px;}
  .coffee-help {margin:4px 0 8px; font-size:13px;}
  .coffee-status {font-size:12px;}
  .coffee-toolbar {position:sticky; top:0; z-index:5; background:#f3f6f8; padding:6px 0;
    box-shadow:0 5px 7px #f3f6f8;}
  .coffee-save button {width:100%;}
}
"""

STUDENT_BOOTSTRAP = r"""
(async () => {
  const element=document.querySelector('#coffee-demo');
  const label=document.querySelector('#class-name');
  const joinToken=new URLSearchParams(location.search).get('class');
  const studentConfig={worker_url:'/coffee-worker.js',bundle_url:new URL('/coffee-bundle.zip',location.href).href,
    collecting:false,participant_required:false};
  if (joinToken) {
    try {
      const response=await fetch('/api/session?class='+encodeURIComponent(joinToken),{signal:AbortSignal.timeout(15000)});
      const session=await response.json();
      if (!response.ok) throw new Error('This class link is not valid. Please scan the instructor’s current QR code.');
      label.textContent=session.name+(session.open ? '' : ' · Submissions closed — practice only');
      if (session.open) {
        studentConfig.collecting=true;
        studentConfig.participant_required=!!session.participant_required;
        studentConfig.upload_url='/api/submissions?class='+encodeURIComponent(joinToken);
      }
    } catch(error) {
      label.textContent=error.name==='TimeoutError' ? 'The class website is taking too long to respond. Please reload.' : error.message;
      element.querySelector('.coffee-status').textContent='The demo could not load your class. Check your connection and reload.';
      return;
    }
  } else label.textContent='Practice mode · Use your class QR code to submit to your instructor';
  const participant=element.querySelector('.coffee-participant');
  participant.required=studentConfig.participant_required;
  document.querySelector('#participant-label').textContent=studentConfig.participant_required ? 'Student ID (required)' : 'Participant code (optional)';
  if (!studentConfig.collecting) {
    document.querySelector('#save-heading').textContent='Finish & save';
    document.querySelector('#participant-note').textContent='Practice recordings stay on your phone unless you download and share them.';
    element.querySelector('.save-note').textContent='Saving ends this attempt. No recording is sent to an instructor in practice mode.';
  }
  const props={value:studentConfig};
  const watch=()=>{};
  const trigger=()=>{};
  __CANVAS_JS__
  __CONTROLLER_JS__
})().catch(error=>{
  document.querySelector('.coffee-status').textContent='Could not start the demo. Please reload. '+error.message;
});
""".replace("__CANVAS_JS__", CANVAS_JAVASCRIPT).replace(
    "__CONTROLLER_JS__", CONTROLLER_JAVASCRIPT
)


def build_static_site(destination: str | Path) -> Path:
    """Generate self-contained frontend assets; Pyodide loads from its pinned CDN."""
    from kaist_rl_lab.apps.coffee_instructor import (
        INSTRUCTOR_CSS,
        INSTRUCTOR_HTML,
        INSTRUCTOR_JAVASCRIPT,
    )

    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in {
        "index.html": STUDENT_HTML,
        "styles.css": STUDENT_CSS,
        "student.js": STUDENT_BOOTSTRAP,
        "coffee-worker.js": WORKER_JAVASCRIPT,
        "instructor.html": INSTRUCTOR_HTML,
        "instructor.css": INSTRUCTOR_CSS,
        "instructor.js": INSTRUCTOR_JAVASCRIPT,
    }.items():
        (directory / name).write_text(text, encoding="utf-8")
    (directory / "coffee-bundle.zip").write_bytes(base64.b64decode(browser_bundle()))
    return directory
