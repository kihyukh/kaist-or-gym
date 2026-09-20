# Host the coffee classroom

Students open a classroom QR link, wait for the simulation to load, play, and
tap **Save & share**. Saving uploads the trajectory directly to the instructor and
shows **Shared ✓** with a receipt after the server has stored it. All student setup
happens automatically in the browser. The website
downloads the Python browser runtime on first use; students do not install Python,
run a notebook, or sign in. A network connection is needed for loading and submission.
The runtime, pinned libraries, and simulation source download concurrently; package
resolution is no longer performed in the browser.

The instructor opens `/instructor`, signs in, creates a class, and projects its
QR code. Each class has its own random student link. The dashboard lists submitted
attempts, replays recorded physics, and downloads the original training data.
Closing a class stops new submissions but permits retries of already saved attempts.
The instructor list refreshes every five seconds and when the instructor returns
to the tab. Failed uploads retain the recording for retry or backup download.

The main website (`/`) joins the sole open class automatically and pins its class
link in the address bar. This makes that open class publicly joinable; the class
token is for routing submissions, not instructor access. When multiple classes
are open, students must use their instructor's QR/link to avoid sending to the
wrong class. With no open class, or at `/?practice=1`, recordings stay local.
Explicit class links are never reassigned to another class, even after closing.

Above the training controls, **Generated practice demonstrations** and **Submitted
demonstrations** list individual trajectories with total recorded reward, cup/spill
amounts, outcome, and duration. Generated examples can be inspected even before
creating a class. The student list belongs to the selected class and can be filtered
by participant. Choose **Replay**, then **Play** to visualize either kind of
recording; pause, scrub, or change playback speed to inspect a moment. The viewer
shows cumulative recorded reward alongside simulated time. Downloads preserve the
original `.npz` archive, and generated examples never enter the student dataset.
The first verified frame is streamed immediately, followed by small batches while
the rest of the recording is checked. Play is available before preparation finishes.
A bounded in-memory cache avoids repeating physics work for recently viewed recordings;
closing or switching recordings cancels outstanding preparation.

Displayed rewards are undiscounted sums of archived step rewards, not rewards
recomputed by the viewer. This preserves recorded bonuses and penalties without
adding a new end-of-episode reward. Older submissions acquire their reward summaries
when listed; missing or unreadable archives show a dash rather than a fabricated zero.

The instructor dashboard also includes **Random agent · no learning**. Click
**Start random trial** to load the browser simulator and run the same coffee physics.
All six joint commands are independently uniform over clockwise, hold, and
counter-clockwise; each vector is held for a uniformly random 1–32 simulation steps
(up to one second). Trials stop at 30 simulated seconds or the environment's terminal
condition. Pause/resume, new trials, and a local table of the last ten completed
trials support the lecture. Backgrounding the page pauses the experiment. These
trials never upload as student submissions and do not change the policy: the demo
illustrates that experience without a feedback-driven update is not learning.

Student demonstrations stop automatically after **60 simulated seconds**, or
**1,920 physics steps at 32 Hz**. The toolbar shows time remaining. Pausing or
backgrounding the page pauses the countdown; this is not a 60-second wall-clock
deadline. A timed-out attempt remains downloadable and can be submitted, labeled
**Time limit** and unsuccessful. The default successful-only cloning filter excludes
it. New submissions longer than 60 simulated seconds are rejected; older saved
recordings remain available for replay and download.

## Behavior cloning from demonstrations

After selecting a class, use **Behavior cloning** on the instructor page:

1. Choose **Selected class's student submissions**. Successful attempts are used
   by default; turn that filter off to illustrate imitation of poor demonstrations.
2. Click **Train policy**. The report shows the number of demonstrations and
   state–action pairs, any excluded recordings, and a held-out action error.
3. Click **Run cloned policy** to watch a new, closed-loop rollout in the actual
   browser simulator. Pause/resume and reset are available. Trials stop on success,
   failure, or after 60 simulated seconds.

For a ready-to-use lecture example, choose **Generated practice demonstrations**.
Fifteen bundled practice recordings achieve approximately 672–686 mL with negligible
spill from varied classroom starting poses. They deliberately move cautiously and
begin returning the pot 15–30 mL early, creating real inefficiency and an accuracy
gap for reward learning. Their target remains 700 mL; they meet the existing ±40 mL
success tolerance. These are intentionally imperfect practice, not expert-quality
examples. The learner can improve accuracy even after a run first qualifies as
successful. No rewards, archived states, or target values are modified to create
this difference. They were generated by a reproducible controller
using the unmodified environment and are never added to the student submission
list. Their provenance and regeneration instructions are in
[`coffee_examples/README.md`](../../kaist_rl_lab/apps/coffee_examples/README.md).

This is explicitly **nearest-neighbor behavior cloning**: the fitted policy
selects the demonstrated action for the closest physical state, using fixed
feature scales. It does not use elapsed time, rewards, future observations, an
expert-controller fallback, or a recorded action timeline. It is a simple
nonparametric supervised baseline, not a neural-network or reinforcement-learning
trainer. The browser chooses a fresh action at each 32 Hz physics step.

Training reads only the selected class, requires instructor authentication and
the same-origin request check, and leaves submitted archives unchanged. It uses
up to 20 compatible trajectories (32 Hz, 700 mL), sampled evenly to at most 50,000
state–action pairs. Training runs off the request loop, with one job at a time and
three jobs per minute. Models contain states/actions, not student identities, and
are sent only to the authenticated instructor browser. The fitted model remains
in that tab; reload or switching class/source requires retraining.

With at least two demonstrations, action error is measured on a whole held-out
trajectory before fitting the final policy on all selected examples. This error
is not task success. Student demonstrations and the random-agent experiment sample
independent horizontal and vertical offsets for each vessel. The cup varies by
up to ±9 cm horizontally and ±5 cm vertically, and the pot by ±11 cm and ±7 cm.
Vessels remain upright with an empty cup, full pot, and a 700 mL target. All 16
extreme combinations and 2,000 random samples passed the environment's reset
validation. These larger four-dimensional variations cover visibly different
approaches, rather than shifting both vessels together.

Cloned-policy playback, fine-tuning exploration, and checkpoint evaluation use
one fixed canonical pose: cup (-0.28, 0.28) m and pot (0.26, 0.62) m. Thus changes
in policy behavior are not confounded by changes in the starting pose. Training
examples remain varied and the fitted policy still acts on the current state.
The arm bases are **1.28 m apart**, compared with the original 1.16 m: each base
moves outward by 6 cm while link lengths stay the same. Students, generated examples,
cloned-policy playback, and every fine-tuning/evaluation rollout share this geometry.
Wider spacing increases reach demands, but does not by itself guarantee a larger
fine-tuning gain. The environment reward and 700 mL goal remain the same. Fine-tuning uses the
separate accuracy-first score described below.

Seed, exact initial joints, and arm spacing are preserved in every new recording.
Recordings made with different arm spacing remain available for replay and download,
but are excluded from behavior cloning for the current classroom. Each replay
constructs the environment at its recorded spacing; archives without that field use
the original 1.16 m. New nonlegacy recordings use archive schema 2 so older package
versions reject them instead of silently assuming the old geometry. The reader still
accepts original schema-1 recordings.

## Fine-tuning with reward

After training a cloned policy, use **Fine-tune with reinforcement learning**.
The default is **Policy search · classroom demo**, with **10 iterations**; 25, 50,
and 100 are also available. **Actor–critic · PPO** provides an alternative learning
method. Both start by evaluating the original clone, then complete one exploratory
or candidate trial per iteration and evaluate the resulting current policy without
exploration noise. All trials use the same fixed starting pose and 1.28 m arm spacing.

The comparison reports the **Accuracy / speed score**, fill, target error,
whether the **±5 mL precision goal** is met, spill, duration, and success. Accuracy
comes first: **exactly 700 mL** earns the highest precision component, and pours
within ±5 mL receive a strong bonus. Speed still matters among accurate pours.
The environment's original ±40 mL completion tolerance remains distinct from the
stricter precision goal. No particular learning gain is guaranteed.

**Policy search** learns two positive speed multipliers: one for approaching and
pouring, and one for returning the pot. It always queries the clone with the real
current observation. If the sum of the clone's three pot-control commands is less
than `-1e-4`, it uses the return multiplier; otherwise it uses the approach/pour
multiplier. It scales all six commands by that multiplier and clips to `[-1, 1]`.
Both gains remain between **0.7 and 1.4**. No liquid reading, target value, or
recorded trajectory is changed.

The search tests paired faster/slower candidates along one parameter at a time,
from shared starting settings and with a seeded initial radius. Improvements are
retained. After a pair fails to improve, it halves that coordinate's radius and
rotates to the other coordinate. The second candidate still uses the pair's
original center even if the first was accepted. A fresh deterministic rollout of
the current policy supplies each navy evaluation point. Rejected candidates count
as completed iterations but not as applied updates. The visible diagnostic lists
both candidate multipliers and both shared pair-center values.

**Actor–critic (PPO)** explores state-dependent speed adjustments during a trial.
A critic learns to estimate future Accuracy / speed scores from observed rollouts, and the
actor uses estimated advantages for clipped policy-gradient updates. Its speed
multiplier is `1 + 0.5 * tanh(z)`, bounded between 0.5 and 1.5. Gaussian latent
noise has standard deviation 0.5 and is resampled every 0.5 simulated seconds.
Updates use 12 clipped PPO epochs, learning rate 0.15, a sampled-state Gaussian
KL cap of 0.08, and a global latent-mean change cap of 0.6. The critic and GAE
remain unchanged. These limits bound updates; evaluated performance can rise or
fall. In both methods the frozen
clone is queried at every physics step; zero commands remain zero and motion
directions remain those of the clone. Neither method calls the demonstration
controller, copies future actions from a recording, or uses an expert correction.

A live chart sits beside the simulation on wider screens and stacks above it on
phones. Every candidate is visible by default in amber, including rejected and
failed trials. For policy search, each candidate is a deterministic policy run;
for PPO, amber represents an exploratory rollout with action noise. Navy always
shows the separate current-policy evaluation without noise. Green shows the best
evaluated score so far. Rejected search candidates leave the retained policy
unchanged, so repeated navy scores and genuine plateaus are expected.

Choose score, absolute volume error, or completion time as the plotted measure.
The recent-iteration detail view removes the original-clone and best-score guides
from the axis calculation, making small recent differences easier to see. The
inspector still reports candidate, current-policy, and clone measurements, along
with score changes. Missing evaluations remain missing; no interpolated samples
or smoothed measurements are invented. The curve remains available while watching
a policy and clears when a new experiment or cloning model replaces it.

**Measurements for every tested policy** lists the original clone and separate
rows for each candidate and completed evaluation, including their outcome and
whether an update was applied. Scores display six decimal places, liquid amounts
four, and durations five (enough to represent every 1/32-second physics step).
All scoring, comparisons, and policy selection continue to use the unrounded
floating-point values. Display detail does not change the reward function or the
learner's decisions.

**Training speed** defaults to **Fastest available**. **Policy playback speed** is
separate and defaults to **4×** for **Watch original clone** and **Watch best policy**.
Both selectors offer 4×, 8×, and fastest. Changing one does not change the other's
setting. Requested playback speeds are limited by the device. Every 1/32-second
physics step and policy decision is still computed; speed only changes pacing.
Adaptive batches of up to 32 steps reduce rendering overhead while yielding for
pause, stop, and speed changes. Repeated geometry calculations are cached, and
hidden training steps omit render-only diagnostics; displayed frames still use
the full renderer.

### Reward used for fine-tuning

The instructor page visibly explains the **Accuracy / speed score**, a fine-tuning objective
separate from the environment reward stored in demonstration archives. The original
clone, candidate trials, current-policy evaluations, and policy playback all use
the same accuracy-and-speed reward helper. Archived rewards and submitted trajectories are unchanged.

For each physics step, `dt = 1/32` seconds and `gamma = 0.99 ** dt`. With error and
spill in litres and cup angle in radians, the fine-tuning step reward is:

- `-dt` for elapsed time, or one point per simulated second;
- `-40 * newly spilled liquid`;
- `-0.024 * dt * sum(control**2)` over all six motors;
- `-0.032 * dt * abs(cup angle)`;
- `gamma * Phi(next) - Phi(current)`, where `Phi = -20 * absolute target error`.

At success, failure, or timeout, it also adds:

`(100 + A(error) if success else -100) - 100 * final absolute target error - 14 * total spill`

The continuous precision bonus is `A(error) = 1000 * exp(-0.5 * (error / 0.005)**2)`,
where error is in litres. It is awarded **only on success**: exactly 700 mL gives
1,000 points, and the bonus is at least 606.53 points within ±5 mL. It decreases
smoothly toward zero as accuracy worsens. A failure cannot earn this bonus merely
by briefly reaching the target fill with unsafe spill or tilt.

The potential `Phi(next)` is **zero at every terminal state, including timeout**.
Consequently its discounted sum is the same 14 points for every rollout from an
empty cup with a 700 mL target. This provides intermediate filling feedback without
changing which completed policy is preferred.

The displayed score is `sum(gamma**t * fine_tuning_reward_t)` from step zero. All
terms, including terminal rewards, use the same discount of **0.99 per simulated
second**. Success still requires being within 40 mL of 700 mL, at most 20 mL spilled,
flow at most 8 mL/s, cup tilt at most 8°, and pot tilt at most 12°. Trials have a
60-second limit. Within those limits, even a worst-case successful pour within
±5 mL at 60 seconds scores at least 343.39, while a successful pour at least 10 mL
off target cannot score above 248.34 even before time and control costs. Thus
accuracy takes priority over a fast but inaccurate pour. At equal time and other
costs, exactly 700 mL uniquely maximizes the score. Discounting and elapsed-time
cost still favor faster completion when accuracy is comparable. Animation speed
has no effect.

The trajectory library continues to show **undiscounted recorded environment
reward**, including its original success bonus and penalties. Those numbers use a
different reward function and should not be compared directly with Accuracy / speed scores.

The best completed evaluation is retained, including the original clone. The
comparison measures refinement on one fixed pose, not average performance across
the broader demonstration distribution. Seeds control exploration; results retain
the training seed and fixed environment seed. Exact results can vary slightly
across numerical runtimes.

Training runs in the instructor browser. Backgrounding the page pauses it, and
stop retains fully evaluated checkpoints. Changing class, demonstration source,
cloning policy, or reloading clears the experiment. No RL trials are submitted as
student data. Computation needs no additional server/GPU service; browser speed
depends on the device and number of demonstration samples.

Method references: [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
and [finite horizons and time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Run locally

From the repository root, install once for the operator:

```sh
python -m pip install '.[web]'
```

Set `COFFEE_INSTRUCTOR_PASSWORD` to a unique password of at least 12 characters and
`COFFEE_SESSION_SECRET` to an independent random value of at least 32 characters.
Generate each securely with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`
and save them in your deployment's secret settings. They are never put into the
student page, QR code, or repository.

```sh
export PUBLIC_BASE_URL=http://localhost:8000
export COFFEE_DATA_DIR=/absolute/path/to/coffee-data
coffee-pouring-web
```

Open `http://localhost:8000/instructor`. Localhost links work on the operator's
computer only; use the public HTTPS deployment before generating classroom QR codes.

## Deploy with Docker and persistent storage

Build from the repository root:

```sh
docker build -f deploy/coffee-web/Dockerfile -t coffee-classroom .
docker run --rm -p 8000:8000 --mount source=coffee-data,target=/data \
  -e COFFEE_INSTRUCTOR_PASSWORD -e COFFEE_SESSION_SECRET \
  -e PUBLIC_BASE_URL coffee-classroom
```

Set `PUBLIC_BASE_URL` to the real HTTPS origin before running, such as the URL
assigned by your hosting provider. The server uses this origin for every student
link and QR and rejects unsafe public HTTP configurations. Terminate HTTPS at your
host or reverse proxy. Set its request-body limit to at least 12 MB and its request
timeout to at least 180 seconds. Long recordings can take 1–2 minutes to reconstruct
for the first replay; one replay is prepared at a time. Do not cache `/api/*` responses. The first Pyodide
load fetches pinned runtime assets from jsDelivr and Python dependencies from PyPI;
check that the classroom network permits them before the talk.

Use one running instance and one application worker, backed by a persistent `/data`
volume. SQLite holds class/session metadata; `archives/` holds submitted `.npz`
files. Back up the full volume using a SQLite-aware backup or while the process is
stopped. A restart preserves classes, receipts, and trajectories. Do not use an
ephemeral container disk for the teaching dataset.

The container initializes the mounted directory at startup, then drops to the
unprivileged `coffee` user before starting the server. This handles a fresh disk
whose mount hides the ownership set while building the image. The application
uses one Uvicorn worker and follows the hosting provider's `PORT` setting.

## Deploy on Render

The repository's [`render.yaml`](../../render.yaml) creates one Docker web service
in Singapore, using the smallest paid compute instance (`0.5c-512mb`, 512 MB RAM)
and a 1 GB persistent disk at `/data`. No managed database is needed. The expected
base cost is US$7.25/month before taxes and any usage overages; check the displayed
price in Render before creating the service.

1. Sign in to Render, connect the GitHub repository, and create a **Blueprint**.
   Select the branch containing this website and `render.yaml` (initial deployment:
   `codex/coffee-classroom-render`). The service inherits that branch.
2. Enter `COFFEE_INSTRUCTOR_PASSWORD` when prompted. Use a unique password of at
   least 12 characters and save it in your password manager. Render generates
   `COFFEE_SESSION_SECRET` independently; neither value belongs in Git.
3. Deploy the Blueprint. Render supplies HTTPS, the assigned `onrender.com` address,
   and `PORT`. The app uses `RENDER_EXTERNAL_URL` automatically unless
   `PUBLIC_BASE_URL` is explicitly set. The health-check path is `/health`.
4. Visit `/instructor`, sign in, create the seminar class, and use that class's QR
   code in the slides. Check submission, replay, and download from a phone before
   distributing the QR.

Automatic service deploys are disabled so an unrelated commit cannot interrupt a
class. Also set the Blueprint's **Auto Sync** to **No** after creating it: Blueprint
configuration changes otherwise sync independently of service auto-deploy settings.
Use **Manual Deploy** for a reviewed application update, or **Manual Sync** when the
Blueprint itself changes. A disk-backed service briefly stops during redeployment,
so deploy between classes. Do not increase the instance count or application workers.

For a custom domain, add it to the Render service, complete Render's DNS setup,
then set `PUBLIC_BASE_URL` to its HTTPS origin and redeploy before creating new
QR codes. Keep the `onrender.com` address available until older QR codes have been
replaced. Changing the public origin changes which origin the instructor login accepts.

Keep the disk when redeploying. Export and back up the teaching dataset before
deleting the service or its disk. Monitor disk usage in Render; increase the disk
size if necessary, because submitted archives share the initial 1 GB allocation.

References: [Render Docker deployment](https://render.com/docs/docker),
[Blueprint reference](https://render.com/docs/blueprint-spec),
[persistent disks](https://render.com/docs/disks), and
[Blueprint sync behavior](https://render.com/docs/infrastructure-as-code).

For the existing KAIST lab Cloudflare setup, use a separate public hostname. The
private vault's collaborator login would block the student workflow. This Python
service needs a container/Python host; a Cloudflare Worker would need an R2/SQLite
storage adapter rather than this local-filesystem store.

## Access and data behavior

- Only `/api/session` and archive submission are public. Student links cannot list,
  download, or replay other students' attempts.
- Instructor credentials create a signed, revocable 12-hour HTTP-only cookie.
  HTTPS cookies are Secure and SameSite=Strict. Instructor mutations check Origin.
- Participant codes identify attempts; avoid asking students for full names or
  other unnecessary personal details. The class creator chooses whether a code
  is required.
- Uploads are bounded and validated with the shared NumPy reader. Receipts are
  idempotent, so retrying after a weak mobile connection does not duplicate data.
- Replay samples at most 400 actual physics frames; recordings with a physics
  mismatch are explicitly rejected for replay. Keep the teaching app version
  unchanged throughout a class. The original archive is always available privately.
- The app allows up to 5,000 attempts per class, 240 uploads/minute per connection
  address, and 10 instructor login attempts/minute. Configure your reverse proxy
  deliberately if you need different limits for larger audiences.
