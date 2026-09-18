# Host the coffee classroom

Students open a classroom QR link, wait for the simulation to load, play, and
submit. All student setup happens automatically in the browser. The website
downloads the Python browser runtime on first use; students do not install Python,
run a notebook, or sign in. A network connection is needed for loading and submission.

The instructor opens `/instructor`, signs in, creates a class, and projects its
QR code. Each class has its own random student link. The dashboard lists submitted
attempts, replays recorded physics, and downloads the original training data.
Closing a class stops new submissions but permits retries of already saved attempts.

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
