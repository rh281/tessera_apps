# The hosted demo

`demo/app.py` is a single Streamlit page for `tessera_apps` 0.2.0, in two parts:

- **A map click.** The map opens on the Caldor demo AOI. A click runs the wildfire recovery trajectory of the clicked 10 m pixel ([trajectory.py](trajectory.py)): it reads the one 256 x 256 chunk (2.56 km by 2.56 km) that holds the pixel, for the pre-fire year and every year after it, from the anonymous store `s3://tessera-embeddings/v1.1/dclimate.icechunk` (AWS `us-west-2`) at snapshot `RC6FS07ZB0T76EXT8XRG`, and draws the pixel's similarity to its pre-fire year and its recovery index per year. Which threshold and unburned reference it uses depends on where and when:
  - **Inside the Caldor AOI at the demo's years** (pre-fire 2020, post-fire 2022, at that snapshot), it takes the threshold (0.244, by Otsu over the whole AOI) and each year's unburned reference median of the precomputed Caldor run in [precomputed/](precomputed), so the pixel's burned verdict and recovery index are that run's. Only the 3 x 3 opening and the 1 ha minimum area are applied to the window, so a patch cut by the window's edge can still differ: in the chunk of the burned pixel below, 383 of the run's 33,534 burned pixels stay unburned (99.42% of its pixels agree).
  - **Anywhere else, or at other years,** it applies the wildfire app's own threshold rule and unburned reference to the window alone, with the app's warnings about them (a suspect fallback threshold, an unburned reference less similar to itself after the fire than before it, a radar-free scar). The page says that these can differ from a run over a whole fire. It then also reads the year before the pre-fire year, when the threshold falls back to the stable pair or the window has a burned and unburned split.

  The page prints the chunk payload bytes read and the wall time of every click. A click in a chunk and years already read is served from an in-memory cache and reads nothing.
- **The bundled demos.** A dropdown shows each of the five bundled demos (Caldor, Rondonia, Everglades, Itapebi, Jaru): its preview, summary table, headline numbers with what each is scored against, its warnings and the command that reproduces it. These are files in [precomputed/](precomputed), so choosing a demo reads nothing from the store. The validation references are stand-ins (a Hansen outline for NIFC, a Sentinel-2 dNBR proxy for MTBS, ESA WorldCover for Global Mangrove Watch) and the two restore sites are synthetic; the page says so next to each.

No AWS credentials are needed to build or run the page, and the container image holds none: the store allows anonymous reads. Deploying to AWS takes an AWS account, but the running page uses no credentials there either.

**Network.** The server needs outbound HTTPS to `tessera-embeddings.s3.us-west-2.amazonaws.com` only, and asks `checkip.amazonaws.com` once at start (below), which may fail harmlessly. The viewer's browser loads the map's scripts from public CDNs (`cdn.jsdelivr.net`, `code.jquery.com`, `cdnjs.cloudflare.com` and `netdna.bootstrapcdn.com`, as folium 0.20.0 does) and its tiles from `server.arcgisonline.com` and `tile.openstreetmap.org`. Without the CDNs the map does not appear and nothing can be clicked; without the tile servers the map background is blank, but the outline and the clicks still work.

## What a click reads and costs

Measured on 2026-10-26 with the container image built from [Dockerfile](Dockerfile) (with one step stood in for, see the end), run on the build machine, which is outside `us-west-2`, with every `AWS_*` variable unset. Each number is what the page printed (and the server logged) for that click.

| Click | Read from the store | Time |
|---|---|---|
| First click on a burned Caldor pixel, zone 10N row 504,962 col 57,733, at the default years 2020 to 2025 (the precomputed Caldor run's settings) | 48,444,831 bytes | 3.57 s to read, 3.92 s in all, after 2.70 s to open the store (metadata only, once per server) |
| Second click in the same chunk | 0 bytes (cache) | 0.23 s |
| Two more chunks inside the Caldor AOI, (1965, 216) and (1976, 218) | 48,187,547 and 47,946,266 bytes | 2.24 and 2.03 s to read |
| Open ocean off the coast | 0 bytes; the page says the store has no data there and what to click instead | 0.46 s |
| The burned pixel again with the pre-fire year 2019 (the window's own rules, which also read the stable pair 2018) | 16,307,616 bytes for 2018 and 2019; 2020 to 2025 from the cache | 2.13 s to read, 2.41 s in all |
| West of the AOI at (-120.80, 38.70), chunk (1973, 206), the window's own rules with the stable-pair fallback | 56,031,773 bytes for 2019 to 2025 | 3.10 s to read, 3.37 s in all |
| Honolulu, clicked on the next copy of the world (longitude 202.14, wrapped to -157.86), zone 04N | 56,608,549 bytes for 2019 to 2025 | 4.48 s to read, 4.77 s in all |

Bytes are chunk payload bytes, counted as the app's reports count them. The store's manifests and metadata are not counted: on the build machine they were about 2.2 MB when the store opened and about 2.3 MB more on the first click (loopback bytes against the reader's count, 2026-10-26).

A click therefore reads about 48 to 57 MB and took 2 to 5 s from outside the region in that run; the same first click in a later container, on the image built from the final tree, took 9.40 s to read, so expect the time to vary with the network. Nothing else on the page touches the store. A host in `us-west-2` sits next to the store and should be faster; that was not measured.

**Memory** (the container's cgroup, sampled once a second, with Playwright driving the page; measured 2026-10-26 from outside `us-west-2`). In one session making 41 clicks in 40 new chunks (2,193,012,617 bytes read), the cgroup's peak was 497.7 MB after the first click, 804.8 MB after 6, 942.8 MB after 10, 1,023.5 MB after 19 and 1,108.4 MB after 41. A second session then clicked new chunks at the same moment as the first, 8 pairs (57 clicks in all, 2,971,431,286 bytes, every one printed its bytes): the peak rose to 1,137.7 MB, and the server process's own peak resident memory (VmHWM) was 1.24 GB. The same drive on the image before the click took the precomputed Caldor run's settings (`tessera-apps-demo:cp8`, the same 41 then 16 clicks) peaked higher: 1,212.8 MB for one session, 1,318.7 MB with two, and 1.43 GB for the process. Memory goes on rising after the 256 MB click cache is full (it holds about 28 window-years, five chunks at the default years; the rise is most likely allocator fragmentation across Streamlit's script threads), so the cache size is not a bound on the process. The cgroup did not count the image's shared library pages (they were charged to another cgroup on this host), so the process's peak is the better upper estimate. Plan for about 1.3 to 1.5 GB: a 1 GB limit was exceeded after 10 to 19 clicks.

Set `TESSERA_APPS_DEMO_CACHE_MB` to change the click cache's size (0 keeps nothing), and `TESSERA_APPS_DEMO_SNAPSHOT` to read another snapshot. A smaller cache was not measured.

**Startup:** the container reported healthy 2.5 to 2.9 s after `docker run` (2.93 s, 2.79 s and 2.54 s in three starts). Streamlit itself, when started headless on `0.0.0.0`, asks `checkip.amazonaws.com` for the server's public address once, to print it in the log. No credentials are involved.

## Run it locally

### With pip

From a checkout of the repository (it is private, so cloning needs access to it), with Python 3.12:

```bash
git clone https://github.com/rh281/tessera_apps.git
cd tessera_apps
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt   # the package's tested dependencies
.venv/bin/pip install --no-deps .                # tessera_apps 0.2.0 from the checkout
.venv/bin/pip install -r demo/requirements.txt   # Streamlit 1.66.1 and streamlit-folium 0.27.4
.venv/bin/streamlit run demo/app.py
```

Streamlit serves the page at <http://localhost:8501>. [requirements.txt](requirements.txt) explains why the package comes from the checkout: 0.2.0 is not on PyPI yet. Its first line, `-c ../requirements-lock.txt`, holds every package the lock pins to the lock's version.

On a minimal Linux system, importing rasterio (which the package uses) can fail with `libexpat.so.1: cannot open shared object file`. rasterio 1.5.2's wheel bundles GDAL but links it to the system's libexpat. Most full Linux installs have it; on Debian or Ubuntu, `apt-get install libexpat1` adds it. This is why the Dockerfile installs `libexpat1`.

### With Docker

The Dockerfile needs BuildKit with Dockerfile frontend 1.6 or later (Docker Engine 25 or later): the legacy builder cannot build it because of `RUN --mount`, and an older frontend stops with `unknown flag: start-interval` on its `HEALTHCHECK`. On Docker 29.8.2 here, the Dockerfile with a `# syntax=docker/dockerfile:1.5` line added failed with exactly that parse error, and with `1.6` it parsed.

From the repository root, which holds what the image installs (the lock and the package source) as well as the page:

```bash
docker build -f demo/Dockerfile -t tessera-apps-demo .
docker run --rm -p 8501:8501 tessera-apps-demo
```

Then open <http://localhost:8501>. What the image does:

- It starts from `python:3.12-slim`, pinned by digest.
- It installs the lock, the package with `--no-deps` and the page's requirements (held to the lock by their constraint line) into a virtual environment in a build stage, keeping no pip cache. It installs wheels only, so nothing is compiled.
- It copies that environment, the page and `precomputed/` into a final stage. That stage adds Debian's `libexpat1` and runs as the user `app` (uid 10001), who owns none of the page's files.
- It serves the page headless on `0.0.0.0`, on `$PORT` (8501 when unset), with usage statistics off. A health check polls Streamlit's `/_stcore/health`.

To use another port, pass `-e PORT=8600 -p 8600:8600`. `docker stop` ends it in about a second. The image holds no proxy setting, no CA file other than the base image's own and no AWS variable or credential: `docker history`, `docker inspect` and a search of its files found none.

The image is 2.08 GB on disk (2,079,539,097 bytes, `docker image inspect`) and about 482 MB compressed. Of that, 166 MB is 41 packages that the lock pins for testing and the notebooks (mypy, ruff, debugpy, jedi, ipython, pytest and others) and the page does not use. A build with an empty build cache (the base image already pulled) took 5 min 17 s here on 2026-10-26, 200 s of it installing the lock.

Only `linux/amd64` was built and run. To build an amd64 image on an ARM machine (an Apple silicon Mac, for example), add `--platform linux/amd64` to `docker build`. Other platforms were not tried: the build installs wheels only, so it stops if a pinned package has no wheel for the platform.

### Behind a TLS-intercepting proxy

A normal machine needs none of this section. On a machine whose HTTPS goes through a proxy that re-signs TLS with its own CA (as on the machine this was built on), give the proxy and the CA bundle to the build only. Docker does not record the predefined proxy build arguments in the image history. The Dockerfile mounts the CA as the BuildKit secret `proxy_ca` for the pip steps only, so it is never written to a layer:

```bash
docker build --network host --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --secret id=proxy_ca,src=/path/to/ca-bundle.crt -f demo/Dockerfile -t tessera-apps-demo .
```

The `apt-get` step also needs to reach a Debian mirror. Debian's default sources are plain HTTP, so an HTTP proxy for them goes in `--build-arg http_proxy`. At run time, give the proxy to the container the same way, never to the image: `-e HTTPS_PROXY -e NO_PROXY`, plus `--network host` if the proxy listens on the host's loopback. On the build machine the proxy passed the store's S3 hosts through without re-signing them, so the container needed only those two variables and no CA. A proxy that also re-signs S3 would need its CA trusted inside the container, which was not tested.

## Refresh the precomputed outputs

[precompute.py](precompute.py) runs the five bundled demos with the package's own command line, pinned to snapshot `RC6FS07ZB0T76EXT8XRG`, in a work directory outside the repository. It copies each run's preview, summary table and validation file into `precomputed/<name>/` and writes `precomputed/index.json`, which records the exact commands. The click reads the Caldor run's threshold and unburned reference from there too. From the repository root, with the package installed as above:

```bash
.venv/bin/python demo/precompute.py --work /some/dir/outside/the/repo
```

Every child process runs with every `AWS_*` variable removed. Reruns skip any demo whose outputs are current, and rewrite a copied file only when its bytes change. The script ends by checking that no file under `precomputed/` holds an absolute local path, an AWS credential or an em or en dash. Keep the work directory to make reruns cheap.

A full run is expensive. The last one read 28,735,764,178 bytes in about 20 minutes from outside `us-west-2` (Caldor alone 23,025,809,926 bytes in 913.5 s), with a peak of about 4.1 GB of memory. Building the Caldor dNBR severity proxy adds 612.5 MB read in 2 min 19 s; pass `--dnbr PATH` to reuse an existing one. Pass `--snapshot ID` to precompute another snapshot. The page's click reads `TESSERA_APPS_DEMO_SNAPSHOT`, so set it to the same snapshot; at any other snapshot than the precomputed run's, every click uses the window's own rules.

## Deploy to Streamlit Community Cloud

What the app needs there:

- **The repository connected.** Sign in to Community Cloud with GitHub and allow it to read `rh281/tessera_apps`. The repository is private, so Community Cloud also needs access to private repositories, and an app deployed from a private repository is visible only to the viewers you invite.
- **The entrypoint.** Create an app from repository `rh281/tessera_apps`, branch `main`, main file path `demo/app.py`.
- **The requirements file.** Community Cloud installs the dependency file it finds next to the entrypoint, here `demo/requirements.txt`, before looking at the repository root. It does not install the lock or the checkout itself; the file's constraint line asks for the lock's versions, if its installer honours constraint lines.
- **Python 3.12.** Choose 3.12 under Advanced settings when deploying. The version cannot be changed later without redeploying.
- **No secrets.** The page reads the anonymous store and needs no credentials, keys or environment variables.

**tessera_apps 0.2.0 must come from PyPI there.** `demo/requirements.txt` asks for `tessera_apps==0.2.0`, and on 2026-10-26 PyPI has only 0.1.0, which cannot run this page. Until 0.2.0 is published, the deployment stops at that line with "No matching distribution found for tessera_apps==0.2.0". The alternative that [requirements.txt](requirements.txt) gives, `tessera_apps @ git+https://github.com/rh281/tessera_apps.git@<commit>`, works only while the repository is public. pip on Community Cloud has no credentials for a private git URL, and putting a token in a secret is what this demo avoids. So for the private repository: publish 0.2.0 to PyPI first, then deploy.

**Versions there.** Resolved alone from PyPI on 2026-10-26 (with a local 0.2.0 wheel standing in for PyPI's), `demo/requirements.txt` without its constraint line gave 101 packages, 7 of them at versions neither the tests nor the image ran: pillow 13.0.0 (the lock has 12.3.0), python-dateutil 3.0.0 (2.9.0.post0), pandas 3.1.0 (3.0.6), numcodecs 0.17.1 (0.17.0), multidict 7.2.1 (7.1.0), certifi 2026.10.23 (2026.7.22) and platformdirs 4.13.5 (4.13.0). With the constraint line, uv resolved 100 packages with all 7 at the lock's pins. Before a Community Cloud deploy, run the page's offline tests (`tests/test_demo_trajectory.py`) on the set it installs.

If the app's log shows `libexpat.so.1: cannot open shared object file`, add a file `packages.txt` at the repository root (not in `demo/`: Community Cloud reads it only from the root) holding the line `libexpat1`; Community Cloud installs the Debian packages it lists. If the app runs out of memory, the measured peak is about 1.1 to 1.3 GB in the container's cgroup (1.2 to 1.4 GB for the process; above); lower the cache with `TESSERA_APPS_DEMO_CACHE_MB`, which was not measured.

## Deploy to AWS

**Not on App Runner.** AWS App Runner is closed to new customers since 2026-04-30, and AWS plans no new features for it (AWS's "AWS App Runner availability change" page; it recommends Amazon ECS Express Mode instead). Its WebSocket request, [aws/apprunner-roadmap#13](https://github.com/aws/apprunner-roadmap/issues/13) "Support web sockets", is closed as not planned. Streamlit's page talks to its server over a WebSocket (`/_stcore/stream` in Streamlit 1.66.1's source), so on App Runner it would load but never get past "Connecting", even for an account that can still create App Runner services. Run the image on Amazon ECS on Fargate instead, behind an Application Load Balancer, which carries WebSockets.

The steps below use the AWS CLI v2 in `us-west-2`, the store's region, so that a click's reads stay in the region and take the shortest path. You need AWS credentials to push the image and create the service. The running task needs none: it gets no task role, and the page reads the store anonymously. **None of this was tried**: the build environment has no AWS account. The JSON shapes of steps 2 and 3 match the `ecs` model of botocore 1.43.119 (the release of 2026-10-26); nothing else about them was checked.

1. **Build and push the image to ECR.** Replace `123456789012` with your account id.

   ```bash
   aws ecr create-repository --repository-name tessera-apps-demo --region us-west-2
   aws ecr get-login-password --region us-west-2 \
     | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
   docker build --platform linux/amd64 -f demo/Dockerfile \
     -t 123456789012.dkr.ecr.us-west-2.amazonaws.com/tessera-apps-demo:0.2.0 .
   docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/tessera-apps-demo:0.2.0
   ```

2. **Two roles, neither of them a task role.** An execution role, which ECS uses to pull the image from ECR and send the logs to CloudWatch Logs; it needs the managed policy `AmazonECSTaskExecutionRolePolicy`. For Express Mode, also an infrastructure role, which ECS uses to create the load balancer, target group, security groups and scaling policy; the console offers to create both. Give the task no task role and no environment variables.

3. **Create the service with ECS Express Mode,** which makes the Fargate service, the Application Load Balancer and its target group in one call. The task listens on port 8501, the load balancer's health check is Streamlit's `/_stcore/health`, and the task is 1 vCPU and 2 GB (`1024` CPU units, `2048` MiB). 2 GB because the measured peak is about 1.1 to 1.3 GB (1.2 to 1.4 GB for the process; above), so a 1 GB task would run out of memory. Keep exactly one task: Streamlit keeps each viewer's session, and the click cache, in the memory of one server process, and its own server code warns that several replicas need sticky sessions.

   ```bash
   aws ecs create-express-gateway-service --region us-west-2 \
     --service-name tessera-apps-demo \
     --execution-role-arn arn:aws:iam::123456789012:role/ecsTaskExecutionRole \
     --infrastructure-role-arn arn:aws:iam::123456789012:role/<your infrastructure role> \
     --primary-container '{"image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/tessera-apps-demo:0.2.0", "containerPort": 8501}' \
     --health-check-path /_stcore/health \
     --cpu 1024 --memory 2048 \
     --scaling-target '{"minTaskCount": 1, "maxTaskCount": 1}'
   ```

   The same settings work for a Fargate service you set up yourself behind an Application Load Balancer: one task (desired count 1, or turn on target-group stickiness if you allow more), container port 8501, target-group health check path `/_stcore/health`, task size 1 vCPU and 2 GB, `us-west-2`, no task role.

4. **Open the page** at the endpoint the service reports (`aws ecs describe-express-gateway-service`, under the active configuration's ingress paths). A first click in a chunk should print about 48 MB read; the server's log, with each click's bytes and time, is in CloudWatch Logs.

To update, push a new tag and point the service at it (`aws ecs update-express-gateway-service`, or a new task definition revision for a service you set up yourself).

## What could not be checked

The page, the image and the numbers above were run and measured. The deployment steps were not: the build environment cannot reach docs.streamlit.io or docs.aws.amazon.com, and has no Community Cloud or AWS account to deploy with. Check these before relying on them:

- **App Runner** was checked only through secondary sources: a web search returned AWS's "AWS App Runner availability change" page (docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html, which does not resolve from here) and Wikipedia, both giving the closure to new customers on 2026-04-30; and GitHub showed [aws/apprunner-roadmap#13](https://github.com/aws/apprunner-roadmap/issues/13) closed as not planned (closed 2024-04-26; AWS's last comment, 2022-11-28: "We don't have native support for web sockets at this point"). This is why the App Runner route was dropped.
- **The ECS route:**
  - that `aws ecs create-express-gateway-service` takes the options in step 3 as written (the JSON shapes match botocore's model, the option names follow the CLI's usual naming);
  - whether Express Mode's scaling target holds the service at one task, and whether its load balancer needs stickiness otherwise;
  - the load balancer's idle timeout against Streamlit's WebSocket, which stays open for as long as the page is;
  - the name of the CloudWatch log group, and the managed policy the infrastructure role needs;
  - that 1 vCPU with 2 GB is an offered Fargate task size.
- **Community Cloud details:**
  - that Python is chosen under Advanced settings, and only at deploy time;
  - how apps from private repositories are shared, and how many it allows;
  - whether its installer honours the constraint line in `demo/requirements.txt`;
  - whether its machines have `libexpat1` already;
  - its memory limit per app.

  The dependency-file order (the entrypoint's folder first, then the repository root) and the root-only `packages.txt` come from Streamlit's own docs source, `streamlit/docs` on GitHub, `content/deploy/community-cloud/deploy-your-app/app-dependencies.md` lines 72 and 113, read on 2026-10-26, not from a deployment.
- **Docker versions:** that Docker Engine 25 is the first release whose bundled Dockerfile frontend parses `HEALTHCHECK --start-interval` (Docker's release notes were not read); only frontends 1.5 (fails) and 1.6 (parses) were tried.
- **Region:** the speed of a click from inside `us-west-2`. Every number above was measured from outside the region.
- **The `apt-get install libexpat1` step:** the build machine's proxy refuses every Debian mirror, so the image measured here got that step's result another way. Its runtime stage was swapped, with `docker build --build-context runtime=...`, for the same base image plus the files of Debian trixie's own `libexpat1` package (2.9.0-1~deb13u1, taken from the official `buildpack-deps:trixie-scm` image). Every other step of the Dockerfile ran as written.
