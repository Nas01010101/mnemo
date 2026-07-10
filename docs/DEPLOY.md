# Deploying Tenet on Alibaba Cloud (mandatory Proof-of-Deployment)

## Current status (deployed 2026-07-10)
**Live on Alibaba Cloud Function Compute 3.0** (`ap-southeast-1`), verified working:

- **URL:** https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run
- `GET /health` → `200 {"status":"ok","provider":"qwen","embed_provider":"qwen",...}`
- `POST /ingest` → `200 {"stored":1,"ids":[1]}` — a real `qwen3.6-flash` distillation call,
  live, on Alibaba Cloud compute.
- `GET /` → the belief-ledger web demo UI (`src/tenet/static/index.html`), 200, ~17KB.

**How it was deployed** (once `ALIBABA_CLOUD_ACCESS_KEY_ID`/`_SECRET` were available —
a root-account AccessKey, not RAM-scoped; fine for a hackathon demo, not what you'd run
in production): Function Compute's zip-based **web function** path, no container/ACR
needed. `pip install --target=pkg --platform manylinux2014_x86_64 --abi cp311
--only-binary=:all: openai numpy fastapi uvicorn` (cp311 wheels — `custom.debian12`'s
system `python3` is 3.11; `custom.debian10`'s is 3.7 and silently fails to import
modern `typing` features, `custom.debian11` is 3.9 — **the Python version must match the
`custom.debianNN` variant's system interpreter**, there's no way to pin a Python version
independently in this path), copy `src/tenet` into the package root, a 6-line
`app.py` (`uvicorn.run("tenet.api:app", host="0.0.0.0", port=9000)`), zip (~21MB), then
`CreateFunction` (runtime `custom.debian12`, `customRuntimeConfig.command=["python3",
"app.py"]`, `customRuntimeConfig.port=9000`, `cpu=0.5`, `memorySize=512`) +
`CreateTrigger` (`http`, `authType=anonymous`) via the `alibabacloud_fc20230330` Python
SDK — the `aliyun` CLI's `--body file://…` path silently truncated the ~28MB
base64-encoded inline zip mid-request (malformed-JSON error from the server), so the SDK
was used directly instead. Full reproduction script referenced below.

**Known caveat, stated plainly:** `TENET_DB_PATH=/tmp/tenet.db` — FC's per-instance
`/tmp` is ephemeral and resets on cold start (the function scales to zero after idle).
The demo works for a live session but isn't durable storage; **OSS-backed snapshot/
restore is not wired up on this deploy** because OSS itself returned `UserDisable` for
this account (the OSS product needs a separate one-time console activation this account
hasn't done — a human step, not a credentials problem). `src/tenet/alicloud_oss.py`
still stands as the proof-of-Alibaba-Cloud-OSS-usage code file; wiring `snapshot()`/
`restore()` into `api.py`'s startup/shutdown hooks is the next step once OSS is
activated (tracked as a `what's next` item, not required for the submission).

**What's already satisfied independent of this:** the "uses Alibaba Cloud
services/APIs" proof does *not* require a deployed backend — DashScope (Qwen Cloud) IS
Alibaba Cloud Model Studio, and every model/embedding call already hits
`dashscope-intl.aliyuncs.com` (`src/tenet/config.py`, `src/tenet/memory.py`,
`src/tenet/distill.py`). The deployed backend above is bonus polish on top of that,
matching `docs/hackathon/COMPETITION.md`'s verbatim requirement (a live URL is not the
literal ask — a linked code file demonstrating Alibaba Cloud API usage is).

## Credentials reality (what you actually need)
- **To RUN Tenet: only `DASHSCOPE_API_KEY`.** Nothing else. The app never calls OSS
  unless you explicitly invoke `src/tenet/alicloud_oss.py`.
- **Embeddings on a serverless deploy: use `EMBED_PROVIDER=qwen` (the shipped
  default), not `local`.** `EMBED_PROVIDER=local` pulls in `sentence-transformers` +
  a ~130MB model — fine on ECS (persistent disk), a bad fit for Function Compute's
  read-only/ephemeral filesystem and cold-start budget. `.env.example` already ships
  `EMBED_PROVIDER=qwen`, so no config change is needed for either deploy path;
  `EMBED_PROVIDER=local` is only for the offline/zero-key demo.
- **"Uses Alibaba Cloud services and APIs" proof — already satisfied.** Qwen Cloud /
  DashScope *is* Alibaba Cloud Model Studio; every model + embedding call in
  `src/tenet/config.py`, `src/tenet/memory.py`, `src/tenet/distill.py` hits
  `dashscope-intl.aliyuncs.com` (an Alibaba Cloud API). That is the linkable proof file.
  `src/tenet/alicloud_oss.py` (OSS) is an *optional* second, stronger proof — use only if you
  want it.
- **"Backend running on Alibaba Cloud" (compute) — the one optional add.** Needs an
  Alibaba Cloud AccessKey, which is generated from the **same Qwen Cloud account you
  already have** (Qwen Cloud = Alibaba Cloud): https://ram.console.aliyun.com/manage/ak
  (~2 min). If you provide one later, deploy is one command (below). If not, the entry
  still uses Alibaba Cloud services throughout via DashScope.

---

The hackathon asks for the backend to run on Alibaba Cloud + a short recording +
a repo code file that uses Alibaba Cloud services/APIs. The primary proof file is the
DashScope integration itself; [`src/tenet/alicloud_oss.py`](../src/tenet/alicloud_oss.py) (OSS) is
an optional stronger proof.

Two paths. **Path B (Function Compute) is what's actually deployed and live** (URL
above) — cheaper/serverless, no idle cost. Path A (ECS) is documented as an
alternative/fallback (e.g. if a proof video specifically wants to show a public IP +
`docker ps`) but untested on this pass.

## Prereqs (both paths)
- Alibaba Cloud account + an AccessKey (ID + Secret). Ours was a **root-account** key
  (not a scoped RAM user — the docs elsewhere recommend RAM + least privilege; use what
  you have, root works for a hackathon demo).
- OSS bucket: *not required* for Path B as deployed (ephemeral `/tmp`, see "Current
  status" above) — only needed if you wire up snapshot/restore.
- Env values: `DASHSCOPE_API_KEY`, `ALIBABA_CLOUD_ACCESS_KEY_ID/_SECRET`,
  `ALIBABA_CLOUD_REGION`.

## Path A — ECS (recommended for the demo)
```bash
# 1. Create a small ECS instance (Ubuntu 22.04, ecs.t6-c1m1.large is plenty),
#    open port 8000 in its security group, note the public IP.
# 2. SSH in, install Docker, clone the repo:
ssh root@<PUBLIC_IP>
apt-get update && apt-get install -y docker.io git
git clone <YOUR_REPO_URL> tenet && cd tenet
# 3. Run the backend (env vars injected, never baked in):
docker build -t tenet .
docker run -d --name tenet -p 8000:8000 \
  -e DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY \
  -e ALIBABA_CLOUD_ACCESS_KEY_ID=$AK_ID \
  -e ALIBABA_CLOUD_ACCESS_KEY_SECRET=$AK_SECRET \
  -e OSS_ENDPOINT=$OSS_ENDPOINT -e OSS_BUCKET=$OSS_BUCKET \
  tenet
# 4. Verify from your laptop (this is the proof shot):
curl http://<PUBLIC_IP>:8000/health
curl -X POST http://<PUBLIC_IP>:8000/ingest -H 'content-type: application/json' \
  -d '{"message":"I moved to Toronto last week."}'
```
For the proof recording: show `curl http://<PUBLIC_IP>:8000/health` returning ok, the
`docker ps` / `docker logs tenet` on the ECS box, and one `python -m tenet.alicloud_oss snapshot`
writing to OSS (visible in the OSS console).

## Path B — Function Compute (serverless, cheapest — this is what's live)
No Docker/ACR needed: FC's zip-based **web function** with a `custom.debianNN` runtime
runs a plain `pip install --target` package directly. This is what's actually deployed
(URL above). Reproduce or redeploy:
```bash
# 1. package (cp311 wheels — matches custom.debian12's system python3.11)
pip install --target=pkg --python-version 3.11 --implementation cp --abi cp311 \
  --platform manylinux2014_x86_64 --only-binary=:all: openai numpy fastapi uvicorn
cp -r src/tenet pkg/tenet
cat > pkg/app.py <<'EOF'
import os, uvicorn
uvicorn.run("tenet.api:app", host="0.0.0.0", port=int(os.environ.get("FC_SERVER_PORT", "9000")))
EOF
cd pkg && zip -rq ../code.zip . && cd ..

# 2. deploy — pip install alibabacloud_fc20230330 alibabacloud_tea_openapi first.
# (the `aliyun` CLI's `--body file://` truncates large inline zips silently — use the SDK)
python3 - <<'PY'
import base64, os
from alibabacloud_fc20230330.client import Client
from alibabacloud_fc20230330 import models as m
from alibabacloud_tea_openapi import models as om

region = os.environ["ALIBABA_CLOUD_REGION"]
account_id = "<YOUR_ACCOUNT_ID>"          # aliyun sts GetCallerIdentity
client = Client(om.Config(
    access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
    access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
    endpoint=f"{account_id}.{region}.fc.aliyuncs.com", region_id=region,
    read_timeout=120000, connect_timeout=20000,
))
zip_b64 = base64.b64encode(open("code.zip", "rb").read()).decode()
req = m.CreateFunctionInput(
    function_name="tenet-demo", runtime="custom.debian12", handler="app.handler",
    code=m.InputCodeLocation(zip_file=zip_b64),
    custom_runtime_config=m.CustomRuntimeConfig(command=["python3", "app.py"], port=9000),
    cpu=0.5, memory_size=512, timeout=60, disk_size=512, internet_access=True,
    environment_variables={
        "DASHSCOPE_API_KEY": os.environ["DASHSCOPE_API_KEY"],
        "QWEN_BASE_URL": os.environ["QWEN_BASE_URL"], "QWEN_MODEL": os.environ["QWEN_MODEL"],
        "EMBED_PROVIDER": "qwen", "LLM_PROVIDER": "qwen", "TENET_DB_PATH": "/tmp/tenet.db",
    },
)
client.create_function(m.CreateFunctionRequest(body=req))
client.create_trigger("tenet-demo", m.CreateTriggerRequest(body=m.CreateTriggerInput(
    trigger_name="httpTrigger", trigger_type="http",
    trigger_config='{"authType":"anonymous","methods":["GET","POST","PUT","DELETE"]}',
)))
PY

# 3. verify (this is the proof shot)
curl https://<function-url>.fcapp.run/health
curl -X POST https://<function-url>.fcapp.run/ingest -H 'content-type: application/json' \
  -d '{"message":"I moved to Toronto last week."}'
```
FC scales to zero (cheapest — free-tier eligible, no ECS/hour billing while idle). The
ephemeral `/tmp` means memory doesn't survive a cold start unless OSS snapshot/restore
(`src/tenet/alicloud_oss.py`) is wired into `api.py`'s startup — not done on this deploy
(OSS wasn't activated for this account); see "Current status" above.

## Proof-of-deploy checklist
- [x] Backend reachable on an Alibaba Cloud public URL (`/health` returns ok) —
      https://tenet-demo-wrenarokun.ap-southeast-1.fcapp.run/health
- [ ] Short recording: the service running on Alibaba Cloud + a live request (for the
      demo video — the `curl` commands above are the shot)
- [x] `src/tenet/alicloud_oss.py` linked as the "uses Alibaba Cloud services/APIs" file
- [ ] One OSS snapshot visible in the Alibaba Cloud console (blocked — OSS product not
      activated for this account; not required by the verbatim rules, see above)
