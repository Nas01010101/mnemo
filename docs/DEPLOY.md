# Deploying Mnemo on Alibaba Cloud (mandatory Proof-of-Deployment)

The hackathon requires the backend to **run on Alibaba Cloud** + a short recording +
a repo code file that uses Alibaba Cloud services/APIs (that file is
[`src/alicloud_oss.py`](../src/alicloud_oss.py), using OSS).

Two paths. **ECS is recommended for the proof video** (easy to show a public IP +
the running process); Function Compute is cheaper/serverless.

## Prereqs (both paths)
- Alibaba Cloud account + a **RAM user** with an AccessKey (ID + Secret), scoped to
  ECS/FC + OSS (don't use the root AccessKey).
- An OSS bucket (for memory snapshots), e.g. `mnemo-<you>` in `ap-southeast-1`.
- Env values ready: `DASHSCOPE_API_KEY`, `ALIBABA_CLOUD_ACCESS_KEY_ID/_SECRET`,
  `OSS_ENDPOINT` (e.g. `https://oss-ap-southeast-1.aliyuncs.com`), `OSS_BUCKET`.

## Path A — ECS (recommended for the demo)
```bash
# 1. Create a small ECS instance (Ubuntu 22.04, ecs.t6-c1m1.large is plenty),
#    open port 8000 in its security group, note the public IP.
# 2. SSH in, install Docker, clone the repo:
ssh root@<PUBLIC_IP>
apt-get update && apt-get install -y docker.io git
git clone <YOUR_REPO_URL> mnemo && cd mnemo
# 3. Run the backend (env vars injected, never baked in):
docker build -t mnemo .
docker run -d --name mnemo -p 8000:8000 \
  -e DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY \
  -e ALIBABA_CLOUD_ACCESS_KEY_ID=$AK_ID \
  -e ALIBABA_CLOUD_ACCESS_KEY_SECRET=$AK_SECRET \
  -e OSS_ENDPOINT=$OSS_ENDPOINT -e OSS_BUCKET=$OSS_BUCKET \
  mnemo
# 4. Verify from your laptop (this is the proof shot):
curl http://<PUBLIC_IP>:8000/health
curl -X POST http://<PUBLIC_IP>:8000/ingest -H 'content-type: application/json' \
  -d '{"message":"I moved to Toronto last week."}'
```
For the proof recording: show `curl http://<PUBLIC_IP>:8000/health` returning ok, the
`docker ps` / `docker logs mnemo` on the ECS box, and one `src/alicloud_oss.py snapshot`
writing to OSS (visible in the OSS console).

## Path B — Function Compute (serverless, cheapest)
Deploy the same container to FC (custom-container runtime) via Serverless Devs (`s`):
```bash
# install: npm i -g @serverless-devs/s ; s config add (with your AK)
# push image to ACR, then `s deploy` with an http trigger. FC gives a public URL.
```
FC scales to zero (cheapest) but the ephemeral filesystem means memory must be
snapshotted to OSS (that's exactly what `src/alicloud_oss.py` is for) and restored on
cold start.

## Proof-of-deploy checklist
- [ ] Backend reachable on an Alibaba Cloud public URL/IP (`/health` returns ok)
- [ ] Short recording: the service running on Alibaba Cloud + a live request
- [ ] `src/alicloud_oss.py` linked as the "uses Alibaba Cloud services/APIs" file
- [ ] One OSS snapshot visible in the Alibaba Cloud console
