# Phase 0 — Early Environment Deployment

## Why This Comes Before Training

The hackathon guide is explicit: **deploy your environment before training seriously**.

> "A good habit is to deploy an early version of the environment before training seriously. That catches API and packaging issues early."

Deployment is not a final step — it is infrastructure that the entire team shares. If you wait until Phase 5, you will discover packaging bugs while the clock is running. Deploy now while the environment is already stable.

---

## Goal

Get the existing environment running on HuggingFace Spaces so that:
- All team members can hit the same environment endpoint
- The FastAPI server works remotely before any training code depends on it
- Docker packaging issues are caught now, not at demo time

**Exit criteria:** `POST /reset` and `POST /step` return correct responses from the deployed Space URL.

---

## Step 0.1 — Verify Local Server Works

Before touching HuggingFace, confirm the local FastAPI server runs cleanly:

```bash
python app.py
```

In a separate terminal, run a quick smoke test:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "propose_price", "payload": {"price": 18.50}}'
```

Confirm both return valid JSON. If either fails — fix before continuing.

Also verify the `/health` endpoint responds:
```bash
curl http://localhost:7860/health
```

---

## Step 0.2 — Run Environment Stability Checks

Before deployment, confirm all environment contracts hold. This is the checklist from the hackathon guide under "Scale only after environment is stable":

```bash
python scripts/quick_test.py
python -m pytest tests/ -v
```

Verify manually:
- [ ] `reset()` returns valid `Observation` with all 23 fields populated
- [ ] `step()` with valid price returns `StepResult` with `reward`, `done`, `info`
- [ ] Rewards are sensible (not all zeros, not all extreme values)
- [ ] Timeouts fire correctly at `max_steps`
- [ ] Patience decay reaches 0.0 and triggers cancellation correctly
- [ ] All three tasks (easy / medium / hard) work

**Do not proceed to deployment if any check fails.**

---

## Step 0.3 — Docker Build and Local Container Test

```bash
docker build -t dynamic-pricing-env .

docker run -p 7860:7860 \
  -e PORT=7860 \
  dynamic-pricing-env
```

Run the same smoke test curl commands against `localhost:7860`. This catches dependency issues (missing packages, wrong Python version) that only surface inside the container.

Common failures to check for:
- `openenv-core` version mismatch inside container
- Missing `python-dotenv` in Dockerfile
- Port binding issues

---

## Step 0.4 — Deploy to HuggingFace Spaces

```bash
# Create space if it doesn't exist
huggingface-cli repo create dynamic-pricing-env --type space --space_sdk docker

# Add remote
git remote add space https://huggingface.co/spaces/<org>/dynamic-pricing-env

# Push — Space builds from Dockerfile automatically
git push space main
```

Environment variables to set in Space Settings → Variables:
```
PORT = 7860
DEBUG = false
```

**No API keys should be in the Space at this stage** — the environment does not need them. Model API keys are only needed during training (local).

---

## Step 0.5 — Verify Remote Endpoint

Once the Space builds (watch build logs in HuggingFace UI):

```bash
SPACE_URL="https://<org>-dynamic-pricing-env.hf.space"

curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

curl -X POST $SPACE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"type": "propose_price", "payload": {"price": 18.50}}'
```

Share the Space URL with all team members — they now have a shared environment endpoint.

---

## Step 0.6 — Record the Space URL

Add to `.env`:
```
SPACE_URL=https://<org>-dynamic-pricing-env.hf.space
```

Update `client.py` to point to Space URL when `SPACE_URL` is set:
```python
import os
API_BASE_URL = os.getenv("SPACE_URL") or os.getenv("API_BASE_URL", "http://localhost:7860")
```

---

## Step 0.7 — Stability Gate Before Any Training

Before moving to Phase 1 and starting training, complete this checklist. This is from the hackathon guide's "Scale only after environment is stable":

- [ ] `reset()` works locally
- [ ] `step()` works locally  
- [ ] Rewards are sensible (positive on deal close, −5.0 on cancel, −2.0 on timeout)
- [ ] Timeouts fire at correct step count
- [ ] Logs are visible (reward, done, info printed or logged)
- [ ] Docker container runs locally
- [ ] Remote Space endpoint responds correctly
- [ ] All 3 tasks work on remote endpoint

**This checklist must be fully checked before Phase 1 begins. Training against a broken environment wastes all compute.**

---

## What This Phase Produces

```
Deployed:  https://<org>-dynamic-pricing-env.hf.space  (live environment)
Updated:   .env  (SPACE_URL added)
Updated:   client.py  (reads SPACE_URL)
Verified:  all stability checks pass
```

## Deliverables for Phase 0

- [ ] Local server smoke test passes
- [ ] Docker container builds and runs locally
- [ ] Environment deployed to HuggingFace Spaces
- [ ] Remote endpoint verified via curl
- [ ] Space URL shared with all team members
- [ ] Stability gate checklist completed
