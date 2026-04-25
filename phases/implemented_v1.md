# Implementation Record — v1

Complete record of everything built, why each decision was made, and exact
commands to test locally and deploy to HuggingFace Spaces.

---

## 1. What Has Been Implemented

### 1.1 Environment (`ride_hailing_env/`)

| File | What it does |
|---|---|
| `environment.py` | Core OpenEnv loop — `reset()`, `step()`, `state()`, `get_hidden_state()`. Supports optional `override_decision` param so simulator LLM can supply accept/reject instead of rule-based logic. Fully backward compatible. |
| `simulator.py` | Deterministic accept/reject + patience decay. Added `simulate_step_with_decisions()` for Phase 3 — takes externally supplied booleans but keeps all patience/cancellation logic unchanged. |
| `reward.py` | Four reward layers: `compute_step_reward()` (intermediate shaping), `compute_terminal_reward()` (efficiency-adjusted profit), `compute_process_reward()` (directional convergence + urgency), `compute_simulator_reward()` (surplus × efficiency for simulator LLM). |
| `models.py` | Pydantic models: `Observation` (23 fields), `HiddenState`, `Action`, `StepResult`, `EpisodeOutcome`. |
| `scenario_generator.py` | Generates randomised episodes per task difficulty. |
| `config.py` | Task configs for easy / medium / hard. Reward constants. |

**Key design choices:**
- Hidden state (`rider_max_willingness`, `driver_min_willingness`) never exposed to the platform LLM — only the simulator sees it.
- `override_decision` is optional — all existing tests and rule-based scripts work unchanged.

---

### 1.2 API Server (`app.py`)

FastAPI server exposing the environment over HTTP.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Status, loaded checkpoints, endpoint list |
| `/reset` | POST | Start new episode, returns observation + episode_id |
| `/step` | POST | Propose price `{"type": "propose_price", "payload": {"price": X}}` |
| `/state` | GET | Current observation |
| `/schema` | GET | JSON schemas for action and observation |
| `/health` | GET | `{"status": "ok", "models_loaded": bool}` |
| `/models` | GET | Which checkpoints are loaded |
| `/demo/run` | POST | Run a full LLM episode, returns step-by-step trace |

**Phase 5 additions:** At startup, loads trained platform LoRA and simulator LoRA from paths set via env vars `PLATFORM_CKPT` and `SIM_CKPT`. Falls back gracefully if checkpoints don't exist (rule-based mode). Set `LOAD_MODELS=0` to skip model loading entirely (lightweight env-only mode).

---

### 1.3 Training Stack (`training/`)

| File | What it does |
|---|---|
| `model_loader.py` | Loads Qwen2.5-1.5B (platform, LoRA r=16) and Qwen2.5-0.5B (simulator, LoRA r=8) via Unsloth 4-bit quantization. `enable_inference_mode()` calls `FastLanguageModel.for_inference()` for ~2× rollout speedup on frozen models. Import is lazy — safe on non-GPU machines. |
| `prompt_builders.py` | `build_platform_prompt(obs)` — formats all 23 observation fields for the platform LLM. `build_simulator_prompt(hidden, obs, price)` — shows hidden thresholds + surplus analysis + safe rejection count. |
| `anti_hack.py` | 4 independent anti-hacking checks: price bounds, hardcoded exploit patterns, price-not-repeated, reasonable range. `score_format_compliance()` returns ±0.1 based on valid JSON with price key. `run_all_checks()` aggregates to (penalty, violations_list) at −0.5 per violation. |
| `rollout.py` | Rollout collection for all training phases. `collect_platform_rollout()` — single episode, platform vs rule-based. `collect_batch()` — batch of terminal samples. `collect_multiagent_rollout()` — both LLMs active, returns separate platform + simulator sample lists. `collect_multiagent_batch()` + `collect_simulator_batch()` — batch variants. `inspect_generations()` — human-readable print every 100 steps. `detect_reward_drift()` — returns True if reward dropped >30% from previous window. |
| `simulator_agent.py` | `SimulatorAgent` wraps Qwen2.5-0.5B. `decide(hidden, obs, price)` generates JSON accept/reject. `_parse()` has 3-tier fallback: full JSON → keyword scan → honest threshold check. Never silently breaks. |
| `grpo_utils.py` | Manual GRPO implementation. `grpo_setup_optimizer()` returns AdamW over trainable params. `grpo_step()` computes group-relative advantages, policy gradient loss over (prompt, completion, reward) triples, clips gradients, steps optimizer. **Why manual:** TRL's `GRPOTrainer` generates completions internally from a static dataset — it cannot interleave with stateful multi-turn `env.step()` calls. The math is identical to GRPO (Shao et al., 2024). |
| `train_platform.py` | Phase 2 (rule-based opponent) and Phase 4 (strategic simulator opponent) via `--simulator_ckpt` flag. 8-column monitoring every 50 steps. `inspect_generations()` every 100 steps. Rolling checkpoint every 200 steps. Drift detection + adapter rollback. Pre/post eval against the correct opponent. |
| `train_simulator.py` | Phase 3 simulator training against frozen platform. Bluff rate + collapse rate monitoring with health warnings. Same rolling checkpoint + drift detection pattern. |
| `evaluate.py` | 4-stage comparison table (A→D). Loads each model combo, runs `n_episodes` per stage per task, prints completion rate and avg reward tables. Detects and labels disruption (C < B) and recovery (D > B). Saves to `data/final_evaluation.json`. |

---

### 1.4 Demo & Visualisation

| File | What it does |
|---|---|
| `demo.py` | CLI demo. 4 modes map to the 4 evaluation stages. Shows step-by-step pricing, hidden thresholds, bluff detection (labels each decision "honest" or "bluffing"), deal outcome. Prints safeguards summary at the end of every run. |
| `scripts/plot_training_curves.py` | 4-panel matplotlib chart from training JSON files: platform reward curve, completion/cancel/timeout rates, simulator reward, bluff+collapse rates. |

---

### 1.5 Pipeline Script

| File | What it does |
|---|---|
| `run_training.sh` | Runs Phase 2 → 3 → 4 → evaluation in sequence. `set -e` stops on any failure. Configurable step counts at the top. |

---

### 1.6 Baselines (pre-existing)

| File | What it does |
|---|---|
| `baselines/midpoint_policy.py` | Always proposes midpoint of rider + driver quotes. |
| `baselines/adaptive_policy.py` | Rule-based policy that adjusts based on last response. |
| `scripts/quick_test.py` | Runs midpoint + adaptive across easy/medium/hard, 50 episodes each. Use this to verify the environment is working before training. |

---

## 2. What Is NOT Implemented (by design)

| Item | Reason |
|---|---|
| TRL `GRPOTrainer` | Cannot work with multi-turn stateful environment — replaced with mathematically equivalent manual GRPO in `grpo_utils.py` |
| Automated curriculum gating | Manual check is safer — user verifies bluff rate and completion rate before scaling to medium/hard |
| Simulator v2 fine-tuning | Optional — Phase 4b can be run manually if time permits |
| HF Spaces push | Requires user credentials — all code is ready, one `git push` away |
| Trained checkpoints | Require GPU training run — see Section 4 |

---

## 3. Local Testing — No GPU Required

These tests verify the environment, rewards, and anti-hacking logic without any model weights.

### 3.1 Install dependencies (CPU-only)

```bash
cd dynamic_pricing
pip install numpy pydantic fastapi uvicorn openenv-core==0.2.3
```

### 3.2 Verify environment works

```bash
# Run baseline policies across all 3 task difficulties (50 episodes each)
python scripts/quick_test.py
```

Expected output: a table showing Midpoint and Adaptive completion rates. Midpoint should complete ~50–70% on easy.

### 3.3 Run environment unit tests

```bash
python -m pytest tests/ -v
```

Key tests:
- `tests/test_environment.py` — reset/step/done/reward logic
- `tests/test_graders.py` — reward function correctness

### 3.4 Verify anti-hacking checks

```python
# Quick inline smoke test
from training.anti_hack import run_all_checks, score_format_compliance

penalty, violations = run_all_checks(
    price=1.0,            # at lower bound — should be fine
    completion='{"price": 1.0}',
    last_price=10.0,
    rider_quote=15.0,
    driver_quote=12.0,
)
print(penalty, violations)   # 0.0, []

penalty, violations = run_all_checks(
    price=0.001,          # below hard min — should trigger
    completion='{"price": 0.001}',
    last_price=10.0,
    rider_quote=15.0,
    driver_quote=12.0,
)
print(penalty, violations)   # -0.5, ['price_out_of_bounds']

print(score_format_compliance('{"price": 14.5}'))   # 0.1
print(score_format_compliance('not json'))           # -0.1
```

### 3.5 Verify reward functions

```python
from ride_hailing_env.reward import (
    compute_terminal_reward, compute_process_reward, compute_simulator_reward
)

# Terminal reward — completed ride
r = compute_terminal_reward(
    proposed_price=18.0, commission_rate=0.20, operational_cost=1.0,
    steps_taken=2, max_steps=5, ride_completed=True, timed_out=False,
    rider_max_willingness=20.0,
)
print(r)   # positive number

# Process reward — moving toward rejecting party
r = compute_process_reward(
    proposed_price=16.0, last_proposed_price=18.0,
    rider_accepted=False, driver_accepted=True,
    rider_patience=0.8, driver_patience=0.9,
    step_number=1, max_steps=5,
)
print(r)   # 0.03 (directional bonus)

# Simulator reward
r = compute_simulator_reward(
    proposed_price=15.0, rider_max_willingness=20.0, driver_min_willingness=10.0,
    ride_completed=True, steps_taken=2, max_steps=5,
)
print(r)   # (5.0 + 5.0) * 2.5 = 25.0
```

### 3.6 Start the API server locally

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Test endpoints:

```bash
# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST "http://localhost:7860/reset?task_name=easy"

# Propose a price
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"type": "propose_price", "payload": {"price": 15.50}}'

# Check loaded models
curl http://localhost:7860/models
```

In rule-based mode (no GPU), the `/demo/run` endpoint will return 503. All other endpoints work fully.

---

## 4. GPU Training — Full Pipeline

Requires: NVIDIA GPU (16 GB VRAM min), CUDA, Python 3.10+

### 4.1 Install training dependencies

```bash
pip install -r requirements.txt
```

Unsloth will auto-install the correct torch + bitsandbytes for your CUDA version.

### 4.2 Verify model loading works

```bash
python scripts/verify_models.py
```

This confirms Unsloth can load Qwen2.5-1.5B and 0.5B in 4-bit.

### 4.3 Run full training pipeline (one command)

```bash
bash run_training.sh
```

Or run phases individually with checkpointing:

```bash
# Phase 2 — Platform v0 vs rule-based simulator (~30–60 min on A100)
python training/train_platform.py \
  --task easy --steps 500 \
  --output_dir checkpoints/phase2

# Phase 3 — Simulator v1 vs frozen platform_v0 (~20–40 min)
python training/train_simulator.py \
  --task easy --steps 500 \
  --platform_ckpt checkpoints/phase2/platform_lora \
  --output_dir checkpoints/phase3

# Phase 4 — Platform v1 vs frozen simulator_v1 (~60–90 min)
python training/train_platform.py \
  --task easy --steps 1000 \
  --platform_ckpt  checkpoints/phase2/platform_lora \
  --simulator_ckpt checkpoints/phase3/simulator_lora \
  --output_dir checkpoints/phase4
```

### 4.4 What to watch during training

**Phase 2 (platform):**
- `completion_rate` should rise from ~25% toward ~60% by step 500
- `grpo_loss` should decrease or stabilise — if it spikes and stays high, learning rate may be too large
- `anti_hack_violations` should stay at 0

**Phase 3 (simulator):**
- `bluff_rate` should land between 20–50% — if 0%, simulator learned to always be honest (reward signal not distinguishing); if >80%, bluffing recklessly
- `collapse_rate` must stay below 60%
- Watch for health warnings printed automatically

**Phase 4 (platform v1):**
- `completion_rate` should dip initially (strategic opponent is harder) then recover to ≥70%
- Compare against Phase 2 baseline — recovery above Phase 2 proves adaptation

### 4.5 Resume from crash

Every 200 steps a rolling checkpoint saves to `checkpoints/last_stable/`. Resume:

```bash
# Resume platform training
python training/train_platform.py \
  --platform_ckpt checkpoints/last_stable/platform_lora \
  --task easy --steps 500 --output_dir checkpoints/phase2

# Resume simulator training
python training/train_simulator.py \
  --simulator_ckpt checkpoints/last_stable/simulator_lora \
  --platform_ckpt checkpoints/phase2/platform_lora \
  --task easy --steps 500 --output_dir checkpoints/phase3
```

### 4.6 Run evaluation (4-stage comparison table)

```bash
# Full table — 50 episodes per stage × 3 tasks
python training/evaluate.py

# Quick smoke-test — 10 episodes, easy only
python training/evaluate.py --n_episodes 10 --tasks easy --skip_missing

# Only stages you have checkpoints for
python training/evaluate.py --stages A B --skip_missing
```

Saves to `data/final_evaluation.json`.

### 4.7 Generate training curves chart

```bash
python scripts/plot_training_curves.py
# → data/training_curves.png
```

### 4.8 Run the demo

```bash
# The 4 presentation clips (one per evaluation stage)
python demo.py --mode untrained  --task easy --episodes 1   # Stage A baseline
python demo.py --mode trained_v0 --task easy --episodes 1   # Stage B platform v0
python demo.py --mode disrupted  --task easy --episodes 1   # Stage C disruption
python demo.py --mode trained    --task easy --episodes 3   # Stage D recovery (hero)

# Hard task demo
python demo.py --mode trained --task hard --episodes 1
```

---

## 5. HuggingFace Spaces Deployment

### 5.1 Prerequisites

- HuggingFace account with a Space created (Docker SDK)
- Trained checkpoints in `checkpoints/phase3/` and `checkpoints/phase4/`

### 5.2 Upload checkpoints to HF Hub

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload checkpoints to a model repo
huggingface-cli upload <your-org>/dynamic-pricing-checkpoints checkpoints/
```

### 5.3 Push code to Space

```bash
git remote add space https://huggingface.co/spaces/<your-org>/<space-name>
git push space main
```

### 5.4 Set Space environment variables

In the Space settings → Variables and secrets:

```
PLATFORM_CKPT = checkpoints/phase4/platform_lora
SIM_CKPT      = checkpoints/phase3/simulator_lora
LOAD_MODELS   = 1
PORT          = 7860
```

If checkpoints are on HF Hub (not in the repo), add:
```
HF_CHECKPOINT_REPO = <your-org>/dynamic-pricing-checkpoints
```

And add checkpoint download logic to `app.py` startup:
```python
from huggingface_hub import snapshot_download
if not Path(PLATFORM_CKPT).exists():
    snapshot_download(
        repo_id=os.getenv("HF_CHECKPOINT_REPO"),
        local_dir="checkpoints"
    )
```

### 5.5 Test the deployed Space

```bash
SPACE_URL="https://<your-org>-<space-name>.hf.space"

# Health check
curl $SPACE_URL/health

# Reset and step
curl -X POST "$SPACE_URL/reset?task_name=easy"
curl -X POST $SPACE_URL/step \
  -H "Content-Type: application/json" \
  -d '{"type": "propose_price", "payload": {"price": 15.50}}'

# Full LLM episode (requires trained models loaded)
curl -X POST $SPACE_URL/demo/run \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "mode": "trained", "show_hidden": true}'
```

### 5.6 Rule-based only mode (no GPU Space)

If the Space has no GPU, set `LOAD_MODELS=0`. The environment endpoints (`/reset`, `/step`, `/state`, `/health`) work fully. Only `/demo/run` requires models.

---

## 6. Checkpoint Directory Structure

```
checkpoints/
├── last_stable/
│   ├── platform_lora/      ← rolling checkpoint, updated every 200 steps
│   └── simulator_lora/
├── phase2/
│   └── platform_lora/      ← platform_v0: trained vs rule-based
├── phase3/
│   └── simulator_lora/     ← simulator_v1: trained vs frozen platform_v0
└── phase4/
    └── platform_lora/      ← platform_v1: trained vs frozen simulator_v1

data/
├── baseline_snapshot_easy.json
├── training_metrics_platform_easy.json
├── training_metrics_simulator_easy.json
├── final_evaluation.json
└── training_curves.png
```

---

## 7. Hackathon Guide Coverage

| Guide Point | Covered By |
|---|---|
| Step-by-step agent action | `env.step()` per price proposal |
| Programmatic success verification | `ride_completed` flag, deterministic reward |
| Curriculum (easy → hard) | `--task` flag, scale gate in training scripts |
| Multiple independent reward functions | Terminal + step + process + format + 4 anti-hack = 7 signals |
| Anti-reward-hacking | `training/anti_hack.py` — 4 checks, generation inspection every 100 steps |
| Rollback on drift | `detect_reward_drift()` + `load_adapter()` in both training scripts |
| Process-aware feedback | `compute_process_reward()` — directional convergence, urgency penalty |
| Unsloth for efficiency | `FastLanguageModel` 4-bit, LoRA, `enable_inference_mode()` on frozen models |
| GRPO algorithm | `training/grpo_utils.py` — manual implementation (TRL's GRPOTrainer incompatible with multi-turn env) |
| Monitor multiple scalars | 8-column logging every 50 steps + generation inspection every 100 steps |
| Save LoRA correctly | `model.save_pretrained()` adapter-only — no naive 4-bit merge |
| OpenEnv / FastAPI interface | `app.py` with all required endpoints |
| Before/after demo | `demo.py` 4 modes + `training/evaluate.py` 4-stage table |
| Safeguards explanation in demo | Printed at end of every `demo.py` run |
| Deploy to HF Spaces | Dockerfile + `app.py` ready — requires `git push space main` |
