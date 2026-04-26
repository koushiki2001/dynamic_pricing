# Training Instructions — Dynamic Pricing Multi-Agent RL

A step-by-step guide to running training, evaluating results, and fixing common problems.

---

## 1. What You're Actually Training

Think of it as two players in a negotiation game:

- **Platform** — the pricing algorithm. Tries to complete rides at a good price.
- **Simulator** — a fake rider + driver. Tries to reject unfair prices.

You train them in 3 rounds (phases):

```
Round 1 (Phase 2): Train Platform vs a dumb rule-based simulator
         → Platform learns the basics of pricing

Round 2 (Phase 3): Train Simulator vs the now-smarter Platform
         → Simulator learns to be strategic / harder to fool

Round 3 (Phase 4): Train Platform again vs the now-smarter Simulator
         → Platform has to learn genuinely good pricing
```

---

## 2. Machine Requirements

You need a GPU. CPU-only will take 10+ hours — not worth it.

| Machine | Notes |
|---|---|
| Google Colab T4 | Free tier, ~15 hrs/day limit |
| Google Colab A100 | Pay-as-you-go (~$0.40/hr in Colab Pro) |
| Kaggle T4 x2 | Free, 30 hrs/week |
| RunPod A100 | ~$1.50/hr, most flexible |
| Local RTX 3060/3090/4090 | Works well if you have one |

---

## 3. One-Time Setup

```bash
# Install dependencies (run once)
pip install -r requirements.txt

# If on Colab, also install:
pip install unsloth openenv-core
```

All commands below should be run from the `dynamic_pricing/` folder.

---

## 4. Running Training

### Option A — Run everything at once (recommended first time)

```bash
bash run_training.sh
```

Runs all 3 rounds + evaluation automatically. Walk away and come back.

**Time estimates:**

| Machine | Full pipeline time |
|---|---|
| Colab T4 (free) | 3–4 hrs |
| Colab A100 | 1–1.5 hrs |
| Kaggle T4 | 2.5–3 hrs |
| RunPod A100 | 1–1.5 hrs |
| Local RTX 4090 | 1.5–2 hrs |
| Local RTX 3060 | 3–4 hrs |

---

### Option B — Run rounds one by one (better for watching progress)

**Round 1: Teach Platform the basics**
```bash
python training/train_platform.py --task easy --steps 200 --output_dir checkpoints/phase2
```

**Round 2: Teach Simulator to be strategic**
```bash
python training/train_simulator.py \
  --task easy --steps 150 \
  --platform_ckpt checkpoints/phase2/platform_lora \
  --output_dir checkpoints/phase3
```

**Round 3: Re-train Platform against the harder Simulator**
```bash
python training/train_platform.py \
  --task easy --steps 200 \
  --platform_ckpt checkpoints/phase2/platform_lora \
  --simulator_ckpt checkpoints/phase3/simulator_lora \
  --output_dir checkpoints/phase4
```

**Time estimates per round (Option B):**

| Round | Steps | Colab T4 | A100 |
|---|---|---|---|
| Phase 2 (platform warmstart) | 200 | ~50 min | ~15 min |
| Phase 3 (simulator) | 150 | ~35 min | ~12 min |
| Phase 4 (platform vs sim) | 200 | ~50 min | ~15 min |
| Evaluation | — | ~10 min | ~5 min |

---

### Hackathon shortcut (verify pattern in ~1 hour on T4)

Edit `run_training.sh` and set:
```bash
PHASE2_STEPS=150
PHASE3_STEPS=150
PHASE4_STEPS=200
```

Then run `bash run_training.sh`. Takes ~45–60 min on T4. Good enough to confirm the result pattern before committing to a full run overnight.

---

## 5. Reading the Live Training Logs

Every 50 steps you'll see a row like this:

```
step | reward | completion% | cancel% | timeout% | bluff% | grpo_loss
  50 |  0.28  |    0.41     |  0.31   |  0.28    |  0.12  |   0.81
 100 |  0.51  |    0.63     |  0.21   |  0.16    |  0.24  |   0.44
 150 |  0.59  |    0.68     |  0.18   |  0.14    |  0.31  |   0.31
```

**What each column means:**

| Column | What it means | Healthy range |
|---|---|---|
| `reward` | Average score per episode | Should rise over time |
| `completion%` | % of rides that completed successfully | > 60% is good |
| `cancel%` | % where rider/driver cancelled | Should fall |
| `timeout%` | % where negotiation ran out of time | Should fall |
| `bluff%` | (Phase 3 only) % of simulator bluffs | 20–40% is healthy |
| `grpo_loss` | How much the model is still changing | Should fall below 0.4 |

**Stop training when:** reward plateaus for 3+ consecutive readings AND completion% > 65%.

---

## 6. Evaluating Results

Run this after training (or after any individual round):

```bash
python training/evaluate.py --tasks easy --n_episodes 30 --output data/final_evaluation.json
```

This prints a 4-row comparison table:

```
Stage | Setup                          | Avg Reward | Completion%
  A   | Untrained vs Rule-based        |   0.18     |   28%
  B   | Platform_v0 vs Rule-based      |   0.52     |   64%
  C   | Platform_v0 vs Simulator_v1    |   0.31     |   45%
  D   | Platform_v1 vs Simulator_v1    |   0.61     |   71%
```

**What you're looking for:**

| Check | Meaning | What to do if it fails |
|---|---|---|
| A < B | Training improved on baseline | If not: something is broken, check setup |
| C < B | Simulator is actually strategic | If not: train Phase 3 longer |
| **D > B** | Platform adapted to harder opponent | If not: train Phase 4 longer |

The key result is **D > B**. That's your proof the multi-agent training worked.

**Decision flowchart:**
```
D > B?
├── YES → Done. Push results.
└── NO  → Is C < B?
          ├── NO  → Simulator not strategic → extend Phase 3
          └── YES → Platform didn't adapt  → extend Phase 4 or lower lr
```

---

## 7. Parameters to Tweak and Where

### `run_training.sh` — top section, lines 24–28

```bash
TASK="easy"           # difficulty: "easy" | "medium" | "hard"
PHASE2_STEPS=500      # Round 1 training steps
PHASE3_STEPS=500      # Round 2 training steps
PHASE4_STEPS=1000     # Round 3 training steps (most important)
EVAL_EPISODES=50      # episodes per eval stage (higher = more reliable but slower)
```

---

### `training/train_platform.py` — inside `main()` function

| Parameter | Location | Default | When to change |
|---|---|---|---|
| `lr` in `grpo_setup_optimizer(model, lr=5e-6)` | ~line 120 | `5e-6` | Lower to `2e-6` if reward is unstable or oscillating |
| `batch_size` | `batch_size=4` near top of loop | `4` | Raise to `8` if loss is noisy; needs more GPU memory |
| `num_generations` in `grpo_step(...)` | inside training loop | `4` | Raise to `8` for stabler gradient signal |
| `drift_threshold` in `detect_reward_drift(...)` | inside training loop | `0.3` | Lower to `0.2` to trigger rollback sooner |

---

### `training/train_simulator.py` — inside `main()` function

| Parameter | Location | Default | When to change |
|---|---|---|---|
| `lr` in `grpo_setup_optimizer(model, lr=5e-6)` | ~line 120 | `5e-6` | Lower to `2e-6` if bluff rate explodes to 90%+ |
| `batch_size` | near top of loop | `4` | Raise to `8` for stabler bluff learning |

---

### `training/grpo_utils.py` — `grpo_step()` function

| Parameter | Default | When to change |
|---|---|---|
| `eps=1e-8` | `1e-8` | Raise to `1e-4` if you see NaN losses |
| `max_grad_norm=1.0` | `1.0` | Lower to `0.5` if loss spikes wildly |
| `kl_coeff=0.0` | `0.0` | Set to `0.01` if model drifts too far from base |

---

## 8. Common Problems and Fixes

### Platform completion rate stuck below 40%

Model is pricing randomly — too high or too low every time.

- **Fix 1**: Run Phase 2 longer. In `run_training.sh`: `PHASE2_STEPS=500` → `800`
- **Fix 2**: Stay on `--task easy` — don't move to medium yet
- **Fix 3**: Raise `batch_size` from `4` to `8` in `training/train_platform.py`

---

### D is not better than B (Phase 4 didn't help)

Platform trained in Round 3 but didn't beat its Round 1 self.

- **Fix 1**: More steps. In `run_training.sh`: `PHASE4_STEPS=1000` → `1500`
- **Fix 2**: Lower learning rate in `training/train_platform.py`: `lr=5e-6` → `lr=2e-6`
- **Fix 3**: Check the simulator is actually loaded in Phase 4 — confirm you passed `--simulator_ckpt` correctly

---

### C is not less than B (simulator not being strategic)

The simulator trained but isn't actually challenging the platform.

- **Fix 1**: Train simulator longer. `PHASE3_STEPS=500` → `800`
- **Fix 2**: Watch `bluff%` in Phase 3 logs. If it's 0% or 100%, the simulator collapsed — lower lr: `5e-6` → `2e-6` in `training/train_simulator.py`
- **Healthy target**: bluff rate 20–40%, collapse rate below 40%

---

### `grpo_loss` never drops below 0.5

Model is not learning — reward signal is too noisy.

- **Fix**: Raise `num_generations` in the `grpo_step(...)` call inside `training/train_platform.py` from `4` to `8`. More samples per update = stabler gradient.

---

### Training crashes / NaN loss

- **Fix 1**: In `training/grpo_utils.py`, raise `eps` from `1e-8` to `1e-4`
- **Fix 2**: Lower `max_grad_norm` from `1.0` to `0.5`
- **Fix 3**: Lower learning rate to `1e-6`

---

## 9. After Training — Push Results for Judges

```bash
# Plot training curves
python scripts/plot_training_curves.py

# Stage and push
git add data/training_metrics_*.json
git add data/final_evaluation.json
git add data/training_curves.png
git add checkpoints/phase2/ checkpoints/phase3/ checkpoints/phase4/
git commit -m "add training results and checkpoints"
git push
```

The README already has a Training Results section with a placeholder table and inline image reference (`data/training_curves.png`) — it will render automatically once the file is pushed.

---

## 10. Colab Notebook Alternative

If you prefer a notebook over scripts, open `notebooks/train_colab.ipynb` in Google Colab.  
Set runtime to **GPU → T4** before running. The notebook covers the full pipeline in 12 cells with inline plots and a before/after evaluation table.
