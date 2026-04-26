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

## 10. Colab Notebook — Step by Step

### What to upload to Colab

**Only the notebook file.** Do NOT upload the entire project.

The notebook clones the repo from GitHub automatically (Cell 3) — so Colab gets all the code it needs.

```
Your GitHub repo (has all the code)
        ↓  Cell 3 (code): git clone branch feature/koushiki_round2_v1
Colab machine (temporary)
        ↓  Cells run top to bottom: train, save results to repo's data/ folder
        ↓  Last cell: git push results back
Your GitHub repo (now has training results too)
```

---

### How to run on Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `notebooks/train_colab.ipynb`
3. Runtime → Change runtime type → **GPU → T4**
4. Run all cells top to bottom (Runtime → Run all, or Shift+Enter each cell)

No other setup needed — the repo URL and branch are already hardcoded in Cell 3.

---

### What each cell does

| Cell | Section | What it does |
|---|---|---|
| 1 (md) | — | Title and overview |
| 2 (code) | 1. Install | Installs Unsloth + OpenEnv + deps |
| 3 (code) | — | Clones `feature/koushiki_round2_v1` branch, sets sys.path, verifies import |
| 4 (md) | — | Section header |
| 5 (code) | 2. Verify Env | Resets env, runs a test step, confirms everything loads |
| 6 (md) | — | Section header |
| 7 (code) | 3. Load Model | Loads Qwen2.5-1.5B-Instruct in 4-bit + LoRA r=16 via Unsloth |
| 8 (md) | — | Section header |
| 9 (code) | 4. Prompt Builder | Defines `build_prompt()` and `parse_price()` |
| 10 (md) | — | Section header |
| 11 (code) | 5. Rollout | Defines `collect_episode()` and `collect_batch()` |
| 12 (md) | — | Section header |
| 13 (code) | 6. GRPO | Defines `grpo_step()` + AdamW optimizer |
| 14 (md) | — | Section header |
| 15 (code) | 7. Baseline Eval | Sets `TASK`, `NUM_STEPS`, `BATCH_SIZE` — runs untrained eval, saves `baseline_reward` |
| 16 (md) | — | Section header |
| 17 (code) | 8. Training Loop (Round 1) | **Platform vs rule-based** — auto-saves metrics every 50 steps |
| 18 (md) | — | Section header |
| 19 (code) | 9. Post Eval | Eval after Round 1, prints before/after table |
| 20 (md) | — | Section header |
| 21 (code) | 9b. Save Eval | Saves `data/colab_evaluation.json` (Stage A + B) |
| 22 (md) | — | Section header |
| 23 (code) | 10. Plot (Round 1) | 3-panel curves for platform only → `data/training_curves.png` |
| 24 (md) | — | Section header |
| 25 (code) | 11. Save Checkpoint | Saves platform_v0 LoRA to `/content/platform_lora_colab/` |
| 26 (md) | — | Section header |
| 27 (code) | 14. Round 2 | **Simulator trains** against frozen platform_v0 |
| 28 (md) | — | Section header |
| 29 (code) | 15. Round 3 | **Platform re-trains** against frozen simulator_v1 |
| 30 (md) | — | Section header |
| 31 (code) | 16. Full Eval | Runs 4-stage A/B/C/D evaluation, saves `data/final_evaluation.json` |
| 32 (md) | — | Section header |
| 33 (code) | 17. Full 4-Panel Plot | All 4 panels (platform + simulator) → overwrites `data/training_curves.png` |
| 34 (md) | — | Section header |
| 35 (code) | 12. Demo | Runs one demo episode with trained platform_v1 |
| 36 (md) | — | Section header |
| 37 (code) | 13. Push to GitHub | **Pushes all results** — fill in `GITHUB_TOKEN` first |

---

### What auto-saves during training (no action needed)

The training loop (Cell 17) writes to disk every 50 steps — so if Colab crashes you don't lose everything:

| File | Location in repo | Saves when |
|---|---|---|
| `training_metrics_platform_easy.json` | `dynamic_pricing/data/` | Every 50 steps (Round 1 and Round 3) |
| `training_metrics_simulator_easy.json` | `dynamic_pricing/data/` | Every 50 steps (Round 2) |
| `colab_evaluation.json` | `dynamic_pricing/data/` | After Round 1 post-eval (Cell 21) |
| `final_evaluation.json` | `dynamic_pricing/data/` | After 4-stage eval (Cell 31) |
| `training_curves.png` | `dynamic_pricing/data/` | After Round 1 plot (Cell 23), overwritten after full plot (Cell 33) |
| LoRA adapter (platform_v0) | `/content/platform_lora_colab/` | After Cell 25 |

---

### Pushing results back to GitHub (Cell 37)

Before running Cell 37, fill in your GitHub token:

```python
GITHUB_TOKEN = ""   # ← paste your token here
```

Get a token: GitHub → Settings → Developer Settings → Personal Access Tokens → Generate new (classic) → tick `repo` scope.

**Security tip:** Never save the token into the notebook permanently. Either:
- Paste it fresh each Colab session and clear it before saving, or
- Use Colab Secrets: Tools → Secrets → add `GITHUB_TOKEN`, then read it with:

```python
from google.colab import userdata
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
```

Cell 37 pushes these 5 files to `feature/koushiki_round2_v1`:
- `dynamic_pricing/data/training_metrics_platform_easy.json`
- `dynamic_pricing/data/training_metrics_simulator_easy.json`
- `dynamic_pricing/data/colab_evaluation.json`
- `dynamic_pricing/data/final_evaluation.json`
- `dynamic_pricing/data/training_curves.png`

Checkpoints (~500 MB) are **not** pushed to GitHub — too large. Upload those to HF Hub separately if needed:

```python
# Uncomment in Cell 25 to upload checkpoints to HF Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/content/platform_lora_colab",
    repo_id="<your-org>/dynamic-pricing-checkpoints",
    repo_type="model",
)
```

---

## 11. Reading Results and Knowing What to Tweak

### The master evaluation table — what to look for

After Cell 31 runs, you see:

```
Stage  Setup                                   Reward    Done%
A      Untrained vs Rule-based                 +0.18     28%
B      Platform_v0 vs Rule-based               +0.52     64%
C      Platform_v0 vs Simulator_v1             +0.31     45%
D      Platform_v1 vs Simulator_v1             +0.61     71%
```

Each row tells a specific story:

| What you see | What it means | What to do |
|---|---|---|
| A is low (reward < 0.3, done% < 35%) | Good — this is the expected untrained baseline | Nothing |
| B > A | Round 1 worked | Nothing |
| B is low (reward < 0.4, done% < 50%) | Platform didn't learn basics | Re-run Round 1 with more steps or lower lr |
| C < B | Simulator is genuinely strategic | Good — this is what you want |
| C >= B | Simulator didn't learn to be strategic | Re-run Round 2 with more steps |
| D > B | Platform adapted to the harder opponent | **This is the key result — you're done** |
| D <= B | Platform didn't improve in Round 3 | Re-run Round 3 with more steps or lower lr |
| D > C but D < B | Partial recovery | More Round 3 steps usually fixes this |

---

### Reading the 4-panel training curves chart

**Panel 1 — Platform Reward (top left)**

```
What healthy looks like:
  ↗ reward rises steadily from left to right
  gray dashed line (baseline) gets left behind early
  green dashed line (post-training target) is reached or exceeded

What each shape means:
  Flat line from the start  → model not learning, lower lr or raise num_generations
  Rising then crashing       → reward drift, re-run from last_stable checkpoint
  Spiky up/down with no trend → batch too small, raise BATCH_SIZE from 4 to 8
  Rising then flat plateau   → converged, stop here or move to harder task
```

**Panel 2 — Platform Episode Outcomes (top right)**

```
3 lines: completion% (green), cancellation% (red), timeout% (orange)

What healthy looks like:
  Green rises above 65%
  Red and orange both fall below 20%

What each shape means:
  Green stuck below 40%          → platform still pricing randomly, more Round 1 steps
  Red very high (> 50%)          → prices consistently too low or too high, check reward fn
  Timeout very high (> 50%)      → model is repeating the same price, check anti-repeat penalty
  Green rises but red also rises → model found a narrow price that completes but is unstable
```

**Panel 3 — Simulator Reward (bottom left)**

```
What healthy looks like:
  ↗ reward rises above the rule-based baseline (gray dashed)
  Steady increase, not spiky

What each shape means:
  Flat at baseline level    → simulator not learning, lower lr in train_simulator.py
  Rising very fast then flat → simulator overfit to platform_v0's patterns (fine for hackathon)
  Falling below baseline    → collapse, simulator being too aggressive, lower lr
```

**Panel 4 — Simulator Bluff & Collapse Rates (bottom right)**

```
2 lines: bluff_rate (purple), collapse_rate (red)
Shaded zone: healthy bluff range 20-50%

What healthy looks like:
  Purple line settles inside the shaded zone (20-50%)
  Red line stays below the danger line (60%)

What each shape means:
  Purple = 0% throughout      → simulator never bluffs, not learning strategy
                                 → lower lr to 2e-6 in train_simulator.py
  Purple = 100% throughout    → simulator always bluffs, everything collapses
                                 → lower lr or raise batch_size
  Red keeps climbing > 60%    → simulator too aggressive, rides failing constantly
                                 → lower lr, add fewer steps
  Purple 20-40%, red < 30%    → perfect, stop training simulator here
```

---

### Step-by-step decision tree after each round

**After Round 1 (Cells 17-21):**

```
Stage B reward > Stage A reward AND completion% > 50%?
├── YES → Round 1 worked. Proceed to Round 2.
└── NO  → What is completion% ?
          < 30%: Platform still random. Double NUM_STEPS in Cell 15, re-run Cells 17-21.
          30-50%: Marginal learning. Lower lr: 5e-6 → 2e-6 in Cell 13, re-run.
```

**After Round 2 (Cell 27):**

```
bluff_rate settled between 20-40% AND collapse_rate < 40%?
├── YES → Simulator is strategic. Proceed to Round 3.
└── NO  → bluff_rate = 0%:   Simulator not learning. Lower PHASE3_STEPS doesn't help —
                               lower lr in train_simulator.py: 5e-6 → 2e-6, re-run Cell 27.
          bluff_rate = 100%: Simulator too aggressive. Same fix — lower lr.
          collapse_rate > 60%: Simulator destroying all rides. Lower PHASE3_STEPS or lr.
```

**After Round 3 (Cell 29):**

```
Stage D reward > Stage B reward AND completion% > 65%?
├── YES → Multi-agent training worked. Run full eval (Cell 31) and plot (Cell 33).
└── NO  → D > C but D < B:  Partial recovery. Raise PHASE4_STEPS by 100, re-run Cell 29.
          D <= C:            Platform not adapting at all.
                              Try: lower lr 5e-6 → 2e-6 in train_platform.py, re-run.
                              Or:  raise PHASE4_STEPS to 500+ and re-run.
```

---

### Quick parameter reference — what to change for each symptom

| Symptom | File | Parameter | Change |
|---|---|---|---|
| Platform reward not rising |  Cell 15 |  | Raise to 300 |
| Platform reward unstable / spiky | Cell 13 () |  | Lower  →  |
| Platform reward rising then crashing | Cell 13 |  in  | Lower to  |
| Completion% stuck below 40% | Cell 15 |  | Raise  →  |
| Simulator bluff_rate = 0% |  |  in  | Lower  →  |
| Simulator bluff_rate = 100% |  |  | Lower  →  |
| Round 2 collapse_rate > 60% |  Cell 27 |  | Lower to 100 |
| D not better than B | Cell 29 |  | Raise to 400 |
| NaN loss | Cell 13 |  in  | Raise  →  |
| Colab runs out of memory | Cell 15 |  | Lower to  |
| Training too slow on T4 | Cell 15 |  | Lower to 100 to test, then scale |

---

### Time estimates for each round on T4 GPU

| Round | Steps | T4 time | What to watch |
|---|---|---|---|
| Round 1 (platform) | 150 steps | ~35 min | Panel 1 rising, completion% > 50% |
| Round 2 (simulator) | 150 steps | ~25 min | bluff_rate 20-40%, collapse_rate < 40% |
| Round 3 (platform) | 200 steps | ~45 min | Panel 1 exceeds Round 1 final value |
| Full eval | 20 ep/stage | ~10 min | D > B |
| **Total** | | **~2 hrs** | |

For full training (500/500/1000 steps): ~5-6 hrs on T4.
