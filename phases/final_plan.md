# 3-Hour Training Plan — Dynamic Pricing Hackathon (Colab T4)

**GPU:** Colab T4 (16 GB VRAM)  
**Models:** Platform = Qwen2.5-1.5B-Instruct 4-bit | Simulator = Qwen2.5-0.5B-Instruct 4-bit  
**Total budget: ~3 hours | Estimated active training: ~90–110 min | Buffer: ~50 min**

---

## T4 Hardware Profile

| Item | Value | Notes |
|------|-------|-------|
| VRAM | 16 GB | Both models load together without issue |
| Platform model VRAM | ~1.5 GB | Qwen2.5-1.5B, 4-bit + LoRA r=16 |
| Simulator model VRAM | ~0.5 GB | Qwen2.5-0.5B, 4-bit + LoRA r=8 |
| Peak (Round 3, both loaded) | ~5–6 GB | Well within limit |
| Generation speed (platform) | ~60–80 tok/s | With Unsloth inference mode |
| Generation speed (simulator) | ~150–200 tok/s | Smaller model, faster |
| Typical time per batch (platform, bs=8) | ~55–70 sec | 8 episodes × ~7 sec + backprop |
| Typical time per batch (simulator, bs=8) | ~20–30 sec | 8 episodes × ~3 sec + backprop |

---

## Timeline Overview (Colab T4)

| Round | What trains | Steps | Batches | Est. time | Cumulative |
|-------|-------------|-------|---------|-----------|------------|
| Setup (install + clone) | — | — | — | ~8 min | 8 min |
| 1 | Platform vs rule-based | 150 | ~19 | ~20–25 min | 30 min |
| 2 | Simulator vs platform_v0 | 120 | ~15 | ~8–10 min | 40 min |
| 3 | Platform vs simulator_v1 | 180 | ~23 | ~30–35 min | 75 min |
| Eval + Plot | All 4 stages | — | — | ~12 min | 87 min |
| Push to GitHub | — | — | — | ~3 min | 90 min |
| **Buffer for tweaks/re-runs** | | | | **~90 min** | |

> **Step math:** `batch_size=8`, so 150 steps = 19 gradient updates. Each update collects 8 full episodes then does one GRPO backward pass.

---

## T4-Specific Settings (Before You Start)

These are the recommended values for Colab T4. Change them before running Cell 12.

| Parameter | Where | T4 value | Default | Why |
|-----------|-------|----------|---------|-----|
| `PHASE2_STEPS` | Cell 12 | `150` | 500 | 3-hr budget |
| `PHASE3_STEPS` | Cell 27 | `120` | 1000 | T4 time limit |
| `PHASE4_STEPS` | Cell 29 | `180` | 1000 | T4 time limit |
| `--batch_size` | Cell 12 / 29 | `8` | 8 | Optimal for T4 |
| `PLATFORM_MAX_SEQ_LEN` | `training/model_loader.py:25` | `1024` | 1024 | Keep as-is |
| `SIMULATOR_MAX_SEQ_LEN` | `training/model_loader.py:26` | `512` | 512 | Keep as-is |
| `PLATFORM_LORA_R` | `training/model_loader.py:28` | `16` | 16 | Keep as-is |
| `SIMULATOR_LORA_R` | `training/model_loader.py:29` | `8` | 8 | Keep as-is |
| `lr` (platform) | `training/train_platform.py:135` | `5e-6` | `5e-6` | Hardcoded, see tweaks |
| `lr` (simulator) | `training/train_simulator.py:141` | `5e-6` | `5e-6` | Hardcoded, see tweaks |

> **Important:** `lr` and `batch_size` for the simulator are **not CLI args** — they are hardcoded in the training scripts. To change them you must edit the source file directly.

---

## Round 1 — Platform vs Rule-Based Simulator (Cells 12–26)

**Steps:** 150 | **Batches:** ~19 | **T4 time:** ~20–25 min

### Log format to watch
```
[STEP    8] avg=-0.812  win=-0.812  done=12%  cancel=62%  timeout=0%  loss=0.0142  (63.4s)
[STEP   16] avg=-0.405  win=-0.608  done=25%  cancel=50%  timeout=0%  loss=0.0118  (61.2s)
...
[STEP  150] avg=0.724   win=0.512   done=68%  cancel=18%  timeout=4%  loss=0.0043  (58.9s)
```

> The number in `(Xs)` is time per batch. On T4 expect **55–70 sec per batch**. If you see > 90 sec, something is wrong (see T4 Slow section below).

### Go/No-Go after Round 1
| Metric | Target | Action if missed |
|--------|--------|-----------------|
| `done` (completion%) | > 40% by step 100 | See tweaks below |
| `avg` reward at step 150 | > 0 | See tweaks below |
| `avg` trending upward | Yes (even slowly) | Continue — still learning |
| `cancel` rate | Falling over time | Good sign |

### Tweaks — Round 1 (all require editing source files)

**Symptom: `done` never rises above 10% after step 80**
- File: `training/train_platform.py`, line 135
- Change: `lr=5e-6` → `lr=2e-5`
- T4 cost: re-run Round 1, ~22 min

**Symptom: `avg` spikes up then crashes (oscillating reward)**
- File: `training/train_platform.py`, line 270 (CLI default)
- Change: `--batch_size` from `8` → `4` in Cell 12 (halves variance in GRPO)
- T4 cost: re-run Round 1, ~40 min (2× batches needed)

**Symptom: `avg` stays negative and flat from step 50 onwards**
- File: `training/train_platform.py`, line 135
- Change: `lr=5e-6` → `5e-5` (10× bigger push)
- T4 cost: re-run Round 1, ~22 min

**Symptom: `loss` printed as `nan` from step 1**
- File: `training/model_loader.py`, line 25
- Change: `PLATFORM_MAX_SEQ_LEN = 1024` → `512`
- This reduces numerical precision issues with very long prompts
- T4 cost: re-run Round 1, ~22 min

**Symptom: time per batch > 90 sec (T4 too slow)**
- Unsloth inference mode may not have activated for the frozen opponent
- Check Cell 15 has: `model = FastLanguageModel.for_inference(model)` (assigned)
- No re-run needed — just restart from Cell 12

---

## Round 2 — Simulator vs Platform_v0 (Cells 27–28)

**Steps:** 120 | **Batches:** ~15 | **T4 time:** ~8–10 min

> Simulator is 0.5B — much faster than the platform. Batches take ~20–30 sec on T4.

### Log format to watch
```
[STEP    8] sim_reward=-1.240  win=-1.240  bluff=12%  collapse=62%  loss=0.0231  (22.1s)
[STEP   56] sim_reward=0.340   win=0.120   bluff=28%  collapse=35%  loss=0.0089  (21.8s)
[STEP  120] sim_reward=0.810   win=0.650   bluff=34%  collapse=28%  loss=0.0054  (20.4s)
```

### Go/No-Go after Round 2
| Metric | Target | Action if missed |
|--------|--------|-----------------|
| `bluff` rate | 20–45% | See tweaks below |
| `collapse` rate | < 40% | See tweaks below |
| `sim_reward` trending up | Yes | Good — proceed |

### Tweaks — Round 2

**Symptom: `bluff` stuck at 0–5% (simulator always honest)**
- File: `training/train_simulator.py`, line 141
- Change: `lr=5e-6` → `lr=2e-5`
- T4 cost: re-run Round 2, ~10 min

**Symptom: `bluff` > 60% AND `collapse` > 50% (simulator too aggressive)**
- File: `training/train_simulator.py`, line 141
- Change: `lr=5e-6` → `lr=1e-6` (much slower learning)
- T4 cost: re-run Round 2, ~10 min

**Symptom: `collapse` > 60% but `bluff` is normal**
- Cell 27: change `PHASE3_STEPS=120` → `PHASE3_STEPS=80` (fewer steps before it over-learns)
- T4 cost: re-run Round 2, ~7 min

**Symptom: `sim_reward` never rises above baseline**
- This is acceptable for Round 2 — proceed if `bluff` and `collapse` are in range
- The simulator doesn't need high reward, just strategic behavior

---

## Round 3 — Platform vs Simulator_v1 (Cells 29–30)

**Steps:** 180 | **Batches:** ~23 | **T4 time:** ~30–35 min

> Both models are loaded simultaneously in Round 3. Platform trains, simulator is frozen.
> T4 has ~5–6 GB used out of 16 GB — no memory risk.

### Log format to watch
```
[STEP    8] avg=-0.310  win=-0.310  done=25%  cancel=62%  timeout=4%  loss=0.0198  (71.2s)
[STEP   88] avg=0.412   win=0.201   done=52%  cancel=28%  timeout=8%  loss=0.0067  (69.4s)
[STEP  180] avg=0.891   win=0.744   done=71%  cancel=12%  timeout=6%  loss=0.0031  (68.8s)
```

> Batches are slower in Round 3 (~65–75 sec) because both LLMs generate per episode.

### Go/No-Go after Round 3
| Metric | Target | Meaning |
|--------|--------|---------|
| `avg` at step 180 > Round 1 `avg` at step 150 | Yes | Platform improved against harder opponent |
| `done` > 55% | Yes | Platform reliably closes deals |
| Stage D reward > Stage B reward (Cell 31) | Yes | **Key judge metric** |

### Tweaks — Round 3

**Symptom: `avg` immediately drops below 0 and stays (catastrophic forgetting)**
- The platform is "forgetting" Round 1 learning against the harder opponent
- File: `training/train_platform.py`, line 135
- Change: `lr=5e-6` → `lr=1e-6` (very slow update to preserve earlier weights)
- T4 cost: re-run Round 3, ~32 min

**Symptom: `avg` plateaus after step 100, no improvement**
- Cell 29: change `PHASE4_STEPS=180` → `PHASE4_STEPS=240`
- T4 cost: +35 min on top of original (~67 min total for Round 3)
- Only do this if you have > 50 min of buffer remaining

**Symptom: `avg` is positive and rising but `done` stays low**
- Platform is earning partial rewards but not closing deals
- File: `training/train_platform.py`, line 135
- Change: `lr=5e-6` → `lr=1e-5`
- T4 cost: re-run Round 3, ~32 min

**Symptom: time per batch > 90 sec in Round 3 (both models loading slowly)**
- Check Cell 19: `model = FastLanguageModel.for_inference(model)` must be assigned
- Check Cell 27: simulator loaded with `enable_inference_mode()`
- These are already in the notebook but verify they ran without error

---

## Evaluation & Plots (Cells 31–34)

**T4 time:** ~12 min (20 eval episodes × 4 stages = 80 rollouts)

Run Cell 31 → Cell 33 immediately after Round 3 completes.

### Target evaluation table

| Stage | JSON key | What it measures | T4 target |
|-------|----------|-----------------|-----------|
| A | `A_untrained_vs_rulebased` | Untrained platform | completion ~10–20% |
| B | `B_platformv0_vs_rulebased` | After Round 1 | completion > 50% |
| C | `C_platformv0_vs_simv1` | Round 1 platform vs trained sim | May drop vs B (normal) |
| D | `D_platformv1_vs_simv1` | After Round 3 | **Must beat B** |

**The judge metric:** `D.avg_reward > B.avg_reward`  
**Secondary:** `D.completion_rate > B.completion_rate`

### Reading the 4-panel plot

| Panel | What to look for | Good sign | Bad sign |
|-------|-----------------|-----------|----------|
| Top-left: Training curve | Reward across all 3 rounds | Upward trend, may dip between rounds | Flat or falling in Round 3 |
| Top-right: Completion rate | % of deals closed | Rising from R1 to R3 | Stuck below 40% |
| Bottom-left: Bluff rate | Simulator behavior | 20–45% zone | 0% or > 70% |
| Bottom-right: Stage bars A→D | The 4-stage comparison | Bar D taller than Bar B | D ≤ B |

---

## Fallback Plan (Time-Based)

### If you have 90 min left after a Round 1 re-run
- Skip Round 2 entirely
- Set `PHASE4_STEPS=100` in Cell 29, run Round 3
- You still show D > B improvement, just less dramatic
- T4 time: ~12 min for Round 3, ~12 min eval = 24 min total

### If you have 60 min left after Round 1 issues
- Do not run Round 3
- Run eval (Cell 31) on stages A and B only
- Shows model learned something vs untrained baseline
- T4 time: ~6 min for 2-stage eval + plot

### If Colab runtime disconnects mid-training
- Auto-save fires every 50 steps → `data/metrics_checkpoint_*.json` (safe)
- Rolling checkpoint saved every 200 steps → `checkpoints/last_stable/`
- Restart: re-run setup cells 1–11, then resume from the last completed round's cell
- Reduce step count: if Round 1 completed 100/150 steps, set `PHASE2_STEPS=50` to finish it

### If T4 gives CUDA OOM (unlikely but possible)
- File: `training/model_loader.py`, line 25
- Change: `PLATFORM_MAX_SEQ_LEN = 1024` → `512`
- This halves the attention memory footprint
- Push change to GitHub first, then re-clone in Cell 3

---

## Final Push (Cell 37)

After Cells 31–34 complete:
1. Set `GITHUB_TOKEN` in Cell 37 (your PAT with `repo` scope)
2. Run Cell 37
3. Files pushed:
   - `dynamic_pricing/data/colab_evaluation.json`
   - `dynamic_pricing/data/final_evaluation.json`
   - `dynamic_pricing/data/training_curves.png`
   - `dynamic_pricing/checkpoints/phase2/` (platform_lora)
   - `dynamic_pricing/checkpoints/phase3/` (simulator_lora)
   - `dynamic_pricing/checkpoints/phase4/` (platform_lora v2)
4. Verify at: `https://github.com/koushiki2001/dynamic_pricing/tree/milan_v1/dynamic_pricing/data`

---

## Complete T4 Parameter Reference

### Parameters you can change via CLI (notebook cells)

| Parameter | Cell | Default | Tweak range | Effect |
|-----------|------|---------|-------------|--------|
| `PHASE2_STEPS` | 12 | `150` | 100–300 | Round 1 length |
| `PHASE3_STEPS` | 27 | `120` | 80–200 | Round 2 length |
| `PHASE4_STEPS` | 29 | `180` | 100–300 | Round 3 length |
| `--batch_size` | 12, 29 | `8` | 4–16 | Samples per GRPO update |

### Parameters that require editing source files

| Parameter | File | Line | Default | Safe range on T4 |
|-----------|------|------|---------|-----------------|
| Platform `lr` | `training/train_platform.py` | 135 | `5e-6` | `1e-6` – `5e-5` |
| Simulator `lr` | `training/train_simulator.py` | 141 | `5e-6` | `1e-6` – `2e-5` |
| `PLATFORM_MAX_SEQ_LEN` | `training/model_loader.py` | 25 | `1024` | 256–1024 |
| `SIMULATOR_MAX_SEQ_LEN` | `training/model_loader.py` | 26 | `512` | 256–512 |
| `PLATFORM_LORA_R` | `training/model_loader.py` | 28 | `16` | 8–32 (32 uses more VRAM) |
| `SIMULATOR_LORA_R` | `training/model_loader.py` | 29 | `8` | 4–16 |
| Simulator `batch_size` | `training/train_simulator.py` | 152, 162 | `8` | 4–16 |

> **How to edit source files in Colab:** After Cell 3 clones the repo, use `!sed -i 's/lr=5e-6/lr=2e-5/' /content/dynamic_pricing_env/dynamic_pricing/training/train_platform.py` or open the Files panel on the left and edit directly.

---

## Quick Decision Flowchart

```
START
  │
  ├─ Run Round 1 (150 steps, ~22 min)
  │     done > 40% by step 100?
  │     ├─ YES → proceed to Round 2
  │     └─ NO  → avg trending up?
  │               ├─ YES → add 50 steps (PHASE2_STEPS=200), continue
  │               └─ NO  → raise lr to 2e-5, re-run (~22 min)
  │
  ├─ Run Round 2 (120 steps, ~10 min)
  │     bluff 20-45% AND collapse < 40%?
  │     ├─ YES → proceed to Round 3
  │     └─ NO  → bluff=0: raise lr to 2e-5, re-run (~10 min)
  │               bluff>60%: lower lr to 1e-6, re-run (~10 min)
  │               collapse>60%: reduce PHASE3_STEPS=80, re-run (~7 min)
  │
  ├─ Run Round 3 (180 steps, ~32 min)
  │     avg Round 3 > avg Round 1?
  │     ├─ YES → run eval + plot
  │     └─ NO  → add steps (PHASE4_STEPS=240) if > 35 min buffer
  │               OR lower lr to 1e-6 and re-run (~32 min)
  │
  └─ Run Eval (Cell 31) + Plot (Cell 33) + Push (Cell 37)
        D > B?
        ├─ YES → done ✓
        └─ NO  → add 50 steps to Round 3, re-run (~12 min), re-eval
```
