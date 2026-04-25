# Phase 4 — Full Multi-Agent Training Cycle

## Goal
Run the complete sequential training cycle: train platform against trained simulator, then optionally fine-tune simulator against the smarter platform. Produce the final model checkpoints used for evaluation and demo.

**Exit criteria:** Platform LLM completion rate ≥ 70% on easy task when facing the strategic simulator, with average reward higher than the Phase 2 (rule-based opponent) baseline.

---

## The Full Training Arc

```
Phase 2: platform_v0  trained vs  rule-based simulator
              ↓
Phase 3: simulator_v1 trained vs  frozen platform_v0
              ↓
Phase 4: platform_v1  trained vs  frozen simulator_v1   ← THIS PHASE
              ↓ (optional)
Phase 4b: simulator_v2 fine-tuned vs frozen platform_v1
```

Each arrow represents one GRPO training run. The non-stationarity problem is controlled by always freezing one model while training the other.

---

## Step 4.1 — Train Platform v1 Against Simulator v1

Create `training/train_platform_v2.py` (or extend `train_platform.py` with a `--simulator_ckpt` argument).

Key difference from Phase 2: the environment now uses the simulator LLM instead of rule-based logic.

```python
"""
Phase 4: Re-train platform LLM against frozen simulator_v1.

Usage:
  python training/train_platform.py \
    --task easy \
    --steps 1000 \
    --simulator_ckpt checkpoints/phase3/simulator_lora \
    --platform_ckpt  checkpoints/phase2/platform_lora \
    --output_dir     checkpoints/phase4
"""

import argparse
from pathlib import Path
from training.model_loader import load_platform_model, load_simulator_model
from training.simulator_agent import SimulatorAgent
from training.rollout import collect_multiagent_rollout_platform_only
from ride_hailing_env.environment import DynamicPricingEnv


def train(task, num_steps, simulator_ckpt, platform_ckpt, output_dir):
    # Load simulator — FROZEN
    sim_model, sim_tokenizer = load_simulator_model(lora_path=simulator_ckpt)
    sim_model.eval()
    simulator_agent = SimulatorAgent(sim_model, sim_tokenizer)

    # Load platform — TRAINABLE, starting from phase2 checkpoint
    platform_model, platform_tokenizer = load_platform_model(lora_path=platform_ckpt)

    env = DynamicPricingEnv()

    # Measure platform_v0 performance against simulator_v1 (pre-training)
    print("[EVAL] Platform v0 vs Simulator v1 (before phase 4 training)...")
    pre_rewards = []
    for _ in range(30):
        result = collect_multiagent_rollout(
            platform_model, platform_tokenizer, simulator_agent, env, task
        )
        if result["platform_samples"]:
            pre_rewards.append(result["platform_samples"][-1]["reward"])
    pre_avg = sum(pre_rewards) / len(pre_rewards) if pre_rewards else 0
    print(f"[EVAL] Pre-training avg reward vs strategic simulator: {pre_avg:.3f}")

    # GRPO training on platform only
    step = 0
    reward_history = []

    while step < num_steps:
        # Collect batch of episodes with simulator_v1 as opponent
        batch_rewards = []
        batch_prompts, batch_completions = [], []

        for _ in range(8):
            result = collect_multiagent_rollout(
                platform_model, platform_tokenizer, simulator_agent, env, task
            )
            p_samples = result["platform_samples"]
            if p_samples:
                terminal = p_samples[-1]
                batch_prompts.append(terminal["prompt"])
                batch_completions.append(terminal["completion"])
                batch_rewards.append(terminal["reward"])

        # GRPO update on platform
        # [GRPOTrainer step here]

        reward_history.extend(batch_rewards)
        avg = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0

        if step % 50 == 0:
            window = reward_history[-100:]
            print(f"[STEP {step}] avg_reward={avg:.3f}, "
                  f"100-window={sum(window)/len(window):.3f}")

        step += 8

    # Post-training evaluation
    print("[EVAL] Platform v1 vs Simulator v1 (after phase 4 training)...")
    post_rewards = []
    for _ in range(30):
        result = collect_multiagent_rollout(
            platform_model, platform_tokenizer, simulator_agent, env, task
        )
        if result["platform_samples"]:
            post_rewards.append(result["platform_samples"][-1]["reward"])
    post_avg = sum(post_rewards) / len(post_rewards) if post_rewards else 0

    print(f"\n{'='*50}")
    print(f"PHASE 4 RESULTS — Platform v1 vs Simulator v1")
    print(f"Pre-training (v0 vs strategic):  {pre_avg:.3f}")
    print(f"Post-training (v1 vs strategic): {post_avg:.3f}")
    print(f"Improvement:                     {post_avg - pre_avg:+.3f}")
    print(f"{'='*50}\n")

    # Save platform_v1
    save_path = Path(output_dir) / "platform_v1_lora"
    platform_model.save_pretrained(str(save_path))
    platform_tokenizer.save_pretrained(str(save_path))
    print(f"[SAVE] Platform v1 LoRA saved to {save_path}")
```

---

## Step 4.2 — Extend to Medium and Hard Tasks

Once easy task is stable, repeat the cycle for medium then hard.

```bash
# Easy cycle (already done in phase 2 + 3 + 4)
python training/train_platform.py --task easy --simulator_ckpt checkpoints/phase3/simulator_lora ...

# Medium cycle
python training/train_platform.py --task medium --steps 500 \
  --simulator_ckpt checkpoints/phase3/simulator_lora \
  --platform_ckpt  checkpoints/phase4/platform_v1_lora \
  --output_dir     checkpoints/phase4_medium

# Hard cycle (only if medium completion rate > 40%)
python training/train_platform.py --task hard --steps 300 \
  --simulator_ckpt checkpoints/phase3/simulator_lora \
  --platform_ckpt  checkpoints/phase4_medium/platform_v1_lora \
  --output_dir     checkpoints/phase4_hard
```

---

## Step 4.3 — Optional: Fine-Tune Simulator v2

If time permits, one more simulator training cycle against platform_v1 produces a more adversarial opponent and a richer training story.

```bash
python training/train_simulator.py \
  --task easy \
  --steps 500 \
  --platform_ckpt checkpoints/phase4/platform_v1_lora \
  --output_dir    checkpoints/phase4b
```

This is optional — skip if time-constrained. The phase 2→3→4 arc already demonstrates the full RL loop.

---

## Step 4.4 — Comprehensive Evaluation Script

Create `training/evaluate.py` to produce the final comparison table used in the demo.

```python
"""
Final evaluation across all training stages.
Produces the before/after comparison for the presentation.
"""

import json
from pathlib import Path
from training.model_loader import load_platform_model, load_simulator_model
from training.simulator_agent import SimulatorAgent
from training.rollout import collect_platform_rollout, collect_multiagent_rollout
from ride_hailing_env.environment import DynamicPricingEnv


def evaluate_stage(
    platform_model, platform_tokenizer,
    simulator_agent,        # None = use rule-based
    env, task, n_episodes=50
):
    rewards, completions, steps = [], [], []

    for _ in range(n_episodes):
        if simulator_agent is None:
            episode = collect_platform_rollout(platform_model, platform_tokenizer, env, task)
        else:
            result  = collect_multiagent_rollout(platform_model, platform_tokenizer,
                                                 simulator_agent, env, task)
            episode = result["platform_samples"]

        if episode:
            final = episode[-1]
            rewards.append(final["reward"])
            completions.append(1 if final["reward"] > 0 else 0)

    return {
        "avg_reward":       sum(rewards) / len(rewards) if rewards else 0,
        "completion_rate":  sum(completions) / len(completions) if completions else 0,
        "n_episodes":       n_episodes,
    }


def run_full_evaluation(output_path="data/final_evaluation.json"):
    env  = DynamicPricingEnv()
    results = {}

    for task in ["easy", "medium", "hard"]:
        results[task] = {}

        # Stage A: untrained platform vs rule-based simulator
        untrained_platform, tok = load_platform_model()   # base model, no LoRA
        results[task]["A_untrained_vs_rulebased"] = evaluate_stage(
            untrained_platform, tok, None, env, task
        )

        # Stage B: platform_v0 vs rule-based simulator (Phase 2 result)
        p0_model, p0_tok = load_platform_model(lora_path="checkpoints/phase2/platform_lora")
        results[task]["B_platformv0_vs_rulebased"] = evaluate_stage(
            p0_model, p0_tok, None, env, task
        )

        # Stage C: platform_v0 vs simulator_v1 (Phase 3 disruption)
        s1_model, s1_tok = load_simulator_model(lora_path="checkpoints/phase3/simulator_lora")
        sim_v1 = SimulatorAgent(s1_model, s1_tok)
        results[task]["C_platformv0_vs_simv1"] = evaluate_stage(
            p0_model, p0_tok, sim_v1, env, task
        )

        # Stage D: platform_v1 vs simulator_v1 (Phase 4 recovery)
        p1_model, p1_tok = load_platform_model(lora_path="checkpoints/phase4/platform_v1_lora")
        results[task]["D_platformv1_vs_simv1"] = evaluate_stage(
            p1_model, p1_tok, sim_v1, env, task
        )

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'STAGE':<35} {'EASY':>8} {'MEDIUM':>8} {'HARD':>8}")
    print(f"{'='*70}")
    stages = [
        ("A: Untrained vs Rule-based",     "A_untrained_vs_rulebased"),
        ("B: Platform v0 vs Rule-based",   "B_platformv0_vs_rulebased"),
        ("C: Platform v0 vs Simulator v1", "C_platformv0_vs_simv1"),
        ("D: Platform v1 vs Simulator v1", "D_platformv1_vs_simv1"),
    ]
    for label, key in stages:
        row = f"{label:<35}"
        for task in ["easy", "medium", "hard"]:
            cr = results[task].get(key, {}).get("completion_rate", 0)
            row += f"  {cr*100:5.1f}%"
        print(row)
    print(f"{'='*70}\n")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Full evaluation saved to {output_path}")


if __name__ == "__main__":
    run_full_evaluation()
```

---

## Step 4.5 — Checkpoint Directory Structure

```
checkpoints/
├── phase2/
│   └── platform_lora/          # Platform v0 — trained vs rule-based
├── phase3/
│   └── simulator_lora/         # Simulator v1 — trained vs platform v0
├── phase4/
│   └── platform_v1_lora/       # Platform v1 — trained vs simulator v1
└── phase4b/                    # Optional
    └── simulator_v2_lora/      # Simulator v2 — fine-tuned vs platform v1

data/
├── baseline_snapshot_phase2.json   # Untrained platform metrics
├── training_metrics_phase2.json    # Phase 2 reward curve
├── training_metrics_phase3.json    # Phase 3 simulator reward curve
├── training_metrics_phase4.json    # Phase 4 platform reward curve
└── final_evaluation.json           # Full before/after comparison table
```

---

## Expected Results Table

| Stage | Easy Completion | Medium Completion | Hard Completion |
|---|---|---|---|
| A: Untrained vs Rule-based | ~25% | ~15% | ~8% |
| B: Platform v0 vs Rule-based | ~60% | ~40% | ~20% |
| C: Platform v0 vs Simulator v1 | ~40% | ~25% | ~12% |
| D: Platform v1 vs Simulator v1 | ~70% | ~50% | ~28% |

Row C dropping below Row B is the **disruption moment** — proof the simulator LLM is genuinely strategic. Row D recovering above Row B is the **adaptation story** — proof the platform RL loop is working.

---

## Deliverables for Phase 4

- [ ] `training/train_platform.py` updated with `--simulator_ckpt` argument
- [ ] `training/evaluate.py` — comprehensive evaluation script
- [ ] Platform v1 LoRA saved to `checkpoints/phase4/`
- [ ] `data/final_evaluation.json` — full 4-stage comparison table
- [ ] Easy task completion rate ≥ 70% for platform_v1 vs simulator_v1
