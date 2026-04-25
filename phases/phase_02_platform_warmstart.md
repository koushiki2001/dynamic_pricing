# Phase 2 — Platform LLM Warm Start (vs Rule-Based Simulator)

## Goal
Train the platform LLM using GRPO against the **existing rule-based simulator** (`simulator.py`). The rule-based simulator is kept unchanged — this is the simplest possible training setup. 

**Why start here:** GRPO needs successful trajectories to learn from. The rule-based simulator is deterministic and easy to exploit — this guarantees the platform model sees enough positive rewards before we introduce the strategic simulator LLM in Phase 3.

**Exit criteria:** Platform LLM achieves >50% completion rate on easy task, with average reward clearly above the untrained baseline.

---

## What Changes vs Current Codebase

| Component | Current State | After Phase 2 |
|---|---|---|
| `baselines/openai_policy.py` | Calls external API | Kept as-is (for comparison baseline) |
| `simulator.py` | Rule-based threshold check | **Unchanged** |
| `training/rollout.py` | Does not exist | **New** — connects env to GRPO |
| `training/train_platform.py` | Does not exist | **New** — GRPO training script |
| Platform model weights | None (API model) | Saved LoRA adapters after training |

---

## Step 2.1 — Build the Rollout Function

Create `training/rollout.py`.

This is the bridge between the OpenEnv environment and the GRPO trainer. It runs complete episodes and returns `(prompt, completion, reward)` triples — exactly what `GRPOTrainer` needs.

```python
"""Rollout: runs episodes and returns GRPO-compatible training samples."""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Tuple

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.models import Observation
from training.prompt_builders import build_platform_prompt


def parse_price_from_output(text: str, fallback: float) -> float:
    """Extract price float from model output. Returns fallback on failure."""
    text = text.strip()
    # Try JSON parse first
    try:
        match = re.search(r'\{.*?"price"\s*:\s*([\d.]+).*?\}', text)
        if match:
            return float(match.group(1))
        parsed = json.loads(text)
        return float(parsed["price"])
    except Exception:
        pass
    # Try bare number
    try:
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        if numbers:
            return float(numbers[0])
    except Exception:
        pass
    return fallback


def collect_platform_rollout(
    model,
    tokenizer,
    env: DynamicPricingEnv,
    task: str = "easy",
    max_new_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """
    Run one complete episode using the platform model.
    Returns list of (prompt, completion, reward) dicts — one per step.
    
    Reward is assigned only at the terminal step (sparse).
    Intermediate steps get step-shaping reward from reward.py.
    """
    obs_dict = env.reset(task=task)
    obs = obs_dict if isinstance(obs_dict, dict) else obs_dict.model_dump()

    samples = []
    done = False
    last_price = (obs["rider_quoted_price"] + obs["driver_quoted_price"]) / 2.0

    while not done:
        prompt = build_platform_prompt(
            Observation(**obs) if not isinstance(obs, Observation) else obs
        )

        # Generate from model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only new tokens
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        price = parse_price_from_output(completion, fallback=last_price)
        price = max(1.0, min(500.0, price))
        last_price = price

        # Step the environment
        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        step_result = env.step(action)

        reward = step_result.reward
        done   = step_result.done
        obs    = step_result.observation.model_dump() if hasattr(step_result.observation, "model_dump") else step_result.observation

        samples.append({
            "prompt":     prompt,
            "completion": completion,
            "reward":     reward,
            "price":      price,
            "done":       done,
        })

    return samples


def collect_batch(
    model,
    tokenizer,
    env: DynamicPricingEnv,
    task: str = "easy",
    batch_size: int = 16,
) -> Tuple[List[str], List[str], List[float]]:
    """
    Collect batch_size episodes and flatten into (prompts, completions, rewards).
    Only terminal step rewards are used for GRPO update.
    """
    prompts, completions, rewards = [], [], []

    for _ in range(batch_size):
        episode = collect_platform_rollout(model, tokenizer, env, task)
        if not episode:
            continue
        # Use only the terminal step for GRPO (has the meaningful reward signal)
        terminal = episode[-1]
        prompts.append(terminal["prompt"])
        completions.append(terminal["completion"])
        rewards.append(terminal["reward"])

    return prompts, completions, rewards
```

---

## Step 2.2 — Platform GRPO Training Script

Create `training/train_platform.py`.

```python
"""
Phase 2: Train platform LLM with GRPO against rule-based simulator.

Usage:
  python training/train_platform.py --task easy --steps 500 --batch_size 8
  python training/train_platform.py --task easy --steps 1000 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from ride_hailing_env.environment import DynamicPricingEnv
from training.model_loader import load_platform_model
from training.rollout import collect_batch


def reward_fn(completions: list, prompts: list = None, **kwargs) -> list[float]:
    """
    Reward function passed to GRPOTrainer.
    Rewards are pre-computed during rollout and stored in kwargs.
    This wrapper just returns them.
    """
    # Rewards injected via dataset during rollout collection
    return kwargs.get("rewards", [0.0] * len(completions))


def train(task: str, num_steps: int, batch_size: int, output_dir: str):
    print(f"[TRAIN] task={task}, steps={num_steps}, batch={batch_size}")

    # Load model
    model, tokenizer = load_platform_model()
    env = DynamicPricingEnv()

    # Collect initial baseline metrics
    print("[EVAL] Measuring baseline (untrained) performance...")
    baseline_rewards = []
    for _ in range(20):
        episode = collect_platform_rollout(model, tokenizer, env, task)
        if episode:
            baseline_rewards.append(episode[-1]["reward"])
    baseline_avg = sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0
    print(f"[EVAL] Baseline avg reward: {baseline_avg:.3f}")

    # Training configuration
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=100,
        max_completion_length=64,
        num_generations=4,           # GRPO samples 4 completions per prompt
        temperature=0.7,
        report_to="none",            # set to "wandb" if tracking
    )

    # Training loop — manual rollout collection + GRPO update
    step = 0
    reward_history = []

    while step < num_steps:
        prompts, completions, rewards = collect_batch(
            model, tokenizer, env, task, batch_size
        )
        reward_history.extend(rewards)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0

        if step % 50 == 0:
            window = reward_history[-100:] if len(reward_history) >= 100 else reward_history
            print(f"[STEP {step}] avg_reward={avg_reward:.3f}, "
                  f"window_avg={sum(window)/len(window):.3f}, "
                  f"completion_rate={sum(1 for r in rewards if r > 0)/len(rewards):.2f}")

        step += batch_size

    # Save LoRA adapters
    save_path = Path(output_dir) / "platform_lora"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[SAVE] Platform LoRA adapters saved to {save_path}")

    # Post-training evaluation
    print("[EVAL] Measuring post-training performance...")
    trained_rewards = []
    for _ in range(20):
        episode = collect_platform_rollout(model, tokenizer, env, task)
        if episode:
            trained_rewards.append(episode[-1]["reward"])
    trained_avg = sum(trained_rewards) / len(trained_rewards) if trained_rewards else 0

    print(f"\n{'='*50}")
    print(f"PHASE 2 RESULTS — Task: {task}")
    print(f"Baseline avg reward:  {baseline_avg:.3f}")
    print(f"Trained avg reward:   {trained_avg:.3f}")
    print(f"Improvement:          {trained_avg - baseline_avg:+.3f}")
    print(f"{'='*50}\n")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",       default="easy", choices=["easy","medium","hard"])
    parser.add_argument("--steps",      type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="checkpoints/phase2")
    args = parser.parse_args()

    train(args.task, args.steps, args.batch_size, args.output_dir)
```

---

## Step 2.3 — Metrics to Track

Log these at every 50 steps. They form the baseline benchmark all later phases are compared against.

| Metric | What it tells you |
|---|---|
| `avg_reward` | Overall training signal — should trend upward |
| `completion_rate` | % of episodes where deal was closed — most readable metric |
| `cancellation_rate` | % of episodes where a party ran out of patience — platform too aggressive |
| `timeout_rate` | % of episodes that hit max_steps — platform too passive |
| `avg_steps_to_close` | Efficiency — should decrease as model improves |
| `avg_profit_per_completed_ride` | Revenue quality — not just closing deals, but closing well |

Save these to `data/training_metrics_phase2.json` for the demo comparison.

---

## Step 2.4 — Curriculum Order

Run in this sequence — do not skip to hard:

```bash
# Start with easy — establish non-zero reward
python training/train_platform.py --task easy --steps 500

# Only proceed if completion_rate > 50% on easy
python training/train_platform.py --task medium --steps 300

# Only proceed if completion_rate > 30% on medium
python training/train_platform.py --task hard --steps 200
```

If completion rate on easy stays near 0 after 200 steps, the prompt format or generation parsing has a bug — fix before continuing.

---

## Step 2.5 — Save Baseline Snapshot

Before training starts, capture a baseline run and save it:

```bash
python scripts/evaluate_baselines.py --policy openai --episodes 20 --task easy
```

Save output to `data/baseline_snapshot_phase2.json`. This is the "before" number for the demo.

---

## Deliverables for Phase 2

- [ ] `training/rollout.py` — episode collection working end-to-end
- [ ] `training/train_platform.py` — GRPO training script running without errors
- [ ] Baseline metrics captured and saved to `data/`
- [ ] Platform LLM trained on easy task, completion rate > 50%
- [ ] LoRA adapters saved to `checkpoints/phase2/platform_lora/`
- [ ] Metrics log saved showing reward progression

---

## What Phase 2 Does NOT Do

- Does not change `simulator.py` — rule-based simulator is kept
- Does not train the simulator LLM — that is Phase 3
- Does not run medium/hard training unless easy is working
