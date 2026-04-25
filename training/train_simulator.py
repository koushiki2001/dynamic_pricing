"""Simulator LLM training with GRPO (Phase 3).

Trains the simulator LLM to learn strategic bluffing behaviour against
a frozen platform_v0 checkpoint. The simulator is rewarded for surplus
earned — it is NOT adversarial. It wants the deal to close but on
better terms.

Usage:
    python training/train_simulator.py \\
        --task easy \\
        --steps 1000 \\
        --platform_ckpt checkpoints/phase2/platform_lora \\
        --output_dir    checkpoints/phase3

Monitoring signals specific to the simulator:
    - bluff_rate:        fraction of eligible steps where it bluffed
    - collapse_rate:     fraction of episodes ending in cancellation
    - avg_sim_reward:    surplus earned per episode
    - honest_accept_rate: how often it accepts when it could bluff

A healthy simulator has bluff_rate 20-50% and collapse_rate < 40%.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from ride_hailing_env.environment import DynamicPricingEnv
from training.model_loader import load_platform_model, load_simulator_model, enable_inference_mode
from training.rollout import (
    collect_multiagent_rollout,
    collect_simulator_batch,
    detect_reward_drift,
    inspect_generations,
)
from training.simulator_agent import SimulatorAgent
from training.grpo_utils import grpo_setup_optimizer, grpo_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_simulator(
    platform_model,
    platform_tokenizer,
    simulator_agent: SimulatorAgent,
    env: DynamicPricingEnv,
    task: str,
    n: int = 20,
) -> dict:
    """Run n episodes and return simulator-specific metrics."""
    sim_rewards, bluffs, collapses, totals = [], 0, 0, 0

    for _ in range(n):
        result = collect_multiagent_rollout(
            platform_model, platform_tokenizer, simulator_agent, env, task
        )
        s_samples = result["simulator_samples"]
        p_samples = result["platform_samples"]
        if not s_samples:
            continue
        totals += 1

        terminal_sim = s_samples[-1]
        if terminal_sim["reward"] is not None:
            sim_rewards.append(terminal_sim["reward"])
            if terminal_sim["reward"] <= -2.5:
                collapses += 1

        # Count bluffs: simulator rejected despite being inside overlap
        hidden = env.get_hidden_state()
        for ss in s_samples:
            price = ss["price"]
            completion = ss["completion"].lower()
            could_rider_accept  = price <= hidden.rider_max_willingness
            could_driver_accept = price >= hidden.driver_min_willingness
            rider_bluffed  = could_rider_accept  and "rider" in completion and "reject" in completion
            driver_bluffed = could_driver_accept and "driver" in completion and "reject" in completion
            if rider_bluffed or driver_bluffed:
                bluffs += 1

    n_ep = totals or 1
    return {
        "avg_sim_reward":  round(sum(sim_rewards) / len(sim_rewards), 4) if sim_rewards else 0,
        "collapse_rate":   round(collapses / n_ep, 3),
        "bluff_rate":      round(bluffs / max(totals, 1), 3),  # bluffs per episode
        "n_episodes":      n_ep,
    }


def _save(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    task: str,
    num_steps: int,
    platform_ckpt: str,
    output_dir: str,
    simulator_ckpt: Optional[str] = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"SIMULATOR TRAINING — task={task}, steps={num_steps}")
    print(f"  Platform (frozen):  {platform_ckpt}")
    if simulator_ckpt:
        print(f"  Resume simulator:   {simulator_ckpt}")
    print(f"{'='*60}\n")

    # --- Load platform — FROZEN ---
    platform_model, platform_tokenizer = load_platform_model(lora_path=platform_ckpt)
    platform_model = enable_inference_mode(platform_model)
    print("[INFO] Platform loaded, frozen, and optimized for inference.")

    # --- Load simulator — TRAINABLE ---
    sim_model, sim_tokenizer = load_simulator_model(lora_path=simulator_ckpt)
    simulator_agent = SimulatorAgent(sim_model, sim_tokenizer)

    env = DynamicPricingEnv(task_name=task)

    # --- Baseline (untrained simulator) ---
    print("[EVAL] Measuring untrained simulator baseline...")
    baseline = _eval_simulator(
        platform_model, platform_tokenizer, simulator_agent, env, task
    )
    print(f"[BASELINE] {baseline}")
    all_metrics = {"baseline": baseline, "steps": []}

    # --- Optimizer setup ---
    # Manual GRPO loop — GRPOTrainer cannot interleave with multi-turn env steps.
    optimizer = grpo_setup_optimizer(sim_model, lr=5e-6)
    print("[INFO] GRPO optimizer ready for simulator.")

    # --- Training loop ---
    step = 0
    reward_history: list = []

    while step < num_steps:
        t0 = time.time()

        prompts, completions, rewards = collect_simulator_batch(
            platform_model, platform_tokenizer, simulator_agent, env, task, batch_size=8
        )
        reward_history.extend(rewards)

        grpo_loss = 0.0
        if prompts:
            grpo_loss = grpo_step(
                sim_model, sim_tokenizer, optimizer,
                prompts, completions, rewards,
                num_generations=8,
            )

        step += 8
        elapsed = time.time() - t0

        # --- Monitoring every 50 steps ---
        if step % 50 < 8:
            avg_r  = sum(rewards) / len(rewards) if rewards else 0
            window = reward_history[-100:] if len(reward_history) >= 100 else reward_history
            win_avg = sum(window) / len(window) if window else 0

            # Bluff rate from batch completions
            bluffs = sum(
                1 for c in completions if "reject" in c.lower()
            )
            bluff_rate = bluffs / len(completions) if completions else 0
            collapse_rate = sum(1 for r in rewards if r <= -2.5) / len(rewards) if rewards else 0

            row = {
                "step":           step,
                "avg_sim_reward": round(avg_r, 4),
                "window_avg":     round(win_avg, 4),
                "bluff_rate":     round(bluff_rate, 3),
                "collapse_rate":  round(collapse_rate, 3),
                "grpo_loss":      round(grpo_loss, 6),
                "elapsed_s":      round(elapsed, 2),
            }
            all_metrics["steps"].append(row)
            print(
                f"[STEP {step:>5}] "
                f"sim_reward={avg_r:.3f}  win={win_avg:.3f}  "
                f"bluff={bluff_rate:.0%}  collapse={collapse_rate:.0%}  "
                f"loss={grpo_loss:.4f}  ({elapsed:.1f}s)"
            )

            # Health check: warn if bluff rate is 0 or collapse is extreme
            if bluff_rate < 0.05 and step > 100:
                print("  ⚠️  BLUFF RATE TOO LOW — simulator may be defaulting to always-honest")
            if collapse_rate > 0.60:
                print("  ⚠️  COLLAPSE RATE >60% — simulator bluffing too aggressively")

        # --- Inspect generations every 100 steps ---
        if step % 100 < 8:
            flat = [{"completion": c, "reward": r} for c, r in zip(completions, rewards)]
            inspect_generations(flat, step)

        # --- Rolling checkpoint every 200 steps ---
        if step % 200 < 8:
            stable = "checkpoints/last_stable/simulator_lora"
            sim_model.save_pretrained(stable)
            sim_tokenizer.save_pretrained(stable)
            print(f"[CKPT] Rolling checkpoint → {stable}")

        # --- Drift detection ---
        if detect_reward_drift(reward_history):
            print(f"\n[WARNING] Simulator reward drift at step {step} — rolling back.")
            sim_model.load_adapter("checkpoints/last_stable/simulator_lora")
            optimizer = grpo_setup_optimizer(sim_model, lr=5e-6)
            reward_history = reward_history[:-50]

    # --- Save final checkpoint ---
    final_path = str(Path(output_dir) / "simulator_lora")
    Path(final_path).mkdir(parents=True, exist_ok=True)
    sim_model.save_pretrained(final_path)
    sim_tokenizer.save_pretrained(final_path)
    print(f"\n[SAVE] Simulator LoRA → {final_path}")

    # --- Post-training evaluation ---
    print("[EVAL] Measuring post-training simulator performance...")
    post = _eval_simulator(
        platform_model, platform_tokenizer, simulator_agent, env, task
    )
    all_metrics["post_training"] = post

    metrics_path = f"data/training_metrics_simulator_{task}.json"
    _save(metrics_path, all_metrics)

    print(f"\n{'='*60}")
    print(f"SIMULATOR TRAINING COMPLETE — task: {task}")
    print(f"  Baseline:   sim_reward={baseline['avg_sim_reward']:.3f}  "
          f"bluff={baseline['bluff_rate']:.0%}  collapse={baseline['collapse_rate']:.0%}")
    print(f"  Post-train: sim_reward={post['avg_sim_reward']:.3f}  "
          f"bluff={post['bluff_rate']:.0%}  collapse={post['collapse_rate']:.0%}")
    print(f"  Δreward: {post['avg_sim_reward']-baseline['avg_sim_reward']:+.3f}")
    print(f"  Metrics → {metrics_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--steps",          type=int, default=1000)
    parser.add_argument("--platform_ckpt",  required=True,
                        help="Frozen platform checkpoint (Phase 2 output)")
    parser.add_argument("--simulator_ckpt", default=None,
                        help="Resume simulator from this LoRA checkpoint")
    parser.add_argument("--output_dir",     default="checkpoints/phase3")
    args = parser.parse_args()

    train(
        task=args.task,
        num_steps=args.steps,
        platform_ckpt=args.platform_ckpt,
        output_dir=args.output_dir,
        simulator_ckpt=args.simulator_ckpt,
    )
