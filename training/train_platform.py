"""Platform LLM training with GRPO (Phase 2 + Phase 4).

Phase 2 usage — warm start vs rule-based simulator:
    python training/train_platform.py --task easy --steps 500

Phase 4 usage — re-train vs frozen simulator_v1:
    python training/train_platform.py \\
        --task easy --steps 1000 \\
        --simulator_ckpt checkpoints/phase3/simulator_lora \\
        --platform_ckpt  checkpoints/phase2/platform_lora \\
        --output_dir     checkpoints/phase4

Monitoring:
    - Logs 8 metric columns every 50 steps.
    - Calls inspect_generations() every 100 steps (requires human review).
    - Saves a rolling "last_stable" checkpoint every 200 steps for rollback.
    - Detects reward drift and warns loudly if training is degrading.
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
    collect_batch,
    collect_platform_rollout,
    detect_reward_drift,
    inspect_generations,
)
from training.grpo_utils import grpo_setup_optimizer, grpo_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_platform(model, tokenizer, env: DynamicPricingEnv, task: str, n: int = 20,
                   simulator_agent=None):
    """Quick evaluation: run n episodes and return summary dict.

    simulator_agent=None uses rule-based simulator (Phase 2).
    Passing a SimulatorAgent evaluates against the strategic opponent (Phase 4).
    """
    rewards, completions, cancels, timeouts = [], 0, 0, 0
    for _ in range(n):
        if simulator_agent is None:
            episode = collect_platform_rollout(model, tokenizer, env, task)
        else:
            from training.rollout import collect_multiagent_rollout
            result = collect_multiagent_rollout(model, tokenizer, simulator_agent, env, task)
            episode = result["platform_samples"]
        if not episode:
            continue
        terminal = episode[-1]
        r = terminal["reward"]
        rewards.append(r)
        if r > 0:
            completions += 1
        elif r <= -4.0:
            cancels += 1
        elif r <= -1.5:
            timeouts += 1

    n_ep = len(rewards) or 1
    return {
        "avg_reward":      round(sum(rewards) / n_ep, 4),
        "completion_rate": round(completions / n_ep, 3),
        "cancel_rate":     round(cancels / n_ep, 3),
        "timeout_rate":    round(timeouts / n_ep, 3),
        "n_episodes":      n_ep,
    }


def _save_metrics(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    task: str,
    num_steps: int,
    batch_size: int,
    output_dir: str,
    platform_ckpt: Optional[str] = None,
    simulator_ckpt: Optional[str] = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"PLATFORM TRAINING — task={task}, steps={num_steps}, batch={batch_size}")
    if platform_ckpt:
        print(f"  Resume platform from: {platform_ckpt}")
    if simulator_ckpt:
        print(f"  Simulator opponent:   {simulator_ckpt}  (frozen)")
    print(f"{'='*60}\n")

    # --- Load models ---
    platform_model, platform_tokenizer = load_platform_model(lora_path=platform_ckpt)
    env = DynamicPricingEnv(task_name=task)

    # If a simulator checkpoint is provided (Phase 4), load and freeze it
    simulator_agent = None
    if simulator_ckpt and Path(simulator_ckpt).exists():
        from training.simulator_agent import SimulatorAgent
        sim_model, sim_tokenizer = load_simulator_model(lora_path=simulator_ckpt)
        sim_model = enable_inference_mode(sim_model)
        simulator_agent = SimulatorAgent(sim_model, sim_tokenizer)
        print(f"[INFO] Simulator loaded, frozen, and optimized for inference.")

    # --- Baseline (pre-training) ---
    print("[EVAL] Measuring baseline performance (pre-training)...")
    baseline = _eval_platform(platform_model, platform_tokenizer, env, task,
                               simulator_agent=simulator_agent)
    print(f"[BASELINE] {baseline}")
    all_metrics = {"baseline": baseline, "steps": []}

    # Capture baseline snapshot for the demo "before" number
    snap_path = f"data/baseline_snapshot_{task}.json"
    _save_metrics(snap_path, baseline)
    print(f"[SAVE] Baseline snapshot → {snap_path}")

    # --- Optimizer setup ---
    # We use a manual GRPO update loop rather than TRL's GRPOTrainer because
    # our environment is multi-turn and stateful — generation must be interleaved
    # with env.step() calls, which GRPOTrainer's internal generation loop cannot do.
    optimizer = grpo_setup_optimizer(platform_model, lr=5e-6)
    print("[INFO] GRPO optimizer ready.")

    # --- Training loop ---
    step = 0
    reward_history = []
    all_samples_this_window = []

    while step < num_steps:
        t0 = time.time()

        if simulator_agent is None:
            prompts, completions, rewards = collect_batch(
                platform_model, platform_tokenizer, env, task, batch_size
            )
        else:
            from training.rollout import collect_multiagent_batch
            prompts, completions, rewards = collect_multiagent_batch(
                platform_model, platform_tokenizer, simulator_agent, env, task, batch_size
            )

        reward_history.extend(rewards)
        all_samples_this_window.extend(
            [{"completion": c, "reward": r, "price": None} for c, r in zip(completions, rewards)]
        )

        # GRPO weight update — manual policy gradient over collected samples
        grpo_loss = 0.0
        if prompts:
            grpo_loss = grpo_step(
                platform_model, platform_tokenizer, optimizer,
                prompts, completions, rewards,
                num_generations=batch_size,
            )

        step += batch_size
        elapsed = time.time() - t0

        # --- Monitoring: every 50 steps ---
        if step % 50 < batch_size:
            window = reward_history[-100:] if len(reward_history) >= 100 else reward_history
            avg_r = sum(rewards) / len(rewards) if rewards else 0
            win_avg = sum(window) / len(window) if window else 0
            comp_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
            cancel_rate = sum(1 for r in rewards if r <= -4.0) / len(rewards) if rewards else 0
            timeout_rate = sum(1 for r in rewards if -2.5 < r <= -1.5) / len(rewards) if rewards else 0
            violation_count = sum(
                len(s.get("violations", [])) for s in all_samples_this_window
            )

            row = {
                "step": step,
                "avg_reward": round(avg_r, 4),
                "window_avg": round(win_avg, 4),
                "completion_rate": round(comp_rate, 3),
                "cancel_rate": round(cancel_rate, 3),
                "timeout_rate": round(timeout_rate, 3),
                "anti_hack_violations": violation_count,
                "grpo_loss": round(grpo_loss, 6),
                "elapsed_s": round(elapsed, 2),
            }
            all_metrics["steps"].append(row)
            print(
                f"[STEP {step:>5}] "
                f"avg={avg_r:.3f}  win={win_avg:.3f}  "
                f"done={comp_rate:.0%}  cancel={cancel_rate:.0%}  "
                f"timeout={timeout_rate:.0%}  "
                f"loss={grpo_loss:.4f}  hacks={violation_count}  ({elapsed:.1f}s)"
            )
            all_samples_this_window = []

        # --- Inspect generations every 100 steps ---
        if step % 100 < batch_size:
            flat = [{"completion": c, "reward": r, "price": None}
                    for c, r in zip(completions, rewards)]
            inspect_generations(flat, step)

        # --- Rolling checkpoint every 200 steps ---
        if step % 200 < batch_size:
            stable_path = "checkpoints/last_stable/platform_lora"
            platform_model.save_pretrained(stable_path)
            platform_tokenizer.save_pretrained(stable_path)
            print(f"[CKPT] Rolling checkpoint → {stable_path}")

        # --- Drift detection ---
        if detect_reward_drift(reward_history):
            print(
                f"\n[WARNING] Reward drift detected at step {step}! "
                "Rolling back to last stable checkpoint."
            )
            platform_model.load_adapter("checkpoints/last_stable/platform_lora")
            optimizer = grpo_setup_optimizer(platform_model, lr=5e-6)
            reward_history = reward_history[:-50]  # trim the drifted window

    # --- Save final checkpoint ---
    final_path = str(Path(output_dir) / "platform_lora")
    Path(final_path).mkdir(parents=True, exist_ok=True)
    platform_model.save_pretrained(final_path)
    platform_tokenizer.save_pretrained(final_path)
    print(f"\n[SAVE] Final platform LoRA → {final_path}")

    # --- Post-training evaluation ---
    print("[EVAL] Measuring post-training performance...")
    post = _eval_platform(platform_model, platform_tokenizer, env, task,
                           simulator_agent=simulator_agent)
    all_metrics["post_training"] = post

    metrics_path = f"data/training_metrics_platform_{task}.json"
    _save_metrics(metrics_path, all_metrics)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — task: {task}")
    print(f"  Baseline:     avg_reward={baseline['avg_reward']:.3f}  "
          f"completion={baseline['completion_rate']:.0%}")
    print(f"  Post-train:   avg_reward={post['avg_reward']:.3f}  "
          f"completion={post['completion_rate']:.0%}")
    print(f"  Improvement:  Δreward={post['avg_reward']-baseline['avg_reward']:+.3f}  "
          f"Δcompletion={post['completion_rate']-baseline['completion_rate']:+.0%}")
    print(f"  Metrics saved → {metrics_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",            default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--steps",           type=int, default=500)
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--output_dir",      default="checkpoints/phase2")
    parser.add_argument("--platform_ckpt",   default=None,
                        help="Resume platform from this LoRA checkpoint")
    parser.add_argument("--simulator_ckpt",  default=None,
                        help="Freeze this simulator checkpoint as opponent (Phase 4)")
    args = parser.parse_args()

    train(
        task=args.task,
        num_steps=args.steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        platform_ckpt=args.platform_ckpt,
        simulator_ckpt=args.simulator_ckpt,
    )
