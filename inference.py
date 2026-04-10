"""LLM baseline inference script (hackathon requirement).

This is the ROOT inference.py required by the HF Space validator.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    API_BASE_URL=https://openrouter.ai/api/v1 MODEL_NAME=google/gemini-2.0-flash-001 HF_TOKEN=sk-... python inference.py

STDOUT FORMAT (required by platform):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

os.environ.setdefault("API_BASE_URL", "https://router.huggingface.co/v1")
os.environ.setdefault("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

from ride_hailing_env.config import TASK_CONFIG
from ride_hailing_env.environment import DynamicPricingEnv
from baselines.openai_policy import OpenAIPolicy

BENCHMARK = "ride_hailing_dynamic_pricing"


def _sigmoid_score(raw: float) -> float:
    """Map raw score [0, 1] to strictly (0, 1) via sigmoid.

    Stretches [0, 1] to [-6, 6] before sigmoid, giving output in (0.0025, 0.9975).
    Can never return exactly 0.0 or 1.0.
    """
    x = raw * 12.0 - 6.0
    return 1.0 / (1.0 + math.exp(-x))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def run_inference(task_name: str, num_episodes: int = 20, seed: int | None = None) -> dict:
    """Run LLM agent on a task, emit required stdout logs, return grading results."""
    cfg = TASK_CONFIG[task_name]
    seed = seed or cfg["seed"]
    model_name = os.environ.get("MODEL_NAME", "unknown")

    policy = OpenAIPolicy()

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    completed_count = 0
    total_passed = 0
    total_penalty = 0.0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step_num = 0
        rewards: List[float] = []

        policy.reset()

        log_start(task=task_name, env=BENCHMARK, model=model_name)

        try:
            while not done:
                obs_dict = obs.model_dump()
                action = policy(obs_dict)

                # Handle duplicate-price rejection: nudge price to break the loop
                result = env.step(action)
                if result.info.get("duplicate_price"):
                    p = action["payload"]["price"]
                    for nudge in [0.5, -0.5, 1.0]:
                        action["payload"]["price"] = round(p + nudge, 2)
                        result = env.step(action)
                        if not result.info.get("duplicate_price"):
                            break

                step_num += 1
                price = action["payload"]["price"]
                reward = result.reward or 0.0
                done = result.done
                episode_reward += reward
                rewards.append(reward)

                log_step(
                    step=step_num,
                    action=f"propose_price({price:.2f})",
                    reward=reward,
                    done=done,
                    error=None,
                )

                obs = result.observation

            outcome = result.info["outcome"]
            missed_revenue_penalty = result.info["missed_revenue_penalty"]
            reward_threshold = result.info["reward_threshold"]
            penalty_threshold = result.info["penalty_threshold"]

            profit_val = outcome.get("platform_profit") or 0.0
            episode_passed = (
                outcome["ride_completed"]
                and episode_reward > reward_threshold
                and missed_revenue_penalty < penalty_threshold
            )
            success = outcome["ride_completed"]

            # Per-episode score via sigmoid — strictly in (0, 1)
            raw = 1.0 if episode_passed else (0.4 if outcome["ride_completed"] else 0.0)
            episode_score = _sigmoid_score(raw)

            total_penalty += missed_revenue_penalty

            if outcome["ride_completed"]:
                total_completed += 1
                completed_count += 1
                total_profit += profit_val
                if episode_passed:
                    total_passed += 1
            elif outcome["timed_out"]:
                total_timed_out += 1
            else:
                total_cancelled += 1

        except Exception as exc:
            success = False
            episode_score = _sigmoid_score(0.0)
            log_step(step=step_num + 1, action="error", reward=0.0, done=True, error=str(exc))

        finally:
            log_end(
                success=success,
                steps=step_num,
                score=episode_score,
                rewards=rewards,
            )

    # Aggregate score across all episodes — sigmoid ensures strictly (0, 1)
    raw_score = total_passed / num_episodes
    score = _sigmoid_score(raw_score)

    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0

    return {
        "task": task_name,
        "score": round(score, 4),
        "pass_rate": round(total_passed / num_episodes, 4),
        "completion_rate": round(completion_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "timeout_rate": round(total_timed_out / num_episodes, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_penalty": round(total_penalty / num_episodes, 4),
        "num_episodes": num_episodes,
    }


def main():
    if not os.getenv("HF_TOKEN"):
        print("ERROR: Set HF_TOKEN environment variable.")
        sys.exit(1)
    if not os.getenv("API_BASE_URL"):
        print("ERROR: Set API_BASE_URL environment variable.")
        sys.exit(1)
    if not os.getenv("MODEL_NAME"):
        print("ERROR: Set MODEL_NAME environment variable.")
        sys.exit(1)

    num_episodes = int(os.getenv("NUM_EPISODES", "20"))
    tasks = ["easy", "medium", "hard"]

    print(f"[CONFIG] api_base_url={os.environ['API_BASE_URL']} "
          f"model={os.environ['MODEL_NAME']} "
          f"episodes_per_task={num_episodes}", flush=True)

    all_results = []
    for task in tasks:
        result = run_inference(task, num_episodes=num_episodes)
        all_results.append(result)

    avg_score = sum(r["score"] for r in all_results) / len(all_results)

    print("\n" + "=" * 88)
    print(f"{'RESULTS SUMMARY':^88}")
    print("=" * 88)
    print(f"{'Task':<10} {'Score':>7} {'Pass%':>7} {'Complete%':>10} {'Cancel%':>8} {'Timeout%':>9} {'AvgProfit':>10} {'AvgPenalty':>11}")
    print("-" * 88)
    for r in all_results:
        print(f"{r['task']:<10} {r['score']:>7.4f} {r['pass_rate']*100:>6.1f}% "
              f"{r['completion_rate']*100:>9.1f}% {r['cancellation_rate']*100:>7.1f}% "
              f"{r['timeout_rate']*100:>8.1f}% ${r['avg_profit']:>9.2f} ${r['avg_penalty']:>10.4f}")
    print("-" * 88)
    print(f"{'AVERAGE':<10} {avg_score:>7.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
