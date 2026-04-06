"""LLM baseline inference script (hackathon requirement).

This is the ROOT inference.py required by the HF Space validator.

Usage:
    OPENROUTER_API_KEY=sk-or-... python inference.py
    OPENROUTER_API_KEY=sk-or-... MODEL_NAME=openrouter/free python inference.py

Runs the LLM agent (via OpenRouter) against all 3 tasks and reports scores.
"""

from __future__ import annotations

import os
import sys

from ride_hailing_env.config import TASK_CONFIG, MAX_EFFICIENCY_BONUS
from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.tasks.graders import GRADER_WEIGHTS
from ride_hailing_env.utils import normalize_revenue
from baselines.openai_policy import OpenAIPolicy


def run_inference(task_name: str, num_episodes: int = 20, seed: int | None = None) -> dict:
    """Run OpenAI agent on a task, return grading results."""
    cfg = TASK_CONFIG[task_name]
    seed = seed or cfg["seed"]

    policy = OpenAIPolicy()

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    total_efficiency = 0.0
    completed_count = 0
    total_reward = 0.0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False
        episode_reward = 0.0

        policy.reset()

        while not done:
            action = policy(obs.model_dump())
            result = env.step(action)
            obs = result.observation
            done = result.done
            episode_reward += result.reward

        total_reward += episode_reward
        outcome = result.info["outcome"]

        status = outcome["termination_reason"]
        steps = outcome["steps_taken"]
        profit_str = f"${outcome['platform_profit']:.2f}" if outcome["platform_profit"] else "N/A"
        print(f"  Episode {ep+1:3d}/{num_episodes}: {status:>10s}  "
              f"steps={steps}  profit={profit_str}  reward={episode_reward:.4f}")

        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

    # Compute grader score
    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
    avg_efficiency = total_efficiency / completed_count if completed_count > 0 else 0.0
    avg_efficiency_norm = min(avg_efficiency / MAX_EFFICIENCY_BONUS, 1.0)
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0
    profit_norm = normalize_revenue(avg_profit)

    w = GRADER_WEIGHTS[task_name]
    score = (
        w["completion"] * completion_rate
        + w["efficiency"] * avg_efficiency_norm
        + w["profit"] * profit_norm
        + w["no_cancel"] * (1.0 - cancellation_rate)
    )
    score = max(0.0, min(1.0, score))

    return {
        "task": task_name,
        "score": round(score, 4),
        "completion_rate": round(completion_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_reward": round(total_reward / num_episodes, 4),
        "num_episodes": num_episodes,
    }


def main():
    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: Set OPENROUTER_API_KEY (or OPENAI_API_KEY) environment variable.")
        sys.exit(1)

    # Use fewer episodes for inference (API cost), more for final eval
    num_episodes = int(os.getenv("NUM_EPISODES", "20"))
    tasks = ["easy", "medium", "hard"]

    print("=" * 70)
    print("LLM BASELINE INFERENCE (OpenRouter)")
    print(f"Model: {os.getenv('MODEL_NAME', 'openrouter/free')}")
    print(f"Episodes per task: {num_episodes}")
    print("=" * 70)

    all_scores = []
    for task in tasks:
        print(f"\n--- Task: {task} ---")
        result = run_inference(task, num_episodes=num_episodes)
        all_scores.append(result["score"])
        print(f"\n  SCORE: {result['score']:.4f}  "
              f"completion={result['completion_rate']:.2%}  "
              f"cancel={result['cancellation_rate']:.2%}  "
              f"profit=${result['avg_profit']:.2f}  "
              f"avg_reward={result['avg_reward']:.4f}")

    print("\n" + "=" * 70)
    print(f"FINAL AVERAGE SCORE: {sum(all_scores)/len(all_scores):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
