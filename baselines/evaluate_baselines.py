"""Evaluate all baseline policies across all tasks."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.tasks.graders import grade_task
from baselines.midpoint_policy import midpoint_policy
from baselines.adaptive_policy import AdaptivePolicy


def run_stateless_baseline(task_name: str, policy_fn, num_episodes=None, seed=None):
    return grade_task(task_name, policy_fn, num_episodes, seed)


def run_stateful_baseline(task_name: str, policy_class, num_episodes=None, seed=None):
    """For stateful policies that need reset() between episodes."""
    from ride_hailing_env.config import TASK_CONFIG
    from ride_hailing_env.environment import DynamicPricingEnv
    from ride_hailing_env.utils import normalize_revenue
    from ride_hailing_env.tasks.graders import GRADER_WEIGHTS
    from ride_hailing_env.config import MAX_EFFICIENCY_BONUS

    cfg = TASK_CONFIG[task_name]
    num_episodes = num_episodes or cfg["num_eval_episodes"]
    seed = seed or cfg["seed"]

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    total_efficiency = 0.0
    completed_count = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False

        policy = policy_class()
        policy.reset()

        while not done:
            action = policy(obs.model_dump())
            result = env.step(action)
            obs = result.observation
            done = result.done

        outcome = result.info["outcome"]
        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

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
        "timeout_rate": round(total_timed_out / num_episodes, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_efficiency": round(avg_efficiency, 4),
        "num_episodes": num_episodes,
    }


def main():
    tasks = ["easy", "medium", "hard"]
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)

    # --- Midpoint baseline ---
    print("\n--- Midpoint Policy ---")
    midpoint_scores = []
    for task in tasks:
        result = run_stateless_baseline(task, midpoint_policy)
        midpoint_scores.append(result["score"])
        print(f"  {task:>8s}: score={result['score']:.4f}  "
              f"completion={result['completion_rate']:.2%}  "
              f"cancel={result['cancellation_rate']:.2%}  "
              f"profit=${result['avg_profit']:.2f}")
    print(f"  {'avg':>8s}: {sum(midpoint_scores)/len(midpoint_scores):.4f}")

    # --- Adaptive baseline ---
    print("\n--- Adaptive Policy ---")
    adaptive_scores = []
    for task in tasks:
        result = run_stateful_baseline(task, AdaptivePolicy)
        adaptive_scores.append(result["score"])
        print(f"  {task:>8s}: score={result['score']:.4f}  "
              f"completion={result['completion_rate']:.2%}  "
              f"cancel={result['cancellation_rate']:.2%}  "
              f"profit=${result['avg_profit']:.2f}")
    print(f"  {'avg':>8s}: {sum(adaptive_scores)/len(adaptive_scores):.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
