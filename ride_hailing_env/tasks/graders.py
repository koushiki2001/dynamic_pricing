"""Graders: run N episodes, compute 0.0–1.0 score as fraction of passing episodes.

An episode passes if and only if:
  1. ride_completed = True
  2. episode_reward > reward_threshold  (earned enough profit efficiently)
  3. missed_revenue_penalty < penalty_threshold  (didn't leave too much on the table)

Both thresholds are scenario-specific, derived from the hidden state at generation time.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..config import TASK_CONFIG
from ..environment import DynamicPricingEnv


def grade_task(
    task_name: str,
    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_episodes: int | None = None,
    seed: int | None = None,
) -> Dict[str, float]:
    """Run a policy through N episodes and return a composite score in [0.0, 1.0].

    policy_fn: takes observation dict, returns action dict {"type": "propose_price", "payload": {"price": X}}
    """
    cfg = TASK_CONFIG[task_name]
    num_episodes = num_episodes or cfg["num_eval_episodes"]
    seed = seed or cfg["seed"]

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    total_efficiency = 0.0
    completed_count = 0
    total_passed = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = policy_fn(obs.model_dump())
            result = env.step(action)
            obs = result.observation
            done = result.done
            episode_reward += result.reward

        outcome = result.info["outcome"]
        missed_revenue_penalty = result.info["missed_revenue_penalty"]
        reward_threshold = result.info["reward_threshold"]
        penalty_threshold = result.info["penalty_threshold"]

        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]

            # Per-episode pass/fail: agent must earn enough AND not leave too much on the table
            if episode_reward > reward_threshold and missed_revenue_penalty < penalty_threshold:
                total_passed += 1
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

    # Score = fraction of episodes that passed both criteria
    # score = max(0.0, min(1.0, total_passed / num_episodes))
    # Score = fraction of episodes that passed both criteria,
    # clamped to be strictly between 0 and 1
    EPS = 1e-4
    raw_score = total_passed / num_episodes
    score = max(EPS, min(1.0 - EPS, raw_score))

    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
    avg_efficiency = (total_efficiency / completed_count if completed_count > 0 else 0.0)
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0

    return {
        "task": task_name,
        "score": round(score, 4),
        "pass_rate": round(total_passed / num_episodes, 4),
        "completion_rate": round(completion_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "timeout_rate": round(total_timed_out / num_episodes, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_efficiency": round(avg_efficiency, 4),
        "num_episodes": num_episodes,
    }


def grade_easy(policy_fn: Callable, num_episodes: int | None = None, seed: int | None = None) -> Dict[str, float]:
    return grade_task("easy", policy_fn, num_episodes, seed)


def grade_medium(policy_fn: Callable, num_episodes: int | None = None, seed: int | None = None) -> Dict[str, float]:
    return grade_task("medium", policy_fn, num_episodes, seed)


def grade_hard(policy_fn: Callable, num_episodes: int | None = None, seed: int | None = None) -> Dict[str, float]:
    return grade_task("hard", policy_fn, num_episodes, seed)
