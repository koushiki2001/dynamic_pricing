"""Graders: run N episodes, compute composite 0.0–1.0 score per task."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..config import TASK_CONFIG, MAX_EFFICIENCY_BONUS
from ..environment import DynamicPricingEnv
from ..utils import normalize_revenue


# Per-task grading weights
GRADER_WEIGHTS = {
    "easy": {"completion": 0.40, "efficiency": 0.30, "profit": 0.20, "no_cancel": 0.10},
    "medium": {"completion": 0.30, "efficiency": 0.25, "profit": 0.30, "no_cancel": 0.15},
    "hard": {"completion": 0.25, "efficiency": 0.20, "profit": 0.35, "no_cancel": 0.20},
}


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

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False

        while not done:
            action = policy_fn(obs.model_dump())
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

    # Compute metrics
    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
    avg_efficiency = (total_efficiency / completed_count if completed_count > 0 else 0.0)
    avg_efficiency_norm = min(avg_efficiency / MAX_EFFICIENCY_BONUS, 1.0)
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0
    profit_norm = normalize_revenue(avg_profit)

    # Composite score
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


def grade_easy(policy_fn: Callable, num_episodes: int | None = None, seed: int | None = None) -> Dict[str, float]:
    return grade_task("easy", policy_fn, num_episodes, seed)


def grade_medium(policy_fn: Callable, num_episodes: int | None = None, seed: int | None = None) -> Dict[str, float]:
    return grade_task("medium", policy_fn, num_episodes, seed)


def grade_hard(policy_fn: Callable, num_episodes: int | None = None, seed: int | None = None) -> Dict[str, float]:
    return grade_task("hard", policy_fn, num_episodes, seed)
