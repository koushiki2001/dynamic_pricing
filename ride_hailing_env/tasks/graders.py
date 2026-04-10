"""Graders: run N episodes, compute score as fraction of passing episodes.

Each episode passes if and only if:
  1. ride_completed = True
  2. episode_reward > reward_threshold   (earned enough profit efficiently)
  3. missed_revenue_penalty < penalty_threshold  (didn't leave too much on the table)

Both thresholds are scenario-specific, computed from the hidden state at
generation time — making pass/fail fully deterministic per episode.

Score = total_passed / num_episodes, clamped strictly to (0.0, 1.0).

Threshold fractions by task:
  easy   — reward_threshold_fraction=0.30, penalty_threshold_fraction=1.00
  medium — reward_threshold_fraction=0.50, penalty_threshold_fraction=0.75
  hard   — reward_threshold_fraction=0.70, penalty_threshold_fraction=0.50
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..config import TASK_CONFIG
from ..environment import DynamicPricingEnv


# Per-task grading weights (used by pre_submission_check.py)
GRADER_WEIGHTS = {
    "easy":   {"completion": 0.40, "efficiency": 0.30, "profit": 0.20, "no_cancel": 0.10},
    "medium": {"completion": 0.30, "efficiency": 0.25, "profit": 0.30, "no_cancel": 0.15},
    "hard":   {"completion": 0.25, "efficiency": 0.20, "profit": 0.35, "no_cancel": 0.20},
}


def grade_task(
    task_name: str,
    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_episodes: int | None = None,
    seed: int | None = None,
) -> float:
    """Run a policy through N episodes and return a score strictly in (0.0, 1.0).

    An episode passes when:
      - ride_completed is True
      - cumulative episode_reward > reward_threshold
      - missed_revenue_penalty < penalty_threshold

    Args:
        task_name: One of "easy", "medium", "hard".
        policy_fn: Callable that takes an observation dict and returns an action dict.
        num_episodes: Number of episodes to run (defaults to task config value).
        seed: Base random seed (defaults to task config value).

    Returns:
        Float strictly in (0.0, 1.0): fraction of episodes that passed.
    """
    cfg = TASK_CONFIG[task_name]
    num_episodes = num_episodes or cfg["num_eval_episodes"]
    seed = seed or cfg["seed"]

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

        if (
            outcome["ride_completed"]
            and episode_reward > reward_threshold
            and missed_revenue_penalty < penalty_threshold
        ):
            total_passed += 1

    # Clamp strictly to (0.0, 1.0) — never exactly 0.0 or 1.0
    raw_score = total_passed / num_episodes
    return max(1e-6, min(1.0 - 1e-6, raw_score))


def grade_easy(
    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_episodes: int | None = None,
    seed: int | None = None,
) -> float:
    """Grade the easy task. Returns float strictly in (0.0, 1.0)."""
    return grade_task("easy", policy_fn, num_episodes, seed)


def grade_medium(
    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_episodes: int | None = None,
    seed: int | None = None,
) -> float:
    """Grade the medium task. Returns float strictly in (0.0, 1.0)."""
    return grade_task("medium", policy_fn, num_episodes, seed)


def grade_hard(
    policy_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    num_episodes: int | None = None,
    seed: int | None = None,
) -> float:
    """Grade the hard task. Returns float strictly in (0.0, 1.0)."""
    return grade_task("hard", policy_fn, num_episodes, seed)
