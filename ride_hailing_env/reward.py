"""Reward computation: per-step shaping + terminal reward."""

from __future__ import annotations

from .config import (
    CANCELLATION_PENALTY,
    TIMEOUT_PENALTY,
    PARTIAL_ACCEPT_REWARD,
    DOUBLE_REJECT_PENALTY,
    MAX_EFFICIENCY_BONUS,
)
from .simulator import SimResult


def compute_step_reward(sim_result: SimResult) -> float:
    """Shaping reward for intermediate steps (not terminal)."""
    if sim_result.rider_accepted and not sim_result.driver_accepted:
        return PARTIAL_ACCEPT_REWARD
    if not sim_result.rider_accepted and sim_result.driver_accepted:
        return PARTIAL_ACCEPT_REWARD
    if not sim_result.rider_accepted and not sim_result.driver_accepted:
        return DOUBLE_REJECT_PENALTY
    return 0.0


def compute_terminal_reward(
    proposed_price: float,
    commission_rate: float,
    operational_cost: float,
    steps_taken: int,
    max_steps: int,
    ride_completed: bool,
    timed_out: bool,
) -> float:
    """Terminal reward at end of episode."""
    if ride_completed:
        platform_profit = proposed_price * commission_rate - operational_cost
        efficiency_bonus = min(max_steps / max(steps_taken, 1), MAX_EFFICIENCY_BONUS)
        return platform_profit * efficiency_bonus
    if timed_out:
        return TIMEOUT_PENALTY
    # Cancellation
    return CANCELLATION_PENALTY
