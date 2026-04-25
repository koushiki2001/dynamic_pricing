"""Reward computation: per-step shaping + terminal reward + process + simulator."""

from __future__ import annotations
from typing import Optional

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
    rider_max_willingness: float = 0.0,
) -> float:
    """Terminal reward at end of episode."""
    if ride_completed:
        platform_profit = proposed_price * commission_rate - operational_cost
        efficiency_bonus = min(max_steps / max(steps_taken, 1), MAX_EFFICIENCY_BONUS)
        missed_revenue_penalty = max(0.0, rider_max_willingness - proposed_price) * commission_rate
        return platform_profit * efficiency_bonus - missed_revenue_penalty
    if timed_out:
        return TIMEOUT_PENALTY
    # Cancellation
    return CANCELLATION_PENALTY


def compute_process_reward(
    proposed_price: float,
    last_proposed_price: Optional[float],
    rider_accepted: bool,
    driver_accepted: bool,
    rider_patience: float,
    driver_patience: float,
    step_number: int,
    max_steps: int,
) -> float:
    """Step-level process supervision signals (Phase 2b).

    Rewards directional convergence and urgency-aware behaviour.
    Kept small so it shapes without dominating the terminal reward.
    """
    reward = 0.0

    if last_proposed_price is not None:
        delta = proposed_price - last_proposed_price
        # Moving toward the rejecting party is directionally correct
        if not rider_accepted and delta < 0:
            reward += 0.03
        if not driver_accepted and delta > 0:
            reward += 0.03

    # Penalise failing to close when patience is critical and steps are almost gone
    min_patience = min(rider_patience, driver_patience)
    steps_remaining = max_steps - step_number
    if min_patience < 0.4 and steps_remaining <= 2 and not (rider_accepted and driver_accepted):
        reward -= 0.10

    # Penalise moving away from the party that already accepted
    if last_proposed_price is not None:
        delta = proposed_price - last_proposed_price
        if rider_accepted and not driver_accepted and delta < 0:
            reward -= 0.02
        if driver_accepted and not rider_accepted and delta > 0:
            reward -= 0.02

    return reward


def compute_simulator_reward(
    proposed_price: float,
    rider_max_willingness: float,
    driver_min_willingness: float,
    ride_completed: bool,
    steps_taken: int,
    max_steps: int,
) -> float:
    """Terminal reward for the simulator LLM (Phase 3+).

    Rewards surplus earned by both parties. Penalises deal collapse to
    prevent the simulator from bluffing recklessly.
    """
    if ride_completed:
        rider_surplus = max(0.0, rider_max_willingness - proposed_price)
        driver_surplus = max(0.0, proposed_price - driver_min_willingness)
        efficiency = min(max_steps / max(steps_taken, 1), 2.0)
        return (rider_surplus + driver_surplus) * efficiency
    # Deal collapsed or timed out — stronger penalty than platform's −5 to
    # discourage the simulator from bluffing when patience is already low
    return -3.0
