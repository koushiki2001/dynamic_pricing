"""Prompt builders for the platform LLM and simulator LLM.

Both functions return a string prompt suitable for direct tokenization by the
caller. They never reach out to a model — pure text formatting.
"""

from __future__ import annotations

from ride_hailing_env.models import HiddenState, Observation


WEATHER = ["clear", "rain", "storm"]
TRAFFIC = ["low", "medium", "heavy"]
DEMAND = ["low", "medium", "high"]
SUPPLY = ["low", "medium", "high"]
TIME = ["morning", "afternoon", "evening", "night"]
DAY = ["weekday", "weekend", "holiday"]


def build_platform_prompt(obs: Observation) -> str:
    """Convert an Observation into the prompt the platform LLM sees each step."""
    lines = [
        "You are a ride-hailing platform. Propose a price both rider and driver will accept.",
        "",
        f"Rider quoted: ${obs.rider_quoted_price:.2f}",
        f"Driver quoted: ${obs.driver_quoted_price:.2f}",
        f"Gap: ${obs.price_gap:.2f}",
        "",
        f"Trip: {obs.distance_km}km, {obs.estimated_duration_min}min, ETA {obs.pickup_eta_min}min",
        f"Context: weather={WEATHER[obs.weather_condition]}, traffic={TRAFFIC[obs.traffic_level]}",
        f"Market: demand={DEMAND[obs.demand_level]}, supply={SUPPLY[obs.supply_level]}, surge={obs.surge_multiplier}x",
        f"Time: {TIME[obs.time_of_day]}, {DAY[obs.day_type]}",
        f"Commission: {obs.commission_rate*100:.0f}%, Op cost: ${obs.operational_cost:.2f}",
        "",
        f"Step {obs.step_number}/{obs.max_steps}",
        f"Rider patience: {obs.rider_patience:.2f} ({obs.rider_mood})",
        f"Driver patience: {obs.driver_patience:.2f} ({obs.driver_mood})",
    ]
    if obs.last_proposed_price is not None:
        lines += [
            "",
            f"Last proposal: ${obs.last_proposed_price:.2f}",
            f"  Rider: {obs.last_rider_response}",
            f"  Driver: {obs.last_driver_response}",
        ]
    lines += ["", 'Respond with JSON only: {"price": <number>}']
    return "\n".join(lines)


def build_simulator_prompt(
    hidden: HiddenState,
    obs: Observation,
    proposed_price: float,
) -> str:
    """Convert hidden state + obs + proposed price into the simulator LLM prompt.

    The simulator agent plays both rider and driver — it sees the true hidden
    thresholds (which the platform never sees) and decides whether each party
    accepts honestly or bluffs (rejects despite being able to accept).
    """
    rider_can_accept = proposed_price <= hidden.rider_max_willingness
    driver_can_accept = proposed_price >= hidden.driver_min_willingness
    rider_surplus = max(0.0, hidden.rider_max_willingness - proposed_price)
    driver_surplus = max(0.0, proposed_price - hidden.driver_min_willingness)
    rider_safe_steps = (
        int(obs.rider_patience / hidden.rider_patience_decay)
        if hidden.rider_patience_decay > 0
        else 99
    )
    driver_safe_steps = (
        int(obs.driver_patience / hidden.driver_patience_decay)
        if hidden.driver_patience_decay > 0
        else 99
    )
    steps_remaining = obs.max_steps - obs.step_number

    lines = [
        "You are the rider and driver in a ride-hailing negotiation.",
        "You know your true thresholds. Decide whether to accept or bluff (reject despite being able to accept).",
        "",
        "=== YOUR HIDDEN STATE ===",
        f"Rider true max willingness: ${hidden.rider_max_willingness:.2f}",
        f"Driver true min willingness: ${hidden.driver_min_willingness:.2f}",
        f"Rider patience: {obs.rider_patience:.2f}, decay per rejection: {hidden.rider_patience_decay:.2f}",
        f"  -> ~{rider_safe_steps} safe rejections remaining before cancellation risk",
        f"Driver patience: {obs.driver_patience:.2f}, decay per rejection: {hidden.driver_patience_decay:.2f}",
        f"  -> ~{driver_safe_steps} safe rejections remaining",
        "",
        "=== PLATFORM PROPOSAL ===",
        f"Proposed price: ${proposed_price:.2f}",
        f"Steps remaining: {steps_remaining} of {obs.max_steps}",
        "",
        "=== DECISION ANALYSIS ===",
        f"Rider CAN accept: {rider_can_accept} (surplus if accepted: ${rider_surplus:.2f})",
        f"Driver CAN accept: {driver_can_accept} (surplus if accepted: ${driver_surplus:.2f})",
        "",
        "BLUFFING means rejecting even though you could accept, hoping for a better deal.",
        "RISK: patience decreases. If patience hits 0, the deal collapses and you earn nothing.",
        "",
        'Respond with JSON only: {"rider": "accept" or "reject", "driver": "accept" or "reject"}',
    ]
    return "\n".join(lines)
