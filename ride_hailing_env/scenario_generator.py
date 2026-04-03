"""Generate ride scenarios per task difficulty."""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .config import TASK_CONFIG, DEFAULT_COMMISSION_RATE, DEFAULT_OPERATIONAL_COST
from .models import Observation, HiddenState
from .utils import patience_to_mood


class ScenarioGenerator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, task_name: str) -> Tuple[Observation, HiddenState]:
        cfg = TASK_CONFIG[task_name]
        return self._build_scenario(cfg)

    def _build_scenario(self, cfg: dict) -> Tuple[Observation, HiddenState]:
        rng = self.rng

        # Trip basics
        distance = rng.uniform(*cfg["distance_range"])
        duration = rng.uniform(*cfg["duration_range"])
        eta = rng.uniform(*cfg["eta_range"])

        # Base fair price from distance + duration
        base_price = 3.0 + 1.5 * distance + 0.3 * duration

        # Context — randomly sampled from option lists per task
        surge = rng.uniform(*cfg["surge_range"])
        weather = int(rng.choice(cfg["weather_condition_options"]))
        traffic = int(rng.choice(cfg["traffic_level_options"]))
        demand = int(rng.choice(cfg["demand_level_options"]))
        supply = int(rng.choice(cfg["supply_level_options"]))
        time_of_day = int(rng.choice(cfg["time_of_day_options"]))
        day_type = int(rng.choice(cfg["day_type_options"]))

        # Variable commission and operational cost
        commission = round(rng.uniform(*cfg["commission_rate_range"]), 2)
        op_cost = round(rng.uniform(*cfg["operational_cost_range"]), 2)

        # Context adjustments to prices
        weather_rider_adj = {0: 0.0, 1: 0.05, 2: 0.12}[weather]
        weather_driver_adj = {0: 0.0, 1: 0.08, 2: 0.15}[weather]
        traffic_driver_adj = {0: 0.0, 1: 0.06, 2: 0.12}[traffic]
        demand_rider_adj = {0: -0.05, 1: 0.0, 2: 0.08}[demand]
        supply_driver_adj = {0: 0.10, 1: 0.0, 2: -0.05}[supply]
        night_adj = 0.08 if time_of_day == 3 else 0.0

        # Rider: quotes LOWER (wants to pay less)
        gap = rng.uniform(*cfg["price_gap_range"])
        rider_quote = base_price - gap / 2.0
        rider_quote *= (1.0 + weather_rider_adj + demand_rider_adj)
        rider_quote = max(rider_quote, 5.0)

        # Driver: quotes HIGHER (wants to earn more)
        driver_quote = base_price + gap / 2.0
        driver_quote *= (1.0 + weather_driver_adj + traffic_driver_adj + supply_driver_adj + night_adj)
        driver_quote = max(driver_quote, rider_quote + 1.0)

        # Hidden willingness (slack beyond quoted prices)
        rider_slack = rng.uniform(*cfg["rider_slack_range"])
        driver_slack = rng.uniform(*cfg["driver_slack_range"])
        rider_max_willingness = rider_quote + rider_slack
        driver_min_willingness = driver_quote - driver_slack

        # Asymmetric offset: shift the acceptance zone away from the naive
        # midpoint so the agent must explore, not just split the difference.
        # offset_range controls how far the zone center drifts from midpoint.
        offset_frac = rng.uniform(-0.35, 0.35)
        shift = offset_frac * abs(rider_slack + driver_slack)
        rider_max_willingness += shift
        driver_min_willingness += shift

        # Guarantee a feasible overlap zone exists
        min_overlap = cfg.get("min_overlap", 2.0)
        overlap = rider_max_willingness - driver_min_willingness
        if overlap < min_overlap:
            deficit = min_overlap - overlap
            rider_max_willingness += deficit / 2.0
            driver_min_willingness -= deficit / 2.0

        # Clamp to valid range
        driver_min_willingness = max(driver_min_willingness, 0.0)

        # Asymmetric patience — rider and driver can start at different levels
        rider_patience_init = rng.uniform(*cfg["rider_patience_init_range"])
        driver_patience_init = rng.uniform(*cfg["driver_patience_init_range"])

        # Patience decay
        rider_patience_decay = rng.uniform(*cfg["rider_patience_decay_range"])
        driver_patience_decay = rng.uniform(*cfg["driver_patience_decay_range"])

        # Acceptance noise
        rider_noise = rng.uniform(*cfg["acceptance_noise_range"])
        driver_noise = rng.uniform(*cfg["acceptance_noise_range"])

        observation = Observation(
            rider_quoted_price=round(rider_quote, 2),
            driver_quoted_price=round(driver_quote, 2),
            price_gap=round(abs(driver_quote - rider_quote), 2),
            distance_km=round(distance, 1),
            estimated_duration_min=round(duration, 1),
            pickup_eta_min=round(eta, 1),
            demand_level=demand,
            supply_level=supply,
            surge_multiplier=round(surge, 2),
            weather_condition=weather,
            traffic_level=traffic,
            time_of_day=time_of_day,
            day_type=day_type,
            commission_rate=commission,
            operational_cost=op_cost,
            max_steps=cfg["max_steps"],
            step_number=0,
            rider_patience=round(rider_patience_init, 2),
            driver_patience=round(driver_patience_init, 2),
            rider_mood=patience_to_mood(rider_patience_init),
            driver_mood=patience_to_mood(driver_patience_init),
            last_rider_response=None,
            last_driver_response=None,
            last_proposed_price=None,
        )

        hidden = HiddenState(
            rider_max_willingness=round(rider_max_willingness, 2),
            driver_min_willingness=round(driver_min_willingness, 2),
            rider_patience_decay=round(rider_patience_decay, 4),
            driver_patience_decay=round(driver_patience_decay, 4),
            rider_acceptance_noise=round(rider_noise, 2),
            driver_acceptance_noise=round(driver_noise, 2),
        )

        return observation, hidden
