"""Simulator: accept/reject logic, patience decay, mood derivation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .models import Observation, HiddenState
from .utils import clamp, patience_to_mood


@dataclass
class SimResult:
    rider_accepted: bool
    driver_accepted: bool
    rider_cancelled: bool
    driver_cancelled: bool
    new_rider_patience: float
    new_driver_patience: float
    new_rider_mood: str
    new_driver_mood: str
    both_accepted: bool
    any_cancelled: bool


class Simulator:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def simulate_step(
        self,
        observation: Observation,
        hidden: HiddenState,
        proposed_price: float,
    ) -> SimResult:
        rng = self.rng

        # --- Rider decision ---
        # Rider accepts if price ≤ their hidden max willingness (+ noise)
        rider_threshold = hidden.rider_max_willingness + rng.uniform(
            -hidden.rider_acceptance_noise, hidden.rider_acceptance_noise
        )
        rider_accepted = proposed_price <= rider_threshold

        # --- Driver decision ---
        # Driver accepts if proposed price meets their minimum acceptable fare
        driver_threshold = hidden.driver_min_willingness + rng.uniform(
            -hidden.driver_acceptance_noise, hidden.driver_acceptance_noise
        )
        driver_accepted = proposed_price >= driver_threshold

        # --- Patience updates (only on rejection) ---
        new_rider_patience = observation.rider_patience
        if not rider_accepted:
            new_rider_patience = clamp(observation.rider_patience - hidden.rider_patience_decay, 0.0, 1.0)

        new_driver_patience = observation.driver_patience
        if not driver_accepted:
            new_driver_patience = clamp(observation.driver_patience - hidden.driver_patience_decay, 0.0, 1.0)

        # --- Cancellation check ---
        rider_cancelled = (not rider_accepted) and new_rider_patience <= 0.0
        driver_cancelled = (not driver_accepted) and new_driver_patience <= 0.0

        # --- Mood derivation ---
        new_rider_mood = patience_to_mood(new_rider_patience)
        new_driver_mood = patience_to_mood(new_driver_patience)

        both_accepted = rider_accepted and driver_accepted
        any_cancelled = rider_cancelled or driver_cancelled

        return SimResult(
            rider_accepted=rider_accepted,
            driver_accepted=driver_accepted,
            rider_cancelled=rider_cancelled,
            driver_cancelled=driver_cancelled,
            new_rider_patience=round(new_rider_patience, 4),
            new_driver_patience=round(new_driver_patience, 4),
            new_rider_mood=new_rider_mood,
            new_driver_mood=new_driver_mood,
            both_accepted=both_accepted,
            any_cancelled=any_cancelled,
        )
