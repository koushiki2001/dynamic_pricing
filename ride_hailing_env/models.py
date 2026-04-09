from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Observable state the agent receives each step."""

    # --- Fixed per episode ---
    rider_quoted_price: float = Field(..., ge=0.0, description="Rider's stated max willingness (fixed)")
    driver_quoted_price: float = Field(..., ge=0.0, description="Driver's stated min ask (fixed)")
    price_gap: float = Field(..., ge=0.0, description="|rider_quoted - driver_quoted|")
    distance_km: float = Field(..., ge=0.1)
    estimated_duration_min: float = Field(..., ge=1.0)
    pickup_eta_min: float = Field(..., ge=0.0)
    demand_level: int = Field(..., ge=0, le=2, description="0=low, 1=medium, 2=high")
    supply_level: int = Field(..., ge=0, le=2, description="0=low, 1=medium, 2=high")
    surge_multiplier: float = Field(..., ge=1.0)
    weather_condition: int = Field(..., ge=0, le=2, description="0=clear, 1=rain, 2=storm")
    traffic_level: int = Field(..., ge=0, le=2, description="0=low, 1=medium, 2=heavy")
    time_of_day: int = Field(..., ge=0, le=3, description="0=morning, 1=afternoon, 2=evening, 3=night")
    day_type: int = Field(..., ge=0, le=2, description="0=weekday, 1=weekend, 2=holiday")
    commission_rate: float = Field(..., ge=0.0, le=0.5)
    operational_cost: float = Field(..., ge=0.0)
    max_steps: int = Field(..., ge=1)

    # --- Change every step ---
    step_number: int = Field(..., ge=0)
    rider_patience: float = Field(..., ge=0.0, le=1.0)
    driver_patience: float = Field(..., ge=0.0, le=1.0)
    rider_mood: str = Field(..., description="willing / hesitant / frustrated")
    driver_mood: str = Field(..., description="willing / hesitant / frustrated")
    last_rider_response: Optional[str] = Field(None, description="accepted / rejected / null")
    last_driver_response: Optional[str] = Field(None, description="accepted / rejected / null")
    last_proposed_price: Optional[float] = Field(None, description="Agent's previous proposal")


class HiddenState(BaseModel):
    """Hidden state the agent cannot observe.

    rider_max_willingness and driver_min_willingness are fixed for the entire
    episode — noise is baked in at scenario generation time, making per-step
    acceptance fully deterministic.

    rider_noise_bound: the unpredictability magnitude of this rider (how far
        their true ceiling was shifted from the base willingness). Used to
        derive penalty_threshold — a more unpredictable rider earns more slack.
    reward_threshold: minimum terminal reward the agent must exceed for a pass.
    penalty_threshold: maximum missed revenue (rider_max_willingness - price)
        the agent is allowed before the episode counts as a fail.
    """

    rider_max_willingness: float = Field(..., ge=0.0)
    driver_min_willingness: float = Field(..., ge=0.0)
    rider_patience_decay: float = Field(..., ge=0.0, le=1.0)
    driver_patience_decay: float = Field(..., ge=0.0, le=1.0)
    rider_noise_bound: float = Field(..., ge=0.0)
    reward_threshold: float = Field(...)
    penalty_threshold: float = Field(..., ge=0.0)


class Action(BaseModel):
    """Action the agent takes each step."""

    type: str = Field("propose_price", pattern="^propose_price$")
    payload: Dict[str, float] = Field(..., description='{"price": <float>}')

    @property
    def price(self) -> float:
        return self.payload.get("price", 0.0)


class StepResult(BaseModel):
    """Result returned by env.step()."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeOutcome(BaseModel):
    """Terminal outcome of an episode."""

    ride_completed: bool
    rider_accepted: bool
    driver_accepted: bool
    rider_cancelled: bool
    driver_cancelled: bool
    timed_out: bool
    final_price: Optional[float] = None
    platform_profit: Optional[float] = None
    steps_taken: int
    termination_reason: str
