"""FastAPI server for the Dynamic Pricing environment — no openenv dependency."""

import random
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


# ── Data models (replaces openenv base classes) ──

class PricingAction(BaseModel):
    move: str  # "decrease", "hold", "increase"


class PricingObservation(BaseModel):
    demand: int
    supply: int
    market_factor: float
    price_multiplier: float
    accepted_rides: int
    revenue: float
    message: str


class PricingState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    demand: int = 20
    supply: int = 15
    market_factor: float = 1.0
    price_multiplier: float = 1.0
    accepted_rides: int = 0
    revenue: float = 0.0
    total_reward: float = 0.0
    max_steps: int = 10


# ── Environment ──

class DynamicPricingEnvironment:
    def __init__(self):
        self._state = PricingState(episode_id=str(uuid.uuid4()))

    def _make_observation(self, message: str) -> PricingObservation:
        return PricingObservation(
            demand=self._state.demand,
            supply=self._state.supply,
            market_factor=round(self._state.market_factor, 2),
            price_multiplier=round(self._state.price_multiplier, 2),
            accepted_rides=self._state.accepted_rides,
            revenue=round(self._state.revenue, 2),
            message=message,
        )

    def reset(self) -> PricingObservation:
        self._state = PricingState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            demand=random.randint(15, 25),
            supply=random.randint(12, 20),
            market_factor=round(random.uniform(0.8, 1.3), 2),
            price_multiplier=1.0,
            accepted_rides=0,
            revenue=0.0,
            total_reward=0.0,
            max_steps=10,
        )
        return self._make_observation("New pricing episode started.")

    def step(self, action: PricingAction) -> PricingObservation:
        self._state.step_count += 1

        if action.move == "decrease":
            self._state.price_multiplier = max(0.7, self._state.price_multiplier - 0.1)
        elif action.move == "increase":
            self._state.price_multiplier = min(1.8, self._state.price_multiplier + 0.1)
        elif action.move == "hold":
            pass
        else:
            return self._make_observation("Invalid action. Use one of: decrease, hold, increase.")

        raw_acceptance = self._state.market_factor - 0.6 * (self._state.price_multiplier - 1.0)
        acceptance_rate = max(0.1, min(1.0, raw_acceptance))
        accepted_rides = min(self._state.supply, int(self._state.demand * acceptance_rate))
        revenue = accepted_rides * 10.0 * self._state.price_multiplier
        unserved_demand = max(0, self._state.demand - accepted_rides)
        overpricing_penalty = max(0.0, (self._state.price_multiplier - 1.3) * 8.0)
        reward = revenue - (unserved_demand * 2.0) - overpricing_penalty

        self._state.accepted_rides = accepted_rides
        self._state.revenue = revenue
        self._state.total_reward += reward
        self._state.demand = max(5, self._state.demand + random.randint(-3, 4))
        self._state.supply = max(5, self._state.supply + random.randint(-2, 3))
        self._state.market_factor = round(
            min(1.5, max(0.7, self._state.market_factor + random.uniform(-0.1, 0.1))), 2
        )

        done = self._state.step_count >= self._state.max_steps
        if done:
            return self._make_observation(f"Episode finished. Total reward = {round(self._state.total_reward, 2)}")
        return self._make_observation(f"Action={action.move}, reward={round(reward, 2)}")

    @property
    def state(self) -> PricingState:
        return self._state


# ── FastAPI app ──

app = FastAPI(title="Ride-Hailing Dynamic Pricing Environment")
env = DynamicPricingEnvironment()


@app.get("/")
def root():
    return {
        "name": "Ride-Hailing Dynamic Pricing Environment",
        "status": "running",
        "endpoints": {
            "POST /reset": "Reset the environment and start a new episode",
            "POST /step": "Take an action (decrease, hold, increase)",
            "GET /state": "Get current environment state",
            "GET /schema": "Get action/observation JSON schemas",
            "GET /health": "Health check",
        },
    }


@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step")
def step(action: PricingAction):
    obs = env.step(action)
    done = env.state.step_count >= env.state.max_steps
    return {
        "observation": obs.model_dump(),
        "done": done,
        "reward": env.state.total_reward,
    }


@app.get("/state")
def get_state():
    return env.state.model_dump()


@app.get("/schema")
def get_schema():
    return {
        "action_schema": PricingAction.model_json_schema(),
        "observation_schema": PricingObservation.model_json_schema(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
