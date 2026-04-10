"""FastAPI server for the Ride-Hailing Dynamic Pricing environment.

Exposes the DynamicPricingEnv (ride negotiation) over HTTP so the platform
can call /reset, /step, and /state.

Action format:
    POST /step  {"type": "propose_price", "payload": {"price": 18.50}}
"""

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.models import Observation


# ── Request / Response models ──────────────────────────────────────────────

class StepRequest(BaseModel):
    type: str = "propose_price"
    payload: Dict[str, float]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    episode_id: str


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ── App & environment ───────────────────────────────────────────────────────

app = FastAPI(title="Ride-Hailing Dynamic Pricing Environment")

# One environment instance per server process (single-session)
_env: Optional[DynamicPricingEnv] = None
_episode_id: str = ""


def _get_env(task_name: str = "easy") -> DynamicPricingEnv:
    global _env
    if _env is None:
        _env = DynamicPricingEnv(task_name=task_name)
    return _env


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Ride-Hailing Dynamic Pricing Environment",
        "status": "running",
        "endpoints": {
            "POST /reset": "Reset the environment and start a new episode",
            "POST /step": "Propose a price: {type: propose_price, payload: {price: X}}",
            "GET /state": "Get current environment state",
            "GET /schema": "Get action/observation JSON schemas",
            "GET /health": "Health check",
        },
    }


@app.post("/reset")
def reset(task_name: str = "easy", seed: Optional[int] = None):
    global _env, _episode_id
    _env = DynamicPricingEnv(task_name=task_name, seed=seed or 42)
    obs = _env.reset()
    _episode_id = str(uuid.uuid4())
    return ResetResponse(observation=obs.model_dump(), episode_id=_episode_id)


@app.post("/step")
def step(action: StepRequest):
    env = _get_env()
    result = env.step({"type": action.type, "payload": action.payload})
    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def get_state():
    env = _get_env()
    return env.state().model_dump()


@app.get("/schema")
def get_schema():
    return {
        "action_schema": {
            "type": "propose_price",
            "payload": {"price": "float (1.0 – 500.0)"},
        },
        "observation_schema": Observation.model_json_schema(),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
