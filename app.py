"""FastAPI server for the Ride-Hailing Dynamic Pricing environment.

Exposes the DynamicPricingEnv (ride negotiation) over HTTP so the platform
can call /reset, /step, and /state.

Action format:
    POST /step  {"type": "propose_price", "payload": {"price": X}}

Phase 5 additions:
    POST /demo/run   Run a full episode with the trained platform LLM and
                     return the complete step-by-step trace.
    GET  /models     Report which model checkpoints are loaded.

Environment variables:
    PLATFORM_CKPT   Path to trained platform LoRA (default: checkpoints/phase4/platform_lora)
    SIM_CKPT        Path to trained simulator LoRA (default: checkpoints/phase3/simulator_lora)
    LOAD_MODELS     Set to "0" to skip model loading (rule-based only mode)
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.models import Observation


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

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


class DemoRequest(BaseModel):
    task: str = "easy"
    mode: str = "trained"   # "trained" | "untrained" | "rule_based"
    show_hidden: bool = True


class DemoStepRecord(BaseModel):
    step: int
    proposed_price: float
    rider_accepted: bool
    driver_accepted: bool
    reward: float
    done: bool


class DemoResponse(BaseModel):
    task: str
    mode: str
    steps: List[DemoStepRecord]
    outcome: str            # "completed" | "cancelled" | "timed_out"
    final_reward: float
    platform_profit: Optional[float]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ride-Hailing Dynamic Pricing",
    description="Multi-agent RL dynamic pricing environment with trained LLM platform",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# Model loading at startup
# ---------------------------------------------------------------------------

PLATFORM_CKPT = os.getenv("PLATFORM_CKPT", "checkpoints/phase4/platform_lora")
SIM_CKPT      = os.getenv("SIM_CKPT",      "checkpoints/phase3/simulator_lora")
LOAD_MODELS   = os.getenv("LOAD_MODELS", "1") != "0"

_platform_model     = None
_platform_tokenizer = None
_simulator_agent    = None
_models_loaded      = False


@app.on_event("startup")
def _load_trained_models():
    global _platform_model, _platform_tokenizer, _simulator_agent, _models_loaded
    if not LOAD_MODELS:
        print("[APP] LOAD_MODELS=0 — running in rule-based only mode.")
        return
    try:
        from training.model_loader import load_platform_model, load_simulator_model
        from training.simulator_agent import SimulatorAgent

        p_lora = PLATFORM_CKPT if Path(PLATFORM_CKPT).exists() else None
        _platform_model, _platform_tokenizer = load_platform_model(lora_path=p_lora)
        print(f"[APP] Platform loaded: {'trained LoRA' if p_lora else 'base model (no checkpoint found)'}")

        if Path(SIM_CKPT).exists():
            sim_model, sim_tok = load_simulator_model(lora_path=SIM_CKPT)
            sim_model.eval()
            _simulator_agent = SimulatorAgent(sim_model, sim_tok)
            print(f"[APP] Simulator loaded from {SIM_CKPT}")
        else:
            print(f"[APP] Simulator checkpoint not found ({SIM_CKPT}) — /demo will use rule-based fallback")

        _models_loaded = True
    except Exception as e:
        print(f"[APP] Model loading failed (rule-based fallback active): {e}")


# ---------------------------------------------------------------------------
# Environment (rule-based, for /reset /step /state)
# ---------------------------------------------------------------------------

_env: Optional[DynamicPricingEnv] = None
_episode_id: str = ""


def _get_env(task_name: str = "easy") -> DynamicPricingEnv:
    global _env
    if _env is None:
        _env = DynamicPricingEnv(task_name=task_name)
    return _env


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Ride-Hailing Dynamic Pricing Environment",
        "version": "2.0.0",
        "models_loaded": _models_loaded,
        "platform_ckpt": PLATFORM_CKPT if Path(PLATFORM_CKPT).exists() else None,
        "simulator_ckpt": SIM_CKPT if Path(SIM_CKPT).exists() else None,
        "endpoints": {
            "POST /reset":    "Reset environment, start new episode",
            "POST /step":     "Propose a price: {type, payload: {price}}",
            "GET  /state":    "Current observation",
            "GET  /schema":   "Action/observation schemas",
            "GET  /health":   "Health check",
            "GET  /models":   "Which model checkpoints are loaded",
            "POST /demo/run": "Run a full LLM episode, return step trace",
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
    return {"status": "ok", "models_loaded": _models_loaded}


@app.get("/models")
def models_status():
    return {
        "platform_checkpoint": PLATFORM_CKPT,
        "platform_loaded":     _platform_model is not None,
        "simulator_checkpoint": SIM_CKPT,
        "simulator_loaded":    _simulator_agent is not None,
        "mode": "trained_llm" if _models_loaded else "rule_based_only",
    }


# ---------------------------------------------------------------------------
# Demo endpoint
# ---------------------------------------------------------------------------

@app.post("/demo/run", response_model=DemoResponse)
def demo_run(req: DemoRequest):
    """Run a full negotiation episode with the platform LLM.

    Returns the complete step-by-step trace — suitable for frontend visualisation.
    Uses rule-based simulator fallback if simulator checkpoint is not loaded.
    """
    if req.mode == "trained" and not _models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Trained models not loaded. Ensure checkpoints exist and LOAD_MODELS=1.",
        )

    import json as _json
    import re

    try:
        from training.prompt_builders import build_platform_prompt
        import torch
    except ImportError:
        raise HTTPException(status_code=503, detail="Training dependencies not installed.")

    env    = DynamicPricingEnv(task_name=req.task)
    obs    = env.reset()
    hidden = env.get_hidden_state()

    p_model   = _platform_model
    p_tok     = _platform_tokenizer
    sim_agent = _simulator_agent if req.mode != "rule_based" else None

    if p_model is None:
        raise HTTPException(status_code=503, detail="Platform model not available.")

    steps_trace: List[DemoStepRecord] = []
    done = False
    final_reward = 0.0
    outcome_str = "timed_out"
    platform_profit = None

    while not done:
        prompt = build_platform_prompt(obs)
        inputs = p_tok(prompt, return_tensors="pt").to(p_model.device)

        with torch.inference_mode():
            out = p_model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.7,
                do_sample=True,
                pad_token_id=p_tok.eos_token_id,
            )
        completion = p_tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        price = None
        try:
            price = float(_json.loads(completion).get("price", 0))
        except Exception:
            nums = re.findall(r"\b\d+\.?\d*\b", completion)
            if nums:
                price = float(nums[0])
        if not price:
            price = (obs.rider_quoted_price + obs.driver_quoted_price) / 2.0
        price = max(1.0, min(500.0, price))

        override = None
        if sim_agent is not None:
            decision = sim_agent.decide(hidden, obs, price)
            override  = {"rider": decision.rider_accept, "driver": decision.driver_accept}
            r_accept  = decision.rider_accept
            d_accept  = decision.driver_accept
        else:
            r_accept = price <= hidden.rider_max_willingness
            d_accept = price >= hidden.driver_min_willingness

        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        result = env.step(action, override_decision=override)

        steps_trace.append(DemoStepRecord(
            step=obs.step_number + 1,
            proposed_price=round(price, 2),
            rider_accepted=r_accept,
            driver_accepted=d_accept,
            reward=result.reward,
            done=result.done,
        ))

        final_reward = result.reward
        done         = result.done
        obs          = result.observation

        if done:
            ep_outcome = result.info.get("outcome", {})
            if ep_outcome.get("ride_completed"):
                outcome_str     = "completed"
                platform_profit = ep_outcome.get("platform_profit")
            elif ep_outcome.get("rider_cancelled") or ep_outcome.get("driver_cancelled"):
                outcome_str = "cancelled"
            else:
                outcome_str = "timed_out"

    return DemoResponse(
        task=req.task,
        mode=req.mode,
        steps=steps_trace,
        outcome=outcome_str,
        final_reward=round(final_reward, 4),
        platform_profit=round(platform_profit, 4) if platform_profit is not None else None,
    )


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
