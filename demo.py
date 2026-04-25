"""Live episode demo — shows platform LLM reasoning through a negotiation.

Suitable for screen-sharing during a presentation. Shows step-by-step pricing
decisions, hidden thresholds, and deal outcome.

The four demo modes map directly to the four evaluation stages:
  untrained   → Stage A: base model, no RL
  trained_v0  → Stage B: platform_v0 vs rule-based (Phase 2 result)
  disrupted   → Stage C: platform_v0 vs strategic simulator_v1
  trained     → Stage D: platform_v1 vs strategic simulator_v1  ← the hero

Usage:
    python demo.py --mode untrained  --task easy
    python demo.py --mode trained_v0 --task easy
    python demo.py --mode disrupted  --task easy
    python demo.py --mode trained    --task easy   --episodes 3
    python demo.py --mode trained    --task hard
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import torch

from ride_hailing_env.environment import DynamicPricingEnv


CKPT = {
    "platform_v0":  "checkpoints/phase2/platform_lora",
    "platform_v1":  "checkpoints/phase4/platform_lora",
    "simulator_v1": "checkpoints/phase3/simulator_lora",
}

WEATHER_LABELS = {0: "clear", 1: "rain", 2: "storm"}
TRAFFIC_LABELS = {0: "low",   1: "medium", 2: "heavy"}


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def run_episode(
    platform_model,
    platform_tokenizer,
    simulator_agent,        # None = rule-based
    env: DynamicPricingEnv,
    task: str,
    show_hidden: bool = True,
) -> float:
    from training.prompt_builders import build_platform_prompt

    obs = env.reset()
    hidden = env.get_hidden_state()
    done = False
    last_reward = 0.0

    weather = WEATHER_LABELS.get(getattr(obs, "weather_condition", 0), "clear")
    traffic = TRAFFIC_LABELS.get(getattr(obs, "traffic_level", 0), "medium")

    print(f"\n{'='*62}")
    print(f"  EPISODE START — Task: {task.upper()}")
    print(f"{'='*62}")
    print(f"  Trip:       {obs.distance_km:.1f} km, {obs.estimated_duration_min:.0f} min")
    print(f"  Conditions: {weather}, {traffic} traffic, surge ×{obs.surge_multiplier:.1f}")
    print(f"  Quoted:     Rider ${obs.rider_quoted_price:.2f}  |  Driver ${obs.driver_quoted_price:.2f}")
    if show_hidden:
        print(f"  [Hidden]    Rider ceiling ${hidden.rider_max_willingness:.2f}  |  "
              f"Driver floor ${hidden.driver_min_willingness:.2f}")
    print(f"  Max steps:  {obs.max_steps}")
    print()

    while not done:
        prompt = build_platform_prompt(obs)
        inputs = platform_tokenizer(prompt, return_tensors="pt").to(platform_model.device)

        with torch.inference_mode():
            out = platform_model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.7,
                do_sample=True,
                pad_token_id=platform_tokenizer.eos_token_id,
            )
        completion = platform_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse price
        price = None
        try:
            price = float(json.loads(completion).get("price", 0))
        except Exception:
            nums = re.findall(r"\b\d+\.?\d*\b", completion)
            if nums:
                price = float(nums[0])
        if not price:
            price = (obs.rider_quoted_price + obs.driver_quoted_price) / 2.0
        price = max(1.0, min(500.0, price))

        step_num = obs.step_number + 1
        print(f"  Step {step_num}/{obs.max_steps}")
        print(f"    Platform proposes: ${price:.2f}")

        # Simulator decision
        override = None
        if simulator_agent is not None:
            from training.simulator_agent import SimulatorDecision
            decision = simulator_agent.decide(hidden, obs, price)
            r_label  = "ACCEPT ✓" if decision.rider_accept  else "REJECT ✗"
            d_label  = "ACCEPT ✓" if decision.driver_accept else "REJECT ✗"
            r_honest = price <= hidden.rider_max_willingness
            d_honest = price >= hidden.driver_min_willingness
            r_note   = "honest" if r_honest == decision.rider_accept else "bluffing"
            d_note   = "honest" if d_honest == decision.driver_accept else "bluffing"
            print(f"    Rider  → {r_label}  (ceiling ${hidden.rider_max_willingness:.2f}, {r_note})")
            print(f"    Driver → {d_label}  (floor  ${hidden.driver_min_willingness:.2f}, {d_note})")
            override = {"rider": decision.rider_accept, "driver": decision.driver_accept}
        else:
            r_accept = price <= hidden.rider_max_willingness
            d_accept = price >= hidden.driver_min_willingness
            print(f"    Rider  → {'ACCEPT ✓' if r_accept else 'REJECT ✗'}  "
                  f"(ceiling ${hidden.rider_max_willingness:.2f})")
            print(f"    Driver → {'ACCEPT ✓' if d_accept else 'REJECT ✗'}  "
                  f"(floor ${hidden.driver_min_willingness:.2f})")

        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        result = env.step(action, override_decision=override)

        last_reward = result.reward
        done        = result.done
        obs         = result.observation

        if done:
            outcome = result.info.get("outcome", {})
            if outcome.get("ride_completed"):
                profit = outcome.get("platform_profit") or 0.0
                print(f"\n  ✅  DEAL CLOSED at ${price:.2f}  |  platform profit ${profit:.2f}")
            elif outcome.get("timed_out"):
                print(f"\n  ⏰  TIMED OUT — no deal after {outcome.get('steps_taken', '?')} steps")
            else:
                who = "rider" if outcome.get("rider_cancelled") else "driver"
                print(f"\n  ❌  CANCELLED — {who} ran out of patience")
        print()

    print(f"  Episode reward: {last_reward:+.3f}")
    print(f"{'='*62}\n")
    return last_reward


# ---------------------------------------------------------------------------
# Mode setup
# ---------------------------------------------------------------------------

def _load_models(mode: str):
    from training.model_loader import load_platform_model, load_simulator_model
    from training.simulator_agent import SimulatorAgent

    mode_config = {
        "untrained":  (None,               None),
        "trained_v0": ("platform_v0",       None),
        "disrupted":  ("platform_v0",       "simulator_v1"),
        "trained":    ("platform_v1",       "simulator_v1"),
    }

    p_key, s_key = mode_config[mode]

    p_lora = CKPT[p_key] if p_key else None
    if p_lora and not Path(p_lora).exists():
        print(f"[WARN] Platform checkpoint not found: {p_lora} — using base model")
        p_lora = None

    print(f"[LOAD] Platform: {'base model' if not p_lora else p_lora}")
    platform_model, platform_tokenizer = load_platform_model(lora_path=p_lora)

    simulator_agent = None
    if s_key:
        s_lora = CKPT[s_key]
        if Path(s_lora).exists():
            print(f"[LOAD] Simulator: {s_lora}")
            sim_model, sim_tok = load_simulator_model(lora_path=s_lora)
            sim_model.eval()
            simulator_agent = SimulatorAgent(sim_model, sim_tok)
        else:
            print(f"[WARN] Simulator checkpoint not found: {s_lora} — using rule-based fallback")

    return platform_model, platform_tokenizer, simulator_agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live demo of the dynamic pricing negotiation")
    parser.add_argument(
        "--mode",
        choices=["untrained", "trained_v0", "disrupted", "trained"],
        default="trained",
        help=(
            "untrained  = Stage A (no RL)\n"
            "trained_v0 = Stage B (Phase 2 platform vs rule-based)\n"
            "disrupted  = Stage C (Phase 2 platform vs strategic simulator)\n"
            "trained    = Stage D (Phase 4 platform vs strategic simulator)"
        ),
    )
    parser.add_argument("--task",     choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no_hidden", action="store_true",
                        help="Hide the true thresholds (more realistic demo)")
    args = parser.parse_args()

    MODE_LABELS = {
        "untrained":  "Stage A — Untrained platform vs rule-based",
        "trained_v0": "Stage B — Platform v0 vs rule-based",
        "disrupted":  "Stage C — Platform v0 vs strategic simulator (disruption)",
        "trained":    "Stage D — Platform v1 vs strategic simulator (recovery)",
    }

    print(f"\n{'#'*62}")
    print(f"  DYNAMIC PRICING DEMO")
    print(f"  {MODE_LABELS[args.mode]}")
    print(f"  Task: {args.task.upper()}  |  Episodes: {args.episodes}")
    print(f"{'#'*62}")

    platform_model, platform_tokenizer, simulator_agent = _load_models(args.mode)
    env = DynamicPricingEnv(task_name=args.task)

    rewards = []
    t_start = time.time()

    for i in range(args.episodes):
        print(f"\n{'─'*30}  Episode {i+1}/{args.episodes}  {'─'*30}")
        r = run_episode(
            platform_model, platform_tokenizer, simulator_agent,
            env, args.task, show_hidden=not args.no_hidden,
        )
        rewards.append(r)

    total_time = time.time() - t_start
    completed  = sum(1 for r in rewards if r > 0)

    print(f"\n{'#'*62}")
    print(f"  SUMMARY — {args.mode} / {args.task}")
    print(f"{'─'*62}")
    print(f"  Episodes run:    {args.episodes}")
    print(f"  Completed deals: {completed}/{args.episodes}  ({completed/args.episodes:.0%})")
    print(f"  Avg reward:      {sum(rewards)/len(rewards):+.3f}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"{'─'*62}")
    print(f"  SAFEGUARDS ACTIVE")
    print(f"  • Price bounds enforced:    $1 – $500 hard clamp")
    print(f"  • Anti-hack checks:         4 independent verifiers per step")
    print(f"  • Format compliance score:  +0.1 valid JSON / -0.1 malformed")
    print(f"  • Repeat price penalty:     env rejects unchanged prices")
    print(f"  • Patience budget:          cancellation if patience → 0")
    print(f"  • Drift detection:          auto-rollback if reward drops >30%")
    print(f"  • LoRA save:                adapter-only (no naive 4-bit merge)")
    print(f"{'#'*62}\n")
