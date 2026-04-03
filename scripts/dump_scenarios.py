"""Dump all training and evaluation test scenarios to JSON files.

For each task (easy/medium/hard), generates:
  - data/scenarios_{task}_train.json   — all training episode scenarios
  - data/scenarios_{task}_eval.json    — all eval episodes with full trajectories

Each entry includes:
  - scenario context (observation + hidden state)
  - evaluation entries include step-by-step actions, responses, and outcome

Usage:
    python3 scripts/dump_scenarios.py
    python3 scripts/dump_scenarios.py --task easy --train-episodes 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.config import TASK_CONFIG
from ride_hailing_env.scenario_generator import ScenarioGenerator


WEATHER_MAP = {0: "clear", 1: "rain", 2: "storm"}
TRAFFIC_MAP = {0: "low", 1: "moderate", 2: "heavy"}
DEMAND_MAP = {0: "low", 1: "medium", 2: "high"}
SUPPLY_MAP = {0: "scarce", 1: "balanced", 2: "surplus"}
TIME_MAP = {0: "morning", 1: "afternoon", 2: "evening", 3: "night"}
DAY_MAP = {0: "weekday", 1: "weekend", 2: "holiday"}


def obs_to_dict(obs) -> dict:
    """Convert observation to a human-readable dict."""
    d = obs.model_dump()
    d["weather_label"] = WEATHER_MAP.get(d["weather_condition"], str(d["weather_condition"]))
    d["traffic_label"] = TRAFFIC_MAP.get(d["traffic_level"], str(d["traffic_level"]))
    d["demand_label"] = DEMAND_MAP.get(d["demand_level"], str(d["demand_level"]))
    d["supply_label"] = SUPPLY_MAP.get(d["supply_level"], str(d["supply_level"]))
    d["time_label"] = TIME_MAP.get(d["time_of_day"], str(d["time_of_day"]))
    d["day_label"] = DAY_MAP.get(d["day_type"], str(d["day_type"]))
    return d


def hidden_to_dict(hidden) -> dict:
    return hidden.model_dump()


def dump_training_scenarios(task_name: str, num_episodes: int, seed: int) -> list[dict]:
    """Generate all training scenarios (observation + hidden) for a task."""
    scenarios = []
    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep
        gen = ScenarioGenerator(ep_seed)
        obs, hidden = gen.generate(task_name)
        scenarios.append({
            "episode": ep,
            "seed": ep_seed,
            "observation": obs_to_dict(obs),
            "hidden_state": hidden_to_dict(hidden),
        })
    return scenarios


def dump_eval_scenarios(task_name: str, num_episodes: int, seed: int) -> list[dict]:
    """Run evaluation episodes, recording full trajectories and outcomes.

    Uses a simple midpoint policy to produce realistic trajectories so we
    can see how each scenario plays out end-to-end.
    """
    scenarios = []
    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        init_obs = obs_to_dict(obs)
        init_hidden = hidden_to_dict(env._hidden)

        steps_log = []
        done = False
        step_num = 0

        while not done:
            step_num += 1
            obs_dict = obs.model_dump()
            rider_q = obs_dict["rider_quoted_price"]
            driver_q = obs_dict["driver_quoted_price"]

            # Midpoint policy (simple deterministic for reproducibility)
            price = round((rider_q + driver_q) / 2.0, 2)

            action = {"type": "propose_price", "payload": {"price": price}}
            result = env.step(action)

            step_entry = {
                "step": step_num,
                "proposed_price": price,
                "rider_response": result.observation.last_rider_response,
                "driver_response": result.observation.last_driver_response,
                "rider_patience": result.observation.rider_patience,
                "driver_patience": result.observation.driver_patience,
                "rider_mood": result.observation.rider_mood,
                "driver_mood": result.observation.driver_mood,
                "reward": round(result.reward, 4),
                "done": result.done,
            }
            steps_log.append(step_entry)

            obs = result.observation
            done = result.done

        outcome = result.info["outcome"]
        scenario_entry = {
            "episode": ep + 1,
            "seed": ep_seed,
            "initial_observation": init_obs,
            "hidden_state": init_hidden,
            "steps": steps_log,
            "outcome": {
                "ride_completed": outcome["ride_completed"],
                "rider_cancelled": outcome.get("rider_cancelled", False),
                "driver_cancelled": outcome.get("driver_cancelled", False),
                "timed_out": outcome.get("timed_out", False),
                "termination_reason": outcome["termination_reason"],
                "final_price": outcome.get("final_price"),
                "platform_profit": outcome.get("platform_profit"),
                "steps_taken": outcome["steps_taken"],
            },
        }
        scenarios.append(scenario_entry)

    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Dump training/eval scenarios to JSON")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--train-episodes", type=int, default=5000,
                        help="Number of training scenarios to dump per task")
    parser.add_argument("--eval-episodes", type=int, default=200,
                        help="Number of eval episodes with full trajectory per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=99999)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"DUMPING SCENARIOS: {task}")
        print(f"{'='*70}")

        # --- Training scenarios (observation + hidden only) ---
        t0 = time.time()
        print(f"  Generating {args.train_episodes} training scenarios...", end=" ", flush=True)
        train_scenarios = dump_training_scenarios(task, args.train_episodes, args.seed)
        train_path = os.path.join(data_dir, f"scenarios_{task}_train.json")
        with open(train_path, "w") as f:
            json.dump(train_scenarios, f, indent=2, default=str)
        print(f"done ({time.time()-t0:.1f}s) → {train_path}")

        # --- Count diversity stats ---
        weathers = set()
        traffics = set()
        demands = set()
        supplies = set()
        times = set()
        days = set()
        for s in train_scenarios:
            o = s["observation"]
            weathers.add(o["weather_label"])
            traffics.add(o["traffic_label"])
            demands.add(o["demand_label"])
            supplies.add(o["supply_label"])
            times.add(o["time_label"])
            days.add(o["day_label"])
        print(f"  Diversity: weather={weathers} traffic={traffics} "
              f"demand={demands} supply={supplies} time={times} day={days}")

        # --- Eval scenarios (with full trajectories) ---
        t0 = time.time()
        print(f"  Running {args.eval_episodes} eval episodes with trajectories...", end=" ", flush=True)
        eval_scenarios = dump_eval_scenarios(task, args.eval_episodes, args.eval_seed)
        eval_path = os.path.join(data_dir, f"scenarios_{task}_eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_scenarios, f, indent=2, default=str)
        print(f"done ({time.time()-t0:.1f}s) → {eval_path}")

        # --- Outcome summary ---
        completed = sum(1 for s in eval_scenarios if s["outcome"]["ride_completed"])
        cancelled = sum(1 for s in eval_scenarios if s["outcome"]["termination_reason"] == "cancelled")
        rider_cancel = sum(1 for s in eval_scenarios if s["outcome"]["rider_cancelled"])
        driver_cancel = sum(1 for s in eval_scenarios if s["outcome"]["driver_cancelled"])
        timed_out = sum(1 for s in eval_scenarios if s["outcome"]["timed_out"])
        n = len(eval_scenarios)

        print(f"\n  Eval Outcome Summary ({n} episodes):")
        print(f"    Completed:          {completed:4d} ({completed/n:.1%})")
        print(f"    Cancelled:          {cancelled:4d} ({cancelled/n:.1%})")
        print(f"      → Rider cancelled:  {rider_cancel:4d}")
        print(f"      → Driver cancelled: {driver_cancel:4d}")
        print(f"    Timed out:          {timed_out:4d} ({timed_out/n:.1%})")

        # --- Cancellation deep-dive ---
        if cancelled > 0:
            print(f"\n  Cancellation Details (first 5):")
            cancel_episodes = [s for s in eval_scenarios if s["outcome"]["termination_reason"] == "cancelled"]
            for s in cancel_episodes[:5]:
                o = s["initial_observation"]
                print(f"    Episode {s['episode']} (seed={s['seed']}): "
                      f"gap=${o['price_gap']:.2f}, "
                      f"weather={o['weather_label']}, traffic={o['traffic_label']}, "
                      f"demand={o['demand_label']}, supply={o['supply_label']}, "
                      f"rider_cancel={s['outcome']['rider_cancelled']}, "
                      f"driver_cancel={s['outcome']['driver_cancelled']}, "
                      f"steps={s['outcome']['steps_taken']}")

    print(f"\n{'='*70}")
    print(f"All scenarios dumped to: {os.path.abspath(data_dir)}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
