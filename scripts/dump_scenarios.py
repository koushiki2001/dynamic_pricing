"""Dump all training and evaluation scenario snapshots to JSON files.

For each task (easy/medium/hard), generates:
  - data/scenarios_{task}_train.json   — training scenario snapshots
  - data/scenarios_{task}_eval.json    — held-out eval scenario snapshots (unseen)

Both files contain only (observation, hidden_state) pairs — no policy is run
during generation. Eval scenarios use a different seed range so they are
guaranteed unseen by any agent trained on the training set.

To evaluate a policy against these scenarios, use graders.py or inference.py.

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
    """Generate held-out eval scenario snapshots (observation + hidden state only).

    Uses a separate seed range from training so these scenarios are unseen.
    No policy is run — evaluation happens separately via graders.py / inference.py.
    """
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
        if args.train_episodes > 0:
            t0 = time.time()
            print(f"  Generating {args.train_episodes} training scenarios...", end=" ", flush=True)
            train_scenarios = dump_training_scenarios(task, args.train_episodes, args.seed)
            train_path = os.path.join(data_dir, f"scenarios_{task}_train.json")
            with open(train_path, "w") as f:
                json.dump(train_scenarios, f, indent=2, default=str)
            print(f"done ({time.time()-t0:.1f}s) → {train_path}")

            # --- Count diversity stats ---
            weathers = set(); traffics = set(); demands = set()
            supplies = set(); times = set(); days = set()
            for s in train_scenarios:
                o = s["observation"]
                weathers.add(o["weather_label"]); traffics.add(o["traffic_label"])
                demands.add(o["demand_label"]); supplies.add(o["supply_label"])
                times.add(o["time_label"]); days.add(o["day_label"])
            print(f"  Diversity: weather={weathers} traffic={traffics} "
                  f"demand={demands} supply={supplies} time={times} day={days}")

        # --- Eval scenarios (with full trajectories) ---
        if args.eval_episodes > 0:
            t0 = time.time()
            print(f"  Running {args.eval_episodes} eval episodes with trajectories...", end=" ", flush=True)
            eval_scenarios = dump_eval_scenarios(task, args.eval_episodes, args.eval_seed)
            eval_path = os.path.join(data_dir, f"scenarios_{task}_eval.json")
            with open(eval_path, "w") as f:
                json.dump(eval_scenarios, f, indent=2, default=str)
            print(f"done ({time.time()-t0:.1f}s) → {eval_path}")


    print(f"\n{'='*70}")
    print(f"All scenarios dumped to: {os.path.abspath(data_dir)}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
