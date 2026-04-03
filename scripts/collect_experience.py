"""Collect experience episodes for the reward-guided LLM policy.

Runs N episodes per task using the adaptive policy (best non-LLM policy),
records scenario features → price → outcome → reward, and saves them as
a JSON experience file that the RewardGuidedLLMPolicy reads at inference.

Usage:
    python3 scripts/collect_experience.py
    python3 scripts/collect_experience.py --task easy --episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.config import TASK_CONFIG
from baselines.adaptive_policy import AdaptivePolicy
from baselines.midpoint_policy import midpoint_policy

WEATHER = {0: "clear", 1: "rain", 2: "storm"}
TRAFFIC = {0: "low", 1: "medium", 2: "heavy"}
DEMAND  = {0: "low", 1: "medium", 2: "high"}
SUPPLY  = {0: "scarce", 1: "balanced", 2: "surplus"}


def collect_for_task(
    task_name: str,
    num_episodes: int,
    seed: int,
) -> list[dict]:
    """Run episodes and collect experience entries."""
    experiences = []
    completed = 0
    cancelled = 0
    timed_out = 0

    print(f"\n  Collecting {num_episodes} episodes for {task_name}...", flush=True)
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        init_obs = obs.model_dump()

        # Use adaptive policy (stateful, good completion rate)
        policy = AdaptivePolicy()
        policy.reset()

        done = False
        total_reward = 0.0
        prices_tried = []

        while not done:
            od = obs.model_dump()
            action = policy(od)
            price = action["payload"]["price"]
            prices_tried.append(price)

            result = env.step(action)
            total_reward += result.reward
            obs = result.observation
            done = result.done

        outcome = result.info["outcome"]

        # Build context summary string
        ctx = (f"{WEATHER[init_obs['weather_condition']]}, "
               f"{TRAFFIC[init_obs['traffic_level']]}, "
               f"demand={DEMAND[init_obs['demand_level']]}, "
               f"supply={SUPPLY[init_obs['supply_level']]}")

        entry = {
            "task": task_name,
            "seed": ep_seed,
            "initial_obs": {
                "rider_quoted_price": init_obs["rider_quoted_price"],
                "driver_quoted_price": init_obs["driver_quoted_price"],
                "price_gap": init_obs["price_gap"],
                "distance_km": init_obs["distance_km"],
                "estimated_duration_min": init_obs["estimated_duration_min"],
                "weather_condition": init_obs["weather_condition"],
                "traffic_level": init_obs["traffic_level"],
                "demand_level": init_obs["demand_level"],
                "supply_level": init_obs["supply_level"],
                "surge_multiplier": init_obs["surge_multiplier"],
                "commission_rate": init_obs["commission_rate"],
                "operational_cost": init_obs["operational_cost"],
                "max_steps": init_obs["max_steps"],
            },
            "gap": round(init_obs["price_gap"], 2),
            "context": ctx,
            "prices_tried": [round(p, 2) for p in prices_tried],
            "winning_price": outcome.get("final_price"),
            "completed": outcome["ride_completed"],
            "rider_cancelled": outcome.get("rider_cancelled", False),
            "driver_cancelled": outcome.get("driver_cancelled", False),
            "timed_out": outcome.get("timed_out", False),
            "termination_reason": outcome["termination_reason"],
            "steps_taken": outcome["steps_taken"],
            "platform_profit": outcome.get("platform_profit"),
            "total_reward": round(total_reward, 4),
        }
        experiences.append(entry)

        if outcome["ride_completed"]:
            completed += 1
        elif outcome["timed_out"]:
            timed_out += 1
        else:
            cancelled += 1

        if ep % 100 == 0:
            print(f"    ep {ep:>5d}/{num_episodes}  "
                  f"completed={completed}  cancelled={cancelled}  "
                  f"timed_out={timed_out}", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s: "
          f"{completed} completed, {cancelled} cancelled, {timed_out} timed out")

    # Sort by reward descending so best episodes are first
    experiences.sort(key=lambda e: e["total_reward"], reverse=True)
    return experiences


def main():
    parser = argparse.ArgumentParser(description="Collect experience for reward-guided LLM")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=500,
                        help="Episodes per task to collect")
    parser.add_argument("--seed", type=int, default=77777)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print("EXPERIENCE COLLECTION (Adaptive Policy)")
    print("=" * 70)

    for task in tasks:
        experiences = collect_for_task(task, args.episodes, args.seed)

        # Save all episodes
        path = os.path.join(data_dir, f"experience_{task}.json")
        with open(path, "w") as f:
            json.dump(experiences, f, indent=2, default=str)
        print(f"  Saved → {path} ({len(experiences)} episodes)")

        # Stats on completed
        completed_eps = [e for e in experiences if e["completed"]]
        if completed_eps:
            avg_reward = sum(e["total_reward"] for e in completed_eps) / len(completed_eps)
            avg_profit = sum(e["platform_profit"] for e in completed_eps) / len(completed_eps)
            avg_steps = sum(e["steps_taken"] for e in completed_eps) / len(completed_eps)
            print(f"  Completed: {len(completed_eps)}/{len(experiences)} "
                  f"(avg_reward={avg_reward:+.2f}, avg_profit=${avg_profit:.2f}, "
                  f"avg_steps={avg_steps:.1f})")
            print(f"  Best reward: {completed_eps[0]['total_reward']:+.2f} "
                  f"(price=${completed_eps[0]['winning_price']:.2f}, "
                  f"gap=${completed_eps[0]['gap']:.2f})")

    print(f"\nAll experience files in: {os.path.abspath(data_dir)}/")


if __name__ == "__main__":
    main()
