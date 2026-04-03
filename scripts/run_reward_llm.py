"""Run the reward-guided LLM policy and compare against base OpenAI policy.

Evaluates both the base LLM (zero-shot) and the reward-guided wrapper
side by side, printing verbose per-episode results.

Usage:
    # First collect experience (fast, no API calls):
    python3 scripts/collect_experience.py

    # Then run the comparison:
    python3 scripts/run_reward_llm.py --task easy --episodes 10
    python3 scripts/run_reward_llm.py --task all --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.config import TASK_CONFIG
from baselines.openai_policy import OpenAIPolicy
from baselines.reward_guided_llm_policy import RewardGuidedLLMPolicy

WEATHER = {0: "clear", 1: "rain", 2: "storm"}
TRAFFIC = {0: "low", 1: "moderate", 2: "heavy"}
DEMAND  = {0: "low", 1: "medium", 2: "high"}
SUPPLY  = {0: "scarce", 1: "balanced", 2: "surplus"}


def run_policy(
    task_name: str,
    policy,
    policy_name: str,
    num_episodes: int,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    results = {
        "policy": policy_name, "task": task_name,
        "episodes": [], "num_episodes": num_episodes,
    }
    totals = {"completed": 0, "cancelled": 0, "rider_cancel": 0,
              "driver_cancel": 0, "timeout": 0, "reward": 0.0,
              "profit": 0.0, "comp_cnt": 0, "steps": 0}

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        init_obs = obs.model_dump()

        policy.reset()
        done = False
        ep_reward = 0.0
        steps_log = []

        while not done:
            od = obs.model_dump()
            t0 = time.time()
            action = policy(od)
            latency = time.time() - t0
            price = action["payload"]["price"]

            result = env.step(action)
            ep_reward += result.reward

            steps_log.append({
                "step": result.observation.step_number,
                "price": round(price, 2),
                "rider": result.observation.last_rider_response,
                "driver": result.observation.last_driver_response,
                "reward": round(result.reward, 4),
                "latency_s": round(latency, 2),
            })

            obs = result.observation
            done = result.done

        outcome = result.info["outcome"]
        reason = outcome["termination_reason"]
        fp = outcome.get("final_price")

        if outcome["ride_completed"]:
            totals["completed"] += 1
            totals["comp_cnt"] += 1
            totals["profit"] += outcome.get("platform_profit", 0)
        elif outcome["timed_out"]:
            totals["timeout"] += 1
        else:
            totals["cancelled"] += 1
            if outcome.get("rider_cancelled"):
                totals["rider_cancel"] += 1
            if outcome.get("driver_cancelled"):
                totals["driver_cancel"] += 1
        totals["reward"] += ep_reward
        totals["steps"] += outcome["steps_taken"]

        results["episodes"].append({
            "episode": ep, "seed": ep_seed,
            "steps": steps_log,
            "outcome": reason,
            "final_price": fp,
            "episode_reward": round(ep_reward, 4),
        })

        if verbose:
            w = WEATHER.get(init_obs["weather_condition"], "?")
            t = TRAFFIC.get(init_obs["traffic_level"], "?")
            d = DEMAND.get(init_obs["demand_level"], "?")
            s = SUPPLY.get(init_obs["supply_level"], "?")
            gap = init_obs["price_gap"]

            cancel_who = ""
            if reason == "cancelled":
                cancel_who = " (rider)" if outcome.get("rider_cancelled") else " (driver)"

            price_str = f"${fp:.2f}" if fp else "N/A"
            step_prices = " → ".join(f"${s['price']}" for s in steps_log)

            print(f"    ep {ep:>3d}/{num_episodes} │ "
                  f"gap=${gap:6.2f} {w:<5s} {t:<8s} {d}/{s} │ "
                  f"{reason}{cancel_who:<12s} {outcome['steps_taken']} steps │ "
                  f"price={price_str} │ rwd={ep_reward:+.2f}")
            print(f"           │ proposals: {step_prices}")

    n = num_episodes
    cc = totals["comp_cnt"] or 1
    results["summary"] = {
        "completion_rate": round(totals["completed"] / n, 4),
        "cancellation_rate": round(totals["cancelled"] / n, 4),
        "rider_cancellations": totals["rider_cancel"],
        "driver_cancellations": totals["driver_cancel"],
        "timeout_rate": round(totals["timeout"] / n, 4),
        "avg_reward": round(totals["reward"] / n, 4),
        "avg_profit": round(totals["profit"] / cc, 4) if totals["completed"] else 0.0,
        "avg_steps": round(totals["steps"] / n, 2),
    }
    return results


def print_summary(res: Dict):
    s = res["summary"]
    n = res["num_episodes"]
    c = int(s["completion_rate"] * n)
    x = int(s["cancellation_rate"] * n)
    t = int(s["timeout_rate"] * n)
    print(f"\n    ┌─────────────────────────────────────────────────────────┐")
    print(f"    │  {res['policy']:^20s}  on  {res['task']:^8s}  ({n} episodes)   │")
    print(f"    ├─────────────────────────────────────────────────────────┤")
    print(f"    │  Completed:    {c:>4d}/{n}  ({s['completion_rate']:.1%})                   │")
    print(f"    │  Cancelled:    {x:>4d}/{n}  ({s['cancellation_rate']:.1%})                   │")
    print(f"    │    → rider:    {s['rider_cancellations']:>4d}                                │")
    print(f"    │    → driver:   {s['driver_cancellations']:>4d}                                │")
    print(f"    │  Timed out:    {t:>4d}/{n}  ({s['timeout_rate']:.1%})                   │")
    print(f"    │  Avg reward:   {s['avg_reward']:>+8.4f}                            │")
    print(f"    │  Avg profit:   ${s['avg_profit']:>8.2f}  (completed only)         │")
    print(f"    │  Avg steps:    {s['avg_steps']:>5.2f}                               │")
    print(f"    └─────────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description="Compare base LLM vs reward-guided LLM")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=55555)
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the base OpenAI policy (only run reward-guided)")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    all_results = {}

    for task in tasks:
        print(f"\n{'='*74}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*74}")

        # ── Base OpenAI Policy ──
        if not args.skip_base:
            print(f"\n  ── Base LLM (zero-shot) ──")
            base_policy = OpenAIPolicy()
            base_res = run_policy(task, base_policy, "Base LLM", args.episodes, args.seed)
            print_summary(base_res)
        else:
            base_res = None

        # ── Reward-Guided LLM ──
        exp_path = os.path.join(data_dir, f"experience_{task}.json")
        if os.path.exists(exp_path):
            print(f"\n  ── Reward-Guided LLM (with {exp_path}) ──")
        else:
            print(f"\n  ── Reward-Guided LLM (no experience file — run collect_experience.py first) ──")
            exp_path = None

        guided_policy = RewardGuidedLLMPolicy(experience_path=exp_path, n_candidates=1)
        guided_res = run_policy(task, guided_policy, "Reward-Guided LLM",
                                args.episodes, args.seed)
        print_summary(guided_res)

        all_results[task] = {
            "base_llm": base_res,
            "reward_guided_llm": guided_res,
        }

    # ── Comparison table ──
    print(f"\n{'='*74}")
    print(f"  COMPARISON: Base LLM vs Reward-Guided LLM")
    print(f"{'='*74}")
    print(f"  {'Policy':<24s} {'Task':<8s} {'Compl%':>7s} {'Cancel%':>8s} "
          f"{'Timeout%':>9s} {'AvgRwd':>9s} {'AvgProfit':>10s}")
    print(f"  {'─'*75}")
    for task in tasks:
        for key, label in [("base_llm", "Base LLM"), ("reward_guided_llm", "Reward-Guided LLM")]:
            r = all_results[task].get(key)
            if r is None:
                continue
            s = r["summary"]
            print(f"  {label:<24s} {task:<8s} "
                  f"{s['completion_rate']:>6.1%} "
                  f"{s['cancellation_rate']:>7.1%} "
                  f"{s['timeout_rate']:>8.1%} "
                  f"{s['avg_reward']:>+9.2f} "
                  f"${s['avg_profit']:>8.2f}")
        print(f"  {'─'*75}")

    # ── Save results JSON ──
    out_path = os.path.join(data_dir, "results_llm_comparison.json")
    save_data = {}
    for task in tasks:
        save_data[task] = {}
        for key in ["base_llm", "reward_guided_llm"]:
            r = all_results[task].get(key)
            if r:
                save_data[task][key] = r
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
