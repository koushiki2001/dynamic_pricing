"""Comprehensive test runner: evaluate ALL policies on BOTH configs.

Runs midpoint, adaptive, and Q-learning policies across easy/medium/hard
for the original config (max $500) and the alternate config (max $1000).

For every episode, logs a one-line description.  Dumps all scenarios to
data/results_{config_name}.json.

Usage:
    python3 scripts/run_all_tests.py
    python3 scripts/run_all_tests.py --eval-episodes 100
    python3 scripts/run_all_tests.py --config alt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── label helpers ──────────────────────────────────────────────────────
WEATHER = {0: "clear", 1: "rain", 2: "storm"}
TRAFFIC = {0: "low", 1: "moderate", 2: "heavy"}
DEMAND  = {0: "low", 1: "medium", 2: "high"}
SUPPLY  = {0: "scarce", 1: "balanced", 2: "surplus"}
TIME_OF = {0: "morning", 1: "afternoon", 2: "evening", 3: "night"}
DAY_OF  = {0: "weekday", 1: "weekend", 2: "holiday"}


# ── config loader ──────────────────────────────────────────────────────
def load_config(config_name: str):
    """Monkey-patches ride_hailing_env.config IN-PLACE.

    Modules that did ``from .config import TASK_CONFIG`` hold a reference
    to the *dict object*.  We must mutate it in-place (.clear() + .update())
    rather than rebinding the name, so every module sees the change.
    """
    import ride_hailing_env.config as cfg_mod

    if config_name == "original":
        from ride_hailing_env import config as src
    elif config_name == "alt":
        from ride_hailing_env import config_alt as src
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Scalar patches (rebind is fine — environment reads them via the module)
    cfg_mod.MIN_OFFER_PRICE = src.MIN_OFFER_PRICE
    cfg_mod.MAX_OFFER_PRICE = src.MAX_OFFER_PRICE
    cfg_mod.DEFAULT_COMMISSION_RATE = src.DEFAULT_COMMISSION_RATE
    cfg_mod.DEFAULT_OPERATIONAL_COST = src.DEFAULT_OPERATIONAL_COST
    cfg_mod.MAX_EFFICIENCY_BONUS = src.MAX_EFFICIENCY_BONUS
    cfg_mod.CANCELLATION_PENALTY = src.CANCELLATION_PENALTY
    cfg_mod.TIMEOUT_PENALTY = src.TIMEOUT_PENALTY
    cfg_mod.PARTIAL_ACCEPT_REWARD = src.PARTIAL_ACCEPT_REWARD
    cfg_mod.DOUBLE_REJECT_PENALTY = src.DOUBLE_REJECT_PENALTY

    # CRITICAL: mutate TASK_CONFIG dict in-place so all cached references update
    # Copy first — src and cfg_mod may be the same module (for "original")
    new_tasks = {k: dict(v) for k, v in src.TASK_CONFIG.items()}
    cfg_mod.TASK_CONFIG.clear()
    cfg_mod.TASK_CONFIG.update(new_tasks)

    # Also patch scalar imports bound locally in sub-modules
    import ride_hailing_env.environment as env_mod
    env_mod.MIN_OFFER_PRICE = src.MIN_OFFER_PRICE
    env_mod.MAX_OFFER_PRICE = src.MAX_OFFER_PRICE

    import ride_hailing_env.scenario_generator as sg_mod
    sg_mod.DEFAULT_COMMISSION_RATE = src.DEFAULT_COMMISSION_RATE
    sg_mod.DEFAULT_OPERATIONAL_COST = src.DEFAULT_OPERATIONAL_COST

    import ride_hailing_env.reward as rw_mod
    rw_mod.CANCELLATION_PENALTY = src.CANCELLATION_PENALTY
    rw_mod.TIMEOUT_PENALTY = src.TIMEOUT_PENALTY
    rw_mod.PARTIAL_ACCEPT_REWARD = src.PARTIAL_ACCEPT_REWARD
    rw_mod.DOUBLE_REJECT_PENALTY = src.DOUBLE_REJECT_PENALTY
    rw_mod.MAX_EFFICIENCY_BONUS = src.MAX_EFFICIENCY_BONUS

    return src


# ── policies ───────────────────────────────────────────────────────────
from baselines.midpoint_policy import midpoint_policy
from baselines.adaptive_policy import AdaptivePolicy


# ── Q-learning (inline, self-contained) ────────────────────────────────
def discretize_obs(obs: Dict[str, Any]) -> Tuple:
    gap = obs["price_gap"]
    step = obs["step_number"]
    rp = int(obs["rider_patience"] * 3)
    dp = int(obs["driver_patience"] * 3)
    gap_bin = 0 if gap < 6 else (1 if gap < 12 else (2 if gap < 20 else 3))
    ctx = obs["weather_condition"] + obs["traffic_level"]
    ctx_bin = 0 if ctx <= 1 else (1 if ctx <= 3 else 2)
    imb = obs["demand_level"] - obs["supply_level"]
    imb_bin = 0 if imb <= 0 else (1 if imb == 1 else 2)
    lr = obs.get("last_rider_response") or "none"
    ld = obs.get("last_driver_response") or "none"
    if lr == "accepted" and ld == "accepted":
        rb = 0
    elif lr == "accepted" or ld == "accepted":
        rb = 1
    elif lr == "none":
        rb = 2
    else:
        rb = 3
    return (gap_bin, min(step, 4), rp, dp, rb, ctx_bin, imb_bin)


def get_price_actions(obs: Dict[str, Any], n: int = 11) -> List[float]:
    lo = obs["rider_quoted_price"] - 2.0
    hi = obs["driver_quoted_price"] + 2.0
    return [round(p, 2) for p in np.linspace(lo, hi, n).tolist()]


class QLearner:
    def __init__(self, n_act=11, lr=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9995):
        self.n_act = n_act
        self.lr, self.gamma = lr, gamma
        self.eps = eps_start
        self.eps_end, self.eps_decay = eps_end, eps_decay
        self.q: Dict[Tuple, np.ndarray] = defaultdict(lambda: np.zeros(self.n_act))

    def act(self, state, rng):
        if rng.random() < self.eps:
            return int(rng.integers(0, self.n_act))
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s2, done):
        t = r if done else r + self.gamma * np.max(self.q[s2])
        self.q[s][a] += self.lr * (t - self.q[s][a])

    def decay(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_qlearner(task_name: str, episodes: int, seed: int, n_act: int = 11,
                   print_every: int = 500) -> QLearner:
    from ride_hailing_env.environment import DynamicPricingEnv

    agent = QLearner(n_act=n_act)
    rng = np.random.default_rng(seed)
    rewards, comps = [], []
    w = min(100, episodes)

    print(f"      Training Q-learner: {episodes} episodes ...", flush=True)
    t0 = time.time()

    for ep in range(1, episodes + 1):
        env = DynamicPricingEnv(task_name=task_name, seed=seed + ep)
        obs = env.reset()
        od = obs.model_dump()
        st = discretize_obs(od)
        pa = get_price_actions(od, n_act)
        done, tr = False, 0.0
        while not done:
            ai = agent.act(st, rng)
            res = env.step({"type": "propose_price", "payload": {"price": pa[min(ai, len(pa)-1)]}})
            ns = discretize_obs(res.observation.model_dump())
            agent.update(st, ai, res.reward, ns, res.done)
            st = ns
            tr += res.reward
            done = res.done
        agent.decay()
        rewards.append(tr)
        comps.append(1 if res.info["outcome"]["ride_completed"] else 0)
        if ep % print_every == 0:
            r = rewards[-w:]
            c = comps[-w:]
            print(f"        ep {ep:>5d}/{episodes}  "
                  f"avg_rwd={sum(r)/len(r):+7.2f}  "
                  f"compl={sum(c)/len(c):5.1%}  "
                  f"ε={agent.eps:.3f}  "
                  f"Q={len(agent.q)} states", flush=True)

    agent.eps = 0.0  # greedy for eval
    print(f"      Training done in {time.time()-t0:.1f}s  "
          f"({len(agent.q)} Q-states)", flush=True)
    return agent


# ── episode runner ─────────────────────────────────────────────────────
def run_episode(env, policy_fn, obs) -> Tuple[Dict, List[Dict]]:
    """Run a single episode, returning (outcome_dict, steps_log)."""
    steps = []
    done = False
    max_iter = 50  # safety: prevent infinite loops from duplicate-price guard
    iters = 0
    while not done and iters < max_iter:
        iters += 1
        od = obs.model_dump()
        action = policy_fn(od)
        result = env.step(action)
        # Skip duplicate-price rejections (no step consumed)
        if result.info.get("duplicate_price"):
            # Nudge price slightly to break the tie
            p = action["payload"]["price"]
            action["payload"]["price"] = round(p + 0.5, 2)
            result = env.step(action)
            if result.info.get("duplicate_price"):
                action["payload"]["price"] = round(p - 0.5, 2)
                result = env.step(action)
        steps.append({
            "step": result.observation.step_number,
            "proposed_price": round(action["payload"]["price"], 2),
            "rider_response": result.observation.last_rider_response,
            "driver_response": result.observation.last_driver_response,
            "rider_patience": result.observation.rider_patience,
            "driver_patience": result.observation.driver_patience,
            "rider_mood": result.observation.rider_mood,
            "driver_mood": result.observation.driver_mood,
            "reward": round(result.reward, 4),
        })
        obs = result.observation
        done = result.done
    outcome = result.info["outcome"]
    return outcome, steps


def episode_description(ep_num: int, obs_dict: dict, outcome: dict) -> str:
    """One-line human-readable episode description."""
    dist = obs_dict["distance_km"]
    gap = obs_dict["price_gap"]
    w = WEATHER.get(obs_dict["weather_condition"], "?")
    t = TRAFFIC.get(obs_dict["traffic_level"], "?")
    d = DEMAND.get(obs_dict["demand_level"], "?")
    s = SUPPLY.get(obs_dict["supply_level"], "?")
    tm = TIME_OF.get(obs_dict["time_of_day"], "?")
    dy = DAY_OF.get(obs_dict["day_type"], "?")
    reason = outcome["termination_reason"]
    steps = outcome["steps_taken"]
    fp = outcome.get("final_price")
    price_str = f"${fp:.2f}" if fp else "N/A"

    rc = outcome.get("rider_cancelled", False)
    dc = outcome.get("driver_cancelled", False)
    cancel_who = ""
    if reason == "cancelled":
        cancel_who = " (rider)" if rc else " (driver)" if dc else ""

    return (f"  ep {ep_num:>4d} │ {dist:5.1f}km gap=${gap:6.2f} │ "
            f"{w:<5s} {t:<8s} {d:<6s}/{s:<8s} {tm:<9s} {dy:<7s} │ "
            f"{reason}{cancel_who:<12s} in {steps} steps │ price={price_str}")


# ── evaluate one policy on one task ────────────────────────────────────
def evaluate_policy(
    task_name: str,
    policy_name: str,
    policy_fn: Callable,
    num_episodes: int,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    from ride_hailing_env.environment import DynamicPricingEnv
    from ride_hailing_env.config import TASK_CONFIG

    cfg = TASK_CONFIG[task_name]
    results = {
        "policy": policy_name,
        "task": task_name,
        "num_episodes": num_episodes,
        "max_steps": cfg["max_steps"],
        "episodes": [],
    }
    totals = {"completed": 0, "cancelled": 0, "rider_cancel": 0,
              "driver_cancel": 0, "timeout": 0, "reward": 0.0,
              "profit": 0.0, "steps": 0, "completed_cnt": 0}

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        init_obs = obs.model_dump()

        outcome, steps_log = run_episode(env, policy_fn, obs)

        # accumulate
        if outcome["ride_completed"]:
            totals["completed"] += 1
            totals["completed_cnt"] += 1
            totals["profit"] += outcome.get("platform_profit", 0.0)
        elif outcome["timed_out"]:
            totals["timeout"] += 1
        else:
            totals["cancelled"] += 1
            if outcome.get("rider_cancelled"):
                totals["rider_cancel"] += 1
            if outcome.get("driver_cancelled"):
                totals["driver_cancel"] += 1
        totals["steps"] += outcome["steps_taken"]

        # episode reward = sum of step rewards
        ep_reward = sum(s["reward"] for s in steps_log)
        totals["reward"] += ep_reward

        # store for JSON
        results["episodes"].append({
            "episode": ep,
            "seed": ep_seed,
            "initial_observation": init_obs,
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
            "episode_reward": round(ep_reward, 4),
        })

        if verbose:
            print(episode_description(ep, init_obs, outcome))

    n = num_episodes
    cc = totals["completed_cnt"] or 1
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


# ── pretty summary printer ─────────────────────────────────────────────
def print_summary(results: Dict):
    s = results["summary"]
    n = results["num_episodes"]
    c = int(s["completion_rate"] * n)
    x = int(s["cancellation_rate"] * n)
    t = int(s["timeout_rate"] * n)
    print(f"\n    ┌─────────────────────────────────────────────────────────┐")
    print(f"    │  {results['policy']:^20s}  on  {results['task']:^8s}  ({n} episodes)   │")
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


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run ALL policies on original & alt configs")
    parser.add_argument("--config", default="both", choices=["original", "alt", "both"],
                        help="Which config to test")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Episodes per task per policy")
    parser.add_argument("--train-episodes", type=int, default=5000,
                        help="Q-learning training episodes per task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=99999)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-episode lines (show summaries only)")
    args = parser.parse_args()

    configs = ["original", "alt"] if args.config == "both" else [args.config]
    tasks = ["easy", "medium", "hard"]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    grand_results = {}

    for cfg_name in configs:
        print(f"\n{'#'*74}")
        print(f"#  CONFIG: {cfg_name.upper():>10s}  "
              f"(MAX_OFFER_PRICE = "
              f"{'$500' if cfg_name == 'original' else '$1000'})")
        print(f"{'#'*74}")

        src = load_config(cfg_name)
        cfg_results = {}

        for task in tasks:
            print(f"\n{'='*74}")
            print(f"  TASK: {task.upper()}  (max_steps={src.TASK_CONFIG[task]['max_steps']})")
            print(f"{'='*74}")

            # ── 1. Midpoint policy ────────────────────────
            print(f"\n  ── Midpoint Policy ──")
            mid_res = evaluate_policy(
                task, "Midpoint", midpoint_policy,
                args.eval_episodes, args.eval_seed, verbose=not args.quiet)
            print_summary(mid_res)

            # ── 2. Adaptive policy ────────────────────────
            print(f"\n  ── Adaptive Policy ──")
            adaptive = AdaptivePolicy()

            def adaptive_fn(obs_dict):
                return adaptive(obs_dict)

            def run_adaptive(task_n, num_ep, seed_v):
                from ride_hailing_env.environment import DynamicPricingEnv
                results_a = {
                    "policy": "Adaptive", "task": task_n,
                    "num_episodes": num_ep,
                    "max_steps": src.TASK_CONFIG[task_n]["max_steps"],
                    "episodes": [],
                }
                totals = {"completed": 0, "cancelled": 0, "rider_cancel": 0,
                          "driver_cancel": 0, "timeout": 0, "reward": 0.0,
                          "profit": 0.0, "steps": 0, "completed_cnt": 0}

                for ep in range(1, num_ep + 1):
                    ep_seed = seed_v + ep
                    env = DynamicPricingEnv(task_name=task_n, seed=ep_seed)
                    obs = env.reset()
                    init_obs = obs.model_dump()
                    adaptive.reset()
                    outcome, steps_log = run_episode(env, adaptive_fn, obs)

                    if outcome["ride_completed"]:
                        totals["completed"] += 1
                        totals["completed_cnt"] += 1
                        totals["profit"] += outcome.get("platform_profit", 0.0)
                    elif outcome["timed_out"]:
                        totals["timeout"] += 1
                    else:
                        totals["cancelled"] += 1
                        if outcome.get("rider_cancelled"):
                            totals["rider_cancel"] += 1
                        if outcome.get("driver_cancelled"):
                            totals["driver_cancel"] += 1
                    totals["steps"] += outcome["steps_taken"]
                    ep_reward = sum(s["reward"] for s in steps_log)
                    totals["reward"] += ep_reward

                    results_a["episodes"].append({
                        "episode": ep, "seed": ep_seed,
                        "initial_observation": init_obs,
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
                        "episode_reward": round(ep_reward, 4),
                    })
                    if not args.quiet:
                        print(episode_description(ep, init_obs, outcome))

                n = num_ep
                cc = totals["completed_cnt"] or 1
                results_a["summary"] = {
                    "completion_rate": round(totals["completed"] / n, 4),
                    "cancellation_rate": round(totals["cancelled"] / n, 4),
                    "rider_cancellations": totals["rider_cancel"],
                    "driver_cancellations": totals["driver_cancel"],
                    "timeout_rate": round(totals["timeout"] / n, 4),
                    "avg_reward": round(totals["reward"] / n, 4),
                    "avg_profit": round(totals["profit"] / cc, 4) if totals["completed"] else 0.0,
                    "avg_steps": round(totals["steps"] / n, 2),
                }
                return results_a

            adapt_res = run_adaptive(task, args.eval_episodes, args.eval_seed)
            print_summary(adapt_res)

            # ── 3. Q-learning policy ─────────────────────
            print(f"\n  ── Q-Learning Policy ──")
            q_agent = train_qlearner(task, args.train_episodes, args.seed,
                                     print_every=max(args.train_episodes // 10, 100))

            def q_policy(obs_dict, _agent=q_agent):
                st = discretize_obs(obs_dict)
                pa = get_price_actions(obs_dict, _agent.n_act)
                ai = int(np.argmax(_agent.q[st]))
                return {"type": "propose_price",
                        "payload": {"price": pa[min(ai, len(pa)-1)]}}

            q_res = evaluate_policy(
                task, "Q-Learning", q_policy,
                args.eval_episodes, args.eval_seed, verbose=not args.quiet)
            print_summary(q_res)

            cfg_results[task] = {
                "Midpoint": mid_res,
                "Adaptive": adapt_res,
                "Q-Learning": q_res,
            }

        # ── Cross-task comparison table ──────────────────
        print(f"\n{'='*74}")
        print(f"  COMPARISON TABLE — {cfg_name.upper()} CONFIG")
        print(f"{'='*74}")
        header = (f"  {'Policy':<14s} {'Task':<8s} {'Compl%':>7s} {'Cancel%':>8s} "
                  f"{'Timeout%':>9s} {'AvgRwd':>9s} {'AvgProfit':>10s} {'AvgSteps':>9s}")
        print(header)
        print(f"  {'─'*72}")
        for task in tasks:
            for pname in ["Midpoint", "Adaptive", "Q-Learning"]:
                s = cfg_results[task][pname]["summary"]
                print(f"  {pname:<14s} {task:<8s} "
                      f"{s['completion_rate']:>6.1%} "
                      f"{s['cancellation_rate']:>7.1%} "
                      f"{s['timeout_rate']:>8.1%} "
                      f"{s['avg_reward']:>+9.2f} "
                      f"${s['avg_profit']:>8.2f} "
                      f"{s['avg_steps']:>9.2f}")
            if task != "hard":
                print(f"  {'─'*72}")

        # ── Save JSON ────────────────────────────────────
        json_path = os.path.join(data_dir, f"results_{cfg_name}.json")
        # Strip heavy episode data for the summary file
        summary_data = {
            "config": cfg_name,
            "max_offer_price": src.MAX_OFFER_PRICE,
            "eval_episodes": args.eval_episodes,
            "train_episodes": args.train_episodes,
            "tasks": {},
        }
        for task in tasks:
            summary_data["tasks"][task] = {}
            for pname in ["Midpoint", "Adaptive", "Q-Learning"]:
                summary_data["tasks"][task][pname] = cfg_results[task][pname]["summary"]

        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"\n  Summary saved → {json_path}")

        # ── Full episode details JSON ────────────────────
        full_path = os.path.join(data_dir, f"results_{cfg_name}_full.json")
        full_data = {}
        for task in tasks:
            full_data[task] = {}
            for pname in ["Midpoint", "Adaptive", "Q-Learning"]:
                full_data[task][pname] = cfg_results[task][pname]
        with open(full_path, "w") as f:
            json.dump(full_data, f, indent=2, default=str)
        print(f"  Full episodes saved → {full_path}")

        grand_results[cfg_name] = summary_data

    # ── Grand comparison (if both configs) ────────────────
    if len(configs) == 2:
        print(f"\n\n{'#'*74}")
        print(f"#  GRAND COMPARISON: ORIGINAL ($500) vs ALT ($1000)")
        print(f"{'#'*74}")
        header = (f"  {'Config':<10s} {'Policy':<14s} {'Task':<8s} {'Compl%':>7s} "
                  f"{'Cancel%':>8s} {'Timeout%':>9s} {'AvgRwd':>9s} {'AvgProfit':>10s}")
        print(header)
        print(f"  {'─'*76}")
        for cfg_name in configs:
            for task in tasks:
                for pname in ["Midpoint", "Adaptive", "Q-Learning"]:
                    s = grand_results[cfg_name]["tasks"][task][pname]
                    print(f"  {cfg_name:<10s} {pname:<14s} {task:<8s} "
                          f"{s['completion_rate']:>6.1%} "
                          f"{s['cancellation_rate']:>7.1%} "
                          f"{s['timeout_rate']:>8.1%} "
                          f"{s['avg_reward']:>+9.2f} "
                          f"${s['avg_profit']:>8.2f}")
            print(f"  {'─'*76}")

    print(f"\nAll JSON files in: {os.path.abspath(data_dir)}/")
    print("Done.")


if __name__ == "__main__":
    main()
