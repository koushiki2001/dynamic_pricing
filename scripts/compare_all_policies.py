#!/usr/bin/env python3
"""
Consolidated comparison of ALL 4 policies using the official grader scoring.

Policies:
  1. Midpoint       — stateless, proposes (rider + driver) / 2
  2. Adaptive       — stateful, binary-search using rejection feedback
  3. Base LLM       — Gemini Flash zero-shot via OpenRouter
  4. Reward-Guided  — Gemini Flash + adaptive bounds + reward proxy

With optional session context to let LLM policies learn across episodes.

Usage:
    python3 scripts/compare_all_policies.py                  # 10 eps/task
    python3 scripts/compare_all_policies.py --episodes 20    # 20 eps/task
    python3 scripts/compare_all_policies.py --task hard      # hard only
    python3 scripts/compare_all_policies.py --no-llm         # skip LLM policies
    python3 scripts/compare_all_policies.py --with-session   # enable session context
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.config import TASK_CONFIG, MAX_EFFICIENCY_BONUS
from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.utils import normalize_revenue
from ride_hailing_env.tasks.graders import GRADER_WEIGHTS

from baselines.midpoint_policy import midpoint_policy
from baselines.adaptive_policy import AdaptivePolicy
from baselines.session_manager import SessionManager


# ── episode runner (handles duplicate-price guard + stateful reset) ───
def run_episode(env, policy_fn, reset_fn=None, session_manager=None):
    """Run one episode, return (outcome_dict, step_count, prices_list, total_reward)."""
    obs = env.reset()
    if reset_fn:
        reset_fn()
    if session_manager:
        session_manager.reset_episode(task_difficulty=env.task_name)
    
    done = False
    prices = []
    iters = 0
    result = None
    total_reward = 0.0

    while not done and iters < 60:
        iters += 1
        od = obs.model_dump()
        action = policy_fn(od)
        price = action["payload"]["price"]
        result = env.step(action)

        # Handle duplicate-price guard
        if result.info.get("duplicate_price"):
            # Nudge by $1 in direction away from last
            action["payload"]["price"] = round(price + 1.0, 2)
            result = env.step(action)
            if result.info.get("duplicate_price"):
                action["payload"]["price"] = round(price - 1.0, 2)
                result = env.step(action)

        prices.append(round(action["payload"]["price"], 2))
        total_reward += result.reward
        
        # Log to session if available
        if session_manager:
            obs_dict = result.observation.model_dump()
            session_manager.log_step(
                step_number=obs_dict["step_number"],
                proposed_price=action["payload"]["price"],
                rider_response=obs_dict.get("last_rider_response"),
                driver_response=obs_dict.get("last_driver_response"),
                step_reward=result.reward,
                observation=obs_dict,
            )
        
        obs = result.observation
        done = result.done

    # Log episode completion
    if session_manager:
        outcome = result.info.get("outcome", {})
        session_manager.log_episode(
            episode_reward=total_reward,
            completed=outcome.get("ride_completed", False),
            termination_reason=outcome.get("termination_reason", "unknown"),
            initial_price_gap=float(outcome.get("initial_price_gap", 0)),
            final_proposed_price=prices[-1] if prices else None,
        )

    return result, prices, total_reward


def grade_policy(task_name, policy_fn, reset_fn, num_episodes, seed, session_manager=None):
    """Run num_episodes and compute the official grader score."""
    cfg = TASK_CONFIG[task_name]

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    total_efficiency = 0.0
    total_reward = 0.0
    completed_count = 0
    all_prices = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        result, prices, ep_reward = run_episode(env, policy_fn, reset_fn, session_manager)
        all_prices.append(prices)
        total_reward += ep_reward

        outcome = result.info["outcome"]
        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

    n = num_episodes
    completion_rate = total_completed / n
    cancellation_rate = total_cancelled / n
    timeout_rate = total_timed_out / n
    avg_efficiency = total_efficiency / completed_count if completed_count > 0 else 0.0
    avg_efficiency_norm = min(avg_efficiency / MAX_EFFICIENCY_BONUS, 1.0)
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0
    profit_norm = normalize_revenue(avg_profit)

    w = GRADER_WEIGHTS[task_name]
    score = (
        w["completion"] * completion_rate
        + w["efficiency"] * avg_efficiency_norm
        + w["profit"] * profit_norm
        + w["no_cancel"] * (1.0 - cancellation_rate)
    )
    score = max(0.0, min(1.0, score))

    avg_steps = sum(len(p) for p in all_prices) / n

    return {
        "score": round(score, 4),
        "completion": round(completion_rate, 4),
        "cancel": round(cancellation_rate, 4),
        "timeout": round(timeout_rate, 4),
        "profit": round(avg_profit, 2),
        "efficiency": round(avg_efficiency, 2),
        "avg_steps": round(avg_steps, 1),
    }


# ── policy constructors ──────────────────────────────────────────────
def make_midpoint():
    return midpoint_policy, None, "Midpoint"

def make_adaptive():
    pol = AdaptivePolicy()
    return lambda od: pol(od), pol.reset, "Adaptive"

def make_base_llm(session_manager=None):
    from baselines.openai_policy import OpenAIPolicy
    pol = OpenAIPolicy(session_manager=session_manager)
    return lambda od: pol(od), pol.reset, "Base LLM"

def make_reward_guided(session_manager=None):
    from baselines.reward_guided_llm_policy import RewardGuidedLLMPolicy
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    # experience_path is per-task, set in the loop
    pol = RewardGuidedLLMPolicy(session_manager=session_manager)
    return lambda od: pol(od), pol.reset, "Reward-Guided LLM"


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare all 4 policies")
    parser.add_argument("--episodes", "-n", type=int, default=10,
                        help="Episodes per task (default: 10)")
    parser.add_argument("--task", "-t", default="all",
                        choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM policies (midpoint + adaptive only)")
    parser.add_argument("--with-session", action="store_true",
                        help="Enable cross-episode session context for LLM policies")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed (default: per-task config seed)")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    n_eps = args.episodes

    # Build policy list
    # We'll instantiate policies per task to allow session managers
    
    # ── Run all evaluations ──
    # results[task][policy_name] = {score, completion, ...}
    results = {}
    timings = {}

    for task in tasks:
        results[task] = {}
        timings[task] = {}
        seed = args.seed or TASK_CONFIG[task]["seed"]

        print(f"\n{'='*80}")
        print(f"  TASK: {task.upper()}  ({n_eps} episodes, seed={seed})")
        if args.with_session:
            print(f"  [Session context ENABLED for LLM policies]")
        print(f"{'='*80}")

        # Create fresh session manager per task if enabled
        session_manager = SessionManager(max_episodes=20) if args.with_session else None

        # Non-LLM policies (no session needed)
        for name, maker in [("Midpoint", make_midpoint), ("Adaptive", make_adaptive)]:
            policy_fn, reset_fn, _ = maker()

            print(f"\n  Running {name}...", end=" ", flush=True)
            t0 = time.time()
            r = grade_policy(task, policy_fn, reset_fn, n_eps, seed, session_manager=None)
            elapsed = time.time() - t0

            results[task][name] = r
            timings[task][name] = elapsed

            print(
                f"score={r['score']:.4f}  "
                f"compl={r['completion']:.0%}  "
                f"cancel={r['cancel']:.0%}  "
                f"timeout={r['timeout']:.0%}  "
                f"profit=${r['profit']:.2f}  "
                f"steps={r['avg_steps']:.1f}"
            )

        # LLM policies (with optional session)
        if not args.no_llm:
            for name, maker in [("Base LLM", make_base_llm), ("Reward-Guided LLM", make_reward_guided)]:
                policy_fn, reset_fn, _ = maker(session_manager=session_manager)

                print(f"\n  Running {name}...", end=" ", flush=True)
                t0 = time.time()
                r = grade_policy(task, policy_fn, reset_fn, n_eps, seed, session_manager=session_manager)
                elapsed = time.time() - t0

                results[task][name] = r
                timings[task][name] = elapsed

                session_info = " (with session)" if args.with_session else ""
                print(
                    f"score={r['score']:.4f}  "
                    f"compl={r['completion']:.0%}  "
                    f"cancel={r['cancel']:.0%}  "
                    f"timeout={r['timeout']:.0%}  "
                    f"profit=${r['profit']:.2f}  "
                    f"steps={r['avg_steps']:.1f}  ({elapsed:.1f}s){session_info}"
                )
        
        # Print session summary if enabled
        if args.with_session and session_manager:
            summary = session_manager.get_session_summary()
            print(f"\n  SESSION SUMMARY for {task}:")
            print(f"    Episodes: {summary.get('episode_count', 0)}")
            print(f"    Success rate: {summary.get('success_rate', 'N/A')}")
            print(f"    Avg reward: {summary.get('average_reward', 0):.3f}")

    # ── Print comparison table ──
    policy_names = ["Midpoint", "Adaptive"]
    if not args.no_llm:
        policy_names.extend(["Base LLM", "Reward-Guided LLM"])

    print(f"\n\n{'='*100}")
    print(f"  COMPARISON TABLE — Official Grader Scores")
    if args.with_session:
        print(f"  [LLM policies with CROSS-EPISODE SESSION CONTEXT]")
    print(f"  Weights: Easy(compl=0.40 eff=0.30 profit=0.20 nocancel=0.10)")
    print(f"           Med (compl=0.30 eff=0.25 profit=0.30 nocancel=0.15)")
    print(f"           Hard(compl=0.25 eff=0.20 profit=0.35 nocancel=0.20)")
    print(f"{'='*100}")

    header = (
        f"  {'Policy':<22s} {'Task':<7s} "
        f"{'Score':>7s} {'Compl%':>7s} {'Cancel%':>8s} {'Timeout%':>9s} "
        f"{'Profit':>8s} {'Effic':>7s} {'Steps':>6s}"
    )
    print(header)
    print(f"  {'─'*88}")

    for task in tasks:
        for name in policy_names:
            r = results[task][name]
            best_score = max(results[task][n]["score"] for n in policy_names)
            marker = " *" if r["score"] == best_score else "  "
            print(
                f"  {name:<22s} {task:<7s} "
                f"{r['score']:>6.4f}{marker}"
                f"{r['completion']:>6.0%} "
                f"{r['cancel']:>7.0%}  "
                f"{r['timeout']:>7.0%}  "
                f"${r['profit']:>7.2f} "
                f"{r['efficiency']:>6.2f} "
                f"{r['avg_steps']:>5.1f}"
            )
        if task != tasks[-1]:
            print(f"  {'─'*88}")

    # ── Average scores ──
    print(f"\n  {'─'*88}")
    print(f"  {'AVERAGE SCORES':^88s}")
    print(f"  {'─'*88}")

    avg_scores = {}
    for name in policy_names:
        scores = [results[t][name]["score"] for t in tasks]
        avg = sum(scores) / len(scores)
        avg_scores[name] = avg

    best_avg = max(avg_scores.values())
    for name in policy_names:
        avg = avg_scores[name]
        marker = " <-- BEST" if avg == best_avg else ""
        bar = "█" * int(avg * 40)
        print(f"  {name:<22s}  {avg:.4f}  {bar}{marker}")

    # ── Per-task winners ──
    print(f"\n  {'─'*88}")
    print(f"  PER-TASK WINNERS:")
    for task in tasks:
        best_name = max(policy_names, key=lambda n: results[task][n]["score"])
        best_s = results[task][best_name]["score"]
        print(f"    {task:>8s}: {best_name} ({best_s:.4f})")

    # ── Timing summary ──
    if not args.no_llm:
        print(f"\n  TIMING (LLM policies):")
        for task in tasks:
            for name in policy_names:
                if "LLM" in name:
                    t = timings[task][name]
                    per_ep = t / n_eps
                    print(f"    {name:<22s} {task:<7s} {t:>6.1f}s total  ({per_ep:.1f}s/ep)")

    print(f"\n{'='*100}")
    print(f"  Episodes per task: {n_eps}")
    print(f"  Tasks evaluated: {', '.join(tasks)}")
    print(f"  Policies: {', '.join(policy_names)}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
