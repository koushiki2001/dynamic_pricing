"""Test OpenAI (base LLM) and Reward-Guided LLM policies on all tasks.

Runs a small number of episodes per task (default 5) to keep API costs low.
Compares against midpoint/adaptive baselines (instant, no API).

Usage:
    python3 scripts/test_llm_policies.py
    python3 scripts/test_llm_policies.py --episodes 10
    python3 scripts/test_llm_policies.py --task easy
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from baselines.midpoint_policy import midpoint_policy
from baselines.adaptive_policy import AdaptivePolicy
from baselines.openai_policy import OpenAIPolicy
from baselines.reward_guided_llm_policy import RewardGuidedLLMPolicy

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def run_episodes(task, policy_fn, reset_fn, n_eps, seed_base=99999, label=""):
    """Run n_eps episodes, return summary dict."""
    completed, cancelled, timed_out = 0, 0, 0
    total_steps, total_reward, total_profit = 0, 0.0, 0.0
    step_prices = []  # collect (ep, step, price) for negotiation trace

    for ep in range(1, n_eps + 1):
        env = DynamicPricingEnv(task_name=task, seed=seed_base + ep)
        obs = env.reset()
        if reset_fn:
            reset_fn()
        done = False
        ep_reward = 0.0
        iters = 0
        ep_prices = []
        while not done and iters < 50:
            iters += 1
            od = obs.model_dump()
            action = policy_fn(od)
            price = action["payload"]["price"]
            result = env.step(action)
            # Handle duplicate price guard
            if result.info.get("duplicate_price"):
                action["payload"]["price"] = round(price + 0.5, 2)
                result = env.step(action)
                if result.info.get("duplicate_price"):
                    action["payload"]["price"] = round(price - 0.5, 2)
                    result = env.step(action)
            ep_prices.append(round(action["payload"]["price"], 2))
            ep_reward += result.reward
            obs = result.observation
            done = result.done

        oc = result.info["outcome"]
        reason = oc["termination_reason"]
        steps = oc["steps_taken"]
        fp = oc.get("final_price")

        if oc["ride_completed"]:
            completed += 1
            total_profit += oc.get("platform_profit", 0.0)
        elif oc["timed_out"]:
            timed_out += 1
        else:
            cancelled += 1

        total_steps += steps
        total_reward += ep_reward

        # Print per-episode trace
        price_trace = " → ".join(f"${p}" for p in ep_prices)
        fp_str = f"${fp:.2f}" if fp else "N/A"
        print(f"    ep {ep:>2d} │ {reason:<10s} in {steps} steps │ "
              f"prices: {price_trace} │ final={fp_str} │ rwd={ep_reward:+.2f}")

    n = n_eps
    cc = completed or 1
    return {
        "completed": completed, "cancelled": cancelled, "timed_out": timed_out,
        "completion_rate": completed / n,
        "cancellation_rate": cancelled / n,
        "timeout_rate": timed_out / n,
        "avg_steps": round(total_steps / n, 2),
        "avg_reward": round(total_reward / n, 2),
        "avg_profit": round(total_profit / cc, 2) if completed else 0.0,
        "n": n,
    }


def print_row(name, task, r):
    print(f"  {name:<20s} {task:<8s} "
          f"{r['completion_rate']:>6.0%} "
          f"{r['cancellation_rate']:>7.0%} "
          f"{r['timeout_rate']:>8.0%} "
          f"{r['avg_reward']:>+9.2f} "
          f"${r['avg_profit']:>8.2f} "
          f"{r['avg_steps']:>9.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--n-candidates", type=int, default=1,
                        help="LLM candidates for reward-guided (1=fast, 3=better)")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    n_eps = args.episodes

    all_results = {}

    for task in tasks:
        print(f"\n{'=' * 80}")
        print(f"  TASK: {task.upper()} — {n_eps} episodes")
        print(f"{'=' * 80}")

        # 1. Midpoint (instant baseline)
        print(f"\n  ── Midpoint (baseline) ──")
        mid = run_episodes(task, midpoint_policy, None, n_eps)

        # 2. Adaptive (instant baseline)
        print(f"\n  ── Adaptive (baseline) ──")
        adaptive = AdaptivePolicy()
        adp = run_episodes(task, lambda od: adaptive(od), adaptive.reset, n_eps)

        # 3. Base OpenAI LLM
        print(f"\n  ── Base LLM (Gemini Flash) ──")
        oai = OpenAIPolicy()
        t0 = time.time()
        oai_res = run_episodes(task, lambda od: oai(od), oai.reset, n_eps)
        oai_time = time.time() - t0
        print(f"    ⏱ {oai_time:.1f}s total ({oai_time/n_eps:.1f}s/episode)")

        # 4. Reward-Guided LLM
        exp_path = os.path.join(DATA_DIR, f"experience_{task}.json")
        print(f"\n  ── Reward-Guided LLM (Gemini Flash + reward proxy) ──")
        rg = RewardGuidedLLMPolicy(
            experience_path=exp_path,
            n_candidates=args.n_candidates,
        )
        t0 = time.time()
        rg_res = run_episodes(task, lambda od: rg(od), rg.reset, n_eps)
        rg_time = time.time() - t0
        print(f"    ⏱ {rg_time:.1f}s total ({rg_time/n_eps:.1f}s/episode)")

        all_results[task] = {
            "Midpoint": mid,
            "Adaptive": adp,
            "Base LLM": oai_res,
            "Reward-Guided LLM": rg_res,
        }

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 90}")
    header = (f"  {'Policy':<20s} {'Task':<8s} {'Compl%':>7s} {'Cancel%':>8s} "
              f"{'Timeout%':>9s} {'AvgRwd':>9s} {'AvgProfit':>10s} {'AvgSteps':>9s}")
    print(header)
    print(f"  {'─' * 82}")
    for task in tasks:
        for pname in ["Midpoint", "Adaptive", "Base LLM", "Reward-Guided LLM"]:
            print_row(pname, task, all_results[task][pname])
        if task != tasks[-1]:
            print(f"  {'─' * 82}")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    main()
