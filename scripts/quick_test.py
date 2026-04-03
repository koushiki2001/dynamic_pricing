"""Quick test of all policies on the new harder config."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from baselines.midpoint_policy import midpoint_policy
from baselines.adaptive_policy import AdaptivePolicy

def run_policy(task, policy_fn, reset_fn=None, n_eps=30, seed_base=99999):
    completed, cancelled, timed_out = 0, 0, 0
    total_steps, total_reward = 0, 0.0
    for ep in range(1, n_eps + 1):
        env = DynamicPricingEnv(task_name=task, seed=seed_base + ep)
        obs = env.reset()
        if reset_fn:
            reset_fn()
        done = False
        ep_reward = 0.0
        iters = 0
        while not done and iters < 50:
            iters += 1
            od = obs.model_dump()
            action = policy_fn(od)
            result = env.step(action)
            # Handle duplicate price guard
            if result.info.get("duplicate_price"):
                p = action["payload"]["price"]
                action["payload"]["price"] = round(p + 0.5, 2)
                result = env.step(action)
                if result.info.get("duplicate_price"):
                    action["payload"]["price"] = round(p - 0.5, 2)
                    result = env.step(action)
            ep_reward += result.reward
            obs = result.observation
            done = result.done
        oc = result.info["outcome"]
        if oc["ride_completed"]:
            completed += 1
        elif oc["timed_out"]:
            timed_out += 1
        else:
            cancelled += 1
        total_steps += oc["steps_taken"]
        total_reward += ep_reward
    return {
        "completed": completed,
        "cancelled": cancelled,
        "timed_out": timed_out,
        "avg_steps": round(total_steps / n_eps, 2),
        "avg_reward": round(total_reward / n_eps, 2),
        "n": n_eps,
    }

def main():
    tasks = ["easy", "medium", "hard"]
    n_eps = 50

    print("=" * 80)
    print("  NEW HARDER CONFIG — POLICY EVALUATION")
    print("=" * 80)

    # Header
    print(f"\n  {'Policy':<14s} {'Task':<8s} {'Compl%':>7s} {'Cancel%':>8s} "
          f"{'Timeout%':>9s} {'AvgRwd':>9s} {'AvgSteps':>9s}")
    print(f"  {'─' * 68}")

    for task in tasks:
        # Midpoint
        r = run_policy(task, midpoint_policy, n_eps=n_eps)
        print(f"  {'Midpoint':<14s} {task:<8s} "
              f"{r['completed']/r['n']:>6.1%} "
              f"{r['cancelled']/r['n']:>7.1%} "
              f"{r['timed_out']/r['n']:>8.1%} "
              f"{r['avg_reward']:>+9.2f} "
              f"{r['avg_steps']:>9.2f}")

        # Adaptive
        adaptive = AdaptivePolicy()
        r = run_policy(task, lambda od: adaptive(od), reset_fn=adaptive.reset, n_eps=n_eps)
        print(f"  {'Adaptive':<14s} {task:<8s} "
              f"{r['completed']/r['n']:>6.1%} "
              f"{r['cancelled']/r['n']:>7.1%} "
              f"{r['timed_out']/r['n']:>8.1%} "
              f"{r['avg_reward']:>+9.2f} "
              f"{r['avg_steps']:>9.2f}")

        if task != "hard":
            print(f"  {'─' * 68}")

    print(f"\n{'=' * 80}")

if __name__ == "__main__":
    main()
