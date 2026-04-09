"""Local test inference script — same logic as inference.py, loads credentials from .env.

Use this for local testing. Do NOT submit this file — inference.py is the submission entry point.

Usage:
    python test_inference.py
    NUM_EPISODES=5 python test_inference.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Load credentials from .env before anything else
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

from ride_hailing_env.config import TASK_CONFIG
from ride_hailing_env.environment import DynamicPricingEnv
from baselines.openai_policy import OpenAIPolicy


def run_inference(task_name: str, num_episodes: int = 20, seed: int | None = None) -> dict:
    """Run OpenAI agent on a task, return grading results."""
    cfg = TASK_CONFIG[task_name]
    seed = seed or cfg["seed"]

    policy = OpenAIPolicy()

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    total_efficiency = 0.0
    completed_count = 0
    total_passed = 0
    total_penalty = 0.0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step_num = 0

        policy.reset()

        # [START] structured log
        print(f"[START] task={task_name} episode={ep+1} seed={ep_seed}")

        while not done:
            obs_dict = obs.model_dump()
            action = policy(obs_dict)
            result = env.step(action)

            # Handle duplicate-price rejection: nudge price to break the loop
            if result.info.get("duplicate_price"):
                p = action["payload"]["price"]
                action["payload"]["price"] = round(p + 0.5, 2)
                result = env.step(action)
                if result.info.get("duplicate_price"):
                    action["payload"]["price"] = round(p - 0.5, 2)
                    result = env.step(action)
                    if result.info.get("duplicate_price"):
                        action["payload"]["price"] = round(p + 1.0, 2)
                        result = env.step(action)

            step_num += 1
            price = action["payload"]["price"]
            rider_resp = result.observation.last_rider_response or "none"
            driver_resp = result.observation.last_driver_response or "none"

            # [STEP] structured log
            print(f"[STEP] task={task_name} episode={ep+1} step={step_num} "
                  f"price={price:.2f} rider_response={rider_resp} "
                  f"driver_response={driver_resp} reward={result.reward:.4f}")

            obs = result.observation
            done = result.done
            episode_reward += result.reward

        outcome = result.info["outcome"]
        missed_revenue_penalty = result.info["missed_revenue_penalty"]
        reward_threshold = result.info["reward_threshold"]
        penalty_threshold = result.info["penalty_threshold"]

        profit_val = outcome.get("platform_profit") or 0.0
        episode_passed = (
            outcome["ride_completed"]
            and episode_reward > reward_threshold
            and missed_revenue_penalty < penalty_threshold
        )

        # [END] structured log
        print(f"[END] task={task_name} episode={ep+1} "
              f"outcome={outcome['termination_reason']} "
              f"steps={outcome['steps_taken']} "
              f"profit={profit_val:.2f} "
              f"reward={episode_reward:.4f} "
              f"penalty={missed_revenue_penalty:.4f} "
              f"passed={episode_passed}")

        total_penalty += missed_revenue_penalty

        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]
            if episode_passed:
                total_passed += 1
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

    score = max(0.0, min(1.0, total_passed / num_episodes))
    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0

    return {
        "task": task_name,
        "score": round(score, 4),
        "pass_rate": round(total_passed / num_episodes, 4),
        "completion_rate": round(completion_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "timeout_rate": round(total_timed_out / num_episodes, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_penalty": round(total_penalty / num_episodes, 4),
        "num_episodes": num_episodes,
    }


def main():
    if not os.getenv("HF_TOKEN"):
        print("ERROR: HF_TOKEN not found. Check your .env file.")
        sys.exit(1)
    if not os.getenv("API_BASE_URL"):
        print("ERROR: API_BASE_URL not found. Check your .env file.")
        sys.exit(1)
    if not os.getenv("MODEL_NAME"):
        print("ERROR: MODEL_NAME not found. Check your .env file.")
        sys.exit(1)

    num_episodes = int(os.getenv("NUM_EPISODES", "20"))
    tasks = ["easy", "medium", "hard"]

    print(f"[CONFIG] api_base_url={os.environ['API_BASE_URL']} "
          f"model={os.environ['MODEL_NAME']} "
          f"episodes_per_task={num_episodes}")

    all_results = []
    for task in tasks:
        result = run_inference(task, num_episodes=num_episodes)
        all_results.append(result)

    avg_score = sum(r["score"] for r in all_results) / len(all_results)

    print("\n" + "=" * 88)
    print(f"{'RESULTS SUMMARY':^88}")
    print("=" * 88)
    print(f"{'Task':<10} {'Score':>7} {'Pass%':>7} {'Complete%':>10} {'Cancel%':>8} {'Timeout%':>9} {'AvgProfit':>10} {'AvgPenalty':>11}")
    print("-" * 88)
    for r in all_results:
        print(f"{r['task']:<10} {r['score']:>7.4f} {r['pass_rate']*100:>6.1f}% "
              f"{r['completion_rate']*100:>9.1f}% {r['cancellation_rate']*100:>7.1f}% "
              f"{r['timeout_rate']*100:>8.1f}% ${r['avg_profit']:>9.2f} ${r['avg_penalty']:>10.4f}")
    print("-" * 88)
    print(f"{'AVERAGE':<10} {avg_score:>7.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
