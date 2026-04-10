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

# Load credentials from .env before anything else — must happen before inference imports
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
# Re-use all logic directly from inference.py
from inference import run_inference, log_start, log_step, log_end, _sigmoid_score, BENCHMARK


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

    num_episodes = int(os.getenv("NUM_EPISODES", "5"))  # default lower for local testing
    tasks = ["easy", "medium", "hard"]

    print(f"[CONFIG] api_base_url={os.environ['API_BASE_URL']} "
          f"model={os.environ['MODEL_NAME']} "
          f"episodes_per_task={num_episodes}", flush=True)

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
