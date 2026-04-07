"""LLM baseline inference script (hackathon requirement).

This is the ROOT inference.py required by the HF Space validator.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Usage:
    API_BASE_URL=https://openrouter.ai/api/v1 MODEL_NAME=google/gemini-2.0-flash-001 HF_TOKEN=sk-... python inference.py
"""

from __future__ import annotations

import os
import sys

from ride_hailing_env.config import TASK_CONFIG, MAX_EFFICIENCY_BONUS
from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.tasks.graders import GRADER_WEIGHTS
from ride_hailing_env.utils import normalize_revenue
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
    total_reward = 0.0

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

        total_reward += episode_reward
        outcome = result.info["outcome"]

        # [END] structured log
        profit_val = outcome.get("platform_profit") or 0.0
        print(f"[END] task={task_name} episode={ep+1} "
              f"outcome={outcome['termination_reason']} "
              f"steps={outcome['steps_taken']} "
              f"profit={profit_val:.2f} "
              f"reward={episode_reward:.4f}")

        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

    # Compute grader score
    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
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

    return {
        "task": task_name,
        "score": round(score, 4),
        "completion_rate": round(completion_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_reward": round(total_reward / num_episodes, 4),
        "num_episodes": num_episodes,
    }


def main():
    if not os.getenv("HF_TOKEN"):
        print("ERROR: Set HF_TOKEN environment variable.")
        sys.exit(1)
    if not os.getenv("API_BASE_URL"):
        print("ERROR: Set API_BASE_URL environment variable.")
        sys.exit(1)
    if not os.getenv("MODEL_NAME"):
        print("ERROR: Set MODEL_NAME environment variable.")
        sys.exit(1)

    num_episodes = int(os.getenv("NUM_EPISODES", "20"))
    tasks = ["easy", "medium", "hard"]

    print(f"[CONFIG] api_base_url={os.environ['API_BASE_URL']} "
          f"model={os.environ['MODEL_NAME']} "
          f"episodes_per_task={num_episodes}")

    all_scores = []
    for task in tasks:
        result = run_inference(task, num_episodes=num_episodes)
        all_scores.append(result["score"])
        print(f"[RESULT] task={task} score={result['score']:.4f} "
              f"completion_rate={result['completion_rate']:.4f} "
              f"cancellation_rate={result['cancellation_rate']:.4f} "
              f"avg_profit={result['avg_profit']:.2f} "
              f"avg_reward={result['avg_reward']:.4f}")

    avg_score = sum(all_scores) / len(all_scores)
    print(f"[FINAL] average_score={avg_score:.4f}")


if __name__ == "__main__":
    main()
