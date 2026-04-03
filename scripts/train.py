"""Q-learning agent for ride-hailing dynamic pricing.

Discretizes the observation space and learns a price-selection policy
via tabular Q-learning. Outputs live training progress to terminal.

Usage:
    python scripts/train.py
    python scripts/train.py --task medium --episodes 5000
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.config import TASK_CONFIG


# ---------------------------------------------------------------------------
# State / Action discretization
# ---------------------------------------------------------------------------

def discretize_obs(obs: Dict[str, Any]) -> Tuple:
    """Convert continuous observation into a hashable discrete state.

    Coarser bins to keep Q-table manageable with diverse scenarios while
    still capturing the signals that matter for price selection.
    """
    gap = obs["price_gap"]
    step = obs["step_number"]
    rider_patience_bin = int(obs["rider_patience"] * 3)  # 0-3
    driver_patience_bin = int(obs["driver_patience"] * 3)  # 0-3

    # Bin the gap into buckets
    if gap < 6:
        gap_bin = 0
    elif gap < 12:
        gap_bin = 1
    elif gap < 20:
        gap_bin = 2
    else:
        gap_bin = 3

    # Coarse context: combine weather + traffic into a single "difficulty" signal
    context_score = obs["weather_condition"] + obs["traffic_level"]  # 0-4
    if context_score <= 1:
        context_bin = 0  # easy conditions
    elif context_score <= 3:
        context_bin = 1  # moderate
    else:
        context_bin = 2  # harsh

    # Demand-supply imbalance
    imbalance = obs["demand_level"] - obs["supply_level"]  # -2 to +2
    if imbalance <= 0:
        imbalance_bin = 0  # balanced or oversupply
    elif imbalance == 1:
        imbalance_bin = 1  # slight shortage
    else:
        imbalance_bin = 2  # severe shortage

    # Combine last responses into a single signal
    last_rider = obs.get("last_rider_response") or "none"
    last_driver = obs.get("last_driver_response") or "none"
    if last_rider == "accepted" and last_driver == "accepted":
        response_bin = 0
    elif last_rider == "accepted" or last_driver == "accepted":
        response_bin = 1  # partial accept
    elif last_rider == "none":
        response_bin = 2  # first step
    else:
        response_bin = 3  # double reject

    return (
        gap_bin,
        min(step, 4),  # cap step at 4 to share learning across late steps
        rider_patience_bin,
        driver_patience_bin,
        response_bin,
        context_bin,
        imbalance_bin,
    )


def get_price_actions(obs: Dict[str, Any], num_actions: int = 11) -> list[float]:
    """Generate candidate prices between rider and driver quotes.

    Returns evenly spaced prices spanning the full range plus some
    outside-range exploration prices.
    """
    rider_q = obs["rider_quoted_price"]
    driver_q = obs["driver_quoted_price"]
    low = rider_q - 2.0
    high = driver_q + 2.0
    prices = np.linspace(low, high, num_actions).tolist()
    return [round(p, 2) for p in prices]


# ---------------------------------------------------------------------------
# Q-learning trainer
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(
        self,
        num_actions: int = 11,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
    ):
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions)
        )

    def select_action(self, state: Tuple, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: Tuple,
        action_idx: int,
        reward: float,
        next_state: Tuple,
        done: bool,
    ):
        current_q = self.q_table[state][action_idx]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train(
    task_name: str = "easy",
    num_episodes: int = 3000,
    seed: int = 42,
    num_actions: int = 11,
    print_every: int = 100,
) -> QLearningAgent:
    """Train a Q-learning agent and display live progress."""

    agent = QLearningAgent(num_actions=num_actions)
    rng = np.random.default_rng(seed)

    # Tracking
    rewards_history = []
    completions_history = []
    window = min(100, num_episodes)

    cfg = TASK_CONFIG[task_name]
    max_steps = cfg["max_steps"]

    print(f"\n{'='*70}")
    print(f"TRAINING: Q-Learning Agent")
    print(f"Task: {task_name} | Episodes: {num_episodes} | Actions: {num_actions}")
    print(f"Max steps/episode: {max_steps}")
    print(f"{'='*70}\n")

    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        obs_dict = obs.model_dump()

        state = discretize_obs(obs_dict)
        prices = get_price_actions(obs_dict, num_actions)

        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action_idx = agent.select_action(state, rng)
            price = prices[min(action_idx, len(prices) - 1)]
            action = {"type": "propose_price", "payload": {"price": price}}

            result = env.step(action)
            next_obs_dict = result.observation.model_dump()
            next_state = discretize_obs(next_obs_dict)

            agent.update(state, action_idx, result.reward, next_state, result.done)

            state = next_state
            total_reward += result.reward
            done = result.done
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        completed = result.info["outcome"]["ride_completed"]
        completions_history.append(1 if completed else 0)

        # Print progress
        if ep % print_every == 0 or ep == 1:
            recent_rewards = rewards_history[-window:]
            recent_completions = completions_history[-window:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            completion_rate = sum(recent_completions) / len(recent_completions)
            elapsed = time.time() - start_time

            bar_len = 30
            filled = int(bar_len * ep / num_episodes)
            bar = "█" * filled + "░" * (bar_len - filled)

            print(
                f"  [{bar}] {ep:5d}/{num_episodes}  "
                f"avg_reward={avg_reward:+7.2f}  "
                f"completion={completion_rate:5.1%}  "
                f"ε={agent.epsilon:.3f}  "
                f"states={len(agent.q_table):5d}  "
                f"time={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE in {elapsed:.1f}s")
    print(f"Final ε: {agent.epsilon:.4f}")
    print(f"Q-table states: {len(agent.q_table)}")
    final_rewards = rewards_history[-window:]
    final_completions = completions_history[-window:]
    print(f"Last {window} episodes: avg_reward={sum(final_rewards)/len(final_rewards):+.2f}  "
          f"completion={sum(final_completions)/len(final_completions):.1%}")
    print(f"{'='*70}\n")

    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: QLearningAgent,
    task_name: str,
    num_episodes: int = 100,
    seed: int = 99999,
    num_actions: int = 11,
):
    """Evaluate a trained agent (greedy, no exploration)."""
    cfg = TASK_CONFIG[task_name]
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy

    rng = np.random.default_rng(seed)
    total_reward = 0.0
    total_completed = 0
    total_cancelled = 0
    total_rider_cancelled = 0
    total_driver_cancelled = 0
    total_timed_out = 0
    total_steps = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        obs_dict = obs.model_dump()
        state = discretize_obs(obs_dict)
        prices = get_price_actions(obs_dict, num_actions)
        done = False
        ep_reward = 0.0

        while not done:
            action_idx = agent.select_action(state, rng)
            price = prices[min(action_idx, len(prices) - 1)]
            action = {"type": "propose_price", "payload": {"price": price}}
            result = env.step(action)
            next_obs_dict = result.observation.model_dump()
            state = discretize_obs(next_obs_dict)
            ep_reward += result.reward
            done = result.done

        total_reward += ep_reward
        outcome = result.info["outcome"]
        total_steps += outcome["steps_taken"]
        if outcome["ride_completed"]:
            total_completed += 1
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1
            if outcome.get("rider_cancelled"):
                total_rider_cancelled += 1
            if outcome.get("driver_cancelled"):
                total_driver_cancelled += 1

    agent.epsilon = old_epsilon  # Restore

    print(f"\n{'='*70}")
    print(f"EVALUATION: {task_name} ({num_episodes} episodes, greedy)")
    print(f"{'='*70}")
    print(f"  Avg reward:       {total_reward/num_episodes:+.4f}")
    print(f"  Completion rate:  {total_completed/num_episodes:.1%}")
    print(f"  Cancellation:     {total_cancelled/num_episodes:.1%}")
    print(f"    Rider cancelled:  {total_rider_cancelled}")
    print(f"    Driver cancelled: {total_driver_cancelled}")
    print(f"  Timeout:          {total_timed_out/num_episodes:.1%}")
    print(f"  Avg steps:        {total_steps/num_episodes:.2f}")
    print(f"{'='*70}\n")

    return {
        "task": task_name,
        "avg_reward": round(total_reward / num_episodes, 4),
        "completion_rate": round(total_completed / num_episodes, 4),
        "cancellation_rate": round(total_cancelled / num_episodes, 4),
        "rider_cancellations": total_rider_cancelled,
        "driver_cancellations": total_driver_cancelled,
        "timeout_rate": round(total_timed_out / num_episodes, 4),
        "avg_steps": round(total_steps / num_episodes, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Q-learning agent for ride-hailing pricing")
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard", "all"])
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", type=int, default=11, help="Number of discrete price actions")
    parser.add_argument("--print-every", type=int, default=100)
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    all_results = {}
    for task in tasks:
        agent = train(
            task_name=task,
            num_episodes=args.episodes,
            seed=args.seed,
            num_actions=args.actions,
            print_every=args.print_every,
        )
        result = evaluate_agent(
            agent,
            task_name=task,
            num_episodes=args.eval_episodes,
            seed=99999,
            num_actions=args.actions,
        )
        all_results[task] = result

    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("SUMMARY ACROSS ALL TASKS")
        print("=" * 70)
        for task, res in all_results.items():
            print(f"  {task:>8s}: reward={res['avg_reward']:+.4f}  "
                  f"completion={res['completion_rate']:.1%}  "
                  f"cancel={res['cancellation_rate']:.1%}")
        avg_reward = sum(r["avg_reward"] for r in all_results.values()) / len(all_results)
        avg_completion = sum(r["completion_rate"] for r in all_results.values()) / len(all_results)
        print(f"  {'AVERAGE':>8s}: reward={avg_reward:+.4f}  completion={avg_completion:.1%}")
        print("=" * 70)


if __name__ == "__main__":
    main()
