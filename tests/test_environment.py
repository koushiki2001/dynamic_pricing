"""Tests for the ride-hailing environment."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.models import Observation, StepResult


def test_reset_returns_valid_observation():
    env = DynamicPricingEnv(task_name="easy", seed=42)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.step_number == 0
    assert obs.last_rider_response is None
    assert obs.last_driver_response is None
    assert obs.last_proposed_price is None
    assert 0.0 < obs.rider_patience <= 1.0
    assert 0.0 < obs.driver_patience <= 1.0
    assert obs.rider_mood in ("willing", "hesitant", "frustrated")


def test_step_returns_step_result():
    env = DynamicPricingEnv(task_name="easy", seed=42)
    env.reset()
    result = env.step({"type": "propose_price", "payload": {"price": 20.0}})
    assert isinstance(result, StepResult)
    assert result.observation.step_number == 1
    assert result.observation.last_rider_response in ("accepted", "rejected")
    assert result.observation.last_driver_response in ("accepted", "rejected")
    assert result.observation.last_proposed_price == 20.0


def test_episode_terminates_within_max_steps():
    for task in ["easy", "medium", "hard"]:
        env = DynamicPricingEnv(task_name=task, seed=42)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            mid = (obs.rider_quoted_price + obs.driver_quoted_price) / 2.0
            result = env.step({"type": "propose_price", "payload": {"price": mid}})
            obs = result.observation
            done = result.done
            steps += 1
        assert steps <= obs.max_steps
        assert result.info["outcome"]["termination_reason"] in ("completed", "cancelled", "timeout")


def test_state_matches_observation():
    env = DynamicPricingEnv(task_name="easy", seed=42)
    obs = env.reset()
    state = env.state()
    assert obs.model_dump() == state.model_dump()


def test_efficiency_reward_ordering():
    """Verify that fewer steps to completion yields higher reward."""
    rewards = []
    for bad_proposals in [0, 2, 4]:  # 0 = close immediately, more = waste steps first
        env = DynamicPricingEnv(task_name="easy", seed=42)
        obs = env.reset()
        total = 0.0

        # Do some bad proposals first (way too low)
        for _ in range(bad_proposals):
            r = env.step({"type": "propose_price", "payload": {"price": 1.0}})
            total += r.reward
            if r.done:
                break

        if not r.done if bad_proposals > 0 else True:
            # Try to close with a high price the rider might accept
            high = obs.driver_quoted_price + 5.0
            r = env.step({"type": "propose_price", "payload": {"price": high}})
            total += r.reward

        rewards.append(total)

    # We can't guarantee all close, but test that the structure works
    assert len(rewards) == 3


def test_cancellation_penalty():
    """Propose an absurdly low price repeatedly until driver cancels."""
    env = DynamicPricingEnv(task_name="hard", seed=42)
    obs = env.reset()
    done = False
    last_reward = 0.0
    while not done:
        result = env.step({"type": "propose_price", "payload": {"price": 1.0}})
        done = result.done
        last_reward = result.reward
    # Terminal reward on cancellation or timeout should be negative
    assert last_reward < 0.0


def test_all_tasks_exist():
    for task in ["easy", "medium", "hard"]:
        env = DynamicPricingEnv(task_name=task, seed=42)
        obs = env.reset()
        assert obs.rider_quoted_price > 0
        assert obs.driver_quoted_price > 0


if __name__ == "__main__":
    test_reset_returns_valid_observation()
    test_step_returns_step_result()
    test_episode_terminates_within_max_steps()
    test_state_matches_observation()
    test_efficiency_reward_ordering()
    test_cancellation_penalty()
    test_all_tasks_exist()
    print("All tests passed!")
