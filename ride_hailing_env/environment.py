"""Core OpenEnv environment: reset() / step() / state()."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import MIN_OFFER_PRICE, MAX_OFFER_PRICE, TASK_CONFIG
from .models import Action, EpisodeOutcome, HiddenState, Observation, StepResult
from .reward import compute_step_reward, compute_terminal_reward
from .scenario_generator import ScenarioGenerator
from .simulator import Simulator
from .utils import clamp, patience_to_mood


class DynamicPricingEnv:
    """Multi-step ride-hailing pricing negotiation environment.

    The agent plays the platform, proposing prices each round.
    Rider and driver independently accept or reject.
    Episode ends on mutual acceptance, cancellation, or timeout.
    """

    def __init__(self, task_name: str = "easy", seed: int = 42) -> None:
        if task_name not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASK_CONFIG.keys())}")
        self.task_name = task_name
        self.seed = seed
        self._generator = ScenarioGenerator(seed)
        self._simulator = Simulator()
        self._observation: Optional[Observation] = None
        self._hidden: Optional[HiddenState] = None
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history: list[Dict[str, Any]] = []

    def reset(self) -> Observation:
        """Initialize a new episode. Returns the initial observation."""
        self._generator = ScenarioGenerator(self.seed)
        self._simulator = Simulator()
        self._observation, self._hidden = self._generator.generate(self.task_name)
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history = []
        return self._observation

    def state(self) -> Observation:
        """Return the current observable state."""
        if self._observation is None:
            raise RuntimeError("Call reset() before state().")
        return self._observation

    def step(self, action: Any) -> StepResult:
        """Process one negotiation round.

        Args:
            action: Either an Action model, a dict {"type": "propose_price", "payload": {"price": X}},
                    or a raw float.
        """
        if self._observation is None or self._hidden is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Parse action
        price = self._parse_price(action)
        price = clamp(price, MIN_OFFER_PRICE, MAX_OFFER_PRICE)

        # Reject duplicate prices: don't waste a step if the price hasn't changed
        last_price = self._observation.last_proposed_price
        if last_price is not None and abs(price - last_price) < 0.01:
            return StepResult(
                observation=self._observation,
                reward=0.0,
                done=False,
                info={"duplicate_price": True, "message": "Price unchanged — propose a different price."},
            )

        self._step_count += 1

        # Simulate rider/driver reactions
        sim_result = self._simulator.simulate_step(self._observation, self._hidden, price)

        # Determine terminal state
        at_max_steps = self._step_count >= self._observation.max_steps
        ride_completed = sim_result.both_accepted
        cancelled = sim_result.any_cancelled
        timed_out = at_max_steps and not ride_completed and not cancelled

        done = ride_completed or cancelled or timed_out

        # Compute reward
        if done:
            reward = compute_terminal_reward(
                proposed_price=price,
                commission_rate=self._observation.commission_rate,
                operational_cost=self._observation.operational_cost,
                steps_taken=self._step_count,
                max_steps=self._observation.max_steps,
                ride_completed=ride_completed,
                timed_out=timed_out,
                rider_max_willingness=self._hidden.rider_max_willingness,
            )
        else:
            reward = compute_step_reward(sim_result)

        self._cumulative_reward += reward

        # Build outcome info
        outcome = EpisodeOutcome(
            ride_completed=ride_completed,
            rider_accepted=sim_result.rider_accepted,
            driver_accepted=sim_result.driver_accepted,
            rider_cancelled=sim_result.rider_cancelled,
            driver_cancelled=sim_result.driver_cancelled,
            timed_out=timed_out,
            final_price=price if ride_completed else None,
            platform_profit=(price * self._observation.commission_rate - self._observation.operational_cost)
            if ride_completed
            else None,
            steps_taken=self._step_count,
            termination_reason=(
                "completed" if ride_completed else "cancelled" if cancelled else "timeout" if timed_out else "ongoing"
            ),
        )

        # Record action
        self._action_history.append({
            "step": self._step_count,
            "proposed_price": round(price, 2),
            "rider_response": "accepted" if sim_result.rider_accepted else "rejected",
            "driver_response": "accepted" if sim_result.driver_accepted else "rejected",
            "reward": round(reward, 4),
        })

        # Update observation for next step
        self._observation = self._observation.model_copy(update={
            "step_number": self._step_count,
            "rider_patience": sim_result.new_rider_patience,
            "driver_patience": sim_result.new_driver_patience,
            "rider_mood": sim_result.new_rider_mood,
            "driver_mood": sim_result.new_driver_mood,
            "last_rider_response": "accepted" if sim_result.rider_accepted else "rejected",
            "last_driver_response": "accepted" if sim_result.driver_accepted else "rejected",
            "last_proposed_price": round(price, 2),
        })

        self._done = done

        missed_revenue_penalty = (
            round(max(0.0, self._hidden.rider_max_willingness - price) * self._observation.commission_rate, 4)
            if ride_completed else 0.0
        )

        info: Dict[str, Any] = {
            "task_name": self.task_name,
            "seed": self.seed,
            "outcome": outcome.model_dump(),
            "action_history": self._action_history,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "missed_revenue_penalty": missed_revenue_penalty,
            "reward_threshold": self._hidden.reward_threshold,
            "penalty_threshold": self._hidden.penalty_threshold,
        }

        return StepResult(
            observation=self._observation,
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    def _parse_price(self, action: Any) -> float:
        if isinstance(action, Action):
            return action.price
        if isinstance(action, dict):
            payload = action.get("payload", action)
            return float(payload.get("price", payload.get("final_offer_price", 0.0)))
        return float(action)
