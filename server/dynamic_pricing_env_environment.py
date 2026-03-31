# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dynamic Pricing Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

import random
import uuid

from openenv.core.env_server import Environment
from models import PricingAction, PricingObservation, PricingState


class DynamicPricingEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state = PricingState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            demand=20,
            supply=15,
            market_factor=1.0,
            price_multiplier=1.0,
            accepted_rides=0,
            revenue=0.0,
            total_reward=0.0,
            max_steps=10,
        )

    def _make_observation(self, message: str) -> PricingObservation:
        return PricingObservation(
            demand=self._state.demand,
            supply=self._state.supply,
            market_factor=round(self._state.market_factor, 2),
            price_multiplier=round(self._state.price_multiplier, 2),
            accepted_rides=self._state.accepted_rides,
            revenue=round(self._state.revenue, 2),
            message=message,
        )

    def reset(self) -> PricingObservation:
        self._state = PricingState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            demand=random.randint(15, 25),
            supply=random.randint(12, 20),
            market_factor=round(random.uniform(0.8, 1.3), 2),
            price_multiplier=1.0,
            accepted_rides=0,
            revenue=0.0,
            total_reward=0.0,
            max_steps=10,
        )
        return self._make_observation("New pricing episode started.")

    def step(self, action: PricingAction) -> PricingObservation:
        self._state.step_count += 1

        if action.move == "decrease":
            self._state.price_multiplier = max(0.7, self._state.price_multiplier - 0.1)
        elif action.move == "increase":
            self._state.price_multiplier = min(1.8, self._state.price_multiplier + 0.1)
        elif action.move == "hold":
            pass
        else:
            return self._make_observation(
                "Invalid action. Use one of: decrease, hold, increase."
            )

        raw_acceptance = (
            self._state.market_factor
            - 0.6 * (self._state.price_multiplier - 1.0)
        )
        acceptance_rate = max(0.1, min(1.0, raw_acceptance))

        accepted_rides = min(
            self._state.supply,
            int(self._state.demand * acceptance_rate)
        )

        revenue = accepted_rides * 10.0 * self._state.price_multiplier

        unserved_demand = max(0, self._state.demand - accepted_rides)
        overpricing_penalty = max(0.0, (self._state.price_multiplier - 1.3) * 8.0)

        reward = revenue - (unserved_demand * 2.0) - overpricing_penalty

        self._state.accepted_rides = accepted_rides
        self._state.revenue = revenue
        self._state.total_reward += reward

        self._state.demand = max(5, self._state.demand + random.randint(-3, 4))
        self._state.supply = max(5, self._state.supply + random.randint(-2, 3))
        self._state.market_factor = round(
            min(1.5, max(0.7, self._state.market_factor + random.uniform(-0.1, 0.1))),
            2,
        )

        done = self._state.step_count >= self._state.max_steps

        if done:
            return self._make_observation(
                f"Episode finished. Total reward = {round(self._state.total_reward, 2)}"
            )

        return self._make_observation(
            f"Action={action.move}, reward={round(reward, 2)}"
        )

    @property
    def state(self) -> PricingState:
        return self._state
