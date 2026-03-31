# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Dynamic Pricing Env Environment.

The dynamic_pricing_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server import Action, Observation, State


class PricingAction(Action):
    move: str


class PricingObservation(Observation):
    demand: int
    supply: int
    market_factor: float
    price_multiplier: float
    accepted_rides: int
    revenue: float
    message: str


class PricingState(State):
    demand: int = 20
    supply: int = 15
    market_factor: float = 1.0
    price_multiplier: float = 1.0
    accepted_rides: int = 0
    revenue: float = 0.0
    total_reward: float = 0.0
    max_steps: int = 10
