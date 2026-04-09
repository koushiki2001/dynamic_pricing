"""Medium task — tighter negotiation with profit pressure.

Objective:
    Act as the ride-hailing platform's pricing agent. Propose a price each round
    that is acceptable to BOTH the rider and the driver, closing the deal within
    5 steps while achieving positive platform profit.

    Success criteria (per episode):
        - ride_completed = True  (both rider and driver accept the proposed price)
        - platform_profit > 0.0  (price * commission_rate - operational_cost > 0)

    Patience decays faster than easy (0.12–0.22 per step), the price gap is wider
    (10–18 USD), and acceptance noise is higher. The agent must balance speed with
    pricing accuracy to avoid cancellations and unprofitable deals.

    Scoring weights:  completion 30% | efficiency 25% | profit 30% | no-cancel 15%
"""

from ..scenario_generator import ScenarioGenerator
from .graders import grade_medium

OBJECTIVE = (
    "Close a ride deal within 5 negotiation steps at a price that yields positive "
    "platform profit. Parties are moderately patient; balance speed and margin."
)


def sample_medium(seed: int = 84):
    return ScenarioGenerator(seed).generate("medium")


def evaluate_medium(policy_fn, num_episodes: int = 140, seed: int = 84) -> dict:
    return grade_medium(policy_fn, num_episodes=num_episodes, seed=seed)
