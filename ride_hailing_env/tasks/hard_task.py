"""Hard task — impatient parties, high surge, tight step budget.

Objective:
    Act as the ride-hailing platform's pricing agent. Propose a price each round
    that is acceptable to BOTH the rider and the driver, closing the deal within
    4 steps while maximising platform profit under high-surge, adverse conditions.

    Success criteria (per episode):
        - ride_completed = True  (both rider and driver accept the proposed price)
        - platform_profit > 1.0  (meaningful margin after operational cost)
        - steps_taken <= 3       (efficient closure — leaving the last step unused)

    Parties start with low patience (0.55–0.85) and decay rapidly (0.18–0.35 per
    step), meaning a single bad proposal can trigger cancellation. Surge is high
    (1.5–3.0x), price gaps are wide (12–22 USD), and acceptance noise is severe.
    The agent must be decisive and price near-optimally from the first step.

    Scoring weights:  completion 25% | efficiency 20% | profit 35% | no-cancel 20%
"""

from ..scenario_generator import ScenarioGenerator
from .graders import grade_hard

OBJECTIVE = (
    "Close a ride deal within 4 steps under high-surge, impatient conditions. "
    "Prioritise platform profit margin and avoid cancellations — mistakes are "
    "costly with almost no room to recover."
)


def sample_hard(seed: int = 168):
    return ScenarioGenerator(seed).generate("hard")


def evaluate_hard(policy_fn, num_episodes: int = 160, seed: int = 168) -> dict:
    return grade_hard(policy_fn, num_episodes=num_episodes, seed=seed)
