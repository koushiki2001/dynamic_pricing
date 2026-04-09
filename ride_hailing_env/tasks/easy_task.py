"""Easy task — forgiving negotiation with patient parties.

Objective:
    Act as the ride-hailing platform's pricing agent. Propose a price each round
    that is acceptable to BOTH the rider and the driver, closing the deal within
    8 steps.

    Success criteria (per episode):
        - ride_completed = True  (both rider and driver accept the proposed price)

    The rider and driver start with high patience (0.85–1.0) and decay slowly,
    giving the agent room to explore. The price gap is moderate (6–14 USD) with
    generous acceptance slack on both sides.

    Scoring weights:  completion 40% | efficiency 30% | profit 20% | no-cancel 10%
"""

from ..scenario_generator import ScenarioGenerator
from .graders import grade_easy

OBJECTIVE = (
    "Close a ride deal (both rider and driver accept) within 8 negotiation steps. "
    "Rider and driver are patient; focus on finding a mutually acceptable price."
)


def sample_easy(seed: int = 42):
    return ScenarioGenerator(seed).generate("easy")


def evaluate_easy(policy_fn, num_episodes: int = 120, seed: int = 42) -> dict:
    return grade_easy(policy_fn, num_episodes=num_episodes, seed=seed)
