from ..scenario_generator import ScenarioGenerator
from .graders import grade_medium


def sample_medium(seed: int = 84):
    return ScenarioGenerator(seed).generate("medium")


def evaluate_medium(policy_fn, num_episodes: int = 140, seed: int = 84) -> dict:
    return grade_medium(policy_fn, num_episodes=num_episodes, seed=seed)
