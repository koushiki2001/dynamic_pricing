from ..scenario_generator import ScenarioGenerator
from .graders import grade_hard


def sample_hard(seed: int = 168):
    return ScenarioGenerator(seed).generate("hard")


def evaluate_hard(policy_fn, num_episodes: int = 160, seed: int = 168) -> dict:
    return grade_hard(policy_fn, num_episodes=num_episodes, seed=seed)
