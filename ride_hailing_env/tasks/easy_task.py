from ..scenario_generator import ScenarioGenerator
from .graders import grade_easy


def sample_easy(seed: int = 42):
    return ScenarioGenerator(seed).generate("easy")


def evaluate_easy(policy_fn, num_episodes: int = 120, seed: int = 42) -> dict:
    return grade_easy(policy_fn, num_episodes=num_episodes, seed=seed)
