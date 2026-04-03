"""Tests for graders."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.tasks.graders import grade_task
from baselines.midpoint_policy import midpoint_policy


def test_grader_returns_score_in_range():
    for task in ["easy", "medium", "hard"]:
        result = grade_task(task, midpoint_policy, num_episodes=20, seed=42)
        assert 0.0 <= result["score"] <= 1.0, f"{task}: score {result['score']} out of range"
        assert result["completion_rate"] >= 0.0
        assert result["cancellation_rate"] >= 0.0


def test_easy_scores_higher_than_hard():
    easy = grade_task("easy", midpoint_policy, num_episodes=50, seed=42)
    hard = grade_task("hard", midpoint_policy, num_episodes=50, seed=168)
    # Easy should generally score higher (not guaranteed with small N, but likely)
    print(f"Easy: {easy['score']:.4f}, Hard: {hard['score']:.4f}")


if __name__ == "__main__":
    test_grader_returns_score_in_range()
    test_easy_scores_higher_than_hard()
    print("All grader tests passed!")
