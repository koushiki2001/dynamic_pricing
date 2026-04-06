"""Tests for session_manager module."""

import pytest
import json
from baselines.session_manager import SessionManager, StepLog, EpisodeLog, SessionSummaryBuilder


class TestStepLog:
    """Test StepLog dataclass."""
    
    def test_step_log_creation(self):
        """Test creating a StepLog."""
        step = StepLog(
            step_number=1,
            proposed_price=25.5,
            rider_response="accepted",
            driver_response="rejected",
            step_reward=-0.05,
            rider_patience=0.8,
            driver_patience=0.6,
            price_gap=5.0,
            observation_state={"key": "value"},
        )
        assert step.step_number == 1
        assert step.proposed_price == 25.5
        assert step.rider_response == "accepted"
        assert step.step_reward == -0.05


class TestEpisodeLog:
    """Test EpisodeLog dataclass."""
    
    def test_episode_log_creation(self):
        """Test creating an EpisodeLog."""
        steps = [
            StepLog(0, 20.0, None, None, 0.0, 0.9, 0.9, 5.0, {}),
            StepLog(1, 22.5, "rejected", "accepted", 0.05, 0.8, 0.9, 5.0, {}),
        ]
        episode = EpisodeLog(
            episode_id=1,
            task_difficulty="easy",
            steps=steps,
            episode_reward=2.5,
            completed=True,
            termination_reason="deal_accepted",
            initial_price_gap=5.0,
            final_proposed_price=22.5,
            timestamp="2024-01-01T10:00:00",
        )
        assert episode.episode_id == 1
        assert len(episode.steps) == 2
        assert episode.completed is True
        assert episode.episode_reward == 2.5

    def test_episode_log_to_dict(self):
        """Test converting EpisodeLog to dict."""
        steps = [StepLog(0, 20.0, None, None, 0.0, 0.9, 0.9, 5.0, {})]
        episode = EpisodeLog(
            episode_id=1,
            task_difficulty="easy",
            steps=steps,
            episode_reward=2.5,
            completed=True,
            termination_reason="deal_accepted",
            initial_price_gap=5.0,
            final_proposed_price=22.5,
            timestamp="2024-01-01T10:00:00",
        )
        d = episode.to_dict()
        assert d["episode_id"] == 1
        assert d["task_difficulty"] == "easy"
        assert len(d["steps"]) == 1
        assert d["episode_reward"] == 2.5


class TestSessionManager:
    """Test SessionManager functionality."""
    
    def test_init(self):
        """Test SessionManager initialization."""
        manager = SessionManager(max_episodes=10)
        assert manager.max_episodes == 10
        assert len(manager.episodes) == 0
        assert manager.current_episode_id == 0

    def test_reset_episode(self):
        """Test resetting for a new episode."""
        manager = SessionManager()
        manager.reset_episode("easy")
        assert manager._current_task == "easy"
        assert len(manager._current_steps) == 0

    def test_log_step(self):
        """Test logging individual steps."""
        manager = SessionManager()
        manager.reset_episode("easy")
        
        manager.log_step(
            step_number=0,
            proposed_price=20.0,
            rider_response=None,
            driver_response=None,
            step_reward=0.0,
            observation={"step_number": 0, "rider_patience": 0.9, "driver_patience": 0.9, "price_gap": 5.0},
        )
        
        assert len(manager._current_steps) == 1
        assert manager._current_steps[0].proposed_price == 20.0
        assert manager._current_steps[0].step_number == 0

    def test_log_episode(self):
        """Test logging episode completion."""
        manager = SessionManager()
        manager.reset_episode("medium")
        manager.log_step(
            step_number=0,
            proposed_price=20.0,
            rider_response="accepted",
            driver_response="accepted",
            step_reward=0.5,
            observation={"step_number": 0, "rider_patience": 0.9, "driver_patience": 0.9, "price_gap": 5.0},
        )
        
        manager.log_episode(
            episode_reward=2.5,
            completed=True,
            termination_reason="deal_accepted",
            initial_price_gap=5.0,
            final_proposed_price=20.0,
        )
        
        assert len(manager.episodes) == 1
        assert manager.episodes[0].episode_id == 0
        assert manager.episodes[0].completed is True
        assert manager.current_episode_id == 1

    def test_max_episodes_enforcement(self):
        """Test that max_episodes limit is enforced."""
        manager = SessionManager(max_episodes=3)
        
        for i in range(5):
            manager.reset_episode("easy")
            manager.log_step(
                step_number=0, proposed_price=20.0 + i,
                rider_response=None, driver_response=None,
                step_reward=0.0, observation={},
            )
            manager.log_episode(
                episode_reward=float(i), completed=True,
                termination_reason="deal", initial_price_gap=5.0,
            )
        
        assert len(manager.episodes) <= 3
        assert manager.episodes[-1].episode_id == 4

    def test_get_session_summary_empty(self):
        """Test getting summary when no episodes logged."""
        manager = SessionManager()
        summary = manager.get_session_summary()
        assert summary["episode_count"] == 0
        assert "summary" in summary

    def test_get_session_summary_with_data(self):
        """Test getting summary with episode data."""
        manager = SessionManager()
        
        # Log 3 successful episodes
        for i in range(3):
            manager.reset_episode("easy")
            for step in range(2):
                manager.log_step(
                    step_number=step,
                    proposed_price=20.0 + step,
                    rider_response="accepted" if step == 1 else None,
                    driver_response="accepted" if step == 1 else None,
                    step_reward=0.05 if step == 1 else 0.0,
                    observation={
                        "step_number": step,
                        "rider_patience": 0.9 - step * 0.1,
                        "driver_patience": 0.9 - step * 0.1,
                        "price_gap": 5.0,
                    },
                )
            manager.log_episode(
                episode_reward=2.5 + i * 0.1,
                completed=True,
                termination_reason="deal_accepted",
                initial_price_gap=5.0,
                final_proposed_price=21.0,
            )
        
        summary = manager.get_session_summary()
        assert summary["episode_count"] == 3
        assert summary["success_rate"] == "100%"
        assert summary["average_reward"] > 0
        assert "patterns" in summary
        assert "reward_breakdown" in summary

    def test_session_summary_success_rate(self):
        """Test success rate calculation."""
        manager = SessionManager()
        
        # Log 2 completed, 1 failed
        for i in range(3):
            manager.reset_episode("easy")
            manager.log_step(
                step_number=0, proposed_price=20.0,
                rider_response=None, driver_response=None,
                step_reward=0.0, observation={},
            )
            completed = (i < 2)
            manager.log_episode(
                episode_reward=2.5 if completed else -2.0,
                completed=completed,
                termination_reason="deal_accepted" if completed else "timeout",
                initial_price_gap=5.0,
            )
        
        summary = manager.get_session_summary()
        assert "67%" in summary["success_rate"] or summary["success_rate"] == "67%"

    def test_session_summary_patterns(self):
        """Test pattern extraction from sessions."""
        manager = SessionManager()
        
        manager.reset_episode("easy")
        # Successful price
        manager.log_step(0, 22.0, "accepted", "accepted", 0.5, {})
        # Rejected prices
        manager.log_step(1, 18.0, "rejected", None, -0.05, {})
        manager.log_step(2, 25.0, None, "rejected", -0.05, {})
        
        manager.log_episode(2.4, True, "deal", 5.0, 22.0)
        
        summary = manager.get_session_summary()
        patterns = summary.get("patterns", {})
        
        assert patterns.get("total_accepted_steps") == 1
        assert patterns.get("total_rejected_steps") == 2
        assert "successful_price_range" in patterns
        assert patterns["successful_price_range"]["min"] == 22.0
        assert patterns["successful_price_range"]["max"] == 22.0

    def test_session_summary_insights(self):
        """Test insight generation."""
        manager = SessionManager()
        
        # Log 10 episodes with high success rate
        for i in range(10):
            manager.reset_episode("easy")
            manager.log_step(0, 22.0, "accepted", "accepted", 0.5, {})
            manager.log_episode(2.5, True, "deal", 5.0, 22.0)
        
        summary = manager.get_session_summary()
        insights = summary.get("cross_episode_insights", [])
        
        assert len(insights) > 0
        assert any("70%" in i or "completion" in i.lower() for i in insights)

    def test_detailed_history(self):
        """Test getting detailed history."""
        manager = SessionManager()
        
        manager.reset_episode("easy")
        manager.log_step(0, 20.0, None, None, 0.0, {})
        manager.log_episode(2.5, True, "deal", 5.0, 20.0)
        
        history = manager.get_detailed_history()
        assert len(history) == 1
        assert history[0]["episode_id"] == 0
        assert len(history[0]["steps"]) == 1


class TestSessionSummaryBuilder:
    """Test SessionSummaryBuilder functionality."""
    
    def test_empty_episodes(self):
        """Test building summary from empty episodes."""
        builder = SessionSummaryBuilder([])
        summary = builder.build()
        assert summary["episode_count"] == 0

    def test_success_rate_display(self):
        """Test success rate display."""
        episodes = [
            EpisodeLog(0, "easy", [], 2.5, True, "deal", 5.0, 20.0, ""),
            EpisodeLog(1, "easy", [], -2.0, False, "timeout", 5.0, None, ""),
        ]
        builder = SessionSummaryBuilder(episodes)
        assert builder._calculate_success_rate() == "50%"

    def test_acceptance_rate_calculation(self):
        """Test acceptance rate calculation in patterns."""
        steps = [
            StepLog(0, 20.0, "accepted", "accepted", 0.5, 0.9, 0.9, 5.0, {}),
            StepLog(1, 21.0, "rejected", None, -0.05, 0.8, 0.9, 5.0, {}),
        ]
        episodes = [EpisodeLog(0, "easy", steps, 2.45, True, "deal", 5.0, 21.0, "")]
        builder = SessionSummaryBuilder(episodes)
        patterns = builder._extract_patterns()
        
        assert patterns["total_accepted_steps"] == 1
        assert patterns["total_rejected_steps"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
