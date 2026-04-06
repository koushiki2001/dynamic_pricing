"""Session management for tracking cross-episode LLM learning and negotiation patterns.

This module provides persistent session tracking across multiple episodes, allowing
LLM policies to learn from reward feedback and negotiation patterns they encounter.

Key features:
1. **Episode tracking**: Logs all proposals, responses, and rewards per episode
2. **Pattern analysis**: Identifies successful price ranges, common rejection patterns
3. **Session summarization**: Builds LLM-friendly JSON summaries of what's been learned
4. **Reward breakdown**: Tracks step rewards and terminal outcomes for insight

Usage:
    from baselines.session_manager import SessionManager
    session = SessionManager(max_episodes=20)
    
    # During policy.step() calls:
    session.log_step(step_data)
    
    # At episode end:
    session.log_episode(episode_outcome)
    
    # Build context for LLM:
    summary = session.get_session_summary(recent_episodes=5)
    # Use summary in prompt to LLM
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class StepLog:
    """Record of a single step within an episode."""
    step_number: int
    proposed_price: float
    rider_response: str  # "accepted", "rejected", or None
    driver_response: str  # "accepted", "rejected", or None
    step_reward: float
    rider_patience: float
    driver_patience: float
    price_gap: float
    observation_state: Dict[str, Any]  # Full obs for context


@dataclass
class EpisodeLog:
    """Record of a complete episode."""
    episode_id: int
    task_difficulty: str  # "easy", "medium", "hard"
    steps: List[StepLog]
    episode_reward: float
    completed: bool
    termination_reason: str
    initial_price_gap: float
    final_proposed_price: Optional[float]
    timestamp: str
    
    def to_dict(self):
        return {
            "episode_id": self.episode_id,
            "task_difficulty": self.task_difficulty,
            "steps": [asdict(s) for s in self.steps],
            "episode_reward": self.episode_reward,
            "completed": self.completed,
            "termination_reason": self.termination_reason,
            "initial_price_gap": self.initial_price_gap,
            "final_proposed_price": self.final_proposed_price,
            "timestamp": self.timestamp,
        }


class SessionManager:
    """Manages cross-episode session tracking for LLM learning."""

    def __init__(self, max_episodes: int = 20):
        """Initialize session manager.
        
        Args:
            max_episodes: Maximum number of episodes to keep in memory.
                         Older episodes are discarded when limit reached.
        """
        self.max_episodes = max_episodes
        self.episodes: List[EpisodeLog] = []
        self.current_episode_id = 0
        self._current_steps: List[StepLog] = []
        self._current_task: str = "unknown"

    def reset_episode(self, task_difficulty: str = "medium"):
        """Call at the start of each new episode.
        
        Args:
            task_difficulty: "easy", "medium", or "hard"
        """
        self._current_steps = []
        self._current_task = task_difficulty

    def log_step(
        self,
        step_number: int,
        proposed_price: float,
        rider_response: Optional[str],
        driver_response: Optional[str],
        step_reward: float,
        observation: Dict[str, Any],
    ):
        """Log a single step within the current episode.
        
        Args:
            step_number: Step index (0-based or 1-based, must be consistent)
            proposed_price: Price the agent proposed
            rider_response: "accepted", "rejected", or None
            driver_response: "accepted", "rejected", or None
            step_reward: Reward received for this step
            observation: The full observation dict from env
        """
        step_log = StepLog(
            step_number=step_number,
            proposed_price=proposed_price,
            rider_response=rider_response,
            driver_response=driver_response,
            step_reward=step_reward,
            rider_patience=observation.get("rider_patience", 0.0),
            driver_patience=observation.get("driver_patience", 0.0),
            price_gap=observation.get("price_gap", 0.0),
            observation_state=observation,
        )
        self._current_steps.append(step_log)

    def log_episode(
        self,
        episode_reward: float,
        completed: bool,
        termination_reason: str,
        initial_price_gap: float,
        final_proposed_price: Optional[float] = None,
    ):
        """Log the end of an episode and store it.
        
        Args:
            episode_reward: Total reward for this episode
            completed: Whether the ride was successfully completed
            termination_reason: Reason the episode ended (e.g., "deal_accepted", "timeout")
            initial_price_gap: Price gap at episode start
            final_proposed_price: Last price proposed (if any)
        """
        episode_log = EpisodeLog(
            episode_id=self.current_episode_id,
            task_difficulty=self._current_task,
            steps=self._current_steps.copy(),
            episode_reward=episode_reward,
            completed=completed,
            termination_reason=termination_reason,
            initial_price_gap=initial_price_gap,
            final_proposed_price=final_proposed_price,
            timestamp=datetime.now().isoformat(),
        )
        self.episodes.append(episode_log)
        self.current_episode_id += 1
        
        # Enforce max episodes
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
        
        self._current_steps = []

    def get_session_summary(self, recent_episodes: int = 5) -> Dict[str, Any]:
        """Build a structured summary of session history for LLM prompting.
        
        Args:
            recent_episodes: Number of most recent episodes to include in summary
            
        Returns:
            Dictionary with patterns, rewards, and insights for LLM
        """
        if not self.episodes:
            return {"episode_count": 0, "summary": "No previous session history"}
        
        recent = self.episodes[-recent_episodes:] if len(self.episodes) > recent_episodes else self.episodes
        
        return SessionSummaryBuilder(recent).build()

    def get_detailed_history(self) -> List[Dict[str, Any]]:
        """Get complete session history (for analysis/debugging)."""
        return [ep.to_dict() for ep in self.episodes]

    def save_session(self, path: str):
        """Save session history to JSON file."""
        data = {
            "session_metadata": {
                "total_episodes": len(self.episodes),
                "max_episodes": self.max_episodes,
                "session_created": datetime.now().isoformat(),
            },
            "episodes": self.get_detailed_history(),
        }
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class SessionSummaryBuilder:
    """Converts raw session logs into LLM-friendly JSON summaries."""

    def __init__(self, episodes: List[EpisodeLog]):
        self.episodes = episodes

    def build(self) -> Dict[str, Any]:
        """Build structured summary for LLM consumption."""
        if not self.episodes:
            return {"episode_count": 0, "summary": "No session history"}
        
        summary = {
            "episode_count": len(self.episodes),
            "success_rate": self._calculate_success_rate(),
            "average_reward": self._calculate_average_reward(),
            "total_steps_across_episodes": sum(len(ep.steps) for ep in self.episodes),
            "patterns": self._extract_patterns(),
            "reward_breakdown": self._analyze_reward_breakdown(),
            "cross_episode_insights": self._generate_insights(),
        }
        return summary

    def _calculate_success_rate(self) -> str:
        """Fraction of episodes completed successfully."""
        if not self.episodes:
            return "0%"
        completed = sum(1 for ep in self.episodes if ep.completed)
        pct = (completed / len(self.episodes)) * 100
        return f"{pct:.0f}%"

    def _calculate_average_reward(self) -> float:
        """Average episode reward across session."""
        if not self.episodes:
            return 0.0
        avg = sum(ep.episode_reward for ep in self.episodes) / len(self.episodes)
        return round(avg, 3)

    def _extract_patterns(self) -> Dict[str, Any]:
        """Identify successful/unsuccessful negotiation patterns."""
        successful_prices = []
        failed_prices = []
        accepted_steps = 0
        rejected_steps = 0
        
        for episode in self.episodes:
            for step in episode.steps:
                if step.rider_response == "accepted" and step.driver_response == "accepted":
                    successful_prices.append(step.proposed_price)
                    accepted_steps += 1
                elif step.rider_response == "rejected" or step.driver_response == "rejected":
                    failed_prices.append(step.proposed_price)
                    rejected_steps += 1
        
        patterns = {
            "acceptance_rate": f"{(accepted_steps / (accepted_steps + rejected_steps) * 100) if (accepted_steps + rejected_steps) > 0 else 0:.1f}%",
            "total_accepted_steps": accepted_steps,
            "total_rejected_steps": rejected_steps,
        }
        
        if successful_prices:
            patterns["successful_price_range"] = {
                "min": round(min(successful_prices), 2),
                "max": round(max(successful_prices), 2),
                "mean": round(sum(successful_prices) / len(successful_prices), 2),
            }
        
        if failed_prices:
            patterns["rejected_price_range"] = {
                "min": round(min(failed_prices), 2),
                "max": round(max(failed_prices), 2),
                "mean": round(sum(failed_prices) / len(failed_prices), 2),
            }
        
        return patterns

    def _analyze_reward_breakdown(self) -> Dict[str, Any]:
        """Analyze step rewards vs terminal rewards."""
        total_step_rewards = 0.0
        step_count = 0
        total_episode_rewards = 0.0
        
        for episode in self.episodes:
            total_episode_rewards += episode.episode_reward
            for step in episode.steps:
                total_step_rewards += step.step_reward
                step_count += 1
        
        avg_step_reward = (total_step_rewards / step_count) if step_count > 0 else 0.0
        
        return {
            "average_step_reward": round(avg_step_reward, 3),
            "total_step_rewards_collected": round(total_step_rewards, 2),
            "terminal_reward_contribution": round(sum(ep.episode_reward for ep in self.episodes) - total_step_rewards, 2),
            "average_episode_reward": round(total_episode_rewards / len(self.episodes), 3),
        }

    def _generate_insights(self) -> List[str]:
        """Generate human-readable insights from session."""
        insights = []
        
        success_rate = sum(1 for ep in self.episodes if ep.completed) / len(self.episodes)
        if success_rate > 0.7:
            insights.append("Strong performance: >70% deal completion rate. Continue current strategy.")
        elif success_rate < 0.3:
            insights.append("Low completion rate: <30% deals completing. Consider more aggressive initial offers.")
        
        # Identify common failure mode
        timeouts = sum(1 for ep in self.episodes if ep.termination_reason == "timeout")
        cancellations = sum(1 for ep in self.episodes if "cancelled" in ep.termination_reason.lower())
        if timeouts > len(self.episodes) * 0.4:
            insights.append("Pattern: Many timeouts. Proposals may be too conservative; widen bounds faster.")
        if cancellations > len(self.episodes) * 0.3:
            insights.append("Pattern: High cancellations. May need to propose closer to quoted prices.")
        
        # Price convergence
        all_prices = []
        for ep in self.episodes:
            for step in ep.steps:
                all_prices.append(step.proposed_price)
        if all_prices and len(all_prices) > 5:
            early_mean = sum(all_prices[:len(all_prices)//2]) / (len(all_prices)//2)
            late_mean = sum(all_prices[len(all_prices)//2:]) / (len(all_prices) - len(all_prices)//2)
            if late_mean < early_mean:
                insights.append(f"Trend: Proposals decreasing over time (${early_mean:.2f} → ${late_mean:.2f}). Adapting to driver floor pressure.")
        
        # Reward trend
        if len(self.episodes) > 2:
            early_reward = sum(ep.episode_reward for ep in self.episodes[:len(self.episodes)//2]) / (len(self.episodes)//2)
            late_reward = sum(ep.episode_reward for ep in self.episodes[len(self.episodes)//2:]) / (len(self.episodes) - len(self.episodes)//2)
            if late_reward > early_reward * 1.2:
                insights.append(f"Improvement: Rewards up {(late_reward/early_reward - 1)*100:.0f}%. Strategy refinement working.")
        
        if not insights:
            insights.append("Steady performance across session. Maintain current approach.")
        
        return insights
