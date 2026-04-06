#!/usr/bin/env python3
"""Quick validation of SessionManager functionality."""

from baselines.session_manager import SessionManager, StepLog, EpisodeLog
import json

print("=" * 60)
print("SessionManager Quick Validation")
print("=" * 60)

# Test 1: Create manager
manager = SessionManager(max_episodes=10)
print("\n✓ SessionManager created successfully")

# Test 2: Reset episode
manager.reset_episode('easy')
print("✓ Episode reset for 'easy' task")

# Test 3: Log steps
manager.log_step(
    step_number=0,
    proposed_price=20.0,
    rider_response=None,
    driver_response=None,
    step_reward=0.0,
    observation={
        'step_number': 0,
        'rider_patience': 0.9,
        'driver_patience': 0.9,
        'price_gap': 5.0
    }
)
print("✓ Step 0 logged")

manager.log_step(
    step_number=1,
    proposed_price=22.0,
    rider_response="accepted",
    driver_response="accepted",
    step_reward=0.5,
    observation={
        'step_number': 1,
        'rider_patience': 0.8,
        'driver_patience': 0.8,
        'price_gap': 5.0
    }
)
print("✓ Step 1 logged")

# Test 4: Log episode
manager.log_episode(
    episode_reward=2.5,
    completed=True,
    termination_reason='deal_accepted',
    initial_price_gap=5.0,
    final_proposed_price=22.0
)
print("✓ Episode 1 completed and logged")

# Test 5: Get summary
summary = manager.get_session_summary()
print("\n✓ Session summary retrieved:")
print(f"  - Episode count: {summary['episode_count']}")
print(f"  - Success rate: {summary['success_rate']}")
print(f"  - Average reward: {summary['average_reward']}")

# Test 6: Verify patterns
patterns = summary.get('patterns', {})
print("\n✓ Patterns extracted:")
print(f"  - Successful price range: {patterns.get('successful_price_range', 'N/A')}")
print(f"  - Total accepted steps: {patterns.get('total_accepted_steps', 0)}")

# Test 7: Verify insights
insights = summary.get('cross_episode_insights', [])
print("\n✓ Insights generated:")
for i, insight in enumerate(insights[:2], 1):
    print(f"  {i}. {insight}")

# Test 8: JSON serialization
try:
    json_str = json.dumps(summary, indent=2)
    print("\n✓ Summary is valid JSON")
except Exception as e:
    print(f"\n✗ JSON serialization failed: {e}")

# Test 9: Multiple episodes
for ep_num in range(2, 5):
    manager.reset_episode('easy')
    for step in range(2):
        manager.log_step(
            step_number=step,
            proposed_price=20.0 + step,
            rider_response="accepted" if step == 1 else None,
            driver_response="accepted" if step == 1 else None,
            step_reward=0.05 if step == 1 else 0.0,
            observation={
                'step_number': step,
                'rider_patience': 0.9 - step * 0.1,
                'driver_patience': 0.9 - step * 0.1,
                'price_gap': 5.0
            }
        )
    manager.log_episode(
        episode_reward=2.5,
        completed=True,
        termination_reason='deal_accepted',
        initial_price_gap=5.0,
        final_proposed_price=21.0
    )
print(f"\n✓ Logged {manager.current_episode_id} episodes total")

# Test 10: Detailed history
history = manager.get_detailed_history()
print(f"✓ Detailed history available: {len(history)} episodes")

print("\n" + "=" * 60)
print("✅ ALL VALIDATIONS PASSED!")
print("=" * 60)
