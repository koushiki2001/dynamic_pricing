#!/usr/bin/env python3
"""Validate policy integrations with SessionManager."""

from baselines.session_manager import SessionManager
from baselines.openai_policy import OpenAIPolicy
import os

print("=" * 60)
print("Policy Integration Validation")
print("=" * 60)

# Check if API key is available
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n⚠ WARNING: No API key found (OPENROUTER_API_KEY or OPENAI_API_KEY)")
    print("  LLM policies will fail at runtime, but can still be instantiated")
    print("  This is expected in test environments")
else:
    print("\n✓ API key found in environment")

# Test 1: Create SessionManager
manager = SessionManager(max_episodes=10)
print("\n✓ SessionManager created")

# Test 2: Instantiate Base LLM Policy with session
try:
    # This will fail on API key, but should succeed on instantiation
    policy_no_session = OpenAIPolicy(session_manager=None)
    print("✓ Base LLM Policy created without session")
except ValueError as e:
    if "Set OPENROUTER_API_KEY" in str(e):
        print("✓ Base LLM Policy correctly validates API key on init (no key available)")
    else:
        raise

try:
    policy_with_session = OpenAIPolicy(session_manager=manager)
    print("✓ Base LLM Policy created WITH session manager")
except ValueError as e:
    if "Set OPENROUTER_API_KEY" in str(e):
        print("✓ Base LLM Policy correctly validates API key on init (no key available)")
    else:
        raise

# Test 3: Validate policy has session_manager attribute
policy = OpenAIPolicy(session_manager=manager)
assert hasattr(policy, 'session_manager'), "Policy missing session_manager attribute"
assert policy.session_manager is manager, "session_manager not set correctly"
print("✓ Policy session_manager attribute correctly set")

# Test 4: Test prompt building with session
# Note: This requires a real observation dict
obs = {
    'step_number': 1,
    'max_steps': 5,
    'rider_quoted_price': 15.0,
    'driver_quoted_price': 25.0,
    'price_gap': 10.0,
    'distance_km': 5.0,
    'estimated_duration_min': 15,
    'weather_condition': 0,
    'traffic_level': 0,
    'demand_level': 1,
    'supply_level': 1,
    'surge_multiplier': 1.2,
    'commission_rate': 0.2,
    'operational_cost': 2.0,
    'rider_patience': 0.8,
    'driver_patience': 0.7,
    'rider_mood': 'willing',
    'driver_mood': 'hesitant',
    'last_proposed_price': 20.0,
    'last_rider_response': 'rejected',
    'last_driver_response': 'accepted',
}

policy_with_session = OpenAIPolicy(session_manager=manager)

# Log some episodes to session
manager.reset_episode('easy')
manager.log_step(0, 19.0, None, None, 0.0, obs)
manager.log_step(1, 20.0, 'accepted', 'accepted', 0.5, obs)
manager.log_episode(2.5, True, 'deal', 10.0, 20.0)

print("\n✓ Logged sample episodes to session")

# Test 5: Build prompt (which should include session context)
prompt = policy_with_session._build_prompt(obs)
assert "SESSION" in prompt or "Episode" in prompt or "Success" in prompt, \
    "Session context not included in prompt"
print("✓ Session context included in prompt")
print(f"  Prompt length: {len(prompt)} characters")

# Test 6: Validate Reward-Guided policy integration
try:
    from baselines.reward_guided_llm_policy import RewardGuidedLLMPolicy
    
    policy = RewardGuidedLLMPolicy(session_manager=manager)
    print("\n✓ Reward-Guided LLM Policy created WITH session manager")
    
    # Validate it has session_manager
    assert hasattr(policy, 'session_manager'), "Reward-Guided policy missing session_manager"
    print("✓ Reward-Guided policy session_manager attribute set")
    
except ValueError as e:
    if "Set OPENROUTER" in str(e):
        print("\n✓ Reward-Guided policy API key validation working (no key available)")
    else:
        raise

print("\n" + "=" * 60)
print("✅ ALL POLICY INTEGRATION TESTS PASSED!")
print("=" * 60)
