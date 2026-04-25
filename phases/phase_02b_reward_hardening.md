# Phase 2b — Reward Hardening, Anti-Hacking, and Process Supervision

## Why This Phase Exists

The hackathon guide warns:

> "Your reward function is your task specification. If it is weak, incomplete, or easy to exploit, the model will optimize the wrong thing very efficiently."
> "Reward hacking is one of the biggest practical failure modes."

This phase runs **between Phase 2 (warm start) and Phase 3 (simulator LLM)**. By Phase 2 you have a training loop running. Before continuing to Phase 3, harden the reward signal so the platform model cannot find shortcuts.

**Exit criteria:** No reward hacking patterns observable in 50 sampled episodes. All reward components fire independently and correctly.

---

## Step 2b.1 — Audit Current Reward Components

The existing `reward.py` already has multi-component rewards. Map each component against the hackathon guide's checklist:

| Guide Component | Existing Implementation | Gap |
|---|---|---|
| Execution success | `ride_completed` in `compute_terminal_reward()` | ✅ Covered |
| Correctness | `platform_profit × efficiency_bonus` | ✅ Covered |
| Timeouts | `TIMEOUT_PENALTY = -2.0` | ✅ Covered |
| Cancellation | `CANCELLATION_PENALTY = -5.0` | ✅ Covered |
| Missed revenue | `missed_revenue_penalty` | ✅ Covered |
| **Format compliance** | **Not implemented** | ❌ Missing |
| **Anti-cheating checks** | **Not implemented** | ❌ Missing |
| **Step-level process signals** | Only `±0.05` partial accept shaping | ⚠️ Weak |

---

## Step 2b.2 — Add Format Compliance Reward

The platform LLM must output valid JSON: `{"price": <number>}`. If it outputs garbage, the rollout falls back to midpoint — the model is not penalized for bad output format, only rewarded for the fallback price.

This means the model can learn to output anything and rely on the fallback. That is reward hacking.

Add a format compliance component to `training/rollout.py`:

```python
import re, json

def score_format_compliance(completion: str) -> float:
    """
    Returns 0.1 if output is valid JSON with a 'price' key.
    Returns -0.1 if output is malformed (forced fallback).
    Small signal — does not dominate reward, just distinguishes format quality.
    """
    completion = completion.strip()
    try:
        # Strip markdown code fences if present
        if completion.startswith("```"):
            completion = completion.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(completion)
        if "price" in parsed and isinstance(parsed["price"], (int, float)):
            return 0.1
        return -0.05   # parsed but missing price key
    except Exception:
        return -0.1    # completely malformed

def compute_total_reward(env_reward: float, completion: str) -> float:
    """Combine environment reward with format compliance signal."""
    format_score = score_format_compliance(completion)
    return env_reward + format_score
```

Update `collect_platform_rollout()` in `rollout.py` to use `compute_total_reward()` instead of raw `result.reward`.

---

## Step 2b.3 — Anti-Hacking Checks

These are checks that run on every generated output during training. If any check triggers, the episode reward is zeroed or penalized regardless of what the environment returned.

Create `training/anti_hack.py`:

```python
"""
Anti-reward-hacking checks for platform LLM outputs.
Run after every generation during training.
"""

import re
from typing import Tuple

# Price bounds from config
MIN_PRICE = 1.0
MAX_PRICE = 500.0

# Suspicious patterns
HARDCODED_PATTERNS = [
    r'\b(99999|99998|0\.001|0\.01)\b',   # extreme values suggesting exploitation
]

def check_price_in_bounds(price: float) -> Tuple[bool, str]:
    if price < MIN_PRICE or price > MAX_PRICE:
        return False, f"price {price} out of bounds [{MIN_PRICE}, {MAX_PRICE}]"
    return True, ""

def check_no_hardcoded_exploit(completion: str) -> Tuple[bool, str]:
    """Detect if model is proposing known exploit values."""
    for pattern in HARDCODED_PATTERNS:
        if re.search(pattern, completion):
            return False, f"suspicious hardcoded value detected: {pattern}"
    return True, ""

def check_price_not_identical_to_last(price: float, last_price: float) -> Tuple[bool, str]:
    """
    Penalize repeating the exact same price.
    Environment already rejects duplicate prices, but we penalize here too
    so the model actively avoids them rather than relying on env rejection.
    """
    if abs(price - last_price) < 0.01:
        return False, f"identical price repeated: {price}"
    return True, ""

def check_price_reasonable_range(price: float, rider_quote: float, driver_quote: float) -> Tuple[bool, str]:
    """
    Penalize prices wildly outside the visible quote range.
    Proposing $1 when quotes are $20/$25 is not a valid strategy — it is exploiting
    the environment's willingness to accept any price within bounds.
    """
    lower_bound = min(rider_quote, driver_quote) * 0.5
    upper_bound = max(rider_quote, driver_quote) * 2.0
    if price < lower_bound or price > upper_bound:
        return False, f"price {price:.2f} unreasonably far from quotes [{rider_quote:.2f}, {driver_quote:.2f}]"
    return True, ""

def run_all_checks(
    price: float,
    completion: str,
    last_price: float,
    rider_quote: float,
    driver_quote: float,
) -> Tuple[float, list]:
    """
    Run all anti-hacking checks.
    Returns (penalty, list_of_violations).
    penalty is 0.0 if clean, negative if violations found.
    """
    violations = []
    checks = [
        check_price_in_bounds(price),
        check_no_hardcoded_exploit(completion),
        check_price_not_identical_to_last(price, last_price),
        check_price_reasonable_range(price, rider_quote, driver_quote),
    ]
    for passed, reason in checks:
        if not passed:
            violations.append(reason)

    penalty = -0.5 * len(violations)   # -0.5 per violation
    return penalty, violations
```

Update `collect_platform_rollout()` to call `run_all_checks()` at every step and add the penalty to the step reward.

---

## Step 2b.4 — Process-Aware Feedback (Step-Level Verifiers)

The current per-step shaping only gives `±0.05` for partial accept. This is weak — the model gets the same signal whether it just made a good strategic move or stumbled into a partial accept accidentally.

Add richer step-level verifiers to `reward.py`:

```python
def compute_process_reward(
    proposed_price: float,
    last_proposed_price: float,
    rider_accepted: bool,
    driver_accepted: bool,
    rider_patience: float,
    driver_patience: float,
    rider_patience_decay: float,
    driver_patience_decay: float,
    step_number: int,
    max_steps: int,
) -> float:
    """
    Process-level reward signals beyond simple partial accept shaping.
    These reward good intermediate reasoning, not just final outcomes.
    """
    reward = 0.0

    # 1. Convergence direction reward
    #    If rider rejected last time and model moved price DOWN → correct direction
    #    If driver rejected last time and model moved price UP   → correct direction
    if last_proposed_price is not None:
        price_change = proposed_price - last_proposed_price
        if not rider_accepted and price_change < 0:
            reward += 0.03   # moved toward rider after their rejection
        if not driver_accepted and price_change > 0:
            reward += 0.03   # moved toward driver after their rejection

    # 2. Patience-aware urgency reward
    #    Model should converge faster when patience is low
    min_patience = min(rider_patience, driver_patience)
    steps_remaining = max_steps - step_number
    if min_patience < 0.4 and steps_remaining <= 2:
        # Critical window — if model is still not closed, penalize
        if not (rider_accepted and driver_accepted):
            reward -= 0.1   # not closing when it should be urgent

    # 3. Partial accept with wrong direction penalty
    #    Rider accepted but model moves price UP further → wasted rider goodwill
    if rider_accepted and not driver_accepted:
        price_change = proposed_price - (last_proposed_price or proposed_price)
        if price_change < 0:
            reward -= 0.02  # moved away from driver after rider already accepted

    return reward
```

This gives the model signals for:
- Moving in the right direction after a rejection
- Urgency recognition (converge faster when patience is low)
- Not wasting partial accepts by moving away from the accepting party

---

## Step 2b.5 — Generation Inspection Protocol

From the hackathon guide:

> "Also inspect actual generations during training. A rising reward is not enough if the model is learning to exploit bugs."

Add a periodic inspection step to the training loop. Every 100 steps, print 3 sampled generations with their rewards and flag anything suspicious:

```python
def inspect_generations(samples: list, step: int, n_show: int = 3):
    """Print sampled generations for human review during training."""
    print(f"\n{'='*60}")
    print(f"GENERATION INSPECTION — Step {step}")
    print(f"{'='*60}")

    import random
    shown = random.sample(samples, min(n_show, len(samples)))

    for i, s in enumerate(shown):
        print(f"\n[Sample {i+1}]")
        print(f"  Reward:     {s['reward']:.3f}")
        print(f"  Price:      {s.get('price', '?')}")
        print(f"  Completion: {s['completion'][:120]!r}")

        # Flag suspicious patterns
        flags = []
        if s.get('reward', 0) > 5.0:
            flags.append("⚠️  VERY HIGH REWARD — check for exploitation")
        if 'fallback' in s.get('completion', '').lower():
            flags.append("⚠️  FALLBACK USED — model outputting invalid format")
        if s.get('price', 0) < 2.0 or s.get('price', 0) > 200:
            flags.append("⚠️  EXTREME PRICE — possible exploitation")

        for flag in flags:
            print(f"  {flag}")

    print(f"{'='*60}\n")
```

Call `inspect_generations(samples, step)` every 100 training steps inside the training loop. This step requires a human to look at the output — automation cannot replace this.

---

## Step 2b.6 — Rollback Plan

From the hackathon guide: "Terminate or roll back runs if behavior drifts badly."

Add a drift detection check to the training loop:

```python
def detect_reward_drift(reward_history: list, window: int = 50) -> bool:
    """
    Returns True if reward has dropped significantly from its peak.
    Signals potential reward hacking or training instability.
    """
    if len(reward_history) < window * 2:
        return False

    recent_window  = reward_history[-window:]
    previous_window = reward_history[-window*2:-window]

    recent_avg   = sum(recent_window) / len(recent_window)
    previous_avg = sum(previous_window) / len(previous_window)

    # If reward dropped more than 30% from previous window → drift alert
    if previous_avg > 0 and recent_avg < previous_avg * 0.70:
        return True
    return False

# In training loop:
if detect_reward_drift(reward_history):
    print("[WARNING] Reward drift detected — rolling back to last checkpoint")
    model.load_adapter("checkpoints/last_stable/platform_lora")
    # Reset optimizer state as well if accessible
```

Save a "last stable" checkpoint every 200 steps so rollback is always possible:
```python
if step % 200 == 0:
    model.save_pretrained("checkpoints/last_stable/platform_lora")
```

---

## Step 2b.7 — Inference Speed Optimization

From the hackathon guide: "In RL for LLMs, inference can dominate total runtime."

Apply these before starting full training runs:

```python
# 1. Enable Unsloth's fast inference mode after training setup
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)   # speeds up generation 2×

# 2. Limit max_new_tokens tightly — platform only needs a short JSON response
max_new_tokens = 32    # {"price": 18.50} is ~10 tokens — 32 is generous headroom

# 3. Batch generation where possible — generate multiple rollouts in one forward pass
# TRL's GRPOTrainer handles this via num_generations parameter

# 4. Disable gradient computation during rollout collection
with torch.inference_mode():   # faster than torch.no_grad()
    output_ids = model.generate(...)

# 5. Keep environment loop tight — no unnecessary sleep() calls, no debug prints in hot path
```

Benchmark rollout time before and after these changes:
```python
import time
start = time.time()
collect_platform_rollout(model, tokenizer, env, "easy")
print(f"Single rollout: {time.time()-start:.2f}s")
```

Target: < 2 seconds per episode on easy task. If slower, profile where time is spent.

---

## Summary of Changes to Existing Files

| File | Change |
|---|---|
| `ride_hailing_env/reward.py` | Add `compute_process_reward()` |
| `training/rollout.py` | Add format compliance scoring, call anti-hack checks, call process reward |
| `training/anti_hack.py` | **New file** — all anti-hacking checks |
| `training/train_platform.py` | Add `inspect_generations()` every 100 steps, drift detection, rollback |

---

## Deliverables for Phase 2b

- [ ] `training/anti_hack.py` — all 4 anti-hacking checks implemented and tested
- [ ] Format compliance reward wired into rollout reward computation
- [ ] `compute_process_reward()` added to `reward.py` and called in rollout
- [ ] `inspect_generations()` running every 100 training steps
- [ ] Drift detection and rollback logic in training loop
- [ ] Inference speed benchmarked, `FastLanguageModel.for_inference()` applied
- [ ] "Last stable" checkpoint saving every 200 steps confirmed working
- [ ] No reward hacking patterns in 50 manually inspected episodes
