"""Anti-reward-hacking checks for platform LLM outputs (Phase 2b).

Every generated price + completion is run through these checks before the
reward is finalised. Violations add a penalty so the model cannot silently
exploit environment edge-cases.

Usage inside the rollout loop::

    penalty, violations = run_all_checks(
        price=price,
        completion=completion,
        last_price=last_price,
        rider_quote=obs.rider_quoted_price,
        driver_quote=obs.driver_quoted_price,
    )
    total_reward = env_reward + format_score + penalty
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

# Prices wildly outside the visible quote range suggest the model is exploiting
# the environment's broad [1, 500] acceptance window rather than negotiating.
_RANGE_MULTIPLIER_LOW = 0.5
_RANGE_MULTIPLIER_HIGH = 2.0

# Values that appear when models try to exploit numeric edge-cases
_EXPLOIT_PATTERNS = [
    r"\b99999\b",
    r"\b99998\b",
    r"\b0\.001\b",
    r"\b0\.01\b",
]

_PENALTY_PER_VIOLATION = -0.5


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_price_in_bounds(price: float, lo: float = 1.0, hi: float = 500.0) -> Tuple[bool, str]:
    if price < lo or price > hi:
        return False, f"price {price:.2f} outside hard bounds [{lo}, {hi}]"
    return True, ""


def check_no_hardcoded_exploit(completion: str) -> Tuple[bool, str]:
    for pat in _EXPLOIT_PATTERNS:
        if re.search(pat, completion):
            return False, f"exploit pattern detected: {pat}"
    return True, ""


def check_price_not_repeated(price: float, last_price: float) -> Tuple[bool, str]:
    if last_price is not None and abs(price - last_price) < 0.01:
        return False, f"price {price:.2f} identical to last proposal — no movement"
    return True, ""


def check_price_reasonable_range(
    price: float,
    rider_quote: float,
    driver_quote: float,
) -> Tuple[bool, str]:
    lo = min(rider_quote, driver_quote) * _RANGE_MULTIPLIER_LOW
    hi = max(rider_quote, driver_quote) * _RANGE_MULTIPLIER_HIGH
    if price < lo or price > hi:
        return False, (
            f"price {price:.2f} unreasonably far from quotes "
            f"[{rider_quote:.2f}, {driver_quote:.2f}] — expected range [{lo:.2f}, {hi:.2f}]"
        )
    return True, ""


# ---------------------------------------------------------------------------
# Format compliance
# ---------------------------------------------------------------------------

def score_format_compliance(completion: str) -> float:
    """Return +0.1 for valid JSON with a numeric price key, −0.1 otherwise."""
    text = completion.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        if "price" in parsed and isinstance(parsed["price"], (int, float)):
            return 0.1
        return -0.05  # parsed but missing price key
    except Exception:
        return -0.1   # completely malformed


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def run_all_checks(
    price: float,
    completion: str,
    last_price: float,
    rider_quote: float,
    driver_quote: float,
) -> Tuple[float, List[str]]:
    """Run every anti-hacking check and return (total_penalty, violations).

    penalty is 0.0 when clean, negative proportional to number of violations.
    """
    results = [
        check_price_in_bounds(price),
        check_no_hardcoded_exploit(completion),
        check_price_not_repeated(price, last_price),
        check_price_reasonable_range(price, rider_quote, driver_quote),
    ]
    violations = [reason for passed, reason in results if not passed]
    penalty = _PENALTY_PER_VIOLATION * len(violations)
    return penalty, violations
