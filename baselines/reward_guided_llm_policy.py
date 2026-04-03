"""Reward-guided LLM policy (v2): wraps an LLM with adaptive bounds
tracking, reward-aware prompting, experience replay, and candidate scoring.

Key improvements over v1:
1. **Adaptive bounds** — tracks rejection history to narrow the feasible
   price range each step (rider rejected at $X → ceiling drops, etc.)
2. **Bounds-aware candidates** — generates candidates spread across the
   *narrowed* range, not static midpoint±15%.
3. **LLM-first scoring** — LLM candidate gets a trust bonus so the proxy
   only overrides it when a heuristic candidate is significantly better.
4. **Minimum price movement** — enforces ≥$1 change from last price to
   prevent oscillation.

Usage:
    from baselines.reward_guided_llm_policy import RewardGuidedLLMPolicy
    policy = RewardGuidedLLMPolicy(experience_path="data/experience_easy.json")
    policy.reset()
    action = policy(obs_dict)
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"

# ── labels ────────────────────────────────────────────────────────────
WEATHER = {0: "clear", 1: "rain", 2: "storm"}
TRAFFIC = {0: "low", 1: "medium", 2: "heavy"}
DEMAND  = {0: "low", 1: "medium", 2: "high"}
SUPPLY  = {0: "scarce", 1: "balanced", 2: "surplus"}


# ── experience store ──────────────────────────────────────────────────
class ExperienceStore:
    """Holds past (scenario → price → reward) tuples indexed for retrieval."""

    def __init__(self, path: Optional[str] = None):
        self.episodes: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            with open(path) as f:
                self.episodes = json.load(f)

    def add(self, entry: Dict[str, Any]):
        self.episodes.append(entry)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2, default=str)

    def find_similar(self, obs: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Find top_k most similar high-reward episodes."""
        if not self.episodes:
            return []

        scored = []
        for ep in self.episodes:
            if not ep.get("completed"):
                continue
            sim = self._similarity(obs, ep["initial_obs"])
            scored.append((sim, ep))

        scored.sort(key=lambda x: (x[0], -x[1].get("total_reward", 0)))
        return [ep for _, ep in scored[:top_k]]

    @staticmethod
    def _similarity(a: Dict, b: Dict) -> float:
        def safe(d, k, default=0):
            return float(d.get(k, default))
        dist = 0.0
        dist += ((safe(a, "price_gap") - safe(b, "price_gap")) / 50.0) ** 2
        dist += ((safe(a, "distance_km") - safe(b, "distance_km")) / 100.0) ** 2
        for k in ["weather_condition", "traffic_level", "demand_level", "supply_level"]:
            dist += ((safe(a, k) - safe(b, k)) / 2.0) ** 2
        dist += ((safe(a, "surge_multiplier", 1) - safe(b, "surge_multiplier", 1)) / 3.0) ** 2
        return math.sqrt(dist)


# ── adaptive reward proxy ─────────────────────────────────────────────
def estimate_reward_proxy(
    price: float,
    obs: Dict[str, Any],
    low_bound: float,
    high_bound: float,
) -> float:
    """Score a candidate price using adaptive bounds (not static quotes).

    Uses low_bound/high_bound (narrowed by rejection history) instead of
    raw quoted prices, so the proxy learns from feedback each step.
    """
    commission = obs["commission_rate"]
    op_cost = obs["operational_cost"]
    max_steps = obs["max_steps"]
    step = obs["step_number"]
    rider_patience = obs["rider_patience"]
    driver_patience = obs["driver_patience"]

    bound_range = max(high_bound - low_bound, 0.5)
    mid = (low_bound + high_bound) / 2.0

    # Acceptance probability based on position within bounds
    # Center of bounds → highest joint probability
    normalized_pos = (price - mid) / (bound_range / 2.0)  # -1 to +1 ideally

    # Rider prefers lower → penalise being above mid
    rider_accept_prob = 1.0 / (1.0 + math.exp(2.5 * normalized_pos))
    # Driver prefers higher → penalise being below mid
    driver_accept_prob = 1.0 / (1.0 + math.exp(-2.5 * normalized_pos))

    # Penalise prices outside the bounds
    if price < low_bound:
        driver_accept_prob *= 0.3
    if price > high_bound:
        rider_accept_prob *= 0.3

    joint_prob = rider_accept_prob * driver_accept_prob

    # Profit if completed
    profit = price * commission - op_cost
    steps_remaining = max_steps - step
    efficiency = min(max_steps / max(step + 1, 1), 3.0)

    expected_reward = joint_prob * profit * efficiency

    # Cancel risk
    cancel_risk = 0.0
    if rider_patience < 0.3 and price > mid:
        cancel_risk += 0.4
    if driver_patience < 0.3 and price < mid:
        cancel_risk += 0.4
    cancel_penalty = cancel_risk * 5.0

    # Timeout risk
    timeout_risk = 0.0
    if steps_remaining <= 1 and joint_prob < 0.5:
        timeout_risk = (1 - joint_prob) * 2.0

    return expected_reward - cancel_penalty - timeout_risk


# ── main policy ───────────────────────────────────────────────────────
class RewardGuidedLLMPolicy:
    """LLM + adaptive bounds + reward proxy scoring.

    Architecture:
        ┌──────────────────────────────────────────────────────┐
        │  Rejection history  →  Adaptive bounds [low, high]   │
        │                           ↓                          │
        │  Experience Store  →  Reward-Aware Prompt → LLM      │
        │                           ↓                          │
        │  LLM candidate + spread candidates across bounds     │
        │                           ↓                          │
        │  Reward proxy scores all → pick best (LLM gets bonus)│
        │                           ↓                          │
        │  Enforce min price movement from last proposal       │
        └──────────────────────────────────────────────────────┘
    """

    # Trust bonus: proxy must beat LLM by this much to override it
    LLM_TRUST_BONUS = 0.5
    # Minimum price change between consecutive proposals
    MIN_PRICE_MOVEMENT = 1.0

    def __init__(
        self,
        experience_path: Optional[str] = None,
        n_candidates: int = 1,
        top_k_examples: int = 3,
    ):
        api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "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"
        )
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        self._model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
        self._n_candidates = n_candidates
        self._top_k = top_k_examples
        self._store = ExperienceStore(experience_path)
        self._history: List[Dict[str, Any]] = []
        self._episode_log: List[Dict[str, Any]] = []
        # Adaptive bounds — narrowed each step by rejection feedback
        self._low_bound = 0.0
        self._high_bound = 0.0
        self._last_price: Optional[float] = None

    def reset(self):
        self._history = []
        self._episode_log = []
        self._low_bound = 0.0
        self._high_bound = 0.0
        self._last_price = None

    def get_episode_log(self) -> List[Dict[str, Any]]:
        return self._episode_log

    @property
    def experience_store(self) -> ExperienceStore:
        return self._store

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Initialise bounds on first step
        if obs["step_number"] == 0:
            self._low_bound = obs["rider_quoted_price"]
            self._high_bound = obs["driver_quoted_price"]
            self._last_price = None

        # Update bounds from rejection feedback
        self._update_bounds(obs)

        self._history.append({"step": obs["step_number"], "observation": obs})

        lo, hi = self._low_bound, self._high_bound

        # Step 1: Get LLM candidate (single call for speed)
        llm_price = self._get_llm_price(obs)

        # Step 2: Generate spread candidates across narrowed bounds
        candidates = self._generate_spread_candidates(lo, hi, obs)

        # Step 3: Score all candidates (LLM gets trust bonus)
        best_price = (lo + hi) / 2.0
        best_score = -999.0
        llm_score = -999.0
        score_log = []

        for price in candidates:
            score = estimate_reward_proxy(price, obs, lo, hi)
            score_log.append({"price": round(price, 2), "proxy_score": round(score, 4)})
            if score > best_score:
                best_score = score
                best_price = price

        if llm_price is not None:
            llm_raw_score = estimate_reward_proxy(llm_price, obs, lo, hi)
            llm_score = llm_raw_score + self.LLM_TRUST_BONUS
            score_log.append({
                "price": round(llm_price, 2),
                "proxy_score": round(llm_raw_score, 4),
                "source": "llm",
                "with_bonus": round(llm_score, 4),
            })
            if llm_score >= best_score:
                best_price = llm_price
                best_score = llm_score

        # Step 4: Enforce minimum price movement
        best_price = self._enforce_movement(best_price, obs, lo, hi)

        self._last_price = round(best_price, 2)

        self._episode_log.append({
            "step": obs["step_number"],
            "bounds": [round(lo, 2), round(hi, 2)],
            "llm_price": round(llm_price, 2) if llm_price else None,
            "candidates": score_log,
            "selected_price": self._last_price,
        })

        return {"type": "propose_price", "payload": {"price": self._last_price}}

    def _update_bounds(self, obs: Dict[str, Any]):
        """Narrow bounds based on last step's rejection feedback."""
        last_price = obs.get("last_proposed_price")
        if last_price is None:
            return

        lr = obs.get("last_rider_response")
        ld = obs.get("last_driver_response")

        if lr == "rejected" and ld == "accepted":
            # Price too high for rider — lower ceiling
            self._high_bound = min(self._high_bound, last_price - 0.5)
        elif lr == "accepted" and ld == "rejected":
            # Price too low for driver — raise floor
            self._low_bound = max(self._low_bound, last_price + 0.5)
        elif lr == "rejected" and ld == "rejected":
            # Both rejected — tighten both sides toward the middle
            mid = (self._low_bound + self._high_bound) / 2.0
            gap = self._high_bound - self._low_bound
            self._low_bound = mid - gap * 0.35
            self._high_bound = mid + gap * 0.35

        # Safety: keep bounds valid
        if self._low_bound > self._high_bound:
            mid = (self._low_bound + self._high_bound) / 2.0
            self._low_bound = mid - 1.0
            self._high_bound = mid + 1.0

    def _generate_spread_candidates(self, lo: float, hi: float, obs: Dict[str, Any]) -> List[float]:
        """Generate 7 candidates evenly spread across [lo, hi]."""
        spread = max(hi - lo, 1.0)
        mid = (lo + hi) / 2.0
        candidates = [
            lo + spread * 0.05,    # just above floor (driver-friendly minimum)
            lo + spread * 0.2,
            lo + spread * 0.35,
            mid,                    # center
            hi - spread * 0.35,
            hi - spread * 0.2,
            hi - spread * 0.05,    # just below ceiling (rider-friendly maximum)
        ]

        # Add urgency-driven candidates
        rider_p = obs["rider_patience"]
        driver_p = obs["driver_patience"]
        if rider_p < driver_p:
            # Rider more impatient → bias low
            candidates.append(lo + spread * 0.1)
        else:
            # Driver more impatient → bias high
            candidates.append(hi - spread * 0.1)

        return candidates

    def _enforce_movement(self, price: float, obs: Dict[str, Any],
                          lo: float, hi: float) -> float:
        """Ensure the price is at least MIN_PRICE_MOVEMENT away from last."""
        if self._last_price is None:
            return price

        if abs(price - self._last_price) >= self.MIN_PRICE_MOVEMENT:
            return price

        # Determine direction from feedback
        lr = obs.get("last_rider_response")
        ld = obs.get("last_driver_response")

        if lr == "rejected":
            # Rider rejected → move DOWN
            new_price = self._last_price - self.MIN_PRICE_MOVEMENT
        elif ld == "rejected":
            # Driver rejected → move UP
            new_price = self._last_price + self.MIN_PRICE_MOVEMENT
        else:
            # Both accepted last time (shouldn't happen mid-episode) — nudge toward mid
            mid = (lo + hi) / 2.0
            direction = 1.0 if mid > self._last_price else -1.0
            new_price = self._last_price + direction * self.MIN_PRICE_MOVEMENT

        # Clamp within bounds
        new_price = max(lo, min(hi, new_price))
        return new_price

    def _get_llm_price(self, obs: Dict[str, Any]) -> Optional[float]:
        """Get a single price from the LLM."""
        prompt = self._build_reward_aware_prompt(obs)
        system = self._build_system_prompt()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=100,
                )
                content = response.choices[0].message.content
                if content:
                    price = self._parse_price(content)
                    if price is not None:
                        return price
            except Exception:
                pass
            time.sleep(0.5 * (attempt + 1))
        return None

    def _build_system_prompt(self) -> str:
        return (
            "You are a ride-hailing platform pricing agent.\n"
            "\n"
            "YOUR OBJECTIVE: maximise platform reward defined as:\n"
            "  If both rider and driver accept:\n"
            "    reward = (price × commission_rate − operational_cost) × min(max_steps / steps_taken, 3.0)\n"
            "  If either party cancels (patience hits 0): reward = −5.0\n"
            "  If negotiation times out: reward = −2.0\n"
            "\n"
            "KEY INSIGHTS:\n"
            "• Higher prices → more profit BUT higher risk rider rejects.\n"
            "• Closing the deal FAST gives an efficiency bonus (up to 3×).\n"
            "• If someone's patience is low, accommodate them or risk −5.0.\n"
            "• Use rejection feedback: rider rejected → go lower, driver rejected → go higher.\n"
            "• NEVER propose the same price twice. Always adjust by at least $1.\n"
            "\n"
            "Respond ONLY with JSON: {\"price\": <number>}\n"
        )

    def _build_reward_aware_prompt(self, obs: Dict[str, Any]) -> str:
        lines = []

        lines.append("=== CURRENT SCENARIO ===")
        lines.append(f"Rider quoted: ${obs['rider_quoted_price']:.2f}")
        lines.append(f"Driver quoted: ${obs['driver_quoted_price']:.2f}")
        lines.append(f"Gap: ${obs['price_gap']:.2f}")
        lines.append(f"Distance: {obs['distance_km']}km, Duration: {obs['estimated_duration_min']}min")
        lines.append(
            f"Weather: {WEATHER[obs['weather_condition']]}, "
            f"Traffic: {TRAFFIC[obs['traffic_level']]}, "
            f"Demand: {DEMAND[obs['demand_level']]}, "
            f"Supply: {SUPPLY[obs['supply_level']]}"
        )
        lines.append(f"Surge: {obs['surge_multiplier']}x")
        lines.append(f"Commission: {obs['commission_rate']*100:.0f}%, Op cost: ${obs['operational_cost']:.2f}")
        lines.append(f"Step {obs['step_number']}/{obs['max_steps']}")
        lines.append(f"Rider: patience={obs['rider_patience']:.2f} ({obs['rider_mood']})")
        lines.append(f"Driver: patience={obs['driver_patience']:.2f} ({obs['driver_mood']})")

        # Show adaptive bounds to LLM
        lines.append(f"\nEstimated feasible range: ${self._low_bound:.2f} – ${self._high_bound:.2f}")

        # Feedback from previous steps
        if obs.get("last_proposed_price") is not None:
            lines.append(f"\nLast proposal: ${obs['last_proposed_price']:.2f}")
            lines.append(f"Rider: {obs['last_rider_response']}, Driver: {obs['last_driver_response']}")

            lr = obs["last_rider_response"]
            ld = obs["last_driver_response"]
            if lr == "rejected" and ld == "accepted":
                lines.append(f"→ Price was TOO HIGH for rider. Propose BELOW ${obs['last_proposed_price']:.2f}.")
            elif lr == "accepted" and ld == "rejected":
                lines.append(f"→ Price was TOO LOW for driver. Propose ABOVE ${obs['last_proposed_price']:.2f}.")
            elif lr == "rejected" and ld == "rejected":
                lines.append("→ BOTH rejected. Try the center of the feasible range.")

        # Experience replay
        examples = self._store.find_similar(obs, self._top_k)
        if examples:
            lines.append("\n=== SIMILAR HIGH-REWARD EXAMPLES ===")
            for i, ex in enumerate(examples, 1):
                lines.append(
                    f"  Ex {i}: gap=${ex.get('gap', '?')}, "
                    f"winning_price=${ex.get('winning_price', '?')}, "
                    f"reward={ex.get('total_reward', '?')}, "
                    f"steps={ex.get('steps_taken', '?')}"
                )

        # Urgency
        rider_p = obs["rider_patience"]
        driver_p = obs["driver_patience"]
        steps_left = obs["max_steps"] - obs["step_number"]
        if rider_p < 0.3 or driver_p < 0.3:
            who = "RIDER" if rider_p < driver_p else "DRIVER"
            lines.append(f"\n⚠ {who} IS ABOUT TO CANCEL (patience={min(rider_p, driver_p):.2f}).")
            lines.append(f"  Prioritise keeping them. Cancellation penalty = −5.0")
        if steps_left <= 1:
            lines.append(f"\n⚠ LAST STEP. Must close deal NOW or get timeout penalty (−2.0).")

        lines.append(f"\nPropose the optimal price. JSON: {{\"price\": <number>}}")
        return "\n".join(lines)

    @staticmethod
    def _parse_price(content: str) -> Optional[float]:
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            return float(parsed["price"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


def reward_guided_llm_factory(
    experience_path: Optional[str] = None,
) -> RewardGuidedLLMPolicy:
    return RewardGuidedLLMPolicy(experience_path=experience_path)
