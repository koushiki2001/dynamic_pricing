"""LLM-based multi-step negotiation policy via OpenRouter."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


class OpenAIPolicy:
    def __init__(self, session_manager: Optional[Any] = None):
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("Set HF_TOKEN environment variable in .env or environment")
        base_url = os.getenv("API_BASE_URL")
        if not base_url:
            raise ValueError("Set API_BASE_URL environment variable in .env or environment")
        model = os.getenv("MODEL_NAME")
        if not model:
            raise ValueError("Set MODEL_NAME environment variable in .env or environment")
        print(f"[LLM-INIT] base_url={base_url} model={model} api_key={api_key[:8]}...{api_key[-4:]}")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._history: List[Dict[str, Any]] = []
        self.session_manager = session_manager

    def reset(self):
        self._history = []

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._history.append({"step": obs["step_number"], "observation": obs})
        prompt = self._build_prompt(obs)
        fallback = (obs["rider_quoted_price"] + obs["driver_quoted_price"]) / 2.0

        messages = [
            {"role": "system", "content": "You are a ride-hailing platform pricing agent. Respond ONLY with a JSON object: {\"price\": <number>}"},
            {"role": "user", "content": prompt},
        ]

        # Retry up to 3 times for free-tier rate limits / empty responses
        for attempt in range(3):
            try:
                print(f"[LLM-REQ] attempt={attempt+1} model={self._model} step={obs['step_number']}")
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=100,
                )
                content = response.choices[0].message.content
                print(f"[LLM-RESP] content={content!r} usage={getattr(response, 'usage', None)}")
                if content:
                    return self._parse_response(content, obs)
                else:
                    print(f"[LLM-WARN] Empty response from model")
            except Exception as e:
                print(f"[LLM-ERR] attempt={attempt+1} error={type(e).__name__}: {e}")
            time.sleep(1.0 * (attempt + 1))

        print(f"[LLM-FALLBACK] All attempts failed, using midpoint={fallback:.2f}")
        return {"type": "propose_price", "payload": {"price": round(fallback, 2)}}

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        lines = [
            f"Negotiate a ride price acceptable to both rider and driver.",
            f"",
            f"Rider quoted: ${obs['rider_quoted_price']:.2f}",
            f"Driver quoted: ${obs['driver_quoted_price']:.2f}",
            f"Gap: ${obs['price_gap']:.2f}",
            f"",
            f"Context: distance={obs['distance_km']}km, duration={obs['estimated_duration_min']}min",
            f"Weather: {['clear','rain','storm'][obs['weather_condition']]}, "
            f"Traffic: {['low','medium','heavy'][obs['traffic_level']]}, "
            f"Demand: {['low','medium','high'][obs['demand_level']]}, "
            f"Supply: {['low','medium','high'][obs['supply_level']]}",
            f"Surge: {obs['surge_multiplier']}x",
            f"Commission: {obs['commission_rate']*100:.0f}%, Op cost: ${obs['operational_cost']:.2f}",
            f"",
            f"Step {obs['step_number']}/{obs['max_steps']}",
            f"Rider patience: {obs['rider_patience']:.2f} ({obs['rider_mood']})",
            f"Driver patience: {obs['driver_patience']:.2f} ({obs['driver_mood']})",
        ]

        if obs.get("last_proposed_price") is not None:
            lines.extend([
                f"",
                f"Your last proposal: ${obs['last_proposed_price']:.2f}",
                f"Rider response: {obs['last_rider_response']}",
                f"Driver response: {obs['last_driver_response']}",
            ])

        if len(self._history) > 1:
            lines.append(f"\nPrevious steps: {len(self._history) - 1}")

        # Add session context if available
        if self.session_manager:
            session_summary = self.session_manager.get_session_summary(recent_episodes=5)
            if session_summary.get("episode_count", 0) > 0:
                lines.extend([
                    f"",
                    f"=== CROSS-EPISODE SESSION CONTEXT ===",
                    f"Episodes completed: {session_summary.get('episode_count', 0)}",
                    f"Success rate: {session_summary.get('success_rate', 'N/A')}",
                    f"Average reward: {session_summary.get('average_reward', 0):.3f}",
                ])
                
                patterns = session_summary.get("patterns", {})
                if patterns.get("successful_price_range"):
                    sr = patterns["successful_price_range"]
                    lines.append(
                        f"Prices that worked: ${sr['min']:.2f} - ${sr['max']:.2f} "
                        f"(mean: ${sr['mean']:.2f})"
                    )
                
                if patterns.get("rejected_price_range"):
                    jr = patterns["rejected_price_range"]
                    lines.append(
                        f"Prices that failed: ${jr['min']:.2f} - ${jr['max']:.2f} "
                        f"(mean: ${jr['mean']:.2f})"
                    )
                
                insights = session_summary.get("cross_episode_insights", [])
                if insights:
                    lines.append(f"Key insights:")
                    for insight in insights[:2]:  # Limit to top 2 insights
                        lines.append(f"  • {insight}")

        lines.extend([
            f"",
            f"Propose a price. Respond with JSON: {{\"price\": <number>}}",
        ])

        return "\n".join(lines)

    def _parse_response(self, content: str | None, obs: Dict[str, Any]) -> Dict[str, Any]:
        fallback = (obs["rider_quoted_price"] + obs["driver_quoted_price"]) / 2.0
        if not content:
            return {"type": "propose_price", "payload": {"price": round(fallback, 2)}}
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(content)
            price = float(parsed["price"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Fallback: midpoint
            price = (obs["rider_quoted_price"] + obs["driver_quoted_price"]) / 2.0

        return {"type": "propose_price", "payload": {"price": round(price, 2)}}


def openai_policy_factory(session_manager: Optional[Any] = None) -> OpenAIPolicy:
    return OpenAIPolicy(session_manager=session_manager)
