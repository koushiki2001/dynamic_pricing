"""LLM-based multi-step negotiation policy via OpenRouter."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


class OpenAIPolicy:
    def __init__(self):
        api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env or environment")
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
        self._history: List[Dict[str, Any]] = []

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
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=100,
                )
                content = response.choices[0].message.content
                if content:
                    return self._parse_response(content, obs)
            except Exception:
                pass
            time.sleep(1.0 * (attempt + 1))

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


def openai_policy_factory() -> OpenAIPolicy:
    return OpenAIPolicy()
