"""Simulator LLM agent — strategic accept/reject decisions for rider and driver.

The simulator agent wraps the Qwen2.5-0.5B model and produces per-step
decisions by reasoning about hidden thresholds, patience, and expected
surplus from bluffing vs accepting honestly.

Fallback: if parsing fails, falls back to the deterministic threshold check
(identical behaviour to the rule-based simulator.py) so training never
silently breaks due to a bad generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Tuple

from ride_hailing_env.models import HiddenState, Observation
from training.prompt_builders import build_simulator_prompt


@dataclass
class SimulatorDecision:
    rider_accept:  bool
    driver_accept: bool
    raw_output:    str   # kept for inspection and GRPO training samples


class SimulatorAgent:
    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int = 48) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def decide(
        self,
        hidden: HiddenState,
        obs: Observation,
        proposed_price: float,
    ) -> SimulatorDecision:
        """Return accept/reject decisions for rider and driver.

        The model sees the true hidden thresholds and reasons about whether
        bluffing (rejecting an acceptable price) is worth the patience cost.
        Falls back to honest threshold check on any parse failure.
        """
        import torch

        prompt = build_simulator_prompt(hidden, obs, proposed_price)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        rider_accept, driver_accept = self._parse(raw, hidden, proposed_price)
        return SimulatorDecision(
            rider_accept=rider_accept,
            driver_accept=driver_accept,
            raw_output=raw,
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(
        self,
        text: str,
        hidden: HiddenState,
        price: float,
    ) -> Tuple[bool, bool]:
        """Parse JSON from simulator output.

        Expected format: {"rider": "accept", "driver": "reject"}

        Falls back to deterministic threshold check on any failure so
        training never silently stalls due to a bad generation.
        """
        # Strip markdown code fences
        clean = text
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        # Try full JSON parse
        try:
            match = re.search(r"\{[^}]+\}", clean, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                rider  = str(parsed.get("rider",  "reject")).strip().lower() == "accept"
                driver = str(parsed.get("driver", "reject")).strip().lower() == "accept"
                return rider, driver
        except Exception:
            pass

        # Try keyword scan as last resort
        lower = text.lower()
        rider_accept  = "rider.*accept" in lower or (
            "rider" in lower and "accept" in lower and "reject" not in lower.split("rider")[1][:20]
        )
        driver_accept = "driver.*accept" in lower or (
            "driver" in lower and "accept" in lower and "reject" not in lower.split("driver")[1][:20]
        )

        if "rider" in lower or "driver" in lower:
            return rider_accept, driver_accept

        # Hard fallback: honest threshold check
        return (
            price <= hidden.rider_max_willingness,
            price >= hidden.driver_min_willingness,
        )
