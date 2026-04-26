"""OpenEnv client for the Dynamic Pricing environment.

Connects to a running environment server (local or HuggingFace Space) and
exposes the same reset() / step() / state() interface as the local
DynamicPricingEnv class — so training code can switch between local and
remote execution by changing one URL.

Usage:
    # Local server
    client = DynamicPricingClient()
    obs = client.reset(task_name="easy")
    result = client.step({"type": "propose_price", "payload": {"price": 15.5}})

    # HuggingFace Space
    client = DynamicPricingClient(base_url="https://<org>-<space>.hf.space")

    # From environment variable (set SPACE_URL or API_BASE_URL)
    client = DynamicPricingClient()
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _default_base_url() -> str:
    """Precedence: SPACE_URL → API_BASE_URL → localhost."""
    return os.getenv("SPACE_URL") or os.getenv("API_BASE_URL", "http://localhost:7860")


class DynamicPricingClient:
    """HTTP client for the Dynamic Pricing environment server.

    Wraps the FastAPI endpoints as a clean Python interface that mirrors the
    local DynamicPricingEnv API so training scripts can use either without
    changing their rollout logic.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 30) -> None:
        self.base_url = (base_url or _default_base_url()).rstrip("/")
        self.timeout = timeout
        self._episode_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "easy", seed: Optional[int] = None) -> Dict[str, Any]:
        """Start a new episode. Returns the initial observation dict."""
        params: Dict[str, Any] = {"task_name": task_name}
        if seed is not None:
            params["seed"] = seed
        resp = requests.post(f"{self.base_url}/reset", params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        self._episode_id = data.get("episode_id")
        return data["observation"]

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a price. Returns {observation, reward, done, info}.

        action format: {"type": "propose_price", "payload": {"price": 15.5}}
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json=action,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Return the current observation without advancing the episode."""
        resp = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def schema(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/schema", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def propose_price(self, price: float) -> Dict[str, Any]:
        """Shorthand for the only supported action type."""
        return self.step({"type": "propose_price", "payload": {"price": round(price, 2)}})

    @property
    def episode_id(self) -> Optional[str]:
        return self._episode_id

    def __repr__(self) -> str:
        return f"DynamicPricingClient(base_url={self.base_url!r})"
