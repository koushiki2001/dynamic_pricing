"""Adaptive policy: binary-search style adjustment based on who rejected."""

from typing import Any, Dict


class AdaptivePolicy:
    def __init__(self):
        self._low = None
        self._high = None

    def reset(self):
        self._low = None
        self._high = None

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        rider_q = obs["rider_quoted_price"]
        driver_q = obs["driver_quoted_price"]

        # Initialize bounds on first step
        if self._low is None:
            self._low = rider_q
            self._high = driver_q
            price = (rider_q + driver_q) / 2.0
            return {"type": "propose_price", "payload": {"price": round(price, 2)}}

        last_rider = obs.get("last_rider_response")
        last_driver = obs.get("last_driver_response")
        last_price = obs.get("last_proposed_price", (self._low + self._high) / 2.0)

        # Adjust bounds based on feedback
        if last_rider == "rejected" and last_driver == "accepted":
            # Price too high for rider → search lower
            self._high = last_price
        elif last_rider == "accepted" and last_driver == "rejected":
            # Price too low for driver → search higher
            self._low = last_price
        elif last_rider == "rejected" and last_driver == "rejected":
            # Both rejected — reset to midpoint of original quotes
            self._low = rider_q
            self._high = driver_q

        price = (self._low + self._high) / 2.0
        return {"type": "propose_price", "payload": {"price": round(price, 2)}}


def adaptive_policy_factory():
    """Return a stateful adaptive policy instance."""
    return AdaptivePolicy()
