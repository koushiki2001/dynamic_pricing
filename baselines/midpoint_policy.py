"""Midpoint policy: always propose (rider_quoted + driver_quoted) / 2."""

from typing import Any, Dict


def midpoint_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    price = (obs["rider_quoted_price"] + obs["driver_quoted_price"]) / 2.0
    return {"type": "propose_price", "payload": {"price": round(price, 2)}}
