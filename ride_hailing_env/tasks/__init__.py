"""Task definitions and registry."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..config import TASK_CONFIG

TASK_DEFINITIONS: List[Dict[str, object]] = [
    {
        "id": "easy",
        "name": "Friendly Market",
        "difficulty": "easy",
        "description": "Balanced scenarios with small price gaps, high patience, wide overlap. Tests basic price discovery.",
        "max_steps": TASK_CONFIG["easy"]["max_steps"],
        "num_eval_episodes": TASK_CONFIG["easy"]["num_eval_episodes"],
        "seed": TASK_CONFIG["easy"]["seed"],
    },
    {
        "id": "medium",
        "name": "Rush Hour Negotiation",
        "difficulty": "medium",
        "description": "Rain, high demand, moderate gaps. Narrow overlap zone, misleading surge signal.",
        "max_steps": TASK_CONFIG["medium"]["max_steps"],
        "num_eval_episodes": TASK_CONFIG["medium"]["num_eval_episodes"],
        "seed": TASK_CONFIG["medium"]["seed"],
    },
    {
        "id": "hard",
        "name": "Storm Surge Standoff",
        "difficulty": "hard",
        "description": "Storm, extreme conditions, large gaps, very narrow or no overlap. Tests graceful failure.",
        "max_steps": TASK_CONFIG["hard"]["max_steps"],
        "num_eval_episodes": TASK_CONFIG["hard"]["num_eval_episodes"],
        "seed": TASK_CONFIG["hard"]["seed"],
    },
]

TASK_IDS = [t["id"] for t in TASK_DEFINITIONS]


def get_task(task_id: str) -> Optional[Dict[str, object]]:
    return next((t for t in TASK_DEFINITIONS if t["id"] == task_id), None)
