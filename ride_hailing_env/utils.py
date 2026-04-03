"""Utility functions."""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def patience_to_mood(patience: float) -> str:
    if patience > 0.6:
        return "willing"
    elif patience > 0.3:
        return "hesitant"
    return "frustrated"


def normalize_revenue(value: float, low: float = -10.0, high: float = 30.0) -> float:
    return clamp((value - low) / (high - low), 0.0, 1.0)
