"""Constants and configuration for the ride-hailing pricing environment."""

# Price bounds
MIN_OFFER_PRICE = 1.0
MAX_OFFER_PRICE = 500.0

# Commission
DEFAULT_COMMISSION_RATE = 0.20
DEFAULT_OPERATIONAL_COST = 2.0

# Efficiency bonus cap
MAX_EFFICIENCY_BONUS = 3.0

# Terminal rewards / penalties
CANCELLATION_PENALTY = -5.0
TIMEOUT_PENALTY = -2.0

# Per-step shaping rewards
PARTIAL_ACCEPT_REWARD = 0.05
DOUBLE_REJECT_PENALTY = -0.05

# Mood thresholds (derived from patience)
MOOD_WILLING_THRESHOLD = 0.6
MOOD_HESITANT_THRESHOLD = 0.3
# Below 0.3 = "frustrated"

# Task-specific configuration
# Tuned for REALISTIC ride-hailing negotiation:
#   - Easy:   decent overlap but off-center, patient parties → ~80-90% closeable, 2-3 steps avg
#   - Medium: narrow overlap, moderate patience, fewer steps → ~55-70%, 2-4 steps avg
#   - Hard:   very tight overlap, impatient, few steps        → ~30-50%, 2-3 steps avg
# Difficulty = narrow acceptance zone + asymmetric offset + impatience + noise
TASK_CONFIG = {
    "easy": {
        "max_steps": 8,
        "rider_patience_init_range": (0.85, 1.0),
        "driver_patience_init_range": (0.85, 1.0),
        "rider_patience_decay_range": (0.08, 0.14),
        "driver_patience_decay_range": (0.08, 0.14),
        "price_gap_range": (6.0, 14.0),
        "rider_slack_range": (2.0, 5.0),        # rider willing to stretch 2-5 above quote
        "driver_slack_range": (2.0, 5.0),        # driver willing to drop 2-5 below quote
        "acceptance_noise_range": (1.0, 2.5),    # moderate noise — can miss even good prices
        "distance_range": (2.0, 15.0),
        "duration_range": (5.0, 35.0),
        "eta_range": (1.0, 10.0),
        "demand_level_options": [0, 1, 1, 2],
        "supply_level_options": [0, 1, 1, 2],
        "surge_range": (1.0, 1.3),
        "weather_condition_options": [0, 0, 0, 1],
        "traffic_level_options": [0, 0, 1],
        "time_of_day_options": [0, 1, 1, 2],
        "day_type_options": [0, 0, 1],
        "commission_rate_range": (0.15, 0.25),
        "operational_cost_range": (1.5, 2.5),
        "min_overlap": 1.0,
        "num_eval_episodes": 120,
        "seed": 42,
    },
    "medium": {
        "max_steps": 5,
        "rider_patience_init_range": (0.75, 0.95),
        "driver_patience_init_range": (0.75, 0.95),
        "rider_patience_decay_range": (0.12, 0.22),
        "driver_patience_decay_range": (0.12, 0.22),
        "price_gap_range": (10.0, 18.0),
        "rider_slack_range": (1.5, 4.0),
        "driver_slack_range": (1.5, 4.0),
        "acceptance_noise_range": (1.5, 3.0),    # significant noise
        "distance_range": (3.0, 25.0),
        "duration_range": (8.0, 50.0),
        "eta_range": (2.0, 15.0),
        "demand_level_options": [0, 1, 2, 2],
        "supply_level_options": [0, 0, 1, 2],
        "surge_range": (1.1, 2.0),
        "weather_condition_options": [0, 1, 1, 2],
        "traffic_level_options": [0, 1, 1, 2],
        "time_of_day_options": [0, 1, 2, 3],
        "day_type_options": [0, 1, 2],
        "commission_rate_range": (0.15, 0.30),
        "operational_cost_range": (1.5, 3.0),
        "min_overlap": 0.5,
        "num_eval_episodes": 140,
        "seed": 84,
    },
    "hard": {
        "max_steps": 4,
        "rider_patience_init_range": (0.55, 0.85),
        "driver_patience_init_range": (0.55, 0.85),
        "rider_patience_decay_range": (0.18, 0.35),
        "driver_patience_decay_range": (0.18, 0.35),
        "price_gap_range": (12.0, 22.0),
        "rider_slack_range": (1.0, 4.0),
        "driver_slack_range": (1.0, 4.0),
        "acceptance_noise_range": (2.0, 4.0),     # very noisy — outcomes are volatile
        "distance_range": (5.0, 35.0),
        "duration_range": (10.0, 70.0),
        "eta_range": (3.0, 25.0),
        "demand_level_options": [1, 2, 2, 2],
        "supply_level_options": [0, 0, 0, 1],
        "surge_range": (1.5, 3.0),
        "weather_condition_options": [0, 1, 2, 2],
        "traffic_level_options": [1, 2, 2, 2],
        "time_of_day_options": [0, 1, 2, 3],
        "day_type_options": [0, 1, 2],
        "commission_rate_range": (0.18, 0.35),
        "operational_cost_range": (2.0, 4.0),
        "min_overlap": 0.5,
        "num_eval_episodes": 160,
        "seed": 168,
    },
}
