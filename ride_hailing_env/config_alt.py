"""Alternate configuration: MAX_OFFER_PRICE = 1000.

Higher price cap enables long-distance premium rides, airport transfers,
surge-heavy scenarios, and luxury tiers.  Keep all structure identical to
config.py so the environment, simulator, and graders can be swapped
transparently.
"""

# Price bounds
MIN_OFFER_PRICE = 1.0
MAX_OFFER_PRICE = 1000.0

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

# --------------------------------------------------------------------------
# Task-specific configs — tuned for the 1000-cap price range.
# Longer distances, higher surges, bigger gaps, and premium fares.
# --------------------------------------------------------------------------
TASK_CONFIG = {
    "easy": {
        "max_steps": 8,
        "rider_patience_init_range": (0.9, 1.0),
        "driver_patience_init_range": (0.9, 1.0),
        "rider_patience_decay_range": (0.06, 0.12),
        "driver_patience_decay_range": (0.06, 0.12),
        "price_gap_range": (5.0, 20.0),
        "rider_slack_range": (15.0, 35.0),
        "driver_slack_range": (10.0, 25.0),
        "acceptance_noise_range": (1.0, 5.0),
        "distance_range": (5.0, 60.0),
        "duration_range": (10.0, 90.0),
        "eta_range": (1.0, 15.0),
        "demand_level_options": [0, 1, 1, 2],
        "supply_level_options": [0, 1, 1, 2],
        "surge_range": (1.0, 2.0),
        "weather_condition_options": [0, 0, 0, 1],
        "traffic_level_options": [0, 0, 1],
        "time_of_day_options": [0, 1, 1, 2],
        "day_type_options": [0, 0, 1],
        "commission_rate_range": (0.15, 0.25),
        "operational_cost_range": (2.0, 5.0),
        "min_overlap": 15.0,
        "num_eval_episodes": 120,
        "seed": 42,
    },
    "medium": {
        "max_steps": 5,
        "rider_patience_init_range": (0.8, 1.0),
        "driver_patience_init_range": (0.8, 1.0),
        "rider_patience_decay_range": (0.10, 0.20),
        "driver_patience_decay_range": (0.10, 0.20),
        "price_gap_range": (15.0, 50.0),
        "rider_slack_range": (15.0, 40.0),
        "driver_slack_range": (10.0, 30.0),
        "acceptance_noise_range": (3.0, 10.0),
        "distance_range": (10.0, 80.0),
        "duration_range": (15.0, 120.0),
        "eta_range": (2.0, 20.0),
        "demand_level_options": [0, 1, 2, 2],
        "supply_level_options": [0, 0, 1, 2],
        "surge_range": (1.2, 2.5),
        "weather_condition_options": [0, 1, 1, 2],
        "traffic_level_options": [0, 1, 1, 2],
        "time_of_day_options": [0, 1, 2, 3],
        "day_type_options": [0, 1, 2],
        "commission_rate_range": (0.15, 0.30),
        "operational_cost_range": (3.0, 8.0),
        "min_overlap": 8.0,
        "num_eval_episodes": 140,
        "seed": 84,
    },
    "hard": {
        "max_steps": 3,
        "rider_patience_init_range": (0.6, 1.0),
        "driver_patience_init_range": (0.6, 1.0),
        "rider_patience_decay_range": (0.20, 0.40),
        "driver_patience_decay_range": (0.20, 0.40),
        "price_gap_range": (30.0, 80.0),
        "rider_slack_range": (15.0, 40.0),
        "driver_slack_range": (10.0, 30.0),
        "acceptance_noise_range": (5.0, 15.0),
        "distance_range": (15.0, 100.0),
        "duration_range": (20.0, 150.0),
        "eta_range": (3.0, 30.0),
        "demand_level_options": [1, 2, 2, 2],
        "supply_level_options": [0, 0, 0, 1],
        "surge_range": (1.5, 3.5),
        "weather_condition_options": [0, 1, 2, 2],
        "traffic_level_options": [1, 2, 2, 2],
        "time_of_day_options": [0, 1, 2, 3],
        "day_type_options": [0, 1, 2],
        "commission_rate_range": (0.18, 0.35),
        "operational_cost_range": (4.0, 10.0),
        "min_overlap": 5.0,
        "num_eval_episodes": 160,
        "seed": 168,
    },
}
