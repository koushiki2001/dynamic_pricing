# Ride-Hailing Dynamic Pricing — OpenEnv Environment

Multi-step ride-hailing pricing negotiation environment where an AI agent plays the **platform**, proposing prices each round to a rider and driver who independently accept or reject.

## Problem

A ride-hailing platform must find a price acceptable to both rider and driver. The rider quotes a maximum they'd pay; the driver quotes a minimum they'd accept. The platform proposes prices in successive rounds. Both parties have hidden willingness thresholds and losing patience — too many bad proposals and they cancel.

**Real-world factors** affect hidden thresholds: weather, demand/supply, traffic, surge, time of day, distance.

## Observation Space

Each step the agent sees:

| Field | Type | Description |
|-------|------|-------------|
| `rider_quoted_price` | float | Rider's stated max (fixed per episode) |
| `driver_quoted_price` | float | Driver's stated min (fixed per episode) |
| `price_gap` | float | Absolute difference |
| `distance_km` | float | Trip distance |
| `demand_level` | int (0-2) | low / medium / high |
| `supply_level` | int (0-2) | low / medium / high |
| `surge_multiplier` | float | Current surge |
| `weather_condition` | int (0-2) | clear / rain / storm |
| `traffic_level` | int (0-2) | low / medium / heavy |
| `rider_patience` | float (0-1) | Decreases on rejection |
| `driver_patience` | float (0-1) | Decreases on rejection |
| `rider_mood` | string | willing / hesitant / frustrated |
| `driver_mood` | string | willing / hesitant / frustrated |
| `last_rider_response` | string | accepted / rejected / null |
| `last_driver_response` | string | accepted / rejected / null |
| `step_number` | int | Current round |

## Action Space

```json
{"type": "propose_price", "payload": {"price": 18.50}}
```

Single action type. Strategy emerges from which price to propose based on rejection feedback.

## Reward Design

- **Per-step**: +0.05 if one party accepts (partial progress); -0.05 if both reject
- **Terminal success**: `platform_profit × min(max_steps / steps_taken, 3.0)` — fewer steps = higher reward
- **Cancellation**: -5.0 penalty
- **Timeout**: -2.0 penalty

## Tasks

| Task | Difficulty | Gap | Patience | Overlap |
|------|-----------|-----|----------|---------|
| `easy` | Friendly Market | $3-5 | 8 steps | Wide |
| `medium` | Rush Hour | $8-12 | 5 steps | Narrow |
| `hard` | Storm Surge | $15-20 | 3 steps | Very narrow / none |

Each scored 0.0–1.0 by composite grader (completion rate, efficiency, profit, cancellation avoidance).

## Setup

```bash
pip install -r requirements.txt
```

## Train Agent

```bash
python scripts/train.py --task easy --episodes 3000
python scripts/train.py --task all --episodes 3000
```

## Evaluate Baselines

```bash
python baselines/evaluate_baselines.py
```

## OpenAI Inference

```bash
OPENAI_API_KEY=sk-... python scripts/inference.py
```

## Docker

```bash
docker build -t ride-hailing-pricing .
docker run ride-hailing-pricing
```

## OpenEnv Metadata

See `openenv.yaml` for full environment specification.
