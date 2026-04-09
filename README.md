---
title: Ride-Hailing Dynamic Pricing
emoji: 🚕
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---
## Dynamic Pricing OpenEnv — Hackathon Submission

This repository implements a multi-step ride-hailing dynamic pricing environment and LLM-based agent for the OpenEnv Hackathon.

### Features
- **OpenEnv-compliant environment**: Negotiation between platform, rider, and driver with multi-step price proposals.
- **LLM Policy**: Uses OpenAI-compatible API (e.g., HuggingFace router) for price proposals via `baselines/openai_policy.py`.
- **Structured Logging**: Inference logs `[START]`, `[STEP]`, `[END]` for each episode, as required by the hackathon.
- **Duplicate Price Handling**: Inference nudges price if the environment rejects a duplicate proposal.
- **Strict Score Range**: All task scores are strictly between 0 and 1 (never exactly 0 or 1).
- **Configurable via .env**: Set `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` in `.env` or environment.
- **Docker-ready**: Includes Dockerfile for HF Spaces deployment.
- **Validation Script**: `scripts/validate-submission.sh` checks API, Docker build, and OpenEnv compliance.

### Key Files
- `inference.py` — Root inference script (required by HF validator)
- `app.py` — FastAPI server for deployment
- `baselines/openai_policy.py` — LLM-based negotiation policy
- `openenv.yaml` — Environment schema
- `requirements.txt` — All dependencies (including `openenv-core>=0.2.0`)
- `scripts/validate-submission.sh` — Submission validation script

### Usage
1. Set up your `.env` file with:
   - `API_BASE_URL=https://router.huggingface.co/v1`
   - `MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct`
   - `HF_TOKEN=...` (your HuggingFace token)
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r server/requirements.txt
   ```
3. Run inference:
   ```bash
   python inference.py
   ```
4. Validate submission:
   ```bash
   ./scripts/validate-submission.sh <your-hf-space-url> dynamic_pricing
   ```

### Submission Compliance
- Passes all hackathon requirements for:
  - Root `inference.py` with structured logs
  - Required env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
  - Uses OpenAI-compatible client
  - `openenv-core>=0.2.0` in requirements
  - Dockerfile and openenv.yaml present
  - All scores strictly between 0 and 1

---
For any issues, see the code or contact the maintainers.
# Ride-Hailing Dynamic Pricing — OpenEnv Environment

A multi-step negotiation environment where an AI agent acts as a **ride-hailing platform**, proposing prices each round to a rider and a driver who independently accept or reject. The agent must find a price both parties agree on — quickly, profitably, and without anyone walking away.

---

## Table of Contents

- [The Problem](#the-problem)
- [How a Negotiation Works](#how-a-negotiation-works)
- [What the Agent Sees (Observation Space)](#what-the-agent-sees-observation-space)
- [What the Agent Does (Action Space)](#what-the-agent-does-action-space)
- [How Scenarios Are Generated](#how-scenarios-are-generated)
- [Reward Design](#reward-design)
- [Grading & Pass/Fail Criteria](#grading--passfail-criteria)
- [Task Difficulties](#task-difficulties)
- [Policies (Agents)](#policies-agents)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## The Problem

A rider wants to go somewhere. A driver can take them. The rider has a maximum price they would pay; the driver has a minimum price they would accept. These **true willingness thresholds are hidden** — the agent only sees their quoted prices (which are more conservative).

The platform must propose a price each round. Both parties independently accept or reject based on their hidden thresholds. If either party runs out of patience after repeated bad proposals, they **cancel** and the deal is lost. If the maximum number of negotiation rounds is exhausted, the episode **times out**.

Real-world context factors — weather, traffic, demand/supply levels, surge pricing, time of day — influence the hidden thresholds, adding asymmetry to the negotiation.

---

## How a Negotiation Works

```
Episode Start
  │
  ├── Scenario generated (rider quote, driver quote, hidden thresholds, context)
  │
  ├── Round 1: Agent proposes $X
  │     ├── Rider: accept if price ≤ rider_max_willingness
  │     ├── Driver: accept if price ≥ driver_min_willingness
  │     └── Both accept → Ride Completed ✓
  │         Either rejects → patience decreases, move to next round
  │         Patience hits 0 → Cancelled ✗
  │
  ├── Round 2: Agent proposes $Y (informed by previous feedback)
  │     └── ...
  │
  └── Round N (max_steps): No agreement yet → Timed Out ✗
```

Key mechanics:
- **Deterministic acceptance**: Thresholds (`rider_max_willingness`, `driver_min_willingness`) are fixed at scenario generation time — no per-step noise. The same price always produces the same accept/reject outcome within an episode.
- **Patience decay**: Each rejection reduces that party's patience. At 0 → cancellation.
- **Mood**: Derived from patience — "willing" (>0.6), "hesitant" (0.3–0.6), "frustrated" (<0.3).
- **Duplicate price guard**: Re-proposing the same price is rejected without advancing the step counter, forcing the agent to explore.

---

## What the Agent Sees (Observation Space)

Each round, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `rider_quoted_price` | float | Rider's stated maximum (conservative — true willingness is higher) |
| `driver_quoted_price` | float | Driver's stated minimum (conservative — true willingness is lower) |
| `price_gap` | float | Absolute difference between quotes |
| `distance_km` | float | Trip distance |
| `estimated_duration_min` | float | Estimated trip duration |
| `pickup_eta_min` | float | Estimated pickup time |
| `demand_level` | int (0–2) | Low / Medium / High |
| `supply_level` | int (0–2) | Low / Medium / High |
| `surge_multiplier` | float (1.0–3.0) | Current surge pricing factor |
| `weather_condition` | int (0–2) | Clear / Rain / Storm |
| `traffic_level` | int (0–2) | Low / Medium / Heavy |
| `time_of_day` | int (0–3) | Morning / Afternoon / Evening / Night |
| `day_type` | int (0–2) | Weekday / Weekend / Holiday |
| `commission_rate` | float | Platform's commission percentage |
| `operational_cost` | float | Platform's per-ride cost |
| `max_steps` | int | Maximum negotiation rounds |
| `step_number` | int | Current round number |
| `rider_patience` | float (0–1) | Rider's remaining patience |
| `driver_patience` | float (0–1) | Driver's remaining patience |
| `rider_mood` | string | "willing" / "hesitant" / "frustrated" |
| `driver_mood` | string | "willing" / "hesitant" / "frustrated" |
| `last_rider_response` | string | "accepted" / "rejected" / null |
| `last_driver_response` | string | "accepted" / "rejected" / null |
| `last_proposed_price` | float | Previous price proposal (null on first step) |

**Crucially absent**: The rider's true maximum willingness and the driver's true minimum willingness. The agent must infer the acceptable price range from indirect signals and rejection feedback.

---

## What the Agent Does (Action Space)

```json
{"type": "propose_price", "payload": {"price": 18.50}}
```

A single action type — propose a price. The entire strategy is about *which* price to propose given the current feedback and context.

---

## How Scenarios Are Generated

Each episode creates a fresh negotiation scenario via `ScenarioGenerator`:

1. **Base price** computed from trip fundamentals: `3.0 + 1.5 × distance + 0.3 × duration`
2. **Context adjustments** modify pricing: bad weather increases both quotes, high demand raises rider tolerance, low supply raises driver expectations, etc.
3. **Rider quote** (lower) and **driver quote** (higher) are derived from the base price with a task-specific gap.
4. **Hidden willingness** extends beyond quotes by a slack amount:
   - Rider's true max = `rider_quote + slack`
   - Driver's true min = `driver_quote - slack`
5. **Noise baked in at generation time**: A `rider_noise_bound` is sampled once per scenario from the task's `acceptance_noise_range`. This is added to the hidden thresholds *once*, making acceptance fully deterministic within an episode while preserving scenario-to-scenario variability. This models real-world unpredictability (e.g., a rider in a rush willing to pay more than usual) without introducing per-step randomness that would make grading stochastic.
6. **Per-scenario thresholds** computed and stored in `HiddenState`:
   - `reward_threshold` = fraction of the maximum possible profit for that scenario
   - `penalty_threshold` = fraction of the `rider_noise_bound` (more unpredictable rider → more lenient penalty threshold)

---

## Reward Design

The reward function balances three competing objectives: **close the deal**, **close it fast**, and **close it profitably**.

### Per-Step Shaping Rewards

| Outcome | Reward | Signal |
|---------|--------|--------|
| One party accepts, other rejects | **+0.05** | Price is acceptable to one side |
| Both reject | **−0.05** | Price is off for everyone |
| Both accept | 0.0 | Terminal reward takes over |

### Terminal Rewards

**Ride completed** (both accept):

```
platform_profit        = price × commission_rate − operational_cost
efficiency_bonus       = min(max_steps / steps_taken, 3.0)
missed_revenue_penalty = max(0, rider_max_willingness − price) × commission_rate
reward = platform_profit × efficiency_bonus − missed_revenue_penalty
```

The `missed_revenue_penalty` penalizes leaving money on the table — if the rider would have accepted a higher price, the platform lost that revenue by pricing too low. This incentivizes the agent to price closer to the rider's true willingness, not just at the minimum acceptable level.

The `efficiency_bonus` rewards closing quickly — an 8-step task solved in 2 steps earns a 3.0× multiplier (capped).

**Timed out**: **−2.0**

**Cancelled**: **−5.0** — penalized more harshly because it means the agent actively drove a party away.

---

## Grading & Pass/Fail Criteria

Each episode is independently evaluated as **pass or fail** based on three conditions:

```
episode_passed = (
    ride_completed == True
    AND episode_reward > reward_threshold
    AND missed_revenue_penalty < penalty_threshold
)

score = total_passed / num_episodes   # ∈ [0.0, 1.0]
```

Both thresholds are **scenario-specific** — computed from the `HiddenState` at generation time:

- `reward_threshold`: A minimum profit bar scaled to what was achievable in that scenario. Easy tasks require 30% of max possible profit; hard tasks require 70%.
- `penalty_threshold`: A cap on how much revenue the agent is allowed to leave on the table, scaled by how unpredictable the rider was (higher `rider_noise_bound` → more tolerance).

This means completing the ride is necessary but not sufficient — the agent must also price well (high profit, not too low).

### Threshold Fractions by Task

| Task | Reward Threshold | Penalty Threshold |
|------|-----------------|-------------------|
| Easy | 30% of max profit | 100% of noise bound |
| Medium | 50% of max profit | 75% of noise bound |
| Hard | 70% of max profit | 50% of noise bound |

---

## Task Difficulties

| Parameter | Easy ("Friendly Market") | Medium ("Rush Hour") | Hard ("Storm Surge") |
|-----------|--------------------------|----------------------|----------------------|
| Objective | Complete the ride | Complete within 5 steps with profit > 0 | Complete within 4 steps with profit > 1.0 |
| Max steps | 8 | 5 | 4 |
| Price gap | $6–14 | $10–18 | $12–22 |
| Initial patience | 0.85–1.0 | 0.75–0.95 | 0.55–0.85 |
| Patience decay / rejection | 0.08–0.14 | 0.12–0.22 | 0.18–0.35 |
| Acceptance noise range | $1.0–2.5 | $1.5–3.0 | $2.0–4.0 |
| Surge multiplier | 1.0–1.3 | 1.1–2.0 | 1.5–3.0 |
| Reward threshold fraction | 30% | 50% | 70% |
| Penalty threshold fraction | 100% | 75% | 50% |

Harder tasks have: fewer negotiation rounds, larger price gaps, lower patience, faster patience decay, more noise in thresholds, and a stricter profit bar to pass.

---

## Policies (Agents)

### LLM Policy (Submission Agent)

**File**: `baselines/openai_policy.py`

**Model**: `google/gemini-2.0-flash-001` (via OpenRouter / HuggingFace Router)

The primary submission agent. Each step, it receives the full observation as a structured prompt and calls the LLM to reason about what price to propose. The prompt includes:
- Current scenario context (quotes, gap, weather, demand/supply, patience, moods)
- Full rejection history so far
- Instructions to balance closing speed with profit

The LLM returns a price, which is parsed and submitted as the action.

### Q-Learning Agent (Training Baseline)

**File**: `scripts/train.py`

A tabular Q-learning agent that learns a price-selection policy over discrete state buckets — **no LLM involved**. It discretizes the observation into a 7-dimensional state tuple (gap bin, step, rider/driver patience bins, response history, context difficulty, demand-supply imbalance) and learns Q-values over 11 evenly-spaced candidate prices.

Useful for:
- Benchmarking reward function quality without LLM variance
- Fast iteration on environment and reward design
- Verifying that the environment is learnable at all

### Midpoint Baseline

**File**: `baselines/midpoint_policy.py`

Always proposes `(rider_quote + driver_quote) / 2`. Deterministic, no learning. Fails when the true acceptance zone is asymmetrically offset from the midpoint.

### Adaptive Baseline

**File**: `baselines/adaptive_policy.py`

Binary-search-style narrowing using rejection feedback. Maintains bounds `[low, high]` and updates them based on who rejected. Converges reliably but ignores profit optimization entirely.

---

## Project Structure

```
dynamic_pricing/
├── ride_hailing_env/              # Core environment package
│   ├── environment.py             # Main env: reset(), step(), episode management
│   ├── simulator.py               # Deterministic accept/reject logic, patience decay
│   ├── scenario_generator.py      # Procedural scenario generation with baked-in noise
│   ├── models.py                  # Pydantic models (Observation, HiddenState, StepResult)
│   ├── reward.py                  # Step shaping + terminal reward with missed_revenue_penalty
│   ├── config.py                  # Task configs (thresholds, noise ranges, difficulty params)
│   ├── utils.py                   # clamp(), patience_to_mood()
│   └── tasks/
│       ├── graders.py             # Pass/fail grader — score = total_passed / num_episodes
│       ├── easy_task.py           # Easy task definition and objective
│       ├── medium_task.py         # Medium task definition and objective
│       └── hard_task.py           # Hard task definition and objective
│
├── baselines/                     # Policy implementations
│   ├── openai_policy.py           # LLM policy — submission entry point
│   ├── adaptive_policy.py         # Binary-search adaptive baseline
│   ├── midpoint_policy.py         # Static midpoint baseline
│   ├── reward_guided_llm_policy.py # Reward-guided LLM policy (experimental)
│   └── session_manager.py         # Cross-episode LLM memory management
│
├── scripts/
│   ├── train.py                   # Q-learning agent training + tabular evaluation summary
│   ├── dump_scenarios.py          # Pre-generate train/eval scenario snapshots to JSON
│   └── compare_all_policies.py    # Side-by-side policy comparison
│
├── data/
│   ├── scenarios_{task}_train.json  # 5000 pre-generated training scenarios per task
│   ├── scenarios_{task}_eval.json   # 1000 held-out eval scenarios per task (unseen)
│   └── experience_{task}.json       # Past episode data for experience replay
│
├── tests/
│   ├── test_environment.py
│   └── test_graders.py
│
├── inference.py                   # Hackathon submission entry point (LLM policy)
├── openenv.yaml                   # OpenEnv environment specification
├── tasks.json                     # Task metadata
├── Dockerfile                     # Container build
└── requirements.txt               # Dependencies
```

---

## Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: numpy, pydantic (v2), openai, python-dotenv, pytest

### Environment Variables

```bash
export API_BASE_URL=https://openrouter.ai/api/v1
export MODEL_NAME=google/gemini-2.0-flash-001
export HF_TOKEN=<your-api-key>
```

### Run Inference (LLM Agent)

```bash
python inference.py
```

Runs the LLM policy across all three tasks (20 episodes each) and prints a tabular summary:

```
Task        Score   Pass%  Complete%  Cancel%  Timeout%  AvgProfit  AvgPenalty
easy       0.7500  75.0%      85.0%    10.0%      5.0%     $12.34      $0.2100
medium     0.5500  55.0%      70.0%    20.0%     10.0%      $9.87      $0.4300
hard       0.3000  30.0%      50.0%    35.0%     15.0%      $7.12      $0.6700
```

### Train the Q-Learning Agent

```bash
# Train on a single task
python scripts/train.py --task easy --episodes 3000

# Train on all tasks
python scripts/train.py --task all --episodes 3000
```

### Pre-generate Scenario Snapshots

```bash
# Generate 5000 train + 1000 eval scenarios for all tasks
python scripts/dump_scenarios.py

# Generate for a specific task
python scripts/dump_scenarios.py --task medium --train-episodes 5000 --eval-episodes 1000
```

Train and eval scenarios use different seed ranges (`seed=42` vs `seed=99999`) to guarantee eval scenarios are unseen during training.

### Compare All Policies

```bash
python scripts/compare_all_policies.py --episodes 10
```

### Run Tests

```bash
pytest tests/
```

### Docker

```bash
docker build -t ride-hailing-pricing .
docker run -e HF_TOKEN=<key> -e API_BASE_URL=<url> -e MODEL_NAME=<model> ride-hailing-pricing
```

---

## OpenEnv Metadata

See `openenv.yaml` for the full environment specification including observation/action schemas and performance bounds.
