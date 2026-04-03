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
- [Official Grader](#official-grader)
- [Task Difficulties](#task-difficulties)
- [Policies (Agents)](#policies-agents)
  - [1. Midpoint Baseline](#1-midpoint-baseline)
  - [2. Adaptive Baseline](#2-adaptive-baseline)
  - [3. Base LLM Policy](#3-base-llm-policy)
  - [4. Reward-Guided LLM Policy](#4-reward-guided-llm-policy-main-contribution)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)

---

## The Problem

A rider wants to go somewhere. A driver can take them. The rider has a maximum price they'd pay; the driver has a minimum price they'd accept. These **true willingness thresholds are hidden** — the agent only sees their quoted prices (which are more conservative).

The platform must propose a price each round. Both parties independently accept or reject. If both accept, the ride is completed. If either party runs out of patience after repeated bad proposals, they **cancel** and the deal is lost. If the maximum number of negotiation rounds is exhausted, the episode **times out**.

Real-world context factors — weather, traffic, demand/supply levels, surge pricing, time of day — influence the hidden thresholds, adding noise and asymmetry to the negotiation.

---

## How a Negotiation Works

```
Episode Start
  │
  ├── Scenario generated (rider quote, driver quote, hidden thresholds, context)
  │
  ├── Round 1: Agent proposes $X
  │     ├── Rider: accept/reject (based on hidden willingness + random noise)
  │     ├── Driver: accept/reject (based on hidden willingness + random noise)
  │     └── If both accept → Ride Completed ✓
  │         If either rejects → patience decreases, move to next round
  │         If patience hits 0 → Cancelled ✗
  │
  ├── Round 2: Agent proposes $Y (informed by feedback)
  │     └── ...
  │
  └── Round N (max_steps): If still no agreement → Timed Out ✗
```

Key mechanics:
- **Patience decay**: Each rejection reduces that party's patience. Below 0 → cancellation.
- **Mood**: Derived from patience — "willing" (>0.6), "hesitant" (0.3–0.6), "frustrated" (<0.3).
- **Acceptance noise**: Even a "good" price can be randomly rejected, and a "bad" one can be accepted. This models real-world uncertainty.
- **Duplicate price guard**: The environment rejects re-proposing the same price without advancing the step counter, forcing the agent to explore.

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

**Crucially absent**: The rider's true maximum willingness and the driver's true minimum willingness. The agent must infer the acceptable price range from indirect signals.

---

## What the Agent Does (Action Space)

```json
{"type": "propose_price", "payload": {"price": 18.50}}
```

A single action type — propose a price. The entire strategy is about *which* price to propose given the current feedback and context.

---

## How Scenarios Are Generated

Each episode creates a fresh negotiation scenario:

1. **Base price** computed from trip fundamentals: `3.0 + 1.5 × distance + 0.3 × duration`
2. **Context adjustments** modify pricing: bad weather increases both quotes, high demand raises rider tolerance, low supply raises driver expectations, etc.
3. **Rider quote** (lower) = `base_price - gap/2 × (1 + adjustments)`
4. **Driver quote** (higher) = `base_price + gap/2 × (1 + adjustments)`
5. **Hidden willingness** extends beyond quotes by a "slack" amount:
   - Rider would actually pay up to `rider_quote + slack`
   - Driver would actually accept down to `driver_quote - slack`
6. **Asymmetric offset** (`±35%` of combined slack): Shifts the true acceptance zone away from the naive midpoint. This prevents "just split the difference" from always working.
7. **Overlap guarantee**: A minimum overlap between the rider's true max and driver's true min is enforced per task config to ensure the deal is theoretically closeable.

---

## Reward Design

The reward function balances three competing objectives: **close the deal**, **close it fast**, and **close it profitably**.

### Per-Step Shaping Rewards
| Outcome | Reward | Signal |
|---------|--------|--------|
| One party accepts, other rejects | **+0.05** | Getting warmer — price is acceptable to one side |
| Both reject | **−0.05** | Price is off for everyone |
| Both accept | 0.0 | Terminal reward takes over |

### Terminal Rewards

**Ride completed** (both accept):

```
platform_profit = price × commission_rate − operational_cost
efficiency_bonus = min(max_steps / steps_taken, 3.0)
reward = platform_profit × efficiency_bonus
```

The efficiency bonus rewards closing quickly — if an 8-step task is solved in 2 steps, that's a 3.0× multiplier (capped). This prevents agents from wasting rounds when they've already found a good price.

**Timed out** (ran out of steps): **−2.0**

**Cancelled** (patience hit zero): **−5.0** — penalized more harshly than timeout because it means the agent actively drove a party away with bad proposals.

---

## Official Grader

Each policy is evaluated over N episodes by a composite grader that scores performance on a 0.0–1.0 scale:

```
score = w_completion × completion_rate
      + w_efficiency × normalized_efficiency
      + w_profit     × normalized_profit
      + w_no_cancel  × (1 − cancellation_rate)
```

Where:
- `normalized_efficiency` = `min(avg_efficiency / 3.0, 1.0)`
- `normalized_profit` = `clamp((avg_profit − (−10)) / (30 − (−10)), 0, 1)` — maps profit from [−$10, $30] to [0, 1]

The weights shift by difficulty:

| Weight | Easy | Medium | Hard |
|--------|------|--------|------|
| Completion | 0.40 | 0.30 | 0.25 |
| Efficiency | 0.30 | 0.25 | 0.20 |
| Profit | 0.20 | 0.30 | **0.35** |
| No-cancel | 0.10 | 0.15 | **0.20** |

On easy tasks, just completing rides matters most. On hard tasks, maximizing profit and avoiding cancellations becomes dominant.

---

## Task Difficulties

| Parameter | Easy ("Friendly Market") | Medium ("Rush Hour") | Hard ("Storm Surge") |
|-----------|------|--------|------|
| Max steps | 8 | 5 | 4 |
| Price gap (rider−driver quotes) | $6–14 | $10–18 | $12–22 |
| Initial patience | 0.85–1.0 | 0.75–0.95 | 0.55–0.85 |
| Patience decay per rejection | 0.08–0.14 | 0.12–0.22 | 0.18–0.35 |
| Acceptance noise | $1.0–2.5 | $1.5–3.0 | $2.0–4.0 |
| Slack (hidden willingness beyond quote) | $2–5 | $1.5–4 | $1–4 |
| Surge multiplier | 1.0–1.3 | 1.1–2.0 | 1.5–3.0 |
| Target completion rate | 80% | 65% | 45% |
| Eval episodes | 120 | 140 | 160 |

Harder tasks have: fewer negotiation rounds, larger price gaps, lower patience, faster patience decay, more acceptance noise, and higher surge. The combination makes finding the acceptable price range much harder.

---

## Policies (Agents)

We implemented and compared four pricing policies, from simple heuristics to our main contribution.

### 1. Midpoint Baseline

**Strategy**: Always propose `(rider_quote + driver_quote) / 2`.

Simple and deterministic. Ignores all rejection feedback — proposes the same price every round. Works when the true acceptance zone happens to be centered, but fails when the asymmetric offset shifts it away from the midpoint.

### 2. Adaptive Baseline

**Strategy**: Binary-search-style narrowing based on rejection feedback.

- Maintains bounds `[low, high]` initialized to `[rider_quote, driver_quote]`
- If rider rejected → they want a lower price → lower the ceiling
- If driver rejected → they want a higher price → raise the floor
- Proposes the midpoint of the updated bounds each round

Learns from feedback and converges toward the acceptable range. Purely heuristic — no probabilistic reasoning or profit optimization.

### 3. Base LLM Policy

**Strategy**: Zero-shot LLM reasoning via Google Gemini Flash (through OpenRouter).

Builds a narrative prompt with the full scenario context (quotes, patience, moods, rejection history) and asks the LLM to propose a price. The LLM applies semantic understanding of negotiation dynamics but has **no awareness of the reward function** — it doesn't know how its proposals translate to scores.

### 4. Reward-Guided LLM Policy (Main Contribution)

**Strategy**: Combines adaptive bounds tracking, a learned reward proxy, experience replay, and LLM reasoning into a hybrid system.

This is the core innovation of the project. Here's how it works:

```
Step 1: Update Adaptive Bounds
  - Rider rejected at $X → ceiling ≤ X − $0.50
  - Driver rejected at $X → floor ≥ X + $0.50
  - Both rejected → tighten both sides by 35% toward midpoint

Step 2: Build Reward-Aware Prompt
  - Include the reward formula so the LLM knows efficiency bonuses exist
  - Show rejection feedback with directional hints ("price was too high/low")
  - Display the estimated feasible range (adaptive bounds)
  - Add top-3 similar past experiences with their outcomes

Step 3: LLM Proposes a Price

Step 4: Generate 7 Spread Candidates
  - Distribute across the feasible bounds at [5%, 20%, 35%, 50%, 65%, 80%, 95%] positions
  - Add an urgency-biased candidate if patience is critically low

Step 5: Score All Candidates via Reward Proxy
  - Estimates acceptance probability using sigmoid functions
  - Computes expected profit × efficiency bonus
  - Penalizes prices outside bounds, adds cancellation/timeout risk penalties
  - LLM's candidate gets a +0.5 trust bonus (overridden only if a heuristic candidate scores significantly better)

Step 6: Enforce Minimum Price Movement
  - At least $1 change from last proposal to prevent oscillation
  - Direction determined by which party rejected
```

**Key components:**
- **ExperienceStore**: Loads past (scenario → price → reward) data from JSON files. Finds top-k similar high-reward episodes via multi-dimensional similarity scoring to ground the LLM's reasoning in empirical examples.
- **Reward Proxy**: A fast heuristic function that estimates the expected reward for any candidate price using sigmoid-based acceptance probability modeling.
- **LLM Trust Bonus**: The LLM's pick gets a score advantage, so it's only overridden when a heuristic candidate is substantially better. This balances semantic reasoning with quantitative optimization.
- **Episode Logging**: Every decision (bounds, candidates, scores, final choice) is recorded for analysis and debugging.

---

## Benchmark Results

All 4 policies evaluated with the official grader (10 episodes per task, deterministic seeds):

| Policy | Easy | Medium | Hard | **Average** |
|--------|------|--------|------|-------------|
| Midpoint | 0.6891 | 0.7156 | 0.6341 | 0.6796 |
| Adaptive | 0.8675 | **0.7635** | **0.7398** | **0.7903** |
| Base LLM | 0.7381 | 0.5982 | 0.5410 | 0.6258 |
| **Reward-Guided LLM** | **0.8680** | 0.7561 | 0.6424 | 0.7555 |

**Per-task winners:**
- **Easy**: Reward-Guided LLM (0.8680) — 100% completion, 0% cancellation
- **Medium**: Adaptive (0.7635) — Reward-Guided LLM very close at 0.7561
- **Hard**: Adaptive (0.7398) — hard tasks' tight constraints (4 max steps, large gaps, high noise) favor fast deterministic convergence

**Key observations:**
- Reward-Guided LLM is the best LLM-based policy, substantially outperforming the base LLM across all difficulties
- On easy tasks, Reward-Guided LLM matches the Adaptive baseline's 100% completion rate while achieving slightly higher profit
- Base LLM struggles significantly — 60% cancellation on hard, 40% timeout on medium — showing that raw LLM reasoning without reward awareness is insufficient
- The Adaptive baseline's strength on hard tasks comes from its deterministic, fast convergence — with only 4 steps and high noise, the LLM's stochasticity becomes a disadvantage

---

## Project Structure

```
plan_b/
├── ride_hailing_env/              # Core environment package
│   ├── environment.py             # Main env: reset(), step(), observation management
│   ├── simulator.py               # Accept/reject logic, patience decay
│   ├── scenario_generator.py      # Procedural scenario generation with context factors
│   ├── models.py                  # Pydantic data models (Observation, Action, StepResult, etc.)
│   ├── reward.py                  # Step shaping + terminal reward computation
│   ├── config.py                  # Constants and task-specific difficulty configs
│   ├── utils.py                   # clamp(), patience_to_mood(), normalize_revenue()
│   └── tasks/
│       ├── graders.py             # Official composite grader (0.0–1.0 scoring)
│       ├── easy_task.py           # Easy task wrappers
│       ├── medium_task.py         # Medium task wrappers
│       └── hard_task.py           # Hard task wrappers
│
├── baselines/                     # Policy implementations
│   ├── midpoint_policy.py         # Static midpoint baseline
│   ├── adaptive_policy.py         # Binary-search adaptive baseline
│   ├── openai_policy.py           # Zero-shot LLM policy (Gemini Flash via OpenRouter)
│   ├── reward_guided_llm_policy.py # Reward-guided LLM policy (main contribution)
│   └── evaluate_baselines.py      # Baseline evaluation harness
│
├── scripts/                       # Evaluation and analysis scripts
│   ├── compare_all_policies.py    # Consolidated 4-policy comparison with official grader
│   ├── train.py                   # Training script
│   ├── inference.py               # Single-episode inference
│   ├── collect_experience.py      # Experience data collection for replay
│   └── ...                        # Additional analysis/debugging scripts
│
├── data/                          # Experience replay data and evaluation results
│   ├── experience_*.json          # Past episode data per difficulty
│   ├── results_*.json             # Evaluation result snapshots
│   └── scenarios_*.json           # Pre-generated scenario sets
│
├── tests/                         # Test suite
│   ├── test_environment.py        # Environment unit tests
│   └── test_graders.py            # Grader unit tests
│
├── openenv.yaml                   # OpenEnv environment specification
├── tasks.json                     # Task metadata (names, targets, descriptions)
├── Dockerfile                     # Container build (Python 3.12-slim)
├── pyproject.toml                 # Project metadata (Python ≥3.11)
└── requirements.txt               # Dependencies
```

---

## Setup & Usage

### Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: numpy, pydantic (v2), openai, python-dotenv, pytest

### Run the Full Policy Comparison

```bash
python scripts/compare_all_policies.py --episodes 10
```

Options:
- `--episodes N` — number of episodes per task (default: 10)
- `--task {easy|medium|hard|all}` — which difficulty to evaluate (default: all)
- `--no-llm` — skip LLM policies (only run Midpoint and Adaptive)
- `--seed N` — base random seed

### Evaluate Individual Baselines

```bash
python baselines/evaluate_baselines.py
```

### Run a Single Inference Episode

```bash
python scripts/inference.py
```

### Train an Agent

```bash
python scripts/train.py --task easy --episodes 3000
python scripts/train.py --task all --episodes 3000
```

### Docker

```bash
docker build -t ride-hailing-pricing .
docker run ride-hailing-pricing
```

### Run Tests

```bash
pytest tests/
```

---

## OpenEnv Metadata

See `openenv.yaml` for the full environment specification including observation/action schemas and performance bounds.
