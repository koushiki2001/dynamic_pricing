# Behavioral Signals — Real-World Deployment Mapping

## The Core Problem

In training, behavioral signals are explicit and precise:
```
rider_patience = 0.62
rider_mood     = "hesitant"
last_rider_response = "rejected"
```

In the real world, **none of these fields exist**. Uber/Rapido do not negotiate — they quote once. There is no `rider_patience` database column. These signals must be **inferred from observable proxies**.

---

## Signal Mapping: Simulation → Real World

### Patience → Session Behavior Signals

| Patience Range | Simulation Meaning | Real-World Proxy |
|---|---|---|
| 0.8–1.0 (high) | Plenty of rejections before cancel | First app open, browsing multiple ride types, non-urgent destination |
| 0.5–0.7 (moderate) | 2–3 rejections left | Repeated price checks, mild urgency signals |
| 0.3–0.5 (hesitant) | 1 rejection before frustrated | 3+ app opens in 8 minutes, previously viewed price and exited |
| < 0.3 (frustrated) | Near cancellation | Multiple reopens in < 5 min, previously cancelled a ride today |

### Mood → Derived from Patience Proxy + History

| Mood | Simulation Source | Real-World Equivalent |
|---|---|---|
| "willing" | patience > 0.6 | First session, casual context, low historical cancellation rate |
| "hesitant" | 0.3 < patience ≤ 0.6 | Multiple app opens, previously viewed price and left, checking competitor |
| "frustrated" | patience ≤ 0.3 | 3rd+ reopen within 5 minutes, cancelled a ride today |

### Hidden Threshold → Historical + Contextual Estimation

The rider's `max_willingness` and driver's `min_willingness` are never directly observable. They are estimated as a **price acceptance probability curve**:

**Rider threshold estimate from:**
- `avg_price_paid_similar_route_last_90d`
- `max_price_ever_accepted_on_this_route`
- `cancellation_rate_when_surge > 1.5x`
- Current surge multiplier (shifts ceiling up proportionally)
- Context: rain + late night + airport → ceiling shifts higher (fewer alternatives)
- Account tier (Uber One subscribers are less price-sensitive)

**Driver threshold estimate from:**
- Average driver acceptance rate in this zone × time of day
- Last accepted trip fare in this zone
- Time since last completed ride (longer idle = lower floor)
- Distance to pickup (farther = needs more incentive)
- Zone demand level (high demand = can afford to wait = higher floor)

---

## The Feature Assembly Pipeline

```
User opens app + enters destination
          │
          ▼
┌─────────────────────────────────────┐
│      REAL-TIME FEATURE LAYER        │
│                                     │
│  Computed live per session:         │
│  - app_open_count_last_10min        │
│  - time_since_first_open_session    │
│  - destination_entered_time_ms      │
│  - ride_type_switches count         │
│  - screen_dwell_on_price_ms         │
│  - pickup_location_type             │
│    (airport / hospital / home /     │
│     railway / mall / office)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    HISTORICAL FEATURE LAYER         │
│                                     │
│  Precomputed per user (refresh 24h):│
│  - price_sensitivity_score          │
│  - avg_accepted_price_route_bucket  │
│  - surge_cancellation_threshold     │
│  - booking_urgency_pattern          │
│    (last-minute vs planned)         │
│  - ride_frequency_score             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     MARKET FEATURE LAYER            │
│                                     │
│  (Already exists in Uber/Rapido):   │
│  - surge_multiplier                 │
│  - demand_level, supply_level       │
│  - weather_condition                │
│  - traffic_level                    │
│  - time_of_day, day_type            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│       FEATURE ASSEMBLY              │
│                                     │
│  Maps raw signals → simulation-     │
│  space features platform LLM knows  │
│                                     │
│  patience_estimate  = f(session)    │
│  mood_estimate      = f(patience,   │
│                         history)    │
│  threshold_estimate = f(hist,ctx)   │
│  urgency_score      = f(loc, time)  │
└──────────────┬──────────────────────┘
               │
               ▼
         Platform LLM
         → outputs optimal price
```

---

## The Calibration Loop — How Signals Improve Over Time

Every accept / abandon event is a calibration data point.

```
Week 1:
  urgency_score = 0.80 (airport + late + 4 app opens)
  Platform proposes: $24
  Outcome: ACCEPTED quickly

Week 2:
  Similar profile, urgency_score = 0.75
  Platform proposes: $26
  Outcome: ACCEPTED

Week 4:
  urgency_score = 0.78
  Platform proposes: $29
  Outcome: ABANDONED

Calibration:
  For this profile + airport context:
    acceptance ceiling ≈ $26–27
    urgency_score was overestimating willingness to pay
  → Adjust urgency → ceiling mapping downward
```

Over millions of rides, the estimates converge to accurate threshold predictions without ever directly observing the hidden threshold.

---

## A/B Testing Framework for Signal Validation

Signals must be validated in controlled experiments — not assumed.

```
Cohort A (10% of users): base_fare × surge  (current system — control)
Cohort B (10% of users): platform LLM with behavioral signals
Cohort C (10% of users): platform LLM with deliberately perturbed signals
                          (tests which signals the model actually relies on)

Metrics compared:
  - acceptance rate
  - revenue per completed ride
  - driver acceptance rate
  - cancellation rate
  - user LTV at 30 days
```

Cohort C is most important — it identifies spurious signal correlations before they cause pricing errors at scale.

---

## End-to-End Example: Airport at Midnight

```
User: opens Uber at airport terminal, 11:48pm, heavy rain
      4th app open in last 12 minutes (tracked in session)

Feature assembly computes:
  patience_estimate    = 0.35  (low — repeated opens, time pressure)
  mood_estimate        = "hesitant"
  urgency_score        = 0.91  (airport + late + rain + repeat opens)
  threshold_estimate   = $48   (hist avg $38, urgency shifts ceiling +$10)
  surge                = 2.1x
  supply               = 0 (scarce)

Platform LLM receives this observation → proposes: $44
Current system would propose: base_fare($22) × 2.1 = $46.20

Outcome:
  Platform LLM: $44 — confident close, reads patience correctly
  Current:      $46.20 — surge only, no patience awareness
  
Both accepted, but platform LLM:
  - Books faster (less hesitation at lower price)
  - Driver dispatched sooner → better actual ETA
  - User more likely to rebook (positive experience)
```

---

## The Distribution Shift Problem and Mitigations

### The Problem
Training used exact patience values (0.62, 0.40). Deployment uses estimated patience with uncertainty. The model was never trained on noisy feature estimates — distribution shift exists.

### Mitigation 1 — Noise-Augmented Fine-Tuning
Before deployment, re-run training episodes replacing exact patience with `patience_estimate ± noise`. Forces robustness to imprecise signals.

### Mitigation 2 — Confidence-Weighted Blending
```python
if urgency_estimate_confidence > 0.8:
    final_price = platform_llm_price
else:
    final_price = (confidence × llm_price) + ((1 − confidence) × base_surge_price)
```

### Mitigation 3 — Shadow Mode First
Run platform LLM in parallel with current system for 30 days. Do not use its prices. Compare predictions to actual outcomes. Only deploy when calibration error is below acceptable threshold.

---

## Honest Assessment of Real-World Value

### Direct deployment value
A behavioral pricing oracle — better individual acceptance prediction per context vs current aggregate market-only signals.

### Deeper value
The simulation framework itself: a calibratable model of user behavior under price negotiation, trained through RL rather than historical regression. Lets you test pricing strategies without running expensive real experiments on users.

### What requires real data to unlock
The synthetic training thresholds are procedurally generated — not from actual user behavior. Full value requires calibration on real ride data to align the simulation's threshold distributions with actual user willingness-to-pay distributions.

### Key claim that is defensible
Even noisy patience estimates outperform pure surge multiplier pricing because they incorporate a dimension of information — individual session behavior — that the current system completely ignores.
