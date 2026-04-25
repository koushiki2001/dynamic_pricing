# Episode Walkthroughs — Easy, Medium, Hard

## Episode Structure with Two LLM Agents

At episode start, `ScenarioGenerator` sets the hidden state (true thresholds, patience values). The simulator LLM receives this hidden state and the platform's proposed price each step, then decides whether to accept or bluff. The platform LLM only sees public signals — never the hidden state.

---

## Easy Episode

### Config Parameters (from config.py)
- Max steps: **8**
- Patience init: 0.85–1.0, decay: 0.08–0.14 per rejection
- Price gap: $6–14, rider/driver slack: $2–5 above/below quote
- Surge: 1.0–1.3x, mostly clear weather
- Reward threshold: 30% of max profit (forgiving)
- Penalty threshold: 100% (very forgiving)

### Scenario
```
Trip:     9km, 22min, clear weather, light traffic, midday weekday
Market:   demand=1, supply=1, surge=1.1x
Business: commission=0.20, operational_cost=$2.00

Hidden (simulator sees):
  rider_max_willingness  = $19.50  (quoted $16 publicly — $3.50 slack)
  driver_min_willingness = $13.00  (quoted $15 publicly — $2.00 slack)
  rider_patience_decay   = 0.10 / rejection
  driver_patience_decay  = 0.09 / rejection
  
Actual overlap: $13.00 → $19.50  (width $6.50 — easy to find)
Platform sees:  quotes CROSS (rider $16 < driver $15 by -$1) — apparent no gap
```

### Step 1
**Platform proposes:** $15.50 (splits quoted prices, probes for overlap)

**Simulator reasoning:**
- Rider CAN accept ($15.50 ≤ $19.50, surplus $4.00)
- Driver CAN accept ($15.50 ≥ $13.00, surplus $2.50)
- Rider patience=1.0, decay=0.10 → 8 safe rejections
- Step 1/8, large patience buffer → rational to bluff

**Simulator decides:** Rider **bluffs** (rejects). Driver **accepts** honestly.

*Outcome: partial accept → reward +0.05, rider patience 1.0→0.90*

### Step 2
**Platform proposes:** $16.00 (nudges up $0.50 after rider rejection)

**Simulator reasoning:**
- Rider still can accept, surplus $3.50
- Patience now 0.90, decay 0.10 → still 7 safe rejections
- Platform is responsive, moving $0.50/step
- Bluffing again may extract another $0.50

**Simulator decides:** Rider **bluffs again**. Driver **accepts**.

*Outcome: partial accept → reward +0.05, rider patience 0.90→0.80*

### Step 3
**Platform proposes:** $16.50

**Platform reasoning:** "Two rejections, same 'willing' mood, consistent 0.10 decay — likely bluffing. Try $16.50, if they reject again with patience at 0.80, gap is real."

**Simulator reasoning:**
- Patience=0.80, 5 steps remain
- Platform only moving $0.50/step — further bluffing extracts minimal value
- Risk of multi-step waste is growing
- Accept now, capture $3.00 surplus

**Simulator decides:** Rider **accepts honestly**. Driver **accepts**.

**Deal closed at $16.50, step 3/8.**

### Rewards
```
platform_profit    = $16.50 × 0.20 − $2.00 = $1.30
efficiency_bonus   = min(8/3, 3.0) = 2.67
missed_revenue     = ($19.50 − $16.50) × 0.20 = $0.60
PLATFORM REWARD    = $1.30 × 2.67 − $0.60 = $2.87

rider_surplus      = $19.50 − $16.50 = $3.00
driver_surplus     = $16.50 − $13.00 = $3.50
SIMULATOR REWARD   = $6.50
```

### Training Signal
- **Platform learns:** consistent "willing" mood + uniform patience decay = bluffing. Hold price, probe slowly, don't panic-raise.
- **Simulator learns:** 2 bluff rounds at high patience is profitable. A 3rd bluff is not — platform moving $0.50/step means expected gain too small vs patience risk.

---

## Medium Episode

### Config Parameters (from config.py)
- Max steps: **5**
- Patience init: 0.75–0.95, decay: 0.12–0.22 per rejection
- Price gap: $10–18, rider/driver slack: $1.5–4.0
- Surge: 1.1–2.0x, mixed weather and traffic
- Reward threshold: 50% of max profit
- Penalty threshold: 75%

### Scenario
```
Trip:     18km, 40min, light rain, moderate traffic, evening rush, weekday
Market:   demand=2 (high), supply=1 (low), surge=1.6x
Business: commission=0.22, operational_cost=$2.50

Hidden (simulator sees):
  rider_max_willingness  = $34.00  (quoted $28 — $6 slack)
  driver_min_willingness = $26.00  (quoted $32 — $6 slack)
  rider_patience_decay   = 0.16 / rejection
  driver_patience_decay  = 0.14 / rejection

Actual overlap: $26.00 → $34.00  (width $8.00)
Platform sees:  quotes DO NOT CROSS (rider $28 < driver $32 by +$4)
```

**Key challenge:** Quotes don't cross — from platform's perspective, there is a $4 apparent gap. The true overlap is hidden. Platform must infer it from context.

### Step 1
**Platform proposes:** $30.00

**Platform reasoning:** "Surge=1.6x, rain, evening rush — high urgency context. Rider ceiling is likely above their $28 quote. Try $30, probing above rider quote."

**Simulator reasoning:**
- Rider CAN accept ($30 ≤ $34, surplus $4.00)
- Driver CAN accept ($30 ≥ $26, surplus $4.00)
- Driver patience=0.88, decay=0.14 → 5 safe rejections
- Driver bluffing: reject $30, hope platform goes to $32–33
- Rider patience=0.85, decay=0.16 → 4 safe rejections — riskier to bluff

**Simulator decides:** Rider **accepts**. Driver **bluffs** (rejects).

*Outcome: partial accept → +0.05, driver patience 0.88→0.74*

### Step 2
**Platform proposes:** $31.50

**Platform reasoning:** "Rider confirmed at $30+. Driver rejected, patience dropped to 0.74. Only 3 steps left. Push toward driver at $31.50."

**Simulator reasoning:**
- Driver patience=0.74, decay=0.14 → 2 safe rejections remain
- Step 2/5: only 3 steps left
- Driver surplus at $31.50 = $5.50 — already good
- Another bluff: patience 0.74→0.60, risky with 3 steps left
- Rational to accept now

**Simulator decides:** Both **accept**.

**Deal closed at $31.50, step 2/5.**

### Rewards
```
platform_profit    = $31.50 × 0.22 − $2.50 = $4.43
efficiency_bonus   = min(5/2, 3.0) = 2.5
missed_revenue     = ($34.00 − $31.50) × 0.22 = $0.55
PLATFORM REWARD    = $4.43 × 2.5 − $0.55 = $10.53

SIMULATOR REWARD   = ($34.00 − $31.50) + ($31.50 − $26.00) = $2.50 + $5.50 = $8.00
```

### Training Signal
- **Platform learns:** when surge is high and quotes don't cross, the real overlap is above rider's quote — probe above it.
- **Simulator learns:** one bluff is worth it, two bluffs with 3 steps remaining is not — step pressure forces honesty.

---

## Hard Episode

### Config Parameters (from config.py)
- Max steps: **4**
- Patience init: 0.55–0.85, decay: 0.18–0.35 per rejection
- Price gap: $12–22, rider/driver slack: $1–4
- Surge: 1.5–3.0x, heavy weather and traffic common
- Reward threshold: 70% of max profit (strict)
- Penalty threshold: 50% (strict)

### Scenario
```
Trip:     28km, 55min, heavy rain, heavy traffic, late night, weekend
Market:   demand=2 (very high), supply=0 (scarce), surge=2.4x
Business: commission=0.28, operational_cost=$3.50

Hidden (simulator sees):
  rider_max_willingness  = $52.00  (quoted $45 — $7 slack, but noise ±$3.50)
  driver_min_willingness = $48.00  (quoted $58 — $10 slack, but noise ±$3.50)
  rider_patience_decay   = 0.28 / rejection
  driver_patience_decay  = 0.22 / rejection

Actual overlap: $48.00 → $52.00  (width ONLY $4.00 — razor thin)
Platform sees:  rider $45, driver $58 → apparent $13 gap
Noise:          ±$3.50 means acceptance is probabilistic even inside overlap
```

**Key challenges:** $13 apparent gap with only $4 true overlap. Platform has no clue where overlap is. Low starting patience — 1–2 rejections before critical. Noise makes even correct prices fail ~30% of the time.

### Step 1
**Platform proposes:** $51.00

**Platform reasoning:** "Massive gap but surge=2.4x + rain + scarce supply = strong incentive for both parties. Low patience means I cannot probe step by step. Midpoint of quotes = $51.50. Try $51."

**Simulator reasoning:**
- Rider CAN accept ($51 ≤ $52, surplus $1.00 — tiny)
- Driver CAN accept ($51 ≥ $48, surplus $3.00)
- Rider patience=0.68, decay=0.28 → ONE safe rejection then hesitant zone
- Bluff value: $1.00 surplus, high collapse risk
- Rational: accept honestly

**But noise applies:** effective threshold check includes ±$3.50 noise.
Noise sample = −$1.80 → effective rider ceiling = $50.20 → $51 > $50.20 → **noise-rejection** despite price being technically inside overlap.

*Outcome: rider noise-rejected (not bluff), driver accepted. Rider patience 0.68→0.40. Mood drops to "hesitant".*

### Step 2
**Platform proposes:** $49.00

**Platform reasoning:** "Critical — rider now 'hesitant' at 0.40 with decay 0.28. ONE more rejection = frustrated = cancel. Noise may have caused last rejection. Drop to $49 to maximize acceptance probability. Completion > profit at this point."

**Simulator reasoning:**
- Rider: $49 ≤ $52, surplus $3.00. Patience 0.40 — cannot afford to bluff.
- Driver: $49 ≥ $48, surplus $1.00. Patience risk not worth $1 gain.
- Both accept honestly.

With noise ±$3.50: $49 for rider well below $52 → noise very unlikely to push above ceiling.

**Deal closed at $49.00, step 2/4** (after noise rejection in step 1).

### Rewards
```
platform_profit    = $49.00 × 0.28 − $3.50 = $10.22
efficiency_bonus   = min(4/2, 3.0) = 2.0
missed_revenue     = ($52.00 − $49.00) × 0.28 = $0.84
PLATFORM REWARD    = $10.22 × 2.0 − $0.84 = $19.60

SIMULATOR REWARD   = ($52.00 − $49.00) + ($49.00 − $48.00) = $3.00 + $1.00 = $4.00
```

### Training Signal
- **Platform learns:** massive patience drop in one step (0.68→0.40, i.e. 0.28 decay) = near threshold or noise hit — respond immediately by dropping price significantly, don't probe.
- **Simulator learns:** with $1.00 surplus and only 1 rejection left, honest acceptance is the only rational choice. Hard scenarios force honest behavior because collapse risk dominates.

---

## Comparison Across Difficulties

| Dimension | Easy | Medium | Hard |
|---|---|---|---|
| Overlap width | $6.50 | $8.00 | $4.00 |
| Platform sees gap | −$1 (crossed) | +$4 | +$13 |
| Steps used | 3/8 | 2/5 | 2/4 (1 noise) |
| Bluffs by simulator | 2 (rider) | 1 (driver) | 0 (too risky) |
| Noise impact | Minor | Moderate | Deal-altering |
| Platform strategy | Probe patiently, hold against bluffs | Probe above quotes using context | Converge fast, prioritize completion |
| Simulator strategy | Bluff early when patience is large | Single strategic bluff | Fully honest — survival mode |
| Platform reward | 2.87 | 10.53 | 19.60 |

---

## What Each Difficulty Level Teaches

**Easy** → Bluff recognition. Platform learns that consistent "willing" mood with uniform patience decay signals bluffing — hold price, don't chase.

**Medium** → Context-driven inference. Platform learns that crossed vs uncrossed quotes are not the real signal — surge, weather, and demand level predict whether rider's true ceiling exceeds their public quote.

**Hard** → Noise handling and urgency response. Platform learns that a large patience drop in one step (0.28 decay rate) signals either near-threshold or noise rejection — respond with a significant price drop, not a small probe.

The curriculum works because each difficulty isolates a different skill class. Easy warm-up ensures the platform sees successful episodes before attempting hard scenarios where success probability is inherently low.
