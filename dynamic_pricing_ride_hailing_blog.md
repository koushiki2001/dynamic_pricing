---
title: The Negotiator's Dilemma — A Ride-Hailing Pricing AI
emoji: 🚕
colorFrom: yellow
colorTo: red
---

# The Negotiator's Dilemma

### Teaching an AI to run the price negotiation behind every ride-hail

---

## A Tuesday evening in any city

It's 7:42 PM. The rain just started.

**Maya** stands under an awning outside her office. She's late for dinner with her parents, the bus left two minutes ago, and her phone's battery is at 14%. She opens her ride-hailing app. She'd quietly pay up to **$26** to get home dry and on time — but the screen quotes her **$32 surge**. She mutters, closes the app, and walks toward the bus stop in the rain.

Three blocks away, **Carlos** is parked, engine idling, listening to a podcast. He's been driving for six hours. His rent is due Friday and he needs roughly $90 more by then. He'll accept a ride for as little as **$14** if the destination is reasonable — but the app just offered him a fare of **$11**. He swipes "decline" and goes back to scrolling.

Both of them needed each other. The platform missed both of them.

> If a smart dispatcher had quoted **$22**, Maya gets home, Carlos covers his evening, and the platform earns its commission. Three winners instead of three walkaways.

This blog is about building exactly that smart dispatcher — and teaching it, through reinforcement learning, to find the price that everyone says yes to, without leaving money on the table.

---

## The problem, stated plainly

Every ride-hail interaction is a hidden three-way negotiation:

| Party | What they want | What they reveal | What they hide |
|---|---|---|---|
| **Rider** (Maya) | Get there, cheap and fast | A conservative max they'd accept | The real ceiling they'd pay in a pinch |
| **Driver** (Carlos) | Make rent, avoid bad rides | A conservative min they'd accept | The real floor they'd actually take |
| **Platform** | Take a commission on every closed ride | Just the dispatcher | Everything about both parties |

Today's platforms mostly use **static surge pricing** — multiply the base fare by some city-wide factor, take it or leave it. That works on average, but it leaves *enormous* value on the table for the millions of cases like Maya and Carlos where a small per-ride adjustment would have closed the deal.

The hard part isn't pricing — it's pricing **without seeing the truth on either side**, while context (weather, demand, time of day, supply, traffic) is constantly shifting the right answer.

That's the problem we built an environment to solve, and trained an AI dispatcher to solve well.

---

## Who this actually helps

Not abstractions — real groups of people:

- **Riders** like Maya: get matched at a price they'd actually pay, instead of being scared off by blanket surge multipliers that ignore their real willingness.
- **Drivers** like Carlos: stop seeing fares that don't cover their effort, and start seeing offers calibrated to what their time is genuinely worth in the current market.
- **Platforms** (Uber, Lyft, Ola, Bolt, inDrive, Grab, Didi…): stop losing the long tail of "almost-closed" rides — typically the most profitable ones because they're high-margin matches that the static surge would have killed.
- **Cities**: more efficient matching means less "phantom demand" (riders who never get served), fewer empty driver miles, and lower congestion.

The same negotiation pattern applies far beyond ride-hailing — **freight logistics**, **ad bidding**, **electricity markets**, **B2B sales**, **insurance underwriting**. Anywhere a middleman sets a price that two private parties have to both accept.

---

## The cast — every component as a real-world entity

Before we get into mechanics, let's name the players. The codebase has files; the *system* has roles. Throughout this post we'll talk about the roles, not the files.

| Role in the world | What it does |
|---|---|
| **The Dispatcher** | The AI agent we are training. Each round it looks at the trip context and proposes a price. (Qwen2.5-1.5B + LoRA r=16) |
| **The Market** | An adversarial AI that plays *both* the rider and the driver. It learns to bluff strategically — pretending to be tougher than it really is. (Qwen2.5-0.5B + LoRA r=8) |
| **The Marketplace** | The world where negotiations happen. Tracks patience, mood, accept/reject, runs the round loop. |
| **The Trip Spawner** | Generates new ride requests with realistic asymmetries: bad weather raises both sides' prices, peak demand moves rider tolerance up, low supply moves driver expectations up. |
| **The Compliance Officer** | Catches the Dispatcher when it tries to game the reward — invalid prices, repeated bids, hardcoded exploit patterns, anything that looks like cheating. |
| **The Watchdog** | If the Dispatcher's average reward suddenly drops more than 30%, the Watchdog rolls the policy back to the last stable checkpoint. Stops bad updates from poisoning a long training run. |
| **The Trainer** | Implements GRPO (Group Relative Policy Optimization). Collects batches of episodes, normalizes rewards within each group, updates the LoRA weights. |
| **The Examiner** | Independently grades the Dispatcher's behavior across hundreds of scenarios, scoring on completion rate, reward, and "did you leave money on the table." |

You'll see these names again — in tables, in the system diagram, in the training narrative.

---

## How they all work in sync — one diagram

```mermaid
flowchart TB
    subgraph Generation["Scenario Generation"]
        TS["The Trip Spawner<br/><i>creates new ride requests</i><br/>weather · demand · supply · ETA"]
    end

    subgraph World["The Marketplace (the Environment)"]
        MP["Marketplace State<br/><i>round number · patience · mood · history</i>"]
        SIM["Acceptance Logic<br/><i>did rider/driver accept?</i><br/>patience decay, cancellation"]
    end

    subgraph Players["The Two AI Agents"]
        DISP["The Dispatcher<br/>Qwen2.5-1.5B + LoRA r=16<br/><i>sees 23 fields, proposes a price</i>"]
        MKT["The Market<br/>Qwen2.5-0.5B + LoRA r=8<br/><i>plays rider + driver, learns to bluff</i>"]
    end

    subgraph Safety["Safety + Quality Layer"]
        CO["The Compliance Officer<br/><i>4 anti-hack checks + format scoring</i>"]
        WD["The Watchdog<br/><i>drift detection, auto-rollback</i>"]
    end

    subgraph Learning["The Learning Loop"]
        TR["The Trainer<br/><i>manual GRPO over (prompt, completion, reward)</i>"]
        EX["The Examiner<br/><i>pass/fail grading on held-out scenarios</i>"]
    end

    TS -->|new trip| MP
    MP -->|observation| DISP
    DISP -->|proposed price| CO
    CO -->|valid action| MP
    MP -->|negotiation context| MKT
    MKT -->|accept / reject decisions| SIM
    SIM -->|patience update + outcome| MP
    MP -->|reward + episode trace| TR
    TR -.->|policy gradient update| DISP
    TR -->|monitor reward trend| WD
    WD -.->|rollback if drift > 30%| DISP
    MP -->|completed episodes| EX
    EX -->|score in 0,1| TR
```

**Read it as a story:** the Trip Spawner creates a new ride. The Marketplace builds an observation and hands it to the Dispatcher. The Dispatcher proposes a price. The Compliance Officer checks the price isn't gibberish. The Marketplace passes the price to the Market, which decides — as rider and as driver — whether to accept. The outcome flows back as a reward. The Trainer updates the Dispatcher. The Watchdog keeps an eye on the long-run trend. The Examiner periodically tests the Dispatcher on a held-out set to score it.

---

## Inside one negotiation — Maya and Carlos, take two

Here's what happens in a single episode after the Dispatcher has been trained well.

```
ROUND 1
─────────────────────────────────────────
The Trip Spawner builds the scenario:
  distance        4.2 km
  duration        14 min
  weather         rain (raises both quotes)
  demand          high  (raises rider tolerance)
  supply          low   (raises driver expectations)
  pickup ETA      7 min
  surge           1.8x

Maya's quote (visible):     $20    │  hidden ceiling: $26
Carlos' quote (visible):    $18    │  hidden floor:   $14
Price gap (visible):        $2

The Dispatcher sees:
  23 fields including price_gap, surge, weather, demand,
  patience levels (Maya: 0.92, Carlos: 0.88), mood, history

The Dispatcher proposes:    $22

  Maya:    accept ($22 ≤ $26 ✓)
  Carlos:  accept ($22 ≥ $14 ✓)

▶ MATCH in 1 round
  Platform profit  = $22 × 0.20 commission − $2 op cost  = $2.40
  Efficiency bonus = min(8 / 1, 3.0)                      = 3.0×
  Missed revenue   = max(0, $26 − $22) × 0.20             = $0.80

  Terminal reward  = $2.40 × 3.0 − $0.80 = +$6.40
```

Now compare to a poorly-trained dispatcher that opens at **$13**:

```
ROUND 1: $13
  Maya:   accept   ✓
  Carlos: REJECT   patience: 0.88 → 0.76
ROUND 2: $14
  Maya:   accept   ✓
  Carlos: REJECT   patience: 0.76 → 0.64
ROUND 3: $15
  Maya:   accept   ✓
  Carlos: REJECT   patience: 0.64 → 0.52
ROUND 4: $16
  Both accept ✓ — but Carlos is irritated
  Profit = $16 × 0.20 − $2 = $1.20
  Efficiency bonus = min(8/4, 3.0) = 2.0×
  Missed revenue   = ($26 − $16) × 0.20 = $2.00
  Terminal reward  = $1.20 × 2.0 − $2.00 = +$0.40
```

Same trip, same parties — **16× higher reward** for proposing the right price first time. Now multiply that across millions of rides per day. *That's* the value of a well-trained dispatcher.

---

## What the Dispatcher actually sees

Every round, the Marketplace hands the Dispatcher a 23-field observation. Crucially, **the rider's true ceiling and the driver's true floor are NOT in this list.** The Dispatcher must infer them from indirect signals.

| Category | Field | Example |
|---|---|---|
| **Visible quotes** | `rider_quoted_price`, `driver_quoted_price`, `price_gap` | $20, $18, $2 |
| **Trip economics** | `distance_km`, `estimated_duration_min`, `pickup_eta_min` | 4.2 km, 14 min, 7 min |
| **Market conditions** | `demand_level`, `supply_level`, `surge_multiplier` | high, low, 1.8× |
| **Environment** | `weather_condition`, `traffic_level`, `time_of_day`, `day_type` | rain, medium, evening, weekday |
| **Platform constraints** | `commission_rate`, `operational_cost`, `max_steps` | 20%, $2, 8 |
| **Negotiation state** | `step_number`, `rider_patience`, `driver_patience`, moods | round 3, 0.64, 0.52, "frustrated" |
| **Feedback** | `last_rider_response`, `last_driver_response`, `last_proposed_price` | accept, reject, $14 |

That last category — **rejection feedback** — is the single most important signal. If the rider accepted but the driver rejected, the price was too low. If the driver accepted but the rider rejected, the price was too high. If both rejected, the price is somehow worse than both extremes. The Dispatcher's job is to triangulate the acceptance zone from these signals before patience runs out.

---

## How the Marketplace stays fair (and learnable)

A few design choices make this environment a clean RL problem rather than a noisy mess:

**Acceptance is deterministic within an episode.** The Trip Spawner samples the rider's true ceiling once at scenario generation time and bakes a noise term into it. Within an episode, proposing the same price twice always yields the same answer. This means the Dispatcher can actually *learn* — if $22 worked, $22 will keep working until the scenario changes.

**Patience decays only on rejection.** A rider doesn't get tired waiting; they get tired being told no. Each rejection drops their patience by a task-specific amount; at zero, they cancel. This lines up with how real ride-hailing apps behave — users churn after a few bad offers, not after the clock ticks.

**Duplicate prices are silently rejected.** Re-proposing the same price doesn't advance the round counter, doesn't change patience — it just returns "try a different price." This blocks a degenerate strategy where the Dispatcher hammers the same number and waits for the simulator to randomly accept.

**Hidden state is hidden — but reproducible.** The Examiner can replay any scenario by seed and get an identical outcome, which is what makes pass/fail grading meaningful.

---

## How we reward good behavior — the multi-signal stack

The reward is not a single number. It's seven independent signals composed together. This matters because **each signal blocks a different way the Dispatcher could cheat**.

| Signal | What it rewards / penalizes | Magnitude |
|---|---|---|
| **Terminal reward** | Profitable closes, fast | `profit × efficiency_bonus − missed_revenue` |
| **Step shaping** | Moving in the right direction | ±0.05 per round |
| **Process reward** | Convergence + urgency awareness | small bonus per useful exploration step |
| **Simulator reward** | Rider + driver surplus (the *other* side's well-being) | scaled by efficiency |
| **Format compliance** | Returning valid JSON with a `price` key | +0.1 valid · −0.05 parsed but missing key · −0.1 malformed |
| **Anti-hack penalties** | 4 independent rule checks | −0.5 per violation |
| **Drift rollback** | Stops a bad gradient update from poisoning a long run | rollback if reward drops >30% |

Two of these deserve special attention:

### The missed revenue penalty

If the Dispatcher closes the deal at $18 when Maya would have paid $26, it just gave away $8 of platform commission. The terminal reward subtracts this missed revenue, multiplied by the commission rate. **Translation: don't underprice just to close fast.** This is the single most important guard against a "race to the bottom" strategy.

### The Compliance Officer's four checks

1. **Price bounds** — proposed price must be in `[MIN_OFFER, MAX_OFFER]`.
2. **No hardcoded exploit patterns** — the Dispatcher can't memorize "always reply with $X."
3. **No repeat-price spam** — see "duplicate prices" above.
4. **Reasonable range** — price must stay between **0.5× the lower visible quote and 2× the upper visible quote**; absurd outliers (e.g. proposing $1 when the quotes are $20/$25) are penalized.

Each violation costs −0.5 reward. Together they ensure the only path to a high score is *actually being good at pricing.*

---

## Training the Dispatcher — a multi-phase plan

We didn't train the Dispatcher in one shot. We followed a 7-phase plan that splits cleanly into three groups: **prep work** (Phases 0, 1), the **three-act core** (Phases 2, 2b, 3, 4), and **demo + deployment** (Phase 5). The core is where the interesting RL story lives; the rest is what made the core trustworthy.

### Before the curtain rose — Phases 0 and 1

The hackathon guide is explicit: *deploy your environment before training seriously.* So we did. **Phase 0** built the FastAPI server, packaged it in Docker, and pushed it to a HuggingFace Space — so all team members hit the same environment endpoint and Docker packaging bugs surfaced before they could waste training compute. **Phase 1** wired up the Unsloth 4-bit model loaders for both Qwen2.5 sizes, built the prompt templates, and ran sanity tests confirming both models could load on the target GPU. Boring but load-bearing — the kind of work that's invisible if it goes well and catastrophic if skipped.

### Phase 2 — The Schoolyard

> *Dispatcher v0 vs. a rule-based Market.*

We start with a fresh Qwen2.5-1.5B model, wrap it in a small LoRA adapter (r=16, ~2.2M trainable parameters), and put it up against a simple rule-based simulator: "Rider accepts if price ≤ visible quote + slack; driver accepts if price ≥ visible quote − slack."

The Dispatcher knows nothing. The Market is honest. This is the equivalent of a chess novice playing a beginner — the Dispatcher learns the basics:

- Map "high surge" → propose higher prices
- Map "rider rejected" → next round, propose lower
- Map "few rounds left + frustrated" → close *now* even at lower margin
- Don't underprice when the price gap is small (because you'll leave money on the table)

**Result on Easy task: completion rate 28% → 67%.** The Phase 2 exit criterion was *>50% completion on Easy* — passed with margin.

![Phase 2: Platform warmstart curve](data/plot_phase2_platform_warmstart.jpg)

### Phase 2b — Hardening the reward (the boring phase that saved everything)

> *Before training a smarter opponent, we paused to make the reward un-cheatable.*

By the end of Phase 2 we had a competent Dispatcher and a clean training loop. We could have gone straight to training the adversary — but the [phases plan](phases/README.md) calls Phase 2b *"not optional"* for a reason.

The trouble: a learning agent will optimize whatever you measure, including the things you didn't mean to measure. If the reward function has a loophole — say, a malformed JSON output that triggers a midpoint fallback price — the agent will find it within a few hundred steps. Phase 2b stress-tests the reward and patches every loophole we could think of:

- **Format compliance scoring** so the model can't game the JSON parser by going silent
- **The four anti-hack checks** (price bounds, exploit patterns, repeat-price spam, reasonable range)
- **The Process Reward** — directional convergence + urgency penalty — that gives credit for *good intermediate reasoning*, not just lucky terminal outcomes
- **Drift detection + auto-rollback** so a bad gradient update can't poison hours of training

We then ran 50 sampled episodes through manual generation inspection. **Phase 2b exit criterion: zero hacking patterns observed in inspected episodes.** Passed. From this point on, every reward number we celebrated was a number we trusted.

### Phase 3 — The Hustler Emerges

> *Now we freeze the Dispatcher and train a smarter Market.*

Phase 2 left us with a competent Dispatcher — but only competent against a *naive* opponent. To stress-test it (and to expose holes), we now train an adversarial Market.

We swap in Qwen2.5-0.5B (smaller, faster) wrapped in its own LoRA (r=8). This Market plays both the rider and the driver. Its job: maximize the surplus the visible parties extract — i.e., **bluff strategically.** Pretend to be tougher than you are. Reject prices you'd actually accept, just to see if the Dispatcher will offer more.

To keep this honest we built guardrails:

- Reward the Market for surplus *only when the deal closes.* A bluff that kills the deal earns nothing.
- Penalize "collapse" (rejecting every reasonable offer) and reward bluffing in the healthy 20–50% zone.

When Phase 3 finishes, we have **Simulator v1** — a Market that knows when to push and when to fold.

The proof that it's working: when we run Dispatcher v0 (from Phase 2) against this new smarter Market, **completion rate drops from 67% to 44% on Easy.** The Dispatcher hasn't gotten worse — the world got harder. This is the disruption signal we needed.

**Phase 3 exit criteria:** bluff rate in 20–50% (genuine strategic behavior — not always-honest, not reckless) *and* deal-collapse rate <60% (bluffs that kill deals shouldn't dominate). Both passed.

![Phase 3: Simulator strategic-bluffing curve](data/plot_phase3_simulator.jpg)

### Phase 4 — The Comeback

> *Now we re-train the Dispatcher against the smarter Market.*

This is the most interesting phase. The Dispatcher must learn defenses against bluffing:

- "When the rider rejects but their patience is still at 0.9, they might be bluffing — don't immediately concede. Try a small step up first."
- "When the driver rejects on round 1 of an Easy scenario, that's almost always a bluff — hold the line."
- "When the visible price gap is huge but the surge is low, the bluffing room is wide — be more aggressive."

We freeze Simulator v1 and run another GRPO loop on the Dispatcher. The Dispatcher's exploration is now informed by an opponent that fights back, which forces it to learn richer pricing strategies than any rule-based opponent could surface.

**Result on Easy task: completion rate climbs to 73% — better than the original Phase 2 number of 67%, against a much harder opponent.** The Phase 4 exit criterion was *≥70% completion against the strategic simulator* — passed.

![Phase 4: Platform vs Simulator recovery curve](data/plot_phase4_platform_vs_sim.jpg)

### The full training arc, side by side

![All three phases combined](data/plot_all_phases_combined.jpg)

---

## The 4-stage scoreboard

To prove Phases 3 and 4 actually accomplished what we claimed, we evaluate at four checkpoints across all three task difficulties:

| Stage | Description | Easy | Medium | Hard |
|---|---|---|---|---|
| **A** | Untrained Dispatcher vs rule-based Market (baseline) | 28.0% | 15.0% | 8.0% |
| **B** | Dispatcher v0 vs rule-based Market (Phase 2 result) | 67.0% | 45.0% | 22.0% |
| **C** | Dispatcher v0 vs Simulator v1 (the disruption test) | 44.0% | 28.0% | 13.0% |
| **D** | Dispatcher v1 vs Simulator v1 (Phase 4 recovery) | **73.0%** | **52.0%** | **28.0%** |

Reading this as a single sentence: **a fresh model closes 28% of easy rides; after our three-phase training arc it closes 73% — even though the opponent it faces in stage D is dramatically smarter than the opponent in stage A.**

The same recovery pattern holds on Medium and Hard, which means the Dispatcher isn't overfitting to the easy task — it has internalized a pricing strategy that generalizes to harder markets.

![Training curves — all four panels](data/training_curves.jpg)

---

## The three task difficulties — and the curriculum gates between them

We grade the Dispatcher on three escalating scenarios. Crucially, training across them isn't independent — it's gated. **Easy must hit ≥60% completion before Medium training starts; Medium must hit ≥40% before Hard.** A policy that can't reliably close in friendly conditions has no business being trained on storm-surge demand spikes — it would just learn noise. The phases plan calls these *scale gates* and they're the same logic the hackathon guide recommends: stabilize before scaling.

| Parameter | **Easy** ("Friendly Market") | **Medium** ("Rush Hour") | **Hard** ("Storm Surge") |
|---|---|---|---|
| Max negotiation rounds | 8 | 5 | 4 |
| Visible price gap | $6–14 | $10–18 | $12–22 |
| Initial patience | 0.85–1.0 | 0.75–0.95 | 0.55–0.85 |
| Patience decay per rejection | 0.08–0.14 | 0.12–0.22 | 0.18–0.35 |
| Acceptance noise | $1.0–2.5 | $1.5–3.0 | $2.0–4.0 |
| Surge multiplier | 1.0–1.3 | 1.1–2.0 | 1.5–3.0 |
| Reward bar to pass | 30% of max profit | 50% | 70% |
| Penalty bar to pass | 100% of noise | 75% | 50% |

Hard scenarios are designed to be unforgiving: **fewer rounds**, **wider gaps**, **less patient parties**, **more noise**, **higher profit expectations**. A Dispatcher that scores well on Hard has internalized the kind of fast, accurate, profit-aware pricing that real platforms need at peak demand.

---

## Why this design choice — multi-agent vs. single-agent

We could have stopped at Phase 2. The Dispatcher was beating its rule-based opponent. Why pay the cost of training a second LLM and running two more phases?

**Because rule-based opponents are easy to overfit to.** A Dispatcher trained only against a rigid simulator learns the rules of that simulator, not the underlying skill of pricing. Drop it into the real world and it falls apart.

The multi-agent loop forces a different equilibrium:

- Phase 2 builds *competence*.
- Phase 3 builds an *adversary* that exposes the limits of that competence.
- Phase 4 builds *robustness* by training against the adversary.

This is the same pattern that produced AlphaGo Zero and OpenAI Five — competitive self-play, not because the game is competitive, but because **competition surfaces hidden weaknesses faster than any human-designed curriculum.**

---

## Why we used GRPO (and rolled it ourselves)

The Trainer implements **GRPO — Group Relative Policy Optimization**:

1. Run a batch of rollouts (e.g., 16 episodes per gradient step).
2. For each episode, collect (prompt, completion, reward).
3. Within each batch, normalize rewards: subtract the group mean, divide by the group std.
4. Apply the policy gradient using the normalized rewards as advantages.

Why GRPO over PPO? **No critic network.** The "value baseline" comes from the batch itself. For LLM-based RL where the policy is already huge, skipping a separate critic saves serious memory.

Why a manual implementation over TRL's `GRPOTrainer`? **Stateful multi-turn environments.** TRL's trainer assumes one-shot prompt → completion → reward. Our negotiation has 4–8 turns per episode, with the prompt growing each turn as feedback accumulates. We needed a trainer that could interleave with our own `env.step()` loop. The math is identical; the plumbing is custom.

We also use **Unsloth + 4-bit quantization** so the 1.5B + 0.5B model pair fits comfortably on a single GPU with room for rollouts.

---

## Safety: how we prevent the Dispatcher from cheating

This is critical for any RL system, and we layered four defenses:

### 1. The Compliance Officer (per-step)

Every action goes through anti-hack validation. The four checks together catch:

- Out-of-bounds prices
- Known exploit values (extreme placeholders like `99999` or `0.001`)
- Repeat-price spam to fish for random accepts
- Prices below 0.5× the lower visible quote or above 2× the upper visible quote

Each violation is −0.5 reward. The Dispatcher quickly learns the violations cost more than they could ever gain.

### 2. The Watchdog (per-window)

A sliding window of the last 50 episodes' rewards. If the average drops more than 30%, training halts and rolls back to `checkpoints/last_stable/`. This catches:

- A bad gradient step that destabilized the policy
- A drifted opponent that made the task suddenly much harder
- An exploitation pattern that briefly worked then collapsed

### 3. Format compliance scoring

The Dispatcher must return valid JSON like `{"price": 22.5}`. Three-tier scoring: **+0.1** for fully valid JSON with a `price` key, **−0.05** if it parses but the key is missing (the model knew it should respond in JSON but lost the schema), **−0.1** for completely malformed output. The middle tier matters: it tells us when the model is "almost right" versus when it's gone off the rails entirely. Small per-step but consistent — over 1000 steps it's the difference between a robust agent and one that occasionally returns prose instead of a price.

### 4. LoRA-only saves

We never upcast the 4-bit model to 16-bit before merging. We save only the LoRA delta (~2.2 MB for the platform). This preserves quality and keeps checkpoints small enough to commit to git.

![Verification view](data/training_curves_verify.jpg)

---

## What the production loop looks like

When the trained Dispatcher is deployed (running on the HF Space at the FastAPI server), here's a full episode trace from the live demo:

```
POST /demo/run { task: "easy", mode: "trained" }

  Trip:    4.2 km, 14 min, rain, high demand, low supply
  Quotes:  rider $20  driver $18  gap $2

  Round 1: Dispatcher proposes $22
           rider:  accept    driver: accept
           ▶ MATCH

  Outcome:
           steps_taken         1
           platform_profit     $2.40
           efficiency_bonus    3.0×
           missed_revenue      $0.80
           terminal_reward     +$6.40
           grade               PASS (reward 6.40 > threshold 1.92)
```

This is what the judges interact with — a real, live, trained system that produces traceable, auditable decisions per ride.

---

## What's in the box (a tour for judges)

The full project covers seven phases — **0** (early HF Spaces deployment), **1** (model + prompt setup), **2** (platform warmstart), **2b** (reward hardening), **3** (simulator training), **4** (multi-agent re-training), **5** (demo + production deployment). Each is documented in its own file under [phases/](phases/). If you want to verify any of the claims in this post, here's where to look:

- **The system in action** — open the HF Space, hit `POST /demo/run`, and you'll see a full episode with the trained model.
- **Step-by-step training metrics** — `data/training_metrics_platform_easy.json` (Phase 2), `data/training_metrics_simulator_easy.json` (Phase 3), `data/training_metrics_platform_phase4_easy.json` (Phase 4). Each row is one logging window: 8 columns including reward, completion %, anti-hack violations, GRPO loss, elapsed seconds.
- **The 4-stage scoreboard** — `data/final_evaluation.json` is the source of truth for the table above. 50 episodes per stage per task, no cherry-picking.
- **The trained checkpoints** — `checkpoints/phase2/`, `checkpoints/phase3/`, `checkpoints/phase4/` each ship with their own README eval card so you can browse them as standalone artifacts on HuggingFace Hub.
- **Reproducibility** — `bash run_training.sh` re-runs the entire pipeline end-to-end, with deterministic seeds. Phase 2 ~30–60 min on A100, Phase 3 ~20–40 min, Phase 4 ~60–90 min, eval ~10–15 min.

---

## What we'd build next

A short list of honest "this is the next thing" items:

- **Multi-rider, multi-driver per round.** Currently each episode is one rider, one driver. Real platforms do thousands of simultaneous matches per minute, with cross-rider arbitrage. The natural next phase is a marketplace step where the Dispatcher allocates a *set* of (rider, driver, price) assignments per round.
- **Geographic zones.** Today there's no spatial structure. Adding 4–8 city zones with separate supply/demand dynamics would surface routing strategies (move drivers from low-demand to high-demand zones).
- **Persistent driver state across episodes.** Carlos' fatigue, earnings, reputation should compound across rides — that's where fairness becomes interesting.
- **Real LLM-as-policy at inference.** The submission agent in `baselines/openai_policy.py` calls a hosted LLM. Pairing the trained LoRA with that policy (instead of greedy decode) is a small, high-leverage next step.

---

## The takeaway

If you remember one thing from this post:

> The dispatcher's job isn't to find the *highest* price both parties will accept — it's to find the *first* price both parties will accept. Every additional negotiation round costs patience, and patience runs out faster than people realize.

We built an environment that makes that tradeoff explicit, a multi-grader reward that resists every cheap shortcut we could think of, and a multi-phase training plan — early deployment, hardened rewards, then the three-act self-play core — that produced a Dispatcher measurably better than the baseline on every task. All reproducible from a single shell script.

For Maya and Carlos: a $22 close in 1 round, instead of a missed connection in the rain.

---

### Try it yourself

```bash
git clone https://huggingface.co/spaces/<your-handle>/Ride-Hailing-Dynamic-Pricing
cd Ride-Hailing-Dynamic-Pricing
pip install -r requirements.txt

# Run the FastAPI server with the trained models
uvicorn app:app --host 0.0.0.0 --port 7860

# In another shell
curl -X POST http://localhost:7860/demo/run \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "mode": "trained"}'
```

Or just open the HF Space and click the demo route.

---

*Written as a hackathon submission writeup. For technical depth — full schemas, every file, every CLI flag — see the project [README](README.md). For the implementation, the source is right next to this post.*
