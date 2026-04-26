# From Static Surge to Behavioral Negotiation: A Story of Training a Pricing Agent

Most pricing systems in ride-hailing are excellent at *market-level balancing* (supply, demand, surge), but weak at *individual negotiation*.  
This project explores what happens when we train an agent to negotiate each ride like a micro-market: one rider, one driver, hidden willingness on both sides, and only a few rounds before someone walks away.

What follows is the practical journey implemented step-by-step in `phases/` — from environment deployment to multi-agent adaptation.

---

## Problem) what capability gap or interesting domain are you targeting?

### The gap
Traditional pricing logic usually treats users as aggregates. It does not explicitly model dynamic behavior such as:
- how rider/driver patience decays after rejections,
- how strategic bluffing can distort quotes,
- how context (storm, traffic, demand-supply imbalance) changes willingness moment to moment.

So the capability gap is:  
**Can a platform agent learn to infer hidden willingness and propose profitable, deal-closing prices under uncertainty and time pressure?**

### Why this domain is interesting
Ride pricing is not a one-shot prediction problem. It is a sequential decision problem:
- every proposal changes future behavior,
- each rejection consumes patience,
- late correct decisions can still fail if trust is already gone.

That makes it a strong real-world domain for reinforcement learning and agentic reasoning.

---

## Environment) what does the agent see, do, and get rewarded for?

### What the agent sees
At each step, the platform agent observes a structured state including:
- rider/driver quoted prices and the gap,
- patience + mood for both parties,
- last accept/reject responses,
- step number and remaining room to negotiate,
- context signals (surge, weather, traffic, demand, supply, time/day),
- platform economics (commission and operating cost).

It **does not** see the true rider max willingness or driver min willingness.  
Those are hidden and must be inferred through interaction.

### What the agent does
It performs one action repeatedly:
- propose a price (`propose_price`).

That simplicity is intentional. Strategy quality comes from *where* and *when* to move the price after each response.

### What the agent is rewarded for
The reward stack combines short-term and terminal feedback:
- small shaping rewards for partial progress (+0.05) and bad moves (-0.05),
- strong terminal incentives for:
  - completing the ride,
  - doing it quickly (efficiency bonus),
  - doing it profitably,
- penalties for:
  - timing out,
  - cancellations,
  - leaving too much money on the table (missed revenue penalty).

Phase 2b further hardens training with format compliance reward, anti-hacking checks, and process-level reward signals.

---

## Results) what changed after training? Show it.

The key story is not “model X scored Y once.”  
It is **capability growth across phases**:

1. **Baseline / early setup**: environment works; simple policies establish starting behavior.
2. **Platform warmstart (Phase 2)**: platform learns against a rule-based simulator.
3. **Strategic simulator training (Phase 3)**: opponent becomes more realistic (can bluff).
4. **Platform adaptation (Phase 4)**: platform is re-trained against this stronger opponent.

### Concrete evidence from recorded runs
Using repository result snapshots:

- `data/results_original.json` (earlier/original configuration):  
  - Hard task completion:
    - Midpoint: **0.00**
    - Adaptive: **0.00**
    - Q-Learning: **0.00**
  - Q-Learning hard avg reward: **-2.331**

- `data/results_alt.json` (improved configuration/training setup):  
  - Hard task completion:
    - Midpoint: **0.70**
    - Adaptive: **0.61**
    - Q-Learning: **0.61**
  - Q-Learning hard avg reward: **39.5532**

### What changed in plain language
The system moved from “hard scenarios almost always fail” to “hard scenarios are frequently solved with high positive reward.”  
That is a qualitative shift in behavior, not just metric noise.

---

## Why does it matter) who would care, and why?

### Who cares
- **Marketplace pricing teams**: better conversion and profitability under volatile conditions.
- **Operations/product teams**: fewer cancellations and faster deal closure can improve rider/driver experience.
- **Applied RL teams**: a realistic benchmark for partially observable, sequential negotiation.
- **Leadership/stakeholders**: a demo-friendly framework that connects model behavior to business outcomes.

### Why it matters
This project reframes dynamic pricing from static multiplier logic to an **interactive decision policy**.  
Instead of only asking “What should price be now?”, it asks “What price move now maximizes the chance of a profitable agreement before patience collapses?”

In high-friction marketplaces, that shift can be the difference between:
- repeated quote churn and cancellation,
- versus consistent, profitable matches.

---

## The implementation journey (phasewise, at a glance)

- **Phase 0:** Deploy environment early; validate `/reset` and `/step`.
- **Phase 1:** Validate setup, load models, establish training package.
- **Phase 2:** Train platform against rule-based simulator (warmstart).
- **Phase 2b:** Add anti-hacking and process-aware reward hardening.
- **Phase 3:** Train simulator LLM to behave strategically.
- **Phase 4:** Re-train platform against strategic simulator (adaptation cycle).
- **Phase 5:** Demo, evaluation table, deployment-ready story.

This is why the project story is credible: it is not a one-shot claim, but a controlled build-up of capabilities with explicit gates and artifacts.
