# Multi-Agent Architecture — Platform LLM + Simulator LLM

## Core Idea

Instead of one AI agent (the platform) acting against a rule-based simulator, both the platform and the rider/driver parties are represented by independently trained LLM agents. Each has its own observation space, reward signal, and policy.

---

## The Two Agents

### Platform LLM
- **Role:** Proposes prices each round to close the deal
- **Observes:** Public quotes, patience signals, context (surge, weather, traffic), prior round feedback — no hidden information
- **Does NOT know:** Rider's true max willingness, driver's true min willingness
- **Reward:** Platform profit × efficiency bonus − missed revenue penalty
- **Must learn:** Infer hidden thresholds from behavioral signals; converge to optimal price fast

### Simulator LLM
- **Role:** Plays both rider and driver simultaneously
- **Observes:** Hidden thresholds (true max/min willingness), current patience, proposed price, step number
- **Knows everything** the platform does not
- **Reward:** Rider surplus (money saved) + Driver surplus (money earned above floor)
- **Must learn:** When bluffing (rejecting an acceptable price) is rational vs reckless

---

## The Key Difference from Rule-Based Simulator

| | Rule-Based (current) | Simulator LLM (new) |
|---|---|---|
| Accept/reject logic | Deterministic threshold check | Learned strategic decision |
| Bluffing | Never bluffs | Learns optimal bluff thresholds |
| Adaptation | Fixed rules regardless of platform strategy | Adapts to platform's pricing patterns |
| Training signal for platform | Exploitable (static opponent) | Forces generalizable strategies |

---

## Agent Objectives and the Cooperative-Competitive Dynamic

The simulator is **not adversarial** — it wants the deal to complete, but on favorable terms.

```
Rider reward  = max(0, rider_max_willingness − price_paid)     if ride_completed
              = −patience_cost                                  if deal collapses

Driver reward = max(0, price_received − driver_min_willingness) if ride_completed
              = −patience_cost                                  if deal collapses
```

Both parties lose if the deal collapses. This creates a **cooperative-competitive** game:
- Cooperative: everyone wants a completed ride
- Competitive: each party wants better terms within the deal

---

## The Internal Conflict Within the Simulator Agent

Rider wants price **low**. Driver wants price **high**. They are a single agent with opposing sub-objectives.

Resolution: The simulator agent learns to bluff on behalf of whichever party has more patience headroom and more to gain. It naturally learns that coordinated bluffing (both reject simultaneously) is rarely rational — it risks deal collapse for both.

---

## Training Approach — Sequential Cycling

Both models are publicly available small LLMs (e.g., Qwen2.5-0.5B for simulator, Qwen2.5-1.5B for platform). They are trained with GRPO via TRL + Unsloth.

Non-stationarity (each agent's environment shifts as the other trains) is managed by **freezing one while training the other**.

```
Stage 0 (~500 episodes):
  Train platform LLM against rule-based simulator
  Goal: warm start — establish non-zero reward before introducing strategic opponent

Stage 1 (~1000 episodes):
  Freeze platform_v0
  Train simulator LLM against frozen platform_v0
  Goal: simulator learns optimal bluffing strategy

Stage 2 (~1000 episodes):
  Freeze simulator_v1
  Train platform LLM against frozen simulator_v1
  Goal: platform adapts to strategic bluffing opponent

Stage 3 (~500 episodes):
  Optional: fine-tune simulator against smarter platform_v2
```

---

## Why Two Different Model Sizes

The simulator only needs to learn a binary strategic decision (bluff vs honest) conditioned on patience and surplus. This is a simpler task — a 0.5B model is sufficient.

The platform must reason across 23 observation fields, infer hidden thresholds, and plan over multiple steps under uncertainty. This needs more capacity — 1.5B minimum.

Using different sizes also reduces total VRAM pressure during training.

---

## Emergent Behaviors Expected After Full Training

**Simulator learns:**
- Only bluff when patience headroom is large AND price is well above personal threshold
- Never bluff in the final step — too risky
- Rider and driver bluff decisions are made independently based on their own patience

**Platform learns:**
- Consistent "willing" mood + patience decay pattern = bluff, hold price rather than chasing
- Massive patience drop in one step = near floor or noise hit, respond by moving price
- Surge + urgency context = rider ceiling is likely above their public quote

**The emergent equilibrium:**
Platform converges near the true overlap zone without excessive probing. Simulator only bluffs when the expected gain clearly outweighs the collapse risk. Both strategies stabilize across training cycles.

---

## The Demo Story This Enables

```
Before training:
  Platform (base model): naive midpoints, frequent timeouts
  Simulator (base model): always honest, easy to exploit

After Stage 0:
  Platform closes ~60% of deals — learned basic convergence

After Stage 1:
  Simulator bluffs → platform completion drops to ~45%
  Demonstrates simulator is genuinely strategic

After Stage 2:
  Platform reads bluffing signals → completion recovers to ~70%+
  Profit per completed ride also increases

This arc — improvement, disruption, re-adaptation — directly demonstrates the RL loop working.
```
