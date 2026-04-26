---
title: Revolutionizing Ride-Hailing Pricing with Multi-Agent AI
emoji: 🚕
colorFrom: green
colorTo: blue
---

# Fairer, Faster, More Profitable Ride Matching

## The Real-World Problem
Imagine: Rainy rush hour. Rider **Priya** quotes $12, driver **Raj** wants $18. Platform **UberX** surge-prices to $20 — **Raj accepts, Priya cancels**. Lost ride, frustrated users, empty car.

**Current systems fail** because:
- Surge multipliers are **blunt** — punish riders, don't negotiate
- No modeling of **individual patience/urgency** 
- **No strategic adaptation** — drivers learn to hold out, riders walk away
- Platforms lose **$billions** in unmatched supply-demand

**Who benefits?** Platforms (higher profit/ride), riders/drivers (fairer prices, fewer cancellations).

## The AI Negotiation System

```mermaid
graph TD
    A[🚗 Rider Priya&lt;br/&gt;Max willing: $16 hidden] --> B[Platform AI Agent&lt;br/&gt;Proposes $14]
    C[🚙 Driver Raj&lt;br/&gt;Min willing: $14 hidden] --> B
    B --> D{Rider accept?}
    B --> E{Driver accept?}
    D -->|Yes| F[Patience OK → Continue]
    D -->|No| G[Rider patience ↓]
    E -->|Yes| F
    E -->|No| H[Driver patience ↓]
    G -->|0| I[🚫 Cancelled -5 reward]
    H -->|0| I
    F --> J[Both Yes → ✅ Ride +profit bonus]
```

**Entities:**
- **Platform Agent** (Qwen2.5-1.5B LoRA): Learns to propose prices balancing speed +
