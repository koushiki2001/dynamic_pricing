# Project Gap Analysis — Current State vs Hackathon Requirements

## What the Hackathon Actually Requires

From the guide, the required stack is:
```
Environment → verifier/reward functions → TRL trainer → Unsloth → deployment on OpenEnv/Spaces
```

The goal is **not** to use an LLM as a fixed policy. The goal is to **train** an LLM with RL so it measurably improves.

---

## Current Project — What Is Complete

### Environment (Production-Ready) ✅
- `reset()`, `step()`, `state()` fully implemented
- 23-field observation space with surge, patience, weather, traffic, context
- Three difficulty levels (easy / medium / hard) with tuned parameters
- Curriculum learning built in through task progression
- FastAPI server, Docker, OpenEnv YAML spec — all deployment-ready

### Scenario Generation ✅
- Context-aware pricing (surge × weather × demand × supply)
- Hidden thresholds with noise baked in
- Pre-generated scenario JSON files for reproducibility

### Reward Design ✅
- Multi-component verifiers: completion + profit + efficiency bonus + missed revenue penalty
- Cancellation penalty (−5.0) and timeout penalty (−2.0)
- Per-step shaping rewards for partial acceptance

### Baseline Policies ✅
- Adaptive binary-search policy
- Static midpoint policy
- LLM API policy (calls external model — not trained)
- Reward-guided LLM policy (experimental)
- Q-learning tabular agent (for benchmarking)

### Testing and Documentation ✅
- Unit tests for environment mechanics and graders
- Comprehensive README
- Submission entry point with required logging format

---

## Critical Gaps vs Hackathon Requirements

### Gap 1 — No LLM Training (Most Important)
The LLM is currently used as a **fixed oracle** via API calls. The hackathon requires training/fine-tuning a model so it demonstrably improves.

| Current | Required |
|---|---|
| `openai_policy.py` → API call to external model | TRL `GRPOTrainer` → model weights update |
| LLM behavior unchanged across episodes | Model improves measurably over training |
| No before/after performance delta | Baseline vs trained model comparison required |

### Gap 2 — No TRL + Unsloth Pipeline
The required training stack is completely absent. `scripts/train.py` does tabular Q-learning — a baseline tool, not the submission agent.

### Gap 3 — No GRPO Training
GRPO is explicitly recommended for verifiable tasks. The reward functions are already verifiers — they just have not been wired into a GRPO training loop.

### Gap 4 — No Measurable Improvement Story
Judges need to see the model improve. Currently there is no:
- Baseline model performance snapshot (pre-training)
- Post-training comparison
- Training curve showing reward progression

### Gap 5 — Simulator Is Not an Agent
Current `simulator.py` is deterministic rules. It cannot bluff, adapt, or learn strategic behavior. This limits the richness of the training signal the platform model receives.

---

## Feasibility Assessment

| Component | Effort | Feasibility | Notes |
|---|---|---|---|
| Environment | Done | ✅ | Already production-ready |
| Reward verifiers | Done | ✅ | `reward.py` already complete |
| Rollout function (env → GRPO bridge) | Low | ✅ | ~50 lines |
| TRL + GRPO training script | Medium | ✅ | Standard TRL pattern |
| Unsloth integration | Low | ✅ | 2–3 lines on top of TRL |
| Small models (Qwen2.5-0.5B + 1.5B) | Low | ✅ | Run on Colab free tier / A100 |
| Simulator LLM (replace rule-based) | Medium | ✅ | New component, well-defined |
| Before/after demo capture | Low | ✅ | Log pre-train and post-train runs |
| HuggingFace Spaces deployment | Low | ✅ | Docker already written |

**The core environment is ~70% of total work. The missing training loop is the smaller remaining piece.**

---

## What Needs to Be Added

1. **GRPO rollout function** — connects `env.reset()/step()` to GRPO's `(prompt, completion, reward)` format
2. **TRL `GRPOTrainer` script** — one per model (platform + simulator)
3. **Unsloth model loading** — wrap both models for efficient training
4. **Simulator LLM** — replace `simulator.py` deterministic logic with LLM agent
5. **Sequential training schedule** — Stage 0 → Stage 1 → Stage 2 pipeline
6. **Before/after logging** — capture metrics at each training stage
7. **Demo interface** — Gradio or simple CLI showing baseline vs trained behavior

---

## The Strongest Hackathon Claim This Project Can Make

> "We built a multi-agent negotiation environment where a platform LLM and a strategic simulator LLM are both trained with GRPO. The platform learns to infer hidden user thresholds from behavioral signals. The simulator learns optimal bluffing strategies. Together they reach a stable Nash equilibrium that models real-world ride-hailing pricing dynamics more accurately than rule-based surge systems."

This satisfies all hackathon criteria:
- Clear environment design with reset/step/reward ✅
- Verifiable objective reward functions ✅
- Evidence of model improvement (before/after) ✅
- Anti-reward-hacking (multi-component verifiers) ✅
- Reproducible deployment (Docker + HuggingFace Spaces) ✅
- Sharp demo showing convergence arc ✅
