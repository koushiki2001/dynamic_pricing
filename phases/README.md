# Implementation Phases

Sequential execution plan for the full multi-agent RL training system.

## Files

| Phase | File | Goal | Exit Criteria |
|---|---|---|---|
| **0** | [phase_00_early_deployment.md](phase_00_early_deployment.md) | Deploy environment to HuggingFace Spaces BEFORE training | Remote `/reset` and `/step` respond correctly |
| **1** | [phase_01_setup.md](phase_01_setup.md) | Validate env, install TRL+Unsloth, load both Qwen models, create `training/` package | Both models load, all existing tests pass |
| **2** | [phase_02_platform_warmstart.md](phase_02_platform_warmstart.md) | Train platform LLM with GRPO vs rule-based simulator | Completion rate >50% on easy task |
| **2b** | [phase_02b_reward_hardening.md](phase_02b_reward_hardening.md) | Add format compliance reward, anti-hacking checks, process supervision, generation inspection | No hacking patterns in 50 inspected episodes |
| **3** | [phase_03_simulator_llm.md](phase_03_simulator_llm.md) | Build simulator LLM agent, train it vs frozen platform_v0 | Simulator bluffing 20–50% of eligible episodes |
| **4** | [phase_04_multiagent_training.md](phase_04_multiagent_training.md) | Re-train platform vs frozen simulator_v1 — full adaptation cycle with stability gates | Platform v1 completion rate ≥70% vs strategic simulator |
| **5** | [phase_05_demo_deployment.md](phase_05_demo_deployment.md) | Demo script, evaluation table, presentation artifacts | Live demo showing 4-stage arc, evaluation table generated |

---

## Dependency Chain

```
Phase 0 → Phase 1 → Phase 2 → Phase 2b → Phase 3 → Phase 4 → Phase 5
   ↑                               ↑                    ↑
Deploy env                  Harden rewards         Scale gate
BEFORE training             BEFORE simulator       BEFORE medium/hard
```

Phase 0 is not optional — the guide explicitly requires early deployment before training.
Phase 2b is not optional — reward hacking can silently ruin training without it.

---

## Hackathon Guide Coverage

Every recommendation from the guide is now covered:

| Guide Point | Covered In |
|---|---|
| 7 — Multi-component rewards | Phase 2b (format compliance + anti-cheat + process reward) |
| 8 — Reward hacking protection | Phase 2b (anti_hack.py, drift detection, rollback, generation inspection) |
| 9 — Process-aware feedback | Phase 2b (`compute_process_reward()` step-level verifiers) |
| 10 — TRL + Unsloth + OpenEnv | Phase 1 (setup), Phase 2 (training) |
| 11 — GRPO for verifiable tasks | Phase 2 and 4 (GRPOTrainer with reward.py as verifier) |
| 12 — Keep inference fast | Phase 2b (FastLanguageModel.for_inference, max_new_tokens=32, torch.inference_mode) |
| 13 — Deploy environment early | Phase 0 (deploy BEFORE training, not after) |
| 14 — Scale after env is stable | Phase 0 (stability gate checklist), Phase 4 (scale gate before medium/hard) |
| 15 — Monitor right things | Phase 4 (8-column monitoring table, inspect_generations every 100 steps) |
| 16 — Save models correctly | Phase 2 (Unsloth save_pretrained, test inference immediately after save) |
| 17 — Team structure | See Team Roles section below |
| 18 — 1-day execution plan | Covered across all phases in order |

---

## Team Roles (from Hackathon Guide)

Assign one owner per track. Tracks can work in parallel once Phase 0 is done.

| Role | Owner | Phases | Responsibilities |
|---|---|---|---|
| **Environment** | — | 0, 1 | Deployment, stability verification, `override_decision` patch, `get_hidden_state()` |
| **Rewards / Verifiers** | — | 2b, 3 | `anti_hack.py`, `compute_process_reward()`, simulator reward, anti-cheat checks |
| **Training** | — | 2, 3, 4 | GRPO training scripts, rollout collection, metrics logging, drift detection |
| **Demo / Product** | — | 5 | `demo.py`, evaluation table, presentation clips, Space demo polish |

Tracks 2, 3, 4 are sequential and owned by Training. Tracks Environment and Rewards/Verifiers can work ahead of Training to prepare their components.

---

## Checkpoint Outputs

```
checkpoints/
├── last_stable/platform_lora/    ← rolling checkpoint every 200 steps (rollback target)
├── phase2/platform_lora/         ← platform v0, trained vs rule-based
├── phase3/simulator_lora/        ← simulator v1, trained vs platform v0
└── phase4/platform_v1_lora/      ← platform v1, trained vs simulator v1 (FINAL MODEL)

data/
├── baseline_snapshot_phase2.json      ← untrained platform metrics (the "before")
├── training_metrics_phase2.json       ← phase 2 reward curve
├── training_metrics_phase3.json       ← phase 3 simulator reward curve
├── training_metrics_phase4.json       ← phase 4 platform reward curve (8 columns)
└── final_evaluation.json              ← 4-stage comparison table (the demo centrepiece)
```

---

## New Files Created Across All Phases

```
training/
├── __init__.py
├── prompt_builders.py      # Phase 1  — platform + simulator prompt builders
├── model_loader.py         # Phase 1  — Unsloth model loading with optional LoRA path
├── anti_hack.py            # Phase 2b — anti-reward-hacking checks
├── rollout.py              # Phase 2  — GRPO episode collection + format + process reward
├── simulator_agent.py      # Phase 3  — LLM-based accept/reject agent
├── train_platform.py       # Phase 2+4 — platform GRPO training with monitoring
├── train_simulator.py      # Phase 3  — simulator GRPO training
└── evaluate.py             # Phase 5  — full 4-stage comparison evaluation

demo.py                         # Phase 5 — live presentation demo
scripts/verify_models.py        # Phase 1 — model load sanity check
```

## Files Modified (Existing Codebase)

```
ride_hailing_env/environment.py   # Phase 3: override_decision + get_hidden_state()
ride_hailing_env/simulator.py     # Phase 3: simulate_step_with_decisions()
ride_hailing_env/reward.py        # Phase 2b+3: compute_process_reward() + compute_simulator_reward()
requirements.txt                  # Phase 1: trl, unsloth, transformers, bitsandbytes, accelerate
app.py                            # Phase 5: load trained models at startup
client.py                         # Phase 0: reads SPACE_URL env var
.env                              # Phase 0: SPACE_URL added
```

## Do Not Touch

```
ride_hailing_env/models.py             # Pydantic models stable — no changes needed
ride_hailing_env/scenario_generator.py
ride_hailing_env/config.py
baselines/                             # Kept as-is for comparison baseline
```
