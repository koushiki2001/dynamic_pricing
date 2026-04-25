# Implementation Phases

Sequential execution plan for the full multi-agent RL training system.

## Files

| Phase | File | Goal | Exit Criteria |
|---|---|---|---|
| 1 | [phase_01_setup.md](phase_01_setup.md) | Validate env, install TRL+Unsloth, load both Qwen models, create `training/` package | Both models load, all existing tests pass |
| 2 | [phase_02_platform_warmstart.md](phase_02_platform_warmstart.md) | Train platform LLM with GRPO vs rule-based simulator | Completion rate >50% on easy task |
| 3 | [phase_03_simulator_llm.md](phase_03_simulator_llm.md) | Build simulator LLM agent, train it vs frozen platform_v0 | Simulator bluffing 30%+ of eligible episodes |
| 4 | [phase_04_multiagent_training.md](phase_04_multiagent_training.md) | Re-train platform vs frozen simulator_v1 — full adaptation cycle | Platform v1 completion rate ≥70% vs strategic simulator |
| 5 | [phase_05_demo_deployment.md](phase_05_demo_deployment.md) | Demo script, evaluation table, HuggingFace Spaces deployment | Live demo accessible, 4-stage comparison table generated |

## Dependency Chain

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  (must complete in order — each phase depends on previous outputs)
```

## Checkpoint Outputs

```
checkpoints/
├── phase2/platform_lora/        ← output of Phase 2
├── phase3/simulator_lora/       ← output of Phase 3
└── phase4/platform_v1_lora/     ← output of Phase 4 (final model)

data/
├── baseline_snapshot_phase2.json
├── training_metrics_phase2.json
├── training_metrics_phase3.json
├── training_metrics_phase4.json
└── final_evaluation.json        ← presentation comparison table
```

## New Files Created Across All Phases

```
training/
├── __init__.py
├── prompt_builders.py      # Phase 1 — platform + simulator prompt builders
├── model_loader.py         # Phase 1 — Unsloth model loading
├── rollout.py              # Phase 2 — GRPO episode collection
├── simulator_agent.py      # Phase 3 — LLM-based accept/reject agent
├── train_platform.py       # Phase 2+4 — platform GRPO training
├── train_simulator.py      # Phase 3 — simulator GRPO training
└── evaluate.py             # Phase 5 — full comparison evaluation

demo.py                     # Phase 5 — live presentation demo
scripts/verify_models.py    # Phase 1 — model load sanity check
```

## Files Modified (Existing Codebase)

```
ride_hailing_env/environment.py   # Phase 3: add override_decision parameter + get_hidden_state()
ride_hailing_env/simulator.py     # Phase 3: add simulate_step_with_decisions()
ride_hailing_env/reward.py        # Phase 3: add compute_simulator_reward()
requirements.txt                  # Phase 1: add trl, unsloth, transformers, etc.
app.py                            # Phase 5: load trained models at startup
```

## Do Not Touch

```
ride_hailing_env/models.py           # Pydantic models are stable — no changes needed
ride_hailing_env/scenario_generator.py
ride_hailing_env/config.py
baselines/                           # Kept as-is for comparison baseline
```
