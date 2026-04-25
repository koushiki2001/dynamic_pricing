# RL Training Approach — GRPO, TRL, Unsloth

## Why Standard Fine-Tuning Is Not Enough

The platform agent cannot be trained with supervised fine-tuning (SFT) alone because:
- There is no dataset of "correct" prices — the optimal price depends on the hidden state, which the platform never observes
- The task requires sequential decision-making across multiple steps
- Good behavior (efficient deal closure) only emerges from exploring and receiving feedback

RL is the right tool here because outcomes are verifiable (deal completed or not, profit earned) and the platform must learn from those outcomes across many episodes.

---

## The Training Stack

| Component | Role |
|---|---|
| **TRL (GRPOTrainer)** | RL training algorithm — updates model weights based on rewards |
| **Unsloth** | Memory-efficient model loading — reduces VRAM, speeds up training |
| **OpenEnv environment** | Provides the episode loop: reset(), step(), reward |
| **Reward functions** | Verifiers — reward.py already implements these |

---

## GRPO — Why This Algorithm

GRPO (Group Relative Policy Optimization) is preferred over PPO for this task because:

- **No value model needed** — PPO requires a separate critic network; GRPO eliminates this, halving memory use
- **Designed for verifiable rewards** — works directly with outcome-based signals (did the ride complete? what was the profit?)
- **Sample efficient** — generates multiple rollouts per prompt, ranks them, updates toward better ones

The key idea: for each pricing scenario, sample multiple price proposals from the model, execute them in the environment, score each by reward, then update the model to make higher-reward proposals more likely.

---

## What a GRPO Training Sample Looks Like

```python
# Platform GRPO sample
{
  "prompt": """
    Scenario: 12km trip, surge 1.8x, light rain, evening
    Rider quoted: $22 | Driver quoted: $16 | Gap: $6
    Rider patience: 0.62 (hesitant) | Driver patience: 0.85 (willing)
    Last round: You proposed $18.50
      Rider: REJECTED | Driver: ACCEPTED
    Step 2 of 5. Propose next price:
  """,
  "completion": "I propose $19.50",      # sampled from model
  "reward": 2.34                          # computed by reward.py after env.step()
}

# Simulator GRPO sample
{
  "prompt": """
    Hidden state:
      Rider true max: $20.00, patience: 0.62, decay: 0.16/rejection
      Driver true min: $16.50, patience: 0.85, decay: 0.12/rejection
    Platform proposed: $19.50. Step 2/5.
    Rider: $19.50 ≤ $20.00 — CAN accept (surplus $0.50)
    Driver: $19.50 ≥ $16.50 — CAN accept (surplus $3.00)
    Bluff or accept honestly?
  """,
  "completion": "Rider: accept. Driver: accept.",
  "reward": 3.50    # rider_surplus + driver_surplus
}
```

---

## The Rollout Function — Connecting Environment to GRPO

The rollout function bridges the OpenEnv step() loop and the GRPO trainer:

```python
def collect_rollout(platform_model, simulator_model, env, task="easy"):
    obs = env.reset(task=task)
    trajectory = []

    while not done:
        # Platform generates price proposal
        platform_prompt = build_platform_prompt(obs)
        price = platform_model.generate(platform_prompt)

        # Simulator decides accept/reject
        sim_prompt = build_simulator_prompt(hidden_state, price, obs)
        decision = simulator_model.generate(sim_prompt)

        # Execute in environment
        result = env.step(price, override_decision=decision)

        trajectory.append({
            "platform_prompt": platform_prompt,
            "platform_completion": str(price),
            "simulator_prompt": sim_prompt,
            "simulator_completion": decision,
            "reward": result.reward,
            "done": result.done
        })

        obs = result.observation
        done = result.done

    return trajectory
```

---

## Curriculum — Train Easy First

Per hackathon guide recommendation: make success possible before scaling difficulty.

```
Phase 1: Train on easy task only
  Target: platform reaches >50% completion rate
  Why: establishes non-zero reward — GRPO needs successful trajectories to learn from

Phase 2: Add medium task
  Target: completion rate >40% on medium
  
Phase 3: Add hard task
  Target: any non-zero reward on hard
  
Phase 4: Mixed curriculum (easy + medium + hard in each batch)
  Target: stable performance across all three
```

If the model gets zero reward, training stalls — GRPO cannot distinguish good from bad if all outcomes are equally negative.

---

## Reward Functions as Verifiers

The existing reward.py already implements multi-component verification, which is exactly what the hackathon guide recommends over single-signal rewards:

| Component | What it verifies | Anti-hacking role |
|---|---|---|
| `ride_completed` | Binary success check | Must actually close the deal |
| `platform_profit` | Revenue earned | Cannot earn reward with $0 prices |
| `efficiency_bonus` | Steps used vs max | Cannot stall and still get full reward |
| `missed_revenue_penalty` | Left money on table | Cannot just propose max price always |
| `cancellation_penalty` (-5.0) | Drove party away | Punishes aggressive pricing |
| `timeout_penalty` (-2.0) | Never converged | Punishes passive / random pricing |

Multiple independent verifiers make reward hacking significantly harder.

---

## Model Saving — Critical Warning

From hackathon guide: **do not upcast a 4-bit model to 16-bit and naively merge LoRA weights** — this degrades quality severely.

Correct approach with Unsloth:
```python
# Save LoRA adapters only (safe, small files)
model.save_pretrained("platform_lora_adapters")

# OR merge correctly using Unsloth's built-in path
model.save_pretrained_merged("platform_merged", tokenizer, save_method="merged_16bit")
```

Test inference immediately after saving — do not leave export until the end.

---

## Training Schedule Summary

```
Total compute estimate (2× Qwen2.5 models, A100 or 2× T4):

Stage 0: ~500 episodes × easy task         → ~1–2 hours
Stage 1: ~1000 episodes × simulator train  → ~2–3 hours
Stage 2: ~1000 episodes × platform retrain → ~2–3 hours
Stage 3: ~500 episodes × optional cycle    → ~1–2 hours

Total: ~6–10 GPU hours for a full training cycle
Colab A100 (pay-per-use): ~$5–15 for full run
```
