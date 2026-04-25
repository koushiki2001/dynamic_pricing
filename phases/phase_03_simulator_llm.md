# Phase 3 — Simulator LLM Implementation + Training

## Goal
Replace the rule-based `simulator.py` threshold check with a trained LLM agent that reasons strategically about when to accept honestly vs bluff. The simulator LLM is trained against the frozen `platform_v0` from Phase 2.

**Exit criteria:** Simulator LLM bluffs at least 30% of episodes where it has the patience headroom to do so, and deal collapse rate stays below 40% (it's not bluffing recklessly).

---

## What Changes vs Phase 2

| Component | Phase 2 State | After Phase 3 |
|---|---|---|
| `simulator.py` | Rule-based, unchanged | **Still used as fallback** — simulator LLM wraps it |
| `training/simulator_agent.py` | Does not exist | **New** — LLM-based accept/reject agent |
| `training/train_simulator.py` | Does not exist | **New** — GRPO training for simulator |
| Platform model | `checkpoints/phase2/platform_lora/` | **Frozen** — not updated in this phase |
| Simulator model | None | Saved LoRA adapters after training |

---

## Step 3.1 — Simulator Agent Class

Create `training/simulator_agent.py`.

The simulator agent wraps the simulator LLM and produces accept/reject decisions for both rider and driver. It receives the hidden state (which the platform never sees) and the platform's proposed price.

```python
"""Simulator LLM agent — strategic accept/reject for rider and driver."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Tuple

import torch
from ride_hailing_env.models import HiddenState, Observation
from training.prompt_builders import build_simulator_prompt


@dataclass
class SimulatorDecision:
    rider_accept:  bool
    driver_accept: bool
    raw_output:    str


class SimulatorAgent:
    def __init__(self, model, tokenizer, max_new_tokens: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def decide(
        self,
        hidden: HiddenState,
        obs: Observation,
        proposed_price: float,
    ) -> SimulatorDecision:
        """
        Given hidden state and platform proposal, return accept/reject for each party.
        Falls back to honest threshold check if parsing fails.
        """
        prompt = build_simulator_prompt(hidden, obs, proposed_price)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        rider_accept, driver_accept = self._parse_decision(raw, hidden, proposed_price)

        return SimulatorDecision(
            rider_accept=rider_accept,
            driver_accept=driver_accept,
            raw_output=raw,
        )

    def _parse_decision(
        self,
        text: str,
        hidden: HiddenState,
        price: float,
    ) -> Tuple[bool, bool]:
        """
        Parse JSON output from simulator LLM.
        Falls back to deterministic threshold check if parsing fails.
        """
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                rider_accept  = str(parsed.get("rider",  "reject")).lower().strip() == "accept"
                driver_accept = str(parsed.get("driver", "reject")).lower().strip() == "accept"
                return rider_accept, driver_accept
        except Exception:
            pass

        # Fallback: honest threshold check (same as rule-based simulator.py)
        return (
            price <= hidden.rider_max_willingness,
            price >= hidden.driver_min_willingness,
        )
```

---

## Step 3.2 — Patch Environment to Accept External Decisions

The current `environment.py` calls `Simulator.simulate_step()` internally. We need to allow the simulator LLM's decision to override this.

Add an optional parameter to `DynamicPricingEnv.step()`:

```python
# In ride_hailing_env/environment.py — add optional override
def step(
    self,
    action: dict,
    override_decision: dict = None   # NEW — {"rider": bool, "driver": bool}
) -> StepResult:
    ...
    if override_decision is not None:
        # Use LLM simulator decision instead of rule-based check
        rider_accepted  = override_decision["rider"]
        driver_accepted = override_decision["driver"]
        # Still apply patience decay logic from simulator.py
        sim_result = self._simulator.simulate_step_with_decisions(
            self._obs, self._hidden, proposed_price,
            rider_accepted, driver_accepted
        )
    else:
        # Original rule-based path — unchanged
        sim_result = self._simulator.simulate_step(
            self._obs, self._hidden, proposed_price
        )
    ...
```

Also add `simulate_step_with_decisions()` to `simulator.py` — accepts pre-determined accept/reject booleans but still runs patience decay and cancellation logic.

> **Important:** This change is additive — existing code calling `env.step(action)` without `override_decision` continues to work exactly as before. No existing tests break.

---

## Step 3.3 — Simulator Reward Function

The simulator agent is rewarded on **surplus earned** — how much better than its floor/ceiling the final price was.

```python
def compute_simulator_reward(
    proposed_price: float,
    hidden: HiddenState,
    ride_completed: bool,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    Simulator reward = rider surplus + driver surplus if ride completed.
    Penalizes deal collapse to prevent reckless bluffing.
    """
    if ride_completed:
        rider_surplus  = max(0.0, hidden.rider_max_willingness - proposed_price)
        driver_surplus = max(0.0, proposed_price - hidden.driver_min_willingness)
        # Slight efficiency bonus — don't drag out negotiations
        efficiency = min(max_steps / max(steps_taken, 1), 2.0)
        return (rider_surplus + driver_surplus) * efficiency

    # Deal collapsed or timed out — both parties lose
    # Stronger penalty than timeout to discourage reckless bluffing
    return -3.0
```

Add this to `ride_hailing_env/reward.py`.

---

## Step 3.4 — Rollout with Simulator LLM

Update `training/rollout.py` to add `collect_multiagent_rollout()` — runs an episode with both LLM agents active.

```python
def collect_multiagent_rollout(
    platform_model,
    platform_tokenizer,
    simulator_agent: SimulatorAgent,
    env: DynamicPricingEnv,
    task: str = "easy",
) -> dict:
    """
    Run one episode with both platform LLM and simulator LLM active.
    Returns separate sample lists for each model.
    """
    obs_dict = env.reset(task=task)
    obs      = Observation(**obs_dict) if isinstance(obs_dict, dict) else obs_dict
    hidden   = env.get_hidden_state()   # expose hidden state for simulator

    platform_samples  = []
    simulator_samples = []
    done = False
    last_price = (obs.rider_quoted_price + obs.driver_quoted_price) / 2.0

    while not done:
        # Platform proposes price
        platform_prompt = build_platform_prompt(obs)
        inputs = platform_tokenizer(platform_prompt, return_tensors="pt").to(platform_model.device)
        with torch.no_grad():
            out = platform_model.generate(**inputs, max_new_tokens=64, temperature=0.7,
                                          do_sample=True, pad_token_id=platform_tokenizer.eos_token_id)
        platform_completion = platform_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        price = parse_price_from_output(platform_completion, fallback=last_price)
        price = max(1.0, min(500.0, price))
        last_price = price

        # Simulator decides accept/reject
        sim_prompt   = build_simulator_prompt(hidden, obs, price)
        sim_decision = simulator_agent.decide(hidden, obs, price)

        # Step environment with simulator's decision
        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        result = env.step(action, override_decision={
            "rider":  sim_decision.rider_accept,
            "driver": sim_decision.driver_accept,
        })

        platform_samples.append({
            "prompt":     platform_prompt,
            "completion": platform_completion,
            "reward":     result.reward,
            "done":       result.done,
        })
        simulator_samples.append({
            "prompt":     sim_prompt,
            "completion": sim_decision.raw_output,
            "reward":     None,   # filled after episode ends
            "price":      price,
            "done":       result.done,
        })

        obs  = result.observation
        done = result.done

    # Assign simulator reward at terminal step
    if simulator_samples:
        final_price = simulator_samples[-1]["price"]
        sim_reward  = compute_simulator_reward(
            final_price, hidden,
            ride_completed=result.info.get("ride_completed", False),
            steps_taken=obs.step_number,
            max_steps=obs.max_steps,
        )
        simulator_samples[-1]["reward"] = sim_reward

    return {
        "platform_samples":  platform_samples,
        "simulator_samples": simulator_samples,
    }
```

> This requires exposing `env.get_hidden_state()` — add a simple getter to `DynamicPricingEnv` that returns `self._hidden`.

---

## Step 3.5 — Simulator Training Script

Create `training/train_simulator.py`.

```python
"""
Phase 3: Train simulator LLM with GRPO against frozen platform_v0.

Usage:
  python training/train_simulator.py --task easy --steps 1000 --platform_ckpt checkpoints/phase2/platform_lora
"""

import argparse
from pathlib import Path
from unsloth import FastLanguageModel
from training.model_loader import load_platform_model, load_simulator_model
from training.simulator_agent import SimulatorAgent
from training.rollout import collect_multiagent_rollout
from ride_hailing_env.environment import DynamicPricingEnv


def train(task: str, num_steps: int, platform_ckpt: str, output_dir: str):
    print(f"[TRAIN-SIM] task={task}, steps={num_steps}")

    # Load frozen platform (no gradient updates)
    platform_model, platform_tokenizer = load_platform_model(lora_path=platform_ckpt)
    platform_model.eval()   # freeze — no training on platform in this phase

    # Load simulator model for training
    sim_model, sim_tokenizer = load_simulator_model()
    simulator_agent = SimulatorAgent(sim_model, sim_tokenizer)

    env = DynamicPricingEnv()

    # Baseline: how does simulator (untrained) perform against platform_v0?
    print("[EVAL] Measuring untrained simulator performance...")
    baseline_sim_rewards = []
    for _ in range(20):
        result = collect_multiagent_rollout(platform_model, platform_tokenizer,
                                            simulator_agent, env, task)
        sim_samples = result["simulator_samples"]
        if sim_samples and sim_samples[-1]["reward"] is not None:
            baseline_sim_rewards.append(sim_samples[-1]["reward"])
    baseline_avg = sum(baseline_sim_rewards) / len(baseline_sim_rewards) if baseline_sim_rewards else 0
    print(f"[EVAL] Simulator baseline avg reward: {baseline_avg:.3f}")

    # Training loop
    step = 0
    reward_history = []

    while step < num_steps:
        batch_sim_rewards = []
        for _ in range(8):   # batch of 8 episodes
            result = collect_multiagent_rollout(platform_model, platform_tokenizer,
                                                simulator_agent, env, task)
            sim_samples = result["simulator_samples"]
            if sim_samples and sim_samples[-1]["reward"] is not None:
                batch_sim_rewards.append(sim_samples[-1]["reward"])

        # GRPO update on simulator using collected (prompt, completion, reward)
        # [GRPOTrainer integration here — same pattern as train_platform.py]

        reward_history.extend(batch_sim_rewards)
        avg = sum(batch_sim_rewards) / len(batch_sim_rewards) if batch_sim_rewards else 0

        if step % 50 == 0:
            print(f"[SIM STEP {step}] avg_sim_reward={avg:.3f}")

        step += 8

    # Save
    save_path = Path(output_dir) / "simulator_lora"
    sim_model.save_pretrained(str(save_path))
    sim_tokenizer.save_pretrained(str(save_path))
    print(f"[SAVE] Simulator LoRA saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           default="easy")
    parser.add_argument("--steps",          type=int, default=1000)
    parser.add_argument("--platform_ckpt",  default="checkpoints/phase2/platform_lora")
    parser.add_argument("--output_dir",     default="checkpoints/phase3")
    args = parser.parse_args()
    train(args.task, args.steps, args.platform_ckpt, args.output_dir)
```

---

## Step 3.6 — What to Monitor

| Signal | Expected after training | Problem if... |
|---|---|---|
| Simulator avg reward | Increases from baseline | Stays flat → model not learning bluffing |
| Deal collapse rate | 20–40% (strategic bluffing causes some collapse) | > 60% → bluffing too recklessly |
| Bluff frequency | ~30–50% of episodes | < 10% → model defaulting to always-honest |
| Platform completion rate | **Should drop** vs Phase 2 | Doesn't drop → simulator not actually bluffing |

The platform completion rate **dropping** is a healthy sign here — it means the simulator is genuinely harder to deal with. This is the "disruption" moment in the demo story.

---

## Deliverables for Phase 3

- [ ] `training/simulator_agent.py` — LLM wrapper with fallback parsing
- [ ] `ride_hailing_env/environment.py` — `override_decision` parameter added (backward compatible)
- [ ] `ride_hailing_env/simulator.py` — `simulate_step_with_decisions()` added
- [ ] `ride_hailing_env/reward.py` — `compute_simulator_reward()` added
- [ ] `training/rollout.py` — `collect_multiagent_rollout()` added
- [ ] `training/train_simulator.py` — training script running
- [ ] Simulator LoRA saved to `checkpoints/phase3/simulator_lora/`
- [ ] Bluff frequency observable in training logs
