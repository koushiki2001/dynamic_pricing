# Phase 5 — Demo, Evaluation, and Deployment

## Goal
Package everything into a presentable demo that shows the before/after improvement arc clearly. Deploy to HuggingFace Spaces via the existing Docker setup. Produce the presentation artifacts.

**Exit criteria:** A live demo runs showing at least two contrasting episodes (untrained vs trained platform), the evaluation table is generated, and the Space is accessible via URL.

---

## Step 5.1 — Demo Script (CLI)

Create `demo.py` at project root. This runs a single episode with visible step-by-step output — suitable for screen-sharing during a presentation.

```python
"""
Live episode demo — shows platform LLM reasoning through a negotiation.
Loads trained or untrained model based on --mode flag.

Usage:
  python demo.py --mode untrained --task easy
  python demo.py --mode trained   --task easy
  python demo.py --mode trained   --task hard
"""

import argparse
import json
from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.models import Observation
from training.model_loader import load_platform_model, load_simulator_model
from training.simulator_agent import SimulatorAgent
from training.prompt_builders import build_platform_prompt, build_simulator_prompt

WEATHER = ["clear", "rain", "storm"]
TRAFFIC = ["low", "medium", "heavy"]

def run_demo_episode(platform_model, platform_tokenizer, simulator_agent, env, task):
    obs_dict = env.reset(task=task)
    obs      = Observation(**obs_dict) if isinstance(obs_dict, dict) else obs_dict
    hidden   = env.get_hidden_state()
    done     = False

    print(f"\n{'='*60}")
    print(f"EPISODE START — Task: {task.upper()}")
    print(f"{'='*60}")
    print(f"Trip:     {obs.distance_km}km, {obs.estimated_duration_min}min")
    print(f"Context:  {WEATHER[obs.weather_condition]}, {TRAFFIC[obs.traffic_level]}, surge={obs.surge_multiplier}x")
    print(f"Quoted:   Rider ${obs.rider_quoted_price:.2f} | Driver ${obs.driver_quoted_price:.2f}")
    print(f"[Hidden]: Rider ceiling ${hidden.rider_max_willingness:.2f} | Driver floor ${hidden.driver_min_willingness:.2f}")
    print(f"Max steps: {obs.max_steps}")
    print()

    total_reward = 0

    while not done:
        # Platform proposes
        prompt = build_platform_prompt(obs)
        inputs = platform_tokenizer(prompt, return_tensors="pt").to(platform_model.device)
        import torch
        with torch.no_grad():
            out = platform_model.generate(**inputs, max_new_tokens=64, temperature=0.7,
                                          do_sample=True, pad_token_id=platform_tokenizer.eos_token_id)
        completion = platform_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            price = float(json.loads(completion.strip()).get("price", 0))
        except Exception:
            import re
            nums = re.findall(r'\d+\.?\d*', completion)
            price = float(nums[0]) if nums else (obs.rider_quoted_price + obs.driver_quoted_price) / 2

        print(f"Step {obs.step_number + 1}/{obs.max_steps}")
        print(f"  Platform proposes: ${price:.2f}")

        # Simulator decides
        decision = simulator_agent.decide(hidden, obs, price)
        print(f"  Rider   → {'ACCEPT ✓' if decision.rider_accept else 'REJECT ✗'}"
              f"  (ceiling ${hidden.rider_max_willingness:.2f},"
              f" {'honest' if price <= hidden.rider_max_willingness else 'forced reject'})")
        print(f"  Driver  → {'ACCEPT ✓' if decision.driver_accept else 'REJECT ✗'}"
              f"  (floor ${hidden.driver_min_willingness:.2f},"
              f" {'honest' if price >= hidden.driver_min_willingness else 'forced reject'})")

        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        result = env.step(action, override_decision={
            "rider": decision.rider_accept, "driver": decision.driver_accept
        })

        total_reward = result.reward
        done = result.done
        obs  = result.observation

        if done:
            info = result.info
            if info.get("ride_completed"):
                print(f"\n  ✅ DEAL CLOSED at ${price:.2f}")
                print(f"  Platform profit: ${info.get('platform_profit', 0):.2f}")
            elif info.get("timed_out"):
                print(f"\n  ⏰ TIMED OUT — no deal reached")
            else:
                print(f"\n  ❌ CANCELLED — party ran out of patience")
        print()

    print(f"{'='*60}")
    print(f"Episode reward: {total_reward:.3f}")
    print(f"{'='*60}\n")
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["untrained", "trained"], default="trained")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    lora = "checkpoints/phase4/platform_v1_lora" if args.mode == "trained" else None
    platform_model, platform_tokenizer = load_platform_model(lora_path=lora)

    sim_lora = "checkpoints/phase3/simulator_lora"
    sim_model, sim_tokenizer = load_simulator_model(lora_path=sim_lora)
    simulator_agent = SimulatorAgent(sim_model, sim_tokenizer)

    env = DynamicPricingEnv()
    rewards = []
    for i in range(args.episodes):
        print(f"\n--- Episode {i+1}/{args.episodes} ({args.mode} platform) ---")
        r = run_demo_episode(platform_model, platform_tokenizer, simulator_agent, env, args.task)
        rewards.append(r)

    print(f"\nSummary ({args.mode}, {args.task}):")
    print(f"  Avg reward:      {sum(rewards)/len(rewards):.3f}")
    print(f"  Completion rate: {sum(1 for r in rewards if r > 0)/len(rewards)*100:.0f}%")
```

---

## Step 5.2 — Generate Presentation Metrics

Run the full evaluation from Phase 4 and produce a clean summary:

```bash
python training/evaluate.py
```

This writes `data/final_evaluation.json` with the 4-stage comparison table.

Additionally generate training curves:

```bash
python scripts/plot_training_curves.py \
  --metrics data/training_metrics_phase2.json \
            data/training_metrics_phase4.json \
  --output  data/training_curves.png
```

Key numbers to highlight in the presentation:
- Completion rate: Stage A → Stage D (the full arc)
- The dip at Stage C (simulator disruption)
- Recovery at Stage D (platform adaptation)
- Avg reward progression per 100 steps

---

## Step 5.3 — Update app.py for Trained Model

The existing `app.py` serves the environment via FastAPI. Update it to optionally load the trained platform model for interactive demo via HTTP.

```python
# In app.py — add at startup
import os
from training.model_loader import load_platform_model
from training.simulator_agent import SimulatorAgent, load_simulator_model

PLATFORM_CKPT = os.getenv("PLATFORM_CKPT", "checkpoints/phase4/platform_v1_lora")
SIM_CKPT      = os.getenv("SIM_CKPT",      "checkpoints/phase3/simulator_lora")

# Load models at startup if checkpoints exist
platform_model, platform_tokenizer = None, None
simulator_agent = None

if os.path.exists(PLATFORM_CKPT):
    platform_model, platform_tokenizer = load_platform_model(lora_path=PLATFORM_CKPT)
    print(f"[APP] Loaded trained platform from {PLATFORM_CKPT}")

if os.path.exists(SIM_CKPT):
    sm, st = load_simulator_model(lora_path=SIM_CKPT)
    simulator_agent = SimulatorAgent(sm, st)
    print(f"[APP] Loaded trained simulator from {SIM_CKPT}")
```

---

## Step 5.4 — HuggingFace Spaces Deployment

The Dockerfile already exists. Deploy:

```bash
# Tag and push to HuggingFace Space
git remote add space https://huggingface.co/spaces/<org>/<space-name>
git push space main
```

Environment variables to set in the Space settings:
```
PLATFORM_CKPT = checkpoints/phase4/platform_v1_lora
SIM_CKPT      = checkpoints/phase3/simulator_lora
PORT          = 7860
```

Upload model checkpoints to HuggingFace Hub (separate repo):
```bash
huggingface-cli upload <org>/dynamic-pricing-checkpoints checkpoints/
```

Then update `model_loader.py` to pull from Hub if local paths don't exist:
```python
from huggingface_hub import snapshot_download
if not Path(lora_path).exists():
    lora_path = snapshot_download(f"<org>/dynamic-pricing-checkpoints/{lora_path}")
```

---

## Step 5.5 — Presentation Story Structure

Use these four clips / screenshots for the presentation:

**Clip 1 — Untrained Platform (Stage A)**
Run `python demo.py --mode untrained --task easy`
Show: random/naive prices, frequent timeouts or cancellations

**Clip 2 — Platform v0 vs Rule-Based (Stage B)**
Run `python demo.py --mode trained_v0 --task easy` (add `--mode trained_v0` flag)
Show: deals close, but platform can easily exploit predictable simulator

**Clip 3 — Platform v0 vs Simulator v1 (Stage C — Disruption)**
Freeze platform_v0, swap in simulator_v1
Show: completion rate drops — simulator is bluffing, platform confused

**Clip 4 — Platform v1 vs Simulator v1 (Stage D — Recovery)**
Run `python demo.py --mode trained --task easy`
Show: platform reads bluffing signals, holds price, deal closes efficiently

The narrative arc: **naive → improving → disrupted → adapted**

---

## Step 5.6 — Key Metrics for Slides

From `data/final_evaluation.json`, pull these numbers:

| Slide | Metric | Source |
|---|---|---|
| "Problem" slide | Current surge pricing = 0 behavioral signals used | `reference/01_current_pricing_systems.md` |
| "Our approach" slide | 3-party negotiation, 2 trained LLMs | `reference/02_multi_agent_architecture.md` |
| "Results" slide | Completion rate A→D, reward curve | `data/final_evaluation.json` |
| "Real-world value" slide | Signal mapping, deployment path | `reference/05_behavioral_signals_deployment.md` |

---

## Deliverables for Phase 5

- [ ] `demo.py` — live episode demo with visible reasoning
- [ ] `data/final_evaluation.json` — 4-stage comparison table
- [ ] Training curves chart
- [ ] `app.py` updated to load trained models at startup
- [ ] Space deployed and accessible at HuggingFace URL
- [ ] Presentation clips recorded (4 episodes showing the arc)
