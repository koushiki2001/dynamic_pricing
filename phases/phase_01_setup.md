# Phase 1 — Environment Validation + Model Setup

## Goal
Verify the existing environment works end-to-end, install the training stack, and confirm both LLM models load and generate output before writing any training code.

**Exit criteria:** A full episode runs with a dummy LLM policy, both Qwen models load under Unsloth, and the reward pipeline produces non-zero values.

---

## Step 1.1 — Validate Existing Environment

Run the existing quick test to confirm the environment is stable before touching anything.

```bash
python scripts/quick_test.py
python -m pytest tests/ -v
```

Confirm:
- `env.reset()` returns a valid `Observation` object
- `env.step(action)` returns `StepResult` with correct reward types
- All 3 tasks (easy / medium / hard) reset and step without errors
- Graders produce scores in (0, 1)

**If any test fails — fix before proceeding. Do not start Phase 2 with a broken environment.**

---

## Step 1.2 — Install Training Dependencies

Add to `requirements.txt`:

```
trl>=0.12.0
unsloth>=2024.12
transformers>=4.46.0
torch>=2.1.0
datasets>=2.18.0
accelerate>=0.26.0
bitsandbytes>=0.43.0
```

Install:
```bash
pip install trl unsloth transformers datasets accelerate bitsandbytes
```

> On Colab / cloud GPU: use `unsloth[colab-new]` or `unsloth[cu121]` depending on CUDA version.

---

## Step 1.3 — Model Selection

| Role | Model | Size | VRAM (4-bit) | Why |
|---|---|---|---|---|
| Platform LLM | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3 GB | Needs richer reasoning across 23 obs fields |
| Simulator LLM | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~1.5 GB | Simpler binary strategy (bluff/honest) |

Total VRAM needed: ~4.5 GB active + ~2 GB overhead = **~6–7 GB** (fits on T4 or better)

Both models are instruction-tuned — they can follow structured prompts without SFT warm-up.

---

## Step 1.4 — Verify Model Loading

Create `scripts/verify_models.py`:

```python
from unsloth import FastLanguageModel

def load_platform_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
    )
    return model, tokenizer

def load_simulator_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length=512,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
    )
    return model, tokenizer

if __name__ == "__main__":
    print("Loading platform model...")
    pm, pt = load_platform_model()
    print("Platform model loaded.")

    print("Loading simulator model...")
    sm, st = load_simulator_model()
    print("Simulator model loaded.")

    # Quick generation test
    inputs = pt("What price should I propose?", return_tensors="pt")
    out = pm.generate(**inputs, max_new_tokens=20)
    print("Platform generation test:", pt.decode(out[0]))
```

Run: `python scripts/verify_models.py`

---

## Step 1.5 — Create Prompt Builders

Create `training/prompt_builders.py`. These convert environment observations into text prompts for each model.

### Platform Prompt Builder

Takes `Observation` (existing Pydantic model) → string prompt.
Reuses most of the logic already in `baselines/openai_policy.py:_build_prompt()`.

```python
from ride_hailing_env.models import Observation
from typing import Optional

WEATHER = ["clear", "rain", "storm"]
TRAFFIC = ["low", "medium", "heavy"]
DEMAND  = ["low", "medium", "high"]
SUPPLY  = ["low", "medium", "high"]
TIME    = ["morning", "afternoon", "evening", "night"]

def build_platform_prompt(obs: Observation) -> str:
    lines = [
        "You are a ride-hailing platform. Propose a price both rider and driver will accept.",
        "",
        f"Rider quoted: ${obs.rider_quoted_price:.2f}",
        f"Driver quoted: ${obs.driver_quoted_price:.2f}",
        f"Gap: ${obs.price_gap:.2f}",
        "",
        f"Trip: {obs.distance_km}km, {obs.estimated_duration_min}min, ETA {obs.pickup_eta_min}min",
        f"Context: weather={WEATHER[obs.weather_condition]}, traffic={TRAFFIC[obs.traffic_level]}",
        f"Market: demand={DEMAND[obs.demand_level]}, supply={SUPPLY[obs.supply_level]}, surge={obs.surge_multiplier}x",
        f"Time: {TIME[obs.time_of_day]}, {'weekday' if obs.day_type == 0 else 'weekend'}",
        f"Commission: {obs.commission_rate*100:.0f}%, Op cost: ${obs.operational_cost:.2f}",
        "",
        f"Step {obs.step_number}/{obs.max_steps}",
        f"Rider patience: {obs.rider_patience:.2f} ({obs.rider_mood})",
        f"Driver patience: {obs.driver_patience:.2f} ({obs.driver_mood})",
    ]
    if obs.last_proposed_price is not None:
        lines += [
            "",
            f"Last proposal: ${obs.last_proposed_price:.2f}",
            f"  Rider: {obs.last_rider_response}",
            f"  Driver: {obs.last_driver_response}",
        ]
    lines += ["", 'Respond with JSON only: {"price": <number>}']
    return "\n".join(lines)
```

### Simulator Prompt Builder

Takes `HiddenState` + `Observation` + `proposed_price` → string prompt.

```python
from ride_hailing_env.models import HiddenState, Observation

def build_simulator_prompt(
    hidden: HiddenState,
    obs: Observation,
    proposed_price: float
) -> str:
    rider_can_accept  = proposed_price <= hidden.rider_max_willingness
    driver_can_accept = proposed_price >= hidden.driver_min_willingness
    rider_surplus     = max(0, hidden.rider_max_willingness - proposed_price)
    driver_surplus    = max(0, proposed_price - hidden.driver_min_willingness)
    rider_safe_steps  = int(obs.rider_patience / hidden.rider_patience_decay) if hidden.rider_patience_decay > 0 else 99
    driver_safe_steps = int(obs.driver_patience / hidden.driver_patience_decay) if hidden.driver_patience_decay > 0 else 99
    steps_remaining   = obs.max_steps - obs.step_number

    lines = [
        "You are the rider and driver in a ride-hailing negotiation.",
        "You know your true thresholds. Decide whether to accept or bluff (reject despite being able to accept).",
        "",
        "=== YOUR HIDDEN STATE ===",
        f"Rider true max willingness: ${hidden.rider_max_willingness:.2f}",
        f"Driver true min willingness: ${hidden.driver_min_willingness:.2f}",
        f"Rider patience: {obs.rider_patience:.2f}, decay per rejection: {hidden.rider_patience_decay:.2f}",
        f"  → ~{rider_safe_steps} safe rejections remaining before cancellation risk",
        f"Driver patience: {obs.driver_patience:.2f}, decay per rejection: {hidden.driver_patience_decay:.2f}",
        f"  → ~{driver_safe_steps} safe rejections remaining",
        "",
        "=== PLATFORM PROPOSAL ===",
        f"Proposed price: ${proposed_price:.2f}",
        f"Steps remaining: {steps_remaining} of {obs.max_steps}",
        "",
        "=== DECISION ANALYSIS ===",
        f"Rider CAN accept: {rider_can_accept} (surplus if accepted: ${rider_surplus:.2f})",
        f"Driver CAN accept: {driver_can_accept} (surplus if accepted: ${driver_surplus:.2f})",
        "",
        "BLUFFING means rejecting even though you could accept, hoping for a better deal.",
        "RISK: patience decreases. If patience hits 0, the deal collapses and you earn nothing.",
        "",
        'Respond with JSON only: {"rider": "accept" or "reject", "driver": "accept" or "reject"}',
    ]
    return "\n".join(lines)
```

---

## Step 1.6 — Add `training/` Package

Create directory structure:
```
training/
├── __init__.py
├── prompt_builders.py      # Step 1.5 above
├── model_loader.py         # Step 1.4 above (refactored)
├── rollout.py              # Phase 2
├── train_platform.py       # Phase 2
├── train_simulator.py      # Phase 3
└── evaluate.py             # Phase 5
```

---

## Deliverables for Phase 1

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Both Qwen models load under Unsloth without errors
- [ ] `prompt_builders.py` exists and returns non-empty strings for sample observations
- [ ] `training/` package structure created
- [ ] New dependencies added to `requirements.txt`

---

## What Phase 1 Does NOT Do

- No training runs
- No simulator LLM integration
- No changes to `ride_hailing_env/` (environment is stable, do not touch)
