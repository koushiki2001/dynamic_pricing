"""Episode rollout collection for GRPO training (Phase 2+).

Two public functions:

collect_platform_rollout()
    Runs one episode with the platform LLM proposing prices against the
    existing rule-based simulator. Returns a list of step dicts — one per
    step — each containing (prompt, completion, reward).

collect_batch()
    Runs batch_size episodes and flattens terminal steps into the
    (prompts, completions, rewards) triple that GRPOTrainer expects.

Phase 2b additions baked in:
  - format compliance score added to every step reward
  - anti-hacking checks run at every step
  - process reward added to non-terminal steps
  - inspect_generations() for periodic human review
  - drift detection helper
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.models import Observation
from ride_hailing_env.reward import compute_process_reward
from training.anti_hack import run_all_checks, score_format_compliance
from training.prompt_builders import build_platform_prompt


# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------

def parse_price(text: str, fallback: float) -> float:
    """Extract a price float from model output. Returns fallback on failure."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        return float(parsed["price"])
    except Exception:
        pass
    nums = re.findall(r"\b\d+\.?\d*\b", text)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            pass
    return fallback


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------

def collect_platform_rollout(
    model: Any,
    tokenizer: Any,
    env: DynamicPricingEnv,
    task: str = "easy",
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """Run one complete episode with the platform LLM vs rule-based simulator.

    Returns a list of step dicts::

        {
            "prompt":      str,
            "completion":  str,
            "reward":      float,   # env reward + format score + anti-hack penalty + process reward
            "price":       float,
            "done":        bool,
            "violations":  list[str],
        }

    Only the terminal step carries a meaningful reward signal for GRPO.
    Intermediate steps carry shaping + process rewards.
    """
    import torch

    obs: Observation = env.reset()
    samples: List[Dict[str, Any]] = []
    done = False
    last_price: Optional[float] = None

    while not done:
        prompt = build_platform_prompt(obs)
        fallback = (obs.rider_quoted_price + obs.driver_quoted_price) / 2.0

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        price = parse_price(completion, fallback)
        price = max(1.0, min(500.0, price))

        # Phase 2b: format compliance + anti-hacking checks
        format_score = score_format_compliance(completion)
        hack_penalty, violations = run_all_checks(
            price=price,
            completion=completion,
            last_price=last_price if last_price is not None else price + 1,
            rider_quote=obs.rider_quoted_price,
            driver_quote=obs.driver_quoted_price,
        )

        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        result = env.step(action)

        # Phase 2b: process reward on non-terminal steps
        process_r = 0.0
        if not result.done:
            process_r = compute_process_reward(
                proposed_price=price,
                last_proposed_price=last_price,
                rider_accepted=result.info.get("outcome", {}).get("rider_accepted", False),
                driver_accepted=result.info.get("outcome", {}).get("driver_accepted", False),
                rider_patience=result.observation.rider_patience,
                driver_patience=result.observation.driver_patience,
                step_number=result.observation.step_number,
                max_steps=obs.max_steps,
            )

        total_reward = result.reward + format_score + hack_penalty + process_r

        samples.append({
            "prompt":     prompt,
            "completion": completion,
            "reward":     round(total_reward, 4),
            "price":      price,
            "done":       result.done,
            "violations": violations,
        })

        last_price = price
        obs = result.observation
        done = result.done

    return samples


# ---------------------------------------------------------------------------
# Batch collection
# ---------------------------------------------------------------------------

def collect_batch(
    model: Any,
    tokenizer: Any,
    env: DynamicPricingEnv,
    task: str = "easy",
    batch_size: int = 16,
) -> Tuple[List[str], List[str], List[float]]:
    """Collect batch_size terminal-step samples for GRPOTrainer.

    Returns (prompts, completions, rewards) — only terminal steps, since
    GRPO updates on the outcome-determining generation.
    """
    prompts, completions, rewards = [], [], []
    for _ in range(batch_size):
        episode = collect_platform_rollout(model, tokenizer, env, task)
        if episode:
            t = episode[-1]
            prompts.append(t["prompt"])
            completions.append(t["completion"])
            rewards.append(t["reward"])
    return prompts, completions, rewards


# ---------------------------------------------------------------------------
# Phase 3: multi-agent rollout
# ---------------------------------------------------------------------------

def collect_multiagent_rollout(
    platform_model: Any,
    platform_tokenizer: Any,
    simulator_agent: Any,           # SimulatorAgent instance
    env: DynamicPricingEnv,
    task: str = "easy",
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Run one episode with both LLM agents active.

    Returns separate sample lists for each model so each can be trained
    independently::

        {
            "platform_samples":  [{"prompt", "completion", "reward", "price", "done"}],
            "simulator_samples": [{"prompt", "completion", "reward", "price", "done"}],
        }

    Simulator reward is assigned only at the terminal step (sparse outcome reward).
    Platform reward includes format + anti-hack + process signals at every step.
    """
    import torch

    obs: Observation = env.reset()
    hidden = env.get_hidden_state()

    platform_samples:  List[Dict[str, Any]] = []
    simulator_samples: List[Dict[str, Any]] = []
    done      = False
    last_price: Optional[float] = None

    while not done:
        fallback = (obs.rider_quoted_price + obs.driver_quoted_price) / 2.0

        # --- Platform proposes price ---
        p_prompt = build_platform_prompt(obs)
        inputs   = platform_tokenizer(p_prompt, return_tensors="pt").to(platform_model.device)
        with torch.inference_mode():
            out = platform_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=platform_tokenizer.eos_token_id,
            )
        p_completion = platform_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        price = parse_price(p_completion, fallback)
        price = max(1.0, min(500.0, price))

        # Phase 2b checks on platform output
        format_score  = score_format_compliance(p_completion)
        hack_penalty, violations = run_all_checks(
            price=price,
            completion=p_completion,
            last_price=last_price if last_price is not None else price + 1,
            rider_quote=obs.rider_quoted_price,
            driver_quote=obs.driver_quoted_price,
        )

        # --- Simulator decides accept/reject ---
        from training.prompt_builders import build_simulator_prompt
        s_prompt   = build_simulator_prompt(hidden, obs, price)
        decision   = simulator_agent.decide(hidden, obs, price)

        # --- Step environment with simulator decision ---
        action = {"type": "propose_price", "payload": {"price": round(price, 2)}}
        result = env.step(action, override_decision={
            "rider":  decision.rider_accept,
            "driver": decision.driver_accept,
        })

        # Platform process reward on non-terminal steps
        process_r = 0.0
        if not result.done:
            process_r = compute_process_reward(
                proposed_price=price,
                last_proposed_price=last_price,
                rider_accepted=decision.rider_accept,
                driver_accepted=decision.driver_accept,
                rider_patience=result.observation.rider_patience,
                driver_patience=result.observation.driver_patience,
                step_number=result.observation.step_number,
                max_steps=obs.max_steps,
            )

        p_total_reward = result.reward + format_score + hack_penalty + process_r

        platform_samples.append({
            "prompt":     p_prompt,
            "completion": p_completion,
            "reward":     round(p_total_reward, 4),
            "price":      price,
            "done":       result.done,
            "violations": violations,
        })
        simulator_samples.append({
            "prompt":     s_prompt,
            "completion": decision.raw_output,
            "reward":     None,      # filled at terminal step below
            "price":      price,
            "done":       result.done,
        })

        last_price = price
        obs        = result.observation
        done       = result.done

    # Assign simulator terminal reward
    if simulator_samples:
        from ride_hailing_env.reward import compute_simulator_reward
        terminal_info = result.info.get("outcome", {})
        sim_r = compute_simulator_reward(
            proposed_price=simulator_samples[-1]["price"],
            rider_max_willingness=hidden.rider_max_willingness,
            driver_min_willingness=hidden.driver_min_willingness,
            ride_completed=terminal_info.get("ride_completed", False),
            steps_taken=terminal_info.get("steps_taken", obs.step_number),
            max_steps=obs.max_steps,
        )
        simulator_samples[-1]["reward"] = round(sim_r, 4)

    return {
        "platform_samples":  platform_samples,
        "simulator_samples": simulator_samples,
    }


def collect_multiagent_batch(
    platform_model: Any,
    platform_tokenizer: Any,
    simulator_agent: Any,
    env: DynamicPricingEnv,
    task: str = "easy",
    batch_size: int = 8,
) -> Tuple[List[str], List[str], List[float]]:
    """Collect batch_size multi-agent episodes, return platform terminal samples."""
    prompts, completions, rewards = [], [], []
    for _ in range(batch_size):
        result = collect_multiagent_rollout(
            platform_model, platform_tokenizer, simulator_agent, env, task
        )
        p = result["platform_samples"]
        if p:
            t = p[-1]
            prompts.append(t["prompt"])
            completions.append(t["completion"])
            rewards.append(t["reward"])
    return prompts, completions, rewards


def collect_simulator_batch(
    platform_model: Any,
    platform_tokenizer: Any,
    simulator_agent: Any,
    env: DynamicPricingEnv,
    task: str = "easy",
    batch_size: int = 8,
) -> Tuple[List[str], List[str], List[float]]:
    """Collect batch_size episodes, return simulator terminal samples for GRPO."""
    prompts, completions, rewards = [], [], []
    for _ in range(batch_size):
        result = collect_multiagent_rollout(
            platform_model, platform_tokenizer, simulator_agent, env, task
        )
        s = result["simulator_samples"]
        if s and s[-1]["reward"] is not None:
            t = s[-1]
            prompts.append(t["prompt"])
            completions.append(t["completion"])
            rewards.append(t["reward"])
    return prompts, completions, rewards


# ---------------------------------------------------------------------------
# Phase 2b: generation inspection
# ---------------------------------------------------------------------------

def inspect_generations(samples: List[Dict[str, Any]], step: int, n_show: int = 3) -> None:
    """Print sampled generations for human review during training.

    Call every 100 training steps. A rising reward is not enough —
    inspect outputs to catch reward hacking early.
    """
    print(f"\n{'='*60}")
    print(f"GENERATION INSPECTION — training step {step}")
    print(f"{'='*60}")

    shown = random.sample(samples, min(n_show, len(samples)))
    for i, s in enumerate(shown, 1):
        print(f"\n[Sample {i}]")
        print(f"  reward:     {s['reward']:.3f}")
        print(f"  price:      {s.get('price', '?')}")
        print(f"  completion: {s['completion'][:120]!r}")
        if s.get("violations"):
            for v in s["violations"]:
                print(f"  ⚠️  VIOLATION: {v}")
        if s.get("reward", 0) > 8.0:
            print("  ⚠️  VERY HIGH REWARD — inspect for exploitation")
        if s.get("price", 0) < 2.0 or s.get("price", 0) > 200:
            print("  ⚠️  EXTREME PRICE — possible exploitation")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Phase 2b: drift detection
# ---------------------------------------------------------------------------

def detect_reward_drift(reward_history: List[float], window: int = 50) -> bool:
    """Return True if reward dropped >30% from the previous window.

    Signals training instability or reward hacking taking hold.
    """
    if len(reward_history) < window * 2:
        return False
    recent = reward_history[-window:]
    previous = reward_history[-window * 2:-window]
    avg_recent = sum(recent) / len(recent)
    avg_prev = sum(previous) / len(previous)
    return avg_prev > 0 and avg_recent < avg_prev * 0.70
