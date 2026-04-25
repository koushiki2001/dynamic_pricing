"""Comprehensive 4-stage evaluation script (Phase 4).

Produces the before/after comparison table used in the demo and presentation.

The four stages tell the full training story:
  A — Untrained platform vs rule-based simulator   (pre-RL baseline)
  B — Platform v0 vs rule-based simulator          (Phase 2 result)
  C — Platform v0 vs Simulator v1                  (disruption: strategic opponent)
  D — Platform v1 vs Simulator v1                  (Phase 4 recovery)

Row C dropping below Row B proves the simulator LLM is genuinely strategic.
Row D recovering above Row B proves the platform RL adaptation loop is working.

Usage:
    # Full 4-stage table (requires all checkpoints):
    python training/evaluate.py

    # Quick smoke-test (n=10 episodes per stage, easy only):
    python training/evaluate.py --n_episodes 10 --tasks easy

    # Single stage:
    python training/evaluate.py --stages A B --tasks easy medium

    # Skip missing checkpoints instead of crashing:
    python training/evaluate.py --skip_missing
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from ride_hailing_env.environment import DynamicPricingEnv
from training.model_loader import load_platform_model, load_simulator_model
from training.rollout import collect_platform_rollout, collect_multiagent_rollout
from training.simulator_agent import SimulatorAgent


# ---------------------------------------------------------------------------
# Checkpoint paths — single place to update if structure changes
# ---------------------------------------------------------------------------

CKPT = {
    "platform_v0":   "checkpoints/phase2/platform_lora",
    "platform_v1":   "checkpoints/phase4/platform_lora",
    "simulator_v1":  "checkpoints/phase3/simulator_lora",
}


# ---------------------------------------------------------------------------
# Per-stage evaluation
# ---------------------------------------------------------------------------

def evaluate_stage(
    platform_model: Any,
    platform_tokenizer: Any,
    simulator_agent: Optional[SimulatorAgent],
    env: DynamicPricingEnv,
    task: str,
    n_episodes: int = 50,
) -> Dict[str, Any]:
    """Run n_episodes and return per-stage metrics dict.

    simulator_agent=None → use the rule-based deterministic simulator.
    """
    rewards: List[float] = []
    completed = 0
    cancelled = 0
    timed_out = 0
    total_steps = 0
    format_ok = 0

    for _ in range(n_episodes):
        if simulator_agent is None:
            episode = collect_platform_rollout(
                platform_model, platform_tokenizer, env, task
            )
        else:
            result = collect_multiagent_rollout(
                platform_model, platform_tokenizer, simulator_agent, env, task
            )
            episode = result["platform_samples"]

        if not episode:
            continue

        terminal = episode[-1]
        r = terminal["reward"]
        rewards.append(r)
        total_steps += len(episode)

        if r > 0:
            completed += 1
        elif r <= -4.0:
            cancelled += 1
        else:
            timed_out += 1

        # Format compliance: last completion should parse as JSON with a price key
        try:
            import json as _json
            parsed = _json.loads(terminal["completion"])
            if "price" in parsed:
                format_ok += 1
        except Exception:
            # Non-JSON output still acceptable if price was extracted
            if terminal.get("price") is not None:
                format_ok += 1

    n = len(rewards) or 1
    return {
        "avg_reward":          round(sum(rewards) / n, 4),
        "completion_rate":     round(completed / n, 3),
        "cancellation_rate":   round(cancelled / n, 3),
        "timeout_rate":        round(timed_out / n, 3),
        "avg_steps_per_ep":    round(total_steps / n, 2),
        "format_compliance":   round(format_ok / n, 3),
        "n_episodes":          n,
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_full_evaluation(
    tasks: List[str] = None,
    stages: List[str] = None,
    n_episodes: int = 50,
    output_path: str = "data/final_evaluation.json",
    skip_missing: bool = False,
) -> Dict:
    tasks  = tasks  or ["easy", "medium", "hard"]
    stages = stages or ["A", "B", "C", "D"]

    results: Dict[str, Any] = {}
    env = DynamicPricingEnv()

    # --- Load models once (shared across tasks) ---

    def _load_platform(ckpt_key: Optional[str]) -> Optional[tuple]:
        """Load platform model. ckpt_key=None loads the base model with no LoRA."""
        lora_path = None
        if ckpt_key is not None:
            path = CKPT.get(ckpt_key)
            if path and Path(path).exists():
                lora_path = path
            elif path:
                if skip_missing:
                    print(f"[SKIP] Checkpoint not found: {path}")
                    return None
                raise FileNotFoundError(
                    f"Checkpoint not found: {path}\n"
                    f"Run the corresponding training phase first, or pass --skip_missing."
                )
        return load_platform_model(lora_path=lora_path)

    def _load_simulator(ckpt_key: str) -> Optional[SimulatorAgent]:
        path = CKPT.get(ckpt_key)
        if not path or not Path(path).exists():
            if skip_missing:
                print(f"[SKIP] Simulator checkpoint not found: {path}")
                return None
            raise FileNotFoundError(
                f"Simulator checkpoint not found: {path}\n"
                f"Run Phase 3 training first, or pass --skip_missing."
            )
        sim_model, sim_tok = load_simulator_model(lora_path=path)
        sim_model.eval()
        return SimulatorAgent(sim_model, sim_tok)

    # Pre-load models needed across stages
    print("[LOAD] Loading models for evaluation...")
    t_load = time.time()

    platform_untrained = None
    platform_v0        = None
    platform_v1        = None
    simulator_v1       = None

    if "A" in stages:
        platform_untrained = _load_platform(None)    # base model, no LoRA

    if "B" in stages or "C" in stages:
        platform_v0 = _load_platform("platform_v0")

    if "C" in stages or "D" in stages:
        simulator_v1 = _load_simulator("simulator_v1")

    if "D" in stages:
        platform_v1 = _load_platform("platform_v1")

    print(f"[LOAD] Done in {time.time()-t_load:.1f}s\n")

    # --- Run evaluations ---
    stage_map = {
        "A": ("A_untrained_vs_rulebased",    platform_untrained, None),
        "B": ("B_platformv0_vs_rulebased",   platform_v0,        None),
        "C": ("C_platformv0_vs_simv1",       platform_v0,        simulator_v1),
        "D": ("D_platformv1_vs_simv1",       platform_v1,        simulator_v1),
    }

    for task in tasks:
        results[task] = {}
        env = DynamicPricingEnv(task_name=task)
        print(f"\n{'─'*50}")
        print(f"Task: {task.upper()}")
        print(f"{'─'*50}")

        for stage_id in stages:
            key, plat_tuple, sim_agent = stage_map[stage_id]
            if plat_tuple is None:
                print(f"  [{stage_id}] SKIPPED — model not loaded")
                continue
            p_model, p_tok = plat_tuple

            t0 = time.time()
            metrics = evaluate_stage(
                p_model, p_tok, sim_agent, env, task, n_episodes
            )
            elapsed = time.time() - t0

            results[task][key] = metrics
            print(
                f"  [{stage_id}] cr={metrics['completion_rate']:.0%}  "
                f"cancel={metrics['cancellation_rate']:.0%}  "
                f"timeout={metrics['timeout_rate']:.0%}  "
                f"avg_r={metrics['avg_reward']:.3f}  "
                f"steps={metrics['avg_steps_per_ep']:.1f}  "
                f"({elapsed:.1f}s)"
            )

    # --- Print summary table ---
    _print_table(results, tasks, stages, stage_map)

    # --- Save ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Evaluation results → {output_path}")

    return results


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------

STAGE_LABELS = {
    "A": "A: Untrained vs Rule-based",
    "B": "B: Platform v0 vs Rule-based",
    "C": "C: Platform v0 vs Simulator v1",
    "D": "D: Platform v1 vs Simulator v1",
}

STAGE_KEYS = {
    "A": "A_untrained_vs_rulebased",
    "B": "B_platformv0_vs_rulebased",
    "C": "C_platformv0_vs_simv1",
    "D": "D_platformv1_vs_simv1",
}


def _print_table(
    results: Dict,
    tasks: List[str],
    stages: List[str],
    stage_map: Dict,
) -> None:
    col_w = 10
    header_cols = "".join(f"{t.upper():>{col_w}}" for t in tasks)
    width = 38 + col_w * len(tasks)

    print(f"\n{'='*width}")
    print(f"  COMPLETION RATE COMPARISON")
    print(f"{'='*width}")
    print(f"  {'STAGE':<36}{header_cols}")
    print(f"{'─'*width}")

    for stage_id in stages:
        label = STAGE_LABELS.get(stage_id, stage_id)
        key   = STAGE_KEYS.get(stage_id, "")
        row   = f"  {label:<36}"
        for task in tasks:
            cr = results.get(task, {}).get(key, {}).get("completion_rate")
            if cr is None:
                row += f"{'—':>{col_w}}"
            else:
                row += f"  {cr*100:6.1f}%"
        print(row)

    print(f"{'─'*width}")

    # Highlight disruption and recovery
    if "C" in stages and "B" in stages and "easy" in tasks:
        b_cr = results.get("easy", {}).get(STAGE_KEYS["B"], {}).get("completion_rate", 0)
        c_cr = results.get("easy", {}).get(STAGE_KEYS["C"], {}).get("completion_rate", 0)
        d_cr = results.get("easy", {}).get(STAGE_KEYS["D"], {}).get("completion_rate", 0) if "D" in stages else None
        print()
        if c_cr < b_cr:
            print(f"  ✓ Disruption confirmed: C ({c_cr:.0%}) < B ({b_cr:.0%}) — simulator is strategic")
        if d_cr is not None and d_cr > b_cr:
            print(f"  ✓ Recovery confirmed:   D ({d_cr:.0%}) > B ({b_cr:.0%}) — platform adapted")
        elif d_cr is not None:
            print(f"  ⚠ Recovery incomplete:  D ({d_cr:.0%}) ≤ B ({b_cr:.0%}) — continue training")

    print(f"{'='*width}\n")

    # Secondary table: avg reward
    print(f"{'='*width}")
    print(f"  AVERAGE REWARD COMPARISON")
    print(f"{'='*width}")
    print(f"  {'STAGE':<36}{header_cols}")
    print(f"{'─'*width}")
    for stage_id in stages:
        label = STAGE_LABELS.get(stage_id, stage_id)
        key   = STAGE_KEYS.get(stage_id, "")
        row   = f"  {label:<36}"
        for task in tasks:
            avg_r = results.get(task, {}).get(key, {}).get("avg_reward")
            if avg_r is None:
                row += f"{'—':>{col_w}}"
            else:
                row += f"  {avg_r:+7.3f}"
        print(row)
    print(f"{'='*width}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-stage platform evaluation")
    parser.add_argument(
        "--tasks", nargs="+", default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Which task difficulties to evaluate"
    )
    parser.add_argument(
        "--stages", nargs="+", default=["A", "B", "C", "D"],
        choices=["A", "B", "C", "D"],
        help="Which evaluation stages to run"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50,
        help="Episodes per stage per task (default 50; use 10 for quick smoke-test)"
    )
    parser.add_argument(
        "--output", default="data/final_evaluation.json",
        help="Where to save the JSON results"
    )
    parser.add_argument(
        "--skip_missing", action="store_true",
        help="Skip stages whose checkpoints don't exist instead of crashing"
    )
    args = parser.parse_args()

    run_full_evaluation(
        tasks=args.tasks,
        stages=args.stages,
        n_episodes=args.n_episodes,
        output_path=args.output,
        skip_missing=args.skip_missing,
    )
