"""Plot training reward curves from phase metrics JSON files.

Reads the step-level metrics saved by train_platform.py and train_simulator.py
and produces a multi-panel chart suitable for presentations.

Usage:
    python scripts/plot_training_curves.py                        # default paths
    python scripts/plot_training_curves.py --output data/curves.png
    python scripts/plot_training_curves.py \\
        --platform_metrics data/training_metrics_platform_easy.json \\
        --simulator_metrics data/training_metrics_simulator_easy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


def _load(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[SKIP] Not found: {path}")
        return None
    with open(p) as f:
        return json.load(f)


def plot(
    platform_metrics_path: str = "data/training_metrics_platform_easy.json",
    simulator_metrics_path: str = "data/training_metrics_simulator_easy.json",
    output_path: str = "data/training_curves.png",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
        return

    platform_data  = _load(platform_metrics_path)
    simulator_data = _load(simulator_metrics_path)

    if not platform_data and not simulator_data:
        print("[ERROR] No metrics files found. Run training first.")
        return

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Dynamic Pricing — Multi-Agent RL Training Curves", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── Panel 1: Platform reward curve ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if platform_data and platform_data.get("steps"):
        steps_p = [r["step"] for r in platform_data["steps"]]
        avg_r   = [r["avg_reward"]   for r in platform_data["steps"]]
        win_r   = [r["window_avg"]   for r in platform_data["steps"]]
        ax1.plot(steps_p, avg_r, alpha=0.4, color="#4C72B0", label="batch avg")
        ax1.plot(steps_p, win_r, linewidth=2, color="#4C72B0", label="100-step window")

        # Baseline + post-training reference lines
        if "baseline" in platform_data:
            ax1.axhline(platform_data["baseline"]["avg_reward"], color="gray",
                        linestyle="--", linewidth=1, label=f"baseline {platform_data['baseline']['avg_reward']:.2f}")
        if "post_training" in platform_data:
            ax1.axhline(platform_data["post_training"]["avg_reward"], color="#55A868",
                        linestyle="--", linewidth=1, label=f"post-train {platform_data['post_training']['avg_reward']:.2f}")

        ax1.set_title("Platform — Avg Reward", fontsize=11)
        ax1.set_xlabel("Training step")
        ax1.set_ylabel("Reward")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No platform metrics", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Platform — Avg Reward")

    # ── Panel 2: Platform completion / cancel / timeout rates ───────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if platform_data and platform_data.get("steps"):
        steps_p    = [r["step"] for r in platform_data["steps"]]
        comp_rate  = [r["completion_rate"] * 100 for r in platform_data["steps"]]
        cancel_rate= [r["cancel_rate"]     * 100 for r in platform_data["steps"]]
        timeout_rate=[r["timeout_rate"]    * 100 for r in platform_data["steps"]]
        ax2.plot(steps_p, comp_rate,   linewidth=2, color="#55A868", label="completion %")
        ax2.plot(steps_p, cancel_rate, linewidth=2, color="#C44E52", label="cancellation %")
        ax2.plot(steps_p, timeout_rate,linewidth=2, color="#DD8452", label="timeout %")
        ax2.axhline(70, color="#55A868", linestyle=":", linewidth=1, alpha=0.6, label="target 70%")
        ax2.set_title("Platform — Episode Outcomes", fontsize=11)
        ax2.set_xlabel("Training step")
        ax2.set_ylabel("Rate (%)")
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No platform metrics", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Platform — Episode Outcomes")

    # ── Panel 3: Simulator reward curve ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if simulator_data and simulator_data.get("steps"):
        steps_s  = [r["step"]           for r in simulator_data["steps"]]
        sim_r    = [r["avg_sim_reward"] for r in simulator_data["steps"]]
        bluff_r  = [r["bluff_rate"] * 100 for r in simulator_data["steps"]]
        ax3.plot(steps_s, sim_r, linewidth=2, color="#8172B2", label="avg sim reward")
        if "baseline" in simulator_data:
            ax3.axhline(simulator_data["baseline"]["avg_sim_reward"], color="gray",
                        linestyle="--", linewidth=1,
                        label=f"baseline {simulator_data['baseline']['avg_sim_reward']:.2f}")
        ax3.set_title("Simulator — Avg Reward (Surplus)", fontsize=11)
        ax3.set_xlabel("Training step")
        ax3.set_ylabel("Reward")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No simulator metrics", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Simulator — Avg Reward")

    # ── Panel 4: Simulator bluff rate + collapse rate ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if simulator_data and simulator_data.get("steps"):
        steps_s     = [r["step"]              for r in simulator_data["steps"]]
        bluff_rate  = [r["bluff_rate"]   * 100 for r in simulator_data["steps"]]
        collapse_rate=[r["collapse_rate"] * 100 for r in simulator_data["steps"]]
        ax4.plot(steps_s, bluff_rate,   linewidth=2, color="#8172B2", label="bluff rate %")
        ax4.plot(steps_s, collapse_rate,linewidth=2, color="#C44E52", label="collapse rate %")
        ax4.axhspan(20, 50, alpha=0.08, color="#8172B2", label="healthy bluff zone (20–50%)")
        ax4.axhline(60, color="#C44E52", linestyle=":", linewidth=1, alpha=0.7, label="collapse danger (60%)")
        ax4.set_title("Simulator — Bluff & Collapse Rates", fontsize=11)
        ax4.set_xlabel("Training step")
        ax4.set_ylabel("Rate (%)")
        ax4.set_ylim(0, 105)
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No simulator metrics", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Simulator — Bluff & Collapse Rates")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[SAVE] Training curves → {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform_metrics",
                        default="data/training_metrics_platform_easy.json")
    parser.add_argument("--simulator_metrics",
                        default="data/training_metrics_simulator_easy.json")
    parser.add_argument("--output", default="data/training_curves.png")
    args = parser.parse_args()

    plot(
        platform_metrics_path=args.platform_metrics,
        simulator_metrics_path=args.simulator_metrics,
        output_path=args.output,
    )
