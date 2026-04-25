#!/usr/bin/env bash
# =============================================================================
# Full training pipeline — Phase 2 → 3 → 4 → Evaluate
#
# Run this from the dynamic_pricing/ directory on a machine with:
#   - NVIDIA GPU (16 GB VRAM minimum; 24 GB recommended)
#   - Python 3.10+
#   - All dependencies installed (see: pip install -r requirements.txt)
#
# Expected wall-clock time:
#   Phase 2 (easy, 500 steps)  ~30-60 min on A100; ~2-3 hrs on T4
#   Phase 3 (easy, 500 steps)  ~20-40 min
#   Phase 4 (easy, 1000 steps) ~60-90 min
#   Evaluation                 ~10-15 min
#
# Checkpoints are saved every 200 steps — safe to resume with --platform_ckpt
# or --simulator_ckpt pointing at checkpoints/last_stable/ if a run crashes.
# =============================================================================

set -e   # stop on first error
set -u   # treat unset variables as errors

# --- Config ---
TASK="easy"
PHASE2_STEPS=500
PHASE3_STEPS=500
PHASE4_STEPS=1000
EVAL_EPISODES=50

PHASE2_OUT="checkpoints/phase2"
PHASE3_OUT="checkpoints/phase3"
PHASE4_OUT="checkpoints/phase4"

echo ""
echo "============================================================"
echo " DYNAMIC PRICING — FULL TRAINING PIPELINE"
echo "============================================================"
echo " Task:         $TASK"
echo " Phase2 steps: $PHASE2_STEPS"
echo " Phase3 steps: $PHASE3_STEPS"
echo " Phase4 steps: $PHASE4_STEPS"
echo "============================================================"
echo ""

# --- Phase 2: Platform v0 vs rule-based simulator ---
echo ">>> PHASE 2: Training platform_v0 vs rule-based simulator..."
python training/train_platform.py \
  --task        "$TASK" \
  --steps       "$PHASE2_STEPS" \
  --output_dir  "$PHASE2_OUT"

echo ""
echo ">>> Phase 2 done. Checkpoint: $PHASE2_OUT/platform_lora"
echo ""

# --- Phase 3: Simulator v1 vs frozen platform_v0 ---
echo ">>> PHASE 3: Training simulator_v1 vs frozen platform_v0..."
python training/train_simulator.py \
  --task           "$TASK" \
  --steps          "$PHASE3_STEPS" \
  --platform_ckpt  "$PHASE2_OUT/platform_lora" \
  --output_dir     "$PHASE3_OUT"

echo ""
echo ">>> Phase 3 done. Checkpoint: $PHASE3_OUT/simulator_lora"
echo ""

# --- Phase 4: Platform v1 vs frozen simulator_v1 ---
echo ">>> PHASE 4: Training platform_v1 vs frozen simulator_v1..."
python training/train_platform.py \
  --task            "$TASK" \
  --steps           "$PHASE4_STEPS" \
  --platform_ckpt   "$PHASE2_OUT/platform_lora" \
  --simulator_ckpt  "$PHASE3_OUT/simulator_lora" \
  --output_dir      "$PHASE4_OUT"

echo ""
echo ">>> Phase 4 done. Checkpoint: $PHASE4_OUT/platform_lora"
echo ""

# --- Evaluation: 4-stage comparison table ---
echo ">>> EVALUATION: Running 4-stage comparison..."
python training/evaluate.py \
  --tasks       "$TASK" \
  --n_episodes  "$EVAL_EPISODES" \
  --output      "data/final_evaluation.json"

echo ""
echo "============================================================"
echo " ALL DONE"
echo " Results: data/final_evaluation.json"
echo "============================================================"
