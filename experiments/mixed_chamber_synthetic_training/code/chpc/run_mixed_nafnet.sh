#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?run name required}"
REAL_STEPS="${2:?real optimizer-step count required}"
SYNTHETIC_STEPS="${3:?synthetic optimizer-step count required}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/activate_env.sh"
cd "$PACKAGE_ROOT"

MIXED_OUTPUT_ROOT="${MIXED_OUTPUT_ROOT:-${BENCH_ROOT}/mixed_results_nafnet_20260725}"
REAL_FOG_ROOT="${REAL_FOG_ROOT:-${BENCH_ROOT}/VerticalFilter_MediumFog_Redo_3-21-26_aligned}"
REAL_GT_ROOT="${REAL_GT_ROOT:-${BENCH_ROOT}/archive}"

python mixed_train_nafnet.py \
  --run-name "$RUN_NAME" \
  --model-key nafnet_fc \
  --real-fog-root "$REAL_FOG_ROOT" \
  --real-gt-root "$REAL_GT_ROOT" \
  --holdout-every 10 \
  --mapillary-root "$MAPILLARY_ROOT" \
  --depth-root "$DEPTH_ROOT" \
  --preset-json "$PACKAGE_ROOT/presets/original_randomized_v19.json" \
  --benchmark-script "$BENCHMARK_SCRIPT" \
  --run6-root "$RUN6_ROOT" \
  --output-root "$MIXED_OUTPUT_ROOT" \
  --real-optimizer-steps "$REAL_STEPS" \
  --synthetic-optimizer-steps "$SYNTHETIC_STEPS" \
  --synthetic-accumulation-steps 2 \
  --crop-size 512 \
  --num-workers 4 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --checkpoint-every 5000 \
  --max-real-eval-samples 0 \
  --max-synthetic-val-batches 300 \
  --max-synthetic-test-batches 500 \
  --seed 734
