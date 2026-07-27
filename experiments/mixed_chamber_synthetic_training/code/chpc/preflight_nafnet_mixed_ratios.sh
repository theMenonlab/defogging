#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PACKAGE_ROOT"

MIXED_OUTPUT_ROOT="${MIXED_OUTPUT_ROOT:-${BENCH_ROOT}/mixed_results_nafnet_20260725}"
REAL_FOG_ROOT="${REAL_FOG_ROOT:-${BENCH_ROOT}/VerticalFilter_MediumFog_Redo_3-21-26_aligned}"
REAL_GT_ROOT="${REAL_GT_ROOT:-${BENCH_ROOT}/archive}"
MAPILLARY_IMAGES_ROOT="$MAPILLARY_ROOT"
if [[ ! -d "$MAPILLARY_IMAGES_ROOT/training/images" ]] && [[ -d "$MAPILLARY_IMAGES_ROOT/Mapillary Vistas" ]]; then
  MAPILLARY_IMAGES_ROOT="$MAPILLARY_IMAGES_ROOT/Mapillary Vistas"
fi

required=(
  "$BENCHMARK_SCRIPT"
  "$RUN6_ROOT/collect_fc_results.py"
  "$REAL_FOG_ROOT"
  "$REAL_GT_ROOT"
  "$MAPILLARY_IMAGES_ROOT/training/images"
  "$MAPILLARY_IMAGES_ROOT/validation/images"
  "$MAPILLARY_IMAGES_ROOT/testing/images"
  "$PACKAGE_ROOT/mixed_train_nafnet.py"
  "$PACKAGE_ROOT/presets/original_randomized_v19.json"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "MISSING: $path" >&2
    exit 1
  fi
done

expected=(18000 2000 5000)
splits=(training validation testing)
for index in 0 1 2; do
  count=$(find "$MAPILLARY_IMAGES_ROOT/${splits[$index]}/images" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  echo "Mapillary ${splits[$index]} images: $count / ${expected[$index]}"
  if [[ "$count" -ne "${expected[$index]}" ]]; then
    echo "Mapillary dataset count mismatch." >&2
    exit 1
  fi
done

for run_name in current_ratio synthetic_5x synthetic_0p2x; do
  run_dir="$MIXED_OUTPUT_ROOT/$run_name"
  if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -print -quit)" ]]; then
    echo "REFUSING TO OVERWRITE NONEMPTY RUN: $run_dir" >&2
    exit 1
  fi
done

source "$SCRIPT_DIR/activate_env.sh"
python -m py_compile mixed_train_nafnet.py synthetic_finetune_all_models.py fog_synthesis.py
python - "$BENCHMARK_SCRIPT" "$RUN6_ROOT" "$REAL_FOG_ROOT" "$REAL_GT_ROOT" <<'PY'
import sys
from pathlib import Path

from synthetic_finetune_all_models import import_benchmark
from mixed_train_nafnet import is_synthetic_step

benchmark = import_benchmark(Path(sys.argv[1]))
run6_root = Path(sys.argv[2])
real_fog_root = Path(sys.argv[3])
real_gt_root = Path(sys.argv[4])

train = benchmark.FogRGBPairedDataset(real_fog_root, real_gt_root, split="train", holdout_every=10)
test = benchmark.FogRGBPairedDataset(real_fog_root, real_gt_root, split="test", holdout_every=10)
print(f"Chamber training pairs: {len(train)} / 4943")
print(f"Chamber holdout pairs: {len(test)} / 552")
if len(train) != 4943 or len(test) != 552:
    raise SystemExit("Chamber split count mismatch")

runs = {row.model_key: row for row in benchmark.load_model_runs(run6_root)}
model = benchmark.build_model_for_run(run6_root, runs["nafnet_fc"])
print(f"Fresh NAFNet parameter count: {sum(parameter.numel() for parameter in model.parameters()):,}")

for name, real_steps, synthetic_steps in (
    ("current_ratio", 247150, 2600),
    ("synthetic_5x", 247150, 13000),
    ("synthetic_0p2x", 247150, 520),
):
    total = real_steps + synthetic_steps
    actual = sum(is_synthetic_step(index, total, synthetic_steps) for index in range(total))
    if actual != synthetic_steps:
        raise SystemExit(f"{name}: schedule emitted {actual}, expected {synthetic_steps}")
    print(
        f"{name}: real={real_steps}, synthetic={synthetic_steps}, total={total}, "
        f"synthetic/real={synthetic_steps / real_steps:.6f}"
    )
PY

echo
echo "Preflight PASSED."
echo "All three runs start from the same seed-controlled fresh NAFNet initialization."
echo "Each run uses 247,150 chamber updates; synthetic updates are 520, 2,600, or 13,000."
echo "Output root: $MIXED_OUTPUT_ROOT"
