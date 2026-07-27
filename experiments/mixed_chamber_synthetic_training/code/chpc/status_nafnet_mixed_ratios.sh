#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
MIXED_OUTPUT_ROOT="${MIXED_OUTPUT_ROOT:-${BENCH_ROOT}/mixed_results_nafnet_20260725}"

squeue -u "$USER" -n naf_mix_1x,naf_mix_5x,naf_mix_0p2x \
  -o "%.18i %.18j %.2t %.10M %.10l %R"

echo
for run_name in current_ratio synthetic_5x synthetic_0p2x; do
  summary="$MIXED_OUTPUT_ROOT/$run_name/summary.json"
  checkpoint="$MIXED_OUTPUT_ROOT/$run_name/last.pth"
  if [[ -f "$summary" ]]; then
    echo "$run_name: COMPLETE"
  elif [[ -f "$checkpoint" ]]; then
    echo "$run_name: training; periodic checkpoint exists"
  else
    echo "$run_name: no completed output yet"
  fi
done
