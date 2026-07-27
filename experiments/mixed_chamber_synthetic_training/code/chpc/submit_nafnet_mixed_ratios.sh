#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/config.sh"
export MIXED_OUTPUT_ROOT="${MIXED_OUTPUT_ROOT:-${BENCH_ROOT}/mixed_results_nafnet_20260725}"
cd "$PACKAGE_ROOT"

active_names=(naf_mix_1x naf_mix_5x naf_mix_0p2x)
for job_name in "${active_names[@]}"; do
  if squeue -h -u "$USER" -n "$job_name" | grep -q .; then
    echo "An active $job_name job already exists; refusing duplicate submissions." >&2
    squeue -u "$USER" -n "$job_name"
    exit 1
  fi
done

bash chpc/preflight_nafnet_mixed_ratios.sh
mkdir -p "$MIXED_OUTPUT_ROOT"

job_current=$(sbatch --parsable chpc/nafnet_mixed_current_ratio.slurm)
job_more=$(sbatch --parsable chpc/nafnet_mixed_5x_more_synthetic.slurm)
job_less=$(sbatch --parsable chpc/nafnet_mixed_5x_less_synthetic.slurm)

record="$MIXED_OUTPUT_ROOT/submitted_jobs_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "current_ratio=$job_current"
  echo "synthetic_5x=$job_more"
  echo "synthetic_0p2x=$job_less"
} | tee "$record"

echo
echo "Submitted all three independent mixed-data NAFNet jobs."
echo "Monitor with: bash chpc/status_nafnet_mixed_ratios.sh"
