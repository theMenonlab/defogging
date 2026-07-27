#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=naf_mix_5x
#SBATCH --output=naf_mix_5x_%j.log

set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:?export PACKAGE_ROOT before submitting}"
export MIXED_OUTPUT_ROOT="${MIXED_OUTPUT_ROOT:?export MIXED_OUTPUT_ROOT before submitting}"
mkdir -p "$MIXED_OUTPUT_ROOT/slurm"

# Keep chamber exposure fixed and use exactly 5 x 2,600 synthetic updates.
bash "$PACKAGE_ROOT/chpc/run_mixed_nafnet.sh" synthetic_5x 247150 13000
