#!/bin/bash
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=naf_mix_1x
#SBATCH --output=naf_mix_1x_%j.log

set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:?export PACKAGE_ROOT before submitting}"
export MIXED_OUTPUT_ROOT="${MIXED_OUTPUT_ROOT:?export MIXED_OUTPUT_ROOT before submitting}"
mkdir -p "$MIXED_OUTPUT_ROOT/slurm"

# Matches the optimizer-update counts in the current sequential pipeline:
# 4,943 chamber pairs x 50 epochs = 247,150 real updates; 2,600 synthetic updates.
bash "$PACKAGE_ROOT/chpc/run_mixed_nafnet.sh" current_ratio 247150 2600
