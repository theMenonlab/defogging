#!/usr/bin/env bash
set -euo pipefail

set +u
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source activate
fi

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda activate "$CONDA_ENV"
elif conda env list | awk '{print $1}' | grep -qx pix2pix; then
  echo "Conda environment '$CONDA_ENV' not found; using pix2pix." >&2
  conda activate pix2pix
else
  echo "Neither '$CONDA_ENV' nor 'pix2pix' exists on CHPC." >&2
  conda info --envs >&2
  exit 1
fi
set -u
