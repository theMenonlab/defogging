#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BENCH_ROOT="${BENCH_ROOT:?set BENCH_ROOT to the directory containing the datasets and outputs}"
export MAPILLARY_ROOT="${MAPILLARY_ROOT:-${BENCH_ROOT}/mapillary_vistas}"
export DEPTH_ROOT="${DEPTH_ROOT:-${BENCH_ROOT}/mapillary_vistas_depth_zoe}"
export RUN6_ROOT="${RUN6_ROOT:-${PACKAGE_ROOT}/benchmark/run6}"
export BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-${PACKAGE_ROOT}/benchmark/fog_rgb_benchmark.py}"
export CONDA_ENV="${CONDA_ENV:-pix2pix}"
