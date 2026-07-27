# Mixed-training code

`mixed_train_nafnet.py` is the exact trainer used for the three completed
ratio experiments. It builds a fresh direct-RGB NAFNet through the included
minimal benchmark registry and interleaves chamber and randomized-synthetic
optimizer updates deterministically.

## Inputs

- Paired fog-chamber roots with matching category/filename structure.
- Mapillary Vistas with `training/images`, `validation/images`, and
  `testing/images`.
- `benchmark/` and `benchmark/run6/`, included here.

`--depth-root` is retained by the shared synthetic dataset interface but is not
read in `randomized` mode.

## Smoke test

From this `code/` directory:

```bash
python mixed_train_nafnet.py \
  --run-name mixed_smoke \
  --model-key nafnet_fc \
  --real-fog-root /path/to/fog_chamber/foggy \
  --real-gt-root /path/to/fog_chamber/ground_truth \
  --holdout-every 10 \
  --mapillary-root /path/to/mapillary_vistas \
  --depth-root /path/to/unused_or_depth_cache \
  --preset-json presets/original_randomized_v19.json \
  --benchmark-script benchmark/fog_rgb_benchmark.py \
  --run6-root benchmark/run6 \
  --output-root /path/to/output \
  --real-optimizer-steps 247150 \
  --synthetic-optimizer-steps 2600 \
  --smoke
```

The smoke flag reduces training and evaluation to a few small steps while
exercising model construction, both data streams, checkpoint writing, and
metrics.

## Reproduce the matched-budget checkpoint

Run the same command without `--smoke` and add:

```text
--synthetic-accumulation-steps 2
--crop-size 512
--num-workers 4
--learning-rate 0.0001
--weight-decay 0.0001
--checkpoint-every 5000
--max-real-eval-samples 0
--max-synthetic-val-batches 300
--max-synthetic-test-batches 500
--seed 734
```

For the ratio sensitivity runs, change only `--run-name` and
`--synthetic-optimizer-steps` to `520` or `13000`.

## Slurm

The `chpc/` directory contains three independent Slurm jobs plus preflight,
submit, and status helpers. Before submission, export:

```bash
export PACKAGE_ROOT=/absolute/path/to/this/code
export BENCH_ROOT=/absolute/path/to/benchmark_data_root
export MIXED_OUTPUT_ROOT=/absolute/path/to/mixed_results
```

Then run:

```bash
bash chpc/submit_nafnet_mixed_ratios.sh
```

Edit the `#SBATCH --account` and `--partition` lines if your cluster differs.
