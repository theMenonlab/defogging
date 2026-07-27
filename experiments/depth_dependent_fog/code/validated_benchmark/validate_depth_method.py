#!/usr/bin/env python3
"""Create depth-fog previews and reject degenerate all-clear/all-fog synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fog_synthesis import jitter_preset, load_preset, normalize_map, synthesize_fog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapillary-root", required=True, type=Path)
    parser.add_argument("--depth-root", required=True, type=Path)
    parser.add_argument("--preset-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    image_root = args.mapillary_root / "validation" / "images"
    paths = sorted(p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.samples]
    if len(paths) < args.samples:
        raise ValueError(f"Found only {len(paths)} validation images")
    args.output.mkdir(parents=True, exist_ok=True)
    preset = load_preset(args.preset_json)
    rows = []
    for index, image_path in enumerate(paths):
        depth_path = args.depth_root / "validation" / image_path.relative_to(image_root).with_suffix(".npy")
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        clear = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != clear.shape[:2]:
            depth = np.asarray(Image.fromarray(depth).resize((clear.shape[1], clear.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
        sample_preset, _ = jitter_preset(
            preset, np.random.default_rng(1000 + index), beta_mult_min=1.0, beta_mult_max=1.0,
            variation_mult_min=1.0, variation_mult_max=1.0, light_fog_prob=0.0, identity_prob=0.0,
        )
        randomized, randomized_stats = synthesize_fog(clear, sample_preset, "randomized")
        depth_fog, depth_stats = synthesize_fog(clear, sample_preset, "andrew_depth", depth)
        depth_rgb = np.repeat(normalize_map(depth)[:, :, None], 3, axis=2)
        panel = np.concatenate([clear, randomized, depth_fog, depth_rgb], axis=1)
        Image.fromarray(np.clip(panel * 255.0, 0, 255).astype(np.uint8)).save(args.output / f"preview_{index:02d}.jpg", quality=92)
        rows.append(
            {
                "image": str(image_path),
                "depth": str(depth_path),
                "raw_depth_min": float(np.nanmin(depth)),
                "raw_depth_mean": float(np.nanmean(depth)),
                "raw_depth_max": float(np.nanmax(depth)),
                "randomized": randomized_stats,
                "andrew_depth": depth_stats,
            }
        )
    report = {"panel_order": ["clear", "randomized", "andrew_depth", "normalized_depth"], "rows": rows}
    (args.output / "depth_method_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    means = [row["andrew_depth"]["transmission_mean"] for row in rows]
    if min(means) < 0.01 or max(means) > 0.99:
        raise SystemExit(f"Degenerate depth synthesis transmission means: {means}")
    print(json.dumps({"samples": len(rows), "depth_transmission_means": means}, indent=2))


if __name__ == "__main__":
    main()
