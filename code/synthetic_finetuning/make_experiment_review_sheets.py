#!/usr/bin/env python3
"""Build review sheets for one synthetic-fog experiment output set."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(font_path, size) if Path(font_path).exists() else ImageFont.load_default()


TITLE_FONT = get_font(34, True)
LABEL_FONT = get_font(18)
SMALL_FONT = get_font(15)


def evenly_sample(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    return [paths[round(i * (len(paths) - 1) / (count - 1))] for i in range(count)]


def fit(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), "white")
    tile.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return tile


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    bbox = draw.textbbox(xy, text, font=SMALL_FONT)
    pad = 5
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill="white")
    draw.text(xy, text, font=SMALL_FONT, fill=(20, 20, 20))


def make_sheet(title: str, paths: list[Path], out_path: Path, max_items: int = 12, columns: int = 3) -> None:
    selected = evenly_sample(paths, max_items)
    if not selected:
        return
    tile_w, tile_h = 560, 360
    gap, margin, header = 22, 28, 70
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (margin * 2 + columns * tile_w + (columns - 1) * gap, header + margin + rows * tile_h + (rows - 1) * gap), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), f"{title} ({len(paths)} available; showing {len(selected)})", font=TITLE_FONT, fill=(15, 15, 15))
    for idx, path in enumerate(selected):
        row, col = divmod(idx, columns)
        x = margin + col * (tile_w + gap)
        y = header + row * (tile_h + gap)
        canvas.paste(fit(path, tile_w, tile_h), (x, y))
        draw_label(draw, (x + 10, y + 10), path.stem.replace("_compare", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=94)


def make_overview(title: str, datasets: dict[str, list[Path]], out_path: Path) -> None:
    tile_w, tile_h = 440, 280
    label_w, gap, margin, header = 235, 16, 28, 72
    canvas = Image.new("RGB", (margin * 2 + label_w + 4 * tile_w + 4 * gap, header + margin + len(datasets) * tile_h + (len(datasets) - 1) * gap), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), title, font=TITLE_FONT, fill=(15, 15, 15))
    for row, (name, paths) in enumerate(datasets.items()):
        y = header + row * (tile_h + gap)
        draw.text((margin, y + 12), name, font=LABEL_FONT, fill=(15, 15, 15))
        draw.text((margin, y + 40), f"{len(paths)} available", font=SMALL_FONT, fill=(70, 70, 70))
        for col, path in enumerate(evenly_sample(paths, 4)):
            x = margin + label_w + gap + col * (tile_w + gap)
            canvas.paste(fit(path, tile_w, tile_h), (x, y))
            draw_label(draw, (x + 8, y + 8), path.stem.replace("_compare", ""))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--paired-dir", required=True, type=Path)
    parser.add_argument("--airplane-dir", required=True, type=Path)
    parser.add_argument("--free-fog-dir", required=True, type=Path)
    parser.add_argument("--sheet-dir", required=True, type=Path)
    args = parser.parse_args()

    paired = args.paired_dir / "comparisons"
    datasets = {
        "O-HAZE": sorted(paired.glob("O-HAZE_*_compare.jpg")),
        "NH-HAZE": sorted(paired.glob("NH-HAZE_*_compare.jpg")),
        "NTIRE26-NH": sorted(paired.glob("NTIRE26-NH_*_compare.jpg")),
        "Airplane all": sorted(args.airplane_dir.glob("*_compare.jpg")),
        "Fog machine": sorted(args.free_fog_dir.glob("*_compare.jpg")),
    }
    args.sheet_dir.mkdir(parents=True, exist_ok=True)
    for name, paths in datasets.items():
        safe = name.lower().replace(" ", "_").replace("-", "_")
        make_sheet(name, paths, args.sheet_dir / f"{safe}_options.jpg")
    make_overview(args.run_name, datasets, args.sheet_dir / "all5_dataset_overview.jpg")
    synthetic_visuals = sorted((args.run_dir / "visuals").glob("val_*.png")) + sorted((args.run_dir / "visuals").glob("test_*.png"))
    make_sheet("Mapillary synthetic val/test strips", synthetic_visuals, args.sheet_dir / "mapillary_synthetic_val_test_options.jpg", max_items=24, columns=4)


if __name__ == "__main__":
    main()
