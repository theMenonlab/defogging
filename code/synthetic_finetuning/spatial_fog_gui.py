#!/usr/bin/env python3
"""Tkinter GUI for tuning spatially varying synthetic fog presets."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from tkinter import BOTH, HORIZONTAL, LEFT, RIGHT, X, Button, DoubleVar, Frame, Label, Scale, Tk, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from spatial_fog_model import (
    SpatialFogPreset,
    make_preview_panel,
    preset_from_json,
    save_preset_json,
    synthesize_spatial_fog,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = ROOT.parents[0] / "synthetic_fog" / "clear_senterra"
FALLBACK_IMAGE_DIR = ROOT.parents[1] / "synthetic_fog" / "clear_senterra"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


SLIDERS: list[tuple[str, str, float, float, float]] = [
    ("beta_mean", "Mean fog density", 0.05, 5.0, 0.01),
    ("beta_variation", "Spatial fog variation", 0.0, 1.5, 0.01),
    ("field_scale_px", "Fog patch scale px", 32.0, 1400.0, 1.0),
    ("field_octaves", "Fog field octaves", 1.0, 6.0, 1.0),
    ("field_contrast", "Fog field contrast", 0.2, 3.0, 0.01),
    ("vertical_gradient", "Top/bottom gradient", -1.0, 1.0, 0.01),
    ("horizon_bias", "Horizon band fog", -0.5, 1.5, 0.01),
    ("airlight_r", "Airlight red", 0.0, 1.0, 0.005),
    ("airlight_g", "Airlight green", 0.0, 1.0, 0.005),
    ("airlight_b", "Airlight blue", 0.0, 1.0, 0.005),
    ("airlight_variation", "Local color variation", 0.0, 0.35, 0.005),
    ("warmth_bias", "Warm/cool fog bias", -0.25, 0.25, 0.005),
    ("bloom_strength", "Bloom strength", 0.0, 0.6, 0.005),
    ("bloom_radius", "Bloom radius", 0.0, 24.0, 0.1),
    ("blur_radius", "Blur radius", 0.0, 6.0, 0.05),
    ("blur_fog_coupling", "Blur follows fog", 0.0, 0.8, 0.005),
    ("saturation_mix", "Desaturation", 0.0, 0.85, 0.005),
    ("contrast_gamma", "Gamma/contrast", 0.55, 1.45, 0.005),
    ("noise_strength", "Sensor noise", 0.0, 0.06, 0.001),
    ("edge_veil_strength", "Edge veil", 0.0, 0.6, 0.005),
    ("seed", "Random field seed", 0.0, 10000.0, 1.0),
]


class SpatialFogGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Spatial Synthetic Fog Tuner")
        self.image_paths = self._collect_default_images()
        self.image_index = 0
        self.clear_rgb: np.ndarray | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.pending_after: str | None = None
        self.vars: dict[str, DoubleVar] = {}

        self._build_layout()
        self._load_current_image()
        self._schedule_update()

    def _collect_default_images(self) -> list[Path]:
        image_dir = DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.exists() else FALLBACK_IMAGE_DIR
        if not image_dir.exists():
            return []
        return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in VALID_SUFFIXES)

    def _build_layout(self) -> None:
        self.root.geometry("1500x980")
        left = Frame(self.root, padx=10, pady=8)
        left.pack(side=LEFT, fill="y")
        right = Frame(self.root, padx=8, pady=8)
        right.pack(side=RIGHT, fill=BOTH, expand=True)

        button_row = Frame(left)
        button_row.pack(fill=X, pady=(0, 6))
        Button(button_row, text="Open image", command=self.open_image).pack(side=LEFT, padx=2)
        Button(button_row, text="Open folder", command=self.open_folder).pack(side=LEFT, padx=2)
        Button(button_row, text="Prev", command=self.prev_image).pack(side=LEFT, padx=2)
        Button(button_row, text="Next", command=self.next_image).pack(side=LEFT, padx=2)

        button_row2 = Frame(left)
        button_row2.pack(fill=X, pady=(0, 6))
        Button(button_row2, text="Random seed", command=self.random_seed).pack(side=LEFT, padx=2)
        Button(button_row2, text="Reset", command=self.reset).pack(side=LEFT, padx=2)
        Button(button_row2, text="Save preset", command=self.save_preset).pack(side=LEFT, padx=2)
        Button(button_row2, text="Load preset", command=self.load_preset).pack(side=LEFT, padx=2)

        button_row3 = Frame(left)
        button_row3.pack(fill=X, pady=(0, 8))
        Button(button_row3, text="Save foggy", command=self.save_foggy).pack(side=LEFT, padx=2)
        Button(button_row3, text="Save preview", command=self.save_preview).pack(side=LEFT, padx=2)

        self.status = Label(left, text="", anchor="w", justify=LEFT, wraplength=520)
        self.status.pack(fill=X, pady=(0, 6))

        slider_frame = ttk.Frame(left)
        slider_frame.pack(fill=BOTH, expand=True)
        preset = SpatialFogPreset()
        for key, label, low, high, resolution in SLIDERS:
            row = Frame(slider_frame)
            row.pack(fill=X, pady=1)
            Label(row, text=label, width=22, anchor="w").pack(side=LEFT)
            var = DoubleVar(value=float(getattr(preset, key)))
            self.vars[key] = var
            scale = Scale(
                row,
                from_=low,
                to=high,
                resolution=resolution,
                orient=HORIZONTAL,
                length=300,
                variable=var,
                command=lambda _value: self._schedule_update(),
            )
            scale.pack(side=LEFT, fill=X, expand=True)

        self.preview_label = Label(right, bg="#eeeeee")
        self.preview_label.pack(fill=BOTH, expand=True)
        self.caption = Label(right, text="Preview columns: clear | synthetic fog | smooth fog field | fog amount map", anchor="w")
        self.caption.pack(fill=X)

    def _preset(self) -> SpatialFogPreset:
        payload = {}
        for field in fields(SpatialFogPreset):
            value = self.vars[field.name].get() if field.name in self.vars else getattr(SpatialFogPreset(), field.name)
            if field.name in {"field_octaves", "seed"}:
                value = int(round(float(value)))
            payload[field.name] = value
        return SpatialFogPreset(**payload)

    def _load_current_image(self) -> None:
        if not self.image_paths:
            self.status.config(text="No default images found. Use Open image or Open folder.")
            return
        path = self.image_paths[self.image_index % len(self.image_paths)]
        image = Image.open(path).convert("RGB")
        image.thumbnail((1100, 720), Image.Resampling.LANCZOS)
        self.clear_rgb = np.asarray(image, dtype=np.float32) / 255.0
        self.status.config(text=f"{path.name}  ({self.image_index + 1}/{len(self.image_paths)})")

    def _schedule_update(self) -> None:
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
        self.pending_after = self.root.after(160, self.update_preview)

    def update_preview(self) -> None:
        self.pending_after = None
        if self.clear_rgb is None:
            return
        preset = self._preset()
        foggy, field, fog_amount = synthesize_spatial_fog(self.clear_rgb, preset)
        panel = make_preview_panel(self.clear_rgb, foggy, field, fog_amount)
        panel.thumbnail((1220, 900), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(panel)
        self.preview_label.config(image=self.preview_photo)

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open clear image",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_paths = [Path(path)]
        self.image_index = 0
        self._load_current_image()
        self._schedule_update()

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Open clear-image folder")
        if not folder:
            return
        paths = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in VALID_SUFFIXES)
        if not paths:
            messagebox.showerror("No images", f"No jpg/png images found in {folder}")
            return
        self.image_paths = paths
        self.image_index = 0
        self._load_current_image()
        self._schedule_update()

    def prev_image(self) -> None:
        if self.image_paths:
            self.image_index = (self.image_index - 1) % len(self.image_paths)
            self._load_current_image()
            self._schedule_update()

    def next_image(self) -> None:
        if self.image_paths:
            self.image_index = (self.image_index + 1) % len(self.image_paths)
            self._load_current_image()
            self._schedule_update()

    def random_seed(self) -> None:
        self.vars["seed"].set(float((int(self.vars["seed"].get()) * 1664525 + 1013904223) % 10000))
        self._schedule_update()

    def reset(self) -> None:
        preset = SpatialFogPreset()
        for key, var in self.vars.items():
            var.set(float(getattr(preset, key)))
        self._schedule_update()

    def load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load spatial fog preset",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        preset = preset_from_json(Path(path))
        for key, var in self.vars.items():
            if hasattr(preset, key):
                var.set(float(getattr(preset, key)))
        self._schedule_update()

    def save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save spatial fog preset",
            initialdir=str(ROOT),
            initialfile="spatial_fog_preset.json",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        save_preset_json(Path(path), self._preset())

    def _current_outputs(self) -> tuple[Image.Image, Image.Image]:
        if self.clear_rgb is None:
            raise RuntimeError("No image loaded")
        foggy, field, fog_amount = synthesize_spatial_fog(self.clear_rgb, self._preset())
        foggy_image = Image.fromarray(np.clip(foggy * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        preview = make_preview_panel(self.clear_rgb, foggy, field, fog_amount)
        return foggy_image, preview

    def save_foggy(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save foggy image",
            initialdir=str(ROOT / "outputs"),
            initialfile="spatial_foggy.jpg",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if not path:
            return
        foggy_image, _preview = self._current_outputs()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        foggy_image.save(path, quality=95)

    def save_preview(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save preview panel",
            initialdir=str(ROOT / "outputs"),
            initialfile="spatial_fog_preview.jpg",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
        )
        if not path:
            return
        _foggy, preview = self._current_outputs()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        preview.save(path, quality=94)


def main() -> None:
    root = Tk()
    SpatialFogGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
