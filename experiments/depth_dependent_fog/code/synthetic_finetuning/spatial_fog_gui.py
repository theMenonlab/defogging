#!/usr/bin/env python3
"""Tkinter GUI for tuning spatially varying synthetic fog presets."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from tkinter import BOTH, HORIZONTAL, LEFT, RIGHT, X, Button, DoubleVar, Frame, Label, Scale, Tk, filedialog, messagebox, ttk, Canvas, Y
from tkinter import Toplevel
import numpy as np
from PIL import Image, ImageTk

# deep bump height map functions
from transformers import pipeline
#import module_normals_to_height
#import module_lowres_to_highres

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
    ("paint_weight", "Paint Weighting", 0.0, 5, 0.1), 
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
        
        # stuff for painting depth onto the image
        self.paint_window = None
        self.paint_canvas = None
        self.paint_photo = None

        self.paint_mask = None          # float32 HxW, values 0-1
        self.paint_radius = 30
        self.paint_strength = 0.05
        self._build_layout()
        self._load_current_image()
        self.depth_model = pipeline(
            "depth-estimation",
            model="Intel/zoedepth-nyu-kitti"
        )
        self._schedule_update()

    def _collect_default_images(self) -> list[Path]:
        image_dir = DEFAULT_IMAGE_DIR if DEFAULT_IMAGE_DIR.exists() else FALLBACK_IMAGE_DIR
        if not image_dir.exists():
            return []
        return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in VALID_SUFFIXES)
    """
    def load_fog_mask(self):
        # HWC -> CHW
        inp = np.transpose(self.clear_rgb, (2, 0, 1)).astype(np.float32)

        if inp.max() > 1.0:
            inp /= 255.0

        normals = module_color_to_normals.apply(inp, "LARGE", None)
        curvature = module_normals_to_height.apply(normals, "SMALL", None)

        # CHW -> HWC
        curvature = np.transpose(curvature, (1, 2, 0))

        mask = curvature.mean(axis=2)
        mask = (mask - mask.min()) / max(mask.max() - mask.min(), 1e-6)
        mask = mask

        self.paint_mask = mask
        self._schedule_update()
    """
    def load_fog_mask(self):
        if self.clear_rgb is None:
            return

        image = Image.fromarray(
            (self.clear_rgb * 255).astype(np.uint8)
        )

        result = self.depth_model(image)

        depth = np.array(result["depth"]).astype(np.float32)

        # normalize to 0-1
        depth -= depth.min()
        depth /= max(depth.max(), 1e-6)

        # Optional:
        # Near = white, far = black.
        # If you want farther objects to get MORE fog:
        #depth = 1.0 - depth

        self.paint_mask = depth

        self._schedule_update()
    def open_paint(self):

        if self.clear_rgb is None:
            return

        if self.paint_window is not None:
            self.paint_window.lift()
            return

        self.paint_window = Toplevel(self.root)
        self.paint_window.title("Paint Extra Fog Depth")

        rgb = (self.clear_rgb * 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        self.paint_photo = ImageTk.PhotoImage(image)

        self.paint_canvas = Canvas(
            self.paint_window,
            width=image.width,
            height=image.height,
        )

        self.paint_canvas.pack()

        self.paint_canvas.create_image(
            0,
            0,
            image=self.paint_photo,
            anchor="nw",
        )

        self.paint_canvas.bind("<B1-Motion>", self.paint_draw)
        self.paint_canvas.bind("<Button-1>", self.paint_draw)

        self.paint_window.protocol(
            "WM_DELETE_WINDOW",
            self.close_paint_window,
        )
    def paint_draw(self, event):

        if self.paint_mask is None:
            return

        x = int(event.x)
        y = int(event.y)

        r = self.paint_radius

        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        brush = xx*xx + yy*yy <= r*r

        x0 = max(0, x-r)
        y0 = max(0, y-r)
        x1 = min(self.paint_mask.shape[1], x+r+1)
        y1 = min(self.paint_mask.shape[0], y+r+1)

        bx0 = x0-(x-r)
        by0 = y0-(y-r)
        bx1 = bx0+(x1-x0)
        by1 = by0+(y1-y0)

        self.paint_mask[y0:y1, x0:x1] += (
            brush[by0:by1, bx0:bx1]
            * self.paint_strength
        )

        np.clip(self.paint_mask, 0, 1, out=self.paint_mask)

        self.paint_canvas.create_oval(
            x-r,
            y-r,
            x+r,
            y+r,
            outline="red",
        )

        self._schedule_update()
    def close_paint_window(self):
        self.paint_window.destroy()
        self.paint_window = None
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
        Button(button_row3, text="Paint Fog Depth", command = self.open_paint).pack(side=LEFT, padx=2)
        Button(button_row3, text="Load Mask", command = self.load_fog_mask).pack(side=LEFT, padx=2)

        self.status = Label(left, text="", anchor="w", justify=LEFT, wraplength=520)
        self.status.pack(fill=X, pady=(0, 6))


        canvas = Canvas(left)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)

        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        slider_frame = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=slider_frame, anchor="nw")

        def update_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        slider_frame.bind("<Configure>", update_scrollregion)

        def resize_frame(event):
            canvas.itemconfigure(window, width=event.width)

        canvas.bind("<Configure>", resize_frame)


        preset=SpatialFogPreset()
        for key, label, low, high, resolution in SLIDERS:
            row = Frame(slider_frame)
            row.pack(fill=X, pady=0.0)
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
        h, w = self.clear_rgb.shape[:2]
        self.paint_mask = np.zeros((h, w), dtype=np.float32)

    def _schedule_update(self) -> None:
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
        self.pending_after = self.root.after(160, self.update_preview)

    def update_preview(self) -> None:
        self.pending_after = None
        if self.clear_rgb is None:
            return
        preset = self._preset()
        foggy, field, fog_amount = synthesize_spatial_fog(self.clear_rgb, preset, extra_depth=self.paint_mask)
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
        self.update_preview()

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

        foggy, field, fog_amount = synthesize_spatial_fog(
            self.clear_rgb,
            self._preset(),
            extra_depth=self.paint_mask,
        )

        foggy_image = Image.fromarray(
            np.clip(foggy * 255.0, 0, 255).astype(np.uint8),
            mode="RGB",
        )
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
