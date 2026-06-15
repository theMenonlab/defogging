#!/usr/bin/env python3
"""Verify that the cleaned fog-chamber checkpoint loads into ResidualNAFNetRGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from nafnet_arch import NAFNet


class ResidualNAFNetRGB(nn.Module):
    def __init__(self, width: int, middle_blocks: int, enc_blocks: list[int], dec_blocks: list[int]) -> None:
        super().__init__()
        self.core = NAFNet(
            in_channels=3,
            out_channels=3,
            width=width,
            middle_blk_num=middle_blocks,
            enc_blk_nums=enc_blocks,
            dec_blk_nums=dec_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.core(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    provenance = checkpoint.get("provenance", {}) if isinstance(checkpoint, dict) else {}
    width = int(provenance.get("width", 32))
    middle_blocks = int(provenance.get("middle_blocks", 12))
    enc_blocks = list(provenance.get("enc_blocks", [2, 2, 4, 8]))
    dec_blocks = list(provenance.get("dec_blocks", [2, 2, 2, 2]))

    model = ResidualNAFNetRGB(width, middle_blocks, enc_blocks, dec_blocks)
    result = model.load_state_dict(state, strict=True)
    report = {
        "checkpoint": str(args.checkpoint),
        "strict_load_ok": True,
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "model_type": "ResidualNAFNetRGB",
        "width": width,
        "middle_blocks": middle_blocks,
        "enc_blocks": enc_blocks,
        "dec_blocks": dec_blocks,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
