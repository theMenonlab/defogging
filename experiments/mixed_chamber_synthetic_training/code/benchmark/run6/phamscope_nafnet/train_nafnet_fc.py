"""Minimal NAFNet builder retained for the mixed-training model registry."""

from __future__ import annotations

from nafnet_arch import NAFNet


def build_model() -> NAFNet:
    """Build the spectral-boundary core rewired by fog_rgb_benchmark.py."""
    return NAFNet(
        in_channels=1,
        out_channels=120,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )
