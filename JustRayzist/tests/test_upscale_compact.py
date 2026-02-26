from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import torch
from PIL import Image

from app.core.upscale import run_upscale_test


def _build_zero_compact_checkpoint(path: Path) -> None:
    state: dict[str, torch.Tensor] = {}
    for index in range(35):
        if index % 2 == 0:
            if index == 0:
                out_channels, in_channels = 64, 3
            elif index == 34:
                out_channels, in_channels = 12, 64
            else:
                out_channels, in_channels = 64, 64
            state[f"body.{index}.weight"] = torch.zeros(
                (out_channels, in_channels, 3, 3),
                dtype=torch.float32,
            )
            state[f"body.{index}.bias"] = torch.zeros((out_channels,), dtype=torch.float32)
        else:
            state[f"body.{index}.weight"] = torch.ones((64,), dtype=torch.float32)
    torch.save({"params": state}, path)


def test_compact_upscaler_preserves_nearest_base_when_residual_is_zero() -> None:
    root = Path.cwd() / "data" / f"test_upscale_{uuid4().hex}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        input_path = root / "input.png"
        checkpoint_path = root / "compact_2x.pth"

        image = Image.new("RGB", (4, 4))
        image.putdata(
            [
                (0, 10, 20),
                (30, 40, 50),
                (60, 70, 80),
                (90, 100, 110),
                (120, 130, 140),
                (150, 160, 170),
                (180, 190, 200),
                (210, 220, 230),
                (15, 25, 35),
                (45, 55, 65),
                (75, 85, 95),
                (105, 115, 125),
                (135, 145, 155),
                (165, 175, 185),
                (195, 205, 215),
                (225, 235, 245),
            ]
        )
        image.save(input_path, format="PNG")
        _build_zero_compact_checkpoint(checkpoint_path)

        result = run_upscale_test(
            input_image_path=input_path,
            checkpoint_path=checkpoint_path,
            profile_name="constrained",
        )

        assert result.scale_factor == 2
        assert result.output_width == 8
        assert result.output_height == 8

        expected = image.resize((8, 8), resample=Image.Resampling.NEAREST)
        assert list(result.image.getdata()) == list(expected.getdata())
    finally:
        shutil.rmtree(root, ignore_errors=True)
