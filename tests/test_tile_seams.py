from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.tile_seams import soften_tile_grid_seams, tile_seam_positions


def _col_jump(image: Image.Image, column: int) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    deltas = np.abs(np.diff(gray, axis=1))
    index = max(0, min(deltas.shape[1] - 1, int(column) - 1))
    return float(deltas[:, index].mean())


def _row_jump(image: Image.Image, row: int) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    deltas = np.abs(np.diff(gray, axis=0))
    index = max(0, min(deltas.shape[0] - 1, int(row) - 1))
    return float(deltas[index, :].mean())


def test_tile_seam_positions_match_overlap_grid() -> None:
    assert tile_seam_positions(2048, 1024, 128) == (896, 1792)
    assert tile_seam_positions(3072, 1024, 128) == (896, 1792, 2688)


def test_soften_tile_grid_seams_reduces_synthetic_band_jumps() -> None:
    base = np.full((256, 256, 3), 180, dtype=np.uint8)
    base[:, 96:, :] = np.clip(base[:, 96:, :] + 24, 0, 255)
    base[96:, :, :] = np.clip(base[96:, :, :] - 20, 0, 255)
    for row in range(base.shape[0]):
        base[row, :, 1] = np.clip(base[row, :, 1] + (row // 6), 0, 255)
    image = Image.fromarray(base, mode="RGB")

    smoothed = soften_tile_grid_seams(
        image,
        tile_width=128,
        tile_height=128,
        tile_overlap_x=32,
        tile_overlap_y=32,
    )

    assert _col_jump(smoothed, 96) < (_col_jump(image, 96) * 0.75)
    assert _row_jump(smoothed, 96) < (_row_jump(image, 96) * 0.75)
