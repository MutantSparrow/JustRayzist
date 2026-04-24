from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image


def tile_seam_positions(length: int, tile_size: int, tile_overlap: int) -> tuple[int, ...]:
    safe_length = max(0, int(length))
    safe_tile_size = max(0, int(tile_size))
    safe_tile_overlap = max(0, int(tile_overlap))
    if safe_length <= 0 or safe_tile_size <= 0:
        return ()
    step = safe_tile_size - safe_tile_overlap
    if step <= 0 or step >= safe_length:
        return ()
    positions: list[int] = []
    current = step
    while current < safe_length:
        positions.append(int(current))
        current += step
    return tuple(positions)


def tile_seam_band_radius(tile_overlap: int) -> int:
    safe_overlap = max(0, int(tile_overlap))
    return max(6, min(max(1, safe_overlap // 4), 48))


def soften_tile_grid_seams(
    image: Image.Image,
    *,
    tile_width: int,
    tile_height: int | None = None,
    tile_overlap_x: int,
    tile_overlap_y: int | None = None,
) -> Image.Image:
    height = tile_height if tile_height is not None else tile_width
    overlap_y = tile_overlap_y if tile_overlap_y is not None else tile_overlap_x
    vertical_seams = tile_seam_positions(image.width, tile_width, tile_overlap_x)
    horizontal_seams = tile_seam_positions(image.height, height, overlap_y)
    band_radius = tile_seam_band_radius(max(tile_overlap_x, overlap_y))
    return soften_tile_seams(
        image,
        vertical_seams=vertical_seams,
        horizontal_seams=horizontal_seams,
        band_radius=band_radius,
    )


def soften_tile_seams(
    image: Image.Image,
    *,
    vertical_seams: Iterable[int] = (),
    horizontal_seams: Iterable[int] = (),
    band_radius: int,
) -> Image.Image:
    normalized_vertical = tuple(sorted({int(value) for value in vertical_seams if int(value) > 0}))
    normalized_horizontal = tuple(sorted({int(value) for value in horizontal_seams if int(value) > 0}))
    safe_radius = max(1, int(band_radius))
    if not normalized_vertical and not normalized_horizontal:
        return image.copy()

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.size == 0:
        return image.copy()
    working = rgb.astype(np.float32) / 255.0
    height, width = working.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    threshold = float(np.quantile(grad, 0.80))
    if threshold <= 1e-6:
        edge_guard = np.ones_like(gray, dtype=np.float32)
    else:
        edge_guard = 1.0 - np.clip(grad / threshold, 0.0, 1.0)
        edge_guard = cv2.GaussianBlur(edge_guard, (0, 0), sigmaX=1.0, sigmaY=1.0)

    blur_sigma = max(1.0, float(safe_radius) / 2.5)
    kernel_radius = max(1, int(round(blur_sigma * 3.0)))
    kernel_size = (kernel_radius * 2) + 1
    kernel = cv2.getGaussianKernel(kernel_size, blur_sigma).astype(np.float32)
    identity = np.array([1.0], dtype=np.float32)
    horizontal_blur = cv2.sepFilter2D(working, -1, kernel, identity, borderType=cv2.BORDER_REFLECT)
    vertical_blur = cv2.sepFilter2D(working, -1, identity, kernel, borderType=cv2.BORDER_REFLECT)

    result = working.copy()
    if normalized_vertical:
        vertical_mask = _axis_seam_mask(
            width=width,
            height=height,
            seams=normalized_vertical,
            band_radius=safe_radius,
            axis="x",
        )
        vertical_mask = np.clip(vertical_mask * edge_guard, 0.0, 1.0)[..., None]
        result = (result * (1.0 - vertical_mask)) + (horizontal_blur * vertical_mask)
    if normalized_horizontal:
        horizontal_mask = _axis_seam_mask(
            width=width,
            height=height,
            seams=normalized_horizontal,
            band_radius=safe_radius,
            axis="y",
        )
        horizontal_mask = np.clip(horizontal_mask * edge_guard, 0.0, 1.0)[..., None]
        result = (result * (1.0 - horizontal_mask)) + (vertical_blur * horizontal_mask)
    return Image.fromarray(np.clip(result * 255.0, 0.0, 255.0).round().astype(np.uint8), mode="RGB")


def _axis_seam_mask(
    *,
    width: int,
    height: int,
    seams: tuple[int, ...],
    band_radius: int,
    axis: str,
) -> np.ndarray:
    axis_length = width if axis == "x" else height
    mask_1d = np.zeros(axis_length, dtype=np.float32)
    for seam in seams:
        if seam >= axis_length:
            continue
        start = max(0, seam - band_radius)
        end = min(axis_length, seam + band_radius + 1)
        coords = np.arange(start, end, dtype=np.float32)
        distance = np.abs(coords - float(seam)) / float(max(1, band_radius))
        weights = 0.5 * (1.0 + np.cos(np.pi * np.clip(distance, 0.0, 1.0)))
        mask_1d[start:end] = np.maximum(mask_1d[start:end], weights)
    if axis == "x":
        return np.broadcast_to(mask_1d[None, :], (height, width))
    return np.broadcast_to(mask_1d[:, None], (height, width))
