from __future__ import annotations

import math
import re
import inspect
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.memory import (
    CudaMemorySnapshot,
    ProcessMemorySnapshot,
    cuda_memory_snapshot,
    now_perf,
    process_memory_snapshot,
)


@dataclass(frozen=True)
class UpscalePolicy:
    tile_size: int
    tile_overlap: int
    prefer_half_precision: bool


@dataclass
class UpscaleResult:
    image: Image.Image
    scale_factor: int
    device: str
    precision: str
    tile_size: int
    tile_overlap: int
    duration_ms: int
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    architecture: str
    cuda_memory_before: CudaMemorySnapshot | None
    cuda_memory_after: CudaMemorySnapshot | None
    process_memory_before: ProcessMemorySnapshot | None
    process_memory_after: ProcessMemorySnapshot | None

    def telemetry_dict(self) -> dict[str, Any]:
        return {
            "scale_factor": self.scale_factor,
            "device": self.device,
            "precision": self.precision,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "duration_ms": self.duration_ms,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "architecture": self.architecture,
            "cuda_memory_before": (
                self.cuda_memory_before.to_dict() if self.cuda_memory_before else None
            ),
            "cuda_memory_after": (
                self.cuda_memory_after.to_dict() if self.cuda_memory_after else None
            ),
            "process_memory_before": (
                self.process_memory_before.to_dict() if self.process_memory_before else None
            ),
            "process_memory_after": (
                self.process_memory_after.to_dict() if self.process_memory_after else None
            ),
        }


def resolve_upscale_policy(profile_name: str, architecture: str) -> UpscalePolicy:
    normalized = profile_name.strip().lower()
    architecture_name = architecture.strip().lower()

    if architecture_name == "rrdb":
        if normalized == "high":
            return UpscalePolicy(tile_size=768, tile_overlap=24, prefer_half_precision=True)
        if normalized == "balanced":
            return UpscalePolicy(tile_size=512, tile_overlap=24, prefer_half_precision=True)
        if normalized == "constrained":
            return UpscalePolicy(tile_size=256, tile_overlap=16, prefer_half_precision=True)
        return UpscalePolicy(tile_size=384, tile_overlap=24, prefer_half_precision=True)

    if architecture_name == "plksr":
        if normalized == "high":
            return UpscalePolicy(tile_size=0, tile_overlap=24, prefer_half_precision=False)
        if normalized == "balanced":
            return UpscalePolicy(tile_size=512, tile_overlap=24, prefer_half_precision=False)
        if normalized == "constrained":
            return UpscalePolicy(tile_size=256, tile_overlap=16, prefer_half_precision=False)
        return UpscalePolicy(tile_size=384, tile_overlap=24, prefer_half_precision=False)

    if normalized == "high":
        return UpscalePolicy(tile_size=0, tile_overlap=24, prefer_half_precision=True)
    if normalized == "balanced":
        return UpscalePolicy(tile_size=640, tile_overlap=24, prefer_half_precision=True)
    if normalized == "constrained":
        return UpscalePolicy(tile_size=384, tile_overlap=24, prefer_half_precision=False)
    return UpscalePolicy(tile_size=512, tile_overlap=24, prefer_half_precision=False)


def _extract_state_dict(torch_module: Any, checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported upscaler checkpoint payload.")

    for key in ("params_ema", "params", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            if all(torch_module.is_tensor(value) for value in candidate.values()):
                return candidate

    if checkpoint and all(torch_module.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise ValueError("Unable to locate tensor state_dict in upscaler checkpoint.")


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        candidate = key.removeprefix("module.")
        candidate = candidate.replace(".layer_norm.", ".norm.")
        if candidate not in normalized:
            normalized[candidate] = value
    return normalized


def _detect_upscaler_architecture(state_dict: dict[str, Any]) -> str:
    if "conv_first.weight" in state_dict and "conv_last.weight" in state_dict:
        return "rrdb"
    if (
        "feats.0.weight" in state_dict
        and any(re.match(r"^feats\.\d+\.channel_mixer\.0\.weight$", key) for key in state_dict)
        and any(re.match(r"^feats\.\d+\.lk\.conv\.weight$", key) for key in state_dict)
        and any(re.match(r"^feats\.\d+\.weight$", key) for key in state_dict)
    ):
        return "plksr"
    body_layer_pattern = re.compile(r"^body\.(\d+)\.weight$")
    if any(body_layer_pattern.match(key) for key in state_dict):
        return "compact"
    raise ValueError("Unsupported upscaler architecture in checkpoint.")


def _build_compact_network(torch_module: Any, state_dict: dict[str, Any]) -> tuple[Any, int]:
    nn = torch_module.nn
    pattern = re.compile(r"^body\.(\d+)\.weight$")
    layer_indices: set[int] = set()
    for key in state_dict:
        match = pattern.match(key)
        if match:
            layer_indices.add(int(match.group(1)))

    if not layer_indices:
        raise ValueError("Upscaler checkpoint is missing body layers.")

    layers: list[Any] = []
    last_conv_out_channels: int | None = None
    for index in sorted(layer_indices):
        weight_key = f"body.{index}.weight"
        weight = state_dict[weight_key]
        if weight.ndim == 4:
            out_channels, in_channels, kernel_h, kernel_w = [int(value) for value in weight.shape]
            padding = (kernel_h // 2, kernel_w // 2)
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_h, kernel_w),
                    stride=1,
                    padding=padding,
                )
            )
            last_conv_out_channels = out_channels
            continue
        if weight.ndim == 1:
            layers.append(nn.PReLU(num_parameters=int(weight.shape[0])))
            continue
        raise ValueError(f"Unsupported layer shape for {weight_key}: {tuple(weight.shape)}")

    if last_conv_out_channels is None:
        raise ValueError("Upscaler checkpoint does not include a final convolution layer.")

    scale_squared = last_conv_out_channels / 3.0
    scale = int(math.sqrt(scale_squared))
    if scale * scale * 3 != last_conv_out_channels or scale < 1:
        raise ValueError(
            "Unable to infer upscale factor from final convolution channels "
            f"({last_conv_out_channels})."
        )

    class CompactUpscaleNet(torch_module.nn.Module):
        def __init__(self, body_layers: list[Any], scale_value: int):
            super().__init__()
            self.body = torch_module.nn.Sequential(*body_layers)
            self.upsample = torch_module.nn.PixelShuffle(scale_value)
            self.upscale = int(scale_value)

        def forward(self, x: Any) -> Any:
            residual = self.upsample(self.body(x))
            base = torch_module.nn.functional.interpolate(
                x,
                scale_factor=self.upscale,
                mode="nearest",
            )
            return residual + base

    model = CompactUpscaleNet(layers, scale)
    return model, scale


def _build_rrdb_network(torch_module: Any, state_dict: dict[str, Any]) -> tuple[Any, int]:
    nn = torch_module.nn
    functional = torch_module.nn.functional

    conv_first_shape = tuple(state_dict["conv_first.weight"].shape)
    conv_last_shape = tuple(state_dict["conv_last.weight"].shape)
    num_feat = int(conv_first_shape[0])
    conv_first_in = int(conv_first_shape[1])
    num_out_ch = int(conv_last_shape[0])
    if num_out_ch <= 0:
        raise ValueError("Invalid conv_last output channels for RRDB upscaler.")

    growth_shape = tuple(state_dict["body.0.rdb1.conv1.weight"].shape)
    num_grow_ch = int(growth_shape[0])

    block_indices: set[int] = set()
    block_pattern = re.compile(r"^body\.(\d+)\.rdb1\.conv1\.weight$")
    for key in state_dict:
        match = block_pattern.match(key)
        if match:
            block_indices.add(int(match.group(1)))
    if not block_indices:
        raise ValueError("RRDB checkpoint is missing residual dense blocks.")
    num_block = max(block_indices) + 1

    upsample_indices: set[int] = set()
    up_pattern = re.compile(r"^conv_up(\d+)\.weight$")
    for key in state_dict:
        match = up_pattern.match(key)
        if match:
            upsample_indices.add(int(match.group(1)))
    if not upsample_indices:
        raise ValueError("RRDB checkpoint is missing conv_up layers.")
    num_upsample = max(upsample_indices)

    num_in_ch = num_out_ch
    unshuffle_squared = conv_first_in / max(1, num_in_ch)
    unshuffle_factor = int(round(math.sqrt(unshuffle_squared)))
    if unshuffle_factor < 1 or (unshuffle_factor * unshuffle_factor * num_in_ch) != conv_first_in:
        raise ValueError(
            "Unable to infer RRDB unshuffle factor from conv_first channels: "
            f"in={conv_first_in}, out={num_in_ch}."
        )

    scale_numerator = 2**num_upsample
    if scale_numerator % unshuffle_factor != 0:
        raise ValueError(
            "Incompatible RRDB scaling factors: "
            f"upsample={scale_numerator}, unshuffle={unshuffle_factor}."
        )
    scale_factor = scale_numerator // unshuffle_factor
    if scale_factor < 1:
        raise ValueError("Invalid inferred RRDB scale factor.")

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat_value: int, num_grow_value: int):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat_value, num_grow_value, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat_value + num_grow_value, num_grow_value, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat_value + (num_grow_value * 2), num_grow_value, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat_value + (num_grow_value * 3), num_grow_value, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat_value + (num_grow_value * 4), num_feat_value, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x: Any) -> Any:
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch_module.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch_module.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch_module.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch_module.cat((x, x1, x2, x3, x4), 1))
            return (x5 * 0.2) + x

    class RRDB(nn.Module):
        def __init__(self, num_feat_value: int, num_grow_value: int):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat_value, num_grow_value)
            self.rdb2 = ResidualDenseBlock(num_feat_value, num_grow_value)
            self.rdb3 = ResidualDenseBlock(num_feat_value, num_grow_value)

        def forward(self, x: Any) -> Any:
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return (out * 0.2) + x

    class RRDBNet(nn.Module):
        def __init__(
            self,
            *,
            num_in_ch_value: int,
            num_out_ch_value: int,
            num_feat_value: int,
            num_block_value: int,
            num_grow_value: int,
            num_upsample_value: int,
            unshuffle_factor_value: int,
        ):
            super().__init__()
            self.num_upsample = int(num_upsample_value)
            self.unshuffle_factor = int(unshuffle_factor_value)
            self.conv_first = nn.Conv2d(
                num_in_ch_value * (self.unshuffle_factor**2),
                num_feat_value,
                3,
                1,
                1,
            )
            self.body = nn.Sequential(
                *[RRDB(num_feat_value, num_grow_value) for _ in range(num_block_value)]
            )
            self.conv_body = nn.Conv2d(num_feat_value, num_feat_value, 3, 1, 1)
            for index in range(1, self.num_upsample + 1):
                setattr(self, f"conv_up{index}", nn.Conv2d(num_feat_value, num_feat_value, 3, 1, 1))
            self.conv_hr = nn.Conv2d(num_feat_value, num_feat_value, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat_value, num_out_ch_value, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x: Any) -> Any:
            if self.unshuffle_factor > 1:
                feat = functional.pixel_unshuffle(x, self.unshuffle_factor)
            else:
                feat = x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            for index in range(1, self.num_upsample + 1):
                conv_up = getattr(self, f"conv_up{index}")
                feat = self.lrelu(conv_up(functional.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_hr(feat))
            return self.conv_last(feat)

    model = RRDBNet(
        num_in_ch_value=num_in_ch,
        num_out_ch_value=num_out_ch,
        num_feat_value=num_feat,
        num_block_value=num_block,
        num_grow_value=num_grow_ch,
        num_upsample_value=num_upsample,
        unshuffle_factor_value=unshuffle_factor,
    )
    return model, scale_factor


def _build_plksr_network(torch_module: Any, state_dict: dict[str, Any]) -> tuple[Any, int]:
    nn = torch_module.nn

    conv_index_pattern = re.compile(r"^feats\.(\d+)\.weight$")
    indexed_conv_keys = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := conv_index_pattern.match(key)) is not None
    )
    if not indexed_conv_keys:
        raise ValueError("PLKSR checkpoint is missing feats convolution layers.")

    first_conv_key = "feats.0.weight"
    if first_conv_key not in state_dict:
        raise ValueError("PLKSR checkpoint is missing feats.0.weight.")
    num_feat = int(state_dict[first_conv_key].shape[0])

    final_conv_index = indexed_conv_keys[-1]
    final_conv_weight = state_dict[f"feats.{final_conv_index}.weight"]
    if final_conv_weight.ndim != 4:
        raise ValueError("Invalid final PLKSR convolution shape.")
    final_out_channels = int(final_conv_weight.shape[0])
    scale_squared = final_out_channels / 3.0
    scale_factor = int(round(math.sqrt(scale_squared)))
    if scale_factor < 1 or (scale_factor * scale_factor * 3) != final_out_channels:
        raise ValueError(
            "Unable to infer PLKSR upscale factor from final convolution channels "
            f"({final_out_channels})."
        )

    max_feature_index = max(
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"^feats\.(\d+)\.", key)) is not None
    )

    class EA(nn.Module):
        def __init__(self, dim_value: int):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(dim_value, dim_value, 3, 1, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: Any) -> Any:
            return x * self.f(x)

    class PLKConv2d(nn.Module):
        def __init__(self, partial_dim_value: int, kernel_size_value: int):
            super().__init__()
            self.conv = nn.Conv2d(
                partial_dim_value,
                partial_dim_value,
                kernel_size_value,
                1,
                kernel_size_value // 2,
            )

        def forward(self, x: Any) -> Any:
            x1, x2 = torch_module.split(
                x,
                [self.conv.in_channels, x.size(1) - self.conv.in_channels],
                dim=1,
            )
            x1 = self.conv(x1)
            return torch_module.cat((x1, x2), dim=1)

    class DCCM(nn.Sequential):
        def __init__(self, dim_value: int):
            super().__init__(
                nn.Conv2d(dim_value, dim_value * 2, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim_value * 2, dim_value, 3, 1, 1),
            )

    class PLKBlock(nn.Module):
        def __init__(
            self,
            dim_value: int,
            n_split_value: int,
            kernel_size_value: int,
            *,
            use_ea_value: bool,
            norm_groups_value: int,
        ):
            super().__init__()
            self.channel_mixer = DCCM(dim_value)
            self.lk = PLKConv2d(n_split_value, kernel_size_value)
            self.attn = EA(dim_value) if use_ea_value else nn.Identity()
            self.refine = nn.Conv2d(dim_value, dim_value, 1, 1, 0)
            self.norm = nn.GroupNorm(norm_groups_value, dim_value)

        def forward(self, x: Any) -> Any:
            x_skip = x
            x = self.channel_mixer(x)
            x = self.lk(x)
            x = self.attn(x)
            x = self.refine(x)
            x = self.norm(x)
            return x + x_skip

    modules: list[Any] = []
    for index in range(max_feature_index + 1):
        conv_weight_key = f"feats.{index}.weight"
        conv_bias_key = f"feats.{index}.bias"
        if conv_weight_key in state_dict and conv_bias_key in state_dict:
            weight = state_dict[conv_weight_key]
            if weight.ndim != 4:
                raise ValueError(f"Unsupported PLKSR conv shape for {conv_weight_key}: {tuple(weight.shape)}")
            out_channels, in_channels, kernel_h, kernel_w = [int(value) for value in weight.shape]
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_h, kernel_w),
                    stride=1,
                    padding=(kernel_h // 2, kernel_w // 2),
                )
            )
            continue

        block_prefix = f"feats.{index}."
        mixer_weight_key = f"{block_prefix}channel_mixer.0.weight"
        lk_weight_key = f"{block_prefix}lk.conv.weight"
        if mixer_weight_key in state_dict and lk_weight_key in state_dict:
            lk_weight = state_dict[lk_weight_key]
            if lk_weight.ndim != 4:
                raise ValueError(f"Unsupported PLKSR lk conv shape for {lk_weight_key}: {tuple(lk_weight.shape)}")
            n_split = int(lk_weight.shape[0])
            kernel_size = int(lk_weight.shape[-1])
            if n_split < 1 or n_split > num_feat:
                raise ValueError(
                    f"Invalid PLKSR partial channel count for {lk_weight_key}: {n_split} (dim={num_feat})."
                )
            use_ea = f"{block_prefix}attn.f.0.weight" in state_dict
            norm_groups = 4 if num_feat % 4 == 0 else 1
            modules.append(
                PLKBlock(
                    dim_value=num_feat,
                    n_split_value=n_split,
                    kernel_size_value=kernel_size,
                    use_ea_value=use_ea,
                    norm_groups_value=norm_groups,
                )
            )
            continue

        # Preserve feature index alignment for parameterless modules in exported checkpoints.
        modules.append(nn.Identity())

    class RealPLKSRNet(nn.Module):
        def __init__(self, body_modules: list[Any], scale_value: int):
            super().__init__()
            self.scale = int(scale_value)
            self.repeat_op = partial(torch_module.repeat_interleave, repeats=self.scale**2, dim=1)
            self.feats = nn.Sequential(*body_modules)
            self.to_img = nn.PixelShuffle(self.scale)

        def forward(self, x: Any) -> Any:
            x = self.feats(x) + self.repeat_op(x)
            return self.to_img(x)

    model = RealPLKSRNet(modules, scale_factor)
    return model, scale_factor


def _build_upscaler_network(torch_module: Any, state_dict: dict[str, Any]) -> tuple[Any, int]:
    architecture = _detect_upscaler_architecture(state_dict)
    if architecture == "compact":
        return _build_compact_network(torch_module, state_dict)
    if architecture == "rrdb":
        return _build_rrdb_network(torch_module, state_dict)
    if architecture == "plksr":
        return _build_plksr_network(torch_module, state_dict)
    raise ValueError(f"Unsupported upscaler architecture: {architecture}")


def _image_to_tensor(torch_module: Any, image: Image.Image, device: str, dtype: Any) -> Any:
    import numpy as np

    array = np.asarray(image, dtype="float32") / 255.0
    tensor = torch_module.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _tensor_to_image(torch_module: Any, tensor: Any) -> Image.Image:
    sanitized = torch_module.nan_to_num(tensor.detach(), nan=0.0, posinf=1.0, neginf=0.0)
    clipped = sanitized.to(device="cpu", dtype=torch_module.float32).clamp(0.0, 1.0)
    output = clipped.squeeze(0).permute(1, 2, 0).numpy()
    output = (output * 255.0).round().astype("uint8")
    return Image.fromarray(output, mode="RGB")


def _run_tiled(
    torch_module: Any,
    model: Any,
    input_tensor: Any,
    scale_factor: int,
    tile_size: int,
    tile_overlap: int,
) -> Any:
    _, _, height, width = [int(value) for value in input_tensor.shape]
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return model(input_tensor).to(device="cpu", dtype=torch_module.float32)

    output = torch_module.zeros(
        (1, 3, height * scale_factor, width * scale_factor),
        dtype=torch_module.float32,
        device="cpu",
    )

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile_height = min(tile_size, height - y)
            tile_width = min(tile_size, width - x)

            in_y0 = max(y - tile_overlap, 0)
            in_x0 = max(x - tile_overlap, 0)
            in_y1 = min(y + tile_height + tile_overlap, height)
            in_x1 = min(x + tile_width + tile_overlap, width)

            tile = input_tensor[:, :, in_y0:in_y1, in_x0:in_x1]
            pred = model(tile)

            crop_y0 = (y - in_y0) * scale_factor
            crop_x0 = (x - in_x0) * scale_factor
            crop_y1 = crop_y0 + (tile_height * scale_factor)
            crop_x1 = crop_x0 + (tile_width * scale_factor)

            out_y0 = y * scale_factor
            out_x0 = x * scale_factor
            out_y1 = out_y0 + (tile_height * scale_factor)
            out_x1 = out_x0 + (tile_width * scale_factor)

            core = pred[:, :, crop_y0:crop_y1, crop_x0:crop_x1]
            output[:, :, out_y0:out_y1, out_x0:out_x1] = core.to(
                device="cpu",
                dtype=torch_module.float32,
            )

    return output


def run_upscale_test(
    *,
    input_image_path: Path,
    checkpoint_path: Path,
    profile_name: str,
) -> UpscaleResult:
    with Image.open(input_image_path) as source_file:
        source = source_file.convert("RGB")
    return upscale_image(
        image=source,
        checkpoint_path=checkpoint_path,
        profile_name=profile_name,
    )


def upscale_image(
    *,
    image: Image.Image,
    checkpoint_path: Path,
    profile_name: str,
) -> UpscaleResult:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors_file
        except Exception as exc:
            raise RuntimeError(
                "safetensors checkpoint requested but safetensors is not available."
            ) from exc
        checkpoint = load_safetensors_file(str(checkpoint_path), device="cpu")
    else:
        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        try:
            load_parameters = inspect.signature(torch.load).parameters
        except Exception:
            load_parameters = {}
        if "weights_only" in load_parameters:
            load_kwargs["weights_only"] = True
        checkpoint = torch.load(checkpoint_path, **load_kwargs)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(torch, checkpoint))
    architecture = _detect_upscaler_architecture(state_dict)
    policy = resolve_upscale_policy(profile_name, architecture=architecture)

    use_half = device == "cuda" and policy.prefer_half_precision
    dtype = torch.float16 if use_half else torch.float32
    model, scale_factor = _build_upscaler_network(torch, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device)
    if use_half:
        model = model.half()
    model.eval()

    source = image.convert("RGB")
    input_tensor = _image_to_tensor(torch, source, device=device, dtype=dtype)

    cuda_before = cuda_memory_snapshot(torch)
    process_before = process_memory_snapshot()
    started = now_perf()
    with torch.inference_mode():
        output_tensor = _run_tiled(
            torch_module=torch,
            model=model,
            input_tensor=input_tensor,
            scale_factor=scale_factor,
            tile_size=policy.tile_size,
            tile_overlap=policy.tile_overlap,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration_ms = int((now_perf() - started) * 1000)
    cuda_after = cuda_memory_snapshot(torch)
    process_after = process_memory_snapshot()

    output_image = _tensor_to_image(torch, output_tensor)
    return UpscaleResult(
        image=output_image,
        scale_factor=scale_factor,
        device=device,
        precision="fp16" if use_half else "fp32",
        tile_size=policy.tile_size,
        tile_overlap=policy.tile_overlap,
        duration_ms=duration_ms,
        source_width=int(source.width),
        source_height=int(source.height),
        output_width=int(output_image.width),
        output_height=int(output_image.height),
        architecture=architecture,
        cuda_memory_before=cuda_before,
        cuda_memory_after=cuda_after,
        process_memory_before=process_before,
        process_memory_after=process_after,
    )
