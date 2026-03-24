from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file


FP8_DTYPE = torch.float8_e4m3fn
BF16_DTYPE = torch.bfloat16


def _default_paths(root: Path) -> tuple[Path, Path, Path]:
    input_path = root / "models" / "packs" / "Rayzist_bf16" / "weights" / "Rayzist.v1.0.safetensors"
    mixed_output = root / "models" / "packs" / "Rayzist_fp8_mixed" / "weights" / "Rayzist.v1.0.fp8_e4m3fn.mixed.safetensors"
    full_output = root / "models" / "packs" / "Rayzist_fp8_full" / "weights" / "Rayzist.v1.0.fp8_e4m3fn.full.safetensors"
    return input_path, mixed_output, full_output


def _is_mixed_bf16_tensor(key: str, tensor: torch.Tensor) -> bool:
    normalized = key.strip().lower()
    if not torch.is_floating_point(tensor):
        return True
    if tensor.ndim < 2:
        return True
    if normalized.endswith(".bias") or normalized.endswith("_bias"):
        return True
    if "norm" in normalized:
        return True
    if "token" in normalized:
        return True
    return False


def convert_transformer_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, torch.Tensor]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"mixed", "full"}:
        raise ValueError(f"Unsupported conversion mode '{mode}'. Expected 'mixed' or 'full'.")

    converted: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            converted[key] = tensor
            continue
        if normalized_mode == "mixed" and _is_mixed_bf16_tensor(key, tensor):
            converted[key] = tensor.to(dtype=BF16_DTYPE)
            continue
        converted[key] = tensor.to(dtype=FP8_DTYPE)
    return converted


def summarize_dtypes(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    counter = Counter(str(tensor.dtype) for tensor in state_dict.values())
    return dict(sorted(counter.items()))


def convert_checkpoint(
    *,
    input_path: Path,
    output_path: Path,
    mode: str,
) -> dict[str, int]:
    state_dict = load_file(str(input_path))
    converted = convert_transformer_state_dict(state_dict, mode=mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    return summarize_dtypes(converted)


def _write_checkpoint_pair(
    *,
    input_path: Path,
    mixed_output: Path,
    full_output: Path,
) -> list[tuple[str, Path, dict[str, int]]]:
    state_dict = load_file(str(input_path))
    results: list[tuple[str, Path, dict[str, int]]] = []
    for mode, output_path in (("mixed", mixed_output), ("full", full_output)):
        converted = convert_transformer_state_dict(state_dict, mode=mode)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(converted, str(output_path))
        results.append((mode, output_path, summarize_dtypes(converted)))
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    input_path, mixed_output, full_output = _default_paths(root)
    parser = argparse.ArgumentParser(
        description="Convert the Rayzist BF16 transformer checkpoint into FP8 E4M3 checkpoint variants.",
    )
    parser.add_argument("--input", type=Path, default=input_path, help="Source transformer safetensors path.")
    parser.add_argument(
        "--mixed-output",
        type=Path,
        default=mixed_output,
        help="Destination safetensors path for the mixed FP8 checkpoint.",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=full_output,
        help="Destination safetensors path for the full FP8 checkpoint.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).resolve()
    mixed_output = Path(args.mixed_output).resolve()
    full_output = Path(args.full_output).resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing source checkpoint: {input_path}")

    results = _write_checkpoint_pair(
        input_path=input_path,
        mixed_output=mixed_output,
        full_output=full_output,
    )
    print(f"Source: {input_path}")
    for mode, output_path, dtype_counts in results:
        print(f"{mode}: {output_path}")
        print(f"  dtype_counts={dtype_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
