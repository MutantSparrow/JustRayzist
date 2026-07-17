"""ComfyUI/native -> diffusers state-dict conversion for Krea2-Turbo transformers.

Follows the same pattern as ``zimage._convert_prefixed_fused_zimage_state_dict``: detect the native
checkpoint key layout and remap it to the keys ``diffusers.Krea2Transformer2DModel`` expects.

The native layout is the one produced by the upstream Krea 2 Turbo release and used verbatim by
ComfyUI fp8 repacks (``blocks.N.*``, ``first``, ``last``, ``tmlp``, ``tproj``, ``txtfusion``,
``txtmlp``). The mapping below was derived by aligning, key-by-key and by tensor shape, the
official 430-key ``turbo.safetensors`` header against a freshly instantiated
``Krea2Transformer2DModel`` (430 keys) — it is not inferred.

Verified structural facts (diffusers 0.39.0):
- ``blocks.N`` (28) -> ``transformer_blocks.N``; ``txtfusion.layerwise_blocks/refiner_blocks`` and
  ``txtfusion.projector`` map 1:1 into ``text_fusion.*``.
- attention: ``wq/wk/wv -> to_q/to_k/to_v``, ``wo -> to_out.0``, ``gate -> to_gate``,
  ``qknorm.qnorm.scale/knorm.scale -> norm_q.weight/norm_k.weight``.
- ffn: ``mlp.gate/up/down -> ff.gate/up/down``.
- norms: ``prenorm.scale -> norm1.weight``, ``postnorm.scale -> norm2.weight`` (and the RMSNorm
  ``.scale`` parameter is stored as ``.weight`` in diffusers).
- ``mod.lin`` ``(6*H,)`` -> per-block ``scale_shift_table`` ``(6, H)`` (reshape).
- globals: ``first -> img_in``; ``tmlp.0/tmlp.2 -> time_embed.linear_1/linear_2``;
  ``tproj.1 -> time_mod_proj``; ``txtmlp.0.scale/txtmlp.1/txtmlp.3 ->
  txt_in.norm.weight/linear_1/linear_2``; ``last.linear -> final_layer.linear``,
  ``last.norm.scale -> final_layer.norm.weight``, ``last.modulation.lin ->
  final_layer.scale_shift_table``.

Extra tensors: some fp8 repacks carry two additional ``last.up.weight`` / ``last.down.weight``
matrices that the official checkpoint and the diffusers model do NOT have. They are dropped
(with a warning) rather than force-fit.
"""

from __future__ import annotations

import logging
import re
from typing import Any

LOGGER = logging.getLogger(__name__)

# Native top-level markers that identify a ComfyUI/native Krea2 transformer checkpoint.
_NATIVE_MARKERS = ("blocks.", "txtfusion.", "txtmlp.", "tproj.", "tmlp.")

# Per-block attention/ffn/norm suffix renames (apply within a `blocks.N.` or txtfusion block).
_SUFFIX_RENAMES: tuple[tuple[str, str], ...] = (
    (".attn.wq.weight", ".attn.to_q.weight"),
    (".attn.wk.weight", ".attn.to_k.weight"),
    (".attn.wv.weight", ".attn.to_v.weight"),
    (".attn.wo.weight", ".attn.to_out.0.weight"),
    (".attn.gate.weight", ".attn.to_gate.weight"),
    (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
    (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"),
    (".mlp.gate.weight", ".ff.gate.weight"),
    (".mlp.up.weight", ".ff.up.weight"),
    (".mlp.down.weight", ".ff.down.weight"),
    (".prenorm.scale", ".norm1.weight"),
    (".postnorm.scale", ".norm2.weight"),
)

# Global (non-repeating) key renames.
_GLOBAL_RENAMES: dict[str, str] = {
    "first.weight": "img_in.weight",
    "first.bias": "img_in.bias",
    "tmlp.0.weight": "time_embed.linear_1.weight",
    "tmlp.0.bias": "time_embed.linear_1.bias",
    "tmlp.2.weight": "time_embed.linear_2.weight",
    "tmlp.2.bias": "time_embed.linear_2.bias",
    "tproj.1.weight": "time_mod_proj.weight",
    "tproj.1.bias": "time_mod_proj.bias",
    "txtmlp.0.scale": "txt_in.norm.weight",
    "txtmlp.1.weight": "txt_in.linear_1.weight",
    "txtmlp.1.bias": "txt_in.linear_1.bias",
    "txtmlp.3.weight": "txt_in.linear_2.weight",
    "txtmlp.3.bias": "txt_in.linear_2.bias",
    "last.linear.weight": "final_layer.linear.weight",
    "last.linear.bias": "final_layer.linear.bias",
    "last.norm.scale": "final_layer.norm.weight",
    "last.modulation.lin": "final_layer.scale_shift_table",
}

# Native tensors present in some fp8 repacks but absent from the official checkpoint and the
# diffusers model. Dropped during conversion.
_DROP_KEYS: frozenset[str] = frozenset({"last.up.weight", "last.down.weight"})

_TXTFUSION_PREFIX = "txtfusion."
_TEXT_FUSION_PREFIX = "text_fusion."


def is_native_krea2_transformer(state_dict_keys: Any) -> bool:
    """True if the key set looks like a ComfyUI/native Krea2 transformer (not diffusers-native)."""
    keys = list(state_dict_keys)
    if any(k.startswith("transformer_blocks.") for k in keys):
        return False
    return any(any(k.startswith(m) for m in _NATIVE_MARKERS) for k in keys)


def _apply_suffix_renames(name: str) -> str:
    for src, dst in _SUFFIX_RENAMES:
        if name.endswith(src):
            return name[: -len(src)] + dst
    return name


def convert_native_krea2_transformer_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Convert a native/ComfyUI Krea2 transformer state dict to diffusers key names.

    ``mod.lin`` tensors of shape ``(6*H,)`` are reshaped to ``(6, H)`` for ``scale_shift_table``.
    Unknown keys raise, so a checkpoint that does not match the expected layout fails loudly rather
    than loading a partially-wrong model.
    """
    converted: dict[str, Any] = {}
    dropped: list[str] = []

    for key, tensor in raw_state.items():
        if key in _DROP_KEYS:
            dropped.append(key)
            continue

        # txtfusion.* -> text_fusion.* (then per-block suffix renames apply to the remainder).
        if key.startswith(_TXTFUSION_PREFIX):
            remainder = key[len(_TXTFUSION_PREFIX) :]
            if remainder == "projector.weight":
                converted["text_fusion.projector.weight"] = tensor
                continue
            new_key = _TEXT_FUSION_PREFIX + _apply_suffix_renames(remainder)
            converted[new_key] = tensor
            continue

        # Main transformer blocks: blocks.N.<suffix> -> transformer_blocks.N.<renamed suffix>.
        block_match = re.match(r"^blocks\.(\d+)\.(.+)$", key)
        if block_match:
            idx, suffix = block_match.group(1), block_match.group(2)
            base = f"transformer_blocks.{idx}."
            if suffix == "mod.lin":
                # (6*H,) -> (6, H)
                converted[base + "scale_shift_table"] = tensor.reshape(6, -1)
                continue
            new_suffix = _apply_suffix_renames("." + suffix)[1:]
            converted[base + new_suffix] = tensor
            continue

        # Global keys.
        if key in _GLOBAL_RENAMES:
            converted[_GLOBAL_RENAMES[key]] = tensor
            continue

        raise ValueError(
            f"Unrecognized native Krea2 transformer key '{key}'. "
            "The checkpoint layout does not match the expected ComfyUI/native Krea2 format; "
            "refusing to load a partially-converted model."
        )

    if dropped:
        LOGGER.warning(
            "Dropped %d native Krea2 transformer tensor(s) with no diffusers counterpart: %s",
            len(dropped),
            ", ".join(sorted(dropped)),
        )
    return converted


# --- Qwen3VL text encoder (ComfyUI scaled-fp8) conversion ---
# ComfyUI fp8 encoder repacks store, per quantized linear: a fp8 `.weight`, a scalar `.weight_scale`
# (dequant = weight.float() * weight_scale), and a `.comfy_quant` U8 tensor that is JSON metadata
# ({"format": "float8_e4m3fn", ...}), NOT weight data. Base key names are already standard Qwen3VL
# (`model.layers.*`, `model.visual.*`), so only dequantization + dropping the metadata key is needed.


def is_comfy_scaled_fp8_encoder(state_dict_keys: Any) -> bool:
    """True if the encoder checkpoint uses the ComfyUI scaled-fp8 layout (`.comfy_quant` markers)."""
    return any(str(k).endswith(".comfy_quant") for k in state_dict_keys)


def convert_comfy_scaled_fp8_encoder_state_dict(
    raw_state: dict[str, Any],
    *,
    compute_dtype: Any,
) -> tuple[dict[str, Any], int]:
    """Dequantize a ComfyUI scaled-fp8 Qwen3VL encoder state dict to ``compute_dtype``.

    Returns ``(converted_state_dict, num_dequantized)``. ``.comfy_quant`` metadata tensors and
    ``.weight_scale`` scalars are consumed (not emitted); each fp8 ``.weight`` with a matching scale
    is dequantized to ``compute_dtype``; all other tensors are cast to ``compute_dtype`` if floating.
    """
    converted: dict[str, Any] = {}
    dequantized = 0

    for key, tensor in raw_state.items():
        if key.endswith(".comfy_quant") or key.endswith(".weight_scale"):
            continue

        # Prefix remap: the ComfyUI checkpoint nests everything under `model.` — `model.visual.*`
        # is the vision tower (-> `visual.*`) and the remaining `model.*` is the LM
        # (-> `language_model.*`), matching diffusers/transformers Qwen3VLModel.
        new_key = _remap_qwen3vl_encoder_key(key)

        scale = raw_state.get(f"{key}_scale") if key.endswith(".weight") else None
        if scale is not None and hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            # Dequantize: fp8 weight * scalar scale -> compute dtype.
            converted[new_key] = tensor.to(dtype=compute_dtype) * scale.to(dtype=compute_dtype)
            dequantized += 1
            continue

        if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            converted[new_key] = tensor.to(dtype=compute_dtype)
        else:
            converted[new_key] = tensor

    return converted, dequantized


def _remap_qwen3vl_encoder_key(key: str) -> str:
    if key.startswith("model.visual."):
        return "visual." + key[len("model.visual.") :]
    if key.startswith("model."):
        return "language_model." + key[len("model.") :]
    return key
