from __future__ import annotations

from pathlib import Path
import warnings

from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest
import torch

from app.config import load_settings
from app.core.backends import diffusers_zimage as zimage_module
from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.worker.types import LoraSelection
from app.storage.gallery_index import get_image, sync_outputs_to_gallery


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_q = torch.nn.LayerNorm(2, elementwise_affine=True)
        self.norm_k = torch.nn.LayerNorm(2, elementwise_affine=True)
        self.to_out = torch.nn.Sequential(torch.nn.Linear(6, 6, bias=False))


class _FakeTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _FakeAttention()
        self.attention_norm1 = torch.nn.LayerNorm(6, elementwise_affine=True)
        self.ffn_norm1 = torch.nn.LayerNorm(6, elementwise_affine=True)
        self.adaLN_modulation = torch.nn.Sequential(torch.nn.Linear(3, 24, bias=True))


class _FakeFinalLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adaLN_modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(3, 6, bias=True))
        self.linear = torch.nn.Linear(6, 4, bias=True)


class _FakeTimestepEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(3, 5, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(5, 3, bias=True),
        )


class _FakeTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cap_embedder = torch.nn.Sequential(
            torch.nn.LayerNorm(4, elementwise_affine=True),
            torch.nn.Linear(4, 6, bias=True),
        )
        self.layers = torch.nn.ModuleList([_FakeTransformerBlock()])
        self.context_refiner = torch.nn.ModuleList([_FakeTransformerBlock()])
        self.noise_refiner = torch.nn.ModuleList([_FakeTransformerBlock()])
        self.t_embedder = _FakeTimestepEmbedder()
        self.all_x_embedder = torch.nn.ModuleDict({"2-1": torch.nn.Linear(4, 6, bias=True)})
        self.all_final_layer = torch.nn.ModuleDict({"2-1": _FakeFinalLayer()})


class _FakeLoraPipe:
    def __init__(self) -> None:
        self.load_calls: list[dict[str, object]] = []
        self.adapter_names = None
        self.adapter_weights = None
        self.fuse_calls: list[dict[str, object]] = []
        self.delete_calls: list[list[str]] = []
        self.loaded_adapters: set[str] = set()
        self.disable_calls = 0
        self.enable_calls = 0
        self.transformer = _FakeTransformer()

    def load_lora_weights(self, source, **kwargs) -> None:
        adapter_name = kwargs.get("adapter_name")
        if isinstance(adapter_name, str) and adapter_name in self.loaded_adapters:
            raise ValueError(f"Adapter name {adapter_name} already in use in the model - please select a new adapter name.")
        self.load_calls.append({"source": source, **kwargs})
        if isinstance(adapter_name, str):
            self.loaded_adapters.add(adapter_name)

    def set_adapters(self, adapter_names, adapter_weights=None) -> None:
        self.adapter_names = list(adapter_names)
        self.adapter_weights = list(adapter_weights or [])

    def fuse_lora(self, **kwargs) -> None:
        self.fuse_calls.append(dict(kwargs))

    def unfuse_lora(self, **kwargs) -> None:
        return None

    def delete_adapters(self, adapter_names) -> None:
        names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        self.delete_calls.append(names)
        for name in names:
            self.loaded_adapters.discard(name)
        return None

    def disable_lora(self) -> None:
        self.disable_calls += 1

    def enable_lora(self) -> None:
        self.enable_calls += 1

    def get_list_adapters(self) -> dict[str, list[str]]:
        return {"transformer": sorted(self.loaded_adapters)}


def test_append_lora_triggers_adds_unique_missing_trigger_words() -> None:
    prompt, triggers = DiffusersZImageBackend._append_lora_triggers(
        "portrait of a traveler",
        (
            LoraSelection(
                id="cinematic-style",
                path=Path("cinematic-style.safetensors"),
                trigger_words=("cinematic style", "moody light"),
            ),
            LoraSelection(
                id="portrait-helper",
                path=Path("portrait-helper.safetensors"),
                trigger_words=("moody light", "sharp focus"),
            ),
        ),
    )

    assert prompt == "portrait of a traveler, cinematic style, moody light, sharp focus"
    assert triggers == ("cinematic style", "moody light", "sharp focus")


def test_append_lora_triggers_skips_words_already_in_prompt() -> None:
    prompt, triggers = DiffusersZImageBackend._append_lora_triggers(
        "portrait of a traveler, cinematic style",
        (
            LoraSelection(
                id="cinematic-style",
                path=Path("cinematic-style.safetensors"),
                trigger_words=("cinematic style", "moody light"),
            ),
        ),
    )

    assert prompt == "portrait of a traveler, cinematic style, moody light"
    assert triggers == ("moody light",)


def test_normalize_legacy_zimage_lora_state_dict_rewrites_dotted_lora_keys_and_out_alias() -> None:
    normalized, changed = DiffusersZImageBackend._normalize_legacy_zimage_lora_state_dict(
        {
            "layers.0.attention.out.lora.down.weight": object(),
            "layers.0.attention.out.lora.up.weight": object(),
            "layers.0.attention.to_out.0.alpha": object(),
        }
    )

    assert changed is True
    assert "layers.0.attention.to_out.0.lora_down.weight" in normalized
    assert "layers.0.attention.to_out.0.lora_up.weight" in normalized
    assert "layers.0.attention.to_out.0.alpha" in normalized


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_handles_lora_unet_keys() -> None:
    attention_out_down = torch.ones((2, 3))
    attention_out_up = torch.full((3, 2), 2.0)
    qkv_down = torch.full((2, 4), 3.0)
    qkv_up = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    w1_down = torch.full((2, 5), 5.0)
    w1_up = torch.full((5, 2), 6.0)
    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "eldritch-style",
        {
            "lora_unet_layers_0_attention_out.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_attention_out.lora_down.weight": attention_out_down,
            "lora_unet_layers_0_attention_out.lora_up.weight": attention_out_up,
            "lora_unet_layers_0_attention_qkv.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_attention_qkv.lora_down.weight": qkv_down,
            "lora_unet_layers_0_attention_qkv.lora_up.weight": qkv_up,
            "lora_unet_layers_0_feed_forward_w1.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_feed_forward_w1.lora_down.weight": w1_down,
            "lora_unet_layers_0_feed_forward_w1.lora_up.weight": w1_up,
        },
    )

    assert format_label == "lora_unet"
    assert set(converted.keys()) == {
        "transformer.layers.0.attention.to_out.0.lora_A.weight",
        "transformer.layers.0.attention.to_out.0.lora_B.weight",
        "transformer.layers.0.attention.to_q.lora_A.weight",
        "transformer.layers.0.attention.to_q.lora_B.weight",
        "transformer.layers.0.attention.to_k.lora_A.weight",
        "transformer.layers.0.attention.to_k.lora_B.weight",
        "transformer.layers.0.attention.to_v.lora_A.weight",
        "transformer.layers.0.attention.to_v.lora_B.weight",
        "transformer.layers.0.feed_forward.w1.lora_A.weight",
        "transformer.layers.0.feed_forward.w1.lora_B.weight",
    }
    assert torch.equal(converted["transformer.layers.0.attention.to_out.0.lora_A.weight"], attention_out_down)
    assert torch.equal(converted["transformer.layers.0.attention.to_out.0.lora_B.weight"], attention_out_up)
    assert torch.equal(converted["transformer.layers.0.attention.to_q.lora_A.weight"], qkv_down)
    assert torch.equal(converted["transformer.layers.0.attention.to_k.lora_A.weight"], qkv_down)
    assert torch.equal(converted["transformer.layers.0.attention.to_v.lora_A.weight"], qkv_down)
    q_up, k_up, v_up = qkv_up.chunk(3, dim=0)
    assert torch.equal(converted["transformer.layers.0.attention.to_q.lora_B.weight"], q_up)
    assert torch.equal(converted["transformer.layers.0.attention.to_k.lora_B.weight"], k_up)
    assert torch.equal(converted["transformer.layers.0.attention.to_v.lora_B.weight"], v_up)
    assert torch.equal(converted["transformer.layers.0.feed_forward.w1.lora_A.weight"], w1_down)
    assert torch.equal(converted["transformer.layers.0.feed_forward.w1.lora_B.weight"], w1_up)
    assert not any(".attention.qkv." in key for key in converted)


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_accepts_diffusion_model_prefixed_diffusers_keys() -> None:
    diffusers_prefixed_state_dict = {
        "diffusion_model.layers.0.adaLN_modulation.0.lora_A.weight": torch.ones((2, 2)),
        "diffusion_model.layers.0.adaLN_modulation.0.lora_B.weight": torch.full((2, 2), 2.0),
        "diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.full((2, 2), 3.0),
        "diffusion_model.layers.0.attention.to_q.lora_B.weight": torch.full((2, 2), 4.0),
    }

    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "dark-fantasy-style",
        diffusers_prefixed_state_dict,
    )

    assert format_label == "legacy-zimage"
    assert set(converted.keys()) == {
        "transformer.layers.0.adaLN_modulation.0.lora_A.weight",
        "transformer.layers.0.adaLN_modulation.0.lora_B.weight",
        "transformer.layers.0.attention.to_q.lora_A.weight",
        "transformer.layers.0.attention.to_q.lora_B.weight",
    }
    for key, value in converted.items():
        source_key = f"diffusion_model.{key.removeprefix('transformer.')}"
        assert torch.equal(value, diffusers_prefixed_state_dict[source_key])


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_applies_alpha_scaling() -> None:
    down_weight = torch.ones((2, 2))
    up_weight = torch.full((2, 2), 3.0)
    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "scaled-style",
        {
            "layers.0.attention.out.lora.down.weight": down_weight,
            "layers.0.attention.out.lora.up.weight": up_weight,
            "layers.0.attention.to_out.0.alpha": torch.tensor(4.0),
        },
    )

    assert format_label == "legacy-zimage"
    assert torch.equal(
        converted["transformer.layers.0.attention.to_out.0.lora_A.weight"],
        down_weight * 2.0,
    )
    assert torch.equal(
        converted["transformer.layers.0.attention.to_out.0.lora_B.weight"],
        up_weight,
    )


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_defaults_missing_alpha_to_rank() -> None:
    down_weight = torch.ones((2, 2))
    up_weight = torch.full((2, 2), 3.0)
    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "rank-default-style",
        {
            "layers.0.attention.out.lora.down.weight": down_weight,
            "layers.0.attention.out.lora.up.weight": up_weight,
        },
    )

    assert format_label == "legacy-zimage"
    assert torch.equal(
        converted["transformer.layers.0.attention.to_out.0.lora_A.weight"],
        down_weight,
    )
    assert torch.equal(
        converted["transformer.layers.0.attention.to_out.0.lora_B.weight"],
        up_weight,
    )


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_splits_qkv_without_alpha() -> None:
    qkv_down = torch.full((2, 4), 3.0)
    qkv_up = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "rank-default-qkv-style",
        {
            "layers.0.attention.qkv.lora_down.weight": qkv_down,
            "layers.0.attention.qkv.lora_up.weight": qkv_up,
        },
    )

    assert format_label == "legacy-zimage"
    assert torch.equal(converted["transformer.layers.0.attention.to_q.lora_A.weight"], qkv_down)
    assert torch.equal(converted["transformer.layers.0.attention.to_k.lora_A.weight"], qkv_down)
    assert torch.equal(converted["transformer.layers.0.attention.to_v.lora_A.weight"], qkv_down)
    q_up, k_up, v_up = qkv_up.chunk(3, dim=0)
    assert torch.equal(converted["transformer.layers.0.attention.to_q.lora_B.weight"], q_up)
    assert torch.equal(converted["transformer.layers.0.attention.to_k.lora_B.weight"], k_up)
    assert torch.equal(converted["transformer.layers.0.attention.to_v.lora_B.weight"], v_up)


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_preserves_diffusers_keys() -> None:
    diffusers_state_dict = {
        "transformer.layers.0.attention.to_q.lora_A.weight": torch.ones((2, 2)),
        "transformer.layers.0.attention.to_q.lora_B.weight": torch.full((2, 2), 2.0),
    }

    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "diffusers-style",
        diffusers_state_dict,
    )

    assert format_label == "diffusers-native"
    assert set(converted.keys()) == set(diffusers_state_dict.keys())
    for key, value in diffusers_state_dict.items():
        assert torch.equal(converted[key], value)


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_applies_alpha_to_diffusers_keys() -> None:
    diffusers_state_dict = {
        "diffusion_model.layers.0.attention.to_q.lora_A.weight": torch.ones((2, 2)),
        "diffusion_model.layers.0.attention.to_q.lora_B.weight": torch.full((2, 2), 3.0),
        "diffusion_model.layers.0.attention.to_q.alpha": torch.tensor(4.0),
    }

    converted, format_label = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "diffusers-alpha-style",
        diffusers_state_dict,
    )

    assert format_label == "legacy-zimage"
    assert torch.equal(
        converted["transformer.layers.0.attention.to_q.lora_A.weight"],
        torch.ones((2, 2)) * 2.0,
    )
    assert torch.equal(
        converted["transformer.layers.0.attention.to_q.lora_B.weight"],
        torch.full((2, 2), 3.0),
    )


def test_convert_zimage_legacy_lora_state_dict_to_diffusers_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="unsupported Z-Image key layout near 'layers.0.attention.out.weight'"):
        DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
            "broken-style",
            {"layers.0.attention.out.weight": torch.ones((2, 2))},
        )


def test_prepare_zimage_lora_compat_state_collects_direct_deltas_with_runtime_remaps() -> None:
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=96,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=3,
        n_kv_heads=3,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=32,
        axes_dims=[8, 8, 16],
        axes_lens=[16, 16, 16],
    )
    prepared = DiffusersZImageBackend._prepare_zimage_lora_compat_state(
        "extracted-style",
        {
            "diffusion_model.cap_embedder.0.diff": torch.ones((32,)),
            "diffusion_model.cap_embedder.1.lora_down.weight": torch.ones((2, 32)),
            "diffusion_model.cap_embedder.1.lora_up.weight": torch.ones((96, 2)),
            "diffusion_model.layers.0.attention.q_norm.diff": torch.ones((32,)),
            "diffusion_model.layers.0.attention.k_norm.diff": torch.ones((32,)),
            "diffusion_model.x_embedder.diff_b": torch.ones((96,)),
            "diffusion_model.final_layer.linear.diff_b": torch.ones((16,)),
        },
    )

    assert prepared.format_label == "legacy-zimage-extracted"
    assert prepared.dropped_keys == ()
    assert set(prepared.adapter_state_dict.keys()) == {
        "transformer.cap_embedder.1.lora_A.weight",
        "transformer.cap_embedder.1.lora_B.weight",
    }
    direct_targets = {delta.target_key for delta in prepared.direct_param_deltas}
    assert direct_targets == {
        "transformer.cap_embedder.0.weight",
        "transformer.layers.0.attention.norm_q.weight",
        "transformer.layers.0.attention.norm_k.weight",
        "transformer.all_x_embedder.2-1.bias",
        "transformer.all_final_layer.2-1.linear.bias",
    }
    runtime_targets = {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}
    assert {target.removeprefix("transformer.") for target in direct_targets} <= runtime_targets


def test_load_lora_adapters_always_passes_state_dict_payload(monkeypatch, caplog) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    lora = LoraSelection(
        id="cinematic-style",
        path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/cinematic-style.safetensors"),
        weight=1.25,
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "transformer.layers.0.attn.to_q.lora_A.weight": object(),
            "transformer.layers.0.attn.to_q.lora_B.weight": object(),
        },
    )

    with caplog.at_level("INFO"):
        backend._load_lora_adapters(pipe, (lora,))

    assert len(pipe.load_calls) == 1
    state_dict_source = pipe.load_calls[0]["source"]
    assert isinstance(state_dict_source, dict)
    assert "transformer.layers.0.attn.to_q.lora_A.weight" in state_dict_source
    assert pipe.load_calls[0]["adapter_name"] == "cinematic-style"
    assert pipe.load_calls[0]["local_files_only"] is True
    assert pipe.adapter_names == ["cinematic-style"]
    assert pipe.adapter_weights == [1.25]
    assert pipe.enable_calls == 1
    assert pipe.fuse_calls == []
    assert "Activating LoRAs ids=['cinematic-style'] weights=[1.25]" in caplog.text
    assert "runtime_path=unfused" in caplog.text


def test_load_lora_adapters_warns_and_drops_norm_final_diff(monkeypatch, caplog) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    lora = LoraSelection(
        id="compat-style",
        path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/compat-style.safetensors"),
        weight=1.0,
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "diffusion_model.cap_embedder.1.lora_down.weight": torch.ones((2, 4)),
            "diffusion_model.cap_embedder.1.lora_up.weight": torch.ones((6, 2)),
            "diffusion_model.norm_final.diff": torch.tensor([float("inf"), 0.0, 0.0, 0.0]),
        },
    )

    with caplog.at_level("WARNING"):
        backend._load_lora_adapters(pipe, (lora,))

    assert len(pipe.load_calls) == 1
    assert "compat dropped unsupported extracted tensors" in caplog.text
    assert "norm_final.diff" in caplog.text


def test_load_lora_adapters_applies_and_reverts_direct_deltas_linearly(monkeypatch) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    cap_weight_before = pipe.transformer.cap_embedder[0].weight.detach().clone()
    x_bias_before = pipe.transformer.all_x_embedder["2-1"].bias.detach().clone()

    loras = (
        LoraSelection(
            id="style-one",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-one.safetensors"),
            weight=0.5,
        ),
        LoraSelection(
            id="style-two",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-two.safetensors"),
            weight=1.5,
        ),
    )

    state_dicts = {
        "style-one.safetensors": {
            "diffusion_model.cap_embedder.1.lora_down.weight": torch.ones((2, 4)),
            "diffusion_model.cap_embedder.1.lora_up.weight": torch.ones((6, 2)),
            "diffusion_model.cap_embedder.0.diff": torch.ones((4,)),
            "diffusion_model.x_embedder.diff_b": torch.full((6,), 2.0),
        },
        "style-two.safetensors": {
            "diffusion_model.cap_embedder.1.lora_down.weight": torch.full((2, 4), 3.0),
            "diffusion_model.cap_embedder.1.lora_up.weight": torch.full((6, 2), 4.0),
            "diffusion_model.cap_embedder.0.diff": torch.full((4,), 3.0),
            "diffusion_model.x_embedder.diff_b": torch.full((6,), 5.0),
        },
    }

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": state_dicts[Path(path).name],
    )

    backend._load_lora_adapters(pipe, loras)

    assert torch.allclose(
        pipe.transformer.cap_embedder[0].weight.detach(),
        cap_weight_before + torch.full_like(cap_weight_before, 5.0),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        pipe.transformer.all_x_embedder["2-1"].bias.detach(),
        x_bias_before + torch.full_like(x_bias_before, 8.5),
        atol=1e-6,
        rtol=0.0,
    )

    backend._clear_lora_adapters(pipe, adapter_names=["style-one", "style-two"])

    assert torch.allclose(pipe.transformer.cap_embedder[0].weight.detach(), cap_weight_before, atol=1e-6, rtol=0.0)
    assert torch.allclose(
        pipe.transformer.all_x_embedder["2-1"].bias.detach(),
        x_bias_before,
        atol=1e-6,
        rtol=0.0,
    )
    assert pipe.delete_calls == [["style-one", "style-two"]]
    assert backend._applied_lora_direct_deltas_by_transformer == {}


def test_load_lora_adapters_passes_exact_user_weights_for_stacked_adapters(monkeypatch) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    loras = (
        LoraSelection(
            id="style-one",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-one.safetensors"),
            weight=0.5,
        ),
        LoraSelection(
            id="style-two",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-two.safetensors"),
            weight=1.0,
        ),
        LoraSelection(
            id="style-three",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-three.safetensors"),
            weight=1.5,
        ),
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "transformer.layers.0.attention.to_q.lora_A.weight": torch.ones((2, 2)),
            "transformer.layers.0.attention.to_q.lora_B.weight": torch.ones((2, 2)),
        },
    )

    backend._load_lora_adapters(pipe, loras)

    assert pipe.adapter_names == ["style-one", "style-two", "style-three"]
    assert pipe.adapter_weights == [0.5, 1.0, 1.5]
    assert pipe.enable_calls == 1
    assert pipe.fuse_calls == []


def test_load_lora_adapters_clears_stale_conflicting_adapter_names_before_reload(monkeypatch, caplog) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    pipe.loaded_adapters.add("natalia-lora")
    lora = LoraSelection(
        id="natalia-lora",
        path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/natalia-lora.safetensors"),
        weight=1.0,
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "transformer.layers.0.attention.to_q.lora_A.weight": torch.ones((2, 2)),
            "transformer.layers.0.attention.to_q.lora_B.weight": torch.ones((2, 2)),
        },
    )

    with caplog.at_level("INFO"):
        backend._load_lora_adapters(pipe, (lora,))

    assert pipe.delete_calls == [["natalia-lora"]]
    assert pipe.disable_calls == 1
    assert pipe.enable_calls == 1
    assert pipe.adapter_names == ["natalia-lora"]
    assert pipe.adapter_weights == [1.0]
    assert "Clearing stale LoRA adapters before reload ids=['natalia-lora']" in caplog.text


def test_load_lora_adapters_normalizes_legacy_dotted_lora_keys(monkeypatch) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    lora = LoraSelection(
        id="legacy-style",
        path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/legacy-style.safetensors"),
        weight=0.75,
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "layers.0.attention.out.lora.down.weight": torch.ones((2, 2)),
            "layers.0.attention.out.lora.up.weight": torch.full((2, 2), 2.0),
            "layers.0.attention.to_out.0.alpha": torch.tensor(2.0),
        },
    )

    backend._load_lora_adapters(pipe, (lora,))

    assert len(pipe.load_calls) == 1
    normalized_source = pipe.load_calls[0]["source"]
    assert isinstance(normalized_source, dict)
    assert set(normalized_source.keys()) == {
        "transformer.layers.0.attention.to_out.0.lora_A.weight",
        "transformer.layers.0.attention.to_out.0.lora_B.weight",
    }
    assert pipe.load_calls[0]["adapter_name"] == "legacy-style"
    assert pipe.load_calls[0]["local_files_only"] is True


def test_load_lora_adapters_converts_lora_unet_keys_to_diffusers(monkeypatch, caplog) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeLoraPipe()
    lora = LoraSelection(
        id="eldritch-style",
        path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/eldritch-style.safetensors"),
        weight=0.9,
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "lora_unet_layers_0_attention_out.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_attention_out.lora_down.weight": torch.ones((2, 3)),
            "lora_unet_layers_0_attention_out.lora_up.weight": torch.full((3, 2), 2.0),
            "lora_unet_layers_0_attention_qkv.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_attention_qkv.lora_down.weight": torch.ones((2, 3)),
            "lora_unet_layers_0_attention_qkv.lora_up.weight": torch.arange(18, dtype=torch.float32).reshape(6, 3),
        },
    )

    with caplog.at_level("INFO"):
        backend._load_lora_adapters(pipe, (lora,))

    assert len(pipe.load_calls) == 1
    converted_source = pipe.load_calls[0]["source"]
    assert isinstance(converted_source, dict)
    assert set(converted_source.keys()) == {
        "transformer.layers.0.attention.to_out.0.lora_A.weight",
        "transformer.layers.0.attention.to_out.0.lora_B.weight",
        "transformer.layers.0.attention.to_q.lora_A.weight",
        "transformer.layers.0.attention.to_q.lora_B.weight",
        "transformer.layers.0.attention.to_k.lora_A.weight",
        "transformer.layers.0.attention.to_k.lora_B.weight",
        "transformer.layers.0.attention.to_v.lora_A.weight",
        "transformer.layers.0.attention.to_v.lora_B.weight",
    }
    assert not any(".attention.qkv." in key for key in converted_source)
    assert "Detected lora_unet LoRA format for 'eldritch-style'" in caplog.text


def test_tiny_zimage_transformer_loads_split_qkv_lora_without_unexpected_key_warning(caplog) -> None:
    pytest.importorskip("peft")
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=96,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=3,
        n_kv_heads=3,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=32,
        axes_dims=[8, 8, 16],
        axes_lens=[16, 16, 16],
    )
    converted_source, _ = DiffusersZImageBackend._convert_zimage_legacy_lora_state_dict_to_diffusers(
        "probe",
        {
            "lora_unet_layers_0_attention_out.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_attention_out.lora_down.weight": torch.ones((2, 96)),
            "lora_unet_layers_0_attention_out.lora_up.weight": torch.ones((96, 2)),
            "lora_unet_layers_0_attention_qkv.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_attention_qkv.lora_down.weight": torch.ones((2, 96)),
            "lora_unet_layers_0_attention_qkv.lora_up.weight": torch.ones((288, 2)),
            "lora_unet_layers_0_feed_forward_w1.alpha": torch.tensor(2.0),
            "lora_unet_layers_0_feed_forward_w1.lora_down.weight": torch.ones((2, 96)),
            "lora_unet_layers_0_feed_forward_w1.lora_up.weight": torch.ones((256, 2)),
        },
    )

    with caplog.at_level("WARNING"):
        model.load_lora_adapter(converted_source, prefix="transformer", adapter_name="probe")

    assert "unexpected keys found in the model" not in caplog.text
    assert not any("attention.qkv" in message for message in caplog.messages)



class _FakeMultiAdapterWarningPipe(_FakeLoraPipe):
    def load_lora_weights(self, source, **kwargs) -> None:
        super().load_lora_weights(source, **kwargs)
        if len(self.load_calls) > 1:
            warnings.warn(
                "Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!",
                UserWarning,
                stacklevel=1,
            )


class _FakeOtherWarningPipe(_FakeLoraPipe):
    def load_lora_weights(self, source, **kwargs) -> None:
        super().load_lora_weights(source, **kwargs)
        warnings.warn("some other loader warning", UserWarning, stacklevel=1)


def test_load_lora_adapters_suppresses_expected_peft_multi_adapter_warning(monkeypatch) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeMultiAdapterWarningPipe()
    loras = (
        LoraSelection(
            id="style-one",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-one.safetensors"),
            weight=1.0,
        ),
        LoraSelection(
            id="style-two",
            path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-two.safetensors"),
            weight=0.8,
        ),
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "transformer.layers.0.attention.to_q.lora_A.weight": torch.ones((2, 2)),
            "transformer.layers.0.attention.to_q.lora_B.weight": torch.ones((2, 2)),
        },
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend._load_lora_adapters(pipe, loras)

    assert [call["adapter_name"] for call in pipe.load_calls] == ["style-one", "style-two"]
    assert not any("Already found a `peft_config` attribute" in str(item.message) for item in caught)


def test_load_lora_adapters_does_not_suppress_unrelated_warnings(monkeypatch) -> None:
    backend = object.__new__(DiffusersZImageBackend)
    pipe = _FakeOtherWarningPipe()
    lora = LoraSelection(
        id="style-one",
        path=Path("S:/STABLEDIFFUSION/JustRayzist/models/loras/style-one.safetensors"),
        weight=1.0,
    )

    monkeypatch.setattr(
        zimage_module,
        "load_safetensors_file",
        lambda path, device="cpu": {
            "transformer.layers.0.attention.to_q.lora_A.weight": torch.ones((2, 2)),
            "transformer.layers.0.attention.to_q.lora_B.weight": torch.ones((2, 2)),
        },
    )

    with pytest.warns(UserWarning, match="some other loader warning"):
        backend._load_lora_adapters(pipe, (lora,))
def test_gallery_sync_persists_lora_metadata(monkeypatch, workspace_tmp_path: Path) -> None:
    root = workspace_tmp_path / "gallery-lora-metadata"
    monkeypatch.setenv("JUSTRAYZIST_ROOT", str(root))
    settings = load_settings()

    image_path = settings.paths.outputs_dir / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), color=(120, 140, 180))
    metadata = PngInfo()
    metadata.add_text("timestamp", "2026-03-26T00:00:00+00:00")
    metadata.add_text("prompt", "A cinematic portrait")
    metadata.add_text("application_name", "JustRayzist")
    metadata.add_text("application_version", "0.1.0")
    metadata.add_text("width", "64")
    metadata.add_text("height", "64")
    metadata.add_text("model_pack", "Rayzist_bf16")
    metadata.add_text(
        "loras_json",
        '[{"id":"cinematic-style","name":"cinematic-style","weight":1.0}]',
    )
    metadata.add_text("lora_count", "1")
    image.save(image_path, format="PNG", pnginfo=metadata)

    indexed = sync_outputs_to_gallery(settings)

    assert indexed == 1
    row = get_image(settings, "sample.png")
    assert row is not None
    assert row["loras_json"] == '[{"id":"cinematic-style","name":"cinematic-style","weight":1.0}]'
    assert row["lora_count"] == 1

