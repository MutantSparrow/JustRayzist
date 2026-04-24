from __future__ import annotations

from app.core import seedvr2 as seedvr2_core


def _effective_attempts(*, runtime_preset: str):
    attempts = seedvr2_core._attempts_for_runtime_preset(
        "balanced",
        max_dim=1024,
        attention_mode="sdpa",
        runtime_preset=runtime_preset,
    )
    limit = seedvr2_core._resolve_max_attempts("balanced", len(attempts))
    return attempts[:limit]


def test_current_baseline_does_not_force_tiling_for_balanced_moderate_images() -> None:
    attempts = _effective_attempts(
        runtime_preset=seedvr2_core.SEEDVR2_RUNTIME_PRESET_CURRENT_BASELINE,
    )

    assert len(attempts) == 2
    assert [attempt.vae_encode_tiled for attempt in attempts] == [False, False]
    assert [attempt.vae_decode_tiled for attempt in attempts] == [False, False]


def test_highres_auto_still_forces_tiling_for_balanced_moderate_images() -> None:
    attempts = _effective_attempts(
        runtime_preset=seedvr2_core.SEEDVR2_RUNTIME_PRESET_HIGHRES_AUTO,
    )

    assert len(attempts) == 2
    assert all(attempt.vae_encode_tiled for attempt in attempts)
    assert all(attempt.vae_decode_tiled for attempt in attempts)
    assert {attempt.vae_encode_tile_size for attempt in attempts} == {1024}
    assert {attempt.vae_decode_tile_size for attempt in attempts} == {1024}
    assert {attempt.vae_encode_tile_overlap for attempt in attempts} == {128}
    assert {attempt.vae_decode_tile_overlap for attempt in attempts} == {256}


def test_constrained_profile_uses_decode_overlap_table_while_encode_stays_128() -> None:
    attempts = seedvr2_core._attempts_for_profile(
        "constrained",
        max_dim=4096,
        attention_mode="sdpa",
    )

    assert [attempt.vae_encode_tile_size for attempt in attempts] == [896, 768, 640]
    assert [attempt.vae_decode_tile_size for attempt in attempts] == [896, 768, 640]
    assert [attempt.vae_encode_tile_overlap for attempt in attempts] == [128, 128, 128]
    assert [attempt.vae_decode_tile_overlap for attempt in attempts] == [224, 192, 160]
