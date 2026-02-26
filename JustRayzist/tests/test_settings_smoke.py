from app.config import load_settings


def test_load_settings_smoke() -> None:
    settings = load_settings()
    assert settings.app_name == "JustRayzist"
    assert settings.paths.models_dir.exists()
    assert settings.paths.outputs_dir.exists()


def test_profile_soak_defaults_are_tuned() -> None:
    high = load_settings(profile_name="high").runtime_profile
    balanced = load_settings(profile_name="balanced").runtime_profile
    constrained = load_settings(profile_name="constrained").runtime_profile

    assert high.default_soak_drift_threshold_mb == 256
    assert balanced.default_soak_drift_threshold_mb == 128
    assert constrained.default_soak_drift_threshold_mb == 64
    assert constrained.default_soak_recycle_every == 24
