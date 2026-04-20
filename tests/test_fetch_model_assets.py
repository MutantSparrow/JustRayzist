from __future__ import annotations

from scripts.portable.fetch_model_assets import ASSETS


def test_fetch_model_assets_excludes_deprecated_clarity_checkpoint() -> None:
    assert all(asset.repo_id != "Phips/1xDeJPG_realplksr_otf" for asset in ASSETS)
