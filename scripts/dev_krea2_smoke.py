"""Dev smoke test: generate one image with the Krea2_Turbo pack through the real app backend.

Not a pytest (needs GPU + 18GB weights). Run manually on a CUDA box:
    .venv-cuda/Scripts/python.exe scripts/dev_krea2_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config.profiles import RUNTIME_PROFILES  # noqa: E402
from app.core.backends import create_backend  # noqa: E402
from app.core.model_registry import load_model_pack  # noqa: E402
from app.core.worker.types import GenerationRequest  # noqa: E402


class _Settings:
    """Minimal settings object exposing what the backend reads."""

    def __init__(self, profile):
        self.runtime_profile = profile
        self.resource_tier = profile

        class _Ctl:
            def current(_self):
                return profile

        self.resource_tier_controller = _Ctl()


def main() -> int:
    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    pack = load_model_pack(ROOT / "models" / "packs" / "Krea2_Turbo" / "modelpack.yaml")
    print("pack:", pack.name, pack.architecture, pack.backend_preference)

    profile = RUNTIME_PROFILES["constrained"]  # 16GB card -> most aggressive offload
    settings = _Settings(profile)
    backend = create_backend(settings=settings, model_pack=pack, resource_tier=profile)
    print("backend:", type(backend).__name__)

    req = GenerationRequest(
        prompt="a red fox sitting in fresh snow, golden hour, photorealistic",
        width=1024,
        height=1024,
        steps=8,
        guidance_scale=0.0,
        seed=1234,
    )
    t0 = time.time()
    result = backend.generate(req)
    dt = time.time() - t0
    out = ROOT / "outputs" / "krea2_smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out)
    print(f"generated in {dt:.1f}s -> {out}")
    print("image size:", result.image.size, "backend:", result.backend, "steps:", result.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
