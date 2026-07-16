"""WP-5 smoke: Krea2 with a style-reference image (Qwen3VL image conditioning).

Not a pytest (needs GPU + 18GB weights + a reference image on disk). Run manually:
    .venv/Scripts/python.exe scripts/dev_krea2_wp5_smoke.py [--context path/to/ref.png]

Default context: outputs/krea2_smoke.png (produced by scripts/dev_krea2_smoke.py).
"""

from __future__ import annotations

import argparse
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
    def __init__(self, profile):
        self.runtime_profile = profile
        self.resource_tier = profile

        class _Ctl:
            def current(_self):
                return profile

        self.resource_tier_controller = _Ctl()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        type=Path,
        default=ROOT / "outputs" / "krea2_smoke.png",
        help="Reference image for Qwen3VL style-conditioning.",
    )
    parser.add_argument(
        "--prompt",
        default="a majestic wolf standing on a mountain ridge at sunrise, photorealistic",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "krea2_wp5_smoke.png")
    args = parser.parse_args()

    if not args.context.exists():
        raise SystemExit(f"context image not found: {args.context}")

    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    print("context:", args.context)

    pack = load_model_pack(ROOT / "models" / "packs" / "Krea2_Turbo" / "modelpack.yaml")
    profile = RUNTIME_PROFILES["constrained"]
    settings = _Settings(profile)
    backend = create_backend(settings=settings, model_pack=pack, resource_tier=profile)
    print("backend:", type(backend).__name__)

    req = GenerationRequest(
        prompt=args.prompt,
        width=1024,
        height=1024,
        steps=8,
        guidance_scale=0.0,
        seed=4242,
        context_image=args.context,
    )
    t0 = time.time()
    result = backend.generate(req)
    dt = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.out)
    print(f"generated in {dt:.1f}s -> {args.out}")
    print("image size:", result.image.size, "backend:", result.backend, "steps:", result.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
