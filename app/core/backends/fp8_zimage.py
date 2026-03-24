from __future__ import annotations

from app.core.backends.diffusers_zimage import DiffusersZImageBackend
from app.core.pipeline_factory import LoadedZImagePipeline, build_fp8_zimage_pipeline


class Fp8ZImageBackend(DiffusersZImageBackend):
    BACKEND_NAME = "fp8_zimage"

    def _build_pipeline(self) -> LoadedZImagePipeline:
        return build_fp8_zimage_pipeline(
            self._model_pack,
            self._resource_profile(),
        )
