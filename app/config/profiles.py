from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    description: str
    target_vram_gb_min: int
    target_vram_gb_max: int
    enable_cpu_offload: bool
    enable_sequential_offload: bool
    guidance_scale_default: float
    steps_default: int
    default_soak_drift_threshold_mb: int
    default_soak_recycle_every: int
    high_reserved_vram_ratio_threshold: float = 0.80
    high_force_mode: str = "auto"


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "high": RuntimeProfile(
        name="high",
        description="24GB-class profile with minimal offload and highest throughput.",
        target_vram_gb_min=20,
        target_vram_gb_max=48,
        enable_cpu_offload=False,
        enable_sequential_offload=False,
        guidance_scale_default=0.0,
        steps_default=9,
        default_soak_drift_threshold_mb=256,
        default_soak_recycle_every=0,
        high_reserved_vram_ratio_threshold=0.80,
        high_force_mode="auto",
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        description="16GB-class profile with moderate offload and stable throughput.",
        target_vram_gb_min=14,
        target_vram_gb_max=24,
        enable_cpu_offload=True,
        enable_sequential_offload=False,
        guidance_scale_default=0.0,
        steps_default=9,
        default_soak_drift_threshold_mb=128,
        default_soak_recycle_every=0,
    ),
    "constrained": RuntimeProfile(
        name="constrained",
        description="12GB-class profile prioritizing reliability with stronger offload.",
        target_vram_gb_min=10,
        target_vram_gb_max=16,
        enable_cpu_offload=True,
        enable_sequential_offload=True,
        guidance_scale_default=0.0,
        steps_default=9,
        default_soak_drift_threshold_mb=64,
        default_soak_recycle_every=24,
    ),
}
