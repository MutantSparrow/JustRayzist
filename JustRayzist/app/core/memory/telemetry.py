from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CudaMemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "max_allocated_bytes": self.max_allocated_bytes,
            "max_reserved_bytes": self.max_reserved_bytes,
        }


@dataclass(frozen=True)
class ProcessMemorySnapshot:
    rss_bytes: int
    vms_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "rss_bytes": self.rss_bytes,
            "vms_bytes": self.vms_bytes,
        }


def now_perf() -> float:
    return time.perf_counter()


def cuda_memory_snapshot(torch_module: Any) -> CudaMemorySnapshot | None:
    if not torch_module.cuda.is_available():
        return None
    device = torch_module.cuda.current_device()
    return CudaMemorySnapshot(
        allocated_bytes=int(torch_module.cuda.memory_allocated(device)),
        reserved_bytes=int(torch_module.cuda.memory_reserved(device)),
        max_allocated_bytes=int(torch_module.cuda.max_memory_allocated(device)),
        max_reserved_bytes=int(torch_module.cuda.max_memory_reserved(device)),
    )


def process_memory_snapshot() -> ProcessMemorySnapshot | None:
    try:
        import os

        import psutil
    except Exception:
        return None

    try:
        info = psutil.Process(os.getpid()).memory_info()
        return ProcessMemorySnapshot(
            rss_bytes=int(getattr(info, "rss", 0)),
            vms_bytes=int(getattr(info, "vms", 0)),
        )
    except Exception:
        return None
