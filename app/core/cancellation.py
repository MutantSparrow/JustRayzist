from __future__ import annotations


class GenerationCancelledError(RuntimeError):
    """Raised when an active generation or upscale job is cancelled."""

