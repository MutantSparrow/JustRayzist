from app.core.worker.types import GenerationRequest

__all__ = ["GenerationRequest", "GenerationSession", "SessionStats"]


def __getattr__(name: str):
    if name == "GenerationSession":
        from app.core.worker.session import GenerationSession

        return GenerationSession
    if name == "SessionStats":
        from app.core.worker.session import SessionStats

        return SessionStats
    raise AttributeError(f"module 'app.core.worker' has no attribute {name!r}")
