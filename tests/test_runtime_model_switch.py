"""Weightless tests for the runtime model switch (WP-7).

Uses fake backends + packs and monkeypatches ``create_backend`` so no torch/diffusers/weights are
needed to assert the tier-adaptive lifecycle: keep-resident on high-VRAM tiers, unload-then-load on
constrained tiers, in-flight cancellation before a swap, and full release on recycle.
"""

from __future__ import annotations

from types import SimpleNamespace

import app.core.worker.session as session_module
from app.config.profiles import RUNTIME_PROFILES
from app.core.worker.session import GenerationSession


class _FakeBackend:
    def __init__(self, pack_name: str) -> None:
        self.pack_name = pack_name
        self.cancelled = 0
        self.torn_down = 0

    def cancel_active(self) -> None:
        self.cancelled += 1

    def teardown(self) -> None:
        self.torn_down += 1


def _fake_pack(name: str):
    return SimpleNamespace(name=name, base_name=name, architecture="krea2_turbo")


def _install_fake_create_backend(monkeypatch):
    created: list[_FakeBackend] = []

    def _factory(*, settings, model_pack, resource_tier):
        backend = _FakeBackend(model_pack.name)
        created.append(backend)
        return backend

    monkeypatch.setattr(session_module, "create_backend", _factory)
    return created


def _make_session(tier_name: str):
    tier = RUNTIME_PROFILES[tier_name]
    settings = SimpleNamespace(
        resource_tier_controller=SimpleNamespace(current=lambda: tier),
    )
    return GenerationSession(settings, _fake_pack("Z_Image"), resource_tier=tier)


def test_switch_on_high_tier_keeps_outgoing_resident(monkeypatch) -> None:
    created = _install_fake_create_backend(monkeypatch)
    session = _make_session("high")

    first = session._ensure_backend()
    session.switch_model_pack(_fake_pack("Krea2_Turbo"))
    assert session.model_pack.name == "Krea2_Turbo"
    assert session.stats.switch_count == 1
    # Outgoing was cancelled but NOT torn down (kept resident for instant switch-back).
    assert first.cancelled == 1
    assert first.torn_down == 0

    # Switching back reuses the cached backend rather than building a new one.
    backends_before = len(created)
    session.switch_model_pack(_fake_pack("Z_Image"))
    assert session.model_pack.name == "Z_Image"
    assert len(created) == backends_before  # no new backend built
    assert session._backend is first


def test_switch_on_constrained_tier_releases_before_load(monkeypatch) -> None:
    _install_fake_create_backend(monkeypatch)
    session = _make_session("constrained")

    first = session._ensure_backend()
    session.switch_model_pack(_fake_pack("Krea2_Turbo"))

    # Outgoing was cancelled AND torn down; nothing left resident (never two large models at once).
    assert first.cancelled == 1
    assert first.torn_down == 1
    assert session._resident_backends == {}


def test_switch_to_same_pack_is_noop(monkeypatch) -> None:
    created = _install_fake_create_backend(monkeypatch)
    session = _make_session("high")
    session._ensure_backend()
    count_before = len(created)
    session.switch_model_pack(_fake_pack("Z_Image"))
    assert len(created) == count_before
    assert session.stats.switch_count == 0


def test_recycle_releases_resident_backends(monkeypatch) -> None:
    _install_fake_create_backend(monkeypatch)
    session = _make_session("high")
    session._ensure_backend()
    session.switch_model_pack(_fake_pack("Krea2_Turbo"))
    # One backend now resident in the cache.
    assert session._resident_backends
    resident = next(iter(session._resident_backends.values()))

    session.recycle("test")
    assert session._resident_backends == {}
    assert resident.torn_down == 1
    assert session._backend is None
