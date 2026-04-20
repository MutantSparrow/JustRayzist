from __future__ import annotations

from app.entrypoints import web_entry


def test_web_entry_main_invokes_serve_with_host_and_port(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_serve(host: str, port: int) -> None:
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(web_entry, "serve", fake_serve)
    monkeypatch.setattr("sys.argv", ["justrayzist-web", "--host", "0.0.0.0", "--port", "38888"])

    web_entry.main()

    assert captured == {"host": "0.0.0.0", "port": 38888}
