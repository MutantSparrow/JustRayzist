from __future__ import annotations

import json

from app.storage.chat_history import (
    MAX_CHAT_EXCHANGES,
    append_chat_exchange,
    chat_history_path,
    chat_messages_for_context,
    clear_chat_history,
    load_chat_history,
)
from app.storage.chat_context import DEFAULT_CHAT_CONTEXT, chat_context_path, load_chat_context


def test_chat_history_appends_numbered_exchanges(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    history = append_chat_exchange(
        settings,
        "Example-Client",
        user_content="hello",
        assistant_content="hi",
    )
    history = append_chat_exchange(
        settings,
        "Example-Client",
        user_content="again",
        assistant_content="second",
    )

    assert history["owner_id"] == "example-client"
    assert history["next_number"] == 5
    assert [item["user"]["number"] for item in history["exchanges"]] == [1, 3]
    assert [item["assistant"]["number"] for item in history["exchanges"]] == [2, 4]
    assert chat_history_path(settings, "Example-Client").exists()


def test_chat_history_trims_to_latest_500_without_resetting_numbers(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    history = {}
    for index in range(MAX_CHAT_EXCHANGES + 2):
        history = append_chat_exchange(
            settings,
            "Example-Client",
            user_content=f"user {index}",
            assistant_content=f"assistant {index}",
        )

    assert history["exchange_count"] == MAX_CHAT_EXCHANGES
    assert history["next_number"] == ((MAX_CHAT_EXCHANGES + 2) * 2) + 1
    assert history["exchanges"][0]["user"]["content"] == "user 2"
    assert history["exchanges"][-1]["assistant"]["content"] == f"assistant {MAX_CHAT_EXCHANGES + 1}"


def test_chat_context_skips_error_assistant_messages(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    append_chat_exchange(settings, "Example-Client", user_content="first", assistant_content="failed", assistant_error=True)
    history = append_chat_exchange(settings, "Example-Client", user_content="second", assistant_content="ok")

    assert chat_messages_for_context(history, max_exchanges=10) == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]


def test_chat_context_skips_stale_clarity_prompt_misinformation(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    history = append_chat_exchange(
        settings,
        "Example-Client",
        user_content="what does clarity do",
        assistant_content="Clarity refines or expands prompts. It's a separate tool that runs after generation.",
    )

    assert chat_messages_for_context(history, max_exchanges=10) == [
        {"role": "user", "content": "what does clarity do"},
    ]


def test_chat_history_strips_leaked_action_markup_from_assistant_content(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    history = append_chat_exchange(
        settings,
        "Example-Client",
        user_content="what does clarity do",
        assistant_content=(
            "Clarity works on existing images.\n\n"
            "<rayzist-actions>{\"actions\":[{\"type\":\"set_prompt\",\"prompt\":\"bad visible json\"}]}"
        ),
        assistant_actions=[{"type": "set_prompt", "prompt": "bad visible json"}],
    )

    assistant = history["exchanges"][0]["assistant"]
    assert assistant["content"] == "Clarity works on existing images."
    assert assistant["actions"] == [{"type": "set_prompt", "prompt": "bad visible json", "label": "Use Prompt"}]


def test_chat_history_preserves_valid_assistant_actions(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    history = append_chat_exchange(
        settings,
        "Example-Client",
        user_content="make prompt",
        assistant_content="Use this.",
        assistant_actions=[
            {"type": "set_prompt", "label": "Use", "prompt": "rainy neon street"},
            {"type": "open_route", "href": "/API"},
            {"type": "open_route", "href": "https://example.com"},
        ],
    )

    actions = history["exchanges"][0]["assistant"]["actions"]
    assert actions == [
        {"type": "set_prompt", "prompt": "rainy neon street", "label": "Use"},
        {"type": "open_route", "href": "/API", "label": "Open API"},
    ]
    assert load_chat_history(settings, "Example-Client")["exchanges"][0]["assistant"]["actions"] == actions


def test_clear_chat_history_removes_file(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    append_chat_exchange(settings, "Example-Client", user_content="hello", assistant_content="hi")
    path = chat_history_path(settings, "Example-Client")
    assert path.exists()

    history = clear_chat_history(settings, "Example-Client")

    assert not path.exists()
    assert history["next_number"] == 1
    assert load_chat_history(settings, "Example-Client")["exchanges"] == []


def test_chat_history_ignores_corrupt_json(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    path = chat_history_path(settings, "Example-Client")
    path.write_text("{not-json", encoding="utf-8")

    history = load_chat_history(settings, "Example-Client")

    assert history["owner_id"] == "example-client"
    assert history["next_number"] == 1
    assert history["exchange_count"] == 0

    path.write_text(json.dumps({"next_number": 7, "exchanges": []}), encoding="utf-8")
    assert load_chat_history(settings, "Example-Client")["next_number"] == 7


def test_default_chat_context_file_is_created(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    context = load_chat_context(settings)

    assert "Just Rayzist" in context
    assert "/API" in context
    assert "Clarity is an image refinement step" in context
    assert "does not rewrite, expand, clean, or improve prompts" in context
    assert "explain the visible UI workflow first" in context
    assert "To create a wildcard in the UI" in context
    assert "Use Clarity when the user wants a prompt cleaned up or expanded" not in context
    assert context == DEFAULT_CHAT_CONTEXT.strip()
    assert chat_context_path(settings).exists()


def test_chat_context_migrates_old_clarity_prompt_guidance(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    path = chat_context_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Core workflow:\n"
        "3. Use Clarity when the user wants a prompt cleaned up or expanded.\n",
        encoding="utf-8",
    )

    context = load_chat_context(settings)

    assert "Use Clarity when the user wants a prompt cleaned up or expanded" not in context
    assert "Use Prompt Enhancer when the user wants the prompt expanded before generation" in context
    assert "Clarity is an image refinement step for existing gallery images" in context
    assert "To create a wildcard in the UI" in context
