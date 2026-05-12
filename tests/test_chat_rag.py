from __future__ import annotations

from app.storage.chat_rag import build_chat_document_context, retrieve_chat_documentation


def test_chat_rag_retrieves_relevant_markdown_snippets(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (temp_app_paths.root_dir / "README.md").write_text(
        "# JustRayzist\n\nGeneral startup notes.\n",
        encoding="utf-8",
    )
    (docs_dir / "USAGE.md").write_text(
        "# Usage\n\n"
        "## Clarity\n\n"
        "Clarity is an image refinement step for existing gallery images after generation. "
        "It does not rewrite prompts.\n\n"
        "## Wildcards\n\n"
        "Wildcards insert reusable prompt fragments.\n",
        encoding="utf-8",
    )

    snippets = retrieve_chat_documentation(settings, "what does clarity do?")

    assert snippets
    assert snippets[0].source == "docs/USAGE.md"
    assert snippets[0].heading == "Clarity"
    assert "does not rewrite prompts" in snippets[0].text


def test_chat_rag_skips_prompt_draft_requests(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    (temp_app_paths.root_dir / "README.md").write_text(
        "# Prompt Help\n\n"
        "Good prompts mention subject, setting, and lighting.\n",
        encoding="utf-8",
    )

    assert retrieve_chat_documentation(settings, "write me a prompt about a castle") == []


def test_chat_rag_formats_context_with_sources(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    (temp_app_paths.root_dir / "README.md").write_text(
        "# API\n\n"
        "Open /API to view local API route documentation.\n",
        encoding="utf-8",
    )

    context = build_chat_document_context(settings, "where is the api page?")

    assert "Relevant documentation excerpts" in context
    assert "[README.md > API]" in context
    assert "/API" in context


def test_chat_rag_prefers_assistant_ui_guide_over_api_docs(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "CHAT_ASSISTANT.md").write_text(
        "# Rayzist Chat Assistant Guide\n\n"
        "## Wildcards Drawer Workflow\n\n"
        "To create a wildcard, open the Wildcard drawer on the right, choose the create action, "
        "add one entry per line, then save.\n",
        encoding="utf-8",
    )
    (docs_dir / "USAGE.md").write_text(
        "# API\n\n"
        "## POST /wildcards\n\n"
        "Create one wildcard with a JSON request body through the API route.\n",
        encoding="utf-8",
    )

    snippets = retrieve_chat_documentation(settings, "how do I create wildcards?")
    context = build_chat_document_context(settings, "how do I create wildcards?")

    assert snippets[0].source == "docs/CHAT_ASSISTANT.md"
    assert snippets[0].heading == "Wildcards Drawer Workflow"
    assert "open the Wildcard drawer" in context
    assert "POST /wildcards" not in context


def test_chat_rag_keeps_api_docs_when_user_asks_for_api(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "USAGE.md").write_text(
        "# API\n\n"
        "## POST /wildcards\n\n"
        "Create one wildcard with a JSON request body through the API route.\n",
        encoding="utf-8",
    )

    context = build_chat_document_context(settings, "what API endpoint creates wildcards?")

    assert "docs/USAGE.md > POST /wildcards" in context
    assert "JSON request body" in context


def test_chat_rag_retrieves_creative_rplus_specifics(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "CHAT_ASSISTANT.md").write_text(
        "# Rayzist Chat Assistant Guide\n\n"
        "## R+ Mode\n\n"
        "R+ is an alternate image inference mode for normal Generate jobs. "
        "Creative Mode and R+ can compound. Creative 3 with R+ can produce stronger changes "
        "than the same Creative level in standard generation.\n",
        encoding="utf-8",
    )

    context = build_chat_document_context(settings, "why is crea 3 r+ so strong?")

    assert "docs/CHAT_ASSISTANT.md > R+ Mode" in context
    assert "Creative 3 with R+" in context

    short_context = build_chat_document_context(settings, "what is R+ mode?")
    assert "alternate image inference mode" in short_context


def test_chat_rag_retrieves_gallery_client_scope(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "CHAT_ASSISTANT.md").write_text(
        "# Rayzist Chat Assistant Guide\n\n"
        "## Gallery And Client Scope\n\n"
        "Each browser client gets its own gallery scope on a server. "
        "Changing browser or accessing the same server through localhost, computer name, or LAN IP "
        "can create a different client id and therefore show a different gallery.\n\n"
        "## Gallery Migration And Repair\n\n"
        "/gallery/import-sources lists available source galleries, /gallery/import copies images "
        "from another gallery source, and /gallery/rebuild rebuilds the current client gallery index.\n",
        encoding="utf-8",
    )

    context = build_chat_document_context(settings, "why is my gallery different on localhost vs ip?")

    assert "docs/CHAT_ASSISTANT.md > Gallery And Client Scope" in context
    assert "different client id" in context


def test_chat_rag_retrieves_gallery_migration_tools(temp_app_paths, make_app_settings) -> None:
    settings = make_app_settings(paths=temp_app_paths)
    docs_dir = temp_app_paths.root_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "CHAT_ASSISTANT.md").write_text(
        "# Rayzist Chat Assistant Guide\n\n"
        "## Gallery Migration And Repair\n\n"
        "The API page includes gallery tools for recovery and migration. "
        "/gallery/import-sources lists available source galleries, /gallery/import copies images "
        "from another gallery source into the current client gallery, and /gallery/rebuild rebuilds "
        "the current client gallery index after manual PNG copies.\n",
        encoding="utf-8",
    )

    context = build_chat_document_context(settings, "how do I migrate my gallery from another browser?")

    assert "docs/CHAT_ASSISTANT.md > Gallery Migration And Repair" in context
    assert "/gallery/import" in context
