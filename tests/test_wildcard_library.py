from __future__ import annotations

import pytest

from app.storage.wildcard_library import (
    create_wildcard,
    delete_wildcard,
    get_wildcard,
    get_wildcard_by_token,
    list_wildcards,
    update_wildcard,
)


def test_create_update_and_delete_wildcard_preserves_hidden_id(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    created = create_wildcard(
        settings,
        display_name="Picturesque Locations",
        token="picturesque locations",
        content_text="a cabin in the Schwarzwald\n\n a chalet in the French Alps \n",
    )

    assert created["display_name"] == "Picturesque Locations"
    assert created["token"] == "picturesque-locations"
    assert created["placeholder"] == "__picturesque-locations__"
    assert created["entry_count"] == 2

    stored = get_wildcard(settings, created["id"])
    assert stored is not None
    assert stored["entries"] == ["a cabin in the Schwarzwald", "a chalet in the French Alps"]

    updated = update_wildcard(
        settings,
        wildcard_id=created["id"],
        display_name="Picturesque Spots",
        token="picturesque-spots",
        content_text="a cabin in the Schwarzwald\na white sandy beach in Bora-Bora",
    )

    assert updated["id"] == created["id"]
    assert updated["display_name"] == "Picturesque Spots"
    assert updated["token"] == "picturesque-spots"
    assert updated["placeholder"] == "__picturesque-spots__"
    assert get_wildcard_by_token(settings, "picturesque locations") is None
    assert get_wildcard_by_token(settings, "picturesque-spots") is not None

    listed = list_wildcards(settings)
    assert [item["id"] for item in listed] == [created["id"]]

    deleted = delete_wildcard(settings, created["id"])
    assert deleted == {"id": created["id"], "deleted": True}
    assert list_wildcards(settings) == []


def test_wildcard_library_rejects_duplicate_tokens_and_empty_content(
    temp_app_paths,
    make_app_settings,
) -> None:
    settings = make_app_settings(paths=temp_app_paths)

    first = create_wildcard(
        settings,
        display_name="Landscapes",
        token="landscapes",
        content_text="rolling hills\nfoggy valley",
    )
    second = create_wildcard(
        settings,
        display_name="Portrait Backdrops",
        token="portrait-backdrops",
        content_text="studio backdrop\nwindow light",
    )

    with pytest.raises(ValueError, match="already exists"):
        create_wildcard(
            settings,
            display_name="Duplicate",
            token="landscapes",
            content_text="duplicate entry",
        )

    with pytest.raises(ValueError, match="already exists"):
        update_wildcard(
            settings,
            wildcard_id=second["id"],
            display_name="Portrait Backdrops",
            token="landscapes",
            content_text="studio backdrop\nwindow light",
        )

    with pytest.raises(ValueError, match="at least one non-empty line"):
        update_wildcard(
            settings,
            wildcard_id=first["id"],
            display_name="Landscapes",
            token="landscapes",
            content_text=" \n \r\n ",
        )
