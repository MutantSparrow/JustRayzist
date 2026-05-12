from __future__ import annotations

from pathlib import Path

from app.config.settings import AppSettings

MAX_CHAT_CONTEXT_CHARS = 6000

DEFAULT_CHAT_CONTEXT = """Just Rayzist app context:

Just Rayzist is a local image generation app for Rayzist model packs. The main prompt box controls text to image generation. The Generate button queues an image job using the current prompt, resolution, seed mode, prompt enhancer state, R+ state, reference image state, and active LoRAs.

Core workflow:
1. Write a clear prompt in the main prompt box.
2. Add subject, setting, style, camera, lighting, material, and mood details.
3. Use Prompt Enhancer when the user wants the prompt expanded before generation.
4. Use the LoRA drawer to apply style or subject adapters.
5. Use the Wildcards drawer to insert reusable prompt placeholders.
6. Use the Gallery to inspect, favorite, reuse prompts, upscale, or run Clarity on generated images.

Clarity:
Clarity is an image refinement step for existing gallery images. It runs after image generation on a selected image. It sharpens and refines image detail through the clarity pipeline. It does not rewrite, expand, clean, or improve prompts. Prompt rewriting is handled by Prompt Enhancer, not Clarity.

Prompt guidance:
Good prompts usually name the subject first, then setting, lighting, camera or composition, material details, color mood, and style constraints. Keep important words near the front. Avoid vague filler. Use commas for prompt phrases. Mention unwanted traits only if the app has a negative prompt control available in the visible UI.

Useful places:
/API opens the local API reference page.
The LoRA drawer opens from the right with installed adapters.
The Wildcards drawer opens from the right with reusable prompt fragments.
Generated images are shown in the Gallery and are also stored in the configured outputs folder.

UI-first behavior:
When the user asks how to use the app, explain the visible UI workflow first. Do not send users to /API unless they specifically ask for API routes, automation, integrations, scripts, or raw request payloads.

Wildcard drawer workflow:
To create a wildcard in the UI, open the Wildcard drawer on the right, choose the create or add action, enter a friendly display name, confirm or edit the prompt token, add one entry per line, then save. Use the wildcard token in the main prompt box to expand those entries during generation.

Chat behavior:
Chat waits behind active generation work, but it does not count against the image generation queue cap. Chat cannot inspect files, generated images, or current UI state unless that state is provided in the chat context.
"""


def chat_context_path(settings: AppSettings) -> Path:
    return settings.paths.data_dir / "chat" / "context.md"


def ensure_chat_context(settings: AppSettings) -> Path:
    path = chat_context_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"{DEFAULT_CHAT_CONTEXT.rstrip()}\n", encoding="utf-8")
    else:
        _migrate_chat_context(path)
    return path


def _migrate_chat_context(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return
    changed = False
    wrong = "3. Use Clarity when the user wants a prompt cleaned up or expanded."
    if wrong in text:
        fixed = (
            "3. Use Prompt Enhancer when the user wants the prompt expanded before generation.\n"
            "\n"
            "Clarity:\n"
            "Clarity is an image refinement step for existing gallery images. It runs after image generation on a "
            "selected image. It sharpens and refines image detail through the clarity pipeline. It does not rewrite, "
            "expand, clean, or improve prompts. Prompt rewriting is handled by Prompt Enhancer, not Clarity."
        )
        text = text.replace(wrong, fixed)
        changed = True
    if "Wildcard drawer workflow:" not in text:
        text = (
            text.rstrip()
            + "\n\n"
            + "UI-first behavior:\n"
            + "When the user asks how to use the app, explain the visible UI workflow first. Do not send users to /API "
            + "unless they specifically ask for API routes, automation, integrations, scripts, or raw request payloads.\n\n"
            + "Wildcard drawer workflow:\n"
            + "To create a wildcard in the UI, open the Wildcard drawer on the right, choose the create or add action, "
            + "enter a friendly display name, confirm or edit the prompt token, add one entry per line, then save. Use "
            + "the wildcard token in the main prompt box to expand those entries during generation.\n"
        )
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")


def load_chat_context(settings: AppSettings) -> str:
    path = ensure_chat_context(settings)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        text = DEFAULT_CHAT_CONTEXT
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > MAX_CHAT_CONTEXT_CHARS:
        return text[:MAX_CHAT_CONTEXT_CHARS].rstrip()
    return text


__all__ = [
    "DEFAULT_CHAT_CONTEXT",
    "MAX_CHAT_CONTEXT_CHARS",
    "chat_context_path",
    "ensure_chat_context",
    "load_chat_context",
]
