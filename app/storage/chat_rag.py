from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import AppSettings

MAX_CHAT_DOC_CONTEXT_CHARS = 3600
MAX_CHAT_DOC_SNIPPETS = 4
MAX_CHAT_DOC_CHUNK_CHARS = 1400

_DOC_RELATIVE_PATHS: tuple[str, ...] = (
    "docs/CHAT_ASSISTANT.md",
    "README.md",
    "docs/USAGE.md",
    "docs/TROUBLESHOOTING.md",
    "docs/PACKAGING.md",
    "docs/PNG_METADATA.md",
)

_SOURCE_SCORE_BOOSTS = {
    "docs/CHAT_ASSISTANT.md": 8,
}

_STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "any",
    "are",
    "can",
    "could",
    "does",
    "for",
    "from",
    "have",
    "help",
    "how",
    "into",
    "just",
    "like",
    "more",
    "that",
    "the",
    "this",
    "use",
    "uses",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

_QUESTION_TERMS = {
    "api",
    "bias",
    "browser",
    "client",
    "clarity",
    "crea",
    "creative",
    "creativity",
    "docs",
    "documentation",
    "gallery",
    "generate",
    "generation",
    "image",
    "img2img",
    "install",
    "ip",
    "lora",
    "localhost",
    "migrate",
    "migration",
    "model",
    "pack",
    "prompt",
    "queue",
    "rebuild",
    "reference",
    "rplus",
    "scheduler",
    "seed",
    "setup",
    "start",
    "upscale",
    "vibrance",
    "vram",
    "wildcard",
}


@dataclass(frozen=True)
class ChatDocSnippet:
    source: str
    heading: str
    text: str
    score: int


def _normalize_terms_text(text: str) -> str:
    return re.sub(r"(?<![a-z0-9])r\+(?![a-z0-9])", "rplus", text.lower())


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_terms_text(text)
    terms = re.findall(r"[a-z0-9][a-z0-9_+/-]{1,}", normalized)
    return [term for term in terms if len(term) >= 3 and term not in _STOP_WORDS]


def _looks_like_doc_question(query: str) -> bool:
    lowered = query.lower()
    if "?" not in query and re.search(
        r"\b(write|make|create|draft|generate|give me)\b.{0,40}\bprompt\b",
        lowered,
    ):
        return False
    if "?" in query:
        return True
    tokens = set(_tokenize(query))
    if tokens & _QUESTION_TERMS:
        return True
    return bool(re.search(r"\b(how|what|where|when|why|can|does|do|is|are)\b", lowered))


def _looks_like_api_question(query: str) -> bool:
    return bool(
        re.search(
            r"\b(api|endpoint|route|payload|request|response|curl|http|automation|integration|script)\b",
            query.lower(),
        )
    )


def _is_api_route_snippet(snippet: ChatDocSnippet) -> bool:
    heading = snippet.heading.strip("` ").lower()
    return bool(re.match(r"^(get|post|patch|delete|put)\s+/", heading))


def _normalize_heading(raw: str) -> str:
    heading = re.sub(r"\s+", " ", raw.strip().strip("#").strip())
    return heading or "Overview"


def _clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_long_chunk(text: str, max_chars: int = MAX_CHAT_DOC_CHUNK_CHARS) -> list[str]:
    cleaned = _clean_markdown(text)
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _chunk_markdown(path: Path, relative_source: str) -> list[ChatDocSnippet]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    snippets: list[ChatDocSnippet] = []
    heading = path.name
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if not content:
            return
        for chunk in _split_long_chunk(content):
            snippets.append(ChatDocSnippet(source=relative_source, heading=heading, text=chunk, score=0))

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = re.match(r"^\s{0,3}(#{1,4})\s+(.+?)\s*$", line)
        if match:
            flush()
            heading = _normalize_heading(match.group(2))
            body = []
            continue
        body.append(line)
    flush()
    return snippets


def _score_snippet(snippet: ChatDocSnippet, query_terms: list[str]) -> int:
    if not query_terms:
        return 0
    title = _normalize_terms_text(snippet.heading)
    text = _normalize_terms_text(snippet.text)
    score = 0
    for term in query_terms:
        if term in title:
            score += 5
        score += min(6, len(re.findall(re.escape(term), text)))
    if score > 0:
        score += _SOURCE_SCORE_BOOSTS.get(snippet.source, 0)
    return score


def _matched_query_terms(snippet: ChatDocSnippet, query_terms: list[str]) -> set[str]:
    haystack = _normalize_terms_text(f"{snippet.heading}\n{snippet.text}")
    return {term for term in query_terms if re.search(re.escape(term), haystack)}


def retrieve_chat_documentation(
    settings: AppSettings,
    query: str,
    *,
    limit: int = MAX_CHAT_DOC_SNIPPETS,
) -> list[ChatDocSnippet]:
    query_text = str(query or "").strip()
    if not query_text or not _looks_like_doc_question(query_text):
        return []

    query_terms = _tokenize(query_text)
    if not query_terms:
        return []

    scored: list[ChatDocSnippet] = []
    for relative in _DOC_RELATIVE_PATHS:
        path = settings.paths.root_dir / relative
        if not path.exists():
            continue
        for snippet in _chunk_markdown(path, relative):
            matched_terms = _matched_query_terms(snippet, query_terms)
            if not _looks_like_api_question(query_text) and len(query_terms) > 1 and len(matched_terms) < 2:
                continue
            score = _score_snippet(snippet, query_terms)
            if score >= 2:
                scored.append(
                    ChatDocSnippet(
                        source=snippet.source,
                        heading=snippet.heading,
                        text=snippet.text,
                        score=score,
                    )
                )
    if not _looks_like_api_question(query_text):
        scored = [snippet for snippet in scored if not _is_api_route_snippet(snippet)]
    scored.sort(key=lambda item: (-item.score, item.source, item.heading))
    return scored[: max(0, int(limit))]


def build_chat_document_context(settings: AppSettings, query: str) -> str:
    snippets = retrieve_chat_documentation(settings, query)
    if not snippets:
        return ""

    parts = [
        "Relevant documentation excerpts for answering app questions. "
        "Use these only when they directly answer the user. If they do not answer it, say the docs do not cover it."
    ]
    remaining = MAX_CHAT_DOC_CONTEXT_CHARS - len(parts[0])
    for snippet in snippets:
        header = f"\n[{snippet.source} > {snippet.heading}]"
        text = snippet.text
        chunk = f"{header}\n{text}"
        if len(chunk) > remaining:
            if remaining > len(header) + 120:
                parts.append(f"{header}\n{text[: remaining - len(header) - 4].rstrip()}...")
            break
        parts.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            break
    return "\n".join(parts).strip()


__all__ = [
    "ChatDocSnippet",
    "MAX_CHAT_DOC_CONTEXT_CHARS",
    "MAX_CHAT_DOC_SNIPPETS",
    "build_chat_document_context",
    "retrieve_chat_documentation",
]
