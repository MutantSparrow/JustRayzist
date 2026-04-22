from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from email import policy
from email.parser import BytesFeedParser
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Literal

from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from app.api.api_manifest import api_manifest_payload
from app.api.inference_service import InferenceService
from app.config import load_settings
from app.core.cancellation import GenerationCancelledError
from app.core.logging import configure_logging
from app.core.model_registry import ModelPackValidationError
from app.storage.lora_library import DEFAULT_LORA_WEIGHT, DEFAULT_MAX_ACTIVE_LORAS, MAX_LORA_WEIGHT, MIN_LORA_WEIGHT
from app.storage.gallery_index import ensure_gallery_schema

configure_logging()
LOGGER = logging.getLogger(__name__)
settings = load_settings()
inference = InferenceService(settings=settings)
_MULTIPART_TEXT_LIMIT_BYTES = 256 * 1024
_LORA_UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024 * 1024
_LORA_THUMBNAIL_LIMIT_BYTES = 10 * 1024 * 1024
_IMG2IMG_UPLOAD_LIMIT_BYTES = 32 * 1024 * 1024
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
_LOCALHOST_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


class _RevisionBroadcaster:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._revision = 0

    def current(self) -> int:
        with self._condition:
            return self._revision

    def notify(self) -> int:
        with self._condition:
            self._revision += 1
            self._condition.notify_all()
            return self._revision

    def wait_for_change(self, current_revision: int, timeout_seconds: float = 20.0) -> int:
        with self._condition:
            if self._revision != current_revision:
                return self._revision
            self._condition.wait(timeout_seconds)
            return self._revision


_LORA_LIBRARY_EVENTS = _RevisionBroadcaster()
_WILDCARD_LIBRARY_EVENTS = _RevisionBroadcaster()


def _notify_lora_library_changed() -> None:
    _LORA_LIBRARY_EVENTS.notify()


def _notify_wildcard_library_changed() -> None:
    _WILDCARD_LIBRARY_EVENTS.notify()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_gallery_schema(settings)
    inference.sync_gallery()
    inference.start_gallery_color_cache_rebuild()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


def _shutdown_server_process(delay_seconds: float = 0.35) -> None:
    def _kill() -> None:
        time.sleep(max(delay_seconds, 0.0))
        os._exit(0)

    threading.Thread(target=_kill, daemon=True, name="justrayzist-shutdown").start()


def _build_server_restart_exec_args() -> tuple[str, list[str], dict[str, str], str]:
    host = str(os.environ.get("JUSTRAYZIST_SERVER_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = str(os.environ.get("JUSTRAYZIST_SERVER_PORT", "37717")).strip() or "37717"
    verbose_logs = str(os.environ.get("JUSTRAYZIST_SERVER_VERBOSE_LOGS", "0")).strip() == "1"
    root_dir = str(settings.paths.root_dir)
    env = os.environ.copy()
    env["JUSTRAYZIST_ROOT"] = root_dir
    executable = sys.executable
    if getattr(sys, "frozen", False):
        args = [executable, "--host", host, "--port", port]
    else:
        args = [executable, "-m", "app.cli.main", "serve", "--host", host, "--port", port]
    if verbose_logs:
        args.append("--verbose-logs")
    return executable, args, env, root_dir


def _restart_server_process(delay_seconds: float = 0.35) -> None:
    executable, args, env, root_dir = _build_server_restart_exec_args()

    def _restart() -> None:
        time.sleep(max(delay_seconds, 0.0))
        try:
            os.chdir(root_dir)
            os.execve(executable, args, env)
        except Exception:
            LOGGER.exception("Server restart failed.")
            os._exit(1)

    threading.Thread(target=_restart, daemon=True, name="justrayzist-restart").start()


class LoraSelectionRequest(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    weight: float = Field(default=DEFAULT_LORA_WEIGHT, ge=MIN_LORA_WEIGHT, le=MAX_LORA_WEIGHT)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int | None = Field(default=None)
    pack: str | None = Field(default=None)
    job_id: str | None = Field(default=None, max_length=255)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] | None = Field(default=None)
    inference_process: Literal["standard", "rplus"] = Field(default="standard")
    enhance_prompt: bool = Field(default=False)
    procedural_creativity: int = Field(default=0, ge=0, le=3)
    rplus_vibrance: float = Field(default=0.0)
    rplus_initial_bias_level: float = Field(default=0.0)
    loras: list[LoraSelectionRequest] = Field(default_factory=list, max_length=DEFAULT_MAX_ACTIVE_LORAS)

    @field_validator("width", "height")
    @classmethod
    def _validate_multiple_of_16(cls, value: int) -> int:
        if value % 16 != 0:
            raise ValueError("Dimension must be a multiple of 16.")
        return value

    @field_validator("loras")
    @classmethod
    def _validate_unique_loras(cls, value: list[LoraSelectionRequest]) -> list[LoraSelectionRequest]:
        ids = [str(item.id).strip().lower() for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("LoRA ids must be unique within a request.")
        return value


class WildcardUpsertRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=128)
    token: str = Field(min_length=1, max_length=128)
    content_text: str = Field(min_length=1, max_length=200_000)


class WildcardSuggestionRequest(BaseModel):
    theme: str = Field(min_length=1, max_length=500)
    format_example: str = Field(min_length=1, max_length=1000)
    seed: int | None = Field(default=None)
    pack: str | None = Field(default=None)
    existing_entries: list[str] = Field(default_factory=list, max_length=5000)


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    pack: str | None = Field(default=None)
    job_id: str | None = Field(default=None, max_length=255)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] | None = Field(default=None)
    enhance_prompt: bool = Field(default=False)


class ClarityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    pack: str | None = Field(default=None)
    job_id: str | None = Field(default=None, max_length=255)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] | None = Field(default=None)
    enhance_prompt: bool = Field(default=False)


class Img2ImgRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    pack: str | None = Field(default=None)
    job_id: str | None = Field(default=None, max_length=255)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] | None = Field(default=None)
    enhance_prompt: bool = Field(default=False)
    similarity: float = Field(default=0.80, ge=0.0, le=1.0)
    loras: list[LoraSelectionRequest] = Field(default_factory=list, max_length=DEFAULT_MAX_ACTIVE_LORAS)

    @field_validator("loras")
    @classmethod
    def _validate_unique_loras(cls, value: list[LoraSelectionRequest]) -> list[LoraSelectionRequest]:
        ids = [str(item.id).strip().lower() for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("LoRA ids must be unique within a request.")
        return value


class DeleteConfirmRequest(BaseModel):
    confirm: str = Field(min_length=1, max_length=32)


class ImportGalleryRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=255)
    dry_run: bool = Field(default=False)


class BulkDownloadRequest(BaseModel):
    filenames: list[str] = Field(min_length=1, max_length=200)

    @field_validator("filenames")
    @classmethod
    def _validate_filenames(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one filename is required.")
        cleaned = [str(item or "").strip() for item in value if str(item or "").strip()]
        if not cleaned:
            raise ValueError("At least one filename is required.")
        return cleaned


class CancelClientJobRequest(BaseModel):
    job_id: str | None = Field(default=None, max_length=255)


class FavoriteImageRequest(BaseModel):
    favorite: bool = Field(default=True)


def _resolve_owner_id(client_header: str | None, client_query: str | None = None) -> str:
    token = str(client_query or client_header or "").strip()
    if not token:
        raise HTTPException(
            status_code=400,
            detail=(
                "Missing client id. Send header 'X-JustRayzist-Client' "
                "or query parameter 'client_id'."
            ),
        )
    try:
        return InferenceService.sanitize_owner_id(token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid client id: {exc}") from exc


def _format_upload_limit(limit_bytes: int) -> str:
    units = (
        (1024 * 1024 * 1024, "GiB"),
        (1024 * 1024, "MiB"),
        (1024, "KiB"),
    )
    for divisor, label in units:
        if limit_bytes >= divisor and limit_bytes % divisor == 0:
            return f"{limit_bytes // divisor} {label}"
    return f"{limit_bytes} bytes"


class _MultipartSizeLimitExceeded(MultiPartException):
    pass


class _LimitedMultipartParser(MultiPartParser):
    def __init__(
        self,
        headers: Headers,
        stream: Any,
        *,
        file_limits: dict[str, int],
        max_files: int,
        max_fields: int,
        max_text_part_size: int,
    ) -> None:
        self._file_limits = {str(name): int(limit) for name, limit in file_limits.items()}
        self._current_file_limit: int | None = None
        self._current_file_size = 0
        super().__init__(
            headers,
            stream,
            max_files=max_files,
            max_fields=max_fields,
            max_part_size=max_text_part_size,
        )

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._current_file_limit = None
        self._current_file_size = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        if self._current_part.file is None:
            return
        self._current_file_limit = self._file_limits.get(self._current_part.field_name)
        if self._current_file_limit is None:
            raise MultiPartException(f"Unexpected multipart file field '{self._current_part.field_name}'.")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._current_file_size += end - start
            if self._current_file_limit is not None and self._current_file_size > self._current_file_limit:
                raise _MultipartSizeLimitExceeded(
                    f"Multipart file field '{self._current_part.field_name}' exceeded the maximum size of "
                    f"{_format_upload_limit(self._current_file_limit)}."
                )
        super().on_part_data(data, start, end)


async def _parse_multipart_form(
    request: Request,
    *,
    file_limits: dict[str, int],
    max_fields: int = 10,
) -> FormData:
    content_type = str(request.headers.get("content-type") or "").strip()
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail="Expected multipart/form-data upload.")
    try:
        parser = _LimitedMultipartParser(
            Headers(request.headers),
            request.stream(),
            file_limits=file_limits,
            max_files=max(1, len(file_limits)),
            max_fields=max_fields,
            max_text_part_size=_MULTIPART_TEXT_LIMIT_BYTES,
        )
    except AssertionError:
        return await _parse_multipart_form_fallback(
            request,
            file_limits=file_limits,
            max_fields=max_fields,
        )
    try:
        return await parser.parse()
    except _MultipartSizeLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except MultiPartException as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _fallback_body_limit(file_limits: dict[str, int], *, max_fields: int) -> int:
    file_budget = sum(max(0, int(limit)) for limit in file_limits.values())
    text_budget = max(0, int(max_fields)) * _MULTIPART_TEXT_LIMIT_BYTES
    overhead_budget = (max(1, len(file_limits)) + max(0, int(max_fields))) * 64 * 1024
    return file_budget + text_budget + overhead_budget + 64 * 1024


def _decode_form_text(payload: bytes, charset: str | None) -> str:
    encoding = str(charset or "utf-8").strip() or "utf-8"
    try:
        return payload.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        return payload.decode("latin-1")


async def _parse_multipart_form_fallback(
    request: Request,
    *,
    file_limits: dict[str, int],
    max_fields: int,
) -> FormData:
    content_type = str(request.headers.get("content-type") or "").strip()
    parser = BytesFeedParser(policy=policy.default)
    parser.feed(f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8"))
    body_limit = _fallback_body_limit(file_limits, max_fields=max_fields)
    body_stream = SpooledTemporaryFile(max_size=min(body_limit, 1024 * 1024))
    total_bytes = 0
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > body_limit:
                raise HTTPException(
                    status_code=413,
                    detail="Multipart request exceeded the fallback parser size limit.",
                )
            body_stream.write(chunk)
        body_stream.seek(0)
        while True:
            chunk = body_stream.read(64 * 1024)
            if not chunk:
                break
            parser.feed(chunk)
    finally:
        body_stream.close()

    message = parser.close()
    if not message.is_multipart():
        raise HTTPException(status_code=400, detail="Malformed multipart payload.")

    items: list[tuple[str, str | UploadFile]] = []
    file_count = 0
    field_count = 0
    max_files = max(1, len(file_limits))
    for part in message.iter_parts():
        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
        if not field_name:
            raise HTTPException(
                status_code=400,
                detail='The Content-Disposition header field "name" must be provided.',
            )
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is None:
            field_count += 1
            if field_count > max_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Too many fields. Maximum number of fields is {max_fields}.",
                )
            if len(payload) > _MULTIPART_TEXT_LIMIT_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Part exceeded maximum size of {int(_MULTIPART_TEXT_LIMIT_BYTES / 1024)}KB.",
                )
            items.append((field_name, _decode_form_text(payload, part.get_content_charset())))
            continue

        file_count += 1
        if file_count > max_files:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum number of files is {max_files}.",
            )
        current_file_limit = file_limits.get(field_name)
        if current_file_limit is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected multipart file field '{field_name}'.",
            )
        if len(payload) > current_file_limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Multipart file field '{field_name}' exceeded the maximum size of "
                    f"{_format_upload_limit(current_file_limit)}."
                ),
            )
        upload_buffer = SpooledTemporaryFile(max_size=min(max(current_file_limit, 1), 1024 * 1024))
        upload_buffer.write(payload)
        upload_buffer.seek(0)
        items.append(
            (
                field_name,
                UploadFile(
                    file=upload_buffer,
                    size=len(payload),
                    filename=filename,
                    headers=Headers({"content-type": part.get_content_type()}),
                ),
            )
        )
    return FormData(items)


def _multipart_file_value(form: FormData, field_name: str, *, required: bool = False) -> UploadFile | None:
    value = form.get(field_name)
    if value is None:
        if required:
            raise ValueError(f"Missing multipart field '{field_name}'.")
        return None
    if not isinstance(value, UploadFile):
        raise ValueError(f"Multipart field '{field_name}' must be a file upload.")
    return value


def _multipart_text_value(form: FormData, field_name: str, *, required: bool = False) -> str | None:
    value = form.get(field_name)
    if value is None:
        if required:
            raise ValueError(f"Missing multipart field '{field_name}'.")
        return None
    if isinstance(value, UploadFile):
        raise ValueError(f"Multipart field '{field_name}' must be text.")
    return str(value)


def _multipart_optional_text(form: FormData, field_name: str) -> str | None:
    value = _multipart_text_value(form, field_name)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _parse_form_bool(raw_value: str | None, field_name: str, *, default: bool = False) -> bool:
    if raw_value is None:
        return default
    normalized = str(raw_value).strip().lower()
    if not normalized:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Field '{field_name}' must be a boolean.")


def _parse_form_int(raw_value: str | None, field_name: str) -> int | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    if not normalized:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"Field '{field_name}' must be an integer.") from exc


def _parse_form_json(raw_value: str | None, field_name: str) -> Any:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    if not normalized:
        return None
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Field '{field_name}' must be valid JSON.") from exc


def _is_local_request(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    return str(client.host or "").strip().lower() in _LOCALHOST_CLIENT_HOSTS


@app.get("/health")
def health() -> dict:
    runtime = inference.runtime_status()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "runtime_profile": runtime.get("runtime_profile"),
        "resource_tier": runtime.get("resource_tier"),
        "active_pack": runtime.get("active_pack"),
        "selected_pack": runtime.get("selected_pack"),
        "effective_pack": runtime.get("effective_pack"),
        "active_backend": runtime.get("active_backend"),
        "fp8_fallback_used": runtime.get("fp8_fallback_used", False),
        "fp8_fallback_reason": runtime.get("fp8_fallback_reason"),
        "fp8_runtime_mode": runtime.get("fp8_runtime_mode"),
        "fp8_storage_preserved_tensor_count": runtime.get("fp8_storage_preserved_tensor_count", 0),
        "fp8_promoted_tensor_count": runtime.get("fp8_promoted_tensor_count", 0),
        "lora_capable": runtime.get("lora_capable", False),
        "gallery_color_cache_active": runtime.get("gallery_color_cache_active", False),
        "gallery_color_cache_version": runtime.get("gallery_color_cache_version"),
        "gallery_color_cache_target_version": runtime.get("gallery_color_cache_target_version"),
        "gallery_color_cache_error": runtime.get("gallery_color_cache_error"),
        "offline_mode": settings.offline_mode,
    }


@app.get("/config")
def config() -> dict:
    payload = settings.to_dict()
    payload["runtime"] = inference.runtime_status()
    return payload


@app.get("/model-packs")
def model_packs() -> dict:
    packs = inference.list_model_packs()
    return {"items": packs, "count": len(packs)}


@app.get("/loras")
def loras() -> JSONResponse:
    items = inference.list_loras()
    return JSONResponse(content={
        "items": items,
        "count": len(items),
        "capabilities": inference.lora_capabilities(),
    }, headers=_NO_STORE_HEADERS)


@app.get("/loras/events")
async def lora_events(request: Request) -> StreamingResponse:
    async def event_stream():
        revision = _LORA_LIBRARY_EVENTS.current()
        yield f"data: {revision}\n\n"
        while True:
            if await request.is_disconnected():
                break
            next_revision = await asyncio.to_thread(
                _LORA_LIBRARY_EVENTS.wait_for_change,
                revision,
                20.0,
            )
            if next_revision != revision:
                revision = next_revision
                yield f"data: {revision}\n\n"
                continue
            yield ": keep-alive\n\n"

    headers = {**_NO_STORE_HEADERS, "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.get("/wildcards")
def wildcards() -> JSONResponse:
    items = inference.list_wildcards()
    return JSONResponse(
        content={
            "items": items,
            "count": len(items),
            "capabilities": inference.wildcard_capabilities(),
        },
        headers=_NO_STORE_HEADERS,
    )


@app.get("/wildcards/events")
async def wildcard_events(request: Request) -> StreamingResponse:
    async def event_stream():
        revision = _WILDCARD_LIBRARY_EVENTS.current()
        yield f"data: {revision}\n\n"
        while True:
            if await request.is_disconnected():
                break
            next_revision = await asyncio.to_thread(
                _WILDCARD_LIBRARY_EVENTS.wait_for_change,
                revision,
                20.0,
            )
            if next_revision != revision:
                revision = next_revision
                yield f"data: {revision}\n\n"
                continue
            yield ": keep-alive\n\n"

    headers = {**_NO_STORE_HEADERS, "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.post("/wildcards")
def wildcards_create(payload: WildcardUpsertRequest) -> dict:
    try:
        item = inference.create_wildcard(
            display_name=payload.display_name,
            token=payload.token,
            content_text=payload.content_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _notify_wildcard_library_changed()
    return {
        "status": "ok",
        "item": item,
        "capabilities": inference.wildcard_capabilities(),
    }


@app.patch("/wildcards/{wildcard_id}")
def wildcards_update(wildcard_id: str, payload: WildcardUpsertRequest) -> dict:
    try:
        item = inference.update_wildcard(
            wildcard_id=wildcard_id,
            display_name=payload.display_name,
            token=payload.token,
            content_text=payload.content_text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _notify_wildcard_library_changed()
    return {"status": "ok", "item": item}


@app.delete("/wildcards/{wildcard_id}")
def wildcards_delete(wildcard_id: str) -> dict:
    try:
        result = inference.delete_wildcard(wildcard_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _notify_wildcard_library_changed()
    return {"status": "ok", **result}


@app.post("/wildcards/suggestions")
def wildcard_suggestions(payload: WildcardSuggestionRequest) -> dict:
    try:
        result = inference.suggest_wildcard_entries(
            theme=payload.theme,
            format_example=payload.format_example,
            existing_entries=payload.existing_entries,
            seed=payload.seed,
            pack_name=payload.pack,
        )
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled wildcard suggestion error.")
        raise HTTPException(status_code=500, detail="Wildcard suggestion generation failed.") from exc
    return {"status": "ok", **result}


@app.post("/lora-drafts")
async def lora_drafts_create(request: Request) -> dict:
    form: FormData | None = None
    try:
        form = await _parse_multipart_form(request, file_limits={"file": _LORA_UPLOAD_LIMIT_BYTES}, max_fields=4)
        upload = _multipart_file_value(form, "file", required=True)
        draft = inference.create_lora_draft(filename=str(upload.filename or ""), content_file=upload.file)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled LoRA draft upload error.")
        raise HTTPException(status_code=500, detail="LoRA draft upload failed.") from exc
    finally:
        if form is not None:
            await form.close()
    return {"status": "ok", "draft": draft}


@app.post("/lora-drafts/{draft_id}/detect-triggers")
def lora_draft_detect_triggers(draft_id: str) -> dict:
    try:
        draft = inference.detect_lora_draft_triggers(draft_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "draft": draft}


@app.post("/loras")
async def loras_create(request: Request) -> dict:
    form: FormData | None = None
    try:
        form = await _parse_multipart_form(request, file_limits={"thumbnail": _LORA_THUMBNAIL_LIMIT_BYTES}, max_fields=8)
        thumbnail = _multipart_file_value(form, "thumbnail")
        preview_content = None
        if thumbnail is not None:
            preview_content = bytes(thumbnail.file.read())
        item = inference.finalize_lora_draft(
            draft_id=_multipart_text_value(form, "draft_id", required=True),
            display_name=_multipart_text_value(form, "display_name", required=True),
            trigger_words=_multipart_text_value(form, "trigger_words"),
            preview_content=preview_content,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled LoRA create error.")
        raise HTTPException(status_code=500, detail="LoRA save failed.") from exc
    finally:
        if form is not None:
            await form.close()
    _notify_lora_library_changed()
    return {
        "status": "ok",
        "item": item,
        "capabilities": inference.lora_capabilities(),
    }


@app.patch("/loras/{lora_id}")
async def loras_update(lora_id: str, request: Request) -> dict:
    form: FormData | None = None
    try:
        form = await _parse_multipart_form(request, file_limits={"thumbnail": _LORA_THUMBNAIL_LIMIT_BYTES}, max_fields=6)
        thumbnail = _multipart_file_value(form, "thumbnail")
        preview_content = None
        if thumbnail is not None:
            preview_content = bytes(thumbnail.file.read())
        item = inference.update_lora(
            lora_id=lora_id,
            display_name=_multipart_text_value(form, "display_name", required=True),
            trigger_words=_multipart_text_value(form, "trigger_words"),
            preview_content=preview_content,
        )
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled LoRA update error.")
        raise HTTPException(status_code=500, detail="LoRA update failed.") from exc
    finally:
        if form is not None:
            await form.close()
    _notify_lora_library_changed()
    return {"status": "ok", "item": item}


@app.get("/loras/{lora_id}/preview")
def lora_preview(lora_id: str) -> FileResponse:
    try:
        preview_path = inference.preview_lora_path(lora_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(
        preview_path,
        media_type="image/png",
        filename=preview_path.name,
        headers=_NO_STORE_HEADERS,
    )


@app.delete("/loras/{lora_id}")
def lora_delete(lora_id: str) -> dict:
    try:
        result = inference.delete_lora(lora_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _notify_lora_library_changed()
    return {"status": "ok", **result}


@app.get("/client-jobs")
def client_jobs(
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    return inference.client_job_status(owner_id=owner_id)


@app.post("/client-jobs/cancel")
def client_jobs_cancel(
    payload: CancelClientJobRequest | None = Body(default=None),
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    return inference.request_cancel_client_job(owner_id=owner_id, job_id=payload.job_id if payload else None)


@app.get("/api-manifest", include_in_schema=False)
def api_manifest() -> dict:
    return api_manifest_payload()


@app.post("/generate")
def generate(
    payload: GenerateRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client)
        generate_kwargs = {
            "owner_id": owner_id,
            "prompt": payload.prompt,
            "width": payload.width,
            "height": payload.height,
            "steps": payload.steps,
            "pack_name": payload.pack,
            "job_id": payload.job_id,
            "seed": payload.seed,
            "scheduler_mode": payload.scheduler_mode,
            "inference_process": payload.inference_process,
            "enhance_prompt": payload.enhance_prompt,
            "procedural_creativity": payload.procedural_creativity,
            "rplus_vibrance": payload.rplus_vibrance,
            "rplus_initial_bias_level": payload.rplus_initial_bias_level,
        }
        if payload.loras:
            generate_kwargs["loras"] = [item.model_dump() for item in payload.loras]
        result = inference.generate(
            **generate_kwargs,
        )
    except HTTPException:
        raise
    except GenerationCancelledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled generation error.")
        raise HTTPException(status_code=500, detail="Generation failed.") from exc
    return result


@app.post("/img2img")
async def img2img(
    request: Request,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    form: FormData | None = None
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client)
        form = await _parse_multipart_form(
            request,
            file_limits={"image": _IMG2IMG_UPLOAD_LIMIT_BYTES},
            max_fields=12,
        )
        upload = _multipart_file_value(form, "image", required=True)
        payload = Img2ImgRequest.model_validate(
            {
                "prompt": _multipart_text_value(form, "prompt", required=True),
                "pack": _multipart_optional_text(form, "pack"),
                "job_id": _multipart_optional_text(form, "job_id"),
                "seed": _parse_form_int(_multipart_optional_text(form, "seed"), "seed"),
                "scheduler_mode": _multipart_optional_text(form, "scheduler_mode"),
                "enhance_prompt": _parse_form_bool(
                    _multipart_optional_text(form, "enhance_prompt"),
                    "enhance_prompt",
                    default=False,
                ),
                "similarity": (
                    _multipart_optional_text(form, "similarity")
                    if _multipart_optional_text(form, "similarity") is not None
                    else 0.80
                ),
                "loras": _parse_form_json(_multipart_optional_text(form, "loras"), "loras") or [],
            }
        )
        image_bytes = await upload.read()
        if not image_bytes:
            raise ValueError("Multipart field 'image' is empty.")
        try:
            with Image.open(io.BytesIO(image_bytes)) as source_file:
                source_image = source_file.convert("RGB")
        except OSError as exc:
            raise ValueError("Multipart field 'image' must be a supported image file.") from exc
        img2img_kwargs = {
            "owner_id": owner_id,
            "prompt": payload.prompt,
            "image": source_image,
            "image_filename": upload.filename,
            "pack_name": payload.pack,
            "job_id": payload.job_id,
            "seed": payload.seed,
            "scheduler_mode": payload.scheduler_mode,
            "enhance_prompt": payload.enhance_prompt,
            "similarity": payload.similarity,
        }
        if payload.loras:
            img2img_kwargs["loras"] = [item.model_dump() for item in payload.loras]
        result = inference.img2img(**img2img_kwargs)
    except HTTPException:
        raise
    except GenerationCancelledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled img2img error.")
        raise HTTPException(status_code=500, detail="Img2img failed.") from exc
    return result


@app.post("/upscale")
def upscale(
    payload: UpscaleRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client)
        upscale_kwargs: dict[str, object] = {
            "owner_id": owner_id,
            "filename": payload.filename,
            "pack_name": payload.pack,
            "job_id": payload.job_id,
            "seed": payload.seed,
            "scheduler_mode": payload.scheduler_mode,
            "enhance_prompt": payload.enhance_prompt,
        }
        result = inference.upscale(**upscale_kwargs)
    except HTTPException:
        raise
    except GenerationCancelledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled upscale error.")
        raise HTTPException(status_code=500, detail="Upscale failed.") from exc
    return result


@app.post("/clarity")
def clarity(
    payload: ClarityRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client)
        clarity_kwargs: dict[str, object] = {
            "owner_id": owner_id,
            "filename": payload.filename,
            "pack_name": payload.pack,
            "job_id": payload.job_id,
            "seed": payload.seed,
            "scheduler_mode": payload.scheduler_mode,
            "enhance_prompt": payload.enhance_prompt,
        }
        result = inference.clarity(**clarity_kwargs)
    except HTTPException:
        raise
    except GenerationCancelledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled clarity error.")
        raise HTTPException(status_code=500, detail="Clarity failed.") from exc
    return result


@app.get("/images")
def images(
    prompt: str | None = Query(default=None),
    color: Literal["black", "white", "red", "yellow", "blue", "green"] | None = Query(default=None),
    favorite: bool = Query(default=False),
    limit: int = Query(default=120, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    newest_first: bool = Query(default=True),
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    list_kwargs = {
        "owner_id": owner_id,
        "prompt_query": prompt,
        "color_filter": color,
        "limit": limit,
        "offset": offset,
        "newest_first": newest_first,
    }
    if favorite:
        list_kwargs["favorites_only"] = True
    rows = inference.list_images(**list_kwargs)
    return {
        "items": rows,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "color_cache": inference.gallery_color_cache_status(),
    }


@app.get("/images/{filename}")
def image_file(
    filename: str,
    client_id: str | None = Query(default=None),
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> FileResponse:
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client, client_query=client_id)
        safe_filename = InferenceService.sanitize_filename(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    row = inference.get_image(safe_filename, owner_id=owner_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Image not found.")

    image_path = Path(str(row["output_path"]))
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
    return FileResponse(image_path, media_type="image/png", filename=safe_filename)


@app.post("/images/{filename}/favorite")
def image_favorite(
    filename: str,
    payload: FavoriteImageRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        safe_filename = InferenceService.sanitize_filename(filename)
        row = inference.set_image_favorite(owner_id=owner_id, filename=safe_filename, favorite=payload.favorite)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "filename": safe_filename,
        "favorite": bool(row.get("favorite")),
        "item": row,
    }


@app.post("/images/download-zip")
def image_download_zip(
    payload: BulkDownloadRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> StreamingResponse:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        files = inference.resolve_download_images(owner_id=owner_id, filenames=payload.filenames)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for safe_filename, image_path in files:
            zip_file.write(image_path, arcname=safe_filename)
    archive.seek(0)

    owner_label = owner_id or "gallery"
    headers = {"Content-Disposition": f'attachment; filename="{owner_label}_selection.zip"'}
    return StreamingResponse(archive, media_type="application/zip", headers=headers)


@app.delete("/gallery")
def gallery_delete(
    payload: DeleteConfirmRequest | None = Body(default=None),
    confirm: str | None = Query(default=None),
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    confirmation = payload.confirm if payload is not None else (confirm or "")
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        result = inference.delete_gallery(owner_id=owner_id, confirm_text=confirmation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "deleted_files": result["deleted_files"],
        "deleted_rows": result["deleted_rows"],
        "remaining_rows": result.get("remaining_rows", 0),
    }


@app.post("/gallery/rebuild")
def gallery_rebuild(
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        result = inference.rebuild_gallery(owner_id=owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", **result}


@app.delete("/images/{filename}")
def image_delete(
    filename: str,
    payload: DeleteConfirmRequest | None = Body(default=None),
    confirm: str | None = Query(default=None),
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    confirmation = payload.confirm if payload is not None else (confirm or "")
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        safe_filename = InferenceService.sanitize_filename(filename)
        result = inference.delete_image(owner_id=owner_id, filename=safe_filename, confirm_text=confirmation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "deleted_files": result["deleted_files"],
        "deleted_rows": result["deleted_rows"],
        "remaining_rows": result.get("remaining_rows", 0),
        "filename": safe_filename,
    }


@app.get("/gallery/import-sources")
def gallery_import_sources(
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        items = inference.list_import_sources(owner_id=owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": items, "count": len(items)}


@app.post("/gallery/import")
def gallery_import(
    payload: ImportGalleryRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    try:
        return inference.import_gallery(
            owner_id=owner_id,
            source_id=payload.source_id,
            dry_run=payload.dry_run,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unhandled gallery import error.")
        raise HTTPException(status_code=500, detail="Import failed.") from exc


@app.post("/server/kill")
def server_kill(request: Request, background_tasks: BackgroundTasks) -> dict:
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="Server shutdown is allowed from the local machine only.")
    background_tasks.add_task(_shutdown_server_process)
    return {"status": "ok", "message": "Server shutdown initiated."}


@app.post("/server/restart")
def server_restart(request: Request, background_tasks: BackgroundTasks) -> dict:
    if not _is_local_request(request):
        raise HTTPException(status_code=403, detail="Server restart is allowed from the local machine only.")
    background_tasks.add_task(_restart_server_process)
    return {"status": "ok", "message": "Server restart initiated."}


ui_dir = Path(settings.paths.ui_dir)
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")

img_dir = settings.paths.root_dir / "img"
if img_dir.exists():
    app.mount("/img", StaticFiles(directory=img_dir), name="img")


@app.get("/")
def index() -> FileResponse:
    index_path = Path(settings.paths.ui_dir) / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail=f"UI entry file not found: {index_path}")
    return FileResponse(index_path, headers=_NO_STORE_HEADERS)


@app.get("/api", include_in_schema=False)
def api_docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/API")


@app.get("/API", include_in_schema=False)
def api_docs_page() -> FileResponse:
    api_path = Path(settings.paths.ui_dir) / "api.html"
    if not api_path.exists():
        raise HTTPException(status_code=500, detail=f"API docs file not found: {api_path}")
    return FileResponse(api_path, headers=_NO_STORE_HEADERS)


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    favicon_path = settings.paths.root_dir / "img" / "favicon.ico"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(favicon_path, media_type="image/x-icon", filename="favicon.ico")
