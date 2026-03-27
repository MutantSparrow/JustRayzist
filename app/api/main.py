from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

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


class LoraSelectionRequest(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    weight: float = Field(default=DEFAULT_LORA_WEIGHT, ge=MIN_LORA_WEIGHT, le=MAX_LORA_WEIGHT)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    pack: str | None = Field(default=None)
    job_id: str | None = Field(default=None, max_length=255)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] | None = Field(default=None)
    enhance_prompt: bool = Field(default=False)
    procedural_creativity: int = Field(default=0, ge=0, le=3)
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


class UpscaleRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    pack: str | None = Field(default=None)
    job_id: str | None = Field(default=None, max_length=255)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] | None = Field(default=None)
    enhance_prompt: bool = Field(default=False)


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


def _extract_content_disposition_param(header_value: str, key: str) -> str | None:
    pattern = rf'{re.escape(key)}="([^"]*)"'
    match = re.search(pattern, header_value, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    for segment in header_value.split(";"):
        name, _sep, value = segment.strip().partition("=")
        if name.strip().lower() != key.lower():
            continue
        return value.strip().strip('"')
    return None


def _parse_multipart_file_upload(content_type: str, body: bytes) -> tuple[str, bytes]:
    parts = _parse_multipart_parts(content_type, body)
    for items in parts.values():
        for part in items:
            filename = str(part.get("filename") or "").strip()
            if filename:
                return filename, bytes(part.get("content") or b"")
    raise ValueError("Upload did not include a file part.")


def _parse_multipart_parts(content_type: str, body: bytes) -> dict[str, list[dict[str, Any]]]:
    match = re.search(r'boundary="?([^";]+)"?', str(content_type or ""), flags=re.IGNORECASE)
    if not match:
        raise ValueError("Upload is missing a multipart boundary.")
    boundary = match.group(1).encode("utf-8")
    delimiter = b"--" + boundary
    parts: dict[str, list[dict[str, Any]]] = {}

    for raw_part in body.split(delimiter):
        part = raw_part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2]
        header_blob, separator, payload = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        header_lines = header_blob.decode("utf-8", errors="replace").split("\r\n")
        headers: dict[str, str] = {}
        for line in header_lines:
            name, _sep, value = line.partition(":")
            if not _sep:
                continue
            headers[name.strip().lower()] = value.strip()
        content_disposition = headers.get("content-disposition", "")
        field_name = _extract_content_disposition_param(content_disposition, "name")
        if not field_name:
            continue
        payload_bytes = payload[:-2] if payload.endswith(b"\r\n") else payload
        parts.setdefault(field_name, []).append(
            {
                "name": field_name,
                "filename": _extract_content_disposition_param(content_disposition, "filename"),
                "content_type": headers.get("content-type", ""),
                "content": payload_bytes,
            }
        )
    return parts


def _get_first_multipart_part(
    parts: dict[str, list[dict[str, Any]]],
    field_name: str,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    items = parts.get(field_name) or []
    if items:
        return items[0]
    if required:
        raise ValueError(f"Missing multipart field '{field_name}'.")
    return None


def _multipart_text_value(
    parts: dict[str, list[dict[str, Any]]],
    field_name: str,
    *,
    required: bool = False,
) -> str | None:
    part = _get_first_multipart_part(parts, field_name, required=required)
    if part is None:
        return None
    return bytes(part.get("content") or b"").decode("utf-8", errors="replace")


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
def loras() -> dict:
    items = inference.list_loras()
    return {
        "items": items,
        "count": len(items),
        "capabilities": inference.lora_capabilities(),
    }


@app.post("/lora-drafts")
async def lora_drafts_create(request: Request) -> dict:
    content_type = str(request.headers.get("content-type") or "").strip()
    try:
        filename, content = _parse_multipart_file_upload(content_type, await request.body())
        draft = inference.create_lora_draft(filename=filename, content=content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled LoRA draft upload error.")
        raise HTTPException(status_code=500, detail="LoRA draft upload failed.") from exc
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
    content_type = str(request.headers.get("content-type") or "").strip()
    try:
        parts = _parse_multipart_parts(content_type, await request.body())
        item = inference.finalize_lora_draft(
            draft_id=_multipart_text_value(parts, "draft_id", required=True),
            display_name=_multipart_text_value(parts, "display_name", required=True),
            trigger_words=_multipart_text_value(parts, "trigger_words"),
            preview_content=(
                bytes(_get_first_multipart_part(parts, "thumbnail").get("content") or b"")
                if _get_first_multipart_part(parts, "thumbnail") is not None
                else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled LoRA create error.")
        raise HTTPException(status_code=500, detail="LoRA save failed.") from exc
    return {
        "status": "ok",
        "item": item,
        "capabilities": inference.lora_capabilities(),
    }


@app.patch("/loras/{lora_id}")
async def loras_update(lora_id: str, request: Request) -> dict:
    content_type = str(request.headers.get("content-type") or "").strip()
    try:
        parts = _parse_multipart_parts(content_type, await request.body())
        item = inference.update_lora(
            lora_id=lora_id,
            display_name=_multipart_text_value(parts, "display_name", required=True),
            trigger_words=_multipart_text_value(parts, "trigger_words"),
            preview_content=(
                bytes(_get_first_multipart_part(parts, "thumbnail").get("content") or b"")
                if _get_first_multipart_part(parts, "thumbnail") is not None
                else None
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        LOGGER.exception("Unhandled LoRA update error.")
        raise HTTPException(status_code=500, detail="LoRA update failed.") from exc
    return {"status": "ok", "item": item}


@app.get("/loras/{lora_id}/preview")
def lora_preview(lora_id: str) -> FileResponse:
    try:
        preview_path = inference.preview_lora_path(lora_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(preview_path, media_type="image/png", filename=preview_path.name)


@app.delete("/loras/{lora_id}")
def lora_delete(lora_id: str) -> dict:
    try:
        result = inference.delete_lora(lora_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
            "pack_name": payload.pack,
            "job_id": payload.job_id,
            "seed": payload.seed,
            "scheduler_mode": payload.scheduler_mode,
            "enhance_prompt": payload.enhance_prompt,
            "procedural_creativity": payload.procedural_creativity,
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


@app.post("/upscale")
def upscale(
    payload: UpscaleRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client)
        result = inference.upscale(
            owner_id=owner_id,
            filename=payload.filename,
            pack_name=payload.pack,
            job_id=payload.job_id,
            seed=payload.seed,
            scheduler_mode=payload.scheduler_mode,
            enhance_prompt=payload.enhance_prompt,
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
        LOGGER.exception("Unhandled upscale error.")
        raise HTTPException(status_code=500, detail="Upscale failed.") from exc
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
def server_kill(background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(_shutdown_server_process)
    return {"status": "ok", "message": "Server shutdown initiated."}


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
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


@app.get("/api", include_in_schema=False)
def api_docs_redirect() -> RedirectResponse:
    return RedirectResponse(url="/API")


@app.get("/API", include_in_schema=False)
def api_docs_page() -> FileResponse:
    api_path = Path(settings.paths.ui_dir) / "api.html"
    if not api_path.exists():
        raise HTTPException(status_code=500, detail=f"API docs file not found: {api_path}")
    return FileResponse(api_path, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    favicon_path = settings.paths.root_dir / "img" / "favicon.ico"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found.")
    return FileResponse(favicon_path, media_type="image/x-icon", filename="favicon.ico")
