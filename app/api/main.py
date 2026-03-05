from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Body, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.config import load_settings
from app.core.logging import configure_logging
from app.core.model_registry import ModelPackValidationError
from app.api.inference_service import InferenceService
from app.storage.gallery_index import ensure_gallery_schema

configure_logging()
settings = load_settings()
inference = InferenceService(settings=settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_gallery_schema(settings)
    inference.sync_gallery()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


def _shutdown_server_process(delay_seconds: float = 0.35) -> None:
    def _kill() -> None:
        time.sleep(max(delay_seconds, 0.0))
        os._exit(0)

    threading.Thread(target=_kill, daemon=True, name="justrayzist-shutdown").start()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    pack: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] = Field(default="euler")
    enhance_prompt: bool = Field(default=False)

    @field_validator("width", "height")
    @classmethod
    def _validate_multiple_of_16(cls, value: int) -> int:
        if value % 16 != 0:
            raise ValueError("Dimension must be a multiple of 16.")
        return value


class UpscaleRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    pack: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    scheduler_mode: Literal["euler", "dpm"] = Field(default="euler")
    enhance_prompt: bool = Field(default=False)


class DeleteConfirmRequest(BaseModel):
    confirm: str = Field(min_length=1, max_length=32)


class ImportGalleryRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=255)
    dry_run: bool = Field(default=False)


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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "profile": settings.runtime_profile.name,
        "offline_mode": settings.offline_mode,
    }


@app.get("/config")
def config() -> dict:
    return settings.to_dict()


@app.get("/model-packs")
def model_packs() -> dict:
    packs = inference.list_model_packs()
    return {"items": packs, "count": len(packs)}


@app.post("/generate")
def generate(
    payload: GenerateRequest,
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    try:
        owner_id = _resolve_owner_id(x_justrayzist_client)
        result = inference.generate(
            owner_id=owner_id,
            prompt=payload.prompt,
            width=payload.width,
            height=payload.height,
            pack_name=payload.pack,
            seed=payload.seed,
            scheduler_mode=payload.scheduler_mode,
            enhance_prompt=payload.enhance_prompt,
        )
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
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
            seed=payload.seed,
            scheduler_mode=payload.scheduler_mode,
            enhance_prompt=payload.enhance_prompt,
        )
    except ModelPackValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upscale failed: {exc}") from exc
    return result


@app.get("/images")
def images(
    prompt: str | None = Query(default=None),
    limit: int = Query(default=120, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    newest_first: bool = Query(default=True),
    x_justrayzist_client: str | None = Header(default=None, alias="X-JustRayzist-Client"),
) -> dict:
    owner_id = _resolve_owner_id(x_justrayzist_client)
    rows = inference.list_images(
        owner_id=owner_id,
        prompt_query=prompt,
        limit=limit,
        offset=offset,
        newest_first=newest_first,
    )
    return {"items": rows, "count": len(rows), "limit": limit, "offset": offset}


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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}") from exc


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
