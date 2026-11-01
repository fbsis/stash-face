"""FastAPI application for Immich-style Face Recognition and Stash integration."""
import base64
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Security, status
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .detector import Detector
from .stash import StashClient
from .store import Store
from .worker import Worker

log = logging.getLogger(__name__)

api_token = APIKeyHeader(name="X-API-Token", auto_error=False)


def check_auth(settings: Settings, provided: Optional[str] = Security(api_token)):
    if not provided or provided != settings.token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


class JobRequest(BaseModel):
    action: str = Field(..., description="'scan', 'cluster', or 'sync_performers'")
    image_id: Optional[str] = None
    force: bool = False


class UpdatePersonRequest(BaseModel):
    name: Optional[str] = None
    hidden: Optional[bool] = None


class LinkPerformerRequest(BaseModel):
    performer_id: str
    performer_name: str


class CreatePerformerRequest(BaseModel):
    name: str


class MergePersonRequest(BaseModel):
    target_id: int


class SuggestionDecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(accepted|rejected|ignored)$")


def worker_process_entry():
    """Load OpenVINO and run the queue outside the HTTP process.

    Keeping model initialization in a spawned process prevents an accelerator
    crash from taking the dashboard/API down during slow GPU startup.
    """
    settings = Settings.from_env()
    ready_path = settings.data_dir / "worker.ready.json"
    ready_path.unlink(missing_ok=True)
    worker = None
    try:
        store = Store(settings.data_dir)
        stash = StashClient(settings)
        detector = Detector(settings)
        worker = Worker(settings, store, stash, detector)
        ready_path.write_text(json.dumps(detector.info()), encoding="utf-8")
        ready_path.chmod(0o600)
        worker.start()
        while any(thread.is_alive() for thread in worker.threads):
            time.sleep(1)
    finally:
        ready_path.unlink(missing_ok=True)
        if worker is not None:
            worker.stop()


def create_app() -> FastAPI:
    settings = Settings.from_env()
    store = Store(settings.data_dir)
    runtime = {"process": None}
    ready_path = settings.data_dir / "worker.ready.json"

    app = FastAPI(title="Stash Face Recognition", version="1.0.0", docs_url=None, redoc_url=None)

    @app.on_event("startup")
    def startup():
        ctx = mp.get_context("spawn")
        process = ctx.Process(target=worker_process_entry, name="stash-face-worker", daemon=True)
        process.start()
        runtime["process"] = process

    @app.on_event("shutdown")
    def shutdown():
        process = runtime.get("process")
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()

    def auth_dep(tok: Optional[str] = Security(api_token)):
        return check_auth(settings, tok)

    @app.get("/health")
    def health(_: None = Depends(auth_dep)):
        process = runtime.get("process")
        ready = bool(process and process.is_alive() and ready_path.exists())
        detector_info = {
            "provider": "insightface", "model_name": settings.model_name,
            "actual_device": "initializing", "execution_provider": "initializing",
            "embeddings": False, "initializing": True,
        }
        if ready:
            try:
                detector_info = json.loads(ready_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ready = False
        process_error = None
        if process is not None and not process.is_alive() and process.exitcode not in (None, 0):
            process_error = f"worker process exited with code {process.exitcode}"
        return {
            "status": "ok" if ready else ("error" if process_error else "initializing"),
            "version": "1.0.0",
            "stash_url": settings.stash_url,
            "worker_alive": ready,
            "worker_threads": settings.workers if ready else 0,
            "detector": detector_info,
            "initialization_error": process_error,
            "stats": store.stats(),
        }

    @app.get("/stats")
    def stats(_: None = Depends(auth_dep)):
        return store.stats()

    @app.get("/progress")
    def progress(_: None = Depends(auth_dep)):
        return store.progress()

    @app.get("/faces")
    def faces(
        face_filter: str = Query("all", alias="filter", pattern="^(all|clustered|unclustered|named|unknown)$"),
        q: str = "",
        limit: int = 100,
        offset: int = 0,
        _: None = Depends(auth_dep),
    ):
        return store.faces_page(face_filter, q, limit, offset)

    @app.get("/media")
    def media(
        status_filter: str = Query("all", alias="status"),
        media_type: str = Query("all", pattern="^(all|image|video)$"),
        limit: int = 100,
        offset: int = 0,
        _: None = Depends(auth_dep),
    ):
        allowed = {"all", "unindexed", "indexed", "clustered", "approved", "no_faces", "ignored", "failed"}
        if status_filter not in allowed:
            raise HTTPException(status_code=400, detail="Unsupported media status")
        return store.media_page(status_filter, media_type, limit, offset)

    @app.get("/suggestions")
    def suggestions(
        status_filter: str = Query("pending", alias="status", pattern="^(pending|accepted|rejected|ignored|all)$"),
        limit: int = 100,
        offset: int = 0,
        _: None = Depends(auth_dep),
    ):
        return store.suggestions_page(status_filter, limit, offset)

    @app.post("/suggestions/{suggestion_id}/decision")
    def decide_suggestion(suggestion_id: int, req: SuggestionDecisionRequest, _: None = Depends(auth_dep)):
        suggestion = store.suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail="Suggestion not found")
        if req.decision == "accepted" and settings.auto_tag_stash and suggestion.get("performer_id"):
            try:
                if suggestion["media_type"] == "image":
                    stash.add_performers([suggestion["media_key"]], [suggestion["performer_id"]])
                else:
                    stash.add_scene_performers([suggestion["source_id"]], [suggestion["performer_id"]])
            except Exception as exc:
                log.warning("Could not apply suggestion %s to Stash: %s", suggestion_id, exc)
                raise HTTPException(status_code=502, detail="Stash rejected the suggested tag") from exc
        result = store.decide_suggestion(suggestion_id, req.decision)
        return {"success": True, "decision": req.decision, "suggestion": result}

    @app.get("/stash/status")
    def stash_status(_: None = Depends(auth_dep)):
        return {"version": stash.gql("{ version { version } }").get("version", {})}

    @app.post("/jobs")
    def enqueue_job(req: JobRequest, _: None = Depends(auth_dep)):
        if req.action not in ("scan", "cluster", "sync_performers", "auto_tag", "image", "video", "forget"):
            raise HTTPException(status_code=400, detail="Unsupported action")

        payload = {"force": req.force}
        if req.action in ("image", "forget"):
            if not req.image_id:
                raise HTTPException(status_code=400, detail=f"image_id required for {req.action} job")
            payload["image_id"] = req.image_id
            dedupe_key = f"{req.action}:{req.image_id}:{req.force}"
        else:
            dedupe_key = f"{req.action}:{req.force}:{int(req.force)}"

        job_id, created = store.enqueue(req.action, payload, dedupe_key)
        return {"job_id": job_id, "created": created, "action": req.action}

    @app.get("/jobs")
    def list_jobs(limit: int = 100, _: None = Depends(auth_dep)):
        return store.jobs(limit=min(max(limit, 1), 500))

    @app.get("/people")
    def list_people(
        q: str = "",
        filter: str = Query("all", regex="^(all|named|unnamed|hidden)$"),
        _: None = Depends(auth_dep),
    ):
        return store.get_clusters(q=q, filter_type=filter)

    @app.get("/people/{cluster_id}")
    def get_person(cluster_id: int, _: None = Depends(auth_dep)):
        person = store.get_cluster(cluster_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        return person

    @app.patch("/people/{cluster_id}")
    def update_person(cluster_id: int, req: UpdatePersonRequest, _: None = Depends(auth_dep)):
        if req.name is not None:
            store.update_cluster_name(cluster_id, req.name)
        if req.hidden is not None:
            store.hide_cluster(cluster_id, req.hidden)
        return {"success": True}

    @app.post("/people/{cluster_id}/link-performer")
    def link_performer(cluster_id: int, req: LinkPerformerRequest, _: None = Depends(auth_dep)):
        image_ids = store.link_cluster_to_performer(cluster_id, req.performer_id, req.performer_name)
        embedding = store.cluster_embedding(cluster_id)
        if embedding is not None:
            performer = stash.performer(req.performer_id) or {}
            store.save_known_performer(req.performer_id, req.performer_name,
                                       performer.get("image_path", ""), embedding)
        if image_ids and settings.auto_tag_stash:
            try:
                stash.add_performers(image_ids, [req.performer_id])
            except Exception as e:
                log.warning("Could not auto-tag Stash images for cluster %s: %s", cluster_id, e)
        return {"success": True, "tagged_images": len(image_ids)}

    @app.post("/people/{cluster_id}/create-performer")
    def create_performer_from_person(cluster_id: int, req: CreatePerformerRequest, _: None = Depends(auth_dep)):
        person = store.get_cluster(cluster_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        # Encode exemplar crop as data URI if present
        data_uri = ""
        if person.get("exemplar_crop"):
            crop_path = store.crops / person["exemplar_crop"]
            if crop_path.exists():
                b64 = base64.b64encode(crop_path.read_bytes()).decode("utf-8")
                data_uri = f"data:image/jpeg;base64,{b64}"

        # Create performer in Stash
        created = stash.create_performer(req.name, data_uri)
        new_id = str(created["id"])
        image_ids = store.link_cluster_to_performer(cluster_id, new_id, req.name)
        embedding = store.cluster_embedding(cluster_id)
        if embedding is not None:
            store.save_known_performer(new_id, req.name, created.get("image_path", ""), embedding)

        if image_ids and settings.auto_tag_stash:
            try:
                stash.add_performers(image_ids, [new_id])
            except Exception as e:
                log.warning("Failed to bulk tag in Stash: %s", e)

        return {"success": True, "performer_id": new_id, "name": req.name, "tagged_images": len(image_ids)}

    @app.post("/people/{cluster_id}/merge")
    def merge_person(cluster_id: int, req: MergePersonRequest, _: None = Depends(auth_dep)):
        if cluster_id == req.target_id:
            raise HTTPException(status_code=400, detail="Cannot merge person into itself")
        success = store.merge_clusters(source_id=cluster_id, target_id=req.target_id)
        return {"success": success}

    @app.get("/faces/{face_id}/crop")
    def get_crop(face_id: int, _: None = Depends(auth_dep)):
        with store.connection() as con:
            row = con.execute("SELECT crop FROM faces WHERE id=?", (face_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Face not found")
        path = store.crops / row["crop"]
        if not path.exists():
            raise HTTPException(status_code=404, detail="Crop file not found")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/crops/{filename}")
    def get_crop_by_filename(filename: str, _: None = Depends(auth_dep)):
        # Sanitization: no directory traversal
        safe_name = Path(filename).name
        path = store.crops / safe_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Crop file not found")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/performers")
    def search_performers(q: str = "", _: None = Depends(auth_dep)):
        return stash.performers(q=q)

    # Static assets
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(static_dir)), name="assets")

        @app.get("/")
        def serve_index():
            return FileResponse(static_dir / "index.html")

    return app
