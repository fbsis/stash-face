"""Background worker for face processing, embedding extraction, and clustering."""
import fcntl
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .clustering import Clusterer
from .stash import UpstreamError, content_key

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Any, store: Any, stash: Any, detector: Any):
        self.settings = settings
        self.store = store
        self.stash = stash
        self.detector = detector
        self.clusterer = Clusterer(eps=settings.cluster_eps, min_samples=settings.min_faces)
        self.stopping = threading.Event()
        self.threads: list[threading.Thread] = []
        self.lock = None
        self.local = threading.local()

    def start(self):
        self.lock = (self.store.directory / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("Another worker owns this database; use uvicorn --workers 1") from None

        recovered = self.store.recover_running_jobs()
        if recovered:
            log.warning("Requeued %d jobs left running by a previous service instance.", recovered)
        worker_count = int(getattr(self.settings, "workers", 1))
        self.threads = [threading.Thread(target=self.run, name=f"face-worker-{index + 1}", daemon=True)
                        for index in range(worker_count)]
        for thread in self.threads:
            thread.start()
        log.info("Face background worker started with %d processing threads.", worker_count)

    def stop(self):
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout=self.settings.timeout * 2 + 5)
        self.stash.close()
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None

    def check_stop(self):
        if self.stopping.is_set():
            raise InterruptedError("Worker stopping")

    def run(self):
        try:
            while not self.stopping.is_set():
                job = self.store.claim()
                if not job:
                    self.stopping.wait(0.2)
                    continue
                try:
                    self.local.job_id = job["id"]
                    self.store.set_job_progress(job["id"], self.stage_for(job), self.detail_for(job))
                    result = self.execute(job)
                    self.store.finish(job["id"], result)
                except InterruptedError:
                    break
                except UpstreamError as exc:
                    log.warning("Job %s upstream error: %s", job["id"], exc)
                    retry_at = (
                        time.time() + self.settings.retry_seconds * (2 ** (job["attempts"] - 1))
                        if exc.retryable and job["attempts"] < self.settings.max_attempts
                        else None
                    )
                    self.store.fail(job["id"], str(exc), retry_at)
                except Exception as exc:
                    log.exception("Job %s failed with unexpected exception", job["id"])
                    self.store.fail(job["id"], str(exc))
                finally:
                    self.local.job_id = None
        finally:
            # The shared Stash client and process lock are closed by stop(), after all threads exit.
            pass

    def progress(self, stage: str, detail: str = ""):
        job_id = getattr(self.local, "job_id", None)
        if job_id:
            self.store.set_job_progress(job_id, stage, detail)

    def stage_for(self, job: dict) -> str:
        return {
            "scan": "enumerating media",
            "image": "detecting and embedding faces",
            "video": "extracting and embedding six frames",
            "cluster": "clustering embeddings",
            "sync_performers": "syncing Performer references",
            "auto_tag": "matching names from file paths",
            "forget": "removing deleted media",
        }.get(job["kind"], "processing")

    def detail_for(self, job: dict) -> str:
        payload = job.get("payload", {})
        return str(payload.get("title") or payload.get("image_id") or payload.get("path") or job["kind"])

    def execute(self, job: dict) -> dict:
        kind = job["kind"]
        payload = job.get("payload", {})

        if kind == "scan":
            return self.scan(payload)
        elif kind == "image":
            return self.process_image(payload["image_id"], payload.get("force", False))
        elif kind == "video":
            return self.process_video(payload)
        elif kind == "cluster":
            return self.run_clustering()
        elif kind == "forget":
            return {"forgotten": self.store.forget_image(payload["image_id"])}
        elif kind == "sync_performers":
            return self.sync_performers()
        elif kind == "auto_tag":
            return self.auto_tag(payload)
        raise ValueError(f"Unknown job kind: {kind}")

    def scan(self, payload: dict) -> dict:
        page, received, seen = 1, 0, set()
        enqueued = skipped = 0
        force = payload.get("force", False)

        while True:
            self.check_stop()
            result = self.stash.image_page(page)
            items = result.get("images", [])
            total = result.get("count", 0)

            if not items and received < total:
                break

            for image in items:
                self.check_stop()
                img_id = str(image["id"])
                seen.add(img_id)

                image_key = content_key(image)
                self.store.register_media(img_id, "image", img_id, image.get("title", ""), image_key)
                if not force and self.store.unchanged(img_id, image_key, self.detector.signature):
                    skipped += 1
                    continue

                key = f"image:{img_id}:{image_key}:{self.detector.signature}:{force}"
                _, created = self.store.enqueue("image", {"image_id": img_id, "force": force}, key)
                enqueued += int(created)

            received += len(items)
            if received >= total or not items:
                break
            page += 1

        # Scenes are indexed separately. Each video gets exactly six evenly-spaced frame records.
        scene_page, scene_received, video_count = 1, 0, 0
        while True:
            self.check_stop()
            result = self.stash.scene_page(scene_page)
            scenes = result.get("scenes", [])
            scene_total = result.get("count", 0)
            if not scenes and scene_received < scene_total:
                break
            for scene in scenes:
                for video in [f for f in scene.get("files", []) if f.get("__typename") == "VideoFile"]:
                    scene_id, file_id = str(scene["id"]), str(video["id"])
                    video_key = f"{video.get('mod_time')}:{video.get('size')}:{video.get('duration')}"
                    if not force and self.store.media_source_unchanged("video", scene_id, file_id, video_key):
                        skipped += self.settings.video_frame_count
                        continue
                    for frame_index in range(self.settings.video_frame_count):
                        frame_key = f"video:{scene_id}:{file_id}:frame:{frame_index}"
                        self.store.register_media(frame_key, "video", scene_id, scene.get("title", ""),
                                                  video_key, file_id, frame_index,
                                                  float(video.get("duration") or 0) * (frame_index + 1) / 7)
                    key = f"video:{scene_id}:{file_id}:{video_key}:{force}"
                    _, created = self.store.enqueue("video", {
                        "scene_id": scene_id, "file_id": file_id, "path": video["path"],
                        "duration": float(video.get("duration") or 0), "title": scene.get("title", ""),
                        "content_key": video_key, "force": force,
                    }, key)
                    enqueued += int(created)
                    video_count += 1
            scene_received += len(scenes)
            if scene_received >= scene_total or not scenes:
                break
            scene_page += 1

        if enqueued > 0:
            self.store.enqueue("cluster", {}, "cluster:after-scan")
            self.store.enqueue("auto_tag", {"paths": list(self.settings.auto_tag_paths)}, "auto_tag:after-scan")

        return {"enqueued": enqueued, "unchanged": skipped, "images_seen": len(seen), "videos_seen": video_count}

    def process_image(self, image_id: str, force: bool = False) -> dict:
        image = self.stash.image(image_id)
        if image is None:
            return {"deleted": True}

        # Check visual files
        visual_files = image.get("visual_files", [])
        if not any(f.get("__typename") == "ImageFile" for f in visual_files):
            return {"skipped": "Only still ImageFile media is supported"}

        ckey = content_key(image)
        if not force and self.store.unchanged(image_id, ckey, self.detector.signature):
            return {"unchanged": True}

        self.progress("downloading photo", image.get("title") or image_id)
        picture = self.stash.download(image)
        self.check_stop()

        self.progress("detecting and embedding faces", image.get("title") or image_id)
        detections = self.detector.detect(picture)
        saved_detections = []
        paths = []
        width, height = picture.size

        try:
            for det in detections:
                box = det.box
                # Crop with a slight margin around the face (15%) for a pleasing portrait
                bw, bh = box[2] - box[0], box[3] - box[1]
                margin = int(max(bw, bh) * 0.15)
                cx1 = max(0, box[0] - margin)
                cy1 = max(0, box[1] - margin)
                cx2 = min(width, box[2] + margin)
                cy2 = min(height, box[3] + margin)

                crop = picture.crop((cx1, cy1, cx2, cy2))
                crop.thumbnail((256, 256), Image.Resampling.LANCZOS)

                filename = uuid.uuid4().hex + ".jpg"
                crop_path = self.store.crops / filename
                crop.save(crop_path, format="JPEG", quality=90)
                crop_path.chmod(0o600)
                paths.append(crop_path)

                saved_detections.append({
                    "box": box,
                    "score": det.score,
                    "crop": filename,
                    "embedding": det.embedding,
                    "gender": det.gender,
                    "age": det.age,
                })

            self.store.save_detection(image_id, ckey, self.detector.signature, image.get("title", ""),
                                      saved_detections, media_type="image", source_id=image_id)

            # Auto-match against known performers if available
            if self.settings.auto_tag_stash:
                self.auto_match_image_faces(image_id, saved_detections)

        except Exception:
            for p in paths:
                p.unlink(missing_ok=True)
            raise

        return {"image_id": image_id, "faces_found": len(saved_detections)}

    def process_video(self, payload: dict) -> dict:
        scene_id = str(payload["scene_id"])
        file_id = str(payload["file_id"])
        frames = self.stash.extract_video_frames(payload["path"], payload.get("duration", 0), self.settings.video_frame_count)
        faces_found = 0
        for frame_index, (frame_time, picture) in enumerate(frames):
            self.check_stop()
            self.progress("detecting video frame", f"frame {frame_index + 1}/6 at {frame_time:.2f}s")
            media_key = f"video:{scene_id}:{file_id}:frame:{frame_index}"
            detections = self.detector.detect(picture)
            saved = self._save_detections(picture, detections)
            self.store.save_detection(media_key, payload["content_key"], self.detector.signature,
                                      payload.get("title", ""), saved, media_type="video",
                                      source_id=scene_id, file_id=file_id,
                                      frame_index=frame_index, frame_time=frame_time)
            faces_found += len(saved)
            if self.settings.auto_tag_stash:
                self.auto_match_scene_faces(scene_id, saved)
        return {"scene_id": scene_id, "file_id": file_id, "frames": len(frames), "faces_found": faces_found}

    def _save_detections(self, picture: Image.Image, detections: list) -> List[dict]:
        saved, width, height = [], *picture.size
        for det in detections:
            box = det.box
            margin = int(max(box[2] - box[0], box[3] - box[1]) * 0.15)
            crop = picture.crop((max(0, box[0] - margin), max(0, box[1] - margin),
                                 min(width, box[2] + margin), min(height, box[3] + margin)))
            crop.thumbnail((256, 256), Image.Resampling.LANCZOS)
            filename = uuid.uuid4().hex + ".jpg"
            crop_path = self.store.crops / filename
            crop.save(crop_path, format="JPEG", quality=90)
            crop_path.chmod(0o600)
            saved.append({"box": box, "score": det.score, "crop": filename,
                          "embedding": det.embedding, "gender": det.gender, "age": det.age})
        return saved

    def auto_match_image_faces(self, image_id: str, detections: List[dict]):
        known = self.store.get_known_performers()
        if not known:
            return

        known_embs = np.stack([k["embedding"] for k in known])
        matched_performer_ids = set()

        for det in detections:
            emb = det.get("embedding")
            if emb is None:
                continue

            # Cosine similarity: dot product of normalized vectors
            sims = np.dot(known_embs, emb)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_dist = 1.0 - best_sim

            if best_dist <= self.settings.performer_threshold:
                perf = known[best_idx]
                matched_performer_ids.add(perf["performer_id"])
                log.info("Auto-matched face on image %s to performer %s (sim: %.3f)",
                         image_id, perf["name"], best_sim)

        if matched_performer_ids:
            try:
                self.stash.add_performers([image_id], list(matched_performer_ids))
            except Exception as e:
                log.warning("Failed to auto-tag image %s in Stash: %s", image_id, e)

    def auto_match_scene_faces(self, scene_id: str, detections: List[dict]):
        known = self.store.get_known_performers()
        if not known:
            return
        known_embs = np.stack([k["embedding"] for k in known])
        matched = set()
        for det in detections:
            emb = det.get("embedding")
            if emb is None:
                continue
            sims = np.dot(known_embs, emb)
            idx = int(np.argmax(sims))
            if 1.0 - float(sims[idx]) <= self.settings.performer_threshold:
                matched.add(known[idx]["performer_id"])
        if matched:
            try:
                self.stash.add_scene_performers([scene_id], list(matched))
            except Exception as exc:
                log.warning("Failed to auto-tag scene %s in Stash: %s", scene_id, exc)

    def auto_tag(self, payload: dict) -> dict:
        paths = payload.get("paths") or list(self.settings.auto_tag_paths)
        self.progress("matching names from file paths", ":".join(paths))
        applied = self.stash.auto_tag_content(paths)
        if not applied:
            raise UpstreamError("Stash did not accept metadata auto-tag request", retryable=True)
        log.info("Requested Stash auto-tag from paths: %s", paths)
        return {"accepted": True, "paths": paths}

    def run_clustering(self) -> dict:
        if self.store.has_pending_media_jobs():
            raise UpstreamError("Waiting for photo/video jobs before clustering", retryable=True)
        log.info("Running DBSCAN face clustering...")
        removed = self.store.reset_unlinked_clusters()
        faces = self.store.get_all_faces_with_embeddings(unclustered_only=True)
        if not faces:
            return {"clusters_created": 0, "faces_evaluated": 0, "clusters_reset": removed}

        assignments, cluster_stats = self.clusterer.cluster(faces)
        created = self.store.apply_clustering(assignments, cluster_stats)
        log.info("DBSCAN clustering finished: %d clusters created from %d faces", created, len(faces))
        return {"clusters_created": created, "faces_evaluated": len(faces), "clusters_reset": removed}

    def sync_performers(self) -> dict:
        log.info("Syncing performers from Stash...")
        page = 1
        synced = 0
        while True:
            self.check_stop()
            result = self.stash.performers(page=page)
            performers = result.get("performers", [])
            if not performers:
                break

            for p in performers:
                self.check_stop()
                img_path = p.get("image_path")
                if not img_path:
                    continue
                try:
                    # Download performer image and detect face
                    pil_img = self.stash.download({"paths": {"image": img_path}})
                    dets = self.detector.detect(pil_img)
                    if dets and dets[0].embedding is not None:
                        self.store.save_known_performer(
                            performer_id=str(p["id"]),
                            name=p["name"],
                            image_url=img_path,
                            embedding=dets[0].embedding,
                        )
                        synced += 1
                except Exception as e:
                    log.debug("Could not process avatar for performer %s: %s", p.get("name"), e)

            page += 1

        log.info("Synced %d performers with facial embeddings from Stash.", synced)
        return {"synced_performers": synced}
