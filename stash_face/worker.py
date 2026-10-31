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
        self.thread: threading.Thread | None = None
        self.lock = None

    def start(self):
        self.lock = (self.store.directory / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("Another worker owns this database; use uvicorn --workers 1") from None

        self.thread = threading.Thread(target=self.run, name="face-worker", daemon=True)
        self.thread.start()
        log.info("Face background worker started.")

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=self.settings.timeout * 2 + 5)

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
            self.stash.close()
            if self.lock:
                self.lock.close()

    def execute(self, job: dict) -> dict:
        kind = job["kind"]
        payload = job.get("payload", {})

        if kind == "scan":
            return self.scan(payload)
        elif kind == "image":
            return self.process_image(payload["image_id"], payload.get("force", False))
        elif kind == "cluster":
            return self.run_clustering()
        elif kind == "forget":
            return {"forgotten": self.store.forget_image(payload["image_id"])}
        elif kind == "sync_performers":
            return self.sync_performers()
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

                if not force and self.store.unchanged(img_id, content_key(image), self.detector.signature):
                    skipped += 1
                    continue

                key = f"image:{img_id}:{content_key(image)}:{self.detector.signature}:{force}"
                _, created = self.store.enqueue("image", {"image_id": img_id, "force": force}, key)
                enqueued += int(created)

            received += len(items)
            if received >= total or not items:
                break
            page += 1

        # Automatically enqueue clustering once scanning is queued
        if enqueued > 0:
            self.store.enqueue("cluster", {}, "cluster:after-scan")

        return {"enqueued": enqueued, "unchanged": skipped, "total_seen": len(seen)}

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

        picture = self.stash.download(image)
        self.check_stop()

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

            self.store.save_detection(image_id, ckey, self.detector.signature, image.get("title", ""), saved_detections)

            # Auto-match against known performers if available
            if self.settings.auto_tag_stash:
                self.auto_match_image_faces(image_id, saved_detections)

        except Exception:
            for p in paths:
                p.unlink(missing_ok=True)
            raise

        return {"image_id": image_id, "faces_found": len(saved_detections)}

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

    def run_clustering(self) -> dict:
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
