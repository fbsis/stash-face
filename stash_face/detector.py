"""Face detection and 512-dimensional ArcFace embedding extraction (Immich stack)."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from PIL import Image
from insightface.app.common import Face
from insightface.utils import face_align

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    box: list[int]  # [x1, y1, x2, y2]
    score: float
    embedding: Optional[np.ndarray] = None  # 512-dim float32 L2-normalized vector
    gender: int = -1  # 0=female, 1=male, -1=unknown
    age: int = -1


def valid_box(box: list[int], width: int, height: int, min_size: int) -> Optional[list[int]]:
    if len(box) != 4 or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return [x1, y1, x2, y2] if min(x2 - x1, y2 - y1) >= min_size else None


class Detector:
    def __init__(self, settings: Any):
        self.settings = settings
        self.provider = getattr(settings, "provider", "insightface").lower()
        self.model_name = getattr(settings, "model_name", "buffalo_l")
        self.models_dir = getattr(settings, "models_dir", Path("/var/lib/stash-face/models"))
        self.min_score = float(getattr(settings, "min_score", 0.5))
        self.min_size = int(getattr(settings, "min_size", 40))
        requested_device = str(getattr(settings, "device", "AUTO")).upper()
        self.device = "CPU"
        self.execution_provider = "CPUExecutionProvider"
        providers = ["CPUExecutionProvider"]

        if requested_device in ("AUTO", "GPU"):
            try:
                import onnxruntime as ort
                import openvino as ov
                available = set(ort.get_available_providers())
                devices = set(ov.Core().available_devices)
                if "OpenVINOExecutionProvider" in available and any(device.startswith("GPU") for device in devices):
                    providers = [("OpenVINOExecutionProvider", {"device_type": "GPU"}),
                                 "CPUExecutionProvider"]
                    self.device = "GPU"
                    self.execution_provider = "OpenVINOExecutionProvider"
                elif requested_device == "GPU":
                    log.warning("FACE_DEVICE=GPU requested, but OpenVINO reports no GPU; falling back to CPU")
            except Exception as exc:
                if requested_device == "GPU":
                    log.warning("GPU provider unavailable (%s); falling back to CPU", exc)

        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=self.model_name,
            root=str(self.models_dir),
            providers=providers,
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        # Provider changes (CPU/GPU) do not change the embedding model output;
        # keep the media fingerprint stable so switching accelerators never requeues
        # an already indexed library.
        self.signature = f"insightface:{self.model_name}:arcface512:{self.min_score}:{self.min_size}:v2"
        log.info("InsightFace initialized with %s on %s", self.model_name, self.device)

    def detect_batch(self, images: List[Image.Image]) -> List[List[Detection]]:
        """Detect faces per image and batch ArcFace recognition across images.

        SCRFD detection remains per image, but the expensive ArcFace embedding
        inference receives all aligned crops in one OpenVINO call. Fall back to
        the established single-image path if the loaded recognition model does
        not expose the batch API.
        """
        if not images:
            return []
        recognition = self.app.models.get("recognition")
        if recognition is None or not hasattr(recognition, "get_feat"):
            return [self.detect(image) for image in images]

        bgr_images = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]
        pending = []
        results: List[List[Detection]] = [[] for _ in images]
        for image_index, bgr in enumerate(bgr_images):
            bboxes, kpss = self.app.det_model.detect(bgr, max_num=0, metric="default")
            for face_index in range(bboxes.shape[0]):
                bbox = bboxes[face_index, 0:4]
                score = float(bboxes[face_index, 4])
                if score < self.min_score:
                    continue
                clamped = valid_box(bbox.tolist(), images[image_index].width,
                                    images[image_index].height, self.min_size)
                if clamped is None or kpss is None:
                    continue
                face = Face(bbox=bbox, kps=kpss[face_index], det_score=score)
                aligned = face_align.norm_crop(bgr, landmark=face.kps,
                                               image_size=recognition.input_size[0])
                pending.append((image_index, face, clamped, aligned))

        if not pending:
            return results
        embeddings = recognition.get_feat([item[3] for item in pending])
        for item, embedding in zip(pending, embeddings):
            image_index, face, clamped, _ = item
            for taskname, model in self.app.models.items():
                if taskname not in ("detection", "recognition"):
                    model.get(bgr_images[image_index], face)
            emb = embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            results[image_index].append(Detection(
                box=clamped,
                score=round(float(face.det_score), 4),
                embedding=emb,
                gender=-1 if getattr(face, "gender", None) is None else int(face.gender),
                age=-1 if getattr(face, "age", None) is None else int(face.age),
            ))
        return results

    def info(self) -> dict:
        return {
            "provider": "insightface",
            "model_name": self.model_name,
            "actual_device": self.device,
            "execution_provider": self.execution_provider,
            "embeddings": True,
            "min_score": self.min_score,
            "min_size": self.min_size,
        }

    def detect(self, image: Image.Image) -> List[Detection]:
        width, height = image.size
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raw_faces = self.app.get(bgr)
        detections = []
        for face in raw_faces:
            score = float(face.det_score)
            if score < self.min_score:
                continue
            clamped = valid_box(face.bbox.tolist(), width, height, self.min_size)
            if clamped is None:
                continue
            emb = face.embedding
            if emb is not None:
                emb = emb.astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            detections.append(
                Detection(
                    box=clamped,
                    score=round(score, 4),
                    embedding=emb,
                    gender=int(face.gender) if face.gender is not None else -1,
                    age=int(face.age) if face.age is not None else -1,
                )
            )
        return detections
