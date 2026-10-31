"""Face detection and 512-dimensional ArcFace embedding extraction (Immich stack)."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from PIL import Image

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
        self.device = "CPU"
        self.app = None

        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=self.model_name,
            root=str(self.models_dir),
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.signature = f"insightface:{self.model_name}:arcface512:{self.min_score}:{self.min_size}:v2"
        log.info("InsightFace initialized with %s on CPU", self.model_name)

    def info(self) -> dict:
        return {
            "provider": "insightface",
            "model_name": self.model_name,
            "actual_device": self.device,
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
