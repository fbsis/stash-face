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
        self.signature = (f"insightface:{self.model_name}:arcface512:{self.min_score}:{self.min_size}:"
                          f"{self.execution_provider}:v3")
        log.info("InsightFace initialized with %s on %s", self.model_name, self.device)

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
