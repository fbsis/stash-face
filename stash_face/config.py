import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be true or false")
    return value in ("true", "1")


@dataclass(frozen=True)
class Settings:
    stash_url: str
    token: str
    data_dir: Path
    stash_api_key: str = ""
    provider: str = "insightface"
    model_name: str = "buffalo_l"
    models_dir: Path = Path("/var/lib/stash-face/models")
    cluster_eps: float = 0.45
    min_faces: int = 2
    performer_threshold: float = 0.42
    auto_tag_stash: bool = True
    device: str = "CPU"
    workers: int = 1
    min_score: float = 0.5
    min_size: int = 40
    timeout: float = 15
    max_attempts: int = 3
    retry_seconds: float = 5
    max_image_bytes: int = 20_000_000
    max_image_pixels: int = 40_000_000
    max_preview_bytes: int = 512_000_000
    video_roots: tuple[Path, ...] = (Path("/mnt"),)
    auto_tag_paths: tuple[str, ...] = ("/mnt",)
    video_frame_count: int = 6

    def __post_init__(self):
        url = urlsplit(self.stash_url)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("STASH_URL must be an HTTP(S) base URL without credentials, query or fragment")
        if len(self.token) < 32 or self.token.startswith(("replace-", "change-")):
            raise ValueError("FACE_TOKEN must contain at least 32 random characters")
        if len(self.token) < 16:
            raise ValueError("FACE_TOKEN must contain at least 16 characters")
        if self.device not in ("AUTO", "CPU", "GPU") or self.workers < 1:
            raise ValueError("FACE_DEVICE must be AUTO, CPU or GPU and FACE_WORKERS must be positive")
        if not 0 <= self.min_score <= 1 or self.min_size < 8:
            raise ValueError("Invalid detection score or minimum size")
        if not 0 < self.cluster_eps <= 2 or self.min_faces < 2:
            raise ValueError("Invalid clustering configuration")
        if self.video_frame_count != 6 or not self.video_roots or not self.auto_tag_paths:
            raise ValueError("Video indexing requires exactly six frames and at least one allowed root")

    @classmethod
    def from_env(cls):
        data_dir = Path(os.getenv("FACE_DATA_DIR", "/var/lib/stash-face/local")).resolve()
        models_dir = Path(os.getenv("FACE_MODELS_DIR", "/var/lib/stash-face/models")).resolve()
        video_roots = tuple(Path(root).resolve() for root in os.getenv("FACE_VIDEO_ROOTS", "/mnt").split(":") if root)
        auto_tag_paths = tuple(path for path in os.getenv("FACE_AUTO_TAG_PATHS", "/mnt").split(":") if path)
        return cls(
            stash_url=os.environ["STASH_URL"].rstrip("/"),
            stash_api_key=os.getenv("STASH_API_KEY", ""),
            token=os.environ["FACE_TOKEN"],
            data_dir=data_dir,
            provider=os.getenv("FACE_PROVIDER", "insightface").lower(),
            model_name=os.getenv("FACE_MODEL_NAME", "buffalo_l"),
            models_dir=models_dir,
            cluster_eps=float(os.getenv("FACE_CLUSTER_EPS", "0.45")),
            min_faces=int(os.getenv("FACE_MIN_FACES", "2")),
            performer_threshold=float(os.getenv("FACE_PERFORMER_THRESHOLD", "0.42")),
            auto_tag_stash=env_bool("FACE_AUTO_TAG_STASH", True),
            device=os.getenv("FACE_DEVICE", "CPU").upper(),
            workers=int(os.getenv("FACE_WORKERS", str(os.cpu_count() or 1))),
            min_score=float(os.getenv("FACE_MIN_SCORE", "0.5")),
            min_size=int(os.getenv("FACE_MIN_SIZE", "40")),
            timeout=float(os.getenv("FACE_TIMEOUT", "15")),
            max_attempts=int(os.getenv("FACE_MAX_ATTEMPTS", "3")),
            retry_seconds=float(os.getenv("FACE_RETRY_SECONDS", "5")),
            max_image_bytes=int(os.getenv("FACE_MAX_IMAGE_BYTES", "20000000")),
            max_image_pixels=int(os.getenv("FACE_MAX_IMAGE_PIXELS", "40000000")),
            max_preview_bytes=int(os.getenv("FACE_MAX_PREVIEW_BYTES", "512000000")),
            video_roots=video_roots,
            auto_tag_paths=auto_tag_paths,
            video_frame_count=6,
        )
