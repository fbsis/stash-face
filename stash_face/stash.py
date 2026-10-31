"""Stash GraphQL and Image API client with Performer management."""
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError

log = logging.getLogger(__name__)

IMAGE_FIELDS = """id title updated_at paths { image } performers { id name }
visual_files { __typename ... on ImageFile { id mod_time size fingerprints { type value } } }"""
IMAGES = "query ReviewImages($page:Int!) { findImages(filter:{page:$page,per_page:100,sort:\"id\",direction:ASC}) { count images { " + IMAGE_FIELDS + " } } }"
IMAGE = "query ReviewImage($id:ID!) { findImage(id:$id) { " + IMAGE_FIELDS + " } }"
SCENE = """query ReviewScene($id:ID!) { findScene(id:$id) { paths { preview } files { __typename id path } } }"""
SCENES = """query ReviewScenes($page:Int!) { findScenes(filter:{page:$page,per_page:100,sort:\"id\",direction:ASC}) {
 count scenes { id title updated_at paths { screenshot preview } files { __typename ... on VideoFile {
 id path mod_time size duration width height } } } } }"""
PERFORMERS = """query ReviewPerformers($page:Int!,$q:String!) {
 findPerformers(filter:{page:$page,per_page:50,q:$q,sort:"name",direction:ASC}) {
 count performers { id name image_path } } }"""
PERFORMER = "query ReviewPerformer($id:ID!) { findPerformer(id:$id) { id name image_path } }"
ADD_PERFORMERS = """mutation ReviewAddPerformers($input:BulkImageUpdateInput!) {
 bulkImageUpdate(input:$input) { id performers { id } } }"""
CREATE_PERFORMER = """mutation CreatePerformer($input:PerformerCreateInput!) {
 performerCreate(input:$input) { id name image_path } }"""


class UpstreamError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def origin(url: str):
    parts = urlsplit(url)
    return parts.scheme, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)


def content_key(image: dict) -> str:
    files = [f for f in image.get("visual_files", []) if f.get("__typename") == "ImageFile"]
    facts = [{k: f.get(k) for k in ("id", "mod_time", "size")} |
             {"fingerprints": sorted(f.get("fingerprints", []), key=lambda x: (x["type"], x["value"]))}
             for f in files]
    facts.sort(key=lambda f: str(f["id"]))
    value = facts or {"updated_at": image.get("updated_at", "")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class StashClient:
    def __init__(self, settings: Any):
        self.settings = settings
        self.base = settings.stash_url.rstrip("/")
        headers = {"ApiKey": settings.stash_api_key} if settings.stash_api_key else {}
        self.client = httpx.Client(
            timeout=settings.timeout,
            follow_redirects=False,
            trust_env=False,
            headers=headers,
        )

    def close(self):
        self.client.close()

    def check(self, response: httpx.Response):
        if response.status_code >= 300:
            raise UpstreamError(f"Stash returned HTTP {response.status_code}",
                                response.status_code == 429 or response.status_code >= 500)

    def gql(self, query: str, variables: Optional[dict] = None) -> dict:
        try:
            response = self.client.post(self.base + "/graphql",
                                        json={"query": query, "variables": variables or {}})
            self.check(response)
            result = response.json()
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Stash connection failed: {exc}", True) from None
        except ValueError:
            raise UpstreamError("Stash returned invalid JSON") from None
        if not isinstance(result, dict) or result.get("errors") or not isinstance(result.get("data"), dict):
            errors = result.get("errors") if isinstance(result, dict) else "unknown error"
            raise UpstreamError(f"Stash rejected GraphQL: {errors}")
        return result["data"]

    def image(self, image_id: str) -> Optional[dict]:
        data = self.gql(IMAGE, {"id": str(image_id)})
        return data.get("findImage")

    def image_page(self, page: int) -> dict:
        data = self.gql(IMAGES, {"page": page})
        result = data.get("findImages", {})
        if not isinstance(result.get("images"), list) or not isinstance(result.get("count"), int):
            raise UpstreamError("Invalid image page from Stash")
        return result

    def scene_page(self, page: int) -> dict:
        data = self.gql(SCENES, {"page": page})
        result = data.get("findScenes", {})
        if not isinstance(result.get("scenes"), list) or not isinstance(result.get("count"), int):
            raise UpstreamError("Invalid scene page from Stash")
        return result

    def scene_preview(self, scene_id: str, file_id: str) -> Optional[str]:
        scene = self.gql(SCENE, {"id": str(scene_id)}).get("findScene") or {}
        paths = scene.get("paths") or {}
        if not paths.get("preview"):
            return None
        if any(str(file.get("id")) == str(file_id) for file in scene.get("files", [])):
            return str(paths["preview"])
        return None

    def performers(self, q: str = "", page: int = 1) -> dict:
        return self.gql(PERFORMERS, {"q": q, "page": page}).get("findPerformers", {"count": 0, "performers": []})

    def performer(self, performer_id: str) -> Optional[dict]:
        return self.gql(PERFORMER, {"id": str(performer_id)}).get("findPerformer")

    def add_performers(self, image_ids: List[str], performer_ids: List[str]) -> bool:
        if not image_ids or not performer_ids:
            return True
        clean_img_ids = [str(x) for x in set(image_ids)]
        clean_perf_ids = [str(x) for x in set(performer_ids)]
        result = self.gql(ADD_PERFORMERS, {"input": {
            "ids": clean_img_ids,
            "performer_ids": {"ids": clean_perf_ids, "mode": "ADD"}
        }})
        return bool(result.get("bulkImageUpdate"))

    def auto_tag_content(self, paths: List[str]) -> bool:
        result = self.gql("""mutation MetadataAutoTag($input:AutoTagMetadataInput!) {
            metadataAutoTag(input:$input)
        }""", {"input": {
            "paths": [str(path) for path in paths],
            "performers": [], "studios": [], "tags": [],
        }})
        return bool(result.get("metadataAutoTag"))

    def add_scene_performers(self, scene_ids: List[str], performer_ids: List[str]) -> bool:
        if not scene_ids or not performer_ids:
            return True
        result = self.gql("""mutation AddScenePerformers($input:BulkSceneUpdateInput!) {
            bulkSceneUpdate(input:$input) { id performers { id } } }""", {"input": {
            "ids": [str(x) for x in set(scene_ids)],
            "performer_ids": {"ids": [str(x) for x in set(performer_ids)], "mode": "ADD"},
        }})
        return bool(result.get("bulkSceneUpdate"))

    def _preview_target(self, preview_url: str) -> str:
        target = urljoin(self.base + "/", preview_url)
        target_parts = urlsplit(target)
        base_parts = urlsplit(self.base)
        base_path = base_parts.path.rstrip("/") + "/"
        if (origin(target) != origin(self.base) or target_parts.username or target_parts.password
                or not target_parts.path.startswith(base_path)):
            raise UpstreamError("Video preview URL is outside the configured Stash origin/base path")
        return target

    def _stage_preview(self, preview_url: str) -> Path:
        target = self._preview_target(preview_url)
        fd, name = tempfile.mkstemp(prefix="stash-face-preview-", suffix=".mp4",
                                    dir=str(self.settings.data_dir))
        os.close(fd)
        staged = Path(name)
        total = 0
        try:
            with self.client.stream("GET", target) as response:
                self.check(response)
                with staged.open("wb") as output:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > self.settings.max_preview_bytes:
                            raise UpstreamError("Video preview exceeds the configured byte limit")
                        output.write(chunk)
            staged.chmod(0o600)
            return staged
        except (httpx.HTTPError, OSError) as exc:
            staged.unlink(missing_ok=True)
            raise UpstreamError(f"Video preview download failed: {type(exc).__name__}", True) from None
        except Exception:
            staged.unlink(missing_ok=True)
            raise

    def extract_video_frames(self, path: str, duration: float, count: int = 6,
                             preview_url: Optional[str] = None) -> List[tuple[float, Image.Image]]:
        real_path = Path(os.path.realpath(path))
        roots = [Path(root).resolve() for root in self.settings.video_roots]
        if not any(real_path == root or root in real_path.parents for root in roots):
            raise UpstreamError("Video path is outside the configured video roots")
        ffmpeg = shutil.which("ffmpeg") or "/mnt/stash-normal/ffmpeg"
        if not os.path.isfile(ffmpeg):
            raise UpstreamError("ffmpeg is not installed for video frame extraction")

        input_path = real_path
        staged_preview = None
        if preview_url:
            try:
                staged_preview = self._stage_preview(preview_url)
                input_path = staged_preview
                log.info("Using existing Stash preview for video frame extraction")
            except UpstreamError as exc:
                log.warning("Existing Stash preview unavailable; falling back to original video: %s", exc)
        if staged_preview is None and not real_path.is_file():
            raise UpstreamError("Video file is not available in the LXC")

        safe_duration = max(float(duration or 0), 1.0)
        if staged_preview:
            ffprobe = shutil.which("ffprobe")
            if ffprobe:
                try:
                    probe = subprocess.run(
                        [ffprobe, "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
                        check=True, capture_output=True, text=True, timeout=30,
                    )
                    preview_duration = float(probe.stdout.strip())
                    if preview_duration > 0:
                        safe_duration = min(safe_duration, preview_duration)
                except (subprocess.SubprocessError, OSError, ValueError):
                    log.warning("Could not determine staged Stash preview duration; using scene duration")
        output = []
        try:
            for index in range(count):
                timestamp = safe_duration * (index + 1) / (count + 1)
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                           "-i", str(input_path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
                try:
                    result = subprocess.run(command, check=True, capture_output=True, timeout=90)
                    with Image.open(io.BytesIO(result.stdout)) as frame:
                        output.append((timestamp, frame.convert("RGB")))
                except (subprocess.SubprocessError, OSError, UnidentifiedImageError) as exc:
                    raise UpstreamError(f"Video frame extraction failed: {type(exc).__name__}", True) from None
            return output
        finally:
            if staged_preview:
                staged_preview.unlink(missing_ok=True)

    def create_performer(self, name: str, image_data_uri: str = "") -> dict:
        input_data = {"name": name.strip()}
        if image_data_uri:
            input_data["image"] = image_data_uri
        result = self.gql(CREATE_PERFORMER, {"input": input_data})
        created = result.get("performerCreate")
        if not created or not created.get("id"):
            raise UpstreamError(f"Failed to create Performer '{name}' in Stash")
        return created

    def download(self, image: dict) -> Image.Image:
        path = image.get("paths", {}).get("image")
        if not path:
            raise UpstreamError("Image has no download URL")
        target = urljoin(self.base + "/", path)
        target_parts = urlsplit(target)
        base_parts = urlsplit(self.base)
        base_path = base_parts.path.rstrip("/") + "/"
        if (origin(target) != origin(self.base) or target_parts.username or target_parts.password
                or not target_parts.path.startswith(base_path)):
            raise UpstreamError("Image URL is outside the configured Stash origin/base path")
        try:
            with self.client.stream("GET", target) as response:
                self.check(response)
                data = bytearray()
                for chunk in response.iter_bytes():
                    data.extend(chunk)
                    if len(data) > self.settings.max_image_bytes:
                        raise UpstreamError("Image exceeds byte limit")
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Image download failed: {exc}", True) from None
        try:
            with Image.open(io.BytesIO(data)) as source:
                return ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise UpstreamError("Invalid or corrupted image") from None
