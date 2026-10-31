import os
from dataclasses import replace

import pytest

from stash_face.config import Settings
from stash_face.detector import Detection


@pytest.fixture
def settings(tmp_path):
    return Settings(stash_url="http://127.0.0.1:49171/stash", token="unit-test-secret-" + "x" * 32,
                    data_dir=tmp_path, stash_api_key="mock-api-key", timeout=0.5, retry_seconds=0.05,
                    min_size=8)


class SyntheticDetector:
    """Deterministic boxes on a blank fixture, NOT a face recognition/detection model."""
    signature = "synthetic-test-detector"
    calls = 0

    def __init__(self, settings):
        self.settings = settings

    def detect(self, image):
        self.calls += 1
        return [Detection((5, 5, 45, 45), 0.9), Detection((60, 60, 100, 100), 0.85)]

    def info(self):
        return {"provider": "synthetic-test", "actual_device": "CPU", "identity_matching": False}


@pytest.fixture
def mock_url():
    url = os.getenv("STASH_MOCK_URL")
    if not url:
        pytest.skip("Run scripts/integration.py for the real HTTP Stash simulation")
    return url.rstrip("/")


@pytest.fixture
def remote_settings(settings, mock_url):
    return replace(settings, stash_url=mock_url)
