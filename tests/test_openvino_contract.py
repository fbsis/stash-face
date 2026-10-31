"""The recognition runtime uses InsightFace/ONNX Runtime rather than OpenVINO."""

from stash_face.config import Settings


def test_recognition_defaults_are_cpu_safe(tmp_path):
    settings = Settings(
        stash_url="https://stash.example.test",
        token="test-token-" + "x" * 32,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
    )
    assert settings.provider == "insightface"
    assert settings.device == "CPU"
    assert settings.model_name == "buffalo_l"
