import os
from dataclasses import replace

import numpy as np
import pytest

from stash_face.clustering import Clusterer
from stash_face.config import Settings
from stash_face.stash import content_key
from stash_face.store import Store


def make_settings(tmp_path):
    return Settings(
        stash_url="https://stash.example.test",
        token="unit-test-secret-" + "x" * 32,
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
    )


def test_settings_rejects_weak_tokens_and_unsafe_urls(tmp_path):
    settings = make_settings(tmp_path)
    with pytest.raises(ValueError):
        replace(settings, token="short")
    with pytest.raises(ValueError):
        replace(settings, token="replace-" + "x" * 40)
    with pytest.raises(ValueError):
        replace(settings, stash_url="https://user:secret@stash.example.test")
    with pytest.raises(ValueError):
        replace(settings, stash_url="https://stash.example.test/?token=secret")


def test_environment_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("STASH_URL", "https://stash.example.test")
    monkeypatch.setenv("FACE_TOKEN", "a" * 40)
    monkeypatch.setenv("FACE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FACE_PROVIDER", "insightface")
    monkeypatch.setenv("FACE_CLUSTER_EPS", "0.45")
    config = Settings.from_env()
    assert config.provider == "insightface"
    assert config.model_name == "buffalo_l"
    assert config.cluster_eps == 0.45


def test_content_key_ignores_performer_metadata_changes():
    image = {
        "updated_at": "before",
        "visual_files": [{
            "__typename": "ImageFile", "id": "1", "mod_time": "1",
            "size": 42, "fingerprints": []
        }],
    }
    old = content_key(image)
    assert content_key(image | {"updated_at": "after", "performers": [{"id": "9"}]}) == old
    image["visual_files"][0]["mod_time"] = "2"
    assert content_key(image) != old


def test_store_persists_embeddings_and_clusters(tmp_path):
    store = Store(tmp_path)
    embeddings = [
        np.array([1, 0] + [0] * 510, dtype=np.float32),
        np.array([0.99, 0.1] + [0] * 510, dtype=np.float32),
        np.array([0.98, 0.2] + [0] * 510, dtype=np.float32),
    ]
    detections = []
    for i, embedding in enumerate(embeddings):
        crop = f"face-{i}.jpg"
        (store.crops / crop).write_bytes(b"jpeg")
        detections.append({"box": [0, 0, 40, 40], "score": 0.9, "crop": crop, "embedding": embedding})
    store.save_detection("1", "content", "insightface:test", "Test", detections)
    faces = store.get_all_faces_with_embeddings()
    assert len(faces) == 3
    assert faces[0]["embedding"].shape == (512,)

    assignments, stats = Clusterer(eps=0.45, min_samples=2).cluster(faces)
    assert len(stats) == 1
    assert set(assignments.values()) == {0}
    assert store.apply_clustering(assignments, stats) == 1
    assert store.get_clusters()[0]["face_count"] == 3


def test_reclustering_preserves_linked_clusters(tmp_path):
    store = Store(tmp_path)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1
    store.crops.joinpath("face.jpg").write_bytes(b"jpeg")
    store.save_detection("1", "content", "detector", "Test", [{
        "box": [0, 0, 40, 40], "score": 0.9, "crop": "face.jpg", "embedding": embedding
    }])
    faces = store.get_all_faces_with_embeddings()
    assignments, stats = Clusterer(min_samples=1).cluster(faces)
    store.apply_clustering(assignments, stats)
    cluster_id = store.get_clusters()[0]["id"]
    image_ids = store.link_cluster_to_performer(cluster_id, "7", "Known Person")
    assert image_ids == ["1"]
    assert store.reset_unlinked_clusters() == 0
    assert store.get_clusters()[0]["performer_id"] == "7"


def test_job_deduplication(tmp_path):
    store = Store(tmp_path)
    first, created = store.enqueue("scan", {}, "scan")
    assert created
    second, created_again = store.enqueue("scan", {}, "scan")
    assert not created_again
    assert first == second
    job = store.claim()
    assert job["id"] == first
    store.finish(first, {"ok": True})
    assert store.jobs()[0]["state"] == "done"
