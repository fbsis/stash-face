"""Immich-style DBSCAN face clustering using 512-dimensional ArcFace embeddings."""
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)


class Clusterer:
    def __init__(self, eps: float = 0.45, min_samples: int = 2):
        """
        eps: maximum cosine distance for faces to be considered the same person (default 0.45, matching Immich).
        min_samples: minimum faces required to form a cluster (default 2).
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, faces: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
        """
        faces: list of dicts with keys: 'id', 'embedding' (numpy array or bytes), 'crop', etc.
        Returns:
          assignments: {face_id: cluster_label} where cluster_label >= 0, or -1 for noise.
          cluster_stats: {cluster_label: {'face_count': int, 'exemplar_crop': str, 'exemplar_face_id': int}}
        """
        if len(faces) < self.min_samples:
            return {f["id"]: -1 for f in faces}, {}

        face_ids = []
        embeddings = []
        crops = []

        for f in faces:
            emb = f.get("embedding")
            if emb is None:
                continue
            if isinstance(emb, (bytes, bytearray)):
                emb = np.frombuffer(emb, dtype=np.float32)
            if emb.shape != (512,):
                continue
            face_ids.append(f["id"])
            embeddings.append(emb)
            crops.append(f.get("crop", ""))

        if len(face_ids) < self.min_samples:
            return {f["id"]: -1 for f in faces}, {}

        X = np.stack(embeddings).astype(np.float32)
        # Ensure L2 normalization for accurate cosine distance
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine", n_jobs=-1)
        labels = db.fit_predict(X)

        assignments: Dict[int, int] = {}
        clusters_indices: Dict[int, List[int]] = {}

        for i, (fid, label) in enumerate(zip(face_ids, labels)):
            lbl = int(label)
            assignments[fid] = lbl
            if lbl >= 0:
                clusters_indices.setdefault(lbl, []).append(i)

        cluster_stats: Dict[int, Dict[str, Any]] = {}
        for lbl, indices in clusters_indices.items():
            cluster_X = X[indices]
            # Centroid of cluster
            centroid = np.mean(cluster_X, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            # Find exemplar: face closest to centroid (medoid)
            similarities = np.dot(cluster_X, centroid)
            best_idx = indices[int(np.argmax(similarities))]

            cluster_stats[lbl] = {
                "face_count": len(indices),
                "exemplar_crop": crops[best_idx],
                "exemplar_face_id": face_ids[best_idx],
                "centroid": centroid,
            }

        return assignments, cluster_stats
