# Architecture

Stash Face Recognition is a standalone service plus a small Stash plugin
dispatcher. The service performs automatic face recognition and clustering; the
plugin only forwards Stash events and task requests.

## Processing pipeline

```text
Stash image
   │
   ├─ Stash Image.Create.Post / Image.Update.Post hook
   │
   ▼
Plugin dispatcher ── authenticated HTTP ──> FastAPI job queue
                                               │
                                               ▼
                                      Stash GraphQL/image client
                                               │
                                               ▼
                                      InsightFace buffalo_l
                                      ├─ SCRFD face detection
                                      ├─ five-point alignment
                                      └─ ArcFace 512-D embedding
                                               │
                                               ▼
                                      SQLite + private face crop
                                               │
                                               ▼
                                      DBSCAN cosine clustering
                                               │
                                               ▼
                                      People dashboard / Stash tagging
```

## Identity model

The service does not claim an identity merely because a face was detected.
Unknown faces become automatic clusters. A user links a cluster to an existing
Stash Performer or creates a new Performer. That action stores the cluster
centroid as a reference embedding and tags the cluster's Stash images.

Future images are compared against linked reference embeddings. A match is
applied only when cosine distance is below the configured threshold. Automatic
assignments are additive: existing Stash Performer links are never removed.

## Storage

Each Stash instance has an independent data directory:

- `review.sqlite3`: images, faces, embeddings, clusters, jobs and reference faces.
- `crops/`: private JPEG face crops, mode `0600`.
- `worker.lock`: exclusive single-worker lock.

The database stores embeddings as normalized `float32` vectors (512 values,
2048 bytes each). SQLite is the source of truth; clustering currently uses
scikit-learn DBSCAN with cosine distance.

## Job model

Jobs are persisted and deduplicated by action and image/model fingerprint.
The worker supports:

- `scan`: enumerate Stash images and enqueue changed images.
- `image`: download and process one image.
- `cluster`: rebuild unnamed clusters while preserving linked clusters.
- `sync_performers`: import usable Performer portraits as references.
- `forget`: remove local state for a deleted Stash image.

The worker runs with one Uvicorn process per data directory. This prevents two
workers from claiming the same SQLite job queue.

## Security boundaries

- All API endpoints require `X-API-Token`.
- Stash API credentials are read only from the environment file.
- Image URLs must resolve to the configured Stash origin.
- The dashboard and crop endpoints are authenticated.
- Crop and embedding data remain under the service data directory.
- HTTPS or a trusted VPN is required when the dashboard is reachable over a network.

## Multiple Stash instances

The systemd template `stash-face@.service` uses one environment file and data
directory per instance. Example:

```text
/etc/stash-face/local.env  -> /var/lib/stash-face/local -> port 8765
/etc/stash-face/remote.env -> /var/lib/stash-face/remote -> port 8766
```

The same source code can therefore serve multiple independent Stash libraries
without sharing embeddings, clusters, tokens, or SQLite state.
