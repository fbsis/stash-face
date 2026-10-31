# Stash Face Recognition

An Immich-style facial recognition service for Stash. It detects faces in Stash
images, creates ArcFace embeddings with InsightFace, clusters similar faces
with DBSCAN, and optionally assigns linked Stash Performers automatically.

## Features

- InsightFace `buffalo_l` face detection and 512-dimensional ArcFace embeddings.
- CPU execution with ONNX Runtime; the service is isolated from the Stash process.
- Automatic background processing through Stash plugin hooks.
- Full-library scan task for existing Stash images.
- Automatic clustering of unknown faces by visual similarity.
- Automatic Performer tagging for clusters linked to a Stash Performer.
- Optional creation of a new Stash Performer from a discovered person cluster.
- Immich-style People grid with counts, search, filters, merge and hide actions.
- Persistent SQLite state and private face crops.
- Multiple independent Stash instances through a systemd service template.

This project performs biometric face recognition. Use it only where you have
appropriate consent and authorization. Similarity thresholds are conservative
but cannot guarantee correct identity assignment; review clusters before linking
them to a Performer.

## Architecture

```text
Stash GraphQL/images
        │
        ▼
Stash plugin dispatcher ── HTTP ──> FastAPI service
                                      │
                                      ├─ InsightFace / ArcFace embeddings
                                      ├─ SQLite embeddings and jobs
                                      ├─ DBSCAN clustering
                                      └─ HTTPS review dashboard
```

The service does not need access to Stash's media filesystem. It downloads image
URLs from the configured Stash API and stores only private crops and embeddings. For video indexing, an existing Stash `paths.preview` is preferred, staged temporarily under `FACE_DATA_DIR` for ffmpeg, and deleted after the six frames are extracted; the original video is used only as a fallback.

## Native installation

Target: Debian or Ubuntu, Python 3.11+.

```bash
git clone https://github.com/fbsis/stash-face.git
cd stash-face
sudo bash scripts/install-native.sh
```

The installer creates `/opt/stash-face/.venv`, the `stash-face` system user,
`/var/lib/stash-face`, `/etc/stash-face/env`, and a systemd unit. It does not
start the service until configuration is complete.

Example configuration:

```dotenv
STASH_URL=https://stash.example.com
STASH_API_KEY=your-stash-api-key
FACE_TOKEN=generate-a-long-random-service-secret
FACE_DATA_DIR=/var/lib/stash-face
FACE_MODELS_DIR=/var/lib/stash-face/models
FACE_BIND=127.0.0.1
FACE_PORT=8765
FACE_PROVIDER=insightface
FACE_MODEL_NAME=buffalo_l
FACE_CLUSTER_EPS=0.45
FACE_MIN_FACES=2
FACE_PERFORMER_THRESHOLD=0.42
FACE_AUTO_TAG_STASH=true
FACE_MIN_SCORE=0.7
FACE_MIN_SIZE=40
```

Generate the service token with `openssl rand -hex 32`. Keep the environment
file owned by `root:stash-face` with mode `0640`. The first service start
downloads the InsightFace model pack into `FACE_MODELS_DIR` if it is absent.

Start and verify:

```bash
sudo systemctl enable --now stash-face
curl -H "X-API-Token: $FACE_TOKEN" http://127.0.0.1:8765/health
```

Use one Uvicorn worker per data directory. A second worker is rejected because
one worker owns the SQLite job queue.

## Stash plugin

Copy these files into the Stash plugin directory:

```text
plugin/stashFaceReview.yml
plugin/stash_face_review.py
plugin/stash_face.json
```

Configure `stash_face.json`:

```json
{
  "service_url": "https://face.example.com",
  "token": "the-same-FACE_TOKEN"
}
```

Reload plugins in Stash. The plugin provides these tasks:

- **Test Remote Connection**: checks the service and configured Stash.
- **Scan & Recognize Faces**: scans all new or changed Stash images.
- **Re-cluster Faces (DBSCAN)**: rebuilds unnamed automatic clusters.
- **Sync Performers from Stash**: imports usable Performer portraits as reference faces.

Image create/update hooks enqueue recognition automatically. Deleted-image hooks
remove the local record. The scan task is resumable: processed images are skipped
when their file fingerprint and model signature are unchanged.

## Recognition workflow

1. Run **Sync Performers from Stash** if reference Performer portraits exist.
2. Run **Scan & Recognize Faces** for the library.
3. The worker detects faces and stores embeddings.
4. DBSCAN creates unnamed People clusters automatically.
5. Open the dashboard, inspect a cluster, and either link it to an existing
   Performer or create a new Performer.
6. Linked clusters are added to the corresponding Stash images. Future matching
   faces are tagged automatically when their distance is below
   `FACE_PERFORMER_THRESHOLD`.

A cluster link stores its centroid as a reference embedding. This is more useful
than relying only on a small Performer portrait and is updated when the cluster
is linked.

## Dashboard endpoints

- `GET /health`
- `GET /stats`
- `POST /jobs` with `scan`, `image`, `cluster`, `sync_performers`, or `forget`
- `GET /people`
- `GET /people/{id}`
- `GET /faces?filter=all|clustered|unclustered|named|unknown`

The dashboard's **All Faces** view displays paginated face crops with filters for
all, grouped, ungrouped, named/linked, and unknown faces. Original Stash media
is read through the Stash API or mounted media path and is not copied into the
service data directory; only private crops, embeddings, and metadata are stored.
- `PATCH /people/{id}`
- `POST /people/{id}/link-performer`
- `POST /people/{id}/create-performer`
- `POST /people/{id}/merge`

All endpoints require `X-API-Token`. Put the dashboard behind HTTPS or a trusted
VPN; do not expose the service token over plain HTTP.

## Data and backups

The service stores SQLite, embeddings, crops, and worker state in
`FACE_DATA_DIR`. Stop the service before making a consistent backup:

```bash
sudo systemctl stop stash-face
sudo tar -czf stash-face-backup.tgz -C /var/lib stash-face
sudo systemctl start stash-face
```

Face crops and embeddings are sensitive biometric data. Protect the data directory
and do not commit environment files, tokens, databases, crops, or model files.
