"""SQLite persistence for faces, 512-dim embeddings, clusters, and jobs."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.crops = self.directory / "crops"
        self.crops.mkdir(exist_ok=True, mode=0o700)
        self.path = self.directory / "review.sqlite3"

        with self.connection() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
            CREATE TABLE IF NOT EXISTS images (
              id TEXT PRIMARY KEY,
              content_key TEXT NOT NULL,
              detector_key TEXT NOT NULL,
              revision INTEGER NOT NULL,
              title TEXT NOT NULL,
              processed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clusters (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              performer_id TEXT,
              exemplar_crop TEXT,
              face_count INTEGER NOT NULL DEFAULT 0,
              hidden INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_clusters_performer ON clusters(performer_id);

            CREATE TABLE IF NOT EXISTS faces (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
              revision INTEGER NOT NULL,
              active INTEGER NOT NULL DEFAULT 1,
              box TEXT NOT NULL,
              score REAL,
              crop TEXT NOT NULL,
              embedding BLOB,
              gender INTEGER DEFAULT -1,
              age INTEGER DEFAULT -1,
              cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
              performer_id TEXT,
              matched_by TEXT,
              created_at REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_faces_active ON faces(active, id);
            CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_faces_performer ON faces(performer_id);

            CREATE TABLE IF NOT EXISTS media (
              media_key TEXT PRIMARY KEY,
              media_type TEXT NOT NULL,
              source_id TEXT NOT NULL,
              file_id TEXT,
              title TEXT NOT NULL DEFAULT '',
              content_key TEXT NOT NULL DEFAULT '',
              detector_key TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL DEFAULT 'unindexed',
              frame_index INTEGER,
              frame_time REAL,
              parent_folder TEXT NOT NULL DEFAULT '',
              face_count INTEGER NOT NULL DEFAULT 0,
              last_error TEXT,
              indexed_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_media_status ON media(status, media_type);
            CREATE INDEX IF NOT EXISTS idx_media_source ON media(media_type, source_id, file_id);

            CREATE TABLE IF NOT EXISTS media_assignments (
              media_key TEXT PRIMARY KEY REFERENCES media(media_key) ON DELETE CASCADE,
              cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
              performer_id TEXT,
              matched_by TEXT NOT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS suggestions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              media_key TEXT NOT NULL REFERENCES media(media_key) ON DELETE CASCADE,
              cluster_id INTEGER NOT NULL REFERENCES clusters(id) ON DELETE CASCADE,
              score REAL NOT NULL,
              reason TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending',
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              UNIQUE(media_key, cluster_id)
            );
            CREATE INDEX IF NOT EXISTS idx_suggestions_status ON suggestions(status, score DESC);

            CREATE TABLE IF NOT EXISTS known_performers (
              performer_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              image_url TEXT,
              embedding BLOB NOT NULL,
              updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              kind TEXT NOT NULL,
              payload TEXT NOT NULL,
              dedupe_key TEXT NOT NULL,
              state TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              rerun INTEGER NOT NULL DEFAULT 0,
              next_attempt REAL NOT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              result TEXT,
              error TEXT,
              stage TEXT NOT NULL DEFAULT 'queued',
              detail TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS active_job_key ON jobs(dedupe_key)
              WHERE state IN ('queued', 'running', 'retry');
            """)

            # Backfill the media inventory for photos indexed before the inventory was added.
            con.execute("""INSERT OR IGNORE INTO media
                (media_key, media_type, source_id, title, content_key, detector_key, status, face_count, indexed_at)
                SELECT id, 'image', id, title, content_key, detector_key,
                       'indexed', (SELECT count(*) FROM faces f WHERE f.image_id=images.id AND f.active=1), processed_at
                FROM images""")

            # Ensure any missing columns exist
            job_cols = {row[1] for row in con.execute("PRAGMA table_info(jobs)")}
            if "stage" not in job_cols:
                con.execute("ALTER TABLE jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'queued'")
            if "detail" not in job_cols:
                con.execute("ALTER TABLE jobs ADD COLUMN detail TEXT")

            media_cols = {row[1] for row in con.execute("PRAGMA table_info(media)")}
            if "parent_folder" not in media_cols:
                con.execute("ALTER TABLE media ADD COLUMN parent_folder TEXT NOT NULL DEFAULT ''")
            con.execute("CREATE INDEX IF NOT EXISTS idx_media_folder ON media(parent_folder, media_type)")

            cols = {row[1] for row in con.execute("PRAGMA table_info(faces)")}
            if "embedding" not in cols:
                con.execute("ALTER TABLE faces ADD COLUMN embedding BLOB")
            if "gender" not in cols:
                con.execute("ALTER TABLE faces ADD COLUMN gender INTEGER DEFAULT -1")
            if "age" not in cols:
                con.execute("ALTER TABLE faces ADD COLUMN age INTEGER DEFAULT -1")
            if "cluster_id" not in cols:
                con.execute("ALTER TABLE faces ADD COLUMN cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL")
            if "performer_id" not in cols:
                con.execute("ALTER TABLE faces ADD COLUMN performer_id TEXT")
            if "matched_by" not in cols:
                con.execute("ALTER TABLE faces ADD COLUMN matched_by TEXT")

        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    def image_ids(self) -> Set[str]:
        with self.connection() as con:
            return {row[0] for row in con.execute("SELECT id FROM images")}

    def unchanged(self, image_id: str, key: str, detector_key: str) -> bool:
        with self.connection() as con:
            row = con.execute("SELECT * FROM images WHERE id=?", (str(image_id),)).fetchone()
        return bool(row and row["content_key"] == key and row["detector_key"] == detector_key)

    def register_media(
        self,
        media_key: str,
        media_type: str,
        source_id: str,
        title: str = "",
        content_key: str = "",
        file_id: Optional[str] = None,
        frame_index: Optional[int] = None,
        frame_time: Optional[float] = None,
        parent_folder: str = "",
    ) -> bool:
        with self.connection() as con:
            row = con.execute("SELECT content_key, status FROM media WHERE media_key=?", (media_key,)).fetchone()
            if row and row["content_key"] == content_key and row["status"] not in ("failed", "unindexed"):
                con.execute("UPDATE media SET parent_folder=?, title=? WHERE media_key=?",
                            (parent_folder or "", title or "", media_key))
                return False
            con.execute(
                """INSERT INTO media
                   (media_key, media_type, source_id, file_id, title, content_key, status,
                    frame_index, frame_time, parent_folder)
                   VALUES (?, ?, ?, ?, ?, ?, 'unindexed', ?, ?, ?)
                   ON CONFLICT(media_key) DO UPDATE SET
                     media_type=excluded.media_type, source_id=excluded.source_id,
                     file_id=excluded.file_id, title=excluded.title,
                     content_key=excluded.content_key, status='unindexed',
                     frame_index=excluded.frame_index, frame_time=excluded.frame_time,
                     parent_folder=excluded.parent_folder, last_error=NULL""",
                (media_key, media_type, str(source_id), file_id, title or "", content_key,
                 frame_index, frame_time, parent_folder or ""),
            )
            return True

    def media_source_unchanged(self, media_type: str, source_id: str, file_id: Optional[str], content_key: str) -> bool:
        with self.connection() as con:
            row = con.execute(
                """SELECT count(*) AS total, sum(status IN ('indexed','clustered','approved','no_faces','ignored')) AS ready
                   FROM media WHERE media_type=? AND source_id=? AND coalesce(file_id, '')=coalesce(?, '')
                   AND content_key=?""",
                (media_type, str(source_id), file_id, content_key),
            ).fetchone()
            return bool(row and row["total"] and row["total"] == row["ready"])

    def faces_page(self, filter_type: str = "all", q: str = "", limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        where = ["f.active=1"]
        params: List[Any] = []
        if filter_type == "clustered":
            where.append("f.cluster_id IS NOT NULL")
        elif filter_type == "unclustered":
            where.append("f.cluster_id IS NULL")
        elif filter_type == "named":
            where.append("(f.performer_id IS NOT NULL OR c.performer_id IS NOT NULL)")
        elif filter_type == "unknown":
            where.append("f.performer_id IS NULL AND c.performer_id IS NULL")
        if q:
            where.append("(coalesce(i.title, '') LIKE ? OR coalesce(c.name, '') LIKE ? OR coalesce(f.performer_id, '') LIKE ?)")
            term = f"%{q}%"
            params.extend([term, term, term])
        clause = " AND ".join(where)
        with self.connection() as con:
            total = con.execute(
                f"""SELECT count(*) FROM faces f
                    LEFT JOIN images i ON i.id=f.image_id
                    LEFT JOIN clusters c ON c.id=f.cluster_id
                    WHERE {clause}""", params,
            ).fetchone()[0]
            rows = con.execute(
                f"""SELECT f.id, f.image_id, f.crop, f.score, f.gender, f.age,
                          f.cluster_id, f.performer_id, f.matched_by,
                          coalesce(i.title, '') AS title, c.name AS cluster_name,
                          c.performer_id AS cluster_performer_id,
                          m.media_type, m.source_id, m.frame_index, m.frame_time
                   FROM faces f
                   LEFT JOIN images i ON i.id=f.image_id
                   LEFT JOIN clusters c ON c.id=f.cluster_id
                   LEFT JOIN media m ON m.media_key=f.image_id
                   WHERE {clause}
                   ORDER BY f.id DESC LIMIT ? OFFSET ?""", params + [min(max(limit, 1), 500), max(offset, 0)],
            ).fetchall()
            return {"total": total, "items": [dict(row) for row in rows]}

    def media_page(self, status: str = "all", media_type: str = "all", limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        where, params = [], []
        if status != "all":
            where.append("status=?")
            params.append(status)
        if media_type != "all":
            where.append("media_type=?")
            params.append(media_type)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self.connection() as con:
            total = con.execute("SELECT count(*) FROM media" + clause, params).fetchone()[0]
            rows = con.execute(
                "SELECT * FROM media" + clause + " ORDER BY media_type, source_id, frame_index LIMIT ? OFFSET ?",
                params + [min(max(limit, 1), 500), max(offset, 0)],
            ).fetchall()
            return {"total": total, "items": [dict(row) for row in rows]}

    def progress(self) -> Dict[str, Any]:
        with self.connection() as con:
            counts = {row["status"]: row["count"] for row in con.execute("SELECT status, count(*) AS count FROM media GROUP BY status")}
            job_counts = {f"{row['kind']}_{row['state']}": row["count"] for row in con.execute("SELECT kind, state, count(*) AS count FROM jobs GROUP BY kind, state")}
            total = sum(counts.values())
            completed = sum(counts.get(state, 0) for state in ("indexed", "clustered", "approved", "no_faces", "ignored"))
            return {
                "media_total": total,
                "media_completed": completed,
                "percent": round(completed * 100 / total, 2) if total else 0,
                "statuses": counts,
                "jobs": job_counts,
            }

    def save_detection(
        self,
        image_id: str,
        content_key: str,
        detector_key: str,
        title: str,
        detections: List[Dict[str, Any]],
        media_type: str = "image",
        source_id: Optional[str] = None,
        file_id: Optional[str] = None,
        frame_index: Optional[int] = None,
        frame_time: Optional[float] = None,
        parent_folder: str = "",
    ) -> List[int]:
        now = time.time()
        inserted_ids = []
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT revision FROM images WHERE id=?", (str(image_id),)).fetchone()
            revision = old["revision"] + 1 if old else 1
            con.execute(
                """INSERT INTO images VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                   content_key=excluded.content_key, detector_key=excluded.detector_key,
                   revision=excluded.revision, title=excluded.title, processed_at=excluded.processed_at""",
                (str(image_id), content_key, detector_key, revision, title or "", now),
            )
            # Deactivate previous faces for this media item and update the inventory.
            con.execute("UPDATE faces SET active=0 WHERE image_id=?", (str(image_id),))
            media_status = "indexed" if detections else "no_faces"
            con.execute(
                """INSERT INTO media
                   (media_key, media_type, source_id, file_id, title, content_key, detector_key,
                    status, frame_index, frame_time, parent_folder, face_count, last_error, indexed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                   ON CONFLICT(media_key) DO UPDATE SET
                     media_type=excluded.media_type, source_id=excluded.source_id,
                     file_id=excluded.file_id, title=excluded.title,
                     content_key=excluded.content_key, detector_key=excluded.detector_key,
                     status=excluded.status, frame_index=excluded.frame_index,
                     frame_time=excluded.frame_time, parent_folder=excluded.parent_folder,
                     face_count=excluded.face_count, last_error=NULL, indexed_at=excluded.indexed_at""",
                (str(image_id), media_type, str(source_id or image_id), file_id, title or "",
                 content_key, detector_key, media_status, frame_index, frame_time,
                 parent_folder or "", len(detections), now),
            )

            for d in detections:
                emb = d.get("embedding")
                emb_bytes = emb.tobytes() if isinstance(emb, np.ndarray) else None
                cur = con.execute(
                    """INSERT INTO faces (image_id, revision, active, box, score, crop, embedding,
                                          gender, age, created_at)
                       VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(image_id),
                        revision,
                        json.dumps(d["box"]),
                        d.get("score"),
                        d["crop"],
                        emb_bytes,
                        d.get("gender", -1),
                        d.get("age", -1),
                        now,
                    ),
                )
                inserted_ids.append(cur.lastrowid)
        return inserted_ids

    def get_all_faces_with_embeddings(self, unclustered_only: bool = False) -> List[Dict[str, Any]]:
        query = """SELECT id, image_id, crop, embedding, cluster_id, performer_id
                   FROM faces WHERE active=1 AND embedding IS NOT NULL"""
        if unclustered_only:
            query += " AND cluster_id IS NULL AND performer_id IS NULL"
        with self.connection() as con:
            rows = con.execute(query).fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "image_id": r["image_id"],
                    "crop": r["crop"],
                    "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
                    "cluster_id": r["cluster_id"],
                    "performer_id": r["performer_id"],
                })
            return results

    def reset_unlinked_clusters(self) -> int:
        """Clear previous automatic clusters while preserving named/linking decisions."""
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("UPDATE faces SET cluster_id=NULL WHERE active=1 AND performer_id IS NULL")
            cur = con.execute("DELETE FROM clusters WHERE performer_id IS NULL")
            return cur.rowcount

    def cluster_embedding(self, cluster_id: int) -> Optional[np.ndarray]:
        with self.connection() as con:
            rows = con.execute(
                "SELECT embedding FROM faces WHERE cluster_id=? AND active=1 AND embedding IS NOT NULL",
                (cluster_id,),
            ).fetchall()
        if not rows:
            return None
        values = np.stack([np.frombuffer(row[0], dtype=np.float32) for row in rows])
        centroid = np.mean(values, axis=0)
        norm = np.linalg.norm(centroid)
        return (centroid / norm).astype(np.float32) if norm else None

    def forget_image(self, image_id: str) -> bool:
        with self.connection() as con:
            cur = con.execute("DELETE FROM images WHERE id=?", (str(image_id),))
            return cur.rowcount > 0

    def apply_clustering(self, assignments: Dict[int, int], cluster_stats: Dict[int, Dict[str, Any]]) -> int:
        now = time.time()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            created_count = 0

            # Map from DBSCAN label (0, 1, 2...) to database cluster_id
            label_to_db_id = {}
            for label, stats in cluster_stats.items():
                cur = con.execute(
                    """INSERT INTO clusters (name, exemplar_crop, face_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"Person #{stats['exemplar_face_id']}", stats["exemplar_crop"], stats["face_count"], now, now),
                )
                label_to_db_id[label] = cur.lastrowid
                created_count += 1

            for fid, label in assignments.items():
                db_id = label_to_db_id.get(label)
                if db_id is not None:
                    con.execute("UPDATE faces SET cluster_id=? WHERE id=?", (db_id, fid))
                elif label == -1:
                    con.execute("UPDATE faces SET cluster_id=NULL WHERE id=?", (fid,))

            con.execute("""UPDATE media SET status='clustered'
                WHERE media_key IN (SELECT DISTINCT image_id FROM faces WHERE cluster_id IS NOT NULL AND active=1)
                AND status='indexed'""")

        return created_count

    def apply_folder_hints(self, max_clusters: int = 5, max_distance: float = 0.45) -> int:
        """Use same-folder image clusters only as a fallback for video faces.

        DBSCAN/known-performer matches remain authoritative. A video face is
        moved only when it is currently unnamed/unmatched, the folder has at
        most ``max_clusters`` image identities, and its embedding agrees with
        one of those image-cluster centroids.
        """
        with self.connection() as con:
            rows = con.execute(
                """SELECT f.id, f.cluster_id, f.embedding, m.parent_folder
                   FROM faces f JOIN media m ON m.media_key=f.image_id
                   WHERE f.active=1 AND f.embedding IS NOT NULL
                     AND m.media_type='video' AND f.performer_id IS NULL"""
            ).fetchall()
            moved = 0
            for face in rows:
                folder = str(face["parent_folder"] or "")
                if not folder:
                    continue
                candidates = con.execute(
                    """SELECT f.cluster_id, f.embedding
                       FROM faces f JOIN media m ON m.media_key=f.image_id
                       WHERE f.active=1 AND f.embedding IS NOT NULL
                         AND m.media_type='image' AND m.parent_folder=?
                         AND f.cluster_id IS NOT NULL""",
                    (folder,),
                ).fetchall()
                cluster_ids = sorted({int(row["cluster_id"]) for row in candidates})
                if not cluster_ids or len(cluster_ids) > max_clusters:
                    continue
                target_vectors = {}
                for row in candidates:
                    target_vectors.setdefault(int(row["cluster_id"]), []).append(
                        np.frombuffer(row["embedding"], dtype=np.float32))
                emb = np.frombuffer(face["embedding"], dtype=np.float32)
                best_id, best_distance = None, 2.0
                for cluster_id, vectors in target_vectors.items():
                    centroid = np.mean(vectors, axis=0)
                    norm = np.linalg.norm(centroid)
                    if not norm:
                        continue
                    distance = 1.0 - float(np.dot(emb, centroid / norm))
                    if distance < best_distance:
                        best_id, best_distance = cluster_id, distance
                if (face["cluster_id"] is None and best_id is not None
                        and best_distance <= max_distance):
                    con.execute("UPDATE faces SET cluster_id=?, matched_by='folder_fallback' WHERE id=?",
                                (best_id, face["id"]))
                    moved += 1

            if moved:
                con.execute("""UPDATE clusters SET face_count=(
                    SELECT count(*) FROM faces WHERE faces.cluster_id=clusters.id AND faces.active=1
                ), updated_at=?""", (time.time(),))
                con.execute("""DELETE FROM clusters
                             WHERE performer_id IS NULL
                               AND NOT EXISTS (SELECT 1 FROM faces WHERE faces.cluster_id=clusters.id AND faces.active=1)""")
                con.execute("""UPDATE media SET status='clustered'
                             WHERE media_type='video' AND media_key IN
                               (SELECT DISTINCT image_id FROM faces WHERE cluster_id IS NOT NULL AND active=1)""")
            return moved

    def create_folder_suggestions(self, max_clusters: int = 5, limit_per_media: int = 5) -> int:
        """Create reviewable same-folder suggestions for unresolved media.

        Existing face clusters are authoritative and excluded. Suggestions are
        deliberately media-level so a no-face image can still be reviewed and
        associated with a person without fabricating a face embedding.
        """
        with self.connection() as con:
            candidates = con.execute(
                """SELECT m.media_key, m.media_type, m.source_id, m.file_id, m.parent_folder
                   FROM media m
                   WHERE m.parent_folder <> ''
                     AND m.status IN ('indexed', 'no_faces', 'clustered')
                     AND NOT EXISTS (SELECT 1 FROM media_assignments a WHERE a.media_key=m.media_key)
                     AND NOT EXISTS (
                       SELECT 1 FROM faces f
                       WHERE f.image_id=m.media_key AND f.active=1 AND f.cluster_id IS NOT NULL
                     )
                     AND (m.media_type='image' OR m.frame_index=0)"""
            ).fetchall()
            created = 0
            now = time.time()
            for media in candidates:
                folder = str(media["parent_folder"])
                clusters = con.execute(
                    """SELECT DISTINCT f.cluster_id, c.name, c.performer_id
                       FROM faces f JOIN media im ON im.media_key=f.image_id
                       JOIN clusters c ON c.id=f.cluster_id
                       WHERE f.active=1 AND f.cluster_id IS NOT NULL
                         AND im.media_type='image' AND im.parent_folder=?
                       ORDER BY c.id""",
                    (folder,),
                ).fetchall()
                if not clusters or len(clusters) > max_clusters:
                    continue
                for candidate in clusters[:limit_per_media]:
                    reason = {
                        "same_folder": folder,
                        "folder_identities": len(clusters),
                        "face_match": False,
                        "performer_linked": bool(candidate["performer_id"]),
                    }
                    score = 0.80 if candidate["performer_id"] else 0.68
                    cur = con.execute(
                        """INSERT INTO suggestions
                           (media_key, cluster_id, score, reason, status, created_at, updated_at)
                           VALUES (?, ?, ?, ?, 'pending', ?, ?)
                           ON CONFLICT(media_key, cluster_id) DO UPDATE SET
                             score=excluded.score, reason=excluded.reason,
                             updated_at=excluded.updated_at
                           WHERE suggestions.status='pending'""",
                        (media["media_key"], candidate["cluster_id"], score,
                         json.dumps(reason, sort_keys=True), now, now),
                    )
                    created += int(cur.rowcount > 0)
            return created

    def suggestions_page(self, status: str = "pending", limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        with self.connection() as con:
            where = "" if status == "all" else "WHERE s.status=?"
            params = [] if status == "all" else [status]
            total = con.execute("SELECT count(*) FROM suggestions s " + where, params).fetchone()[0]
            rows = con.execute(
                """SELECT s.id, s.media_key, s.cluster_id, s.score, s.reason, s.status,
                          s.created_at, s.updated_at, m.media_type, m.source_id, m.file_id,
                          m.title, m.parent_folder, c.name AS cluster_name,
                          c.performer_id, c.exemplar_crop
                   FROM suggestions s JOIN media m ON m.media_key=s.media_key
                   JOIN clusters c ON c.id=s.cluster_id """ + where +
                " ORDER BY s.score DESC, s.id DESC LIMIT ? OFFSET ?",
                params + [min(max(limit, 1), 200), max(offset, 0)],
            ).fetchall()
            return {"total": total, "items": [dict(row) for row in rows]}

    def suggestion(self, suggestion_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as con:
            row = con.execute(
                """SELECT s.*, m.media_type, m.source_id, m.file_id, m.title,
                          c.name AS cluster_name, c.performer_id
                   FROM suggestions s JOIN media m ON m.media_key=s.media_key
                   JOIN clusters c ON c.id=s.cluster_id WHERE s.id=?""",
                (suggestion_id,),
            ).fetchone()
            return dict(row) if row else None

    def decide_suggestion(self, suggestion_id: int, decision: str) -> Optional[Dict[str, Any]]:
        if decision not in ("accepted", "rejected", "ignored"):
            raise ValueError("Unsupported suggestion decision")
        now = time.time()
        with self.connection() as con:
            row = con.execute(
                """SELECT s.*, m.media_type, m.source_id, m.file_id, m.title,
                          c.performer_id
                   FROM suggestions s JOIN media m ON m.media_key=s.media_key
                   JOIN clusters c ON c.id=s.cluster_id WHERE s.id=?""",
                (suggestion_id,),
            ).fetchone()
            if not row:
                return None
            con.execute("UPDATE suggestions SET status=?, updated_at=? WHERE id=?",
                        (decision, now, suggestion_id))
            if decision == "accepted":
                con.execute(
                    """INSERT INTO media_assignments
                       (media_key, cluster_id, performer_id, matched_by, created_at, updated_at)
                       VALUES (?, ?, ?, 'suggestion', ?, ?)
                       ON CONFLICT(media_key) DO UPDATE SET
                         cluster_id=excluded.cluster_id, performer_id=excluded.performer_id,
                         matched_by=excluded.matched_by, updated_at=excluded.updated_at""",
                    (row["media_key"], row["cluster_id"], row["performer_id"], now, now),
                )
                con.execute("UPDATE media SET status='approved' WHERE media_key=?",
                            (row["media_key"],))
            return dict(row)

    def get_clusters(self, q: str = "", filter_type: str = "all") -> List[Dict[str, Any]]:
        query = """SELECT c.id, c.name, c.performer_id, c.exemplar_crop, c.face_count, c.hidden,
                          c.updated_at, COUNT(DISTINCT f.image_id) as image_count
                   FROM clusters c
                   LEFT JOIN faces f ON f.cluster_id = c.id AND f.active = 1
                   WHERE 1=1"""
        params = []
        if filter_type != "hidden":
            query += " AND c.hidden = 0"
        else:
            query += " AND c.hidden = 1"

        if filter_type == "named":
            query += " AND c.performer_id IS NOT NULL"
        elif filter_type == "unnamed":
            query += " AND c.performer_id IS NULL"

        if q:
            query += " AND c.name LIKE ?"
            params.append(f"%{q}%")

        query += " GROUP BY c.id ORDER BY c.face_count DESC, c.id ASC"

        with self.connection() as con:
            rows = con.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def cluster_media_sources(self, cluster_id: int) -> Dict[str, Set[str]]:
        with self.connection() as con:
            rows = con.execute(
                """SELECT DISTINCT m.media_type, m.media_key, m.source_id
                   FROM faces f JOIN media m ON m.media_key=f.image_id
                   WHERE f.cluster_id=? AND f.active=1""",
                (cluster_id,),
            ).fetchall()
        return {
            "images": {str(r[1]) for r in rows if r[0] == "image"},
            "scenes": {str(r[2]) for r in rows if r[0] == "video"},
        }

    def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM clusters WHERE id=?", (cluster_id,)).fetchone()
            if not row:
                return None
            cluster = dict(row)
            faces_rows = con.execute(
                """SELECT f.id, f.image_id, f.crop, f.score, f.gender, f.age,
                          m.media_type, m.source_id, m.frame_index, m.frame_time
                   FROM faces f LEFT JOIN media m ON m.media_key=f.image_id
                   WHERE f.cluster_id=? AND f.active=1 ORDER BY f.id DESC""",
                (cluster_id,),
            ).fetchall()
            cluster["faces"] = [dict(f) for f in faces_rows]
            cluster["image_count"] = len({f["image_id"] for f in faces_rows})
            return cluster

    def update_cluster_name(self, cluster_id: int, name: str) -> bool:
        with self.connection() as con:
            cur = con.execute(
                "UPDATE clusters SET name=?, updated_at=? WHERE id=?",
                (name.strip(), time.time(), cluster_id),
            )
            return cur.rowcount > 0

    def link_cluster_to_performer(self, cluster_id: int, performer_id: str, performer_name: str) -> List[str]:
        now = time.time()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                "UPDATE clusters SET performer_id=?, name=?, updated_at=? WHERE id=?",
                (str(performer_id), performer_name, now, cluster_id),
            )
            con.execute(
                "UPDATE faces SET performer_id=?, matched_by='cluster' WHERE cluster_id=? AND active=1",
                (str(performer_id), cluster_id),
            )
            con.execute(
                "UPDATE media_assignments SET performer_id=?, updated_at=? WHERE cluster_id=?",
                (str(performer_id), now, cluster_id),
            )
            con.execute(
                """UPDATE media SET status='approved' WHERE media_key IN
                   (SELECT DISTINCT image_id FROM faces WHERE cluster_id=? AND active=1)
                   OR media_key IN (SELECT media_key FROM media_assignments WHERE cluster_id=?)""",
                (cluster_id, cluster_id),
            )
            rows = con.execute(
                """SELECT DISTINCT image_id FROM faces WHERE cluster_id=? AND active=1
                   UNION SELECT a.media_key FROM media_assignments a JOIN media m ON m.media_key=a.media_key
                   WHERE a.cluster_id=? AND m.media_type='image'""",
                (cluster_id, cluster_id),
            ).fetchall()
            return [str(r[0]) for r in rows]

    def merge_clusters(self, source_id: int, target_id: int) -> bool:
        now = time.time()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            # Move faces
            con.execute(
                "UPDATE faces SET cluster_id=? WHERE cluster_id=? AND active=1",
                (target_id, source_id),
            )
            # Recalculate target count
            count = con.execute(
                "SELECT count(*) FROM faces WHERE cluster_id=? AND active=1",
                (target_id,),
            ).fetchone()[0]
            con.execute("UPDATE clusters SET face_count=?, updated_at=? WHERE id=?", (count, now, target_id))
            # Delete source
            con.execute("DELETE FROM clusters WHERE id=?", (source_id,))
            return True

    def hide_cluster(self, cluster_id: int, hidden: bool = True) -> bool:
        with self.connection() as con:
            cur = con.execute(
                "UPDATE clusters SET hidden=?, updated_at=? WHERE id=?",
                (1 if hidden else 0, time.time(), cluster_id),
            )
            return cur.rowcount > 0

    def get_known_performers(self) -> List[Dict[str, Any]]:
        with self.connection() as con:
            rows = con.execute("SELECT * FROM known_performers").fetchall()
            results = []
            for r in rows:
                results.append({
                    "performer_id": r["performer_id"],
                    "name": r["name"],
                    "image_url": r["image_url"],
                    "embedding": np.frombuffer(r["embedding"], dtype=np.float32),
                })
            return results

    def save_known_performer(self, performer_id: str, name: str, image_url: str, embedding: np.ndarray):
        with self.connection() as con:
            con.execute(
                """INSERT INTO known_performers (performer_id, name, image_url, embedding, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(performer_id) DO UPDATE SET
                   name=excluded.name, image_url=excluded.image_url,
                   embedding=excluded.embedding, updated_at=excluded.updated_at""",
                (str(performer_id), name, image_url or "", embedding.tobytes(), time.time()),
            )

    # Job Queue
    def enqueue(self, kind: str, payload: dict, dedupe_key: str) -> Tuple[str, bool]:
        now = time.time()
        job_id = uuid.uuid4().hex
        with self.connection() as con:
            try:
                con.execute(
                    """INSERT INTO jobs (id, kind, payload, dedupe_key, state, next_attempt, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)""",
                    (job_id, kind, json.dumps(payload), dedupe_key, now, now, now),
                )
                return job_id, True
            except sqlite3.IntegrityError:
                row = con.execute(
                    "SELECT id FROM jobs WHERE dedupe_key=? AND state IN ('queued','running','retry')",
                    (dedupe_key,),
                ).fetchone()
                return (row[0], False) if row else (job_id, False)

    def recover_running_jobs(self) -> int:
        with self.connection() as con:
            cur = con.execute(
                """UPDATE jobs SET state='queued', stage='queued', detail='recovered after service restart',
                   next_attempt=?, updated_at=? WHERE state='running'""",
                (time.time(), time.time()),
            )
            return cur.rowcount

    def claim(self) -> Optional[dict]:
        now = time.time()
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                """SELECT * FROM jobs
                   WHERE state IN ('queued', 'retry') AND next_attempt <= ?
                   ORDER BY created_at ASC LIMIT 1""",
                (now,),
            ).fetchone()
            if not row:
                return None
            con.execute(
                "UPDATE jobs SET state='running', stage='starting', detail=NULL, attempts=attempts+1, updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            d = dict(row)
            d["payload"] = json.loads(d["payload"])
            return d

    def set_job_progress(self, job_id: str, stage: str, detail: Optional[str] = None):
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET stage=?, detail=?, updated_at=? WHERE id=? AND state='running'",
                (stage, detail, time.time(), job_id),
            )

    def finish(self, job_id: str, result: Optional[dict]):
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET state='done', stage='completed', detail=NULL, result=?, updated_at=? WHERE id=?",
                (json.dumps(result) if result else None, time.time(), job_id),
            )

    def fail(self, job_id: str, error: str, retry_at: Optional[float] = None):
        state = "retry" if retry_at else "failed"
        next_attempt = retry_at or 0.0
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET state=?, stage=?, detail=?, error=?, next_attempt=?, updated_at=? WHERE id=?",
                (state, "retrying" if retry_at else "failed", error, error, next_attempt, time.time(), job_id),
            )

    def has_pending_media_jobs(self) -> bool:
        with self.connection() as con:
            row = con.execute(
                "SELECT 1 FROM jobs WHERE kind IN ('image','video') AND state IN ('queued','running','retry') LIMIT 1"
            ).fetchone()
            return row is not None

    def jobs(self, limit: int = 15) -> List[dict]:
        with self.connection() as con:
            rows = con.execute(
                """SELECT * FROM jobs
                   ORDER BY CASE state WHEN 'running' THEN 0 WHEN 'retry' THEN 1
                                      WHEN 'queued' THEN 2 WHEN 'failed' THEN 3 ELSE 4 END,
                            updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            res = []
            for r in rows:
                d = dict(r)
                d["payload"] = json.loads(d["payload"])
                if d.get("result"):
                    d["result"] = json.loads(d["result"])
                res.append(d)
            return res

    def stats(self) -> dict:
        with self.connection() as con:
            images_count = con.execute("SELECT count(*) FROM media WHERE media_type='image'").fetchone()[0]
            faces_count = con.execute("SELECT count(*) FROM faces WHERE active=1").fetchone()[0]
            clusters_count = con.execute("SELECT count(*) FROM clusters WHERE hidden=0").fetchone()[0]
            named_count = con.execute("SELECT count(*) FROM clusters WHERE performer_id IS NOT NULL AND hidden=0").fetchone()[0]
            jobs_queued = con.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running','retry')").fetchone()[0]
            media_count = con.execute("SELECT count(*) FROM media").fetchone()[0]
            media_done = con.execute("SELECT count(*) FROM media WHERE status IN ('indexed','clustered','approved','no_faces','ignored')").fetchone()[0]
            return {
                "images": images_count,
                "faces": faces_count,
                "clusters": clusters_count,
                "named_people": named_count,
                "active_jobs": jobs_queued,
                "media_total": media_count,
                "media_completed": media_done,
                "media_percent": round(media_done * 100 / media_count, 2) if media_count else 0,
            }
