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
              error TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS active_job_key ON jobs(dedupe_key)
              WHERE state IN ('queued', 'running', 'retry');
            """)

            # Ensure any missing columns exist
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

    def save_detection(
        self,
        image_id: str,
        content_key: str,
        detector_key: str,
        title: str,
        detections: List[Dict[str, Any]],
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
            # Deactivate previous faces for this image
            con.execute("UPDATE faces SET active=0 WHERE image_id=?", (str(image_id),))

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
                    # Noise/unassigned
                    con.execute("UPDATE faces SET cluster_id=NULL WHERE id=?", (fid,))

        return created_count

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

    def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM clusters WHERE id=?", (cluster_id,)).fetchone()
            if not row:
                return None
            cluster = dict(row)
            faces_rows = con.execute(
                """SELECT id, image_id, crop, score, gender, age
                   FROM faces WHERE cluster_id=? AND active=1 ORDER BY id DESC""",
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
            rows = con.execute(
                "SELECT DISTINCT image_id FROM faces WHERE cluster_id=? AND active=1",
                (cluster_id,),
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
                "UPDATE jobs SET state='running', attempts=attempts+1, updated_at=? WHERE id=?",
                (now, row["id"]),
            )
            d = dict(row)
            d["payload"] = json.loads(d["payload"])
            return d

    def finish(self, job_id: str, result: Optional[dict]):
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET state='done', result=?, updated_at=? WHERE id=?",
                (json.dumps(result) if result else None, time.time(), job_id),
            )

    def fail(self, job_id: str, error: str, retry_at: Optional[float] = None):
        state = "retry" if retry_at else "failed"
        next_attempt = retry_at or 0.0
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET state=?, error=?, next_attempt=?, updated_at=? WHERE id=?",
                (state, error, next_attempt, time.time(), job_id),
            )

    def jobs(self, limit: int = 15) -> List[dict]:
        with self.connection() as con:
            rows = con.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
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
            images_count = con.execute("SELECT count(*) FROM images").fetchone()[0]
            faces_count = con.execute("SELECT count(*) FROM faces WHERE active=1").fetchone()[0]
            clusters_count = con.execute("SELECT count(*) FROM clusters WHERE hidden=0").fetchone()[0]
            named_count = con.execute("SELECT count(*) FROM clusters WHERE performer_id IS NOT NULL AND hidden=0").fetchone()[0]
            jobs_queued = con.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running','retry')").fetchone()[0]
            return {
                "images": images_count,
                "faces": faces_count,
                "clusters": clusters_count,
                "named_people": named_count,
                "active_jobs": jobs_queued,
            }
