"""SQLite persistence + sqlite-vec hybrid search.

A single file ``data/anilist_local.db`` holds the relational model
(``media`` / ``user_list`` / ``media_recommendations`` / ``candidates``) and a
``vec_media`` virtual table for cosine similarity over CPU embeddings.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Sequence

import sqlite_vec

from .scoring import Candidate, Work

CREATE_MEDIA = """
CREATE TABLE media (
  id INTEGER PRIMARY KEY,
  type TEXT NOT NULL,
  title_romaji TEXT,
  title_english TEXT,
  format TEXT,
  status TEXT,
  episodes INTEGER,
  chapters INTEGER,
  volumes INTEGER,
  average_score INTEGER,
  popularity INTEGER,
  genres TEXT,
  tags TEXT,
  studios TEXT,
  description TEXT
)
"""

CREATE_USER_LIST = """
CREATE TABLE user_list (
  media_id INTEGER PRIMARY KEY,
  status TEXT,
  score REAL,
  progress INTEGER,
  repeat INTEGER,
  updated_at INTEGER,
  progress_ratio REAL,
  z_score REAL,
  affinity_weight REAL,
  is_favourite INTEGER,
  FOREIGN KEY(media_id) REFERENCES media(id)
)
"""

CREATE_RECOMMENDATIONS = """
CREATE TABLE media_recommendations (
  source_media_id INTEGER,
  target_media_id INTEGER,
  rating INTEGER,
  PRIMARY KEY (source_media_id, target_media_id)
)
"""

CREATE_CANDIDATES = """
CREATE TABLE candidates (
  media_id INTEGER PRIMARY KEY,
  graph_score REAL,
  planning_bonus REAL,
  source_pool TEXT,
  positive_sources TEXT,
  negative_warnings TEXT,
  FOREIGN KEY(media_id) REFERENCES media(id)
)
"""

CREATE_META = """
CREATE TABLE meta (
  key TEXT PRIMARY KEY,
  value TEXT
)
"""


def connect(path: Any) -> sqlite3.Connection:
    """Open the DB and load the sqlite-vec extension."""
    db_path = str(path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def reset_schema(db: sqlite3.Connection, embedding_dim: int) -> None:
    """Drop and recreate every table, sizing the vec0 table to the model."""
    db.execute("PRAGMA foreign_keys=OFF")
    for table in ("candidates", "user_list", "media_recommendations", "media", "meta", "vec_media"):
        db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(CREATE_MEDIA)
    db.execute(CREATE_USER_LIST)
    db.execute(CREATE_RECOMMENDATIONS)
    db.execute(CREATE_CANDIDATES)
    db.execute(CREATE_META)
    db.execute(
        "CREATE VIRTUAL TABLE vec_media USING vec0("
        "media_id INTEGER PRIMARY KEY, "
        f"embedding FLOAT[{embedding_dim}] distance_metric=cosine)"
    )
    db.execute("PRAGMA foreign_keys=ON")


def persist(
    db: sqlite3.Connection,
    works: list[Work],
    candidates: list[Candidate],
    edges: list[tuple[int, int, int]],
    username: str,
    synced_at: str,
    embedding_model: str,
) -> None:
    """Insert everything into the freshly-reset schema."""
    media_rows: dict[int, dict[str, Any]] = {}
    for work in works:
        media_rows[work.media_id] = work.media
    for candidate in candidates:
        media_rows[candidate.media_id] = candidate.media

    for media_id, media in media_rows.items():
        db.execute(
            "INSERT OR REPLACE INTO media (id, type, title_romaji, title_english, format, "
            "status, episodes, chapters, volumes, average_score, popularity, genres, tags, "
            "studios, description) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                media_id,
                media.get("type"),
                media.get("title_romaji"),
                media.get("title_english"),
                media.get("format"),
                media.get("status"),
                media.get("episodes"),
                media.get("chapters"),
                media.get("volumes"),
                media.get("average_score"),
                media.get("popularity"),
                json.dumps(media.get("genres") or []),
                json.dumps(media.get("tags") or []),
                json.dumps(media.get("studios") or []),
                media.get("description") or "",
            ),
        )

    for work in works:
        db.execute(
            "INSERT INTO user_list (media_id, status, score, progress, repeat, updated_at, "
            "progress_ratio, z_score, affinity_weight, is_favourite) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                work.media_id,
                work.status,
                work.score,
                work.progress,
                work.repeat,
                work.updated_at,
                work.progress_ratio,
                work.z_score,
                work.weight,
                int(work.is_favourite),
            ),
        )

    for source_id, target_id, rating in edges:
        db.execute(
            "INSERT OR REPLACE INTO media_recommendations (source_media_id, target_media_id, rating) "
            "VALUES (?,?,?)",
            (source_id, target_id, rating),
        )

    for candidate in candidates:
        db.execute(
            "INSERT INTO candidates (media_id, graph_score, planning_bonus, source_pool, "
            "positive_sources, negative_warnings) VALUES (?,?,?,?,?,?)",
            (
                candidate.media_id,
                candidate.graph_score,
                candidate.planning_bonus,
                candidate.source_pool,
                json.dumps(candidate.positive_sources),
                json.dumps(candidate.negative_warnings),
            ),
        )

    db.execute("INSERT INTO meta (key, value) VALUES ('username', ?)", (username,))
    db.execute("INSERT INTO meta (key, value) VALUES ('synced_at', ?)", (synced_at,))
    db.execute("INSERT INTO meta (key, value) VALUES ('embedding_model', ?)", (embedding_model,))
    db.execute("INSERT INTO meta (key, value) VALUES ('num_works', ?)", (str(len(works)),))
    db.execute("INSERT INTO meta (key, value) VALUES ('num_candidates', ?)", (str(len(candidates)),))
    db.commit()


_CANDIDATE_SELECT = """
SELECT
  c.media_id,
  c.graph_score,
  c.planning_bonus,
  c.source_pool,
  c.positive_sources,
  c.negative_warnings,
  m.type,
  m.title_romaji,
  m.title_english,
  m.format,
  m.status AS release_status,
  m.episodes,
  m.chapters,
  m.volumes,
  m.average_score,
  m.popularity,
  m.genres,
  m.tags,
  m.description
FROM candidates c
JOIN media m ON m.id = c.media_id
"""


def _row_to_candidate(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["genres"] = json.loads(data.get("genres") or "[]")
    data["tags"] = json.loads(data.get("tags") or "[]")
    data["positive_sources"] = json.loads(data.get("positive_sources") or "[]")
    data["negative_warnings"] = json.loads(data.get("negative_warnings") or "[]")
    return data


def search_candidates(
    db: sqlite3.Connection,
    *,
    media_type: str | None = None,
    min_episodes: int | None = None,
    max_episodes: int | None = None,
    min_chapters: int | None = None,
    max_chapters: int | None = None,
    source_pool: str = "BOTH",
    required: Sequence[str] | None = None,
    excluded: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Hard SQL filters, then soft genre/tag matching in Python.

    Length filters apply to the relevant media type only (episodes for ANIME,
    chapters for MANGA) and never drop media with an unknown (NULL) length.
    """
    required_terms = [t.strip().lower() for t in (required or []) if t.strip()]
    excluded_terms = [t.strip().lower() for t in (excluded or []) if t.strip()]

    sql = _CANDIDATE_SELECT + " WHERE 1=1"
    params: list[Any] = []

    if media_type:
        sql += " AND m.type = ?"
        params.append(media_type)

    if source_pool == "PLANNING_ONLY":
        sql += " AND c.source_pool IN ('PLANNING','BOTH')"
    elif source_pool == "DISCOVERY_ONLY":
        sql += " AND c.source_pool IN ('DISCOVERY','BOTH')"

    if max_episodes is not None:
        sql += " AND (m.type != 'ANIME' OR m.episodes IS NULL OR m.episodes <= ?)"
        params.append(max_episodes)
    if min_episodes is not None:
        sql += " AND (m.type != 'ANIME' OR m.episodes IS NULL OR m.episodes >= ?)"
        params.append(min_episodes)
    if max_chapters is not None:
        sql += " AND (m.type != 'MANGA' OR m.chapters IS NULL OR m.chapters <= ?)"
        params.append(max_chapters)
    if min_chapters is not None:
        sql += " AND (m.type != 'MANGA' OR m.chapters IS NULL OR m.chapters >= ?)"
        params.append(min_chapters)

    def _matches_any(terms: list[str], haystack: set[str]) -> bool:
        return any(term in item for term in terms for item in haystack)

    results: list[dict[str, Any]] = []
    for row in db.execute(sql, params).fetchall():
        candidate = _row_to_candidate(row)
        haystack = {g.lower() for g in candidate["genres"]} | {t.lower() for t in candidate["tags"]}
        if required_terms and not _matches_any(required_terms, haystack):
            continue
        if excluded_terms and _matches_any(excluded_terms, haystack):
            continue
        results.append(candidate)
    return results


def read_meta(db: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}


def count_candidates(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()
    return int(row["n"])


def top_by_graph_score(db: sqlite3.Connection, limit: int = 10) -> list[dict[str, Any]]:
    rows = db.execute(
        _CANDIDATE_SELECT + " ORDER BY c.graph_score DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_candidate(r) for r in rows]
