"""SQLite persistence + sqlite-vec hybrid search (V2 omni-media).

A single file ``data/anilist_local.db`` holds the relational model
(``media`` / ``user_list`` / ``media_recommendations`` / ``media_relations`` /
``candidates``) and a ``vec_media`` virtual table for cosine similarity.
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
  duration INTEGER,
  chapters INTEGER,
  volumes INTEGER,
  country_of_origin TEXT,
  average_score INTEGER,
  popularity INTEGER,
  genres TEXT,
  tags TEXT,
  studios TEXT,
  description TEXT,
  prequel_id INTEGER,
  parent_id INTEGER,
  is_root INTEGER DEFAULT 1
)
"""

CREATE_USER_LIST = """
CREATE TABLE user_list (
  media_id INTEGER PRIMARY KEY,
  status TEXT,
  score REAL,
  progress INTEGER,
  progress_volumes INTEGER,
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

CREATE_RELATIONS = """
CREATE TABLE media_relations (
  source_id INTEGER,
  target_id INTEGER,
  relation_type TEXT,
  target_type TEXT,
  target_format TEXT,
  target_country TEXT,
  target_status TEXT,
  target_title TEXT,
  target_episodes INTEGER,
  target_chapters INTEGER,
  target_volumes INTEGER,
  target_score INTEGER,
  PRIMARY KEY (source_id, target_id, relation_type)
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
    db.execute("PRAGMA foreign_keys=OFF")
    for table in ("candidates", "user_list", "media_recommendations", "media_relations", "media", "meta", "vec_media"):
        db.execute(f"DROP TABLE IF EXISTS {table}")
    db.execute(CREATE_MEDIA)
    db.execute(CREATE_USER_LIST)
    db.execute(CREATE_RECOMMENDATIONS)
    db.execute(CREATE_RELATIONS)
    db.execute(CREATE_CANDIDATES)
    db.execute(CREATE_META)
    db.execute(
        "CREATE VIRTUAL TABLE vec_media USING vec0("
        "media_id INTEGER PRIMARY KEY, "
        f"embedding FLOAT[{embedding_dim}] distance_metric=cosine)"
    )
    db.execute("PRAGMA foreign_keys=ON")


_MEDIA_COLUMNS = (
    "id, type, title_romaji, title_english, format, status, episodes, duration, chapters, "
    "volumes, country_of_origin, average_score, popularity, genres, tags, studios, "
    "description, prequel_id, parent_id, is_root"
)


def _media_values(media: dict[str, Any]) -> tuple[Any, ...]:
    return (
        media.get("id"),
        media.get("type"),
        media.get("title_romaji"),
        media.get("title_english"),
        media.get("format"),
        media.get("status"),
        media.get("episodes"),
        media.get("duration"),
        media.get("chapters"),
        media.get("volumes"),
        media.get("country_of_origin"),
        media.get("average_score"),
        media.get("popularity"),
        json.dumps(media.get("genres") or []),
        json.dumps(media.get("tags") or []),
        json.dumps(media.get("studios") or []),
        media.get("description") or "",
        media.get("prequel_id"),
        media.get("parent_id"),
        media.get("is_root") if media.get("is_root") is not None else 1,
    )


def persist(
    db: sqlite3.Connection,
    works: list[Work],
    candidates: list[Candidate],
    edges: list[tuple[int, int, int]],
    relation_rows: list[dict[str, Any]],
    username: str,
    synced_at: str,
    embedding_model: str,
) -> None:
    media_rows: dict[int, dict[str, Any]] = {}
    for work in works:
        media_rows[work.media_id] = work.media
    for candidate in candidates:
        media_rows[candidate.media_id] = candidate.media

    for media in media_rows.values():
        db.execute(
            f"INSERT OR REPLACE INTO media ({_MEDIA_COLUMNS}) "
            f"VALUES ({','.join('?' * 20)})",
            _media_values(media),
        )

    for work in works:
        db.execute(
            "INSERT INTO user_list (media_id, status, score, progress, progress_volumes, repeat, "
            "updated_at, progress_ratio, z_score, affinity_weight, is_favourite) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                work.media_id,
                work.status,
                work.score,
                work.progress,
                work.progress_volumes,
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

    for row in relation_rows:
        db.execute(
            "INSERT OR REPLACE INTO media_relations (source_id, target_id, relation_type, "
            "target_type, target_format, target_country, target_status, target_title, "
            "target_episodes, target_chapters, target_volumes, target_score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["source_id"],
                row["target_id"],
                row["relation_type"],
                row["target_type"],
                row["target_format"],
                row["target_country"],
                row["target_status"],
                row["target_title"],
                row["target_episodes"],
                row["target_chapters"],
                row["target_volumes"],
                row["target_score"],
            ),
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
    db.execute("INSERT INTO meta (key, value) VALUES ('num_relations', ?)", (str(len(relation_rows)),))
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
  m.duration,
  m.chapters,
  m.volumes,
  m.country_of_origin,
  m.average_score,
  m.popularity,
  m.genres,
  m.tags,
  m.description,
  m.prequel_id,
  m.parent_id,
  m.is_root
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


# --- Tool 1: search_new_recommendations -------------------------------------------------

_CATEGORY_MAP: dict[str, tuple[list[str], str]] = {
    "ANIME": (["m.type = 'ANIME'"], "m.episodes"),
    "MANGA": (["m.type = 'MANGA'", "m.format = 'MANGA'", "m.country_of_origin = 'JP'"], "m.chapters"),
    "LIGHT_NOVEL": (["m.type = 'MANGA'", "m.format = 'NOVEL'"], "COALESCE(m.chapters, m.volumes)"),
    "MANHWA": (["m.type = 'MANGA'", "m.format = 'MANGA'", "m.country_of_origin IN ('KR','CN')"], "m.chapters"),
    "ONE_SHOT": (["m.type = 'MANGA'", "m.format = 'ONE_SHOT'"], "m.chapters"),
    "ANY_PRINT": (["m.type = 'MANGA'"], "COALESCE(m.chapters, m.volumes)"),
}

_GATE_SQL = (
    "(c.source_pool != 'DISCOVERY' OR ("
    "(m.prequel_id IS NULL OR m.prequel_id IN (SELECT media_id FROM user_list WHERE status = 'COMPLETED'))"
    " AND (m.parent_id IS NULL OR m.parent_id IN (SELECT media_id FROM user_list WHERE status = 'COMPLETED'))))"
)


def search_new_recommendations(
    db: sqlite3.Connection,
    *,
    category: str = "ANIME",
    include_movies_or_ovas: bool = False,
    min_length: int | None = None,
    max_length: int | None = None,
    max_volumes: int | None = None,
    source_pool: str = "BOTH",
    only_finished: bool = False,
    required: Sequence[str] | None = None,
    excluded: Sequence[str] | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    base, length_expr = _CATEGORY_MAP[category]
    conditions = list(base)
    params: list[Any] = []

    if category == "ANIME" and not include_movies_or_ovas:
        conditions.append("m.format IN ('TV','TV_SHORT','ONA')")
        conditions.append("(m.episodes IS NULL OR m.episodes >= 3)")
    elif category == "ANIME" and include_movies_or_ovas:
        conditions.append("m.format IN ('TV','TV_SHORT','ONA','MOVIE','OVA','SPECIAL')")

    if min_length is not None:
        conditions.append(f"({length_expr} IS NULL OR {length_expr} >= ?)")
        params.append(min_length)
    if max_length is not None:
        conditions.append(f"({length_expr} IS NULL OR {length_expr} <= ?)")
        params.append(max_length)
    if max_volumes is not None:
        conditions.append("(m.volumes IS NULL OR m.volumes <= ?)")
        params.append(max_volumes)
    if only_finished:
        conditions.append("m.status = 'FINISHED'")

    if source_pool == "PLANNING_ONLY":
        conditions.append("c.source_pool IN ('PLANNING','BOTH')")
    elif source_pool == "DISCOVERY_ONLY":
        conditions.append("c.source_pool IN ('DISCOVERY','BOTH')")

    conditions.append(_GATE_SQL)

    sql = _CANDIDATE_SELECT + " WHERE " + " AND ".join(conditions)
    rows = db.execute(sql, params).fetchall()

    required_terms = [t.strip().lower() for t in (required or []) if t.strip()]
    excluded_terms = [t.strip().lower() for t in (excluded or []) if t.strip()]

    def _matches_any(terms: list[str], haystack: set[str]) -> bool:
        return any(term in item for term in terms for item in haystack)

    results: list[dict[str, Any]] = []
    for row in rows:
        candidate = _row_to_candidate(row)
        haystack = {g.lower() for g in candidate["genres"]} | {t.lower() for t in candidate["tags"]}
        if required_terms and not _matches_any(required_terms, haystack):
            continue
        if excluded_terms and _matches_any(excluded_terms, haystack):
            continue
        results.append(candidate)
    return results[:limit]


# --- Tool 2: find_unseen_franchise_entries ----------------------------------------------

def find_unseen_franchise_entries(
    db: sqlite3.Connection,
    *,
    source_type: str = "ANIME",
    relation_types: Sequence[str] | None = None,
    target_formats: Sequence[str] | None = None,
    min_parent_score: int = 0,
    include_unreleased: bool = False,
    limit: int = 25,
) -> list[dict[str, Any]]:
    relation_types = list(relation_types or ["SEQUEL", "SIDE_STORY"])
    params: list[Any] = []

    sql = """
    SELECT mr.*, ul.affinity_weight AS parent_weight, ul.score AS parent_score,
           m.title_romaji AS source_title_romaji, m.title_english AS source_title_english,
           m.format AS source_format, m.type AS source_type
    FROM media_relations mr
    JOIN user_list ul ON ul.media_id = mr.source_id
    JOIN media m ON m.id = mr.source_id
    WHERE ul.status IN ('COMPLETED','CURRENT','PAUSED')
      AND mr.relation_type IN ({placeholders})
      AND mr.target_id NOT IN (SELECT media_id FROM user_list WHERE status = 'COMPLETED')
    """
    sql = sql.replace("{placeholders}", ",".join("?" * len(relation_types)))
    params.extend(relation_types)

    if source_type in ("ANIME", "MANGA"):
        sql += " AND m.type = ?"
        params.append(source_type)

    if target_formats:
        sql += " AND mr.target_format IN ({placeholders})".replace(
            "{placeholders}", ",".join("?" * len(target_formats))
        )
        params.extend(target_formats)

    if min_parent_score:
        sql += " AND ul.score >= ?"
        params.append(min_parent_score)

    if not include_unreleased:
        sql += " AND (mr.target_status IS NULL OR mr.target_status != 'NOT_YET_RELEASED')"

    sql += " ORDER BY ul.affinity_weight DESC, mr.target_score DESC LIMIT ?"
    params.append(limit)

    return [dict(row) for row in db.execute(sql, params).fetchall()]


# --- Tool 3: find_cross_media_adaptations ----------------------------------------------

def find_cross_media_adaptations(
    db: sqlite3.Connection,
    *,
    direction: str = "ANIME_TO_PRINT",
    target_print_format: str = "ANY",
    min_user_score: int = 75,
    only_finished_target: bool = False,
    limit: int = 15,
) -> list[dict[str, Any]]:
    if direction == "ANIME_TO_PRINT":
        # My anime works adapted from a print source (manga/LN).
        sql = """
        SELECT mr.*, ul.affinity_weight AS parent_weight, ul.score AS parent_score,
               m.title_romaji AS source_title_romaji, m.title_english AS source_title_english,
               m.format AS source_format
        FROM media_relations mr
        JOIN user_list ul ON ul.media_id = mr.source_id
        JOIN media m ON m.id = mr.source_id
        WHERE m.type = 'ANIME'
          AND mr.relation_type = 'ADAPTATION'
          AND mr.target_type = 'MANGA'
          AND ul.status IN ('COMPLETED','CURRENT','PAUSED')
          AND mr.target_id NOT IN (SELECT media_id FROM user_list WHERE status = 'COMPLETED')
        """
    else:  # PRINT_TO_ANIME
        sql = """
        SELECT mr.*, ul.affinity_weight AS parent_weight, ul.score AS parent_score,
               m.title_romaji AS source_title_romaji, m.title_english AS source_title_english,
               m.format AS source_format
        FROM media_relations mr
        JOIN user_list ul ON ul.media_id = mr.source_id
        JOIN media m ON m.id = mr.source_id
        WHERE m.type = 'MANGA'
          AND mr.relation_type = 'SOURCE'
          AND mr.target_type = 'ANIME'
          AND ul.status IN ('COMPLETED','CURRENT','PAUSED')
          AND mr.target_id NOT IN (SELECT media_id FROM user_list WHERE status = 'COMPLETED')
        """

    params: list[Any] = []
    if target_print_format != "ANY":
        sql += " AND mr.target_format = ?"
        params.append(target_print_format)
    if min_user_score:
        sql += " AND ul.score >= ?"
        params.append(min_user_score)
    if only_finished_target:
        sql += " AND mr.target_status = 'FINISHED'"

    sql += " ORDER BY ul.affinity_weight DESC LIMIT ?"
    params.append(limit)

    return [dict(row) for row in db.execute(sql, params).fetchall()]


# --- Misc -------------------------------------------------------------------------------

def read_meta(db: sqlite3.Connection) -> dict[str, str]:
    return {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}


def count_candidates(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()
    return int(row["n"])


def top_by_graph_score(
    db: sqlite3.Connection, limit: int = 10, media_type: str | None = None
) -> list[dict[str, Any]]:
    sql = _CANDIDATE_SELECT
    params: list[Any] = []
    if media_type:
        sql += " WHERE m.type = ?"
        params.append(media_type)
    sql += " ORDER BY c.graph_score DESC LIMIT ?"
    params.append(limit)
    return [_row_to_candidate(r) for r in db.execute(sql, params).fetchall()]
