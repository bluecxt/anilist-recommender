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

# Bump when the stored format changes in a way a stale DB can't satisfy (the
# CLI's `_require_synced` guard reads this from the `meta` table).
SCHEMA_VERSION = 6

CREATE_MEDIA = """
CREATE TABLE media (
  id INTEGER PRIMARY KEY,
  type TEXT NOT NULL,
  title_romaji TEXT,
  title_english TEXT,
  title_native TEXT,
  synonyms TEXT,
  format TEXT,
  status TEXT,
  episodes INTEGER,
  duration INTEGER,
  chapters INTEGER,
  volumes INTEGER,
  country_of_origin TEXT,
  average_score INTEGER,
  popularity INTEGER,
  is_adult INTEGER DEFAULT 0,
  genres TEXT,
  tags TEXT,
  studios TEXT,
  description TEXT,
  prequel_id INTEGER,
  parent_id INTEGER,
  is_root INTEGER DEFAULT 1,
  season_number INTEGER,
  franchise_episodes INTEGER
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
  target_title_romaji TEXT,
  target_title_english TEXT,
  target_title_native TEXT,
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

# Never dropped by reset_schema: the cache survives syncs so unchanged
# candidates keep their vectors (see P1 #9 — embedding cache + atomic sync).
CREATE_EMBEDDING_CACHE = """
CREATE TABLE IF NOT EXISTS embedding_cache (
  media_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector TEXT NOT NULL
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
    db.execute("PRAGMA busy_timeout=5000")
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
    db.execute(CREATE_EMBEDDING_CACHE)
    db.execute("PRAGMA foreign_keys=ON")


_MEDIA_COLUMNS = (
    "id, type, title_romaji, title_english, title_native, synonyms, format, status, episodes, "
    "duration, chapters, volumes, country_of_origin, average_score, popularity, is_adult, genres, "
    "tags, studios, description, prequel_id, parent_id, is_root, season_number, franchise_episodes"
)


def _media_values(media: dict[str, Any]) -> tuple[Any, ...]:
    return (
        media.get("id"),
        media.get("type"),
        media.get("title_romaji"),
        media.get("title_english"),
        media.get("title_native"),
        json.dumps(media.get("synonyms") or []),
        media.get("format"),
        media.get("status"),
        media.get("episodes"),
        media.get("duration"),
        media.get("chapters"),
        media.get("volumes"),
        media.get("country_of_origin"),
        media.get("average_score"),
        media.get("popularity"),
        int(media.get("is_adult") or 0),
        json.dumps(media.get("genres") or []),
        json.dumps(media.get("tags") or []),
        json.dumps(media.get("studios") or []),
        media.get("description") or "",
        media.get("prequel_id"),
        media.get("parent_id"),
        media.get("is_root") if media.get("is_root") is not None else 1,
        media.get("season_number"),
        media.get("franchise_episodes"),
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
            f"VALUES ({','.join('?' * 25)})",
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
            "target_type, target_format, target_country, target_status, "
            "target_title_romaji, target_title_english, target_title_native, "
            "target_episodes, target_chapters, target_volumes, target_score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["source_id"],
                row["target_id"],
                row["relation_type"],
                row["target_type"],
                row["target_format"],
                row["target_country"],
                row["target_status"],
                row["target_title_romaji"],
                row["target_title_english"],
                row["target_title_native"],
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
    db.execute("INSERT INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    db.execute("INSERT INTO meta (key, value) VALUES ('num_works', ?)", (str(len(works)),))
    db.execute("INSERT INTO meta (key, value) VALUES ('num_candidates', ?)", (str(len(candidates)),))
    db.execute("INSERT INTO meta (key, value) VALUES ('num_relations', ?)", (str(len(relation_rows)),))
    # No commit here: sync wraps the whole rebuild in ONE transaction so a
    # failure mid-sync rolls everything back (atomic rebuild, P1 #9).


# --- Embedding cache (P1 #9) -----------------------------------------------------------


def load_embedding_cache(db: sqlite3.Connection, model: str) -> dict[int, tuple[str, list[float]]]:
    """media_id → (content_hash, vector) for the given embedding model."""
    rows = db.execute(
        "SELECT media_id, content_hash, vector FROM embedding_cache WHERE model = ?", (model,)
    ).fetchall()
    cache: dict[int, tuple[str, list[float]]] = {}
    for row in rows:
        try:
            cache[int(row["media_id"])] = (str(row["content_hash"]), json.loads(row["vector"]))
        except (TypeError, ValueError):
            continue  # corrupt row → treated as a cache miss
    return cache


def save_embedding_cache(
    db: sqlite3.Connection, model: str, entries: list[tuple[int, str, list[float]]]
) -> None:
    db.executemany(
        "INSERT OR REPLACE INTO embedding_cache (media_id, model, content_hash, vector) "
        "VALUES (?,?,?,?)",
        [(media_id, model, content_hash, json.dumps(vector)) for media_id, content_hash, vector in entries],
    )


def prune_embedding_cache(db: sqlite3.Connection, model: str, keep_ids: set[int]) -> None:
    """Drop cache rows whose media is no longer a candidate."""
    if not keep_ids:
        db.execute("DELETE FROM embedding_cache WHERE model = ?", (model,))
        return
    placeholders = ",".join("?" * len(keep_ids))
    db.execute(
        f"DELETE FROM embedding_cache WHERE model = ? AND media_id NOT IN ({placeholders})",
        [model, *sorted(keep_ids)],
    )


_CANDIDATE_SELECT = """
SELECT
  c.media_id,
  c.graph_score,
  c.planning_bonus,
  c.source_pool,
  c.positive_sources,
  c.negative_warnings,
  (SELECT ul.updated_at FROM user_list ul WHERE ul.media_id = c.media_id) AS list_updated_at,
  m.type,
  m.title_romaji,
  m.title_english,
  m.title_native,
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
  m.is_root,
  m.season_number,
  m.franchise_episodes
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
    "ANY": (["m.type IN ('ANIME','MANGA')"], "COALESCE(m.episodes, m.chapters, m.volumes)"),
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
    allow_adult: bool = False,
    adult_only: bool = False,
) -> list[dict[str, Any]]:
    base, length_expr = _CATEGORY_MAP[category]
    conditions = list(base)
    params: list[Any] = []

    # Adult (hentai) titles are excluded unless explicitly requested.
    if adult_only:
        conditions.append("m.is_adult = 1")
    elif not allow_adult:
        conditions.append("(m.is_adult IS NULL OR m.is_adult = 0)")

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

    # Safety net: the LLM often invents a non-canonical tag (e.g. "Mind Games" instead
    # of AniList's "Psychological"), which would otherwise return an empty list. Drop the
    # required filter — keeping the excluded one — instead of showing nothing.
    if not results and required_terms:
        for row in rows:
            candidate = _row_to_candidate(row)
            haystack = {g.lower() for g in candidate["genres"]} | {t.lower() for t in candidate["tags"]}
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
           m.title_native AS source_title_native,
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

    fetch_limit = min(max(limit * 3, limit), 300)
    sql += " ORDER BY ul.affinity_weight DESC, mr.target_score DESC LIMIT ?"
    params.append(fetch_limit)

    # De-duplicate targets (same sequel reachable from several sources) keeping
    # the highest-affinity source, then cap at the requested limit.
    seen: set[int] = set()
    results: list[dict[str, Any]] = []
    for row in db.execute(sql, params).fetchall():
        if row["target_id"] in seen:
            continue
        seen.add(row["target_id"])
        results.append(dict(row))
        if len(results) >= limit:
            break
    return results


# --- Tool 3: find_cross_media_adaptations ----------------------------------------------

def find_cross_media_adaptations(
    db: sqlite3.Connection,
    *,
    direction: str = "ANIME_TO_PRINT",
    target_print_format: str = "ANY",
    min_user_score: int = 0,
    only_finished_target: bool = False,
    limit: int = 15,
) -> list[dict[str, Any]]:
    if direction == "ANIME_TO_PRINT":
        # My anime works whose SOURCE material is a manga/LN (`anime --SOURCE--> print`).
        sql = """
        SELECT mr.*, ul.affinity_weight AS parent_weight, ul.score AS parent_score,
               m.title_romaji AS source_title_romaji, m.title_english AS source_title_english,
               m.title_native AS source_title_native,
               m.format AS source_format, m.type AS source_type
        FROM media_relations mr
        JOIN user_list ul ON ul.media_id = mr.source_id
        JOIN media m ON m.id = mr.source_id
        WHERE m.type = 'ANIME'
          AND mr.relation_type = 'SOURCE'
          AND mr.target_type = 'MANGA'
          AND ul.status IN ('COMPLETED','CURRENT','PAUSED')
          AND mr.target_id NOT IN (SELECT media_id FROM user_list WHERE status = 'COMPLETED')
        """
    else:  # PRINT_TO_ANIME
        # My print works whose ADAPTATION is an anime (`print --ADAPTATION--> anime`).
        sql = """
        SELECT mr.*, ul.affinity_weight AS parent_weight, ul.score AS parent_score,
               m.title_romaji AS source_title_romaji, m.title_english AS source_title_english,
               m.title_native AS source_title_native,
               m.format AS source_format, m.type AS source_type
        FROM media_relations mr
        JOIN user_list ul ON ul.media_id = mr.source_id
        JOIN media m ON m.id = mr.source_id
        WHERE m.type = 'MANGA'
          AND mr.relation_type = 'ADAPTATION'
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
    fetch_limit = min(max(limit * 3, limit), 300)
    params.append(fetch_limit)

    # De-duplicate targets (the same manga/LN is reachable from several anime entries,
    # e.g. a TV series and its movie both point to the same source) keeping the
    # highest-affinity source, then cap at the requested limit.
    seen: set[int] = set()
    results: list[dict[str, Any]] = []
    for row in db.execute(sql, params).fetchall():
        if row["target_id"] in seen:
            continue
        seen.add(row["target_id"])
        results.append(dict(row))
        if len(results) >= limit:
            break
    return results


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


# --- explain command (P2 #11) ----------------------------------------------------------


def find_media_by_title(
    db: sqlite3.Connection, text: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Fuzzy title lookup across romaji/english/native, most popular first."""
    needle = text.strip()
    if not needle:
        return []
    like = f"%{needle}%"
    rows = db.execute(
        "SELECT id, type, title_romaji, title_english, title_native, format, status, "
        "average_score, popularity, episodes, chapters, volumes, genres, tags, description, "
        "prequel_id, parent_id, is_root, season_number, franchise_episodes "
        "FROM media "
        "WHERE title_romaji LIKE ? OR title_english LIKE ? OR title_native LIKE ? "
        "ORDER BY popularity DESC LIMIT ?",
        (like, like, like, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def resolve_media_by_alias(db: sqlite3.Connection, query: str) -> dict[str, Any] | None:
    """Resolve a nickname/title mentioned in a natural-language query to a local media.

    Matches case-insensitively against every title variant AND AniList's ``synonyms``
    (so "ngnl" → No Game No Life, "tensura" → Slime, "opm" → One Punch Man). The
    longest matched alias wins (most specific). Returns a media dict whose ``genres``
    and ``tags`` are parsed into lists (ready for ``build_embedding_text``), or None.
    """
    text = (query or "").lower().strip()
    if not text:
        return None
    rows = db.execute(
        "SELECT id, type, title_romaji, title_english, title_native, synonyms, "
        "genres, tags, description FROM media"
    ).fetchall()

    def _list(value: Any) -> list[str]:
        try:
            parsed = json.loads(value or "[]")
            return parsed if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            return []

    best: tuple[int, dict[str, Any]] | None = None
    for row in rows:
        media = dict(row)
        names: list[str] = [
            media.get("title_romaji"),
            media.get("title_english"),
            media.get("title_native"),
        ]
        names.extend(_list(media.get("synonyms")))
        for name in names:
            if not name:
                continue
            lowered = str(name).lower().strip()
            if len(lowered) < 3 or lowered not in text:
                continue
            if best is None or len(lowered) > best[0]:
                media["genres"] = _list(media.get("genres"))
                media["tags"] = _list(media.get("tags"))
                best = (len(lowered), media)
    return best[1] if best else None


def get_candidate(db: sqlite3.Connection, media_id: int) -> dict[str, Any] | None:
    rows = db.execute(_CANDIDATE_SELECT + " WHERE c.media_id = ?", (media_id,)).fetchall()
    return _row_to_candidate(rows[0]) if rows else None


def completed_or_dropped_ids(db: sqlite3.Connection) -> set[int]:
    """Media the user already finished or abandoned — never recommend these."""
    rows = db.execute(
        "SELECT media_id FROM user_list WHERE status IN ('COMPLETED','DROPPED')"
    ).fetchall()
    return {int(row["media_id"]) for row in rows}


def recommendation_sources(db: sqlite3.Connection, target_id: int) -> list[dict[str, Any]]:
    """Raw collaborative edges pointing at a candidate, with source context."""
    rows = db.execute(
        "SELECT r.source_media_id, r.rating, ul.affinity_weight, ul.status AS list_status, "
        "ul.score AS user_score, m.title_romaji, m.title_english, m.title_native, "
        "m.type, m.format, m.average_score "
        "FROM media_recommendations r "
        "JOIN user_list ul ON ul.media_id = r.source_media_id "
        "JOIN media m ON m.id = r.source_media_id "
        "WHERE r.target_media_id = ? "
        "ORDER BY ul.affinity_weight DESC",
        (target_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def relation_context(db: sqlite3.Connection, media_id: int) -> list[dict[str, Any]]:
    """Every relation row where the media appears as source or target."""
    rows = db.execute(
        "SELECT source_id, target_id, relation_type, target_title_romaji, target_title_english, "
        "target_title_native, target_format, target_type, target_status "
        "FROM media_relations WHERE source_id = ? OR target_id = ? "
        "ORDER BY relation_type",
        (media_id, media_id),
    ).fetchall()
    return [dict(row) for row in rows]


def franchise_triples(db: sqlite3.Connection) -> list[tuple[int, str, int]]:
    """(source, type, target) triples for the prequel/parent maps, from the DB."""
    rows = db.execute(
        "SELECT source_id, relation_type, target_id FROM media_relations "
        "WHERE relation_type IN ('SEQUEL','PREQUEL','PARENT')"
    ).fetchall()
    return [(int(row["source_id"]), str(row["relation_type"]), int(row["target_id"])) for row in rows]
