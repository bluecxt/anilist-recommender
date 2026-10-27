"""Deterministic scoring — the LLM never guesses your taste.

V2 additions:
* Unified progress ratio (anime episodes / manga chapters / LN volumes / 20-chapter
  engagement threshold for ongoing works).
* Structural relation maps (prequel / parent) from the `relations` graph.
* Prequel Gatekeeping + Root Score Transfer: a discovered sequel whose prequel is
  not completed is removed and its GraphScore is transferred onto the root.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .graphql_client import normalize_media, normalize_relation_node


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def display_title(media: dict[str, Any]) -> str:
    return media.get("title_romaji") or media.get("title_english") or f"#{media.get('id')}"


def compute_progress_ratio(
    media: dict[str, Any], status: str, progress: int, progress_volumes: int = 0
) -> float:
    """Fraction of the media consumed, in [0, 1]."""
    if status == "COMPLETED":
        return 1.0
    progress = progress or 0
    if media.get("type") == "ANIME":
        total = media.get("episodes")
        consumed = progress
    else:
        # MANGA / NOVEL: chapters, else volumes, else a 20-chapter engagement estimate.
        if media.get("chapters"):
            total = media.get("chapters")
            consumed = progress
        elif media.get("volumes"):
            total = media.get("volumes")
            consumed = progress_volumes or 0
        else:
            total = 20
            consumed = progress
    if not total or total <= 0:
        return 0.0
    return min(1.0, consumed / total)


def compute_affinity(
    status: str,
    score: float,
    repeat: int,
    progress_ratio: float,
    mu: float,
    sigma: float,
    is_favourite: bool,
) -> tuple[float, float]:
    sigma = max(sigma, 10.0)
    z = (score - mu) / sigma if score > 0 else 0.0

    if status == "COMPLETED":
        if score > 0:
            weight = clamp(z * 1.2, -1.5, 2.5)
            if abs(z) <= 0.05:
                weight = max(weight, 0.25)
        else:
            weight = 0.5
    elif status == "REPEATING":
        if score > 0:
            weight = clamp(z * 1.2, -1.5, 2.5)
        else:
            weight = 0.5
    elif status == "CURRENT":
        if score > 0:
            weight = clamp(z, -0.5, 1.8)
        else:
            weight = 0.8 if progress_ratio > 0.5 else 0.4
    elif status == "PAUSED":
        if score > 0:
            weight = clamp(z * 0.6, -1.0, 0.8)
        elif progress_ratio < 0.25:
            weight = -0.5
        else:
            weight = 0.1
    elif status == "DROPPED":
        if score == 0:
            weight = -2.0
        else:
            weight = clamp(z * 1.5 - 1.0, -3.0, -0.5)
    else:
        weight = 0.0  # PLANNING

    if status == "REPEATING" or repeat > 0:
        weight = min(weight + 1.5, 3.0)

    if is_favourite:
        weight = max(weight + 2.0, 3.5)

    return weight, z


@dataclass
class Work:
    media_id: int
    media: dict[str, Any]
    status: str
    score: float
    progress: int
    progress_volumes: int
    repeat: int
    updated_at: int
    progress_ratio: float
    weight: float
    z_score: float
    is_favourite: bool


def build_works(
    entries: list[dict[str, Any]],
    fav_ids: set[int],
    stats: dict[str, dict[str, float]],
) -> list[Work]:
    works: list[Work] = []
    for entry in entries:
        media = normalize_media(entry.get("media") or {})
        media_id = media["id"]
        if media_id is None:
            continue
        status = entry.get("status") or ""
        score = float(entry.get("score") or 0.0)
        progress = int(entry.get("progress") or 0)
        progress_volumes = int(entry.get("progressVolumes") or 0)
        repeat = int(entry.get("repeat") or 0)
        updated_at = int(entry.get("updatedAt") or 0)

        stat = stats.get("anime" if media["type"] == "ANIME" else "manga", {})
        mu = float(stat.get("mean_score", 0.0))
        sigma = float(stat.get("std_dev", 0.0))

        ratio = compute_progress_ratio(media, status, progress, progress_volumes)
        is_fav = media_id in fav_ids
        weight, z = compute_affinity(status, score, repeat, ratio, mu, sigma, is_fav)

        works.append(
            Work(
                media_id=media_id,
                media=media,
                status=status,
                score=score,
                progress=progress,
                progress_volumes=progress_volumes,
                repeat=repeat,
                updated_at=updated_at,
                progress_ratio=ratio,
                weight=weight,
                z_score=z,
                is_favourite=is_fav,
            )
        )
    return works


@dataclass
class Candidate:
    media_id: int
    media: dict[str, Any]
    graph_score: float = 0.0
    planning_bonus: float = 0.0
    source_pool: str = "DISCOVERY"
    positive_sources: list[str] = field(default_factory=list)
    negative_warnings: list[str] = field(default_factory=list)


def compute_graph_scores(
    works: list[Work],
    rec_edges: list[tuple[int, int, dict[str, Any]]],
) -> list[Candidate]:
    works_by_id: dict[int, Work] = {w.media_id: w for w in works}
    status_by_id: dict[int, str] = {w.media_id: w.status for w in works}
    planning_ids: set[int] = {w.media_id for w in works if w.status == "PLANNING"}

    edges: dict[int, dict[int, int]] = defaultdict(dict)
    discovered_media: dict[int, dict[str, Any]] = {}

    for source_id, rating, raw_target in rec_edges:
        target = normalize_media(raw_target or {})
        target_id = target["id"]
        if target_id is None:
            continue
        if status_by_id.get(target_id) in ("COMPLETED", "DROPPED"):
            continue
        discovered_media[target_id] = target
        current = edges[target_id].get(source_id, -10**9)
        if rating > current:
            edges[target_id][source_id] = rating

    candidates: dict[int, Candidate] = {}
    for pid in planning_ids:
        candidates[pid] = Candidate(media_id=pid, media=works_by_id[pid].media)

    for did, media in discovered_media.items():
        if did in candidates:
            candidates[did].media = media
        else:
            candidates[did] = Candidate(media_id=did, media=media)

    for media_id, source_ratings in edges.items():
        cand = candidates[media_id]
        total = 0.0
        positive: list[tuple[float, str]] = []
        negative: list[str] = []
        for source_id, rating in source_ratings.items():
            work = works_by_id.get(source_id)
            if work is None:
                continue
            contribution = work.weight * math.log1p(max(0.0, float(rating)))
            total += contribution
            label = display_title(work.media)
            if work.weight > 0.5:
                positive.append((contribution, f"{label} ({work.weight:+.1f}, rating: {rating})"))
            elif work.weight < 0.0:
                negative.append(f"{label} ({work.status}, {work.weight:+.1f}, rating: {rating})")
        cand.graph_score = total
        positive.sort(key=lambda pair: -pair[0])
        cand.positive_sources = [text for _, text in positive]
        cand.negative_warnings = negative

    for media_id, cand in candidates.items():
        in_planning = media_id in planning_ids
        in_discovery = media_id in discovered_media
        cand.planning_bonus = 1.0 if in_planning else 0.0
        if in_planning and in_discovery:
            cand.source_pool = "BOTH"
        elif in_planning:
            cand.source_pool = "PLANNING"
        else:
            cand.source_pool = "DISCOVERY"

    return list(candidates.values())


def build_relation_maps(
    relation_edges: list[tuple[int, str, dict[str, Any]]],
) -> tuple[dict[int, int], dict[int, int], list[dict[str, Any]]]:
    """Derive prequel/parent maps + denormalized `media_relations` rows.

    ``relation_edges`` items are ``(source_id, relation_type, raw_target_node)``.
    """
    prequel_of: dict[int, int] = {}
    parent_of: dict[int, int] = {}
    rows: list[dict[str, Any]] = []

    for source_id, relation_type, raw_target in relation_edges:
        target = normalize_relation_node(raw_target or {})
        target_id = target.get("id")
        if target_id is None:
            continue
        if relation_type == "SEQUEL":
            prequel_of[target_id] = source_id
        elif relation_type == "PREQUEL":
            prequel_of[source_id] = target_id
        elif relation_type == "PARENT":
            parent_of[source_id] = target_id
        rows.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "target_type": target.get("type"),
                "target_format": target.get("format"),
                "target_country": target.get("country"),
                "target_status": target.get("status"),
                "target_title": target.get("title_romaji") or target.get("title_english"),
                "target_episodes": target.get("episodes"),
                "target_chapters": target.get("chapters"),
                "target_volumes": target.get("volumes"),
                "target_score": target.get("average_score"),
            }
        )
    return prequel_of, parent_of, rows


def apply_prequel_gatekeeping(
    works: list[Work],
    candidates: list[Candidate],
    prequel_of: dict[int, int],
    parent_of: dict[int, int],
) -> list[Candidate]:
    """Remove discovered sequels whose prequel/parent is not completed.

    Their GraphScore (+ sources/warnings) is transferred onto the root candidate
    when the root is itself recommendable (e.g. a planned Season 1).
    """
    completed = {w.media_id for w in works if w.status == "COMPLETED"}
    cand_by_id = {c.media_id: c for c in candidates}
    gated: set[int] = set()

    for candidate in candidates:
        if candidate.source_pool != "DISCOVERY":
            continue  # never gate my own planning list
        transfer_to = None
        prequel = prequel_of.get(candidate.media_id)
        parent = parent_of.get(candidate.media_id)
        if prequel is not None and prequel not in completed:
            transfer_to = prequel
        elif parent is not None and parent not in completed:
            transfer_to = parent
        if transfer_to is not None:
            gated.add(candidate.media_id)
            root = cand_by_id.get(transfer_to)
            if root is not None:
                root.graph_score += candidate.graph_score
                root.positive_sources.extend(candidate.positive_sources)
                root.negative_warnings.extend(candidate.negative_warnings)

    return [c for c in candidates if c.media_id not in gated]


def annotate_roots(
    media_items: list[dict[str, Any]],
    prequel_of: dict[int, int],
    parent_of: dict[int, int],
) -> None:
    """Set ``prequel_id`` / ``parent_id`` / ``is_root`` on each media dict in place."""
    for media in media_items:
        media_id = media.get("id")
        media["prequel_id"] = prequel_of.get(media_id)
        media["parent_id"] = parent_of.get(media_id)
        media["is_root"] = 1 if (media["prequel_id"] is None and media["parent_id"] is None) else 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_taste_profile(
    username: str,
    works: list[Work],
    stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    genre_score: dict[str, float] = defaultdict(float)
    genre_count: dict[str, int] = defaultdict(int)
    tag_score: dict[str, float] = defaultdict(float)
    tag_count: dict[str, int] = defaultdict(int)
    anti_tag_count: dict[str, int] = defaultdict(int)

    favourites: list[dict[str, str | None]] = []
    anime_lengths: list[float] = []
    manga_lengths: list[float] = []

    for work in works:
        media = work.media
        if work.is_favourite:
            favourites.append({"romaji": media.get("title_romaji"), "english": media.get("title_english")})
        if work.weight > 0:
            for genre in media.get("genres") or []:
                genre_score[genre] += work.weight
                genre_count[genre] += 1
            for tag in media.get("tags") or []:
                tag_score[tag] += work.weight
                tag_count[tag] += 1
        if work.status == "DROPPED" or work.weight < 0:
            for tag in media.get("tags") or []:
                anti_tag_count[tag] += 1
        if work.weight > 0.5:
            if media.get("type") == "ANIME" and media.get("episodes"):
                anime_lengths.append(float(media["episodes"]))
            elif media.get("type") == "MANGA" and media.get("chapters"):
                manga_lengths.append(float(media["chapters"]))

    def top(score_map: dict[str, float], count_map: dict[str, int], n: int) -> list[dict[str, Any]]:
        items = sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        return [{"name": name, "score": round(score, 2), "count": count_map[name]} for name, score in items]

    anti_tags = sorted(anti_tag_count.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    return {
        "username": username,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": stats,
        "top_genres": top(genre_score, genre_count, 15),
        "top_tags": top(tag_score, tag_count, 15),
        "anti_tags": [{"name": name, "count": count} for name, count in anti_tags],
        "preferred_length": {
            "anime_avg_episodes": round(_mean(anime_lengths), 1),
            "manga_avg_chapters": round(_mean(manga_lengths), 1),
        },
        "favourites": favourites,
    }
