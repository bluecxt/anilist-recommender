"""Deterministic scoring — the LLM never guesses your taste.

Everything the agent consumes is pre-computed here during `sync`:

* ``affinity_weight`` of every list entry (Z-score based, status-aware).
* ``GraphScore`` of every candidate (collaborative convergence from your works).
* ``data/taste_profile.json`` (compact, < ~800 tokens).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .graphql_client import normalize_media


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def display_title(media: dict[str, Any]) -> str:
    """Human-friendly title (romaji preferred, english fallback)."""
    return media.get("title_romaji") or media.get("title_english") or f"#{media.get('id')}"


def compute_progress_ratio(media: dict[str, Any], status: str, progress: int) -> float:
    """Fraction of the media consumed, in [0, 1]."""
    if status == "COMPLETED":
        return 1.0
    total = media.get("episodes") if media.get("type") == "ANIME" else (
        media.get("chapters") or media.get("volumes")
    )
    progress = progress or 0
    if not total or total <= 0:
        return 0.0
    return min(1.0, progress / total)


def compute_affinity(
    status: str,
    score: float,
    repeat: int,
    progress_ratio: float,
    mu: float,
    sigma: float,
    is_favourite: bool,
) -> tuple[float, float]:
    """Return ``(affinity_weight, z_score)`` for one list entry.

    Implements the cumulative, conditional rules from the spec:

    * Z-score is ``(score - mu) / sigma`` with a ``sigma`` floor of 10.
    * Base weight depends on the list status and the score.
    * ``REPEATING`` / ``repeat > 0`` adds ``+1.5`` (capped at ``+3.0``).
    * Favourites add ``+2.0`` and force a floor of ``+3.5``.
    """
    sigma = max(sigma, 10.0)
    z = (score - mu) / sigma if score > 0 else 0.0

    if status == "COMPLETED":
        if score > 0:
            weight = clamp(z * 1.2, -1.5, 2.5)
            if abs(z) <= 0.05:  # scored ≈ your mean → mild positive
                weight = max(weight, 0.25)
        else:
            weight = 0.5  # finished but unrated
    elif status == "REPEATING":
        # A rewatch implies it was finished; treat base like COMPLETED.
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
            weight = -0.5  # "soft drop"
        else:
            weight = 0.1
    elif status == "DROPPED":
        if score == 0:
            weight = -2.0  # implicit strong rejection
        else:
            weight = clamp(z * 1.5 - 1.0, -3.0, -0.5)
    else:
        weight = 0.0  # PLANNING — no signal, excluded from sources

    if status == "REPEATING" or repeat > 0:
        weight = min(weight + 1.5, 3.0)

    if is_favourite:
        weight = max(weight + 2.0, 3.5)

    return weight, z


@dataclass
class Work:
    """A single list entry with its computed affinity signal."""

    media_id: int
    media: dict[str, Any]
    status: str
    score: float
    progress: int
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
    """Normalize + score every list entry (PLANNING included, weight 0)."""
    works: list[Work] = []
    for entry in entries:
        media = normalize_media(entry.get("media") or {})
        media_id = media["id"]
        if media_id is None:
            continue
        status = entry.get("status") or ""
        score = float(entry.get("score") or 0.0)
        progress = int(entry.get("progress") or 0)
        repeat = int(entry.get("repeat") or 0)
        updated_at = int(entry.get("updatedAt") or 0)

        stat = stats.get("anime" if media["type"] == "ANIME" else "manga", {})
        mu = float(stat.get("mean_score", 0.0))
        sigma = float(stat.get("std_dev", 0.0))

        ratio = compute_progress_ratio(media, status, progress)
        is_fav = media_id in fav_ids
        weight, z = compute_affinity(status, score, repeat, ratio, mu, sigma, is_fav)

        works.append(
            Work(
                media_id=media_id,
                media=media,
                status=status,
                score=score,
                progress=progress,
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
    """A recommendable work with its collaborative graph signals."""

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
    """Build candidates from PLANNING + discovered media and score them.

    ``rec_edges`` is a list of ``(source_media_id, rating, raw_target_media)``.

    ``GraphScore(X) = Σ_s weight(s) * ln(1 + max(0, rating_s→X))`` over every
    source ``s`` that recommends ``X``. Candidates that are COMPLETED or DROPPED
    in your list are excluded.
    """
    works_by_id: dict[int, Work] = {w.media_id: w for w in works}
    status_by_id: dict[int, str] = {w.media_id: w.status for w in works}

    planning_ids: set[int] = {w.media_id for w in works if w.status == "PLANNING"}

    # media_id -> {source_id: max_rating}
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
                positive.append(
                    (contribution, f"{label} ({work.weight:+.1f}, rating: {rating})")
                )
            elif work.weight < 0.0:
                negative.append(
                    f"{label} ({work.status}, {work.weight:+.1f}, rating: {rating})"
                )
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_taste_profile(
    username: str,
    works: list[Work],
    stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Produce the compact taste summary injected into the LLM system prompt."""
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
            favourites.append(
                {"romaji": media.get("title_romaji"), "english": media.get("title_english")}
            )
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
        return [
            {"name": name, "score": round(score, 2), "count": count_map[name]}
            for name, score in items
        ]

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
