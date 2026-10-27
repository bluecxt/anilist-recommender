"""AniList GraphQL extraction — omni-media (ANIME + MANGA/Light Novel/Manhwa/Manhua).

Fetches the user profile/statistics, the full MediaListCollection (chunk
paginated, for both ANIME and MANGA), the collaborative `recommendations` graph
and the full structural `relations` graph (preq/sequel/side-story/adaptation).

All HTTP goes through a rate-limited async client (90 req/min, currently
degraded to ~30/min) with 429 backoff.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import logging
import re
import time
from typing import Any

import httpx

from .config import ANILIST_GRAPHQL_URL

logger = logging.getLogger(__name__)


class AniListError(RuntimeError):
    """Raised when the AniList API returns an error or is unreachable."""


# Full `Media` fragment used for list entries and `mediaRecommendation` targets.
MEDIA_FIELDS = """
  id
  type
  title { romaji english native }
  synonyms
  format
  status
  episodes
  duration
  chapters
  volumes
  countryOfOrigin
  averageScore
  popularity
  isAdult
  genres
  studios(isMain: true) { nodes { name } }
  tags { name rank isMediaSpoiler }
  description(asHtml: false)
"""

# Minimal `Media` fragment used for `relations` targets (kept small so the bulk
# `Page(id_in: [...])` relations query stays under AniList's complexity limit).
RELATION_NODE_FIELDS = """
  id
  type
  format
  countryOfOrigin
  status
  episodes
  chapters
  volumes
  averageScore
  title { romaji english native }
"""

# Reduced `Media` fragment for `mediaRecommendation` targets. Drops fields never
# used on candidates (studios/duration/popularity) so two sources can be batched
# per request via GraphQL aliases without exceeding the complexity limit.
RECOMMENDATION_MEDIA_FIELDS = """
  id
  type
  title { romaji english native }
  synonyms
  format
  status
  episodes
  chapters
  volumes
  countryOfOrigin
  averageScore
  isAdult
  genres
  tags { name rank isMediaSpoiler }
  description(asHtml: false)
"""

USER_PROFILE_QUERY = """
query ($userName: String) {
  User(name: $userName) {
    id
    name
    favourites {
      anime(perPage: 25) { nodes { id } }
      manga(perPage: 25) { nodes { id } }
    }
    statistics {
      anime { count meanScore standardDeviation }
      manga { count meanScore standardDeviation }
    }
  }
}
"""

MEDIA_LIST_QUERY = f"""
query ($userName: String, $type: MediaType, $chunk: Int, $perChunk: Int) {{
  MediaListCollection(userName: $userName, type: $type, chunk: $chunk, perChunk: $perChunk) {{
    hasNextChunk
    lists {{
      status
      entries {{
        status
        score(format: POINT_100)
        progress
        progressVolumes
        repeat
        updatedAt
        media {{
{MEDIA_FIELDS}
        }}
      }}
    }}
  }}
}}
"""

RECOMMENDATIONS_QUERY = f"""
query ($mediaId: Int, $perPage: Int) {{
  Media(id: $mediaId) {{
    id
    recommendations(sort: RATING_DESC, perPage: $perPage) {{
      nodes {{
        rating
        mediaRecommendation {{
{MEDIA_FIELDS}
        }}
      }}
    }}
  }}
}}
"""

BULK_RELATIONS_QUERY = f"""
query ($ids: [Int], $perPage: Int) {{
  Page(page: 1, perPage: $perPage) {{
    media(id_in: $ids) {{
      id
      relations {{
        edges {{
          relationType(version: 2)
          node {{
{RELATION_NODE_FIELDS}
          }}
        }}
      }}
    }}
  }}
}}
"""


class AniListClient:
    """Rate-limited async GraphQL client for graphql.anilist.co."""

    def __init__(self, min_interval: float = 0.85) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=20.0),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "anilist-recommender/0.2 (personal CLI)",
            },
        )
        self._min_interval = max(0.0, min_interval)
        # Self-tuning pacing: starts at `min_interval` and grows on every 429 so we
        # converge on AniList's *real* per-minute cap instead of repeatedly bursting
        # into it and relying on backoff.
        self._effective_interval = self._min_interval
        self._max_interval = 6.0
        self._last_request = 0.0
        self._lock = asyncio.Lock()
        # Adaptive rate-limit state, refreshed from X-RateLimit-* headers.
        self._rate_remaining: int | None = None
        self._rate_limit: int | None = None
        self._rate_reset: float | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._effective_interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    @staticmethod
    def _backoff_delay(headers: httpx.Headers, default: float) -> float:
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        reset = headers.get("X-RateLimit-Reset")
        if reset is not None:
            try:
                remaining = float(reset) - time.time()
                if remaining > 0:
                    return remaining
            except ValueError:
                pass
        return default

    def _update_rate_limit(self, headers: httpx.Headers) -> None:
        """Track the sliding-window quota from X-RateLimit-* response headers."""
        try:
            remaining = headers.get("X-RateLimit-Remaining")
            limit = headers.get("X-RateLimit-Limit")
            reset = headers.get("X-RateLimit-Reset")
            if remaining is not None:
                self._rate_remaining = int(remaining)
            if limit is not None:
                self._rate_limit = int(limit)
            if reset is not None:
                self._rate_reset = float(reset)
        except (TypeError, ValueError):
            pass

    async def _wait_for_quota(self) -> None:
        """Sleep until the next rate-limit window before the quota runs out.

        Fires well before exhaustion (10% of the budget left, min 3) so we avoid
        the server-side 429 entirely when the X-RateLimit-* headers are accurate.
        """
        if self._rate_remaining is not None and self._rate_limit:
            threshold = max(3, int(self._rate_limit * 0.1))
            if self._rate_remaining <= threshold:
                if self._rate_reset is not None:
                    wait = self._rate_reset - time.time() + 0.5
                    if wait > 0:
                        logger.info(
                            "rate limit near (%s/%s left) — waiting %.1fs for reset",
                            self._rate_remaining, self._rate_limit, wait,
                        )
                        await asyncio.sleep(wait)
                self._rate_remaining = None
                self._rate_reset = None

    async def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = {"query": query, "variables": variables}
        while True:
            await self._throttle()
            await self._wait_for_quota()
            try:
                response = await self._client.post(ANILIST_GRAPHQL_URL, json=payload)
            except httpx.TransportError as exc:
                raise AniListError(f"network error while calling AniList: {exc}") from exc
            self._update_rate_limit(response.headers)

            if response.status_code == 429:
                self._effective_interval = min(self._max_interval, self._effective_interval * 1.5)
                delay = self._backoff_delay(response.headers, 5.0)
                logger.info(
                    "AniList HTTP 429 — slowing to %.2fs/req, backing off %.1fs",
                    self._effective_interval, delay,
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 500:
                delay = self._backoff_delay(response.headers, 3.0)
                logger.info("AniList HTTP %s — retrying in %.1fs", response.status_code, delay)
                await asyncio.sleep(delay)
                continue

            response.raise_for_status()

            try:
                body = response.json()
            except ValueError as exc:
                raise AniListError("AniList returned non-JSON response") from exc

            errors = body.get("errors")
            if errors:
                messages = [e.get("message", "") or "" for e in errors]
                if any("Too Many Requests" in m or e.get("status") == 429 for m, e in zip(messages, errors)):
                    self._effective_interval = min(self._max_interval, self._effective_interval * 1.5)
                    logger.info(
                        "AniList rate limit (GraphQL) — slowing to %.2fs/req, backing off 5s",
                        self._effective_interval,
                    )
                    await asyncio.sleep(5.0)
                    continue
                raise AniListError("GraphQL error: " + "; ".join(messages))

            data = body.get("data")
            if data is None:
                raise AniListError("AniList returned no `data` payload")
            return data

    async def fetch_user_profile(self, username: str) -> dict[str, Any]:
        data = await self._post(USER_PROFILE_QUERY, {"userName": username})
        user = data.get("User")
        if not user:
            raise AniListError(f"AniList user '{username}' not found")
        return user

    async def fetch_media_list(self, username: str, media_type: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        chunk = 0
        per_chunk = 500
        while True:
            data = await self._post(
                MEDIA_LIST_QUERY,
                {"userName": username, "type": media_type, "chunk": chunk, "perChunk": per_chunk},
            )
            collection = data.get("MediaListCollection") or {}
            for group in collection.get("lists") or []:
                for entry in group.get("entries") or []:
                    if entry.get("media"):
                        entries.append(entry)
            if not collection.get("hasNextChunk"):
                break
            chunk += 1
        return entries

    async def fetch_recommendations(self, media_id: int, per_page: int = 15) -> list[dict[str, Any]]:
        data = await self._post(RECOMMENDATIONS_QUERY, {"mediaId": media_id, "perPage": per_page})
        media = data.get("Media") or {}
        recommendations = media.get("recommendations") or {}
        return recommendations.get("nodes") or []

    @staticmethod
    def _build_recommendations_batch_query(ids: list[int], per_page: int) -> str:
        parts = [
            f"a{i}: Media(id: {mid}) {{ id recommendations(sort: RATING_DESC, perPage: {per_page}) "
            f"{{ nodes {{ rating mediaRecommendation {{ {RECOMMENDATION_MEDIA_FIELDS} }} }} }} }}"
            for i, mid in enumerate(ids)
        ]
        return "query { " + " ".join(parts) + " }"

    async def fetch_recommendations_batch(
        self, ids: list[int], per_page: int = 15
    ) -> list[tuple[int, int, dict[str, Any]]]:
        """Recommendations for several media in one aliased query → (source_id, rating, target)."""
        data = await self._post(self._build_recommendations_batch_query(ids, per_page), {})
        results: list[tuple[int, int, dict[str, Any]]] = []
        for index, media_id in enumerate(ids):
            media = data.get(f"a{index}") or {}
            for node in (media.get("recommendations") or {}).get("nodes") or []:
                rating = node.get("rating")
                target = node.get("mediaRecommendation")
                if rating is not None and target is not None:
                    results.append((media_id, int(rating), target))
        return results

    async def fetch_relations_chunk(self, ids: list[int]) -> list[tuple[int, str, dict[str, Any]]]:
        data = await self._post(BULK_RELATIONS_QUERY, {"ids": ids, "perPage": len(ids)})
        edges: list[tuple[int, str, dict[str, Any]]] = []
        for media in (data.get("Page") or {}).get("media") or []:
            source_id = media.get("id")
            for edge in (media.get("relations") or {}).get("edges") or []:
                relation_type = edge.get("relationType")
                node = edge.get("node")
                if source_id is not None and relation_type and node:
                    edges.append((source_id, relation_type, node))
        return edges

    async def fetch_relations_bulk(
        self, ids: list[int], chunk_size: int = 4
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """Fetch `relations` for many media, batched; retries per-item on failure."""
        edges: list[tuple[int, str, dict[str, Any]]] = []
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            try:
                edges.extend(await self.fetch_relations_chunk(chunk))
            except AniListError as exc:
                logger.info("bulk relations chunk failed (%s) — retrying one-by-one", exc)
                for media_id in chunk:
                    try:
                        edges.extend(await self.fetch_relations_chunk([media_id]))
                    except AniListError as exc2:
                        logger.info("relations for #%s failed: %s", media_id, exc2)
        return edges


# --- Raw GraphQL → normalized flat dict -------------------------------------------------

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPOILER_MARKERS = re.compile(r"~!|!~")
_MULTI_WHITESPACE = re.compile(r"\s+")


def clean_description(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw
    text = _SPOILER_MARKERS.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = _HTML_TAG.sub(" ", text)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    text = html_mod.unescape(text)
    return _MULTI_WHITESPACE.sub(" ", text).strip()


def normalize_media(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw GraphQL `Media` object to the flat dict used everywhere else."""
    title = raw.get("title") or {}
    tags: list[str] = []
    for tag in raw.get("tags") or []:
        name = tag.get("name")
        if not name:
            continue
        if tag.get("isMediaSpoiler") and (tag.get("rank") or 0) < 60:
            continue
        tags.append((tag.get("rank") or 0, name))
    tags.sort(key=lambda pair: pair[0], reverse=True)

    studios = [
        node.get("name")
        for node in (raw.get("studios") or {}).get("nodes") or []
        if node.get("name")
    ]

    return {
        "id": raw.get("id"),
        "type": raw.get("type"),
        "title_romaji": title.get("romaji"),
        "title_english": title.get("english"),
        "title_native": title.get("native"),
        "synonyms": list(raw.get("synonyms") or []),
        "format": raw.get("format"),
        "status": raw.get("status"),
        "episodes": raw.get("episodes"),
        "duration": raw.get("duration"),
        "chapters": raw.get("chapters"),
        "volumes": raw.get("volumes"),
        "country_of_origin": raw.get("countryOfOrigin"),
        "average_score": raw.get("averageScore"),
        "popularity": raw.get("popularity"),
        "is_adult": bool(raw.get("isAdult")),
        "genres": list(raw.get("genres") or []),
        "tags": [name for _, name in tags],
        "studios": studios,
        "description": clean_description(raw.get("description")),
    }


def normalize_relation_node(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a relation target node to the denormalized fields stored in `media_relations`."""
    title = raw.get("title") or {}
    return {
        "id": raw.get("id"),
        "type": raw.get("type"),
        "format": raw.get("format"),
        "country": raw.get("countryOfOrigin"),
        "status": raw.get("status"),
        "episodes": raw.get("episodes"),
        "chapters": raw.get("chapters"),
        "volumes": raw.get("volumes"),
        "average_score": raw.get("averageScore"),
        "title_romaji": title.get("romaji"),
        "title_english": title.get("english"),
        "title_native": title.get("native"),
    }


def favourite_ids(user: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    favourites = user.get("favourites") or {}
    for key in ("anime", "manga"):
        connection = favourites.get(key) or {}
        for node in connection.get("nodes") or []:
            if node.get("id") is not None:
                ids.add(node["id"])
    return ids


def user_statistics(user: dict[str, Any]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    statistics = user.get("statistics") or {}
    for key in ("anime", "manga"):
        block = statistics.get(key) or {}
        stats[key] = {
            "count": block.get("count") or 0,
            "mean_score": block.get("meanScore") or 0.0,
            "std_dev": block.get("standardDeviation") or 0.0,
        }
    return stats
