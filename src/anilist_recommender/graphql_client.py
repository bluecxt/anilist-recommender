"""AniList GraphQL extraction.

Fetches the user profile/statistics, the full MediaListCollection (chunk
paginated) and the collaborative `recommendations` graph. All HTTP is done
through a small async client that throttles to respect AniList's rate limit
(90 req/min, currently degraded to ~30/min) and retries 429s with backoff.
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


# Shared fragment used for every `Media` object we pull down (both list entries
# and the `mediaRecommendation` targets of the collaborative graph).
MEDIA_FIELDS = """
  id
  type
  title { romaji english }
  format
  status
  episodes
  chapters
  volumes
  averageScore
  popularity
  genres
  studios(isMain: true) { nodes { name } }
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


class AniListClient:
    """Rate-limited async GraphQL client for graphql.anilist.co."""

    def __init__(self, min_interval: float = 0.85) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(90.0, connect=20.0),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "anilist-recommender/0.1 (personal CLI)",
            },
        )
        self._min_interval = max(0.0, min_interval)
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

    @staticmethod
    def _backoff_delay(headers: httpx.Headers, default: float) -> float:
        """Derive a retry delay from Retry-After / X-RateLimit-Reset headers."""
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

    async def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = {"query": query, "variables": variables}
        while True:
            await self._throttle()
            try:
                response = await self._client.post(ANILIST_GRAPHQL_URL, json=payload)
            except httpx.TransportError as exc:
                raise AniListError(f"network error while calling AniList: {exc}") from exc

            if response.status_code == 429:
                delay = self._backoff_delay(response.headers, 5.0)
                logger.warning("AniList HTTP 429 — backing off %.1fs", delay)
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 500:
                delay = self._backoff_delay(response.headers, 3.0)
                logger.warning("AniList HTTP %s — retrying in %.1fs", response.status_code, delay)
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
                    logger.warning("AniList rate limit (GraphQL) — backing off 5s")
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
        """Return every MediaList entry for a given type, chunk-paginated."""
        entries: list[dict[str, Any]] = []
        chunk = 0
        per_chunk = 500  # max allowed by the API
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
        """Collaborative recommendations for one media, sorted by net rating."""
        data = await self._post(
            RECOMMENDATIONS_QUERY, {"mediaId": media_id, "perPage": per_page}
        )
        media = data.get("Media") or {}
        recommendations = media.get("recommendations") or {}
        return recommendations.get("nodes") or []


# --- Raw GraphQL → normalized flat dict -------------------------------------------------

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_SPOILER_MARKERS = re.compile(r"~!|!~")
_MULTI_WHITESPACE = re.compile(r"\s+")


def clean_description(raw: str | None) -> str:
    """Strip AniList markdown/HTML/spoiler markers down to plain text."""
    if not raw:
        return ""
    text = raw
    text = _SPOILER_MARKERS.sub(" ", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = text.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    text = _HTML_TAG.sub(" ", text)
    # Markdown emphasis / strikethrough / code markers.
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    text = html_mod.unescape(text)
    return _MULTI_WHITESPACE.sub(" ", text).strip()


def normalize_media(raw: dict[str, Any]) -> dict[str, Any]:
    """Map a raw GraphQL `Media` object to the flat dict used everywhere else.

    Spoiler tags are dropped here (``isMediaSpoiler == True`` and ``rank < 60``)
    and the remaining tags are kept sorted by descending relevance.
    """
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
    tag_names = [name for _, name in tags]

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
        "format": raw.get("format"),
        "status": raw.get("status"),
        "episodes": raw.get("episodes"),
        "chapters": raw.get("chapters"),
        "volumes": raw.get("volumes"),
        "average_score": raw.get("averageScore"),
        "popularity": raw.get("popularity"),
        "genres": list(raw.get("genres") or []),
        "tags": tag_names,
        "studios": studios,
        "description": clean_description(raw.get("description")),
    }


def favourite_ids(user: dict[str, Any]) -> set[int]:
    """Union of favourite anime & manga ids from a `User` payload."""
    ids: set[int] = set()
    favourites = user.get("favourites") or {}
    for key in ("anime", "manga"):
        connection = favourites.get(key) or {}
        for node in connection.get("nodes") or []:
            if node.get("id") is not None:
                ids.add(node["id"])
    return ids


def user_statistics(user: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract `{anime: {count, mean_score, std_dev}, manga: {...}}`."""
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



