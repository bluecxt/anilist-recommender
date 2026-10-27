"""Local LLM agent — V2 multi-tool (search / franchise / cross-media).

Uses the official `openai` SDK pointed at Ollama. Three specialized tools are
exposed; each returns pre-digested rows and the model is forbidden from inventing
anything outside them. Output size is dynamic: detailed per-title analysis for
<=5 results, a full Markdown table/list for larger audits.
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any, Callable

from openai import OpenAI
from rich.markup import escape

from . import db as db_module
from . import embeddings as embeddings_module
from . import i18n
from .config import Settings, pick_title

MAX_TOOL_ROUNDS = 8


def _anilist_url(media_type: str | None, media_id: Any) -> str:
    kind = "manga" if media_type == "MANGA" else "anime"
    return f"https://anilist.co/{kind}/{media_id}"


def anilist_link(title: str, media_type: str | None, media_id: Any) -> str:
    """Wrap a title in an OSC-8 hyperlink (clickable — Ctrl+Click — in modern
    terminals: Konsole, GNOME Terminal, kitty, iTerm2, Windows Terminal, …),
    colored bold cyan so names stand out from the surrounding text."""
    if not media_id or not title:
        return title
    return f"[link={_anilist_url(media_type, media_id)}][bold cyan]{escape(title)}[/bold cyan][/link]"


def _heat_color(value: float) -> str:
    """Hex color on a red → yellow → green ramp, for a value in [0, 1]."""
    value = max(0.0, min(1.0, value))
    hue = value * 120.0  # 0° = red, 60° = yellow, 120° = green
    sat, light = 0.8, 0.45
    chroma = (1.0 - abs(2.0 * light - 1.0)) * sat
    x = chroma * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    m = light - chroma / 2.0
    r, g, b = (chroma, x, 0.0) if hue < 60.0 else (x, chroma, 0.0)
    return f"#{int(round((r + m) * 255)):02x}{int(round((g + m) * 255)):02x}{int(round((b + m) * 255)):02x}"


def _heat(value: float, lo: float, hi: float, text: str) -> str:
    """Wrap `text` in a red→yellow→green color for `value` within [lo, hi].

    Negative → red, neutral → yellow, positive → green (lo < 0 < hi makes the
    midpoint land on yellow). Rich strips these codes when output is piped, so
    files/logs stay clean.
    """
    if hi <= lo:
        return text
    color = _heat_color((value - lo) / (hi - lo))
    return f"[{color}]{text}[/{color}]"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_new_recommendations",
            "description": (
                "Search the local candidate database for NEW anime/manga/LN/manhwa to "
                "watch or read (planning list + collaborative-graph discoveries), ranked "
                "by GraphScore + semantic similarity + planning bonus. Use for any "
                "'recommend me something new' request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["ANIME", "MANGA", "LIGHT_NOVEL", "MANHWA", "ONE_SHOT", "ANY_PRINT"],
                        "description": "Media category. MANHWA covers Korean manhwa + Chinese manhua.",
                    },
                    "include_movies_or_ovas": {
                        "type": "boolean",
                        "description": "For ANIME: also allow MOVIE/OVA/SPECIAL and drop the >=3 episodes rule.",
                    },
                    "min_length": {"type": "integer", "description": "Min episodes (anime) / chapters (print)."},
                    "max_length": {"type": "integer", "description": "Max episodes (anime) / chapters (print)."},
                    "max_volumes": {"type": "integer", "description": "Max volumes (manga/LN only)."},
                    "source_pool": {
                        "type": "string",
                        "enum": ["PLANNING_ONLY", "DISCOVERY_ONLY", "BOTH"],
                        "description": "Restrict to planning list, graph discoveries, or both.",
                    },
                    "only_finished": {
                        "type": "boolean",
                        "description": "Only fully published (FINISHED) titles — useful for manga/LN.",
                    },
                    "required_genres_or_tags": {"type": "array", "items": {"type": "string"}},
                    "excluded_genres_or_tags": {"type": "array", "items": {"type": "string"}},
                    "semantic_query": {"type": "string", "description": "English mood/theme description for vector search."},
                    "limit": {"type": "integer", "description": "Number of results (default 5, up to 30)."},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_unseen_franchise_entries",
            "description": (
                "Find sequels, OVAs, movies, side-stories and spin-offs IN THE SAME MEDIUM as "
                "works the user already watched/read, whose follow-ups they have NOT completed. "
                "For a continuation that SWITCHES medium (anime → manga/LN), use "
                "find_cross_media_adaptations instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_type": {"type": "string", "enum": ["ANIME", "MANGA", "BOTH"]},
                    "relation_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relation types: SEQUEL, SIDE_STORY, SPIN_OFF, ADAPTATION, SOURCE…",
                    },
                    "target_formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter target formats, e.g. [\"OVA\",\"SPECIAL\",\"MOVIE\"] or [\"TV\"].",
                    },
                    "min_parent_score": {
                        "type": "integer",
                        "description": "Only from works the user scored >= this (0-100).",
                    },
                    "include_unreleased": {"type": "boolean", "description": "Include NOT_YET_RELEASED entries."},
                    "limit": {"type": "integer", "description": "Max results (default 25, up to 50)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_cross_media_adaptations",
            "description": (
                "Bridge between an anime and its manga/light-novel source. Use ONLY when the "
                "user has a work in one medium and wants its counterpart in the other. "
                "direction=ANIME_TO_PRINT means: 'I WATCHED an anime, find the manga/LN that is "
                "its SOURCE so I can read the continuation'. direction=PRINT_TO_ANIME means: "
                "'I READ a manga/LN, find its anime adaptation'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["ANIME_TO_PRINT", "PRINT_TO_ANIME"],
                        "description": (
                            "ANIME_TO_PRINT = the user WATCHED an ANIME and wants its MANGA/"
                            "LIGHT-NOVEL source (to read the continuation). PRINT_TO_ANIME = the "
                            "user READ a MANGA/LIGHT-NOVEL and wants its ANIME adaptation. If the "
                            "user names an ANIME and asks for its manga/LN, it is ALWAYS "
                            "ANIME_TO_PRINT — never the reverse."
                        ),
                    },
                    "target_print_format": {
                        "type": "string",
                        "enum": ["ANY", "MANGA", "NOVEL", "LIGHT_NOVEL"],
                        "description": "Target format. NOVEL / LIGHT_NOVEL = light novel (roman). MANGA = manga.",
                    },
                    "min_user_score": {
                        "type": "integer",
                        "description": (
                            "Filter by the SOURCE anime's user score (0-100). 0 = no score filter. "
                            "Only raise it if the user asks for 'mes préférés' / 'bien notés'."
                        ),
                    },
                    "only_finished_target": {
                        "type": "boolean",
                        "description": (
                            "Only show manga/LN targets that are fully published (FINISHED). "
                            "Leave false unless the user wants 'un LN terminé'. NOTE: the user "
                            "having FINISHED the anime is already handled — this only concerns "
                            "the manga/LN target's publication status."
                        ),
                    },
                    "limit": {"type": "integer", "description": "Max results (default 15, up to 40)."},
                },
                "required": ["direction"],
            },
        },
    },
]


def load_profile(settings: Settings) -> dict[str, Any]:
    path = settings.taste_profile_path
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def build_system_prompt(profile: dict[str, Any]) -> str:
    profile_json = json.dumps(profile, ensure_ascii=False)
    return (
        "You are a rigorous, clinical anime/manga/light-novel/manhwa recommendation engine "
        "running fully offline. You reason only from the local database and the taste profile "
        "below, across every medium: TV, movies, OVAs, manga, light novels, Korean manhwa and "
        "Chinese manhua.\n\n"
        "TOOL SELECTION — keyword priority (highest first):\n"
        "1. The words \"suite\", \"sequel\", \"continuation\", \"la suite\", \"lire la suite\" are a "
        "HARD signal. They NEVER map to search_new_recommendations.\n"
        "   - continuation in the SAME medium (anime sequel/OVA/movie/spin-off) → "
        "find_unseen_franchise_entries.\n"
        "   - continuation in ANOTHER medium (\"la suite est en manga / light novel / roman\", "
        "\"dispo uniquement en manga\") → find_cross_media_adaptations(direction=ANIME_TO_PRINT).\n"
        "2. \"conseille-moi\", \"je veux découvrir\", \"que regarder/lire\" WITHOUT any \"suite\" → "
        "search_new_recommendations.\n"
        "3. HAS a manga/LN, WANTS its anime adaptation → "
        "find_cross_media_adaptations(direction=PRINT_TO_ANIME).\n"
        "Rule of thumb: \"conseille-moi une suite …\" is about a CONTINUATION, not a discovery — "
        "use the franchise or cross-media tool, never search.\n"
        "DIRECTION (critical, never invert): if the user names an ANIME they finished/watched and "
        "asks for its manga/LN/roman → direction=ANIME_TO_PRINT. Use PRINT_TO_ANIME ONLY when the "
        "user names a MANGA/LN they READ and asks for its anime.\n\n"
        "STRICT RULES:\n"
        "1. Resolve every recommendation/continuation/adaptation request by calling exactly ONE "
        "tool — never write the list of titles yourself. Only answer directly for a meta question "
        "(greeting, 'how do you work').\n"
        "2. Call the tool ONCE with the best parameters. Keep filters minimal: leave min_user_score "
        "and only_finished_target unset unless the user asks for 'mes préférés' or 'un LN/manga terminé'.\n"
        "3. For search_new_recommendations, ALWAYS leave include_movies_or_ovas=false unless the "
        "user EXPLICITLY asks for films, OVAs, spéciaux or movies. Setting it to true without being "
        "asked floods the list with OVAs/shorts.\n"
        "4. Answer meta questions in FRENCH. Do not add preamble, disclaimer or 'j'espère que ça "
        "vous plaira' filler.\n\n"
        f"<taste_profile>\n{profile_json}\n</taste_profile>"
    )


def _length_label(media: dict[str, Any], lang: str = "fr") -> str:
    if media.get("type") == "ANIME":
        episodes = media.get("episodes")
        base = f"{episodes} {i18n.t(lang, 'eps')}" if episodes else f"? {i18n.t(lang, 'eps')}"
        season = media.get("season_number")
        total = media.get("franchise_episodes")
        extras = []
        if season and season > 1:
            extras.append(i18n.t(lang, "season", n=season))
        if total and (not episodes or total > episodes):
            extras.append(i18n.t(lang, "total_eps", n=total))
        return base if not extras else f"{base} · " + " · ".join(extras)
    chapters = media.get("chapters") or media.get("volumes")
    unit = i18n.t(lang, "chap") if media.get("chapters") else i18n.t(lang, "vol")
    return f"{chapters} {unit}" if chapters else i18n.t(lang, "unknown_len")


def _format_source(source: dict[str, Any], settings: Settings, colorize: bool = False) -> str:
    """Phrase one recommending source in the configured prose language, with the
    title in ``title_language`` (the DB stores the title in all three languages,
    never pre-formatted).

    Legacy guard: a plain string (pre-structured DB) is passed through as-is.
    With ``colorize=True`` the 0-100 score is heat-colored (red→green).
    """
    if isinstance(source, str):
        return source
    lang = settings.language
    title = pick_title(source, settings.title_language)
    if colorize:
        title = anilist_link(title, source.get("type"), source.get("media_id"))
    status = source.get("status")
    score = source.get("score")
    if status == "DROPPED":
        suffix = ", " + i18n.t(lang, "dropped")
    elif status == "PAUSED":
        suffix = ", " + i18n.t(lang, "paused")
    elif score and score > 0:
        score_text = str(int(score))
        if colorize:
            score_text = _heat(float(score), 0.0, 100.0, score_text)
        suffix = ", " + i18n.t(lang, "rated", score=score_text)
    else:
        suffix = ""
    return f"« {title} »{suffix}"


def _format_source_compact(source: dict[str, Any], settings: Settings, colorize: bool = True) -> str:
    """Compact form for a list: ``« Title » 85/100`` (score heat-colored).

    Dropped/paused sources carry a red status instead of a score — an unrated
    drop must not be shown as ``0/100``.
    """
    if isinstance(source, str):
        return source
    lang = settings.language
    title = pick_title(source, settings.title_language)
    if colorize:
        title = anilist_link(title, source.get("type"), source.get("media_id"))
    status = source.get("status")
    score = source.get("score")
    if status == "DROPPED":
        note = f"[red]{i18n.t(lang, 'dropped')}[/red]" if colorize else i18n.t(lang, "dropped")
    elif status == "PAUSED":
        note = f"[red]{i18n.t(lang, 'paused')}[/red]" if colorize else i18n.t(lang, "paused")
    elif score and score > 0:
        text = f"{int(score)}/100"
        if colorize:
            text = _heat(float(score), 0.0, 100.0, text)
        note = text
    else:
        note = ""
    return f"« {title} »" + (f" {note}" if note else "")


def rank_candidates(
    rows: list[dict[str, Any]],
    similarities: dict[int, float],
    has_query: bool,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Final blended score per candidate.

    Components (P2 #10 / #12):
    * graph_norm — rank-percentile of the amortized GraphScore (see scoring.py);
    * similarity — cosine vs the semantic query (only when the user asked one);
    * planning — bonus for titles already in the planning list, **decayed with
      a 2-year half-life** when the entry has been sleeping there untouched
      (``updated_at`` from AniList) — a 3-year-old plan is a weak signal;
    * quality — Bayesian-ish quality factor ``min(1, averageScore/75)``, neutral
      0.5 when the community score is unknown (unreleased/obscure).

    Weights: 0.45/0.30/0.10/0.15 with a semantic query, 0.75/0.15/0.10 without.
    """
    if not rows:
        return []
    scores = [row["graph_score"] or 0.0 for row in rows]
    # Rank-percentile normalization: Min-Max collapses a power-law GraphScore
    # distribution (2-3 mega-hits dwarf 95% of candidates near 0). Rank spreads the
    # filtered set uniformly over [0, 1] so the graph weight stays meaningful.
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    rank_norm = [0.0] * n
    if n > 1:
        for rank, idx in enumerate(order):
            rank_norm[idx] = rank / (n - 1)
    else:
        rank_norm[0] = 1.0 if scores[0] > 0 else 0.0

    now = time.time()
    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        graph_norm = rank_norm[index]
        similarity = similarities.get(row["media_id"], 0.0)
        planning = row["planning_bonus"] or 0.0
        decay = 1.0
        if planning > 0:
            updated = row.get("list_updated_at")
            if updated:
                age_years = max(0.0, (now - float(updated)) / 31_536_000)
                decay = 0.5 ** (age_years / 2.0)  # half-life: 2 years on the shelf
                planning = planning * decay
        average = row.get("average_score") or 0
        quality = min(1.0, float(average) / 75.0) if average else 0.5
        final = (
            graph_norm * 0.45 + similarity * 0.30 + planning * 0.10 + quality * 0.15
            if has_query
            else graph_norm * 0.75 + quality * 0.15 + planning * 0.10
        )
        ranked.append(
            {
                "media_id": row["media_id"],
                "title": pick_title(row, settings.title_language),
                "title_romaji": row["title_romaji"],
                "title_english": row["title_english"],
                "title_native": row["title_native"],
                "type": row["type"],
                "format": row["format"],
                "country": row["country_of_origin"],
                "genres": row["genres"],
                "tags": row["tags"],
                "description": row["description"],
                "length": _length_label(row, settings.language),
                "episodes": row["episodes"],
                "chapters": row["chapters"],
                "volumes": row["volumes"],
                "average_score": row["average_score"],
                "status": "IN_PLANNING" if planning > 0 else "DISCOVERY",
                "graph_score": round(row["graph_score"] or 0.0, 3),
                "semantic_similarity": round(similarity, 3) if has_query else None,
                "quality_factor": round(quality, 3),
                "planning_decay": round(decay, 3) if planning > 0 else None,
                "final_score": round(final, 4),
                "positive_sources": row["positive_sources"],
                "negative_warnings": row["negative_warnings"],
            }
        )
    ranked.sort(key=lambda item: -item["final_score"])
    return ranked


def _to_int(value: Any, default: int | None = None) -> int | None:
    """Coerce a tool arg to int, treating 'null'/'none'/'' and non-numerics as default.

    llama3.1 often emits ``"5"`` (quoted) or ``"null"`` (string) instead of the
    proper JSON ``5`` / ``null`` — passing those raw into SQL broke the filters.
    """
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "null", "none"):
            return default
        try:
            return int(float(s))
        except ValueError:
            return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    """Coerce a tool arg to bool, handling the strings 'true'/'false'/'1'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _to_str_list(value: Any) -> list[str]:
    """Coerce a tool arg to a list of strings; 'null'/'' → []. A bare string → [it]."""
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip().lower() not in ("", "null", "none"):
        return [value]
    return []


def _to_str(value: Any, default: str) -> str:
    """Coerce an enum-ish arg to str, mapping 'null'/'none'/'' to the default."""
    if value is None:
        return default
    s = str(value).strip()
    return default if s.lower() in ("", "null", "none") else s


def _clamp_limit(args: dict[str, Any], default: int, maximum: int) -> int:
    limit = _to_int(args.get("limit"), default)
    return max(1, min(limit, maximum))


_MOVIE_HINT = re.compile(
    r"\b(films?|movie|movies|ova|oav|sp[ée]cia(?:ux|les?)|special|specials|"
    r"court[- ]m[ée]trage)\b",
    re.IGNORECASE,
)


def _wants_movies(query: str) -> bool:
    """The model sometimes flips `include_movies_or_ovas` on without being asked
    (flooding results with OVAs/shorts). Deterministic guardrail: honor the flag
    only when the user's own request mentions films/OVAs/specials."""
    return bool(_MOVIE_HINT.search(query or ""))


_ADULT_HINT = re.compile(
    r"\b(hentai|adult|adulte|ecchi|nudit[ée]|18\+|sexuel(?:le)?s?|"
    r"[ée]rotique|porno?|hardcore)\b",
    re.IGNORECASE,
)


def _wants_adult(query: str) -> bool:
    """Detect an explicit adult-content request (only honoured when allow_adult)."""
    return bool(_ADULT_HINT.search(query or ""))


def _search_new(
    args: dict[str, Any], db: Any, settings: Settings, query: str = ""
) -> list[dict[str, Any]]:
    # Adult content is often OVA/short format and lives in BOTH anime and manga, so
    # an explicit adult request widens the format/category filters and pins the pool
    # to is_adult titles (AniList has no 'hentai' genre the model can pick, and the
    # embedding of 'hentai' alone would otherwise latch onto dark manga like Berserk).
    adult_request = settings.allow_adult and _wants_adult(query)
    if adult_request:
        category = "ANY"
        include_movies = True
        min_length = max_length = max_volumes = None
        required = excluded = None
        adult_only = True
    else:
        category = _to_str(args.get("category"), "ANIME")
        include_movies = _to_bool(args.get("include_movies_or_ovas")) and _wants_movies(query)
        min_length = _to_int(args.get("min_length"))
        max_length = _to_int(args.get("max_length"))
        max_volumes = _to_int(args.get("max_volumes"))
        required = _to_str_list(args.get("required_genres_or_tags"))
        excluded = _to_str_list(args.get("excluded_genres_or_tags"))
        adult_only = False
    rows = db_module.search_new_recommendations(
        db,
        category=category,
        include_movies_or_ovas=include_movies,
        min_length=min_length,
        max_length=max_length,
        max_volumes=max_volumes,
        source_pool=_to_str(args.get("source_pool"), "BOTH"),
        only_finished=_to_bool(args.get("only_finished")),
        required=required,
        excluded=excluded,
        limit=_clamp_limit(args, 10, 30),
        allow_adult=settings.allow_adult,
        adult_only=adult_only,
    )
    semantic = (args.get("semantic_query") or "").strip()
    if adult_request and not semantic:
        # The model has no adult genre/tag to select, so embed the user's own
        # request as the vector query to actually surface adult titles.
        semantic = query
    has_query = bool(semantic)
    similarities: dict[int, float] = {}
    if has_query:
        ids = [row["media_id"] for row in rows]
        similarities = embeddings_module.query_similarities(db, semantic, ids, settings)
    return rank_candidates(rows, similarities, has_query, settings)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_annotations_json(text: str, count: int) -> dict[int, str]:
    """Parse a strict-JSON annotation answer: ``{"1": "phrase", ...}`` or a list
    of ``{"index"/"id", "reason"/"why"/"text"}`` objects. Empty dict when the
    model ignored the JSON contract — the caller falls back to line parsing."""
    try:
        parsed: Any = _load_json_like(_strip_fences(text))
    except (ValueError, SyntaxError):
        return {}

    def _sentence(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("reason", "why", "text", "phrase"):
                if isinstance(value.get(key), str) and value[key].strip():
                    return value[key].strip()
        return ""

    result: dict[int, str] = {}
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            sentence = _sentence(value)
            if 1 <= index <= count and sentence:
                result[index] = sentence
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index") or item.get("id") or item.get("num"))
            except (TypeError, ValueError):
                continue
            sentence = _sentence(item)
            if 1 <= index <= count and sentence:
                result[index] = sentence
    return result


def _annotation_llm(client: Any, settings: Settings, prompt: str) -> str:
    messages = [
        {"role": "system", "content": i18n.t(settings.language, "annotation_system")},
        {"role": "user", "content": prompt},
    ]
    options = {"options": {"num_ctx": 8192, "temperature": 0.3}}
    try:
        # Structured outputs (json_object): Ollama ≥0.5 maps this to `format: json`.
        response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body=options,
        )
    except Exception:
        # Older Ollama or a model that rejects the field — retry without it.
        response = client.chat.completions.create(
            model=settings.ollama_model, messages=messages, extra_body=options
        )
    return (response.choices[0].message.content or "").strip()


def _annotate_search(
    items: list[dict[str, Any]], client: Any, settings: Settings
) -> dict[int, str]:
    """One short 'why' sentence per recommendation, grounded in the data.

    The LLM only phrases the reason; titles/formats stay rendered by the code, so a
    failure here never breaks the list — the caller falls back to `positive_sources`.
    JSON mode is requested and validated (P1 #8b); line-by-line parsing is the
    fallback when the model refuses the JSON contract.
    """
    if not items:
        return {}
    lang = settings.language
    profile = load_profile(settings)
    top_genres = ", ".join(g["name"] for g in (profile.get("top_genres") or [])[:6])
    top_tags = ", ".join(t["name"] for t in (profile.get("top_tags") or [])[:6])

    bullets = []
    for index, item in enumerate(items[:10], start=1):
        genres = ", ".join(item.get("genres") or [])[:120]
        tags = ", ".join((item.get("tags") or [])[:6])
        sources = " · ".join(
            _format_source(s, settings) for s in (item.get("positive_sources") or [])
        )[:160] or "—"
        synopsis = (item.get("description") or "").strip()[:140]
        bullets.append(
            f"{index}. {item['title']} | {item.get('format')} {item['length']} | "
            f"genres: {genres} | tags: {tags} | {i18n.t(lang, 'proche_de')}: {sources} | "
            f"synopsis: {synopsis}"
        )

    prompt = (
        i18n.t(lang, "annotation_instruction")
        + "\n\n"
        + f"{i18n.t(lang, 'preferred_genres')} : {top_genres}\n"
        + f"{i18n.t(lang, 'preferred_tags')} : {top_tags}\n\n"
        + "\n".join(bullets)
        + "\n\n"
        + i18n.t(lang, "annotation_json")
    )

    try:
        text = _annotation_llm(client, settings, prompt)
    except Exception:
        return {}

    annotations = _parse_annotations_json(text, len(items))
    if not annotations:
        # Fallback: line-by-line "1. phrase" when the model ignored JSON mode.
        for line in text.splitlines():
            match = re.match(r"^\s*(\d+)\s*[.):\-]\s*(.+)$", line.strip())
            if not match:
                continue
            index = int(match.group(1))
            sentence = match.group(2).strip()
            if 1 <= index <= len(items) and sentence:
                annotations[index] = sentence

    # Strip a leading "Title | " echo if the model repeated the input title.
    cleaned: dict[int, str] = {}
    for index, sentence in annotations.items():
        if " | " in sentence:
            head, _, rest = sentence.partition(" | ")
            if head.strip() == items[index - 1]["title"]:
                sentence = rest.strip()
        if sentence:
            cleaned[index] = sentence[:200]
    return cleaned


def _render_search(
    items: list[dict[str, Any]], annotations: dict[int, str], has_query: bool, settings: Settings
) -> str:
    lang = settings.language
    lines = [i18n.t(lang, "recommendations_header", n=len(items)) + "\n"]
    for index, item in enumerate(items, start=1):
        meta = " · ".join(b for b in (item.get("format"), item["length"]) if b)
        if has_query and item.get("semantic_similarity") is not None:
            sim = item["semantic_similarity"]
            meta += f" · {i18n.t(lang, 'sim')} {_heat(sim, 0.0, 1.0, f'{sim:.2f}')}"
        score = item["final_score"]
        lines.append(
            f"{index}. {anilist_link(item['title'], item.get('type'), item.get('media_id'))} "
            f"— {meta} · {i18n.t(lang, 'score')} {_heat(score, 0.0, 1.0, f'{score:.2f}')}"
        )
        why = annotations.get(index)
        if why:
            lines.append(f"   → {why}")
        sources = list(item.get("positive_sources") or [])
        sources += item.get("negative_warnings") or []
        if sources:
            # Highest user score first (strongest positive evidence), unrated last.
            sources.sort(key=lambda s: -(s.get("score") or 0))
            proches = " · ".join(_format_source_compact(s, settings) for s in sources[:6])
            lines.append(f"   {i18n.t(lang, 'proche_de')} : {proches}")
        if index < len(items):
            lines.append("")
    return "\n".join(lines)


def _fmt_len(episodes: Any, chapters: Any, volumes: Any, lang: str = "fr") -> str:
    if episodes:
        return f"{episodes} {i18n.t(lang, 'eps')}"
    if chapters:
        return f"{chapters} {i18n.t(lang, 'chap')}"
    if volumes:
        return f"{volumes} {i18n.t(lang, 'vol')}"
    return ""


def _stars(weight: Any) -> str:
    w = float(weight or 0.0)
    if w < 0:
        return "☆"
    if w >= 2.0:
        return "⭐⭐⭐⭐⭐"
    if w >= 1.2:
        return "⭐⭐⭐⭐"
    if w >= 0.5:
        return "⭐⭐⭐"
    if w >= 0.1:
        return "⭐⭐"
    return "⭐"


def _target_title(row: dict[str, Any], language: str) -> str:
    title = pick_title(
        {
            "id": row.get("target_id"),
            "title_romaji": row.get("target_title_romaji"),
            "title_english": row.get("target_title_english"),
            "title_native": row.get("target_title_native"),
        },
        language,
    )
    return anilist_link(title, row.get("target_type"), row.get("target_id"))


def _source_title(row: dict[str, Any], language: str) -> str:
    title = pick_title(
        {
            "id": row.get("source_id"),
            "title_romaji": row.get("source_title_romaji"),
            "title_english": row.get("source_title_english"),
            "title_native": row.get("source_title_native"),
        },
        language,
    )
    return anilist_link(title, row.get("source_type"), row.get("source_id"))


def _target_meta(row: dict[str, Any], lang: str = "fr") -> str:
    bits = [
        b
        for b in (
            row.get("target_format"),
            _fmt_len(row.get("target_episodes"), row.get("target_chapters"), row.get("target_volumes"), lang),
            i18n.status(lang, row.get("target_status") or ""),
        )
        if b
    ]
    meta = " · ".join(bits)
    score = row.get("target_score")
    if score is not None:
        meta += " · " + i18n.t(lang, "note", score=score)
    return meta


def _group_by_source(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    index: dict[Any, dict[str, Any]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if source_id not in index:
            index[source_id] = {"source": row, "targets": []}
            groups.append(index[source_id])
        index[source_id]["targets"].append(row)
    return groups


def _format_franchise(rows: list[dict[str, Any]], settings: Settings) -> str:
    lang = settings.language
    title_lang = settings.title_language
    groups = _group_by_source(rows)
    lines = [i18n.t(lang, "franchise_header", n=len(rows)) + "\n"]
    for group in groups:
        source = group["source"]
        targets = group["targets"]
        src_title = _source_title(source, title_lang)
        stars = _stars(source.get("parent_weight"))
        if len(targets) == 1:
            target = targets[0]
            rel = i18n.relation(lang, target.get("relation_type") or "")
            lines.append(f"- {_target_title(target, title_lang)} — {rel} · {_target_meta(target, lang)} · {stars}")
            lines.append(f"  ↳ {i18n.t(lang, 'from_source')} : {src_title}")
        else:
            lines.append(f"{src_title} — {stars} {i18n.t(lang, 'affinity')}")
            for target in targets:
                rel = i18n.relation(lang, target.get("relation_type") or "")
                lines.append(f"  • {_target_title(target, title_lang)} — {rel} · {_target_meta(target, lang)}")
    return "\n".join(lines)


def _format_cross_media(rows: list[dict[str, Any]], settings: Settings, direction: str) -> str:
    lang = settings.language
    title_lang = settings.title_language
    groups = _group_by_source(rows)
    link = i18n.t(lang, "sequel_of") if direction == "ANIME_TO_PRINT" else i18n.t(lang, "adapted_from")
    if direction == "ANIME_TO_PRINT":
        header = i18n.t(lang, "anime_to_print_header", n=len(rows))
    else:
        header = i18n.t(lang, "print_to_anime_header", n=len(rows))
    lines = [header + "\n"]
    for group in groups:
        source = group["source"]
        targets = group["targets"]
        src_title = _source_title(source, title_lang)
        stars = _stars(source.get("parent_weight"))
        if len(targets) == 1:
            target = targets[0]
            lines.append(f"- {_target_title(target, title_lang)} — {_target_meta(target, lang)} · {stars}")
            lines.append(f"  ↳ {link} : {src_title}")
        else:
            lines.append(f"{src_title} — {stars} {i18n.t(lang, 'affinity')} · {len(targets)} {i18n.t(lang, 'titles')}")
            for target in targets:
                lines.append(f"  • {_target_title(target, title_lang)} — {_target_meta(target, lang)}")
    return "\n".join(lines)


def _franchise(args: dict[str, Any], db: Any, settings: Settings) -> str:
    rows = db_module.find_unseen_franchise_entries(
        db,
        source_type=_to_str(args.get("source_type"), "ANIME"),
        relation_types=_to_str_list(args.get("relation_types")) or ["SEQUEL", "SIDE_STORY"],
        target_formats=_to_str_list(args.get("target_formats")) or None,
        min_parent_score=_to_int(args.get("min_parent_score"), 0),
        include_unreleased=_to_bool(args.get("include_unreleased")),
        limit=_clamp_limit(args, 25, 50),
    )
    if not rows:
        return i18n.t(settings.language, "no_sequels")
    return _format_franchise(rows, settings)


def _adaptations(args: dict[str, Any], db: Any, settings: Settings) -> str:
    fmt = _to_str(args.get("target_print_format"), "ANY").upper()
    if fmt in ("LIGHT_NOVEL", "LN"):
        fmt = "NOVEL"
    direction = _to_str(args.get("direction"), "ANIME_TO_PRINT")
    rows = db_module.find_cross_media_adaptations(
        db,
        direction=direction,
        target_print_format=fmt,
        min_user_score=_to_int(args.get("min_user_score"), 0),
        only_finished_target=_to_bool(args.get("only_finished_target")),
        limit=_clamp_limit(args, 15, 40),
    )
    if not rows:
        return i18n.t(settings.language, "no_results")
    return _format_cross_media(rows, settings, direction)


def execute_tool(
    name: str, args: dict[str, Any], settings: Settings, query: str = ""
) -> str | list[dict[str, Any]]:
    connection = db_module.connect(settings.db_path)
    try:
        if name == "search_new_recommendations":
            return _search_new(args, connection, settings, query)
        if name == "find_unseen_franchise_entries":
            return _franchise(args, connection, settings)
        if name == "find_cross_media_adaptations":
            return _adaptations(args, connection, settings)
        return [{"error": f"unknown tool: {name}"}]
    finally:
        connection.close()


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced ``{…}`` JSON object found in free text.

    Recovers tool calls the model wrapped in prose (``Voici l'appel : {…}.``)
    where a plain ``json.loads`` of the whole content fails.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _load_json_like(text: str) -> Any:
    """Parse a JSON-ish string, tolerating Python-style ``True``/``False``/``None``.

    llama3.1 frequently emits ``False`` instead of ``false`` — strict ``json.loads``
    rejects it, so we fall back to ``ast.literal_eval`` (safe, never executes code).
    """
    try:
        return json.loads(text)
    except ValueError:
        return ast.literal_eval(text)  # raises ValueError / SyntaxError on failure


def _parse_text_tool_call(content: str) -> tuple[str | None, dict[str, Any]]:
    """Recover a tool call emitted as a JSON string instead of native `tool_calls`.

    Some Ollama models dump the call into `message.content` as e.g.
    `{"name": "...", "parameters": {...}}` (or `"arguments"`), possibly wrapped
    in prose or code fences, and often with Python booleans (`False`). Parse it so
    the request still executes instead of printing the raw JSON.
    """
    text = (content or "").strip()
    if not text:
        return None, {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    data: Any = None
    for candidate in (text, _extract_json_object(text)):
        if not candidate:
            continue
        try:
            data = _load_json_like(candidate)
            break
        except (ValueError, SyntaxError):
            continue

    if isinstance(data, dict) and data.get("name"):
        params = data.get("parameters") or data.get("arguments") or {}
        if isinstance(params, str):
            try:
                params = _load_json_like(params)
            except (ValueError, SyntaxError):
                params = {}
        if isinstance(params, dict):
            return str(data["name"]), params
    return None, {}


def run_agent(
    query: str,
    settings: Settings,
    history: list[dict[str, Any]] | None = None,
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
) -> str:
    # A request for adult content without `allow_adult` would only surface unrelated
    # titles (the is_adult exclusion empties the pool, so the vector search latches
    # onto dark manga instead). Refuse early with an actionable hint.
    if _wants_adult(query) and not settings.allow_adult:
        return i18n.t(settings.language, "adult_disabled")

    client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama", timeout=180.0)
    system_prompt = build_system_prompt(load_profile(settings))

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=messages,
            tools=TOOLS,
            extra_body={"options": {"num_ctx": 8192, "temperature": 0.2}},
        )
        message = response.choices[0].message
        raw_content = message.content or ""

        # Resolve the tool call(s): native `message.tool_calls`, else a JSON-string fallback.
        calls: list[tuple[str, str, dict[str, Any]]] = []
        native = bool(message.tool_calls)
        if native:
            for index, call in enumerate(message.tool_calls):
                call_id = getattr(call, "id", None) or f"call_{index}"
                try:
                    args = json.loads(call.function.arguments or "{}")
                except ValueError:
                    args = {}
                calls.append((call_id, call.function.name, args))
        else:
            name, args = _parse_text_tool_call(raw_content)
            if name:
                calls.append(("call_0", name, args))

        if not calls:
            return raw_content.strip()

        if native:
            messages.append(
                {
                    "role": "assistant",
                    "content": raw_content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for call_id, name, args in calls
                    ],
                }
            )
        else:
            messages.append({"role": "assistant", "content": raw_content})

        outputs: list[str] = []
        for call_id, name, args in calls:
            if on_tool_call is not None:
                on_tool_call(name, args)
            result = execute_tool(name, args, settings, query)
            if name == "search_new_recommendations" and isinstance(result, list):
                if not result:
                    content = i18n.t(settings.language, "no_results")
                else:
                    has_query = any(item.get("semantic_similarity") is not None for item in result)
                    annotations = _annotate_search(result, client, settings)
                    content = _render_search(result, annotations, has_query, settings)
            else:
                content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            outputs.append(content)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

        # The tool's output IS the answer. The LLM only selects the tool and its
        # parameters; it never re-summarizes the list — that round-trip caused every synthesis
        # bug (JSON-explaining, truncation, hallucination, filler). Show the tool output directly.
        return "\n\n".join(outputs)

    return "No final answer produced — the model kept issuing tool calls."
