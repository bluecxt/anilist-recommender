"""Local LLM agent — V2 multi-tool (search / franchise / cross-media).

Uses the official `openai` SDK pointed at Ollama. Three specialized tools are
exposed; each returns pre-digested rows and the model is forbidden from inventing
anything outside them. Output size is dynamic: detailed per-title analysis for
<=5 results, a full Markdown table/list for larger audits.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from . import db as db_module
from . import embeddings as embeddings_module
from .config import Settings

MAX_TOOL_ROUNDS = 8

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
                            "ANIME_TO_PRINT: user WATCHED an anime, wants its manga/LN source "
                            "(to read the continuation). PRINT_TO_ANIME: user READ a manga/LN, "
                            "wants its anime adaptation."
                        ),
                    },
                    "target_print_format": {"type": "string", "enum": ["ANY", "MANGA", "NOVEL"]},
                    "min_user_score": {"type": "integer", "description": "Only works the user scored >= this (default 75)."},
                    "only_finished_target": {"type": "boolean", "description": "Only fully published targets."},
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
        "TOOL SELECTION — decide by what the user HAS and what they WANT:\n"
        "- WANTS new titles (\"conseille-moi\", \"je veux découvrir\") → search_new_recommendations.\n"
        "- HAS an anime, WANTS its sequel/OVA/movie/spin-off IN ANIME → "
        "find_unseen_franchise_entries(source_type=ANIME).\n"
        "- HAS an anime, WANTS the manga/LN that CONTINUES the story (\"la suite est en manga / "
        "light novel / roman\") → find_cross_media_adaptations(direction=ANIME_TO_PRINT).\n"
        "- HAS a manga/LN, WANTS its anime adaptation → "
        "find_cross_media_adaptations(direction=PRINT_TO_ANIME).\n"
        "Rule of thumb: if \"manga\" or \"light novel\" designates WHERE THE STORY CONTINUES "
        "(not a side bonus), use find_cross_media_adaptations with direction=ANIME_TO_PRINT.\n\n"
        "STRICT RULES:\n"
        "1. LANGUAGE (critical): answer in the EXACT language the user wrote in (French question "
        "→ French answer). NEVER answer in Chinese or English, even though the titles are "
        "Japanese and the JSON fields are English.\n"
        "2. Never invent a title — recommend ONLY what a tool returns.\n"
        "3. Call each tool ONCE with your best parameters, then write the final answer. Do not "
        "re-call the same tool just to tweak filters.\n"
        "4. OUTPUT SIZE (critical):\n"
        "   - <=5 results: analyze each one in depth — cite graph_score, positive_sources, "
        "negative_warnings, exact format and length.\n"
        "   - >5 results: full Markdown table or exhaustive list, never truncate.\n"
        "5. Presentation: for franchise/cross-media results, one line per item as "
        "\"SOURCE → TARGET (relation, format, length, status, avg score)\".\n"
        "6. Be direct, analytical, zero filler.\n"
        "7. If a tool returns nothing, say so plainly and suggest looser filters.\n\n"
        f"<taste_profile>\n{profile_json}\n</taste_profile>"
    )


def _length_label(media: dict[str, Any]) -> str:
    if media.get("type") == "ANIME":
        episodes = media.get("episodes")
        return f"{episodes} eps" if episodes else "? eps"
    chapters = media.get("chapters") or media.get("volumes")
    unit = "chap" if media.get("chapters") else "vol"
    return f"{chapters} {unit}" if chapters else "? length"


def rank_candidates(
    rows: list[dict[str, Any]],
    similarities: dict[int, float],
    has_query: bool,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    scores = [row["graph_score"] or 0.0 for row in rows]
    gmin, gmax = min(scores), max(scores)
    span = gmax - gmin

    def norm(value: float) -> float:
        if span < 1e-9:
            return 1.0 if gmax > 0 else 0.0
        return (value - gmin) / span

    ranked: list[dict[str, Any]] = []
    for row in rows:
        graph_norm = norm(row["graph_score"] or 0.0)
        similarity = similarities.get(row["media_id"], 0.0)
        planning = row["planning_bonus"] or 0.0
        final = graph_norm * 0.55 + similarity * 0.35 + planning * 0.10 if has_query else (
            graph_norm * 0.85 + planning * 0.15
        )
        ranked.append(
            {
                "title": row["title_romaji"] or row["title_english"] or f"#{row['media_id']}",
                "title_romaji": row["title_romaji"],
                "title_english": row["title_english"],
                "type": row["type"],
                "format": row["format"],
                "country": row["country_of_origin"],
                "length": _length_label(row),
                "episodes": row["episodes"],
                "chapters": row["chapters"],
                "volumes": row["volumes"],
                "average_score": row["average_score"],
                "status": "IN_PLANNING" if planning > 0 else "DISCOVERY",
                "graph_score": round(row["graph_score"] or 0.0, 3),
                "semantic_similarity": round(similarity, 3) if has_query else None,
                "final_score": round(final, 4),
                "positive_sources": row["positive_sources"],
                "negative_warnings": row["negative_warnings"],
            }
        )
    ranked.sort(key=lambda item: -item["final_score"])
    return ranked


def _clamp_limit(args: dict[str, Any], default: int, maximum: int) -> int:
    try:
        limit = int(args.get("limit") or default)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(limit, maximum))


def _search_new(args: dict[str, Any], db: Any, settings: Settings) -> list[dict[str, Any]]:
    rows = db_module.search_new_recommendations(
        db,
        category=args.get("category") or "ANIME",
        include_movies_or_ovas=bool(args.get("include_movies_or_ovas")),
        min_length=args.get("min_length"),
        max_length=args.get("max_length"),
        max_volumes=args.get("max_volumes"),
        source_pool=args.get("source_pool") or "BOTH",
        only_finished=bool(args.get("only_finished")),
        required=args.get("required_genres_or_tags"),
        excluded=args.get("excluded_genres_or_tags"),
        limit=_clamp_limit(args, 10, 30),
    )
    query = (args.get("semantic_query") or "").strip()
    has_query = bool(query)
    similarities: dict[int, float] = {}
    if has_query:
        ids = [row["media_id"] for row in rows]
        similarities = embeddings_module.query_similarities(db, query, ids, settings)
    return rank_candidates(rows, similarities, has_query)


def _franchise(args: dict[str, Any], db: Any) -> list[dict[str, Any]]:
    rows = db_module.find_unseen_franchise_entries(
        db,
        source_type=args.get("source_type") or "ANIME",
        relation_types=args.get("relation_types") or ["SEQUEL", "SIDE_STORY"],
        target_formats=args.get("target_formats"),
        min_parent_score=int(args.get("min_parent_score") or 0),
        include_unreleased=bool(args.get("include_unreleased")),
        limit=_clamp_limit(args, 25, 50),
    )
    return [
        {
            "source_title": r.get("source_title_romaji") or r.get("source_title_english"),
            "source_format": r.get("source_format"),
            "my_score": r.get("parent_score"),
            "my_affinity": round(r.get("parent_weight") or 0.0, 2),
            "relation_type": r.get("relation_type"),
            "target_title": r.get("target_title"),
            "target_type": r.get("target_type"),
            "target_format": r.get("target_format"),
            "target_country": r.get("target_country"),
            "target_status": r.get("target_status"),
            "target_episodes": r.get("target_episodes"),
            "target_chapters": r.get("target_chapters"),
            "target_volumes": r.get("target_volumes"),
            "target_score": r.get("target_score"),
        }
        for r in rows
    ]


def _adaptations(args: dict[str, Any], db: Any) -> list[dict[str, Any]]:
    rows = db_module.find_cross_media_adaptations(
        db,
        direction=args.get("direction") or "ANIME_TO_PRINT",
        target_print_format=args.get("target_print_format") or "ANY",
        min_user_score=int(args.get("min_user_score") or 75),
        only_finished_target=bool(args.get("only_finished_target")),
        limit=_clamp_limit(args, 15, 40),
    )
    return [
        {
            "source_title": r.get("source_title_romaji") or r.get("source_title_english"),
            "source_format": r.get("source_format"),
            "my_score": r.get("parent_score"),
            "my_affinity": round(r.get("parent_weight") or 0.0, 2),
            "relation_type": r.get("relation_type"),
            "target_title": r.get("target_title"),
            "target_format": r.get("target_format"),
            "target_status": r.get("target_status"),
            "target_episodes": r.get("target_episodes"),
            "target_chapters": r.get("target_chapters"),
            "target_volumes": r.get("target_volumes"),
            "target_score": r.get("target_score"),
        }
        for r in rows
    ]


def execute_tool(name: str, args: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    connection = db_module.connect(settings.db_path)
    try:
        if name == "search_new_recommendations":
            return _search_new(args, connection, settings)
        if name == "find_unseen_franchise_entries":
            return _franchise(args, connection)
        if name == "find_cross_media_adaptations":
            return _adaptations(args, connection)
        return [{"error": f"unknown tool: {name}"}]
    finally:
        connection.close()


def run_agent(
    query: str,
    settings: Settings,
    history: list[dict[str, Any]] | None = None,
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
) -> str:
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

        if not message.tool_calls:
            return (message.content or "").strip()

        tool_calls = []
        for index, call in enumerate(message.tool_calls):
            tool_calls.append(
                {
                    "id": getattr(call, "id", None) or f"call_{index}",
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
            )
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": tool_calls})

        for index, call in enumerate(message.tool_calls):
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = {}
            if on_tool_call is not None:
                on_tool_call(call.function.name, args)
            result = execute_tool(call.function.name, args, settings)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", None) or f"call_{index}",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return "No final answer produced — the model kept issuing tool calls."
