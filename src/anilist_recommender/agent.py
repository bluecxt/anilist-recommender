"""Local LLM agent: strict tool-calling against SQLite + profile injection.

Uses the official `openai` SDK pointed at Ollama's OpenAI-compatible endpoint.
The only tool exposed to the model is ``search_candidates``, which runs the
hybrid ranking (hard SQL filters + cosine similarity + GraphScore) in Python
and returns pre-digested candidates — the model is forbidden from inventing
anything outside that set.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI

from . import db as db_module
from . import embeddings as embeddings_module
from .config import Settings

MAX_TOOL_ROUNDS = 6

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": (
                "Search the local candidate database (the user's PLANNING list plus "
                "collaborative-graph discoveries) and return ranked recommendations. "
                "Use this for ANY recommendation request; never recommend a title "
                "that is not in the returned list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "media_type": {
                        "type": "string",
                        "enum": ["ANIME", "MANGA"],
                        "description": "Restrict to anime or manga.",
                    },
                    "min_episodes": {
                        "type": "integer",
                        "description": "Minimum episode count (anime only).",
                    },
                    "max_episodes": {
                        "type": "integer",
                        "description": "Maximum episode count (anime only).",
                    },
                    "min_chapters": {
                        "type": "integer",
                        "description": "Minimum chapter count (manga only).",
                    },
                    "max_chapters": {
                        "type": "integer",
                        "description": "Maximum chapter count (manga only).",
                    },
                    "source_pool": {
                        "type": "string",
                        "enum": ["PLANNING_ONLY", "DISCOVERY_ONLY", "BOTH"],
                        "description": (
                            "PLANNING_ONLY = only titles already on the user's planning list; "
                            "DISCOVERY_ONLY = only collaborative-graph discoveries; BOTH = both."
                        ),
                    },
                    "required_genres_or_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Candidate must contain at least one of these genres/tags "
                            "(English AniList names, e.g. 'Psychological', 'Thriller')."
                        ),
                    },
                    "excluded_genres_or_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude candidates containing any of these genres/tags.",
                    },
                    "semantic_query": {
                        "type": "string",
                        "description": (
                            "Rich English description of the desired mood, protagonist type or "
                            "themes for the vector search (e.g. 'cerebral cat-and-mouse thriller "
                            "with a morally grey protagonist')."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of candidates to return (default 10).",
                    },
                },
                "required": ["media_type"],
            },
        },
    }
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
        "You are a rigorous, clinical anime & manga recommendation engine running fully "
        "offline. You reason only from the local candidate database and the taste profile "
        "injected below.\n\n"
        "STRICT RULES:\n"
        "1. Never invent a title. Recommend ONLY works returned by search_candidates.\n"
        "2. Give 3 to 5 recommendations. For each: justify WHY it fits by citing "
        "positive_sources (the graph bridges from works the user loved), state the length, "
        "and honestly flag any negative_warnings (bridges from dropped/hated works).\n"
        "3. Be direct, analytical, zero filler. No marketing tone.\n"
        "4. Answer in the language the user wrote in.\n"
        "5. If search_candidates returns nothing, say so plainly and suggest looser filters.\n\n"
        f"<taste_profile>\n{profile_json}\n</taste_profile>"
    )


def _length_label(media: dict[str, Any]) -> str:
    if media.get("type") == "ANIME":
        episodes = media.get("episodes")
        return f"{episodes} eps" if episodes else "? eps"
    chapters = media.get("chapters")
    return f"{chapters} chap" if chapters else "? chap"


def rank_candidates(
    rows: list[dict[str, Any]],
    similarities: dict[int, float],
    has_query: bool,
) -> list[dict[str, Any]]:
    """FinalScore = Norm(GraphScore)*0.55 + Sim*0.35 + PlanningBonus*0.10.

    When no semantic query is supplied, the weights are renormalised to
    GraphScore (0.85) and PlanningBonus (0.15).
    """
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
        if has_query:
            final = graph_norm * 0.55 + similarity * 0.35 + planning * 0.10
        else:
            final = graph_norm * 0.85 + planning * 0.15

        ranked.append(
            {
                "title": row["title_romaji"] or row["title_english"] or f"#{row['media_id']}",
                "title_romaji": row["title_romaji"],
                "title_english": row["title_english"],
                "type": row["type"],
                "format": row["format"],
                "length": _length_label(row),
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


def execute_search(args: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    """The tool implementation: hard filters → semantic sim → hybrid rank."""
    connection = db_module.connect(settings.db_path)
    try:
        rows = db_module.search_candidates(
            connection,
            media_type=args.get("media_type"),
            min_episodes=args.get("min_episodes"),
            max_episodes=args.get("max_episodes"),
            min_chapters=args.get("min_chapters"),
            max_chapters=args.get("max_chapters"),
            source_pool=args.get("source_pool") or "BOTH",
            required=args.get("required_genres_or_tags"),
            excluded=args.get("excluded_genres_or_tags"),
        )
        query = (args.get("semantic_query") or "").strip()
        has_query = bool(query)
        similarities: dict[int, float] = {}
        if has_query:
            ids = [row["media_id"] for row in rows]
            similarities = embeddings_module.query_similarities(connection, query, ids, settings)

        ranked = rank_candidates(rows, similarities, has_query)
        try:
            limit = max(1, int(args.get("limit") or 10))
        except (TypeError, ValueError):
            limit = 10
        return ranked[:limit]
    finally:
        connection.close()


def run_agent(
    query: str,
    settings: Settings,
    history: list[dict[str, Any]] | None = None,
    on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
) -> str:
    """One turn of the agent loop, resolving tool calls until a final answer."""
    client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama", timeout=180.0)
    profile = load_profile(settings)
    system_prompt = build_system_prompt(profile)

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=settings.ollama_model,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
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
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
        messages.append(
            {"role": "assistant", "content": message.content or "", "tool_calls": tool_calls}
        )

        for index, call in enumerate(message.tool_calls):
            try:
                args = json.loads(call.function.arguments or "{}")
            except ValueError:
                args = {}
            if on_tool_call is not None:
                on_tool_call(call.function.name, args)
            result = execute_search(args, settings)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", None) or f"call_{index}",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return "No final answer produced — the model kept issuing tool calls."
