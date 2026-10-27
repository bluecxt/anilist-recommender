"""Typer + rich CLI: sync, profile, ask, chat."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table

from . import agent, db as db_module, embeddings, graphql_client as gc, scoring
from .config import get_settings, Settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="anilist-rec",
    help="Local AniList anime/manga recommender (Ollama + sqlite-vec).",
    no_args_is_help=True,
)
console = Console()


# --------------------------------------------------------------------------- sync


async def _sync(username: str, settings: Settings) -> None:
    client = gc.AniListClient(min_interval=settings.request_interval)
    try:
        with console.status("[cyan]Fetching AniList profile & lists…[/cyan]"):
            user = await client.fetch_user_profile(username)
            anime_entries = await client.fetch_media_list(username, "ANIME")
            manga_entries = await client.fetch_media_list(username, "MANGA")
        entries = anime_entries + manga_entries
        fav_ids = gc.favourite_ids(user)
        stats = gc.user_statistics(user)

        works = scoring.build_works(entries, fav_ids, stats)
        sources = [w for w in works if w.status != "PLANNING"]
        ordered_sources = sorted(sources, key=lambda w: -abs(w.weight))[: settings.max_graph_sources]

        console.print(
            f"[cyan]Scored {len(works)} list entries; "
            f"pulling collaborative graph from {len(ordered_sources)} sources…[/cyan]"
        )
        rec_edges: list[tuple[int, int, dict]] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Collaborative graph", total=len(ordered_sources))
            for source in ordered_sources:
                try:
                    nodes = await client.fetch_recommendations(source.media_id, per_page=15)
                except gc.AniListError as exc:
                    logger.warning("skipping #%s: %s", source.media_id, exc)
                    progress.advance(task)
                    continue
                for node in nodes:
                    rating = node.get("rating")
                    target = node.get("mediaRecommendation")
                    if rating is not None and target is not None:
                        rec_edges.append((source.media_id, int(rating), target))
                progress.advance(task)
    finally:
        await client.close()

    candidates = scoring.compute_graph_scores(works, rec_edges)

    candidate_ids = {c.media_id for c in candidates}
    edge_map: dict[tuple[int, int], int] = {}
    for source_id, rating, raw_target in rec_edges:
        target_id = (raw_target or {}).get("id")
        if target_id is None or target_id not in candidate_ids:
            continue
        key = (source_id, target_id)
        edge_map[key] = max(edge_map.get(key, -10**9), rating)
    edges = [(s, t, r) for (s, t), r in edge_map.items()]

    connection = db_module.connect(settings.db_path)
    try:
        db_module.reset_schema(connection, settings.embedding_dim)
        synced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        db_module.persist(
            connection, works, candidates, edges, username, synced_at, settings.embedding_model
        )

        if candidates:
            console.print("[cyan]Loading embedding model on CPU (never touches the GPU)…[/cyan]")
            texts = [embeddings.build_embedding_text(c.media) for c in candidates]
            ids = [c.media_id for c in candidates]
            chunk_size = 32
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Embedding candidates", total=len(candidates))
                for i in range(0, len(candidates), chunk_size):
                    chunk_texts = texts[i : i + chunk_size]
                    chunk_ids = ids[i : i + chunk_size]
                    vectors = embeddings.embed_texts(chunk_texts, settings)
                    embeddings.store_vectors(connection, list(zip(chunk_ids, vectors)))
                    progress.advance(task, len(chunk_texts))
    finally:
        connection.close()

    profile = scoring.build_taste_profile(username, works, stats)
    settings.taste_profile_path.parent.mkdir(parents=True, exist_ok=True)
    settings.taste_profile_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    console.print(
        f"[green]Done.[/green] {len(works)} works, {len(candidates)} candidates, "
        f"taste profile → {settings.taste_profile_path}"
    )


@app.command()
def sync(
    username: str = typer.Argument(None, help="Your AniList username."),
) -> None:
    """Download AniList data, score it, build GraphScore + CPU embeddings."""
    settings = get_settings()
    username = username or os.environ.get("ANILIST_USERNAME")
    if not username:
        console.print("[red]Provide a username: `anilist-rec sync <username>`[/red]")
        raise typer.Exit(1)
    asyncio.run(_sync(username, settings))


# --------------------------------------------------------------------------- profile


def _require_synced(settings: Settings) -> None:
    if not settings.db_path.exists() or not settings.taste_profile_path.exists():
        console.print("[red]No synced data. Run `anilist-rec sync <username>` first.[/red]")
        raise typer.Exit(1)


@app.command()
def profile() -> None:
    """Show an analytical summary of your taste and latent recommendations."""
    settings = get_settings()
    _require_synced(settings)

    profile = agent.load_profile(settings)
    connection = db_module.connect(settings.db_path)
    try:
        meta = db_module.read_meta(connection)
        status_rows = connection.execute(
            "SELECT status, COUNT(*) AS n, ROUND(AVG(affinity_weight), 2) AS avg_w "
            "FROM user_list GROUP BY status ORDER BY n DESC"
        ).fetchall()
        summary = connection.execute(
            "SELECT ROUND(AVG(z_score), 3) AS avg_z, ROUND(AVG(affinity_weight), 3) AS avg_w "
            "FROM user_list WHERE status != 'PLANNING'"
        ).fetchone()
        top = db_module.top_by_graph_score(connection, 10)
    finally:
        connection.close()

    console.print(f"[bold]Profile — {meta.get('username', '?')}[/bold]  (synced {meta.get('synced_at', '?')})")
    console.print(
        f"{meta.get('num_works', '?')} works · {meta.get('num_candidates', '?')} candidates · "
        f"embeddings {meta.get('embedding_model', '?')}"
    )

    stats_table = Table(title="Global statistics", show_lines=False)
    stats_table.add_column("Type")
    stats_table.add_column("Count", justify="right")
    stats_table.add_column("Mean score", justify="right")
    stats_table.add_column("Std dev", justify="right")
    for key, label in (("anime", "ANIME"), ("manga", "MANGA")):
        block = (profile.get("stats") or {}).get(key) or {}
        stats_table.add_row(
            label,
            str(block.get("count") or 0),
            f"{block.get('mean_score', 0):.2f}",
            f"{block.get('std_dev', 0):.2f}",
        )
    stats_table.add_row(
        "ALL",
        "",
        f"avg Z = {summary['avg_z']}",
        f"avg weight = {summary['avg_w']}",
    )
    console.print(stats_table)

    status_table = Table(title="List status distribution")
    status_table.add_column("Status")
    status_table.add_column("Count", justify="right")
    status_table.add_column("Avg weight", justify="right")
    for row in status_rows:
        status_table.add_row(row["status"], str(row["n"]), str(row["avg_w"]))
    console.print(status_table)

    if profile.get("top_tags"):
        tags_table = Table(title="Top affinity tags")
        tags_table.add_column("Tag")
        tags_table.add_column("Score", justify="right")
        tags_table.add_column("Count", justify="right")
        for item in profile["top_tags"][:15]:
            tags_table.add_row(item["name"], str(item["score"]), str(item["count"]))
        console.print(tags_table)

    if profile.get("anti_tags"):
        anti_table = Table(title="Anti-tags (from drops / negative scores)")
        anti_table.add_column("Tag")
        anti_table.add_column("Count", justify="right")
        for item in profile["anti_tags"][:10]:
            anti_table.add_row(item["name"], str(item["count"]))
        console.print(anti_table)

    rec_table = Table(title="Top latent recommendations (highest GraphScore)")
    rec_table.add_column("#", justify="right")
    rec_table.add_column("Title")
    rec_table.add_column("Type")
    rec_table.add_column("GraphScore", justify="right")
    rec_table.add_column("Pool")
    rec_table.add_column("Positive bridges", justify="right")
    for index, candidate in enumerate(top, start=1):
        rec_table.add_row(
            str(index),
            candidate["title_romaji"] or candidate["title_english"] or f"#{candidate['media_id']}",
            candidate["type"],
            f"{candidate['graph_score']:.2f}",
            "IN_PLANNING" if candidate["planning_bonus"] else "DISCOVERY",
            str(len(candidate["positive_sources"])),
        )
    console.print(rec_table)


# --------------------------------------------------------------------------- ask / chat


def _log_tool(name: str, args: dict) -> None:
    payload = json.dumps(args, ensure_ascii=False)
    console.print(f"[dim]→ tool {name}: {payload[:220]}[/dim]")


@app.command()
def ask(query: str = typer.Argument(..., help="Your request in natural language.")) -> None:
    """One-shot recommendation request through the local LLM."""
    settings = get_settings()
    _require_synced(settings)
    console.print("[cyan]Thinking (local LLM)…[/cyan]")
    answer = agent.run_agent(query, settings, on_tool_call=_log_tool)
    console.print(Markdown(answer))


@app.command()
def chat() -> None:
    """Interactive conversation keeping session history to refine recommendations."""
    settings = get_settings()
    _require_synced(settings)
    history: list[dict] = []
    console.print("[bold]chat mode[/bold] — type [italic]/exit[/italic] to quit.")
    while True:
        try:
            query = Prompt.ask("[bold green]you[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        query = query.strip()
        if query.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if not query:
            continue
        answer = agent.run_agent(query, settings, history=history, on_tool_call=_log_tool)
        console.print(Markdown(answer))
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    app()
