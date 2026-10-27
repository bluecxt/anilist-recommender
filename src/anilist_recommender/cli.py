"""Typer + rich CLI (V2): sync, profile, sequels, ask, chat."""

from __future__ import annotations

import asyncio
import json
import logging
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
    help="Local AniList anime/manga/LN/manhwa recommender (Ollama + sqlite-vec).",
    no_args_is_help=True,
)
console = Console()

_RELATION_TYPE_MAP: dict[str, list[str]] = {
    "SEQUEL": ["SEQUEL"],
    "SIDE_STORY": ["SIDE_STORY", "SPIN_OFF"],
    "ADAPTATION": ["ADAPTATION", "SOURCE"],
    "ALL": ["SEQUEL", "SIDE_STORY", "SPIN_OFF", "ADAPTATION", "SOURCE", "ALTERNATIVE"],
}


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

        # --- 1. collaborative graph (top-N sources by |weight|)
        sources = [w for w in works if w.status != "PLANNING"]
        ordered_sources = sorted(sources, key=lambda w: -abs(w.weight))[: settings.max_graph_sources]
        console.print(
            f"[cyan]Scored {len(works)} entries; pulling graph from {len(ordered_sources)} sources…[/cyan]"
        )
        rec_edges: list[tuple[int, int, dict]] = []
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Collaborative graph", total=len(ordered_sources))
            batch_size = 2
            for start in range(0, len(ordered_sources), batch_size):
                chunk = ordered_sources[start : start + batch_size]
                ids = [source.media_id for source in chunk]
                try:
                    rec_edges.extend(await client.fetch_recommendations_batch(ids, per_page=15))
                except gc.AniListError as exc:
                    logger.warning("batch recommendations failed (%s) — retrying one-by-one", exc)
                    for source in chunk:
                        try:
                            nodes = await client.fetch_recommendations(source.media_id, per_page=15)
                        except gc.AniListError as exc2:
                            logger.warning("skipping #%s: %s", source.media_id, exc2)
                            continue
                        for node in nodes:
                            rating = node.get("rating")
                            target = node.get("mediaRecommendation")
                            if rating is not None and target is not None:
                                rec_edges.append((source.media_id, int(rating), target))
                progress.advance(task, len(chunk))

        # --- 2. franchise relations for engaged works (sequels/prequels/adaptations)
        engaged_ids = [w.media_id for w in works if w.status != "PLANNING"]
        console.print(f"[cyan]Fetching franchise relations for {len(engaged_ids)} works…[/cyan]")
        relation_edges: list[tuple[int, str, dict]] = []
        chunk_size = 4
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Franchise relations", total=len(engaged_ids))
            for start in range(0, len(engaged_ids), chunk_size):
                chunk = engaged_ids[start : start + chunk_size]
                try:
                    relation_edges.extend(await client.fetch_relations_chunk(chunk))
                except gc.AniListError as exc:
                    logger.warning("relations chunk failed (%s) — retrying one-by-one", exc)
                    for media_id in chunk:
                        try:
                            relation_edges.extend(await client.fetch_relations_chunk([media_id]))
                        except gc.AniListError as exc2:
                            logger.warning("relations for #%s failed: %s", media_id, exc2)
                progress.advance(task, len(chunk))
    finally:
        await client.close()

    candidates = scoring.compute_graph_scores(works, rec_edges)
    prequel_of, parent_of, relation_rows = scoring.build_relation_maps(relation_edges)

    before = len(candidates)
    candidates = scoring.apply_prequel_gatekeeping(works, candidates, prequel_of, parent_of)
    if before - len(candidates):
        console.print(f"[yellow]Gated {before - len(candidates)} orphan sequels → transferred to roots.[/yellow]")

    all_media = [w.media for w in works] + [c.media for c in candidates]
    scoring.annotate_roots(all_media, prequel_of, parent_of)

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
            connection, works, candidates, edges, relation_rows,
            username, synced_at, settings.embedding_model,
        )

        if candidates:
            console.print("[cyan]Loading embedding model on CPU (never touches the GPU)…[/cyan]")
            texts = [embeddings.build_embedding_text(c.media) for c in candidates]
            ids = [c.media_id for c in candidates]
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("Embedding candidates", total=len(candidates))
                for i in range(0, len(candidates), 32):
                    chunk_texts = texts[i : i + 32]
                    chunk_ids = ids[i : i + 32]
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
        f"{len(relation_rows)} relations → {settings.taste_profile_path}"
    )


@app.command()
def sync(username: str = typer.Argument(None, help="Your AniList username.")) -> None:
    """Download AniList (anime + manga/LN/manhwa), score it, build graph + relations + embeddings."""
    settings = get_settings()
    username = username or settings.username
    if not username:
        console.print("[red]Provide a username: `anilist-rec sync <username>`[/red]")
        raise typer.Exit(1)
    asyncio.run(_sync(username, settings))


# --------------------------------------------------------------------------- profile


def _require_synced(settings: Settings) -> None:
    if not settings.db_path.exists() or not settings.taste_profile_path.exists():
        console.print("[red]No synced data. Run `anilist-rec sync <username>` first.[/red]")
        raise typer.Exit(1)


def _candidate_table(title: str, candidates: list[dict]) -> Table:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Title")
    table.add_column("Type")
    table.add_column("Format")
    table.add_column("GraphScore", justify="right")
    table.add_column("Pool")
    for index, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            candidate["title_romaji"] or candidate["title_english"] or f"#{candidate['media_id']}",
            candidate["type"],
            candidate["format"] or "—",
            f"{candidate['graph_score']:.2f}",
            "IN_PLANNING" if candidate["planning_bonus"] else "DISCOVERY",
        )
    return table


@app.command()
def profile() -> None:
    """Show an analytical summary of your taste, split anime vs manga/LN."""
    settings = get_settings()
    _require_synced(settings)

    profile = agent.load_profile(settings)
    connection = db_module.connect(settings.db_path)
    try:
        meta = db_module.read_meta(connection)
        summary = connection.execute(
            "SELECT ROUND(AVG(z_score), 3) AS avg_z, ROUND(AVG(affinity_weight), 3) AS avg_w "
            "FROM user_list WHERE status != 'PLANNING'"
        ).fetchone()
        top_anime = db_module.top_by_graph_score(connection, 10, "ANIME")
        top_manga = db_module.top_by_graph_score(connection, 10, "MANGA")
    finally:
        connection.close()

    console.print(f"[bold]Profile — {meta.get('username', '?')}[/bold]  (synced {meta.get('synced_at', '?')})")
    console.print(
        f"{meta.get('num_works', '?')} works · {meta.get('num_candidates', '?')} candidates · "
        f"{meta.get('num_relations', '?')} relations"
    )

    stats_table = Table(title="Global statistics")
    stats_table.add_column("Type")
    stats_table.add_column("Count", justify="right")
    stats_table.add_column("Mean score", justify="right")
    stats_table.add_column("Std dev", justify="right")
    for key, label in (("anime", "ANIME"), ("manga", "MANGA / LN / Manhwa")):
        block = (profile.get("stats") or {}).get(key) or {}
        stats_table.add_row(
            label,
            str(block.get("count") or 0),
            f"{block.get('mean_score', 0):.2f}",
            f"{block.get('std_dev', 0):.2f}",
        )
    stats_table.add_row("ALL", "", f"avg Z = {summary['avg_z']}", f"avg weight = {summary['avg_w']}")
    console.print(stats_table)

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

    console.print(_candidate_table("Top 10 latent — ANIME", top_anime))
    console.print(_candidate_table("Top 10 latent — MANGA / LN / Manhwa", top_manga))


# --------------------------------------------------------------------------- sequels


def _target_length(row: dict) -> str:
    if row.get("target_episodes"):
        return f"{row['target_episodes']} eps"
    if row.get("target_chapters"):
        return f"{row['target_chapters']} chap"
    if row.get("target_volumes"):
        return f"{row['target_volumes']} vol"
    return "—"


@app.command()
def sequels(
    media: str = typer.Option("ALL", "--media", help="ANIME | MANGA | ALL"),
    type_: str = typer.Option("ALL", "--type", help="SEQUEL | SIDE_STORY | ADAPTATION | ALL"),
    limit: int = typer.Option(30, "--limit", help="Number of rows to show"),
) -> None:
    """List your unseen sequels / OVAs / movies / spin-offs / adaptations, by affinity."""
    settings = get_settings()
    _require_synced(settings)

    source_type = media.upper()
    if source_type not in ("ANIME", "MANGA", "ALL"):
        console.print("[red]--media must be ANIME, MANGA or ALL.[/red]")
        raise typer.Exit(1)
    relation_types = _RELATION_TYPE_MAP.get(type_.upper())
    if relation_types is None:
        console.print("[red]--type must be SEQUEL, SIDE_STORY, ADAPTATION or ALL.[/red]")
        raise typer.Exit(1)

    connection = db_module.connect(settings.db_path)
    try:
        rows = db_module.find_unseen_franchise_entries(
            connection,
            source_type=source_type if source_type != "ALL" else "BOTH",
            relation_types=relation_types,
            limit=limit,
        )
    finally:
        connection.close()

    table = Table(title=f"Unseen franchise entries ({len(rows)})")
    table.add_column("#", justify="right")
    table.add_column("From (watched/read)")
    table.add_column("Affinity", justify="right")
    table.add_column("Relation")
    table.add_column("Target")
    table.add_column("Format")
    table.add_column("Status")
    table.add_column("Length")
    table.add_column("Avg score", justify="right")
    for index, row in enumerate(rows, start=1):
        table.add_row(
            str(index),
            row["source_title_romaji"] or row["source_title_english"] or f"#{row['source_id']}",
            f"{row['parent_weight']:+.2f}",
            row["relation_type"],
            row["target_title"] or "?",
            row["target_format"] or "—",
            row["target_status"] or "—",
            _target_length(row),
            str(row["target_score"]) if row["target_score"] is not None else "—",
        )
    console.print(table)


# --------------------------------------------------------------------------- ask / chat


def _log_tool(name: str, args: dict) -> None:
    console.print(f"[dim]→ tool {name}: {json.dumps(args, ensure_ascii=False)[:220]}[/dim]")


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
