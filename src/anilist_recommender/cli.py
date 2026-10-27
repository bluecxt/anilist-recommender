"""Typer + rich CLI (V2): sync, profile, sequels, explain, ask, chat."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone

import typer
from rich.console import Console
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
from .config import get_settings, Settings, pick_title

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _set_log_level(verbose: bool) -> None:
    """Show transient AniList rate-limit/retry logs only when `--verbose` is set."""
    logging.getLogger().setLevel(logging.INFO if verbose else logging.WARNING)

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
    "ALL": ["SEQUEL", "SIDE_STORY", "SPIN_OFF", "ADAPTATION", "SOURCE"],
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
                    logger.info("batch recommendations failed (%s) — retrying one-by-one", exc)
                    for source in chunk:
                        try:
                            nodes = await client.fetch_recommendations(source.media_id, per_page=15)
                        except gc.AniListError as exc2:
                            logger.info("skipping #%s: %s", source.media_id, exc2)
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
                    logger.info("relations chunk failed (%s) — retrying one-by-one", exc)
                    for media_id in chunk:
                        try:
                            relation_edges.extend(await client.fetch_relations_chunk([media_id]))
                        except gc.AniListError as exc2:
                            logger.info("relations for #%s failed: %s", media_id, exc2)
                progress.advance(task, len(chunk))
    finally:
        await client.close()

    # Relation maps first: the graph scoring needs the franchise roots to
    # amortize the echo chamber (P0 #3).
    prequel_of, parent_of, relation_rows = scoring.build_relation_maps(relation_edges)
    candidates = scoring.compute_graph_scores(works, rec_edges, prequel_of, parent_of)

    before = len(candidates)
    candidates = scoring.apply_prequel_gatekeeping(works, candidates, prequel_of, parent_of)
    if before - len(candidates):
        console.print(f"[yellow]Gated {before - len(candidates)} orphan sequels → transferred to roots.[/yellow]")

    all_media = [w.media for w in works] + [c.media for c in candidates]
    scoring.annotate_roots(all_media, prequel_of, parent_of)
    scoring.compute_franchise_stats(all_media, prequel_of, parent_of)

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
        # ONE transaction around the whole rebuild: a crash mid-sync (network
        # errors already passed, but embeddings can fail) rolls back to the
        # previous intact DB instead of leaving a half-built one (P1 #9).
        connection.execute("BEGIN IMMEDIATE")
        try:
            db_module.reset_schema(connection, settings.embedding_dim)
            synced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            db_module.persist(
                connection, works, candidates, edges, relation_rows,
                username, synced_at, settings.embedding_model,
            )

            if candidates:
                console.print("[cyan]Embedding candidates on CPU (never touches the GPU)…[/cyan]")
                ids = [c.media_id for c in candidates]
                texts = [embeddings.build_embedding_text(c.media) for c in candidates]
                hashes = [embeddings.content_hash(settings, text) for text in texts]

                cache = db_module.load_embedding_cache(connection, settings.embedding_model)
                reused: list[tuple[int, list[float]]] = []
                to_embed: list[int] = []
                for index, media_id in enumerate(ids):
                    entry = cache.get(media_id)
                    if entry is not None and entry[0] == hashes[index]:
                        reused.append((media_id, entry[1]))
                    else:
                        to_embed.append(index)

                new_entries: list[tuple[int, str, list[float]]] = []
                if reused:
                    embeddings.store_vectors(connection, reused)
                if to_embed:
                    with Progress(
                        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                        BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                    ) as progress:
                        task = progress.add_task("Embedding candidates", total=len(to_embed))
                        for start in range(0, len(to_embed), 32):
                            chunk = to_embed[start : start + 32]
                            chunk_ids = [ids[i] for i in chunk]
                            vectors = embeddings.embed_texts([texts[i] for i in chunk], settings)
                            pairs = list(zip(chunk_ids, vectors))
                            embeddings.store_vectors(connection, pairs)
                            new_entries.extend(
                                (media_id, hashes[index], vector)
                                for (media_id, vector), index in zip(pairs, chunk)
                            )
                            progress.advance(task, len(chunk))
                db_module.save_embedding_cache(connection, settings.embedding_model, new_entries)
                db_module.prune_embedding_cache(connection, settings.embedding_model, set(ids))
                if reused:
                    console.print(
                        f"[dim]Embedding cache: {len(reused)} réutilisés, {len(to_embed)} recalculés.[/dim]"
                    )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
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
def sync(
    username: str = typer.Argument(None, help="Your AniList username."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show AniList rate-limit / retry logs."),
) -> None:
    """Download AniList (anime + manga/LN/manhwa), score it, build graph + relations + embeddings."""
    settings = get_settings()
    _set_log_level(verbose)
    username = username or settings.username
    if not username:
        console.print("[red]Provide a username: `anilist-rec sync <username>`[/red]")
        console.print("[dim]Or set it in config.toml (see config.example.toml) or $ANILIST_USERNAME.[/dim]")
        raise typer.Exit(1)
    asyncio.run(_sync(username, settings))


# --------------------------------------------------------------------------- profile


def _require_synced(settings: Settings) -> None:
    if not settings.db_path.exists() or not settings.taste_profile_path.exists():
        console.print("[red]No synced data. Run `anilist-rec sync <username>` first.[/red]")
        raise typer.Exit(1)
    # Guard against a stale DB from an older schema (e.g. pre title_native).
    try:
        connection = db_module.connect(settings.db_path)
        try:
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(media)")}
            version = int(db_module.read_meta(connection).get("schema_version", "0") or 0)
        finally:
            connection.close()
    except Exception:
        return  # let the actual command surface any real DB error
    if (
        "title_native" not in columns
        or "is_adult" not in columns
        or version < db_module.SCHEMA_VERSION
    ):
        console.print("[red]Database schema is outdated — run `anilist-rec sync` again.[/red]")
        raise typer.Exit(1)


def _candidate_table(title: str, candidates: list[dict], language: str = "english") -> Table:
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
            agent.anilist_link(pick_title(candidate, language), candidate["type"], candidate["media_id"]),
            candidate["type"],
            candidate["format"] or "—",
            agent._heat(candidate["graph_score"] or 0.0, 0.0, 50.0, f"{candidate['graph_score']:.2f}"),
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

    console.print(_candidate_table("Top 10 latent — ANIME", top_anime, settings.title_language))
    console.print(_candidate_table("Top 10 latent — MANGA / LN / Manhwa", top_manga, settings.title_language))


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
            agent.anilist_link(
                pick_title(
                    {
                        "id": row["source_id"],
                        "title_romaji": row["source_title_romaji"],
                        "title_english": row["source_title_english"],
                        "title_native": row["source_title_native"],
                    },
                    settings.title_language,
                ),
                row["source_type"],
                row["source_id"],
            ),
            f"{row['parent_weight']:+.2f}",
            row["relation_type"],
            agent.anilist_link(
                pick_title(
                    {
                        "id": row["target_id"],
                        "title_romaji": row["target_title_romaji"],
                        "title_english": row["target_title_english"],
                        "title_native": row["target_title_native"],
                    },
                    settings.title_language,
                ),
                row["target_type"],
                row["target_id"],
            ),
            row["target_format"] or "—",
            row["target_status"] or "—",
            _target_length(row),
            str(row["target_score"]) if row["target_score"] is not None else "—",
        )
    console.print(table)


# --------------------------------------------------------------------------- explain


def _age_label(timestamp: int | None) -> str:
    if not timestamp:
        return "—"
    days = (time.time() - float(timestamp)) / 86_400
    if days < 1:
        return f"{days * 24:.1f} h"
    if days < 365:
        return f"{days:.0f} j"
    return f"{days / 365:.1f} ans"


@app.command()
def explain(
    title: str = typer.Argument(..., help="Titre (ou fragment) du candidat à expliquer."),
) -> None:
    """Deterministic 'why is this recommended' breakdown — no LLM involved."""
    settings = get_settings()
    _require_synced(settings)

    connection = db_module.connect(settings.db_path)
    try:
        matches = db_module.find_media_by_title(connection, title, limit=10)
        if not matches:
            console.print(f"[red]Aucun média trouvé pour « {title} ».[/red]")
            raise typer.Exit(1)
        needle = title.strip().lower()
        chosen = next(
            (
                media
                for media in matches
                if any(
                    (media.get(key) or "").strip().lower() == needle
                    for key in ("title_romaji", "title_english", "title_native")
                )
            ),
            None,
        )
        if chosen is None:
            if len(matches) > 1:
                console.print("[yellow]Plusieurs correspondances — précise le titre :[/yellow]")
                for media in matches:
                    console.print(
                        f"  • {agent.anilist_link(pick_title(media, settings.title_language), media.get('type'), media.get('id'))} "
                        f"({media['type']}, {media['format'] or '?'})"
                    )
                raise typer.Exit(1)
            chosen = matches[0]
        media_id = int(chosen["id"])

        cand = db_module.get_candidate(connection, media_id)
        if cand is None:
            console.print(
                f"[red]{pick_title(chosen, settings.title_language)} n'est pas un candidat "
                f"(ni dans ta planning, ni découvert par le graphe collaboratif).[/red]"
            )
            raise typer.Exit(1)

        sources = db_module.recommendation_sources(connection, media_id)
        relations = db_module.relation_context(connection, media_id)
        prequel_of, parent_of = scoring.derive_prequel_parent_maps(
            db_module.franchise_triples(connection)
        )
        # Title lookup for every id we need to name (sources, roots, relations).
        needed = {row["source_media_id"] for row in sources}
        needed |= {scoring.chain_root(sid, prequel_of, parent_of) for sid in needed}
        needed |= {row["source_id"] for row in relations}
        needed |= {row["target_id"] for row in relations}
        title_of: dict[int, dict] = {}
        if needed:
            placeholders = ",".join("?" * len(needed))
            title_of = {
                row["id"]: dict(row)
                for row in connection.execute(
                    f"SELECT id, type, title_romaji, title_english, title_native FROM media "
                    f"WHERE id IN ({placeholders})",
                    sorted(needed),
                ).fetchall()
            }

        def name(media_id: int) -> str:
            info = title_of.get(media_id)
            if not info:
                return f"#{media_id}"
            return agent.anilist_link(
                pick_title(info, settings.title_language), info.get("type"), media_id
            )
        # Recompute the amortized graph score from the stored edges (same math
        # as sync — see scoring.compute_graph_scores).
        raw_contribs: list[tuple[int, float]] = []
        root_of: dict[int, int] = {}
        for row in sources:
            sid = int(row["source_media_id"])
            weight = float(row["affinity_weight"] or 0.0)
            rating = float(row["rating"] or 0.0)
            raw_contribs.append((sid, weight * math.log1p(max(0.0, rating))))
            root_of.setdefault(sid, scoring.chain_root(sid, prequel_of, parent_of))
        damped = dict(scoring.amortize_contributions(raw_contribs, root_of))
        own_total = sum(damped.values())
        stored = float(cand["graph_score"] or 0.0)
        transferred = stored - own_total
    finally:
        connection.close()

    # --- header
    fmt = " · ".join(bit for bit in (chosen.get("format"), agent._length_label(chosen, settings.language)) if bit)
    console.print(
        f"[bold]{agent.anilist_link(pick_title(chosen, settings.title_language), chosen.get('type'), media_id)}[/bold] — {fmt}"
    )
    avg = chosen.get("average_score")
    avg_text = agent._heat(float(avg), 0.0, 100.0, str(avg)) if avg else "?"
    console.print(
        f"type [cyan]{chosen['type']}[/cyan] · statut [cyan]{chosen.get('status') or '?'}[/cyan] · "
        f"note communautaire {avg_text} · "
        f"popularité [cyan]{chosen.get('popularity') or '?'}[/cyan]"
    )
    if chosen.get("genres"):
        try:
            genres = json.loads(chosen["genres"]) if isinstance(chosen["genres"], str) else chosen["genres"]
            console.print(f"genres : {', '.join(genres or [])}")
        except ValueError:
            pass
    console.print()

    # --- planning status + stagnation (P2 #12)
    if cand["planning_bonus"]:
        updated = cand.get("list_updated_at")
        if updated:
            age_years = max(0.0, (time.time() - float(updated)) / 31_536_000)
            decay = 0.5 ** (age_years / 2.0)
            console.print(
                f"Dans ta [bold]planning[/bold] depuis {_age_label(updated)} "
                f"→ bonus de planning amorti ×[cyan]{decay:.2f}[/cyan] "
                "(demi-vie 2 ans : une vieille intention compte moins)."
            )
        else:
            console.print(
                "Dans ta [bold]planning[/bold] (date inconnue) → bonus complet ×1.00."
            )
    else:
        console.print("Source du candidat : [bold]découverte collaborative[/bold] (pas dans ta planning).")
    console.print()

    # --- graph score decomposition
    table = Table(title=f"GraphScore = {stored:.3f} (amorti par franchise)")
    table.add_column("Source", overflow="fold")
    table.add_column("Poids", justify="right")
    table.add_column("Note", justify="right")
    table.add_column("Brut", justify="right")
    table.add_column("Racine franchise", overflow="fold")
    table.add_column("Amorti", justify="right")
    ordered = sorted(raw_contribs, key=lambda pair: -pair[1])
    for sid, raw in ordered[:12]:
        row = next(r for r in sources if int(r["source_media_id"]) == sid)
        root_id = root_of[sid]
        root_label = "—" if root_id == sid else name(root_id)
        factor = damped.get(sid, 0.0) / raw if raw else 0.0
        weight = float(row["affinity_weight"] or 0.0)
        damped_val = damped.get(sid, 0.0)
        table.add_row(
            agent.anilist_link(
                pick_title(
                    {
                        "id": sid,
                        "title_romaji": row["title_romaji"],
                        "title_english": row["title_english"],
                        "title_native": row["title_native"],
                    },
                    settings.title_language,
                ),
                row.get("type"),
                sid,
            ),
            agent._heat(weight, -2.0, 2.5, f"{weight:+.2f}"),
            str(row["rating"]),
            agent._heat(raw, -30.0, 30.0, f"{raw:+.3f}"),
            root_label,
            agent._heat(damped_val, -30.0, 30.0, f"{damped_val:+.3f}") + (
                f" {'×' + f'{factor:.2f}' if factor < 0.999 else ''}"
            ),
        )
    console.print(table)
    raw_total = sum(value for _, value in raw_contribs)
    if abs(transferred) > 1e-6:
        if abs(stored - raw_total) <= 1e-6:
            console.print(
                f"[yellow]Écart de {transferred:+.3f} : le score stocké date d'un sync "
                f"antérieur à l'amortissement par franchise — relance `sync` pour l'aligner.[/yellow]"
            )
        else:
            console.print(
                f"[yellow]Dont {transferred:+.3f} transféré depuis des suites retirées par le "
                f"Prequel Gatekeeping (leur préquel n'est pas terminé).[/yellow]"
            )
    elif not sources:
        console.print("[dim]Aucun bord collaboratif — score issu uniquement de ta planning.[/dim]")
    if cand.get("negative_warnings"):
        warnings = " · ".join(
            agent._format_source(s, settings, colorize=True) for s in cand["negative_warnings"][:4]
        )
        console.print(f"[red]⚠ recommandé aussi par : {warnings}[/red]")
    console.print()

    # --- quality factor (P2 #10)
    average = chosen.get("average_score") or 0
    quality = min(1.0, float(average) / 75.0) if average else 0.5
    console.print(
        f"Facteur qualité (bayésien) : {agent._heat(quality, 0.0, 1.0, f'{quality:.2f}')} "
        "= min(1, note/75)" + (" (note inconnue → neutre 0.5)" if not average else "")
    )
    console.print()

    # --- relations context
    if relations:
        console.print("[bold]Contexte relationnel[/bold]")
        for row in relations:
            kind = agent.i18n.relation(settings.language, row["relation_type"])
            if int(row["source_id"]) == media_id:
                label = f"→ {kind} : {name(int(row['target_id']))}"
            else:
                label = f"← {kind} de : {name(int(row['source_id']))}"
            console.print(f"  {label}")
    else:
        console.print("[dim]Aucune relation franchise connue (racine autonome).[/dim]")
    console.print()
    console.print(
        "[dim]Note : le score final affiché par `ask` dépend du lot filtré au moment de la "
        "requête (normalisation par rang + similarité sémantique + bonus de planning).[/dim]"
    )


# --------------------------------------------------------------------------- config


@app.command()
def config() -> None:
    """Show where the config file is read from and the effective settings."""
    settings = get_settings()

    if settings.config_path:
        console.print(f"[bold]Config file:[/bold] {settings.config_path}")
    else:
        console.print("[yellow]No config file found.[/yellow]")
        console.print("Searched (first match wins):")
        console.print("  1. $ANILIST_CONFIG")
        console.print("  2. ./config.toml")
        console.print("  3. ~/.config/anilist-recommender/config.toml")
        console.print("Create one with: cp config.example.toml config.toml")

    table = Table(title="Effective settings")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("username", settings.username or "(unset)")
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("ollama_model", settings.ollama_model)
    table.add_row("ollama_base_url", settings.ollama_base_url)
    table.add_row("embedding_model", settings.embedding_model)
    table.add_row("max_graph_sources", str(settings.max_graph_sources))
    table.add_row("request_interval", str(settings.request_interval))
    table.add_row("title_language", settings.title_language)
    table.add_row("language", settings.language)
    table.add_row("allow_adult", str(settings.allow_adult))
    console.print(table)


# --------------------------------------------------------------------------- ask / chat


def _log_tool(name: str, args: dict) -> None:
    console.print(f"[dim]→ tool {name}: {json.dumps(args, ensure_ascii=False)[:220]}[/dim]")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your request in natural language."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show the LLM's tool calls."),
) -> None:
    """One-shot recommendation request through the local LLM."""
    settings = get_settings()
    _require_synced(settings)
    console.print("[cyan]Thinking (local LLM)…[/cyan]")
    answer = agent.run_agent(query, settings, on_tool_call=_log_tool if verbose else None)
    console.print(answer)


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show the LLM's tool calls."),
) -> None:
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
        answer = agent.run_agent(query, settings, history=history, on_tool_call=_log_tool if verbose else None)
        console.print(answer)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    app()
