"""CPU-only embeddings + sqlite-vec cosine search.

The embedding model is *always* instantiated with ``device="cpu"`` so it never
touches the GPU — 100% of VRAM stays available for Ollama. The model is cached
as a module-level singleton so repeated `ask`/`chat` turns don't reload it.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from sentence_transformers import SentenceTransformer

from .config import Settings

_model: SentenceTransformer | None = None


def get_model(settings: Settings) -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def build_embedding_text(media: dict[str, Any]) -> str:
    """The exact string we embed for each candidate."""
    title_romaji = media.get("title_romaji") or ""
    title_english = media.get("title_english") or ""
    genres = ", ".join(media.get("genres") or [])
    top_tags = (media.get("tags") or [])[:8]
    tags = ", ".join(top_tags)
    synopsis = media.get("description") or ""
    return (
        f"{title_romaji} | {title_english} | Genres: {genres} | "
        f"Tags: {tags} | Synopsis: {synopsis}"
    )


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    """Embed a batch of candidate documents on CPU (normalized to unit norm)."""
    if not texts:
        return []
    model = get_model(settings)
    prefixed = [settings.embedding_doc_prefix + text for text in texts]
    vectors = model.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [vector.tolist() for vector in vectors]


def embed_query(text: str, settings: Settings) -> list[float]:
    """Embed a semantic query on CPU."""
    model = get_model(settings)
    vector = model.encode(
        [settings.embedding_query_prefix + text],
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()


def store_vectors(db: sqlite3.Connection, vectors: list[tuple[int, list[float]]]) -> None:
    """Insert (media_id, embedding) rows into the vec_media virtual table."""
    db.executemany(
        "INSERT INTO vec_media (media_id, embedding) VALUES (?, ?)",
        [(media_id, json.dumps(vector)) for media_id, vector in vectors],
    )
    db.commit()


def query_similarities(
    db: sqlite3.Connection,
    query_text: str,
    candidate_ids: list[int],
    settings: Settings,
) -> dict[int, float]:
    """Cosine similarity in [0, 1] for every candidate vs the semantic query."""
    if not candidate_ids:
        return {}
    query_vector = embed_query(query_text, settings)
    k = max(1, len(candidate_ids))
    sql = (
        "SELECT media_id, distance FROM vec_media "
        f"WHERE embedding MATCH ? AND k = {k}"
    )
    rows = db.execute(sql, (json.dumps(query_vector),)).fetchall()
    similarities: dict[int, float] = {}
    for row in rows:
        similarity = 1.0 - float(row["distance"])
        similarities[row["media_id"]] = max(0.0, min(1.0, similarity))
    return similarities
