"""CPU-only embeddings + sqlite-vec cosine search.

The embedding model is *always* instantiated with ``device="cpu"`` so it never
touches the GPU — 100% of VRAM stays available for Ollama. The model is cached
as a module-level singleton so repeated `ask`/`chat` turns don't reload it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import warnings
from typing import Any

# Set BEFORE importing sentence_transformers/transformers/tqdm so their notices
# and progress bars are silenced from the very first cold load.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402

from .config import Settings

_model: SentenceTransformer | None = None


def _quiet_hf_logs() -> None:
    """Silence huggingface_hub's 'unauthenticated requests' notice — it fires on
    every cold model load (a metadata ping to the HF Hub) even though the model
    is cached locally. Noise, not an error: the anonymous download still works,
    and a free HF_TOKEN only raises rate limits."""
    for name in ("huggingface_hub", "transformers", "sentence_transformers", "filelock"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*")


def get_model(settings: Settings) -> SentenceTransformer:
    global _model
    if _model is None:
        _quiet_hf_logs()
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


def content_hash(settings: Settings, text: str) -> str:
    """Cache key for one candidate: model + prefix are part of the hash, so a
    change of embedding model or prefix automatically invalidates every entry."""
    return hashlib.sha256((settings.embedding_doc_prefix + text).encode("utf-8")).hexdigest()


def store_vectors(db: sqlite3.Connection, vectors: list[tuple[int, list[float]]]) -> None:
    """Insert (media_id, embedding) rows into the vec_media virtual table.

    No commit: `sync` runs inside one transaction so the whole rebuild is atomic
    (P1 #9). The caller commits.
    """
    db.executemany(
        "INSERT INTO vec_media (media_id, embedding) VALUES (?, ?)",
        [(media_id, json.dumps(vector)) for media_id, vector in vectors],
    )


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
    # `k` must cover the WHOLE index, not just the filtered candidate set: the
    # top-N nearest neighbours of the query can be other media, and then the
    # filtered candidates would wrongly fall back to 0.0 similarity.
    total = db.execute("SELECT COUNT(*) AS n FROM vec_media").fetchone()["n"]
    k = max(1, int(total))
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
