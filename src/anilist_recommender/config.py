"""Central configuration for anilist-recommender.

Everything is overridable through environment variables so the same binary
works across machines without editing code. Defaults target the documented
hardware: NixOS + RTX 5070 Laptop (8 GB VRAM, fully dedicated to Ollama) with
CPU-only embeddings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"

# Dense embedding dimension for the models we support. Used to size the
# sqlite-vec virtual table at sync time.
EMBEDDING_DIMS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
    "BAAI/bge-m3": 1024,
}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    taste_profile_path: Path

    # Ollama OpenAI-compatible endpoint. All VRAM is reserved for the LLM.
    ollama_base_url: str
    ollama_model: str

    # Embeddings always run on CPU (device="cpu") — see embeddings.py.
    embedding_model: str
    embedding_dim: int

    # Collaborative graph budget: how many of your list entries (ranked by
    # |affinity_weight|) get their `recommendations` pulled from AniList.
    # One request each — raise it if AniList grants you a higher rate limit.
    max_graph_sources: int

    # Minimum interval between two GraphQL requests (seconds). AniList allows
    # 90/min but is currently degraded to ~30/min; 0.85s is a safe default.
    request_interval: float

    @property
    def embedding_query_prefix(self) -> str:
        """`intfloat/e5-*` models expect a `query: ` / `passage: ` prefix."""
        if self.embedding_model.startswith("intfloat/"):
            return "query: "
        return ""

    @property
    def embedding_doc_prefix(self) -> str:
        if self.embedding_model.startswith("intfloat/"):
            return "passage: "
        return ""


def _resolve_embedding_dim(model: str) -> int:
    return EMBEDDING_DIMS.get(model, 384)


def get_settings() -> Settings:
    """Build the immutable settings object from the environment."""
    data_dir = Path(os.environ.get("ANILIST_DATA_DIR", "data")).expanduser().resolve()
    embedding_model = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "anilist_local.db",
        taste_profile_path=data_dir / "taste_profile.json",
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
        embedding_model=embedding_model,
        embedding_dim=_resolve_embedding_dim(embedding_model),
        max_graph_sources=int(os.environ.get("ANILIST_MAX_GRAPH_SOURCES", "60")),
        request_interval=float(os.environ.get("ANILIST_REQUEST_INTERVAL", "0.85")),
    )
