"""Central configuration for anilist-recommender.

Precedence (highest wins):

    CLI argument > environment variable > config.toml > built-in default

The config file is optional TOML, looked up in this order (first match wins):

    1. ``$ANILIST_CONFIG`` (explicit path)
    2. ``./config.toml`` (current directory)
    3. ``$XDG_CONFIG_HOME/anilist-recommender/config.toml`` (default ``~/.config``)

See ``config.example.toml`` for the available keys.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
CONFIG_FILENAME = "config.toml"

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

    username: str | None

    # Ollama OpenAI-compatible endpoint. All VRAM is reserved for the LLM.
    ollama_base_url: str
    ollama_model: str

    # Embeddings always run on CPU (device="cpu") — see embeddings.py.
    embedding_model: str
    embedding_dim: int

    # Collaborative graph budget: how many list entries (ranked by
    # |affinity_weight|) get their `recommendations` pulled from AniList.
    max_graph_sources: int
    request_interval: float

    config_path: Path | None

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


def _find_config_file() -> Path | None:
    explicit = os.environ.get("ANILIST_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    cwd_config = Path.cwd() / CONFIG_FILENAME
    if cwd_config.is_file():
        return cwd_config

    xdg_base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
    xdg_config = xdg_base / "anilist-recommender" / CONFIG_FILENAME
    if xdg_config.is_file():
        return xdg_config
    return None


def _load_config_file(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (tomllib.TOMLDecodeError, OSError):
        # A malformed config must never crash the CLI — fall back to defaults.
        return {}


def _pick(file_cfg: dict, file_key: str, env_key: str, default):
    if env_key in os.environ:
        return os.environ[env_key]
    if file_key in file_cfg and file_cfg[file_key] is not None:
        return file_cfg[file_key]
    return default


def _pick_int(file_cfg: dict, file_key: str, env_key: str, default: int) -> int:
    value = _pick(file_cfg, file_key, env_key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick_float(file_cfg: dict, file_key: str, env_key: str, default: float) -> float:
    value = _pick(file_cfg, file_key, env_key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_settings() -> Settings:
    """Build the immutable settings object (env + config file + defaults)."""
    config_path = _find_config_file()
    file_cfg = _load_config_file(config_path)

    data_dir = Path(str(_pick(file_cfg, "data_dir", "ANILIST_DATA_DIR", "data"))).expanduser().resolve()
    embedding_model = str(
        _pick(file_cfg, "embedding_model", "EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    )

    username = _pick(file_cfg, "username", "ANILIST_USERNAME", None)
    username = str(username) if username else None

    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "anilist_local.db",
        taste_profile_path=data_dir / "taste_profile.json",
        username=username,
        ollama_base_url=str(
            _pick(file_cfg, "ollama_base_url", "OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ),
        ollama_model=str(_pick(file_cfg, "ollama_model", "OLLAMA_MODEL", "qwen2.5:7b")),
        embedding_model=embedding_model,
        embedding_dim=_resolve_embedding_dim(embedding_model),
        max_graph_sources=_pick_int(file_cfg, "max_graph_sources", "ANILIST_MAX_GRAPH_SOURCES", 60),
        request_interval=_pick_float(file_cfg, "request_interval", "ANILIST_REQUEST_INTERVAL", 0.85),
        config_path=config_path,
    )
