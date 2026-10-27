"""Entry point for `python -m anilist_recommender` (used by the wrapped binary)."""

from .cli import app

if __name__ == "__main__":
    app()
