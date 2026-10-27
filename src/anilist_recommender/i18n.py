"""Output-prose translations for the recommendation surface.

Titles are handled separately by ``title_language`` (english/romaji/native); this
module localizes the *prose* around them — the list headers, labels, source
phrasing, status/relation words, and the LLM "why" annotation prompts — for the
configured ``language`` (``fr``/``en``).
"""

from __future__ import annotations

LANGUAGES = ("fr", "en")

_STATUS = {
    "fr": {
        "FINISHED": "terminé",
        "RELEASING": "en cours",
        "NOT_YET_RELEASED": "à paraître",
        "HIATUS": "en pause",
        "CANCELLED": "annulé",
        "DROPPED": "abandonné",
        "PAUSED": "en pause",
    },
    "en": {
        "FINISHED": "finished",
        "RELEASING": "releasing",
        "NOT_YET_RELEASED": "upcoming",
        "HIATUS": "on hiatus",
        "CANCELLED": "cancelled",
        "DROPPED": "dropped",
        "PAUSED": "paused",
    },
}

_RELATION = {
    "fr": {
        "SEQUEL": "suite",
        "PREQUEL": "préquelle",
        "SIDE_STORY": "histoire annexe",
        "SPIN_OFF": "spin-off",
        "ADAPTATION": "adaptation",
        "SOURCE": "source",
        "PARENT": "parent",
        "SUMMARY": "résumé",
        "ALTERNATIVE": "alternative",
        "CHARACTER": "spin-off perso",
        "OTHER": "lié",
    },
    "en": {
        "SEQUEL": "sequel",
        "PREQUEL": "prequel",
        "SIDE_STORY": "side story",
        "SPIN_OFF": "spin-off",
        "ADAPTATION": "adaptation",
        "SOURCE": "source",
        "PARENT": "parent",
        "SUMMARY": "summary",
        "ALTERNATIVE": "alternative",
        "CHARACTER": "character spin-off",
        "OTHER": "related",
    },
}

_STRINGS = {
    "fr": {
        "recommendations_header": "{n} recommandations (triées par affinité) :",
        "proche_de": "proche de",
        "score": "score",
        "sim": "sim",
        "eps": "eps",
        "chap": "chap",
        "vol": "vol",
        "season": "saison {n}",
        "total_eps": "total {n} eps",
        "unknown_len": "? length",
        "dropped": "abandonné",
        "paused": "en pause",
        "rated": "noté {score}/100",
        "note": "note {score}",
        "franchise_header": "{n} suites (triées par affinité) :",
        "affinity": "affinité",
        "from_source": "depuis",
        "titles": "titres",
        "anime_to_print_header": "{n} mangas/LN à lire (triés par affinité) :",
        "print_to_anime_header": "{n} adaptations anime (triées par affinité) :",
        "sequel_of": "suite de",
        "adapted_from": "adapté de",
        "no_results": "Aucun résultat avec ces filtres.",
        "no_sequels": "Aucune suite trouvée avec ces filtres.",
        "adult_disabled": (
            "Contenu adulte désactivé : ta demande mentionne du contenu hentai/adulte, "
            "mais `allow_adult` est sur `false`.\n"
            "Active-le avec `ANILIST_ALLOW_ADULT=true` (ou `allow_adult = true` dans "
            "config.toml) puis relance ta demande."
        ),
        "annotation_system": "Tu es concis, factuel, tu réponds en français et en JSON strict.",
        "annotation_instruction": (
            "Pour chaque recommandation, écris UNE phrase française courte (≤ 20 mots) "
            "expliquant POURQUOI elle plaira à l'utilisateur. Appuie-toi sur le synopsis "
            "fourni (et sur les genres/tags/œuvres proches). N'invente aucun élément "
            "d'intrigue absent du synopsis, ne répète pas le titre, ne cite pas de chiffres."
        ),
        "preferred_genres": "Genres préférés",
        "preferred_tags": "Tags préférés",
        "annotation_json": (
            "Réponds EXACTEMENT un objet JSON unique, sans texte autour : "
            '{"1": "phrase", "2": "phrase", …} — une entrée par numéro, rien d\u2019autre.'
        ),
    },
    "en": {
        "recommendations_header": "{n} recommendations (sorted by affinity):",
        "proche_de": "close to",
        "score": "score",
        "sim": "sim",
        "eps": "eps",
        "chap": "ch",
        "vol": "vol",
        "season": "season {n}",
        "total_eps": "total {n} eps",
        "unknown_len": "? length",
        "dropped": "dropped",
        "paused": "paused",
        "rated": "rated {score}/100",
        "note": "score {score}",
        "franchise_header": "{n} sequels (sorted by affinity):",
        "affinity": "affinity",
        "from_source": "from",
        "titles": "titles",
        "anime_to_print_header": "{n} manga/LN to read (sorted by affinity):",
        "print_to_anime_header": "{n} anime adaptations (sorted by affinity):",
        "sequel_of": "sequel of",
        "adapted_from": "adapted from",
        "no_results": "No results with these filters.",
        "no_sequels": "No sequels found with these filters.",
        "adult_disabled": (
            "Adult content is disabled: your request mentions hentai/adult content, "
            "but `allow_adult` is `false`.\n"
            "Enable it with `ANILIST_ALLOW_ADULT=true` (or `allow_adult = true` in "
            "config.toml) and try again."
        ),
        "annotation_system": "You are concise and factual; answer in English and strict JSON.",
        "annotation_instruction": (
            "For each recommendation, write ONE short English sentence (≤ 20 words) "
            "explaining WHY the user will like it. Ground it in the provided synopsis "
            "(and the genres/tags/related works). Do not invent plot elements absent "
            "from the synopsis, do not repeat the title, do not cite numbers."
        ),
        "preferred_genres": "Preferred genres",
        "preferred_tags": "Preferred tags",
        "annotation_json": (
            "Answer with EXACTLY one JSON object, no surrounding text: "
            '{"1": "sentence", "2": "sentence", …} — one entry per number, nothing else.'
        ),
    },
}


def t(lang: str, key: str, **fmt: object) -> str:
    """Return the localized string for ``key``, interpolating ``fmt`` if given."""
    table = _STRINGS.get(lang) or _STRINGS["en"]
    value = table.get(key)
    if value is None:
        value = _STRINGS["en"].get(key, key)
    return value.format(**fmt) if fmt else value


def status(lang: str, key: str) -> str:
    table = _STATUS.get(lang) or _STATUS["en"]
    return table.get(key, key or "")


def relation(lang: str, key: str) -> str:
    table = _RELATION.get(lang) or _RELATION["en"]
    return table.get(key, key or "")
