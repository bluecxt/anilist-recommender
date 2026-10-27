# anilist-recommender

Agent local en ligne de commande qui recommande des **animés** et des **mangas**
à partir de ton profil AniList (notes, statuts, favoris), du **graphe collaboratif
de recommandations** d'AniList et d'une **recherche vectorielle locale**. La
synthèse finale est produite par un **LLM local (Ollama)** via *tool calling* strict.

> Stack minimale : `httpx` · `sqlite3` + `sqlite-vec` · `sentence-transformers` (CPU)
> · `pydantic` + `openai` · `typer` + `rich`. **Aucun** LangChain / LlamaIndex / ChromaDB.

---

## Matériel & allocation mémoire (strict)

- **OS** : NixOS.
- **GPU** : RTX 5070 Laptop, **8 Go de VRAM**.
- **100 % de la VRAM → Ollama** (`http://localhost:11434/v1`, modèle par défaut `qwen2.5:7b`).
- **0 Mo de VRAM pour les embeddings** : `sentence-transformers` tourne **sur CPU**
  (`device="cpu"`), modèle par défaut `intfloat/multilingual-e5-small` (384 dims).
  `BAAI/bge-m3` (1024 dims) est disponible via `EMBEDDING_MODEL`.

---

## Installation

```bash
# 1. Entrer dans le shell reproductible Nix
nix develop          # génère flake.lock au premier lancement

# 2. Créer le venv et installer les dépendances (uv lit pyproject.toml)
uv sync

# 3. Lancer Ollama et tirer le modèle (si pas déjà fait)
ollama pull qwen2.5:7b
```

Le `shellHook` du flake exporte `LD_LIBRARY_PATH` (libstdc++ / libz) pour que les
wheels *manylinux* de `torch` et `sqlite-vec` se chargent sur NixOS. Les
**embeddings ne touchent jamais le GPU** : toute la VRAM reste à Ollama.

---

## Utilisation

```bash
uv run anilist-rec --help

# 1. Synchroniser un profil (notes → poids d'affinité → GraphScore → embeddings CPU)
uv run anilist-rec sync <username>

# 2. Afficher le résumé analytique du profil + les reco latentes au plus fort GraphScore
uv run anilist-rec profile

# 3. Une requête ponctuelle
uv run anilist-rec ask "Conseille-moi un anime psychologique de moins de 25 épisodes, dans ma planning list ou proche de mes favoris"

# 4. Mode conversationnel (garde l'historique pour affiner)
uv run anilist-rec chat
```

---

## Binaire (`anilist-rec`) & intégration NixOS

Le flake expose aussi un **paquet** (pas seulement un devShell) :

```bash
# Construit un binaire autonome → result/bin/anilist-rec
nix build .#default --option sandbox false

# Lance directement la commande (équivalent de `anilist-rec ...`)
nix run .#default -- sync <username>
```

Pour l'ajouter à ta **config NixOS** (depuis ton `flake.nix` racine) :

```nix
inputs.anilist-recommender.url = "github:bluecxt/anilist-recommender";

# puis dans le module concerné :
environment.systemPackages = [
  inputs.anilist-recommender.packages.${pkgs.system}.default
];
```

> ⚠️ Le paquet installe `torch` / `sentence-transformers` / `sqlite-vec` depuis
> PyPI **au moment du build** : il a besoin du réseau pendant le build. Sur NixOS,
> soit tu construis une fois avec `--option sandbox false` (et pousses le résultat
> vers un binary cache), soit tu actives `nix.settings.sandbox = "relaxed"`.
> Pour un build 100 % hors-ligne et reproductible, la piste est
> [`uv2nix`](https://github.com/astral-sh/uv2nix).

**Alternative rapide** (sans toucher à ta config) : `uv tool install .` installe
un binaire `anilist-rec` persistant dans `~/.local/bin`.

---

## Configuration (variables d'environnement)

| Variable | Défaut | Rôle |
|---|---|---|
| `ANILIST_USERNAME` | — | Nom d'utilisateur AniList (sinon passé à `sync`) |
| `ANILIST_DATA_DIR` | `data/` | Dossier contenant `anilist_local.db` + `taste_profile.json` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Endpoint OpenAI-compatible d'Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Modèle LLM (doit supporter le tool calling) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modèle d'embeddings CPU (`BAAI/bge-m3` possible) |
| `ANILIST_MAX_GRAPH_SOURCES` | `60` | Nombre d'œuvres (triées par \|poids\|) dont on tire le graphe collaboratif |
| `ANILIST_REQUEST_INTERVAL` | `0.85` | Intervalle minimal entre deux requêtes GraphQL (s) |

---

## Comment ça marche

1. **Extraction** (`graphql_client.py`) — `User` (stats `meanScore` / `standardDeviation`,
   favoris) + `MediaListCollection` paginé par `chunk`/`perChunk`, avec backoff
   exponentiel sur HTTP 429 (`Retry-After`).
2. **Pondération** (`scoring.py`) — `affinity_weight` déterministe par œuvre :
   Z-score `(score − μ) / σ` (plancher `σ = 10`), règles conditionnelles par statut
   (`COMPLETED` / `CURRENT` / `REPEATING` / `PAUSED` / `DROPPED`), bonus favoris `+2.0`.
3. **Graphe collaboratif** — `GraphScore(X) = Σₛ Poids(s)·ln(1 + max(0, rating s→X))`
   pour chaque candidat (`PLANNING` + découvertes), avec `positive_sources`
   (ponts depuis les œuvres aimées) et `negative_warnings` (ponts depuis les drops).
4. **Recherche hybride** (`db.py` + `embeddings.py`) — filtres durs SQL + similarité
   cosinus `sqlite-vec` :
   `FinalScore = Norm(GraphScore)·0.55 + Sim·0.35 + PlanningBonus·0.10`.
5. **Agent** (`agent.py`) — tool calling vers `search_candidates`, profil injecté dans
   le prompt système, interdiction d'inventer des titres hors candidats.

### Pourquoi le graphe collaboratif est tiré œuvre par œuvre

AniList impose une **limite de complexité (~500) par requête** : imbriquer
`recommendations(perPage: 15)` de toute la liste dans `MediaListCollection`
dépasse la limite dès une petite centaine d'œuvres. On tire donc le graphe
**par œuvre, en respectant le rate-limit**, en priorisant les œuvres au plus fort
`|affinity_weight|` (`ANILIST_MAX_GRAPH_SOURCES`, une requête chacune). Augmente la
valeur si AniList t'accorde un rate-limit supérieur.

---

## Fichiers

```
anilist-recommender/
├── flake.nix
├── pyproject.toml
├── README.md
├── data/                      # gitignoré : anilist_local.db + taste_profile.json
└── src/anilist_recommender/
    ├── config.py              # chemins, URL Ollama, modèle LLM, réglages
    ├── graphql_client.py      # extraction AniList + rate-limit
    ├── db.py                  # schéma SQLite + sqlite-vec + requêtes hybrides
    ├── scoring.py             # Z-scores, GraphScore, taste_profile.json
    ├── embeddings.py          # vectorisation CPU + cosinus sqlite-vec
    ├── agent.py               # agent LLM (tool calling) + synthèse
    └── cli.py                 # commandes sync / profile / ask / chat
```
