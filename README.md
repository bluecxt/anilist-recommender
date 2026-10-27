# anilist-recommender

Agent local en ligne de commande qui recommande des **animés, mangas, light novels,
manhwas et manhuas** à partir de ton profil AniList (notes, statuts, favoris), du
**graphe collaboratif de recommandations**, du **graphe relationnel** (suites, OAV,
films, adaptations anime↔manga/LN) et d'une **recherche vectorielle locale**. La
synthèse est produite par un **LLM local (Ollama)** via *tool calling* strict.

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

# 0. Voir où la config est lue + les réglages effectifs
uv run anilist-rec config

# 1. Synchroniser (anime + manga/LN/manhwa, graphe relationnel, embeddings CPU)
uv run anilist-rec sync

# 2. Résumé analytique (stats Anime vs Manga/LN, top tags, anti-tags, top latent)
uv run anilist-rec profile

# 3. Audit des suites / OAV / films / spin-offs non vus (sans LLM)
uv run anilist-rec sequels --media ANIME --type ALL --limit 30

# 4. Requête ponctuelle via le LLM (3 outils : recherche / suites / adaptations)
uv run anilist-rec ask "Conseille-moi un anime psychologique de moins de 25 épisodes"

# 5. Mode conversationnel
uv run anilist-rec chat
```

---

## Binaire (`anilist-rec`) & intégration NixOS

Le flake expose aussi un **paquet** construit **hors-ligne depuis nixpkgs**
(Python 3.13), servi par le binary cache officiel — aucun réseau au build :

```bash
# Construit un binaire autonome → result/bin/anilist-rec (offline, cache.nixos.org)
nix build .#default

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

> ℹ️ Le paquet repose sur `python313Packages` — `sqlite-vec` et
> `sentence-transformers` (avec son `torch` **CPU**) y sont déjà empaquetés, d'où
> le passage de Python 3.12 à **3.13**. Au premier `sync`, le modèle d'embeddings
> (~470 Mo) est téléchargé depuis Hugging Face dans `~/.cache/huggingface`
> (réseau au *runtime*, pas au build).

**Alternative dev** : `nix develop && uv sync && uv run anilist-rec ...`

---

## Configuration

Le plus simple : copie `config.example.toml` en `config.toml` et fixe ton nom
d'utilisateur — plus besoin de le passer à chaque commande.

```bash
cp config.example.toml config.toml
```

```toml
# config.toml
username = "ton-pseudo"            # fixe le profil AniList
# data_dir = "data"
# ollama_model = "qwen2.5:7b"
# embedding_model = "intfloat/multilingual-e5-small"
# max_graph_sources = 60
```

Fichier cherché dans l'ordre (premier trouvé) : `$ANILIST_CONFIG` → `./config.toml`
→ `~/.config/anilist-recommender/config.toml`. Priorité : argument CLI > variable
d'environnement > `config.toml` > défaut.

Les variables d'environnement restent disponibles pour surcharger ponctuellement :

| Variable | Défaut | Rôle |
|---|---|---|
| `ANILIST_USERNAME` | — | Nom d'utilisateur AniList |
| `ANILIST_DATA_DIR` | `data/` | Dossier contenant `anilist_local.db` + `taste_profile.json` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Endpoint OpenAI-compatible d'Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Modèle LLM (doit supporter le tool calling) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Modèle d'embeddings CPU (`BAAI/bge-m3` possible) |
| `ANILIST_MAX_GRAPH_SOURCES` | `60` | Nombre d'œuvres (triées par \|poids\|) dont on tire le graphe collaboratif |
| `ANILIST_REQUEST_INTERVAL` | `0.85` | Intervalle minimal (anti-burst) entre deux requêtes (s) ; le quota est piloté par les headers `X-RateLimit-*` |

---

## Comment ça marche

1. **Extraction** (`graphql_client.py`) — `User` + `MediaListCollection` (anime **et**
   manga/LN/manhwa) paginé par `chunk`/`perChunk`, `countryOfOrigin`/`format`/`duration`,
   recommandations collaboratives **et** graphe `relations` (`relationType(version: 2)`)
   tirés par lots via `Page(id_in:)`, backoff 429.
2. **Pondération** (`scoring.py`) — `affinity_weight` par œuvre (Z-score par média,
   anime vs manga), ratio de progression unifié (épisodes / chapitres / tomes / seuil 20).
3. **Graphe collaboratif + relationnel** — `GraphScore` par candidat, puis **Prequel
   Gatekeeping** : une saison 2/3 dont le préquel n'est pas terminé est retirée et son
   score est **transféré** vers la racine.
4. **Recherche hybride** (`db.py` + `embeddings.py`) — filtres SQL (catégorie, longueur,
   pays) + cosinus `sqlite-vec` : `FinalScore = Norm(GraphScore)·0.55 + Sim·0.35 + PlanningBonus·0.10`.
5. **Agent** (`agent.py`) — 3 outils (`search_new_recommendations`,
   `find_unseen_franchise_entries`, `find_cross_media_adaptations`), `num_ctx=8192`,
   sortie dynamique (analyse détaillée ≤ 5, tableau complet au-delà).

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
    ├── db.py                  # schéma SQLite + media_relations + sqlite-vec + requêtes
    ├── scoring.py             # Z-scores, GraphScore, gatekeeping, taste_profile.json
    ├── embeddings.py          # vectorisation CPU + cosinus sqlite-vec
    ├── agent.py               # agent LLM (3 outils) + synthèse
    └── cli.py                 # commandes sync / profile / sequels / ask / chat
```
