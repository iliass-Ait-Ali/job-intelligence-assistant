# 9. Architecture logicielle et MLOps

## 9.1 Les couches

```
┌──────────── Interfaces ────────────┐
│  Streamlit UI (9 pages)  FastAPI   │  ui/, api/          présentation, validation des entrées
│  CLI `jobintel …`                  │  cli*.py
├──────────── Service ───────────────┤
│  JobIntelService                   │  services.py        ressources partagées, chargement paresseux,
│                                    │                     une seule porte d'entrée pour UI et API
├──────────── Domaine ───────────────┤
│ connectors/ enrich/ search/ match/ │  logique métier pure, testable sans UI
│ market/ assistant/ rag/ evaluation/│
├──────────── Données ───────────────┤
│ raw snapshots · JSONL · SQLite ·   │  data/ (non versionné)
│ Chroma (vecteurs) · FAISS (mémoire)│
└────────────────────────────────────┘
```

Règle : **la logique métier ne dépend jamais de l'interface**. La page Streamlit et l'endpoint FastAPI appellent
la même méthode `JobIntelService.recommend(...)` : impossible qu'ils divergent. Chaque module de domaine est une
fonction ou une classe testable isolément (tests unitaires sans modèle ni réseau : `HashingEmbedder` remplace e5,
un faux re-ranker remplace le cross-encoder, un faux serveur HTTP remplace Ollama).

| Module du cahier des charges | Dossier |
|---|---|
| Data ingestion | `connectors/`, `pipeline.py`, `http.py`, `storage.py` |
| Preprocessing | `textutils.py`, `relevance.py`, `dedup.py`, `enrich/` |
| CV parsing / skill extraction | `match/cv_text.py`, `match/cv_parser.py`, `match/taxonomy.py`, `match/requirements.py`, `match/softskills.py` |
| Embeddings / indexing | `search/embedder.py`, `search/document.py`, `search/index.py`, `search/faiss_index.py` |
| Retrieval / re-ranking | `search/lexical.py`, `search/keyword.py`, `search/retrieve.py`, `search/rerank.py` |
| Matching / recommendation | `match/matcher.py`, `match/scoring.py` |
| Skill gap | `match/gap.py`, `knowledge/skills_kb.yaml` |
| RAG orchestration | `assistant/`, `rag/` |
| API / UI | `api/`, `ui/` |
| Evaluation | `evaluation/`, `reports/` |

## 9.2 L'API REST

`jobintel api` puis http://127.0.0.1:8000/docs (documentation OpenAPI générée depuis les modèles Pydantic).
Sans état : le client renvoie le profil à chaque requête, le serveur ne stocke **aucun** CV. Codes d'erreur
explicites (400 CV illisible, 404 offre inconnue, 422 entrée invalide, 503 index absent).

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health`, `/status` | santé, volumétrie, état du LLM |
| GET | `/jobs`, `/jobs/{id}` | catalogue filtrable (pays, rôle, contrat, niveau, compétence) |
| POST | `/search` | 5 moteurs, filtres d'éligibilité |
| POST | `/cv/parse` | PDF / DOCX / TXT ou texte → profil |
| POST | `/recommendations` | offres classées, composantes du score, exclusions motivées |
| POST | `/gap`, `/plan` | écarts pour une offre, plan d'apprentissage |
| GET | `/market/skills`, `/market/cooccurrence`, `/market/companies` | analytique |
| POST | `/assistant/ask` | réponse citée et vérifiée |

## 9.3 Qualité logicielle

* **Typage** partout (annotations + Pydantic pour les frontières : profil, offre, requêtes API).
* **Configuration centralisée** : `config.py` (variables d'environnement, `.env.example`), `config/scoring.yaml`.
* **Erreurs** : exceptions métier explicites (`CVExtractionError`, `SearchError`, `LLMUnavailable`,
  `LLMPrivacyError`…) avec des messages destinés à l'utilisateur ; une panne LLM ne casse jamais l'interface.
* **Logs** : `logging` standard, bibliothèques bruyantes rendues silencieuses.
* **Lint** : `ruff` (pyflakes, pycodestyle, isort, bugbear, pyupgrade).

## 9.4 Tests : la pyramide

| Niveau | Exemples | Où |
|---|---|---|
| Unitaires | métriques IR calculées à la main, BM25, RRF, taxonomie, parseur, score, tri topologique, routeur | `tests/test_metrics.py`, `test_platform_core.py`, … |
| Intégration | vraie base Chroma + FAISS sur un corpus synthétique ; API FastAPI via `TestClient` ; faux serveur Ollama | `test_api_and_ui.py`, `test_search_index.py`, `test_ollama.py` |
| Bout en bout | chaque page Streamlit exécutée (`AppTest`) ; parcours recommandations → écarts → chat ; script Playwright qui pilote l'appli réelle et prend les captures | `test_api_and_ui.py`, `scripts/capture_screenshots.py` |

Plus de 530 tests, hors ligne, ~1 minute. Les tests ont trouvé de vrais bugs pendant le développement
(DataFrame vide sans colonnes, valeur par défaut absente d'un widget, phrase positive rejetée par la règle des
compétences manquantes).

## 9.5 CI/CD et conteneurs

* **GitHub Actions** (`.github/workflows/ci.yml`) : lint, tests sur Python 3.11 et 3.12 (PyTorch CPU), build de
  l'image Docker.
* **Docker** : image multi-étapes (`python:3.11-slim`, PyTorch CPU, utilisateur non-root, healthcheck) ;
  `docker-compose.yml` : `pipeline` (collect → enrich → index), `api`, `ui`, et `ollama` en profil optionnel.
  Les données et les modèles sont des **volumes**, jamais copiés dans l'image.

## 9.6 Sécurité et données personnelles

* CV traité en mémoire, fichier temporaire supprimé, aucun identifiant personnel extrait, rien n'est persisté.
* LLM : uniquement local (ou hôte explicitement déclaré de confiance par l'opérateur).
* Ports publiés sur `127.0.0.1` par défaut ; HTML échappé dans l'UI ; seuls les liens http(s) sont rendus.
* Secrets : aucun n'est nécessaire (API publiques) ; `.env` ignoré par git.
* Données tierces (offres) jamais versionnées.

## 9.7 Reproductibilité

Graines fixes (échantillons d'annotation, bootstrap, recherche de poids), corpus d'évaluation figé, rapports
JSON générés par `jobintel bench all`, rapport Markdown généré depuis ces JSON, versions minimales des dépendances
dans `pyproject.toml`.
