# 1. Le problème et les données

## 1.1 Le problème métier

Un étudiant en data science cherche un stage ou un premier emploi. Les job boards lui renvoient des listes de
mots-clés : « data » ramène des centaines d'offres, il ne sait pas lesquelles lui correspondent **vraiment**, ce
qui est **obligatoire** (un contrat CDI quand il veut un stage, l'allemand C1) ou seulement **souhaité**, ni quoi
apprendre pour progresser. Trois difficultés scientifiques en découlent :

* **Vocabulaire** : le CV dit « apprentissage automatique », l'offre dit « machine learning » ; le CV dit
  « PostgreSQL », l'offre « SQL ». La recherche lexicale rate ces correspondances → recherche sémantique.
* **Structure** : une offre est du texte libre ; il faut en extraire des champs (compétences requises, années,
  diplôme, langues) sans inventer ce qui n'est pas écrit.
* **Évaluation** : « bonne recommandation » n'a pas de vérité terrain évidente → il faut construire des jugements
  de pertinence et des métriques de classement.

## 1.2 Sources autorisées (et pourquoi c'est important)

| Source | Accès | Conditions respectées dans le code |
|---|---|---|
| [Arbeitnow](https://www.arbeitnow.com) | API publique paginée | lien vers l'offre d'origine conservé |
| [Himalayas](https://himalayas.app) | API publique de recherche, sans clé | « Apply » = `applicationLink` d'origine ; crédit visible ; pas de redistribution à d'autres agrégateurs |
| [Remotive](https://remotive.com) | API publique | ≤ 4 appels/jour (cache de 6 h), lien et nom de la source affichés |

On **ne scrape pas** de sites (LinkedIn, Indeed) : leurs conditions l'interdisent et un recruteur te le
demandera. Les données collectées ne sont **pas versionnées** (`data/` est dans `.gitignore`) : elles
appartiennent aux plateformes et se recréent avec `jobintel collect`.

Fichiers : `src/job_intel/connectors/` (un connecteur par source, contrat commun `Connector.fetch/normalize`),
`src/job_intel/http.py` (client poli : délai minimal entre requêtes, retries avec backoff, timeouts).

## 1.3 Le pipeline ETL

```
fetch (ou snapshot récent en cache) → normalize → filtre de pertinence → déduplication → export JSONL/CSV
```

* **Snapshots bruts gzip** (`data/raw/`) : on garde la réponse brute de chaque API ; relancer le pipeline ne
  refrappe pas l'API si le snapshot est « frais » (TTL par source). Avantage : reproductibilité + politesse.
* **Normalisation** (`textutils.py`) : HTML → texte, dates en ISO-8601 UTC, type de contrat canonique
  (`internship`, `full_time`, …) — avec une règle importante : si le **titre** dit « Stage » ou « Werkstudent »,
  il l'emporte sur l'étiquette de la plateforme (souvent « Full Time » par défaut).
* **Filtre de pertinence** (`relevance.py`) : on garde les métiers data/ML/IA d'après le **titre** (règles
  strictes : « AI Marketing Manager » n'est pas un rôle IA), et les stages seulement s'ils ont un signal data.
* **Déduplication** (`dedup.py`) : même identifiant, même URL canonique (sans paramètres `utm_*`), ou même
  (titre, entreprise, lieu) normalisés. La meilleure copie (score de pertinence) gagne.

Chiffres de la collecte du 2026-11-22 : 3 400 offres récupérées → 723 pertinentes → **720** après
déduplication (Arbeitnow 204, Himalayas 515, Remotive 1).

## 1.4 Enrichissement : du texte aux champs structurés (Module 1)

`src/job_intel/enrich/postings.py` transforme chaque offre en `JobPosting` (modèle **Pydantic** validé) :

| Champ | Comment | Remarque |
|---|---|---|
| pays / région | gazetteer ville → pays (`search/locations.py`) | « Casablanca » → morocco ; « EMEA » → europe |
| contrat | étiquette fournisseur, corrigée par le titre | inconnu = `None`, jamais deviné |
| famille de métier | motifs sur le titre | data_scientist, ml_engineer, … |
| compétences requises / souhaitées / mentionnées | extracteur à sections (`match/requirements.py`) | chaque compétence garde **sa ligne de preuve** |
| soft skills | vocabulaire multilingue (`match/softskills.py`) | affichées, jamais notées |
| niveau d'expérience | mots du titre (senior, junior, stage), sinon années exigées | ≤ 2 ans → junior, 3–5 → mid, ≥ 6 → senior |
| diplôme, langues | uniquement s'ils sont écrits | |
| secteur | heuristique par mots-clés (≥ 2 indices) | **faible** : précision mesurée ≈ 0,33–1,0 sur de petits échantillons |

**Pourquoi des règles et pas un LLM pour extraire ?** Déterministe (même entrée → même sortie), testable,
instantané, gratuit, et chaque champ a une preuve. Inconvénient : seul le vocabulaire connu est reconnu (12–19 %
des compétences des offres sont hors taxonomie, voir chapitre 8). C'est un compromis assumé et mesuré.

**Validation** : une offre invalide (URL non http, titre vide) n'est jamais jetée silencieusement : elle est
listée dans `quality_report.json` avec la raison. Le rapport de complétude montre par exemple que seules 45,8 %
des offres indiquent un nombre d'années et 14,4 % une langue : « inconnu » est la norme, d'où l'importance de
ne pas pénaliser l'inconnu (chapitre 4).

## 1.5 Stockage : pourquoi SQLite (et pas PostgreSQL)

`src/job_intel/enrich/store.py` reconstruit une base SQLite normalisée en une transaction :
`jobs`, `job_skills`, `job_soft_skills`, `job_roles`, `job_languages`, `job_countries`.

* Un fichier, zéro administration, SQL complet pour l'analytique (`SELECT skill, COUNT(DISTINCT job_id) …`).
* Le schéma n'utilise que des types standards → **portable tel quel vers PostgreSQL** si le volume grandit
  (plusieurs écrivains concurrents, centaines de milliers d'offres).
* Le JSONL reste la source de vérité de l'application ; SQLite sert les requêtes analytiques et l'API.

## 1.6 Limites à connaître (et à dire en entretien)

* Échantillon de trois job boards, métiers data/IA seulement ; Himalayas est 100 % remote et très US.
* Les offres expirent : le corpus d'évaluation est **figé** (`data/eval_corpus/2026-10-23`, 191 offres) pour que
  les jugements de pertinence restent valides.
* L'heuristique de secteur est faible ; elle est étiquetée « estimation » dans l'interface.

**Question d'entretien probable :** *« Pourquoi ne pas utiliser un LLM pour structurer les offres ? »* →
coût et latence × 720 offres à chaque collecte, non-déterminisme, risque d'inventer des exigences ; les règles
donnent une preuve par champ et un F1 mesuré de 0,80 sur un test tenu à l'écart. Un LLM pourrait être ajouté
**en complément** pour les compétences hors vocabulaire, avec le même vérificateur.
