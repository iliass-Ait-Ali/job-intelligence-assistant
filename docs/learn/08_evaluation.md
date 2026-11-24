# 8. Évaluation scientifique

Les chiffres exacts sont dans [`reports/evaluation_report.md`](../../reports/evaluation_report.md), **généré** par
`jobintel bench all` à partir des fichiers `reports/*.json`. Ce chapitre explique la méthode et comment lire ces
chiffres sans se tromper — et sans tromper un recruteur.

## 8.1 Le problème de la vérité terrain

Il n'existe pas de « bonne réponse » toute faite pour « cette offre est-elle pertinente pour cette requête ? ».
Il faut des **jugements de pertinence** graduels : 2 = très pertinent, 1 = partiellement, 0 = non pertinent,
selon une **grille écrite par requête** (ex. q04 : 2 = data engineer qui construit des pipelines avec Spark ou
Databricks comme stack principale).

**Qui a jugé ?** Aucun humain pour l'instant. Les jugements ont été rédigés par l'assistant IA qui a construit le
système : **à l'aveugle** (sans voir quel système avait renvoyé l'offre ni son rang), avec les grilles écrites,
une citation de l'offre par jugement, et un niveau de confiance. C'est un **conflit d'intérêts** déclaré dans
chaque rapport ; l'outillage de jugement humain existe (`jobintel label …`) et les notes humaines remplacent
automatiquement les propositions dès qu'elles existent. **À dire en entretien, pas à cacher.**

## 8.2 Les métriques de classement (avec un exemple)

Classement d'un système pour une requête : notes `[2, 0, 1, 0, 2]`, pool jugé `{2, 2, 1, 1, 0, 0}`.

| Métrique | Définition | Calcul sur l'exemple (pertinent = note ≥ 1) |
|---|---|---|
| Precision@5 | pertinents dans le top 5 / 5 | 3/5 = 0,60 |
| Recall@5 (« poolé ») | pertinents du top 5 / pertinents du pool | 3/4 = 0,75 |
| RR | 1 / rang du premier pertinent | 1/1 = 1 |
| AP@5 | moyenne des précisions aux rangs pertinents ÷ min(R, k) | (1/1 + 2/3 + 3/5)/4 = 0,567 |
| DCG@5 | Σ (2^note − 1)/log2(rang+1) | 3/1 + 0 + 1/2 + 0 + 3/2,585 = 4,66 |
| nDCG@5 | DCG / DCG du classement idéal [2,2,1,1,0] | 4,66 / (3 + 1,893 + 0,5 + 0,431) = 4,66 / 5,82 = 0,80 |

* **nDCG** est la métrique principale : elle utilise les notes **graduelles** (un 2 vaut 3 fois un 1) et
  pénalise les bonnes offres classées bas (division par log du rang).
* **MRR** répond à « le premier résultat est-il bon ? ». Ici il **sature** à 1,0 (le premier résultat est presque
  toujours au moins partiellement pertinent) → on publie aussi un MRR strict (note 2 seulement).
* **judged@10** : part du top 10 qui a une note. Une offre non jugée compte comme non pertinente ; si judged@10
  est bas, toutes les métriques sont sous-estimées. On le publie donc systématiquement.

Toutes ces formules sont **implémentées à la main** (`evaluation/metrics.py`) et testées contre des calculs faits
à la main (`tests/test_metrics.py`).

## 8.3 Pooling : pourquoi on ne juge pas les 191 offres

Juger 10 requêtes × 191 offres = 1 910 jugements. Le **pooling** (méthode TREC) ne juge que l'union des top 10 de
tous les systèmes comparés : 187 jugements. Biais connu : un système qui trouve des offres qu'aucun autre ne
trouve est désavantagé s'il reste des trous → on a complété le pool pour les deux nouveaux systèmes (20
jugements ajoutés à l'aveugle) et on publie judged@10.

## 8.4 Incertitude : bootstrap sur les requêtes

10 requêtes, c'est peu. Pour chaque métrique on donne un **intervalle de confiance à 95 % par bootstrap** :
on ré-échantillonne 10 000 fois les 10 requêtes avec remise et on prend les percentiles 2,5 et 97,5 de la moyenne.
Pour comparer deux systèmes, on utilise les différences **appariées** par requête (victoires / égalités /
défaites, p-value de bootstrap apparié), car certaines requêtes sont simplement plus faciles pour tout le monde.

## 8.5 Extraction de compétences : précision, rappel, F1 — et le piège du sur-ajustement

* **Précision** = compétences extraites correctes / extraites ; **rappel** = correctes / compétences réelles ;
  **F1** = moyenne harmonique. *Micro* = on additionne les comptes de toutes les offres ; *macro* = moyenne des F1
  par offre.
* Protocole : **20 offres de développement** (graine 42) → analyse d'erreurs → corrections (alias manquants comme
  « deep-learning », « Vektordatenbanken » ; lignes d'avantages ignorées pour le secteur) → mesure finale sur
  **12 offres de test jamais vues** (graine 43), annotées après les corrections et jamais utilisées pour corriger.
* Les chiffres du jeu de développement *après* corrections sont **optimistes** (on a corrigé en regardant ses
  erreurs) ; le chiffre à citer est celui du **test**. Le rapport montre les trois : dev avant, dev après, test.
* Rappel « dans le vocabulaire » : rappel calculé seulement sur les compétences qui existent dans la taxonomie —
  il sépare les deux causes d'échec : alias manquant (corrigeable) vs compétence inconnue du dictionnaire.

## 8.6 Matching : nDCG sur liste condensée et validation croisée leave-one-out

Voir chapitre 4. Sur les 3 CV synthétiques, on ne classe que les offres éligibles **jugées** (liste condensée :
les non-jugées ne favorisent ni ne pénalisent personne). Les poids ajustés sont évalués **hors échantillon**
(entraînés sur 2 CV, testés sur le 3e). Leçon : un gain en apprentissage (0,79) qui disparaît hors échantillon
(0,64) est du sur-apprentissage.

## 8.7 Assistant RAG : métriques automatiques

Définies dans `evaluation/assistant_bench.py`, sur 26 questions FR/EN : précision du routage, validité des
citations, *faithfulness* et taux d'hallucination (sur les phrases **proposées** avant le vérificateur),
pertinence de la réponse (cosinus e5 question ↔ réponse), pertinence de la récupération (les offres citées
exigent-elles bien les compétences demandées ?), absence d'affirmations interdites (salaire, probabilité),
latence, volume de tokens. Aucun « LLM-juge » : tout est défini par des règles reproductibles.

## 8.8 Latence et ressources

`evaluation/latency_bench.py` : démarrage à froid (modèles, index), médiane et p95 de chaque moteur, recommandations
à froid / à chaud, assistant avec et sans LLM, mémoire GPU, taille des index, et comparaison FAISS exact vs HNSW de
Chroma (rappel@10 et temps).

## 8.9 Comment présenter ces résultats en entretien

1. Commencer par la **méthode** (pooling, grilles, aveugle, bootstrap, dev/test).
2. Donner le résultat **avec son intervalle** et la **taille d'échantillon**.
3. Dire spontanément la **limite principale** (labels non humains, 10 requêtes, 3 CV synthétiques).
4. Dire ce qu'on ferait ensuite : 2–3 annotateurs humains, accord inter-annotateurs (kappa de Cohen), 50+
   requêtes, vrais CV anonymisés, test A/B en ligne (clics, candidatures).
