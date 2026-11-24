# 10. Préparer l'entretien

## 10.1 Le pitch (2 minutes)

> « J'ai construit un moteur de recherche et de recommandation d'offres data/IA, pas un chatbot. Les données
> viennent de trois API publiques autorisées — 720 offres réelles, nettoyées, dédupliquées et structurées
> (compétences requises/souhaitées avec leur ligne de preuve, niveau, pays, langues), stockées en SQLite.
> La recherche combine BM25 et des embeddings multilingues dans FAISS, fusionnés par Reciprocal Rank Fusion, avec
> un re-ranking optionnel par cross-encoder ; sur 10 requêtes jugées, l'hybride passe de 0,75 à 0,84 de nDCG@10
> par rapport à BM25. Le matching applique d'abord les contraintes éliminatoires — inconnu n'est jamais un non —,
> puis un score de compatibilité interprétable dont chaque point est attribuable à un critère ; j'ai testé une
> calibration des poids en validation croisée et je l'ai **rejetée** car elle sur-apprenait. Le système liste les
> compétences manquantes, distingue « absent » de « non mentionné » en interrogeant l'utilisateur, et propose un
> plan d'apprentissage ordonné par tri topologique des prérequis. Un assistant RAG répond en citant ses sources ; un
> vérificateur supprime toute phrase dont les faits ne sont pas dans les documents. Tout tourne en local, avec une
> API FastAPI, une interface Streamlit, plus de 530 tests, une CI et Docker. Limite principale : les jugements de
> pertinence ont été proposés par une IA, pas par des humains — l'outillage d'annotation humaine est prêt. »

## 10.2 Les chiffres à connaître par cœur

| Quoi | Valeur | Contexte à dire avec |
|---|---|---|
| corpus | 720 offres, 490 entreprises, 64 pays | 3 sources, métiers data/IA, collecte du 2026-11-22 |
| hybride vs BM25 | nDCG@10 0,842 vs 0,754 | 10 requêtes, IC 95 % [0,75 ; 0,93], 6 V / 1 N / 3 D |
| cross-encoder | 0,851, ~5× plus lent | gain marginal → optionnel |
| extraction | F1 0,80 (P 0,91 / R 0,70) | test de 12 offres jamais vues ; 19 % des compétences hors taxonomie |
| matching | poids ajustés 0,79 → 0,64 hors échantillon | sur-apprentissage détecté, poids a priori gardés |
| exclusions | précision 1,0, rappel 0,72 | aucune bonne offre cachée ; contraintes en texte libre ratées |
| GPU | 3 833 morceaux encodés en 68 s | RTX 4060, contre ~30 min CPU |

## 10.3 Questions probables et réponses

**« Pourquoi ne pas tout faire avec un LLM ? »** — Coût, latence, non-déterminisme, et surtout : impossible à
évaluer composante par composante et risque d'inventer des exigences. Ici chaque étape est mesurée ; le LLM ne fait
que reformuler des documents fournis, et ce qu'il produit est vérifié.

**« Vos labels sont générés par une IA : vos résultats valent quoi ? »** — Ils sont *indicatifs*. J'ai limité le
biais : jugements à l'aveugle (sans voir le système ni le rang), grilles écrites par requête, citation de l'offre
par jugement, test tenu à l'écart pour l'extraction. Prochaine étape : 2–3 annotateurs humains et un kappa de Cohen.
L'outil d'annotation humaine (`jobintel label`) existe et les notes humaines remplacent automatiquement les miennes.

**« BM25 battu par les mots-clés, c'est normal ? »** — Sur 10 requêtes, les intervalles se recouvrent largement :
ce n'est pas significatif. Explication plausible : petit corpus homogène où les titres contiennent déjà les mots de
la requête, donc l'IDF apporte peu. Je ne conclurais rien sans plus de requêtes.

**« Comment avez-vous choisi les poids ? »** — A priori raisonnés (compétences requises > sémantique > rôle > …),
configurables, puis confrontés à une recherche aléatoire en validation croisée leave-one-candidate-out. Le gain
disparaît hors échantillon, donc je garde les poids a priori. C'est un choix *justifié par l'expérience*.

**« Que se passe-t-il si une offre ne dit pas le type de contrat ? »** — La contrainte vaut *inconnu* : l'offre
n'est pas exclue, et le critère sort du calcul du score (la couverture baisse, et le score de classement se
rapproche de 50).

**« Comment passer à l'échelle ? »** — Collecte planifiée (Airflow/cron), PostgreSQL, Elasticsearch pour BM25,
index FAISS approximatif (IVF-PQ/HNSW) avec rappel mesuré, encodage batch hors ligne, API derrière un load balancer,
cache des embeddings de requêtes, cross-encoder sur GPU limité au top 30.

**« Comment protégez-vous les données des candidats ? »** — CV en mémoire, aucun stockage, aucune coordonnée
extraite, LLM local uniquement (le client refuse les autres hôtes), ports liés à 127.0.0.1, données tierces non
versionnées.

**« Qu'améliorer en premier ? »** — (1) annotation humaine, (2) extraction des compétences hors taxonomie (un LLM
comme *proposeur* + vérificateur), (3) contraintes écrites en texte libre (« freelance position »), (4) un
learning-to-rank quand il y aura des interactions réelles (clics, candidatures).

## 10.4 Démo de 5 minutes (ordre conseillé)

1. **Overview** : architecture et provenance (30 s).
2. **CV & Profile** : charger « fr data engineer », montrer démontré vs listé et une preuve (45 s).
3. **Recommendations** : contraintes, anneau, couverture, puis déplier les exclusions motivées (60 s).
4. **Job Details** : la cascade du score + une explication avec citations (60 s).
5. **Skill Gap** : statut « non mentionné », répondre « j'ai », le plan se réordonne (45 s).
6. **Assistant** : « Quelles compétences me manquent pour l'offre 1 ? » puis montrer les sources (45 s).
7. **Evaluation** : nDCG avec intervalles, et dire la limite des labels (30 s).

## 10.5 Les limites à dire avant qu'on te les dise

Labels non humains ; 10 requêtes et 3 CV synthétiques ; parseur à règles ; taxonomie finie ; secteur faible ;
corpus non représentatif du marché ; petit LLM local (3B) dont une part importante des phrases est rejetée.
Dire une limite, c'est montrer qu'on sait évaluer.
