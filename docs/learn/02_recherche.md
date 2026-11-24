# 2. Recherche d'information : du mot-clé au cross-encoder

Objectif : pour une requête (« stage data analyst à Paris ») ou un CV, ordonner les offres de la plus
pertinente à la moins pertinente. Cinq moteurs sont implémentés et **comparés sur les mêmes requêtes jugées**
(chapitre 8). Code : `src/job_intel/search/`.

## 2.1 Moteur 0 — chevauchement de mots-clés (baseline)

`score(q, d) = |termes(q) ∩ termes(d)|` (nombre de mots distincts de la requête présents dans l'offre).
Aucune pondération. C'est le plancher : si un moteur sophistiqué ne le bat pas, il ne sert à rien.
Tokenisation commune à tous les moteurs lexicaux : minuscules, accents retirés, mots ≥ 2 lettres, liste de mots
vides FR/EN/DE/NL (`lexical.py::tokenize`).

## 2.2 Moteur 1 — BM25

Pour chaque terme `t` de la requête et chaque document `d` :

```
score(q, d) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
IDF(t)      = ln( 1 + (N − df(t) + 0,5) / (df(t) + 0,5) )
```

* **IDF** : un mot rare (« dbt », présent dans 3 % des offres) pèse plus qu'un mot fréquent (« data »).
* **Saturation k1 = 1,5** : répéter « python » 20 fois ne rend pas une offre 20× plus pertinente ; la
  contribution plafonne vers `IDF·(k1+1)`.
* **Normalisation de longueur b = 0,75** : une offre très longue contient mécaniquement plus de mots ; on
  divise sa fréquence par sa longueur relative.

*Exemple numérique réel (corpus du 2026-11-22)* : N = 720 offres. « dbt » apparaît dans df = 92 offres →
IDF = ln(1 + 628,5/92,5) ≈ 2,05 ; « data » dans 654 offres → IDF ≈ 0,10 ; « python » dans 507 → 0,35 ;
« stage » dans 57 → 2,53. Une occurrence de « dbt » pèse donc ~20 fois une occurrence de « data ».

Implémenté **sans dépendance** en 60 lignes (`lexical.py::Bm25Index`) : index inversé `terme → {doc: tf}`.

**Limite** : aucune notion de sens. « stage » ≠ « internship », « ML » ≠ « machine learning ».

## 2.3 Moteur 2 — recherche dense (embeddings)

Un **modèle d'embedding** transforme un texte en vecteur de 768 nombres tel que deux textes de sens proche ont des
vecteurs proches. Modèle : `intfloat/multilingual-e5-base` (multilingue, licence MIT, entraîné par apprentissage
contrastif sur des paires requête/passage). Particularité e5 : préfixes `"query: "` et `"passage: "`.

**Similarité cosinus** : `cos(q, d) = q·d / (‖q‖ ‖d‖)`. Les vecteurs sont **normalisés** (‖v‖ = 1) dès
l'indexation, donc `cos = q·d` (simple produit scalaire) — c'est ce qui permet d'utiliser un index par produit
interne.

**Découpage (chunking)** : une offre fait ~1 300 tokens, le modèle en lit au plus 512. On découpe en morceaux de
≤ 300 tokens (chevauchement 40), chacun préfixé par `titre | entreprise | lieu | contrat` pour garder le
contexte. Score d'une offre = **meilleur** morceau (une offre est pertinente si l'un de ses passages l'est).
Le code **refuse** de tronquer silencieusement : un morceau trop long est une erreur (`ChunkingError`).
720 offres → 3 833 morceaux, encodés en **68 s sur GPU** (RTX 4060) contre ~30 min sur CPU.

### Stockage vs recherche : Chroma + FAISS

* **ChromaDB** stocke textes, métadonnées et vecteurs sur disque, et rend l'indexation **incrémentale et
  idempotente** (hash de contenu par offre : seules les offres nouvelles/modifiées sont ré-encodées).
* **FAISS** fait la recherche : au démarrage, les vecteurs stockés sont chargés dans un `IndexFlatIP`
  — un parcours **exact** de tous les vecteurs (produit interne). 3 833 × 768 flottants = 11 Mo ; une requête
  prend ~1 ms. Un index approximatif (HNSW, IVF) n'apporterait que de la perte de rappel à cette échelle.
* Le **filtrage** (contrat, lieu…) se fait *avant* la recherche (`IDSelectorBatch`) : on ne cherche que parmi les
  offres éligibles, au lieu de filtrer après coup une liste top-k qui pourrait devenir vide.

**HNSW en deux phrases** (pour l'entretien) : un graphe de proximité hiérarchique — on descend des couches
grossières aux couches fines en suivant les voisins les plus proches. Complexité ~log(N) au lieu de N, au prix
d'un rappel < 100 %. Le benchmark de latence mesure le rappel@10 de l'HNSW de Chroma face à la recherche exacte.

## 2.4 Moteur 3 — hybride par Reciprocal Rank Fusion (RRF)

Les scores BM25 (0 à ~30) et cosinus (0,78 à 0,89) ne sont **pas comparables** : les additionner n'a pas de
sens sans calibration. RRF fusionne les **rangs** :

```
RRF(d) = Σ_{s ∈ {BM25, dense}}  1 / (k + rang_s(d)),   k = 60
```

*Exemple* : une offre 1re en BM25 et 10e en dense : 1/61 + 1/70 = 0,0307. Une offre 3e et 3e : 2/63 = 0,0317 →
elle passe devant : RRF récompense le **consensus**. k = 60 est la valeur de l'article original (Cormack et
al., 2009) ; elle n'a **pas** été ajustée sur nos données (pas de sur-apprentissage possible).

## 2.5 Moteur 4 — re-ranking par cross-encoder

* **Bi-encoder** (moteur 2) : requête et offre encodées *séparément* → les vecteurs d'offres se pré-calculent →
  rapide, mais l'interaction requête/offre est compressée dans un produit scalaire.
* **Cross-encoder** : *un seul* transformeur lit `[CLS] requête [SEP] passage` ; chaque mot de la requête
  « regarde » chaque mot du passage (attention croisée) → bien plus fin (négation, « junior » vs « senior »),
  mais un passage du réseau **par paire** → réservé aux 30 premiers candidats de l'hybride.

Modèle : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingue, 118 M paramètres). Score d'une offre = max
sur ses morceaux. Coût mesuré sur les 720 offres (GPU RTX 4060, à chaud) : ~740 ms par requête contre ~70 ms pour
l'hybride, ~34 ms pour le dense seul et ~19 ms pour BM25.

> *Anecdote utile en entretien* : la première mesure donnait 1,2 s pour l'hybride et 58 s pour le re-ranking. Deux
> causes trouvées grâce au benchmark : l'hybride calculait un extrait de texte pour les 720 offres alors que seules
> les 10 affichées en ont besoin (corrigé → 70 ms), et trois processus se partageaient les 8 Go du GPU, qui débordait
> en mémoire système. **Mesurer avant d'optimiser.**

## 2.6 Ce que dit la mesure (détails au chapitre 8)

Sur 10 requêtes jugées (FR/EN/DE, avec filtres) : l'hybride améliore nettement BM25 (nDCG@10 0,842 vs 0,754 ;
6 victoires, 3 défaites ; bootstrap apparié p ≈ 0,008). Le cross-encoder ajoute peu (0,851) pour 5× la latence :
**en production on le garderait optionnel**. Surprise instructive : le simple chevauchement de mots-clés bat BM25
ici (0,786) — petit corpus de titres très « parlants », où l'IDF compte peu ; les intervalles de confiance se
recouvrent largement, donc pas de conclusion forte.

## 2.7 Questions d'entretien

* *Pourquoi RRF plutôt qu'une somme pondérée ?* Pas de calibration nécessaire, robuste, un seul hyperparamètre non
  ajusté. Une somme pondérée demanderait de normaliser des distributions de scores qui changent avec chaque requête.
* *Pourquoi le meilleur morceau et pas la moyenne ?* Une offre longue parle de l'entreprise, des avantages… La
  moyenne dilue le passage pertinent.
* *Comment passer à 10 millions d'offres ?* FAISS IVF-PQ ou HNSW (index approximatif, rappel mesuré), BM25 dans
  Elasticsearch/OpenSearch, encodage en batch hors ligne, cross-encoder seulement sur le top 50.
