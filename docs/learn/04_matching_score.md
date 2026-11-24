# 4. Matching CV ↔ offre et score de compatibilité

Code : `src/job_intel/match/matcher.py` (pipeline), `scoring.py` (score), `config/scoring.yaml` (poids).

## 4.1 Le pipeline en quatre étapes

1. **Contraintes éliminatoires d'abord.** Tu choisis lesquelles sont « dures » (contrat, lieu, expérience,
   diplôme, langue). Pour chaque offre et chaque contrainte : `satisfied`, `violated` ou `unknown`.
   **Seule une violation claire d'une contrainte dure exclut une offre** — et elle apparaît dans la liste
   « Exclues » avec sa raison. *Inconnu n'exclut jamais.* Ce filtrage a lieu **avant** la recherche : sinon la
   plupart des places du top 80 seraient prises par des offres exclues ensuite (bug réel trouvé en phase 4).
2. **Requêtes multiples construites à partir du CV** : une par rôle visé, une pour les compétences, une par
   expérience, une par projet. Chacune interroge BM25 **et** le dense ; tout est fusionné par RRF → un pool de
   candidats. Un CV n'est jamais réduit à un seul sac de mots.
3. **Évaluation de chaque candidat** : compétences (exacte / transférable / non attestée, avec preuves), rôle,
   contraintes souples.
4. **Score de compatibilité** et tri.

Cas particulier traité : sur les job boards « remote » (Himalayas, Remotive), le champ lieu indique **où le
candidat peut résider** (« United States, Canada »), pas un bureau. Un candidat au Maroc est donc *violé* pour
une offre réservée aux États-Unis, *satisfait* pour « Worldwide » ou « Europe » si le pays est européen.

## 4.2 Le score : une moyenne pondérée qui ignore l'inconnu

Neuf composantes `c_i ∈ [0, 1]` :

| Composante | Calcul | Poids par défaut | Justification du poids |
|---|---|---|---|
| compétences requises | (exactes + 0,5·transférables) / nb requises | 0,30 | ce que l'offre exige explicitement : le cœur de l'adéquation |
| sémantique | cosinus du meilleur passage, ramené sur [0,1] | 0,20 | rattrape le vocabulaire hors taxonomie |
| rôle | le titre correspond-il à un rôle visé ? | 0,15 | « bon métier » avant « bonnes compétences » |
| expérience | années du CV / années exigées | 0,10 | critère fréquent mais souvent flexible |
| compétences souhaitées | même formule | 0,08 | bonus, d'où un poids plus faible |
| diplôme | (niveau CV + 1) / (niveau exigé + 1) si insuffisant | 0,05 | souvent négociable |
| langues | part des langues exigées présentes | 0,05 | |
| lieu, contrat | satisfait / violé (préférences souples) | 0,04, 0,03 | les vraies contraintes sont déjà éliminatoires |

```
score     = 100 · Σ_{i connu} w_i·c_i / Σ_{i connu} w_i
couverture = Σ_{i connu} w_i / Σ_i w_i
```

**Pourquoi exclure l'inconnu des deux sommes ?** Si l'offre ne dit rien des langues, compter 0 pénaliserait
injustement ; compter 1 récompenserait l'ignorance. On renormalise sur ce qui est observable et on **affiche la
couverture** (« 85 % des critères observables »).

**Le score sémantique ramené sur [0, 1]** : les cosinus e5 vivent dans une bande étroite (0,80–0,89). Les bornes
(0,81 ; 0,88) sont les **percentiles 5 et 95 mesurés** sur 328 offres récupérées pour les 3 CV synthétiques — pas
des nombres magiques.

### Score de classement (tri)

Problème : une offre à 95/100 avec 40 % de couverture devrait-elle passer devant une offre à 85/100 avec 100 % ?
Non. On trie par :

```
ranking = couverture · score + (1 − couverture) · 50
```

C'est un **rétrécissement (shrinkage) vers une valeur neutre** : chaque composante inconnue compte pour 0,5.
Plus l'information manque, plus on se rapproche de 50. Idée bayésienne simple, mesurée au chapitre 8 : légèrement
meilleure que le score brut sur nos labels (nDCG@10 0,661 vs 0,646).

**Ce n'est pas une probabilité d'embauche.** Aucune donnée d'embauche n'a été observée : rien ne permet de
calibrer le score vers cet événement. L'interface et l'API le disent explicitement.

## 4.3 Les poids sont-ils justifiés ? Calibration expérimentale

Le cahier des charges demande des poids « justifiés, configurables et, si les données le permettent, calibrés ».

* **Configurables** : `config/scoring.yaml` + curseurs dans la page Recommandations (en direct).
* **Calibrés ?** `evaluation/scoring_bench.py` cherche les poids qui maximisent le nDCG@10 (3 000 tirages
  aléatoires de Dirichlet) en **validation croisée leave-one-candidate-out** : on choisit les poids sur 2 CV,
  on les évalue sur le 3e, trois fois.

Résultat réel : poids ajustés = 0,789 **sur les données d'apprentissage** mais 0,637 **hors échantillon**,
contre 0,661 pour les poids par défaut → **sur-apprentissage**. Conclusion écrite dans le rapport : on garde les
poids a priori. C'est un excellent point d'entretien : *savoir ne pas adopter un modèle qui gagne en apparence.*

Autre résultat honnête : l'ordre du retrieval seul (RRF) obtient 0,665 — le score ne fait **pas mieux** que la
recherche pour ordonner sur ces labels. Sa valeur est ailleurs : **l'explication** (chaque point du score est
attribuable à une composante, voir le graphique en cascade de la page Job Details) et la cohérence avec les
contraintes. Avec 3 candidats synthétiques, aucune différence n'est concluante.

## 4.4 Vérifier que les exclusions sont justes

Accord des exclusions avec les labels : **précision 1,0** (aucune offre éligible exclue à tort), **rappel
0,72** (8 offres inéligibles gardées — surtout des contraintes écrites seulement en texte libre, ex. « this is a
freelance position », que le système traite comme *inconnu*). Choix assumé : mieux vaut montrer une offre
douteuse que cacher une bonne offre.

## 4.5 Questions d'entretien

* *Pourquoi pas un modèle appris (learning-to-rank) ?* Pas de données d'interaction (clics, candidatures) ; 84
  paires labellisées par l'IA ne suffisent pas — l'expérience LOCO le montre. La structure est prête : les
  composantes sont exactement les features d'un LTR (LambdaMART) le jour où des données existent.
* *Pourquoi le rôle pèse 0,15 ?* Un data engineer très compétent n'est pas un bon match pour un poste de designer
  qui mentionne « SQL ». Le poids est un a priori, modifiable, et la calibration n'a pas trouvé mieux de façon robuste.
