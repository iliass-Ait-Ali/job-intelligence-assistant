# 6. Intelligence du marché de l'emploi

Code : `src/job_intel/market/analytics.py` (fonctions pures sur DataFrame pandas, testées), page
« Market Intelligence », endpoints `/market/*` de l'API.

## 6.1 Segmenter d'abord

Toutes les statistiques se calculent sur un **segment** filtrable : famille de métier, pays, secteur, niveau,
contrat. `filter_frame(df, role_family=[…], country=[…])` puis n'importe quelle fonction ci-dessous.

## 6.2 Les indicateurs et leurs formules

**Part de demande** : `P(compétence) = nb d'offres qui la demandent / nb d'offres du segment`.
Sur les 720 offres : Python 68 %, machine learning 57 %, SQL 39 %, LLM 36 %, agents IA 27 %, AWS 26 %.

**Probabilité conditionnelle** (heatmap) : `P(B | A) = nb(A et B) / nb(A)` — « parmi les offres qui demandent
Spark, quelle part demande aussi Databricks ? ». La matrice n'est **pas symétrique**.

**Lift** : `lift(A, B) = P(A, B) / (P(A)·P(B))`. 1 = indépendance ; > 1 = demandées ensemble plus souvent que
le hasard. Exemple réel : Azure–GCP (lift 3,3), AWS–Azure (2,9), Databricks–Spark (2,8). Interprétation honnête :
les clouds co-apparaissent parce que les offres les citent **comme alternatives** (« AWS, Azure ou GCP ») —
le lift mesure une association, pas une complémentarité. On impose un **support minimal** (≥ 5 offres) car le
lift explose sur les compétences rares.

**Distinctivité d'une famille de métier** : `P(compétence | famille) / P(compétence)`. Python est partout (≈ 1×) ;
dbt est typique des data engineers, Power BI/Excel des data analysts. Seuil : la compétence doit apparaître dans
≥ 8 % des offres de la famille, sinon une seule offre bizarre domine (bug réel observé : « Hive » comme
compétence typique NLP à cause d'un faux positif isolé).

**Évolution temporelle** : offres par semaine de publication et part de chaque compétence par semaine. Plus une
**comparaison de deux instantanés** (23 octobre vs 22 novembre).

## 6.3 Biais à expliquer (sinon un recruteur le fera)

* **Biais d'échantillon** : 3 job boards, métiers data/IA uniquement ; Himalayas (72 % du corpus) est remote et
  très américain → « United States » domine les pays. Ce n'est pas « le marché », c'est **ce jeu de données**.
* **Biais de survie temporel** : les plateformes retirent les offres expirées ; les semaines anciennes contiennent
  donc peu d'offres (celles encore en ligne). La tendance hebdomadaire est bruitée au début.
* **Instantanés non comparables à l'identique** : le second ajoute Himalayas ; une différence mélange effet temps
  et effet source. L'interface l'écrit sous le graphique.
* **Extraction** : une compétence hors taxonomie n'est jamais comptée (rappel mesuré ≈ 0,70–0,84).

## 6.4 Choix de visualisation

Barres horizontales triées (lecture des libellés longs), une seule teinte pour une seule série (pas de légende
inutile), une rampe séquentielle à une teinte pour la heatmap (0 → 100 %), deux couleurs catégorielles fixes pour
la comparaison d'instantanés, info-bulles sur chaque barre, vue tableau disponible. Palette vérifiée pour le
daltonisme (ordre catégoriel fixe).
