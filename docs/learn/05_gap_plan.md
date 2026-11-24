# 5. Analyse des écarts de compétences et plan d'apprentissage

Code : `src/job_intel/match/gap.py`, base de connaissances `src/job_intel/knowledge/skills_kb.yaml`.

## 5.1 Sept statuts, du plus fort au plus faible

| Statut | Sens | Exemple |
|---|---|---|
| `demonstrated` | utilisé dans une expérience/projet du CV | « Built a churn model with scikit-learn » |
| `listed` | seulement dans la liste de compétences | « Skills: Docker » |
| `implied` | impliqué par une compétence du CV | PostgreSQL ⇒ SQL |
| `transferable` | une compétence voisine | Tableau pour Power BI |
| `self_declared` | absent du CV, mais tu dis l'avoir | → ajoute-le à ton CV ! |
| `not_mentioned` | aucune preuve dans le CV | **ce n'est pas « absent »** |
| `absent` | tu confirmes ne pas l'avoir | entre dans le plan avec priorité pleine |

## 5.2 « Absent » ou « non mentionné » ? Le système ne triche pas

Le cahier des charges demande de distinguer une compétence **réellement absente** d'une compétence **simplement
non mentionnée**. À partir du seul CV, c'est **impossible** : un CV est un résumé. Le système fait donc deux
choses honnêtes :

1. il donne un **indice** : « ton CV cite PyTorch et Keras ; si tu as utilisé TensorFlow, le CV ne le dit
   peut-être simplement pas » (compétences voisines présentes) ;
2. il **te demande** : dans la page Skill Gap, chaque compétence non mentionnée a un choix *j'ai / je n'ai pas /
   je ne sais pas*. « J'ai » → `self_declared` (et le conseil de l'ajouter au CV) ; « je n'ai pas » → `absent`.

## 5.3 Prioriser ce qu'il faut apprendre

Pour les 10 meilleures recommandations, chaque compétence manquante reçoit :

```
priorité(s) = Σ_{offres}  w_niveau(s, offre) · adéquation(offre)  ×  (1 + demande_marché(s))
```

* `w_niveau` = 1 si requise, 0,5 si souhaitée, 0,3 si seulement mentionnée (divisé par 2 si partiellement couverte) ;
* `adéquation` = score de compatibilité / 100 : les lacunes des offres **presque atteignables** comptent plus ;
* `demande_marché` = part des 720 offres qui demandent la compétence : utile au-delà de ces offres-là.

## 5.4 Ordonner : tri topologique (algorithme de Kahn)

On n'apprend pas Kubernetes avant Docker. La base de connaissances donne des **prérequis** (graphe orienté). On
ordonne les compétences retenues pour que tout prérequis vienne avant, et à égalité on prend la plus prioritaire :

```
tant qu'il reste des compétences :
    prêtes = celles dont tous les prérequis (retenus) sont déjà placés
    placer la plus prioritaire parmi les prêtes
```

S'il y avait un cycle dans la base, on retomberait sur l'ordre de priorité (garde-fou testé).

## 5.5 Ressources, projets, entretien

Pour chaque compétence : 1 à 3 **ressources** (documentation officielle ou cours gratuits reconnus — Kaggle Learn,
Hugging Face, PyTorch, docs.docker.com…), un **projet de portfolio** qui produit une preuve visible, et des
**questions d'entretien** typiques. La préparation d'entretien d'une offre met **d'abord** les questions sur les
compétences que tu revendiques : un recruteur teste ce que ton CV affirme.

## 5.6 Limites

La base de connaissances couvre 63 compétences (47 liens distincts, vérifiables avec
`python scripts/check_links.py` : 46/47 répondaient le 2026-11-22, la page Tableau refuse les robots) ; les autres
compétences reçoivent des conseils par catégorie. Les liens peuvent évoluer. Le plan est une **aide à la décision**, pas une
vérité : il dépend des offres du corpus.
