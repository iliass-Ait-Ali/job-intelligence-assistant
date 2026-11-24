# 3. Du CV au profil structuré

Code : `src/job_intel/match/cv_text.py` (lecture), `cv_parser.py` (analyse), `models.py` (Pydantic),
`taxonomy.py` (compétences), `softskills.py`.

## 3.1 Lire le fichier

* **PDF** : `pypdf`, page par page. Un PDF scanné (image) est **détecté** (< 120 caractères/page) et signalé — pas
  d'OCR silencieux qui inventerait du texte.
* **DOCX** : `python-docx`, paragraphes et tableaux dans l'ordre du document ; les puces deviennent `- `, une
  ligne de tableau « Langues | Français, Anglais » devient « Langues: Français, Anglais » pour que le même parseur
  la comprenne.
* **Texte collé**. Limites de sécurité : 10 Mo, 60 000 caractères.

**Confidentialité** : le fichier est lu en mémoire puis le fichier temporaire est supprimé ; aucun nom, e-mail
ou téléphone n'est extrait (inutile pour le matching).

## 3.2 Analyser : un parseur à règles, déterministe

1. **Sections** reconnues par leurs titres en FR/EN/DE (« Expérience », « Berufserfahrung », « Projects »…).
2. **Dates** → bornes de durée : « 2023-06 – 2023-09 » → 3–4 mois. On calcule une borne **basse** (emplois
   seulement) et **haute** (emplois + stages) ; on ne tranche jamais à la place du recruteur.
3. **Diplômes** → niveau (bachelor / master / phd) et statut *en cours* uniquement si le CV le dit.
4. **Compétences** → recherchées dans toutes les lignes (hors langues et loisirs) avec la taxonomie.
5. **Langues**, **certifications**, **secteurs d'expérience** (heuristique du chapitre 1 appliquée à chaque
   expérience), **soft skills déclarées**.

Repli sans titres de section : les expériences et diplômes sont devinés à partir des dates et noms de diplômes —
avec un **avertissement** affiché.

## 3.3 Démontré vs listé

Une compétence citée **dans une expérience ou un projet** est `demonstrated` (« j'ai utilisé PyTorch pour… ») ;
citée seulement dans une liste « Compétences », elle est `listed`. Ce n'est pas un niveau de maîtrise (on n'en
infère jamais), c'est **où se trouve la preuve**. L'interface les colore différemment, l'analyse d'écarts aussi.

## 3.4 La taxonomie : une seule langue commune CV ↔ offres

~150 compétences canoniques avec alias (`sklearn` → scikit-learn, `apprentissage automatique` → Machine
learning, `Vektordatenbanken` → Vector database). **Le même vocabulaire** est utilisé côté CV et côté offre :
c'est ce qui rend « correspondance exacte » bien défini.

Mots ambigus (R, Go, C, Julia, Excel) : comptés seulement dans un contexte technique explicite
(« Python, R and SQL ») — sinon « Excel at teamwork » deviendrait la compétence Excel. Un test le garantit.

### Synonyme, relation, implication : trois choses différentes

| Relation | Exemple | Effet |
|---|---|---|
| **Synonyme** (alias) | « Postgres » = PostgreSQL | même compétence |
| **Implication** (orientée) | PostgreSQL ⇒ SQL ; PyTorch ⇒ deep learning | statut *impliqué* (couverture partielle) |
| **Proximité** (même famille) | Tableau ~ Power BI ; PyTorch ~ TensorFlow | statut *transférable* (compte pour moitié) |

C'est exactement l'exemple du cahier des charges : **PostgreSQL et SQL sont liés mais pas équivalents** — savoir
PostgreSQL implique savoir SQL, l'inverse est faux. Les implications sont dans
`src/job_intel/knowledge/skills_kb.yaml` (champ `implies`), testées dans `test_platform_core.py`.

## 3.5 Mesure

* Côté CV : 40 CV synthétiques × 10 mises en page avec vérité connue (`jobintel regress`) → diplôme, dates,
  langues à 100 % **sur ce corpus de régression** (ce n'est pas une précision sur de vrais CV).
* Côté offres : P/R/F1 sur des offres annotées (chapitre 8) : F1 = 0,80 sur l'échantillon de test.

## 3.6 Limites

Un parseur à règles ne lit que les formats qu'on lui a appris ; un CV en deux colonnes mal extrait par le PDF
peut mélanger les sections. Réponse d'ingénieur : signaler (avertissements), laisser l'utilisateur corriger
(déclarations « j'ai / je n'ai pas »), et mesurer sur de vrais CV anonymisés avant toute mise en production.
