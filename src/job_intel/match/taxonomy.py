"""A curated skill taxonomy shared by the CV parser and the job-requirement extractor.

Using ONE vocabulary on both sides is what makes "exact match", "transferable" and "not evidenced"
well defined: a skill is a canonical name; a job and a CV both mention it through aliases.

* ``exact``       the CV states the same canonical skill.
* ``transferable`` the CV states a different skill from the same *related group* (e.g. PyTorch vs
                   TensorFlow, Power BI vs Tableau). It is evidence of a related skill, never of this one.
* ``not evidenced`` the CV text does not mention it (it may still exist: the CV is the only source).

Ambiguous short names (R, Go, C) are only counted in a technical context, see ``Skill.ambiguous``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# category names
LANG, FRAMEWORK, DATA, DB, BI, CLOUD, GENAI, CONCEPT, TOOL = (
    "programming_language", "framework_library", "data_platform", "database", "bi_tool", "cloud_devops",
    "genai", "concept", "tool",
)  # fmt: skip


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...] = ()
    ambiguous: bool = False  # short/common word: counted only in a technical context
    case_sensitive: bool = False
    pattern: re.Pattern[str] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        alts = "|".join(re.escape(a) for a in sorted({self.name, *self.aliases}, key=len, reverse=True))
        flags = 0 if self.case_sensitive else re.IGNORECASE
        # not preceded/followed by a letter, digit, '+', '#', '.', or a hyphen joining two words
        pattern = re.compile(rf"(?<![\w+#.])(?:{alts})(?![\w+#&]|\.\w|-\w)", flags)
        object.__setattr__(self, "pattern", pattern)


def _s(name: str, category: str, *aliases: str, ambiguous: bool = False, case_sensitive: bool = False) -> Skill:
    return Skill(name, category, tuple(aliases), ambiguous, case_sensitive)


SKILLS: tuple[Skill, ...] = (
    # languages
    _s("Python", LANG, "python3", "python 3"),
    _s("R", LANG, "R programming", "langage R", "RStudio", ambiguous=True, case_sensitive=True),
    _s("SQL", LANG, "T-SQL", "PL/SQL", "SQL queries", "requêtes SQL"),
    _s("Java", LANG), _s("JavaScript", LANG, "ECMAScript", "Node.js", "NodeJS"),
    _s("TypeScript", LANG), _s("C++", LANG, "cpp"), _s("C#", LANG, "csharp"),
    _s("C", LANG, ambiguous=True, case_sensitive=True),
    _s("Go", LANG, "Golang", ambiguous=True, case_sensitive=True),
    _s("Rust", LANG), _s("Scala", LANG), _s("Julia", LANG, ambiguous=True, case_sensitive=True), _s("MATLAB", LANG), _s("PHP", LANG), _s("Kotlin", LANG),
    _s("Bash", LANG, "shell scripting", "shell script"), _s("SAS", LANG, ambiguous=True, case_sensitive=True),
    _s("VBA", LANG),
    # ML / DL libraries
    _s("scikit-learn", FRAMEWORK, "sklearn", "scikit learn"), _s("PyTorch", FRAMEWORK, "pytorch lightning"),
    _s("TensorFlow", FRAMEWORK, "tf.keras"), _s("Keras", FRAMEWORK), _s("JAX", FRAMEWORK, case_sensitive=True),
    _s("XGBoost", FRAMEWORK), _s("LightGBM", FRAMEWORK), _s("CatBoost", FRAMEWORK),
    _s("Hugging Face", FRAMEWORK, "huggingface", "transformers library", "HF transformers"),
    _s("spaCy", FRAMEWORK), _s("NLTK", FRAMEWORK), _s("OpenCV", FRAMEWORK), _s("ONNX", FRAMEWORK),
    _s("MLflow", TOOL), _s("Weights & Biases", TOOL, "wandb"), _s("Optuna", TOOL),
    _s("pandas", FRAMEWORK), _s("NumPy", FRAMEWORK), _s("SciPy", FRAMEWORK), _s("Polars", FRAMEWORK),
    _s("Matplotlib", FRAMEWORK), _s("Seaborn", FRAMEWORK), _s("Plotly", FRAMEWORK), _s("statsmodels", FRAMEWORK),
    # data engineering
    _s("Spark", DATA, "Apache Spark", "PySpark", "Spark SQL"), _s("Hadoop", DATA), _s("Hive", DATA),
    _s("Kafka", DATA, "Apache Kafka"), _s("Flink", DATA, "Apache Flink"), _s("Airflow", DATA, "Apache Airflow"),
    _s("dbt", DATA, "data build tool", case_sensitive=True), _s("Dagster", DATA), _s("Prefect", DATA),
    _s("Databricks", DATA), _s("Snowflake", DATA), _s("BigQuery", DATA), _s("Redshift", DATA),
    _s("Synapse", DATA, "Azure Synapse"), _s("Delta Lake", DATA), _s("NiFi", DATA, "Apache NiFi"),
    _s("Talend", DATA), _s("ETL", CONCEPT, "ELT", "ETL/ELT", "data pipelines", "data pipeline", "pipelines de données"),
    _s("Data warehouse", CONCEPT, "data warehousing", "entrepôt de données", "datawarehouse", "Datenlager"),
    _s("Data modeling", CONCEPT, "data modelling", "modélisation de données", "dimensional modeling"),
    # databases
    _s("PostgreSQL", DB, "Postgres"), _s("MySQL", DB, "MariaDB"), _s("SQL Server", DB, "MSSQL", "Microsoft SQL Server"),
    _s("Oracle", DB, "Oracle DB", "Oracle Database"), _s("SQLite", DB), _s("MongoDB", DB), _s("Redis", DB),
    _s("Elasticsearch", DB, "OpenSearch"), _s("Neo4j", DB), _s("Cassandra", DB),
    _s("ChromaDB", DB, "Chroma"), _s("FAISS", DB), _s("Pinecone", DB), _s("Weaviate", DB), _s("Qdrant", DB),
    _s("Vector database", GENAI, "vector databases", "vector store", "base de données vectorielle"),
    # BI
    _s("Power BI", BI, "PowerBI"), _s("Tableau", BI), _s("Looker", BI, "Looker Studio", "Data Studio"),
    _s("Qlik", BI, "QlikView", "Qlik Sense"), _s("Superset", BI, "Apache Superset"), _s("Metabase", BI),
    _s("Excel", BI, "Microsoft Excel", ambiguous=True, case_sensitive=True), _s("Grafana", BI),
    # cloud / devops / software
    _s("AWS", CLOUD, "Amazon Web Services", "SageMaker"), _s("Azure", CLOUD, "Microsoft Azure"),
    _s("GCP", CLOUD, "Google Cloud", "Google Cloud Platform", "Vertex AI"),
    _s("Docker", CLOUD), _s("Kubernetes", CLOUD, "k8s"), _s("Terraform", CLOUD), _s("Git", CLOUD, "GitHub", "GitLab"),
    _s("CI/CD", CLOUD, "continuous integration", "GitHub Actions", "GitLab CI", "Jenkins"),
    _s("Linux", CLOUD, "Unix"), _s("FastAPI", FRAMEWORK), _s("Flask", FRAMEWORK), _s("Django", FRAMEWORK),
    _s("REST API", FRAMEWORK, "REST APIs", "RESTful", "API REST"), _s("Streamlit", FRAMEWORK), _s("Gradio", FRAMEWORK),
    _s("React", FRAMEWORK, case_sensitive=True, ambiguous=True), _s("Spring Boot", FRAMEWORK),
    # GenAI
    _s("LLM", GENAI, "LLMs", "large language models", "large language model", "grands modèles de langage"),
    _s("RAG", GENAI, "retrieval augmented generation", "retrieval-augmented generation", case_sensitive=True),
    _s("LangChain", GENAI), _s("LlamaIndex", GENAI), _s("OpenAI API", GENAI, "GPT-4", "ChatGPT API"),
    _s("Prompt engineering", GENAI, "prompt design"), _s("Fine-tuning", GENAI, "finetuning", "fine tuning", "LoRA"),
    _s("Embeddings", GENAI, "sentence-transformers", "sentence transformers", "text embeddings"),
    _s("AI agents", GENAI, "agentic", "LLM agents", "agents LLM", "multi-agent"),
    # concepts
    _s("Machine learning", CONCEPT, "ML", "apprentissage automatique", "machine-learning", "maschinelles Lernen",
       "apprentissage statistique"),
    _s("Deep learning", CONCEPT, "apprentissage profond", "neural networks", "réseaux de neurones", "CNN", "RNN", "LSTM"),
    _s("NLP", CONCEPT, "natural language processing", "traitement du langage naturel", "text mining",
       "Sprachverarbeitung", "computational linguistics"),
    _s("Computer vision", CONCEPT, "vision par ordinateur", "image processing", "traitement d'images", "Bildverarbeitung"),
    _s("Statistics", CONCEPT, "statistical modeling", "statistical analysis", "statistiques", "Statistik",
       "statistical modelling", "hypothesis testing", "tests d'hypothèses"),
    _s("Time series", CONCEPT, "séries temporelles", "Zeitreihen", "time-series"),
    _s("Forecasting", CONCEPT, "demand forecasting", "prévision", "prévisions", "Prognose"),
    _s("A/B testing", CONCEPT, "AB testing", "experimentation", "tests A/B", "experimental design"),
    _s("Data visualization", CONCEPT, "data visualisation", "dataviz", "visualisation de données", "Datenvisualisierung"),
    _s("Data analysis", CONCEPT, "data analytics", "analyse de données", "Datenanalyse", "exploratory data analysis", "EDA"),
    _s("Feature engineering", CONCEPT), _s("MLOps", CONCEPT, "ML ops", "model deployment", "model serving", "déploiement de modèles"),
    _s("Recommender systems", CONCEPT, "recommendation systems", "systèmes de recommandation", "recommender"),
    _s("Reinforcement learning", CONCEPT, "apprentissage par renforcement"),
    _s("Data engineering", CONCEPT, "ingénierie des données", "Dateningenieur"),
    _s("Agile", TOOL, "Scrum", "Kanban"),
)

BY_NAME = {s.name: s for s in SKILLS}

# Skills of one group are *related*, not equal. Used only for "transferable" evidence.
RELATED_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(g)
    for g in (
        {"PyTorch", "TensorFlow", "Keras", "JAX"},
        {"XGBoost", "LightGBM", "CatBoost", "scikit-learn"},
        {"Machine learning", "Deep learning", "scikit-learn", "PyTorch", "TensorFlow", "Keras", "XGBoost"},
        {"Power BI", "Tableau", "Looker", "Qlik", "Superset", "Metabase", "Grafana"},
        {"Airflow", "Dagster", "Prefect"},
        {"AWS", "Azure", "GCP"},
        {"Snowflake", "BigQuery", "Redshift", "Synapse", "Databricks", "Data warehouse"},
        {"SQL", "PostgreSQL", "MySQL", "SQL Server", "Oracle", "SQLite"},
        {"PostgreSQL", "MySQL", "SQL Server", "Oracle", "SQLite", "MongoDB"},
        {"Spark", "Hadoop", "Hive", "Flink", "Databricks"},
        {"Kafka", "Flink", "Spark"},
        {"ChromaDB", "FAISS", "Pinecone", "Weaviate", "Qdrant", "Vector database", "Elasticsearch"},
        {"LangChain", "LlamaIndex", "RAG", "LLM", "AI agents"},
        {"LLM", "NLP", "Hugging Face", "Embeddings", "Fine-tuning", "Prompt engineering"},
        {"Flask", "FastAPI", "Django", "REST API"},
        {"Streamlit", "Gradio"},
        {"pandas", "Polars", "NumPy"},
        {"spaCy", "NLTK", "Hugging Face", "NLP"},
        {"Java", "Scala", "Kotlin"},
        {"JavaScript", "TypeScript"},
        {"C++", "C", "Rust", "Go"},
        {"Matplotlib", "Seaborn", "Plotly", "Data visualization"},
        {"ETL", "Airflow", "dbt", "NiFi", "Talend", "Data engineering"},
        {"Statistics", "Data analysis", "A/B testing", "Time series", "statsmodels"},
        {"Forecasting", "Time series", "Statistics"},
        {"Docker", "Kubernetes", "CI/CD", "Terraform", "MLOps"},
        {"Computer vision", "OpenCV", "Deep learning"},
    )
)

_SEP_BEFORE = re.compile(r"(?:[,;/|]|\b(?:and|et|und|or|ou|oder)\b)\s*$", re.I)
_SEP_AFTER = re.compile(r"^\s*(?:[,;/|]|\b(?:and|et|und|or|ou|oder)\b)", re.I)


def find_skills(text: str) -> list[str]:
    """Canonical skills mentioned in `text`, in order of first appearance, without duplicates."""
    return [name for name, _ in find_skill_spans(text)]


def find_skill_spans(text: str) -> list[tuple[str, int]]:
    """(canonical skill, start offset) pairs ordered by position, at most one per skill.

    Ambiguous names ("R", "Go", "C", "Julia", "Excel") only count when written explicitly ("Golang",
    "R programming") or when they sit in a list next to another technical term ("Python, R and SQL").
    """
    found: dict[str, int] = {}
    offset = 0
    for line in text.split("\n"):
        hits = [(skill, m) for skill in SKILLS if (m := skill.pattern.search(line))]
        has_plain = any(not skill.ambiguous for skill, _ in hits)
        for skill, m in hits:
            if skill.ambiguous and m.group(0).casefold() == skill.name.casefold():
                in_list = bool(_SEP_BEFORE.search(line[: m.start()]) or _SEP_AFTER.search(line[m.end() :]))
                if not (has_plain and in_list):
                    continue
            found.setdefault(skill.name, offset + m.start())
        offset += len(line) + 1
    return sorted(found.items(), key=lambda kv: kv[1])


def closest_related(name: str, available: set[str] | frozenset[str]) -> list[str]:
    """Skills of `available` related to `name`, taken from the SMALLEST related group that has any
    (PyTorch for TensorFlow, not "machine learning" in general), at most three, alphabetical."""
    for group in sorted((g for g in RELATED_GROUPS if name in g), key=len):
        found = sorted((group & set(available)) - {name})
        if found:
            return found[:3]
    return []


def related_skills(name: str) -> set[str]:
    """Skills related to `name` through any related group (excluding itself)."""
    out: set[str] = set()
    for group in RELATED_GROUPS:
        if name in group:
            out |= group
    out.discard(name)
    return out


def category_of(name: str) -> str:
    return BY_NAME[name].category if name in BY_NAME else TOOL
