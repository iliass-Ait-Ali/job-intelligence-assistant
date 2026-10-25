"""A terminal tool for entering human relevance judgments, one card at a time.

Design choices that protect the evaluation:
* BLINDED: cards show the query/candidate and the job, never which system returned it, at what rank or with what score.
* RESUMABLE and CRASH-SAFE: every answer is saved immediately (atomic write); re-running continues at the next
  unfinished card; `b` goes back to fix a previous answer.
* NOTHING IS PRE-FILLED: the tool only records what the person types.

Keys at any prompt: the choice keys shown, `s` skip, `b` back, `n` add a note, `?` help, `q` save and quit.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from job_intel.search.lexical import tokenize


@dataclass(frozen=True)
class Field:
    column: str
    question: str
    choices: Mapping[str, str]  # key typed -> value stored, e.g. {"0": "0", "y": "y"}
    legend: str = ""


@dataclass
class LabelSummary:
    labelled: int = 0
    skipped: int = 0
    remaining: int = 0
    quit_early: bool = False


def _is_done(row: Mapping[str, str], fields: Sequence[Field]) -> bool:
    return all((row.get(f.column) or "") != "" for f in fields)


def run_labeling(
    rows: list[dict[str, str]],
    fields: Sequence[Field],
    render_card: Callable[[dict[str, str], int, int], str],
    save: Callable[[list[dict[str, str]]], None],
    *,
    select: Callable[[dict[str, str]], bool] = lambda row: True,
    ask: Callable[[str], str] | None = None,
    show: Callable[[str], None] = print,
    help_text: str = "",
) -> LabelSummary:
    """Walk through the selected rows that are not finished, asking each field in turn."""
    ask = ask or (lambda prompt: input(prompt))  # looked up at call time (so tests can replace input)
    indices = [i for i, r in enumerate(rows) if select(r)]
    summary = LabelSummary()
    todo = [i for i in indices if not _is_done(rows[i], fields)]
    if not todo:
        show("Nothing left to label for this selection.")
        return summary
    position = 0
    order = indices  # the order cards were created in (blinded shuffle), restricted to the selection
    pending = set(todo)
    cursor = order.index(todo[0])
    history: list[int] = []
    while cursor < len(order):
        i = order[cursor]
        row = rows[i]
        if i not in pending and not (history and history[-1] == i):
            cursor += 1
            continue
        done_count = sum(1 for j in order if _is_done(rows[j], fields))
        show("\n" + "=" * 78)
        show(render_card(row, done_count, len(order)))
        answered: dict[str, str] = {}
        action = None
        for f in fields:
            while True:
                keys = "/".join(f.choices)
                reply = ask(f"\n{f.question} [{keys}]  (s skip, b back, n note, ? help, q quit) > ").strip().lower()
                if reply in f.choices:
                    answered[f.column] = f.choices[reply]
                    break
                if reply in {"s", "b", "q"}:
                    action = reply
                    break
                if reply == "n":
                    row["notes"] = ask("note > ").strip()
                    save(rows)
                    continue
                show(f.legend or help_text or "Type one of the listed keys.")
            if action:
                break
        if action == "q":
            summary.quit_early = True
            break
        if action == "s":
            summary.skipped += 1
            cursor += 1
            continue
        if action == "b":
            prev = [j for j in history if j != i]
            if prev:
                cursor = order.index(prev[-1])
                history = [j for j in history if j != prev[-1]]
                pending.add(order[cursor])
            else:
                show("Already at the first card.")
            continue
        row.update(answered)
        pending.discard(i)
        history.append(i)
        summary.labelled += 1
        save(rows)
        cursor += 1
        position += 1
    summary.remaining = sum(1 for j in order if not _is_done(rows[j], fields))
    save(rows)
    return summary


# ---------------------------------------------------------------------------------------------
# cards


def wrap(text: str, indent: str = "  ", width: int = 100) -> str:
    return textwrap.fill(re.sub(r"\s+", " ", text), width=width, initial_indent=indent, subsequent_indent=indent)


def key_lines(description: str, query: str, n: int = 3, max_chars: int = 220) -> list[str]:
    """The description lines that share the most words with `query` (a reading aid, not a ranking signal)."""
    terms = set(tokenize(query))
    scored = []
    for line in (ln.strip(" -\t") for ln in description.split("\n")):
        if len(line) < 25:
            continue
        overlap = len(terms & set(tokenize(line)))
        if overlap:
            scored.append((overlap, line[:max_chars]))
    return [line for _, line in sorted(scored, key=lambda x: -x[0])[:n]]


def job_block(record: Mapping[str, Any], title: str, query_for_lines: str, excerpt_chars: int = 900) -> str:
    desc = str(record.get("description") or "")
    lines = [
        f"JOB: {title}",
        f"     {record.get('company') or 'company n/a'} | {record.get('location') or 'location n/a'} | "
        f"{record.get('job_type') or 'type n/a'}{' | remote' if record.get('remote') else ''}",
        f"     {record.get('source_url', '')}",
        "", "  Opening of the listing:", wrap(desc[:excerpt_chars] + ("…" if len(desc) > excerpt_chars else "")),
    ]
    key = key_lines(desc, query_for_lines)
    if key:
        lines += ["", "  Lines that mention the query's words:"] + [wrap(f"- {k}", "    ") for k in key]
    return "\n".join(lines)


def retrieval_card(queries: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], rubric: str):
    def render(row: dict[str, str], done: int, total: int) -> str:
        q = queries[row["query_id"]]
        head = [f"[{row['query_id']}  card {done + 1}/{total} in this selection]",
                f"QUERY ({q.lang or '?'}): {q.text}", f"  eligibility filters: {q.filters.describe()}", f"  {rubric}", ""]  # fmt: skip
        return "\n".join(head) + job_block(records.get(row["job_id"], {}), row["title"], q.text)

    return render


def matching_card(candidates: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], rubric: str):
    def render(row: dict[str, str], done: int, total: int) -> str:
        cand = candidates[row["query_id"]]
        p, prefs = cand.profile, cand.prefs
        exp = "; ".join(f"{e.title or '?'} ({e.start or '?'}→{e.end or '?'})" for e in p.experiences[:4]) or "none found"
        edu = "; ".join(f"{e.degree or '?'}{' [in progress]' if e.in_progress else ''}" for e in p.education[:3]) or "none found"
        head = [
            f"[{row['query_id']}  card {done + 1}/{total} in this selection]",
            f"CANDIDATE: {p.headline or '(no headline)'}",
            wrap(f"Target roles: {', '.join(prefs.target_roles or p.target_roles) or 'not stated'}"),
            wrap(f"STATED PREFERENCES: job types {prefs.job_types or 'any'}; locations {prefs.locations or 'any'}; remote {prefs.remote}"),
            wrap(f"Education: {edu}"), wrap(f"Experience: {exp}"),
            wrap("Skills in the CV: " + ", ".join(s.name for s in p.skills[:25])),
            wrap("Languages: " + (", ".join(s.language for s in p.spoken_languages) or "none listed")), f"  {rubric}", "",
        ]  # fmt: skip
        return "\n".join(head) + job_block(records.get(row["job_id"], {}), row["title"], " ".join(s.name for s in p.skills))

    return render


def skill_card(row: dict[str, str], done: int, total: int) -> str:
    claim = ("The system says the CV EXACTLY shows this skill." if row["claimed"] == "exact"
             else "The system says the CV shows a RELATED (not identical) skill that transfers.")  # fmt: skip
    return "\n".join([
        f"[{row['candidate_id']}  assertion {done + 1}/{total}]",
        f"SKILL REQUIRED BY THE LISTING: {row['skill']}", f"  {claim}", "",
        "  CV evidence:", wrap(row["cv_evidence"] or "(none)", "    "), "  Listing line:", wrap(row["job_evidence"], "    "),
    ])  # fmt: skip


GRADE_FIELD = Field("grade", "Grade", {"0": "0", "1": "1", "2": "2"}, "0 = not relevant, 1 = partially relevant, 2 = highly relevant")
ROLE_FIELD = Field("role_ok", "Does the job's ROLE match the candidate's target role / field?", {"y": "y", "n": "n", "?": "?"})
ELIGIBLE_FIELD = Field("eligible_ok", "Given the STATED PREFERENCES (type, location, remote), is this job acceptable?", {"y": "y", "n": "n", "?": "?"},
                       "y = acceptable, n = violates a stated preference, ? = cannot tell from the listing")  # fmt: skip
SKILL_FIELD = Field("correct", "Is the system's claim correct?", {"y": "y", "n": "n"}, "y = yes, n = no")
