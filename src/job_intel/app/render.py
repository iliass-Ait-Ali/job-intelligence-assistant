"""HTML rendering for the interface. Everything that comes from a CV or a listing is HTML-escaped,
and links are only produced for http(s) URLs taken from the listing metadata."""

from __future__ import annotations

from html import escape
from urllib.parse import urlsplit

from job_intel.match.matcher import JobMatch, MatchResult
from job_intel.match.models import CandidateProfile
from job_intel.rag.compare import Comparison
from job_intel.rag.explain import Explanation

CSS = """
<style>
.jm-card{border:1px solid var(--border-color-primary,#ccc);border-radius:10px;padding:12px 14px;margin:10px 0}
.jm-head{font-size:1.05em}.jm-rank{opacity:.6;margin-right:6px}
.jm-meta{opacity:.8;font-size:.9em;margin:2px 0 8px}
.jm-badge{display:inline-block;border-radius:6px;padding:1px 8px;font-size:.82em;border:1px solid currentColor;margin-right:6px}
.jm-strong{color:#1a7f37}.jm-moderate{color:#9a6700}.jm-weak{color:#8250df}.jm-insufficient_information{color:#6e7781}
.jm-chip{display:inline-block;border-radius:12px;padding:0 9px;margin:2px 3px 2px 0;font-size:.85em;background:var(--background-fill-secondary,#eee)}
.jm-exact{border:1px solid #1a7f37}.jm-transf{border:1px dashed #9a6700}.jm-miss{border:1px dotted #cf222e}
.jm-ok{color:#1a7f37}.jm-bad{color:#cf222e}.jm-unk{color:#6e7781}
.jm-note{opacity:.75;font-size:.85em}
.jm-cite{font-size:.8em;opacity:.7;margin-left:4px}
table.jm-table{border-collapse:collapse;width:100%}table.jm-table td,table.jm-table th{border:1px solid var(--border-color-primary,#ccc);padding:5px 8px;vertical-align:top;text-align:left}
</style>
"""

LEVEL_TEXT = {
    "strong": "strong evidence", "moderate": "moderate evidence", "weak": "weak evidence",
    "insufficient_information": "not enough information",
}
STATUS_ICON = {"satisfied": ("✔", "jm-ok"), "violated": ("✖", "jm-bad"), "unknown": ("?", "jm-unk")}


def e(value: object) -> str:
    return escape(str(value), quote=True)


def safe_link(url: str | None, label: str) -> str:
    parts = urlsplit(url or "")
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return e(label)
    return f'<a href="{e(url)}" target="_blank" rel="noopener noreferrer">{e(label)}</a>'


def chips(items: list[str], css: str) -> str:
    return "".join(f'<span class="jm-chip {css}">{e(i)}</span>' for i in items) or '<span class="jm-note">none</span>'


def render_profile_summary(profile: CandidateProfile) -> str:
    def row(label: str, value: str) -> str:
        return f"<tr><th>{e(label)}</th><td>{value}</td></tr>"

    exp = profile.experience_months_bounds()
    rows = [
        row("Detected language", e(profile.detected_language)),
        row("Headline", e(profile.headline or "—")),
        row("Target roles stated in the CV", e(", ".join(profile.target_roles) or "none stated")),
        row("Education", "<br>".join(
            e(f"{ed.degree or '?'}{f' — {ed.institution}' if ed.institution else ''} "
              f"({ed.start or '?'}–{ed.end or '?'}{', in progress' if ed.in_progress else ''})") for ed in profile.education) or "none found"),
        row("Programming languages", chips(profile.programming_languages, "jm-exact")),
        row("Frameworks & technologies", chips(profile.frameworks_and_technologies, "jm-exact")),
        row("Concepts", chips([s.name for s in profile.skills if s.category == "concept"], "jm-exact")),
        row("Experience entries", "<br>".join(
            e(f"{x.title or '?'} @ {x.organization or '?'} [{x.kind}] {x.start or '?'}→{x.end or '?'}") for x in profile.experiences) or "none found"),
        row("Projects", "<br>".join(e(p.name or "(unnamed)") for p in profile.projects) or "none found"),
        row("Spoken languages", e(", ".join(f"{s.language}{f' ({s.level_text})' if s.level_text else ''}" for s in profile.spoken_languages) or "none found")),
        row("Stated experience (from dates only)", e(f"{exp[0]}–{exp[1]} months" if exp and exp[1] < 9999 else ("dates incomplete" if exp else "no dates found"))),
    ]
    return f"{CSS}<table class='jm-table'>{''.join(rows)}</table>"


def render_constraints(match: JobMatch) -> str:
    if not match.constraints:
        return '<span class="jm-note">No preference or requirement could be checked.</span>'
    out = []
    for c in match.constraints:
        icon, css = STATUS_ICON[c.status]
        hard = " (hard)" if c.hard else ""
        out.append(f'<div class="{css}">{icon} <b>{e(c.name)}{hard}</b>: {e(c.detail)}</div>')
    return "".join(out)


def render_job_card(rank: int, m: JobMatch) -> str:
    job = m.job
    counts = m.counts
    exact = [s.skill for s in m.skills if s.status == "exact"]
    transf = [f"{s.skill} ← {', '.join(s.via)}" for s in m.skills if s.status == "transferable"]
    missing = [f"{s.skill}{' (required)' if s.job_level == 'required' else ''}" for s in m.skills if s.status == "not_evidenced"]
    published = job.published_date.isoformat() if job.published_date else "date n/a"
    meta = " • ".join(e(x) for x in (job.location or "location n/a", job.job_type or "type n/a", job.source, published))
    roles = f" — matches your target role ({e(', '.join(m.matched_roles))})" if m.role_status == "match" else (
        " — title differs from your target roles" if m.role_status == "no_match" else "")
    evidence_rows = "".join(
        f"<li><b>{e(s.skill)}</b> — CV: “{e(s.cv_evidence[0].text if s.cv_evidence else '')}” · listing: “{e(s.job_evidence)}”</li>"
        for s in m.skills if s.status == "exact")
    notes = "".join(f'<div class="jm-note">ⓘ {e(n)}</div>' for n in m.notes)
    return f"""
<div class="jm-card">
 <div class="jm-head"><span class="jm-rank">#{rank}</span><b>{e(job.title)}</b> — {e(job.company or 'company n/a')}</div>
 <div class="jm-meta">{meta} · {safe_link(job.url, 'Original listing ↗')}</div>
 <div><span class="jm-badge jm-{e(m.evidence_level)}" title="Heuristic ordering from explicit rules; not a probability of being hired.">{e(LEVEL_TEXT[m.evidence_level])}</span>
 <span class="jm-note">{counts['exact']} exact · {counts['transferable']} transferable · {counts['not_evidenced']} not evidenced (of {len(m.base_skills)} skills judged){roles}</span></div>
 <div><b>✔ In your CV</b> {chips(exact, 'jm-exact')}</div>
 <div><b>≈ Related skill in your CV</b> {chips(transf, 'jm-transf')}</div>
 <div><b>? Not evidenced in the CV text</b> {chips(missing, 'jm-miss')}</div>
 <div style="margin-top:6px"><b>Eligibility &amp; requirements</b>{render_constraints(m)}</div>
 <details><summary>Matching evidence</summary><ul>{evidence_rows or '<li class="jm-note">No exact skill match.</li>'}</ul></details>
 {notes}
</div>"""


def render_results(result: MatchResult) -> str:
    parts = [CSS]
    queries = "".join(f"<li>{e(q.label)}</li>" for q in result.queries)
    parts.append(
        f"<p><b>{len(result.matches)}</b> recommendations. Of {result.corpus_size or result.pool_size} indexed listings, "
        f"{len(result.excluded)} are excluded by your hard constraints and {result.eligible_total or result.pool_size} remain eligible "
        f"({result.pool_size} retrieved as candidates). Ordering = evidence level, then role match, then unmet "
        f"requirements, then skill coverage, then retrieval rank. There is no percentage score: it could not be calibrated.</p>"
        f"<details><summary>Searches built from your CV ({len(result.queries)})</summary><ul>{queries}</ul></details>")
    parts += [f'<p class="jm-note">⚠ {e(w)}</p>' for w in result.warnings]
    parts += [render_job_card(i, m) for i, m in enumerate(result.matches, 1)]
    if result.excluded:
        items = "".join(f"<li>{e(m.job.title)} — {e(m.job.company or '')}: {e(m.excluded_because[0])}</li>" for m in result.excluded[:30])
        parts.append(f"<details><summary>Excluded by hard constraints ({len(result.excluded)})</summary><ul>{items}</ul></details>")
    return "".join(parts)


def _cites(cv_refs: list[str], job_refs: list[str]) -> str:
    refs = [*cv_refs, *job_refs]
    return f'<span class="jm-cite">[{e(", ".join(refs))}]</span>' if refs else ""


def render_explanation(exp: Explanation, match: JobMatch) -> str:
    mode = (f"Written by local model <b>{e(exp.model)}</b>; every item was checked against the cited evidence."
            if exp.mode == "llm" else "Deterministic explanation — <b>no language model was used</b>.")
    out = [CSS, f"<h4>{e(exp.job_title)} — {e(match.job.company or '')}</h4>", f'<p class="jm-note">{mode} '
           f"{exp.claims_verified} of {exp.claims_proposed} proposed items passed verification.</p>"]
    out += [f'<p class="jm-note">ⓘ {e(n)}</p>' for n in exp.notes]
    out.append("<h5>Why it matches (with evidence)</h5><ul>" + "".join(
        f"<li>{e(s.claim)}{_cites(s.cv_refs, s.job_refs)}</li>" for s in exp.strengths) + "</ul>" if exp.strengths else
        "<p class='jm-note'>No verifiable match statement.</p>")
    if exp.gaps:
        out.append("<h5>Missing or unverified requirements</h5><ul>" + "".join(
            f"<li>{e(g.requirement)} {('— ' + e(g.note)) if g.note else ''}{_cites(g.cv_refs, g.job_refs)}</li>" for g in exp.gaps) + "</ul>")
    if exp.preparation:
        out.append("<h5>How to prepare</h5><ul>" + "".join(
            f"<li>{e(p.suggestion)}{_cites(p.cv_refs, p.job_refs)}</li>" for p in exp.preparation) + "</ul>")
    cv = "".join(f"<li><b>{e(s.id)}</b> {e(s.text)}</li>" for s in exp.pack.cv)
    job = "".join(f"<li><b>{e(s.id)}</b> {e(s.text)}</li>" for s in exp.pack.job)
    out.append(f"<details><summary>Evidence used ({len(exp.pack.cv)} CV snippets, {len(exp.pack.job)} listing snippets)</summary>"
               f"<b>From your CV</b><ul>{cv}</ul><b>From the listing</b><ul>{job}</ul></details>")
    if exp.rejected:
        rej = "".join(f"<li>{e(r.text)} <span class='jm-note'>— removed: {e(r.reason)}</span></li>" for r in exp.rejected)
        out.append(f"<details><summary>{len(exp.rejected)} item(s) removed by verification</summary><ul>{rej}</ul></details>")
    out.append("<h5>Eligibility &amp; requirements</h5>" + render_constraints(match))
    out.append(f"<p>{safe_link(match.job.url, 'Open the original listing to apply ↗')}</p>")
    return "".join(out)


def render_comparison(comp: Comparison) -> str:
    header = "".join(f"<th>{e(r['label'])}: {e(r['title'])}<br><span class='jm-note'>{e(r['company'])}</span></th>" for r in comp.rows)

    def cells(fn) -> str:
        return "".join(f"<td>{fn(r)}</td>" for r in comp.rows)

    table = (
        f"<table class='jm-table'><tr><th></th>{header}</tr>"
        f"<tr><th>Location / type</th>{cells(lambda r: e(r['location'] + ' · ' + r['job_type']))}</tr>"
        f"<tr><th>Evidence level</th>{cells(lambda r: e(LEVEL_TEXT[r['evidence_level']]))}</tr>"
        f"<tr><th>Skills exact / related / not evidenced</th>{cells(lambda r: e(str(r['exact']) + ' / ' + str(r['transferable']) + ' / ' + str(r['not_evidenced'])))}</tr>"
        f"<tr><th>Required, not evidenced</th>{cells(lambda r: e(', '.join(r['top_missing']) or '—'))}</tr>"
        f"<tr><th>Constraints not met or unknown</th>{cells(lambda r: e('; '.join(r['constraints_not_met']) or '—'))}</tr>"
        f"<tr><th>Listing</th>{cells(lambda r: safe_link(r['url'], 'open ↗'))}</tr></table>")
    out = [CSS, table]
    unique = "".join(f"<li><b>{e(k)}</b>: {e(', '.join(v) or '—')}</li>" for k, v in comp.unique_strengths.items())
    out.append(f"<h5>Facts</h5><ul><li>Skills evidenced for only one of the jobs:<ul>{unique}</ul></li>"
               f"<li>Listed by every job but not evidenced in the CV: {e(', '.join(comp.common_gaps) or '—')}</li></ul>")
    if comp.narrative:
        out.append("<h5>Narrative (verified against the evidence)</h5><ul>" + "".join(
            f"<li>{e(p.statement)}{_cites(p.cv_refs, p.job_refs)}</li>" for p in comp.narrative) + "</ul>")
    out += [f'<p class="jm-note">ⓘ {e(n)}</p>' for n in comp.notes]
    if comp.rejected:
        out.append(f"<p class='jm-note'>{len(comp.rejected)} narrative point(s) removed by verification.</p>")
    return "".join(out)
