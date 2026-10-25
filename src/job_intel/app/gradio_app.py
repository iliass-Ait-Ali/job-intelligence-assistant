"""The Gradio interface. Handlers are thin: they call controller.py and render.py and turn any failure
into a readable message instead of a stack trace."""

from __future__ import annotations

import os

from job_intel.app import controller as ctl
from job_intel.app import render
from job_intel.app.controller import AppContext

PRIVACY = (
    "🔒 **Privacy:** your CV is processed on this computer only. It is kept in memory for this session, uploaded files are "
    "deleted after reading, and text is only sent to a language model running locally through Ollama (if you enable it). "
    "Nothing is saved or sent to an online AI service."
)


def build_app(ctx: AppContext):
    import gradio as gr

    def note(kind: str, text: str) -> str:
        icon = {"error": "❌", "warn": "⚠️", "ok": "✅"}[kind]
        return f"{icon} {text}"

    # ---- handlers -----------------------------------------------------------------------------

    def on_parse(file_path, pasted):
        try:
            parsed = ctl.parse_cv_input(file_path, pasted)
        except Exception as exc:  # never show a stack trace to the user
            return gr.skip(), gr.skip(), gr.skip(), note("error", ctl.user_message_for(exc))
        if parsed.error:
            return gr.skip(), gr.skip(), gr.skip(), note("error", parsed.error)
        msgs = "\n\n".join(note("warn", m) for m in parsed.messages) or note("ok", "CV parsed. Review the profile below and correct it if needed.")
        return parsed.profile, parsed.json_text, render.render_profile_summary(parsed.profile), msgs

    def on_apply(json_text):
        profile, error = ctl.profile_from_json(json_text)
        if error:
            return gr.skip(), gr.skip(), note("error", error)
        return profile, render.render_profile_summary(profile), note("ok", "Profile updated.")

    def on_recommend(profile, roles, job_types, locations, remote, hard, max_results):
        if profile is None:
            return None, "", gr.update(choices=[], value=None), gr.update(choices=[], value=[]), note("error", "Parse or paste a CV first (tab 1).")
        try:
            prefs = ctl.build_preferences(roles, job_types, locations, remote, hard, max_results)
            result = ctl.recommend(ctx, profile, prefs)
        except Exception as exc:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), note("error", ctl.user_message_for(exc))
        labels = ctl.job_labels(result)
        status = note("ok", f"{len(result.matches)} recommendations.") if result.matches else note(
            "warn", "No eligible job found. Relax the hard constraints (unknown values never exclude a job).")
        return (result, render.render_results(result), gr.update(choices=labels, value=labels[0] if labels else None),
                gr.update(choices=labels, value=[]), status)

    def on_explain(profile, result, label, use_llm, language):
        if profile is None or result is None:
            return note("error", "Get recommendations first (tab 3).")
        try:
            exp = ctl.explain_job(ctx, profile, result, label, use_llm=use_llm, language=language)
            match = result.matches[ctl._index_from_label(label, len(result.matches))]
        except Exception as exc:
            return note("error", ctl.user_message_for(exc))
        return render.render_explanation(exp, match)

    def on_compare(profile, result, labels, use_llm, language):
        if profile is None or result is None:
            return note("error", "Get recommendations first (tab 3).")
        try:
            comp = ctl.compare_jobs(ctx, profile, result, labels or [], use_llm=use_llm, language=language)
        except Exception as exc:
            return note("error", ctl.user_message_for(exc))
        return render.render_comparison(comp)

    def on_llm_status():
        ok, message = ctx.llm_status()
        return note("ok" if ok else "warn", message)

    # ---- layout -------------------------------------------------------------------------------

    with gr.Blocks(title="Job & Internship Intelligence Assistant", analytics_enabled=False) as demo:
        gr.Markdown("# Job & Internship Intelligence Assistant\nUpload a CV, set your preferences and get **explainable** recommendations "
                    "from the local job database. Recommendations are ordered by transparent rules; there is no hiring-probability score.")
        gr.Markdown(PRIVACY)
        profile_state = gr.State(None)
        result_state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("1 · Candidate profile"):
                with gr.Row():
                    with gr.Column():
                        cv_file = gr.File(label="CV (PDF or .txt, text-based; scanned PDFs are not supported)",
                                          file_types=[".pdf", ".txt", ".md"], type="filepath")  # fmt: skip
                        cv_text = gr.Textbox(label="…or paste the CV text (French or English)", lines=10)
                        parse_btn = gr.Button("Analyse CV", variant="primary")
                        parse_status = gr.Markdown()
                    with gr.Column():
                        profile_html = gr.HTML(label="Extracted profile")
                with gr.Accordion("Inspect and correct the extracted profile (JSON)", open=False):
                    gr.Markdown("Edit anything that was extracted wrongly, then press **Apply edits**. Nothing is invented: "
                                "what you remove or add here is what the matching uses.")
                    profile_json = gr.Code(language="json", lines=18, interactive=True, label="Profile")
                    apply_btn = gr.Button("Apply edits")
                    apply_status = gr.Markdown()

            with gr.Tab("2 · Preferences"):
                roles = gr.Textbox(label="Target roles (comma-separated; empty = use roles stated in the CV)",
                                   placeholder="Data Scientist, Machine Learning Engineer")  # fmt: skip
                job_types = gr.CheckboxGroup(ctl.JOB_TYPE_CHOICES, label="Job types (empty = any)")
                locations = gr.Textbox(label="Preferred locations (city, region or country; comma-separated)", placeholder="Paris, Switzerland")
                remote = gr.Radio(["No preference", "Prefer remote", "Remote only"], value="No preference", label="Remote work",
                                  info="Listings rarely say 'on-site', so 'Remote only' keeps only listings that explicitly mention remote work.")  # fmt: skip
                hard = gr.CheckboxGroup(ctl.HARD_CHOICES, value=["job_type", "location"], label="Hard constraints",
                                        info="A job is excluded only if it CLEARLY violates one of these. Missing information never excludes a job; other checks are shown as warnings.")  # fmt: skip
                max_results = gr.Slider(5, 40, value=15, step=1, label="Maximum recommendations")

            with gr.Tab("3 · Recommendations"):
                find_btn = gr.Button("Find matching jobs", variant="primary")
                find_status = gr.Markdown()
                results_html = gr.HTML()

            with gr.Tab("4 · AI analysis"):
                with gr.Row():
                    use_llm = gr.Checkbox(value=True, label="Use local language model (Ollama)")
                    language = gr.Radio(list(ctl.LANGUAGES), value="English", label="Language of the analysis")
                    llm_btn = gr.Button("Check local model")
                llm_status = gr.Markdown()
                gr.Markdown("If the local model is unavailable, a clearly labelled **deterministic** explanation is shown instead. "
                            "Every statement cites CV and listing evidence, and unsupported statements are removed.")
                with gr.Row():
                    with gr.Column():
                        job_pick = gr.Dropdown([], label="Explain this recommendation")
                        explain_btn = gr.Button("Explain match", variant="primary")
                        explanation_html = gr.HTML()
                    with gr.Column():
                        compare_pick = gr.CheckboxGroup([], label="Compare (select 2–4)")
                        compare_btn = gr.Button("Compare selected")
                        comparison_html = gr.HTML()

        parse_btn.click(on_parse, [cv_file, cv_text], [profile_state, profile_json, profile_html, parse_status])
        apply_btn.click(on_apply, [profile_json], [profile_state, profile_html, apply_status])
        find_btn.click(on_recommend, [profile_state, roles, job_types, locations, remote, hard, max_results],
                       [result_state, results_html, job_pick, compare_pick, find_status])  # fmt: skip
        explain_btn.click(on_explain, [profile_state, result_state, job_pick, use_llm, language], [explanation_html])
        compare_btn.click(on_compare, [profile_state, result_state, compare_pick, use_llm, language], [comparison_html])
        llm_btn.click(on_llm_status, None, [llm_status])
    return demo


def launch(ctx: AppContext, *, port: int = 7860, inbrowser: bool = False) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_app(ctx)
    demo.queue(default_concurrency_limit=2)
    # Bound to loopback: the app is for local use and is never exposed on the network.
    demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=inbrowser, show_error=False)
