"""
app.py — Gradio UI for the Everstorm GraphRAG chatbot.

Two tabs:
  Ingest Graph  — kick off ingest_pipeline on Union (reads PDFs from data/)
  Chat          — ask questions, see answer + retrieval mode + sources + entities

Run locally:
    python app.py

Deploy to Union:
    python app.py --deploy
"""

import json
import shutil
import sys
from pathlib import Path

import flyte
import flyte.app
import gradio as gr

import config    # loads .env
import workflows  # imported at module level so union register bundles all tasks

# CSS inlined so the deployed app bundle (Python files only) doesn't need a
# separate file on disk.
_CSS = """
/* ── Mode badges ──────────────────────────────────────────────────────── */
.mode-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.mode-hybrid    { background: #dbeafe; color: #1e40af; }
.mode-entity    { background: #ede9fe; color: #5b21b6; }
.mode-community { background: #d1fae5; color: #065f46; }

/* ── Metadata panel ───────────────────────────────────────────────────── */
.meta-panel {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 8px;
    font-size: 0.88em;
}
.meta-label-gap { margin-top: 12px; }
.meta-label {
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued, #888);
    margin-bottom: 6px;
}
.meta-item {
    padding: 3px 0;
    color: var(--body-text-color, #e0e0e0);
}

/* ── Source accordion ─────────────────────────────────────────────────── */
.source-accordion {
    margin-top: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.88em;
}
.source-accordion summary {
    padding: 6px 12px;
    background: rgba(255,255,255,0.07);
    cursor: pointer;
    font-weight: 600;
    color: var(--body-text-color, #e0e0e0);
    list-style: none;
}
.source-accordion summary:hover { background: rgba(255,255,255,0.13); }
.source-item {
    padding: 6px 12px;
    border-top: 1px solid rgba(255,255,255,0.07);
    color: var(--body-text-color-subdued, #aaa);
}

/* ── Run link ─────────────────────────────────────────────────────────── */
.run-link a {
    display: inline-block;
    padding: 6px 14px;
    background: #5865f2;
    color: #fff;
    border-radius: 6px;
    font-weight: 600;
    text-decoration: none;
}
.run-link a:hover { background: #4752c4; }

/* ── Log box ──────────────────────────────────────────────────────────── */
.log-box textarea { font-family: monospace !important; font-size: 0.85em !important; }

/* ── Sidebar divider ──────────────────────────────────────────────────── */
.tab-sidebar { border-right: 1px solid rgba(255,255,255,0.1); padding-right: 16px !important; }
"""

# ── Union App deployment environment ──────────────────────────────────────────

serving_env = flyte.app.AppEnvironment(
    name="everstorm-graphrag-chatbot",
    image="docker.io/johndellenbaugh/graphrag-app:latest",
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="NEO4J_URI",         as_env_var="NEO4J_URI"),
        flyte.Secret(key="NEO4J_USERNAME",    as_env_var="NEO4J_USERNAME"),
        flyte.Secret(key="NEO4J_PASSWORD",    as_env_var="NEO4J_PASSWORD"),
    ],
    env_vars={"FLYTE_BACKEND": "cluster", "APP_VERSION": "1"},
    port=7860,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# ── HTML builders ──────────────────────────────────────────────────────────────

def _mode_badge(mode: str) -> str:
    label = {"hybrid": "Hybrid", "entity": "Entity", "community": "Community"}.get(mode, mode)
    return f'<span class="mode-badge mode-{mode}">{label}</span>'


def build_meta_panel(result: dict) -> str:
    mode = result.get("retrieval_mode", "hybrid")
    sources = result.get("sources", [])
    entities = result.get("entities_used", [])

    source_items = "".join(
        f'<div class="source-item">{s}</div>' for s in sources
    ) or '<div class="source-item">—</div>'

    entity_items = "".join(
        f'<div class="meta-item">· {e}</div>' for e in entities
    ) or '<div class="meta-item">—</div>'

    sources_accordion = (
        f'<details class="source-accordion">'
        f'<summary>📄 {len(sources)} source document(s)</summary>'
        f'{source_items}'
        f'</details>'
    )

    return (
        f'<div class="meta-panel">'
        f'  <div class="meta-label">Retrieval Mode</div>'
        f'  {_mode_badge(mode)}'
        f'  <div class="meta-label meta-label-gap">Entities Used</div>'
        f'  {entity_items}'
        f'  {sources_accordion}'
        f'</div>'
    )


def build_run_link(run) -> str:
    try:
        return f'<a href="{run.url}" target="_blank">🔗 View run on Union</a>'
    except Exception:
        return ""


# ── Ingest handler ─────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"


def run_ingest(uploaded_files):
    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Copy any uploaded PDFs into data/ so the pipeline can read them
    if uploaded_files:
        _DATA_DIR.mkdir(exist_ok=True)
        yield emit(f"📂 Copying {len(uploaded_files)} uploaded file(s) to data/..."), ""
        for file_path in uploaded_files:
            src = Path(file_path)
            dest = _DATA_DIR / src.name
            shutil.copy2(src, dest)
            yield emit(f"   ✅ {src.name}"), ""
    else:
        yield emit("📂 Using PDFs already in data/..."), ""

    yield emit("\n🚀 Dispatching ingest_pipeline → Union cluster..."), ""

    try:
        from workflows import ingest_pipeline
        run = flyte.run(ingest_pipeline, data_dir=str(_DATA_DIR))
        link = build_run_link(run)

        yield emit("⏳ Running on Union — waiting for results..."), link

        run.wait()
        summary = json.loads(run.outputs().o0)

        yield emit(
            f"\n✅ Ingest complete!\n"
            f"   Communities summarized : {summary.get('communities_summarized', '—')}"
        ), link

    except Exception as exc:
        yield emit(f"\n❌ Error: {exc}"), ""


# ── Chat handler ───────────────────────────────────────────────────────────────

def chat(question, history):
    question = question.strip()
    history = list(history or [])

    if not question:
        return history, ""

    history.append({"role": "user", "content": question})

    try:
        from workflows import query_pipeline
        run = flyte.run(query_pipeline, question=question)
        run.wait()
        result = json.loads(run.outputs().o0)

        answer = result["answer"]
        meta = build_meta_panel(result)

        history.append({
            "role": "assistant",
            "content": f"{answer}\n\n{meta}",
        })

    except Exception as exc:
        history.append({
            "role": "assistant",
            "content": f"❌ Error: {exc}",
        })

    return history, ""


# ── UI layout ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Everstorm GraphRAG Chatbot", css=_CSS) as app:

        gr.Markdown("# Everstorm Outfitters — GraphRAG Chatbot")
        gr.Markdown(
            "Knowledge-graph Q&A powered by Neo4j + Claude, compute running on Union."
        )

        with gr.Tabs():

            # ── Tab 1: Ingest ──────────────────────────────────────────────────
            with gr.Tab("📥 Ingest Graph"):

                gr.Markdown(
                    "Upload PDFs to add them to the graph, or leave empty to use "
                    "the PDFs already in `data/`. Extracts entities and relationships, "
                    "writes the graph to Neo4j AuraDB, and generates community summaries."
                )

                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        file_upload = gr.File(
                            file_types=[".pdf"],
                            file_count="multiple",
                            label="Upload PDFs (optional)",
                        )

                    with gr.Column(scale=4):
                        ingest_btn = gr.Button("🚀 Run Ingest on Union", variant="primary")
                        run_link = gr.HTML(elem_classes=["run-link"])
                        status_log = gr.Textbox(
                            label="Status Log",
                            lines=12,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                ingest_btn.click(
                    fn=run_ingest,
                    inputs=[file_upload],
                    outputs=[status_log, run_link],
                )

            # ── Tab 2: Chat ────────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Everstorm GraphRAG",
                            height=500,
                            type="messages",
                        )
                        question_input = gr.Textbox(
                            placeholder="Ask anything about Everstorm Outfitters...",
                            label="Question",
                            submit_btn=True,
                        )
                        clear_btn = gr.Button("🗑 Clear")

                    with gr.Column(scale=1, elem_classes=["tab-sidebar"]):
                        gr.Markdown("**How it works**")
                        gr.Markdown(
                            "Each question is automatically routed to the best "
                            "retrieval strategy:\n\n"
                            "🔵 **Hybrid** — facts, rules, definitions\n\n"
                            "🟣 **Entity** — relationships between named things\n\n"
                            "🟢 **Community** — broad themes and programs"
                        )

                question_input.submit(
                    fn=chat,
                    inputs=[question_input, chatbot],
                    outputs=[chatbot, question_input],
                )

                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, question_input],
                )

    return app


# ── Union cluster entry point ──────────────────────────────────────────────────

@serving_env.server
def _cluster_server():
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


# ── Local entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--deploy" in sys.argv:
        app = flyte.serve(serving_env)
        print(f"App URL: {app.url}")
    else:
        build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)
