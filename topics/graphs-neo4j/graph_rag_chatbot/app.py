"""
app.py — Gradio UI for the Everstorm GraphRAG chatbot.

Two tabs:
  Ingest Graph  — kick off ingest_pipeline on Union (reads PDFs from data/)
  Chat          — ask questions, see answer + retrieval mode badge + sources + entities

Run locally:
    python app.py

Deploy to Union:
    python app.py --deploy
"""

import json
import os
import shutil
import sys
from pathlib import Path

import flyte
import flyte.app
import gradio as gr

import config    # loads .env and calls flyte.init() for the right backend
import workflows  # imported at module level so flyte deploy bundles workflows


def _remote():
    """Return a UnionRemote configured for the current backend."""
    from flytekit.configuration import Config, PlatformConfig
    from union.remote import UnionRemote

    if os.getenv("FLYTE_BACKEND") == "cluster":
        cfg = Config(platform=PlatformConfig(endpoint="host.docker.internal:8090", insecure=True))
    else:
        cfg = Config(platform=PlatformConfig(endpoint=config.UNION_ENDPOINT))

    return UnionRemote(
        config=cfg,
        default_project=config.UNION_PROJECT,
        default_domain=config.UNION_DOMAIN,
    )


def _execution_url(execution) -> str:
    try:
        name = execution.id.name
        return (
            f"https://{config.UNION_ENDPOINT}/v2/domain/{config.UNION_DOMAIN}"
            f"/project/{config.UNION_PROJECT}/executions/{name}"
        )
    except Exception:
        return ""

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

/* ── Entity list ──────────────────────────────────────────────────────── */
.entity-accordion {
    margin-top: 6px;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.88em;
}
.entity-accordion summary {
    padding: 6px 12px;
    background: rgba(255,255,255,0.07);
    cursor: pointer;
    font-weight: 600;
    color: var(--body-text-color, #e0e0e0);
    list-style: none;
}
.entity-accordion summary:hover { background: rgba(255,255,255,0.13); }
.entity-item {
    padding: 4px 12px;
    border-top: 1px solid rgba(255,255,255,0.07);
    color: var(--body-text-color-subdued, #aaa);
    font-size: 0.85em;
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
    env_vars={"FLYTE_BACKEND": "cluster"},
    port=7860,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# ── HTML builders ──────────────────────────────────────────────────────────────

def _mode_badge(mode: str) -> str:
    label = {"hybrid": "Hybrid", "entity": "Entity", "community": "Community"}.get(mode, mode)
    return f'<span class="mode-badge mode-{mode}">{label}</span>'


def build_sources_accordion(sources: list) -> str:
    items = "".join(
        f'<div class="source-item">{s}</div>' for s in sources
    )
    return (
        f'<details class="source-accordion">'
        f'<summary>📄 {len(sources)} source document(s)</summary>'
        f'{items}'
        f'</details>'
    )


def build_entities_accordion(entities: list) -> str:
    if not entities:
        return ""
    items = "".join(f'<div class="entity-item">· {e}</div>' for e in entities)
    return (
        f'<details class="entity-accordion">'
        f'<summary>🔗 {len(entities)} entity/entities used</summary>'
        f'{items}'
        f'</details>'
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

    if uploaded_files:
        _DATA_DIR.mkdir(exist_ok=True)
        yield emit(f"📂 Copying {len(uploaded_files)} uploaded file(s) to data/..."), ""
        for file_path in uploaded_files:
            src = Path(file_path)
            shutil.copy2(src, _DATA_DIR / src.name)
            yield emit(f"   ✅ {src.name}"), ""
    else:
        yield emit("📂 Using PDFs already in data/..."), ""

    yield emit("\n🚀 Dispatching ingest_pipeline → Union cluster..."), ""

    try:
        from workflows import ingest_pipeline
        remote = _remote()
        execution = remote.execute(ingest_pipeline, inputs={"data_dir": str(_DATA_DIR)})
        url = _execution_url(execution)
        link = f'<a href="{url}" target="_blank">🔗 View run on Union</a>' if url else ""

        yield emit("⏳ Running on Union — waiting for results..."), link

        remote.wait(execution)
        summary = json.loads(execution.outputs["o0"])

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
        return history

    history.append({"role": "user", "content": question})

    try:
        from workflows import query_pipeline
        remote = _remote()
        execution = remote.execute(query_pipeline, inputs={"question": question})
        remote.wait(execution)
        result = json.loads(execution.outputs["o0"])

        answer   = result["answer"]
        mode     = result.get("retrieval_mode", "hybrid")
        sources  = result.get("sources", [])
        entities = result.get("entities_used", [])

        badge    = _mode_badge(mode)
        src_html = build_sources_accordion(sources) if sources else ""
        ent_html = build_entities_accordion(entities)

        history.append({
            "role": "assistant",
            "content": f"{answer}\n\n{badge}{src_html}{ent_html}",
        })

    except Exception as exc:
        history.append({
            "role": "assistant",
            "content": f"❌ Error: {exc}",
        })

    return history


# ── UI layout ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Everstorm GraphRAG Chatbot") as app:

        gr.Markdown("# Everstorm Outfitters — GraphRAG Chatbot")
        gr.Markdown(
            "Knowledge-graph Q&A powered by Neo4j + Claude, compute running on Union."
        )

        with gr.Tabs():

            # ── Tab 1: Ingest ──────────────────────────────────────────────────
            with gr.Tab("📥 Ingest Graph"):

                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        file_upload = gr.File(
                            file_types=[".pdf"],
                            file_count="multiple",
                            label="Upload PDFs (optional)",
                        )

                    with gr.Column(scale=4):
                        gr.Markdown(
                            "Upload PDFs to add them to the graph, or leave empty to "
                            "use PDFs already in `data/`. Extracts entities and relationships, "
                            "writes the graph to Neo4j AuraDB, and generates community summaries."
                        )
                        ingest_btn = gr.Button("🚀 Run Ingest on Union", variant="primary")
                        run_link = gr.HTML(elem_classes=["run-link"])
                        status_log = gr.Textbox(
                            label="Status Log",
                            lines=14,
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
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        gr.Markdown(
                            "**Retrieval modes**\n\n"
                            "🔵 **Hybrid** — facts, rules, definitions\n\n"
                            "🟣 **Entity** — relationships between named things\n\n"
                            "🟢 **Community** — broad themes and programs"
                        )
                        clear_btn = gr.Button("🗑 Clear")

                    with gr.Column(scale=4):
                        query_input = gr.Textbox(
                            placeholder="Ask anything about Everstorm Outfitters...",
                            label="Question",
                            submit_btn=True,
                        )
                        chatbot = gr.Chatbot(
                            label="Everstorm GraphRAG",
                            height=480,
                        )

                query_input.submit(
                    fn=chat,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot],
                ).then(
                    fn=lambda: "",
                    outputs=[query_input],
                )

                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, query_input],
                )

    return app


# ── Union cluster entry point ──────────────────────────────────────────────────

@serving_env.server
def _cluster_server():
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=_CSS)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--deploy" in sys.argv:
        try:
            app = flyte.serve(serving_env)
            print(f"App URL: {app.url}")
        except Exception:
            print("App deployed — check console for status:")
            print("https://tryv2.hosted.unionai.cloud/v2/domain/development/project/dellenbaugh/apps/everstorm-graphrag-chatbot")
    else:
        build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=_CSS)
