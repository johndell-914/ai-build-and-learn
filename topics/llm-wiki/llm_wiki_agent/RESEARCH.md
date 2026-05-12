# LLM Wiki — Research Notes

Topic: Karpathy's LLM-maintained Wiki pattern — turning raw sources into a
persistent, compounding knowledge base instead of re-retrieving on every query.

Reference: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

---

## The Core Idea

Traditional RAG re-discovers the same knowledge on every query. An LLM Wiki
inverts that: the LLM reads sources once and writes structured wiki pages that
synthesize, cross-reference, and accumulate insight over time.

**"The wiki is a persistent, compounding artifact."** — Karpathy

The human's job: curate sources, direct analysis, ask good questions.
The LLM's job: everything else — writing, updating, linking, auditing.

---

## Three-Layer Stack

| Layer | What it is |
|---|---|
| **Sources** | Raw, immutable inputs (docs, PDFs, URLs, notes) |
| **Wiki** | LLM-generated markdown pages — the knowledge base |
| **Schema** | Config describing structure, entity types, workflows |

---

## Three Core Operations

**Ingest** — Drop a source in; LLM reads it, writes summaries, updates relevant
wiki pages, logs the action. A single source may touch 10–15 files.

**Query** — Ask questions against the wiki; valuable answers become new pages,
compounding the knowledge base further.

**Lint** — Periodic audit: find contradictions, stale claims, orphaned pages,
and missing cross-references.

---

## How This Differs from RAG

| Dimension | RAG | LLM Wiki |
|---|---|---|
| When work happens | At query time | At ingest time |
| Knowledge state | Ephemeral (per query) | Persistent (grows over time) |
| Cross-doc synthesis | Per-query | Baked into wiki pages |
| Maintenance burden | Low (stateless) | LLM handles it |
| Latency at query time | Higher (retrieval + generation) | Lower (wiki already synthesized) |
| Best for | Arbitrary corpora, live data | Curated, domain-specific knowledge |

---

## Community Implementations to Study

From the gist comments:
- **SwarmVault** — multi-agent wiki maintenance
- **Kompl** — completion-style wiki builder
- **Link** — link-graph approach
- **OmegaWiki** — larger-scale implementation

---

## Open Questions (to answer before building)

- What domain / source corpus will we use?
- File-based wiki (markdown files) vs. database-backed?
- Which operations to implement: ingest only, or ingest + query + lint?
- Orchestration: single script, FastMCP, Flyte, or agent loop?
- UI: CLI only, or Gradio?
- How do we demonstrate the "compounding" property in a demo setting?
