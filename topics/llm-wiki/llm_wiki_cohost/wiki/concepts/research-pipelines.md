---
title: Research Pipelines
first_seen: topics/tavily/
weeks: [tavily]
---

A pipeline pattern for producing synthesized research reports from a broad
query. The canonical shape: plan → parallel research (fan-out) → synthesize →
quality gate → loop or finalize.

What makes it a pipeline rather than a single agent call: the work is
decomposed into parallel sub-tasks, each handled independently, then recombined.
The quality gate introduces a feedback loop — if the synthesis isn't good
enough, the pipeline deepens rather than stopping.

## How it appeared across the series

### Week 2 — Agentic Search with Tavily (2026-04-03)

`langgraph_agent_research/` is the first research pipeline in the series.
Orchestration is split across two layers:

**LangGraph** controls the logic — the pipeline graph (`build_pipeline_graph()`)
defines the nodes and edges: plan → `Send` fan-out → synthesize →
quality_check → (gaps found: identify_gaps → Send again) or (good: finalize).

**Flyte** controls the compute — each `research_topic` node in the pipeline
runs as a separate Flyte task with its own container. LangGraph passes the
Flyte task as a parameter to the pipeline graph; `Send` API fans out to it.
On a cluster, each `Send` becomes a separate container running a ReAct agent.

Key parameters that control the pipeline:
- `num_topics` — how many sub-topics to research in parallel
- `max_searches` — how many Tavily searches each researcher can make
- `max_iterations` — how many quality gate loops before forced finalization

The quality gate is the architecturally novel part: it scores the synthesized
report, identifies gaps, and re-fans out to fill them — the first feedback loop
inside a pipeline in the series.

## Open questions

- How does this pattern scale when sub-topics are highly interdependent?
- Does the series revisit research pipelines with different orchestration
  frameworks (e.g., pure Flyte, pure LangGraph, AutoResearch)?
