---
title: Flyte / Union
weeks: [mcp]
---

Flyte is an open-source workflow orchestration platform. Union is the managed
cloud offering. In the series, "Flyte" and "Union" are used interchangeably —
Union hosts the compute; the Flyte 2.x SDK defines tasks and apps.

Used across the series for two purposes:
1. **Task orchestration** — running ML/AI pipelines as parallelizable tasks
2. **App deployment** — hosting persistent services (Gradio UIs, MCP servers)
   via `flyte.app.AppEnvironment` + `flyte.serve()`

## Usage across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Used to deploy the FastMCP server to Union cloud. `flyte_app.py` wraps the
FastMCP server in a Starlette ASGI app and deploys it as a persistent service.
Once deployed, clients update `MCP_SERVER_URL` in `.env` to point at the remote
URL — no code changes needed since both clients read from that env var.

This was the deployment pattern for the server side only.

### Week 2 — Agentic Search with Tavily (2026-04-03)

Used for task orchestration in `langgraph_agent_research/`. Each `research_topic`
runs as a separate Flyte task with its own container, resources, and
observability. LangGraph's `Send` API fans out to the Flyte task — on a cluster,
each `Send` becomes a separate container running a full ReAct agent.

The pipeline graph accepts the Flyte task as a parameter, keeping LangGraph
(logic) and Flyte (compute) cleanly separated. This is the first use of Flyte
for parallel task fan-out rather than simple app deployment.
