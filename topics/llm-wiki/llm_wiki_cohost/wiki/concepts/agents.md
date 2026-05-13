---
title: Agents
first_seen: topics/mcp/
weeks: [mcp]
---

LLM-powered systems that use tools to take actions, not just generate text.
An agent perceives its environment (context, tool results), decides what to do
(which tool to call or what to say), acts (calls the tool), and observes the
result — repeating until the task is done.

## How it appeared across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Agents appeared as MCP clients — the consumer side of the protocol. Two clients
were built: one using the OpenAI Agents SDK, one using the Anthropic SDK. Both
connected to the same FastMCP server, demonstrating that MCP decouples the
agent (client) from the tool implementation (server).

The agent loop here is minimal: user message → model picks tools → tools execute
→ model synthesizes answer. The MCP server handles tool execution; the SDK
handles the loop.

The data analysis demo showed a multi-step agent: the model chains
load → filter → aggregate → chart across multiple tool calls in one session,
with server-side state persisting between calls.

### Week 2 — Agentic Search with Tavily (2026-04-03)

Agents appeared in two forms, both as web research assistants:

**Claude + FastMCP ReAct agent** — A manual ReAct loop where the agent chooses
between `tavily_search`, `tavily_extract`, and `tavily_crawl` based on the
query. The agent's system prompt specifies when to use each tool and quality
standards (cite sources, cross-verify claims, iterate on poor results). The
loop is visible in the console — each tool call is printed.

**LangGraph ReAct sub-agent** — Each sub-topic researcher in the pipeline is a
LangGraph ReAct subgraph running inside a Flyte task. The subgraph calls
`web_search` (Tavily) in a loop until it has enough information or reaches
`max_searches`. Multiple researchers run in parallel across the fan-out.

Key new idea: a system prompt as a first-class artifact. `system_prompt.py` is
a standalone module imported by `agent.py` — the agent's strategy (which tool
to use when, quality standards) is explicit and separately maintainable.

## Open questions

- What agent frameworks does the series explore beyond the Agents SDK? (LangGraph, custom loops?)
- How does the series handle agent memory across sessions?
- What's the distinction between a "ReAct agent" and a simpler tool-calling loop?
