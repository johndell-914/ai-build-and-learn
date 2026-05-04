# GraphRAG Chatbot — Research Notes

Companion project to `topics/vectorstore/vector_rag_chatbot`. Uses the same
16 Everstorm Outfitters PDFs to demonstrate when and why you need graph
retrieval on top of — or instead of — pure vector search.

---

## The Core Thesis

Neither vector RAG nor graph RAG is universally better. They answer different
question types well:

| Question type | Vector RAG | GraphRAG |
|---|---|---|
| Factual / single-chunk | ✅ | ✅ |
| Relationship-aware | ⚠️ | ✅ |
| Multi-hop / cross-document | ❌ | ✅ |
| Thematic / global | ❌ | ✅ (community summaries) |

The best production systems use both together. Neo4j enables this in a single
database: vector index on chunk embeddings + graph traversal over entity
relationships, combined in one Cypher query.

---

## Hosting Decision — Neo4j AuraDB Free

**Decision: AuraDB Free**

Evaluated three options:

| Option | Cost | GDS library | Setup time | Decision |
|---|---|---|---|---|
| AuraDB Free | $0 | ❌ | Minutes | ✅ Chosen |
| GCP VM + Neo4j Docker | ~$2-5/wk | ✅ | 30-60 min | Post-demo upgrade |
| GCP Marketplace Neo4j | ~$170-340/wk | ✅ | Hours | ❌ Too expensive |

**Why AuraDB Free:**
- Sufficient capacity for demo (200k nodes / 400k relationships limit vs our ~few thousand nodes)
- Ready immediately — no VM, no firewall rules, no SSH
- Accessible from Union cluster (AWS us-east-2) via `neo4j+s://` Bolt over TLS
- Post-demo migration to GCP VM is one line change (connection string only)

**AuraDB Free limits to know:**
- No GDS library (community detection must be done in Python)
- Vector index optimization requires > 4 GB RAM — free tier falls back to
  Lucene-backed HNSW. Performance difference is irrelevant at 127 chunks.
- Auto-pauses on idle

**Connection string format:**
```
neo4j+s://<id>.databases.neo4j.io
```

---

## Neo4j Python Driver

**Version:** 6.1.0 (January 2026, current production line)

**Breaking changes from 5.x:**
- Python 3.10+ required
- Must use `with` blocks — drivers no longer auto-close in destructors
- Create and close driver per Flyte task (tasks are separate processes)

```python
from neo4j import GraphDatabase

URI  = "neo4j+s://xxxxxxxx.databases.neo4j.io"
AUTH = ("neo4j", "<password>")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        session.run("MERGE (c:Chunk {id: $id}) SET c.text = $text",
                    id=chunk_id, text=text)
```

Always use parameterized queries — never f-strings in Cypher.

---

## Neo4j Vector Search

Native HNSW vector indexes via Apache Lucene. Available since Neo4j 5.11.
Supports cosine similarity and Euclidean distance, up to 4096 dimensions.

**Create index:**
```cypher
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

**The key GraphRAG query — vector search + graph traversal in one statement:**
```cypher
CALL db.index.vector.queryNodes('chunk_embedding', 10, $query_embedding)
YIELD node AS chunk, score
MATCH (chunk)-[:MENTIONS]->(entity:Entity)
OPTIONAL MATCH (entity)-[:RELATIONSHIP]->(other:Entity)
RETURN chunk.text, entity.name, entity.type,
       collect(other.name) AS related_entities, score
ORDER BY score DESC
```

This is what pgvector cannot do — vector search and relationship traversal
combined in a single query.

---

## Graph Schema

### Nodes

| Label | Key Properties |
|---|---|
| `Document` | name, source_path |
| `Chunk` | id, text, embedding, chunk_index, source_doc |
| `Entity` | name, type, description |
| `Community` | id, level, summary |

### Relationships

| Relationship | From → To | Meaning |
|---|---|---|
| `HAS_CHUNK` | Document → Chunk | Document contains this chunk |
| `MENTIONS` | Chunk → Entity | Chunk references this entity |
| `RELATIONSHIP` | Entity → Entity | Typed edge with `rel_type` property |
| `BELONGS_TO` | Entity → Community | Entity is member of this community |

### Everstorm Entity Ontology

**Entity types (constrained — better extraction quality than open-ended):**
`PRODUCT`, `POLICY`, `PROGRAM`, `TIER`, `BENEFIT`, `CONDITION`, `PROCESS`

**Relationship types:**
`HAS_POLICY`, `QUALIFIES_FOR`, `REQUIRES`, `APPLIES_TO`, `PART_OF`, `COVERS`

---

## Entity Extraction with Claude

Use Claude's tool use (structured output) to force valid JSON.
Never parse markdown — tool use eliminates JSON parse failures entirely.

**Output schema:**
```json
{
  "entities": [
    {
      "name": "Elite Tier",
      "type": "TIER",
      "description": "Highest loyalty tier with premium benefits"
    }
  ],
  "relationships": [
    {
      "source": "Elite Tier",
      "target": "Free Returns",
      "type": "QUALIFIES_FOR",
      "description": "Elite members receive free return shipping"
    }
  ]
}
```

**Prompt patterns that work:**
1. Define entity and relationship types upfront — constrain to domain ontology
2. Use tool use / structured output — eliminates JSON parse failures
3. Multi-shot examples in `<examples>` tags — 3-5 examples improves consistency
4. Two-pass if needed: entities first, then relationships
5. Run as `map_task` in Flyte — one Claude call per chunk, fully parallel

**Entity normalization (critical):**
Different chunks will refer to the same entity differently — "Everstorm",
"Everstorm Outfitters", "the Company". Without de-duplication the graph
becomes disconnected and traversal produces no benefit.

Fix: embed all extracted entity names with gte-small, merge nodes with
cosine similarity > 0.95. `neo4j-graphrag-python` has a built-in
`EntityResolver` component for this.

---

## Community Detection

GDS library (Leiden/Louvain) not available on AuraDB Free.

**Workaround:** Run community detection in Python using `python-louvain`
on a NetworkX graph built from the entity relationships stored in Neo4j.
Assign community IDs, use Claude to summarize each community, store
Community nodes back in Neo4j.

```python
import community as community_louvain
import networkx as nx

# Build NetworkX graph from Neo4j entity relationships
G = nx.Graph()
# ... add nodes and edges from Cypher query ...

# Detect communities
partition = community_louvain.best_partition(G)
# partition = {entity_name: community_id, ...}

# Summarize each community with Claude, store as Community nodes
```

Post-demo upgrade: migrate to GCP VM + Neo4j Docker to get native GDS
and drop this workaround.

---

## Retrieval Patterns

Three modes, selected by a Claude query router:

**Mode A — Vector + Graph Expansion** (factual / specific questions)
1. Embed query → vector search finds top-k chunks
2. Cypher traversal expands outward → entities + relationships
3. Combined context sent to Claude

**Mode B — Entity Lookup + Traversal** (relationship questions)
1. Extract named entities from question
2. Direct graph lookup → entity neighborhood subgraph
3. Shortest path between entities if multiple mentioned

**Mode C — Community Summary Search** (thematic / global questions)
1. Embed question → find most relevant Community node
2. Return community summary + member entities
3. Answers "what programs does Everstorm have for X?"

---

## Python Libraries

| Library | Version | Role |
|---|---|---|
| `neo4j` | 6.1.0 | Driver — all Cypher queries |
| `neo4j-graphrag` | latest | Pipeline + retrievers + EntityResolver |
| `anthropic` | latest | Entity extraction + generation |
| `sentence-transformers` | latest | gte-small embeddings (384D) |
| `python-louvain` | latest | Community detection (AuraDB workaround) |
| `networkx` | latest | Graph structure for community detection |
| `gradio` | 6.x | UI |
| `flytekit` | latest | Flyte task orchestration |

**Do not use:**
- Microsoft `graphrag` package — too opinionated, expensive LLM calls, not Neo4j native
- `langchain-neo4j` / `llama-index-neo4j` — adds framework overhead we don't need

---

## Workflow Structure (Modular)

Designed modular from day one — learned from vector_rag_chatbot.
Each task has its own file so engineers can open one file and see
exactly one job. `app.py` imports only from `workflows.py`.
`flyte deploy` bundles all modules transitively.

### Project Structure

```
graph_rag_chatbot/
├── app.py                        ← Gradio UI + Union AppEnvironment
├── workflows.py                  ← entry point: imports ingest_pipeline + query_pipeline
├── config.py                     ← constants, secrets, TaskEnvironment
├── schema.py                     ← entity types, relationship types, Cypher templates
├── prompts.py                    ← Claude extraction, routing, generation prompts
│
├── ingest/
│   ├── __init__.py               ← exports ingest_pipeline
│   ├── chunking.py               ← parse_and_chunk_task
│   ├── extraction.py             ← extract_entities_task
│   ├── graph_loader.py           ← load_graph_task + create_vector_index_task
│   ├── enrichment.py             ← resolve_entities_task + detect_communities_task
│   │                                + summarize_communities_task
│   └── pipeline.py               ← ingest_pipeline orchestrator
│
├── query/
│   ├── __init__.py               ← exports query_pipeline
│   ├── routing.py                ← route_query_task
│   ├── retrieval.py              ← hybrid_retrieve_task + entity_retrieve_task
│   │                                + community_retrieve_task
│   ├── generation.py             ← generate_task
│   └── pipeline.py               ← query_pipeline orchestrator
│
├── requirements.txt
├── Dockerfile                    ← task image (gte-small baked in)
├── Dockerfile.app                ← app serving image
├── flyte.yaml
└── data/                         ← 16 Everstorm Outfitters PDFs
```

### Ingest Pipeline

```
ingest_pipeline  (ingest/pipeline.py — orchestrator)
    │
    ├── parse_and_chunk_task      ingest/chunking.py
    │       PyMuPDF → text · RecursiveCharacterTextSplitter
    │       cached per PDF
    │
    ├── extract_entities_task     ingest/extraction.py
    │       map_task — one Claude call per chunk, fully parallel
    │       tool use → {entities, relationships} JSON
    │
    ├── load_graph_task           ingest/graph_loader.py
    │       write Document, Chunk, Entity nodes to Neo4j
    │       create HAS_CHUNK, MENTIONS, RELATIONSHIP edges
    │
    ├── create_vector_index_task  ingest/graph_loader.py
    │       HNSW index on Chunk.embedding (384D, cosine)
    │
    ├── resolve_entities_task     ingest/enrichment.py
    │       embed entity names → merge nodes (cosine similarity > 0.95)
    │
    ├── detect_communities_task   ingest/enrichment.py
    │       python-louvain on NetworkX → assign community IDs
    │
    └── summarize_communities_task  ingest/enrichment.py
            Claude summarizes each community → Community nodes + BELONGS_TO edges
```

### Query Pipeline

```
query_pipeline  (query/pipeline.py — orchestrator)
    │
    ├── route_query_task          query/routing.py
    │       Claude classifier → "hybrid" | "entity" | "community"
    │
    ├── [Mode A] hybrid_retrieve_task     query/retrieval.py
    │       embed query → vector search → Cypher graph expansion
    │       VectorCypherRetriever (neo4j-graphrag)
    │
    ├── [Mode B] entity_retrieve_task     query/retrieval.py
    │       named entity lookup → neighborhood subgraph traversal
    │
    ├── [Mode C] community_retrieve_task  query/retrieval.py
    │       embed question → find relevant Community node → return summary
    │
    └── generate_task             query/generation.py
            Claude RAG prompt → answer + sources + retrieval_mode
```

---

## Demo Story

Same 16 Everstorm PDFs. Side-by-side UI: Vector RAG vs GraphRAG.

**Questions that show the graph advantage:**

| Question | Vector RAG | GraphRAG |
|---|---|---|
| "What is the return window?" | ✅ | ✅ |
| "What benefits do Elite members get on returns?" | ⚠️ | ✅ Tier→Benefit→Policy |
| "Which policies apply to international purchases?" | ❌ | ✅ Multi-hop |
| "What programs does Everstorm have for repeat customers?" | ❌ | ✅ Community summary |

**The narrative:** Vector RAG gets you to the right neighborhood.
Graph traversal tells you how everything in that neighborhood connects.
Community summaries zoom out to the full picture for thematic questions.
Production RAG needs all three.

---

## Cost Summary

| Resource | Cost |
|---|---|
| Neo4j AuraDB Free | $0 |
| Union.ai compute | Existing account |
| Anthropic API (extraction + generation) | ~$0.50-1.00 per full ingest run |
| Docker Hub | Existing account |
| Total | ~$1/run |
