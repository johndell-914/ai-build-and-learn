"""
query/retrieval.py

Tasks: hybrid_retrieve_task, entity_retrieve_task, community_retrieve_task

Responsibility:
    hybrid_retrieve_task (Mode A):
        - Embed query with gte-small
        - Vector search finds top-k Chunk nodes via HNSW index
        - Cypher traversal expands outward: Chunk → Entity → related Entities
        - Returns chunks + entity context as JSON

    entity_retrieve_task (Mode B):
        - Extract named entities from the question via Claude
        - Direct MATCH on Entity nodes by name (fuzzy via CONTAINS)
        - Retrieve entity neighborhood subgraph (up to 2 hops)
        - Returns entity subgraph context as JSON

    community_retrieve_task (Mode C):
        - Embed question with gte-small
        - Fetch all Community summaries from Neo4j
        - Find most relevant Community by cosine similarity
        - Return community summary + member entity names as JSON
"""

import json

import numpy as np
from flytekit import task, Resources
from sentence_transformers import SentenceTransformer

from config import CLAUDE_MODEL, EMBED_MODEL, VECTOR_INDEX_NAME, anthropic_client, neo4j_driver

_TOP_K = 5  # number of chunks returned by vector search


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@task(requests=Resources(cpu="2", mem="2Gi"))
def hybrid_retrieve_task(question: str) -> str:
    """
    Mode A — vector search over Chunk nodes + graph expansion to nearby Entities.

    Returns:
        JSON — {mode, chunks: [{chunk_id, source_doc, text, score}],
                entities: [{name, type, description}]}
    """
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(question).tolist()

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            # HNSW vector search
            chunk_rows = session.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
                YIELD node AS chunk, score
                RETURN chunk.id AS chunk_id,
                       chunk.source_doc AS source_doc,
                       chunk.text AS text,
                       score
                ORDER BY score DESC
                """,
                top_k=_TOP_K,
                embedding=query_vec,
            ).data()

            # Expand from matched chunks to nearby entities
            chunk_ids = [r["chunk_id"] for r in chunk_rows]
            entity_rows = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.id IN $chunk_ids
                OPTIONAL MATCH (e)-[:RELATED]->(neighbor:Entity)
                RETURN DISTINCT e.name AS name, e.type AS type,
                       e.description AS description
                LIMIT 20
                """,
                chunk_ids=chunk_ids,
            ).data()

    return json.dumps({
        "mode": "hybrid",
        "chunks": chunk_rows,
        "entities": entity_rows,
    })


_ENTITY_EXTRACT_TOOL = {
    "name": "extract_entities",
    "description": "Extract named entities mentioned in a question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of entity names mentioned in the question.",
            }
        },
        "required": ["entities"],
    },
}


@task(requests=Resources(cpu="1", mem="1Gi"))
def entity_retrieve_task(question: str) -> str:
    """
    Mode B — extract named entities from question, traverse their neighborhood in Neo4j.

    Returns:
        JSON — {mode, entities: [{name, type, description, neighbors: [...]}]}
    """
    client = anthropic_client()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        tools=[_ENTITY_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_entities"},
        messages=[{"role": "user", "content": question}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    entity_names = tool_block.input.get("entities", [])

    if not entity_names:
        return json.dumps({"mode": "entity", "entities": []})

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            results = []
            for name in entity_names:
                # Fuzzy match — CONTAINS handles partial names
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (e)-[r:RELATED]->(neighbor:Entity)
                    RETURN e.name AS name, e.type AS type, e.description AS description,
                           collect({
                               name: neighbor.name,
                               rel_type: r.type,
                               description: r.description
                           }) AS neighbors
                    LIMIT 5
                    """,
                    name=name,
                ).data()
                results.extend(rows)

    return json.dumps({"mode": "entity", "entities": results})


@task(requests=Resources(cpu="2", mem="2Gi"))
def community_retrieve_task(question: str) -> str:
    """
    Mode C — find the most relevant Community node by embedding similarity.

    Returns:
        JSON — {mode, community_id, summary, member_entities: [name, ...]}
    """
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(question)

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            community_rows = session.run(
                "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary"
            ).data()

            if not community_rows:
                return json.dumps({"mode": "community", "community_id": None,
                                   "summary": "", "member_entities": []})

            # Find most similar community by cosine similarity over summary embeddings
            summaries = [r["summary"] for r in community_rows]
            summary_vecs = model.encode(summaries)

            scores = [_cosine_sim(query_vec, sv) for sv in summary_vecs]
            best_idx = int(np.argmax(scores))
            best = community_rows[best_idx]

            member_rows = session.run(
                """
                MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $cid})
                RETURN e.name AS name
                """,
                cid=best["id"],
            ).data()

    return json.dumps({
        "mode": "community",
        "community_id": best["id"],
        "summary": best["summary"],
        "member_entities": [r["name"] for r in member_rows],
    })
