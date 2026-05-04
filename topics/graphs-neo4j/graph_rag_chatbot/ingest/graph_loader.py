"""
ingest/graph_loader.py

Tasks: load_graph_task, create_vector_index_task

Responsibility:
    load_graph_task:
        - Connect to Neo4j AuraDB
        - Create Document and Chunk nodes
        - Embed each chunk with gte-small (384D)
        - Set chunk.embedding property
        - Create Entity nodes and RELATIONSHIP edges from extraction results
        - Create HAS_CHUNK edges (Document → Chunk)
        - Create MENTIONS edges (Chunk → Entity)

    create_vector_index_task:
        - Create HNSW vector index on Chunk.embedding
        - cosine similarity, 384 dimensions
        - CREATE VECTOR INDEX IF NOT EXISTS (idempotent)
"""

import json
from typing import List

from flytekit import task, Resources
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL,
    EMBED_DIM,
    VECTOR_INDEX_NAME,
    VECTOR_SIMILARITY,
    neo4j_driver,
)

_MERGE_CHUNK_Q = """
MERGE (d:Document {name: $source_doc})
MERGE (c:Chunk {id: $chunk_id})
  ON CREATE SET c.text = $chunk_text,
               c.source_doc = $source_doc,
               c.chunk_index = $chunk_index
MERGE (d)-[:HAS_CHUNK]->(c)
"""

_SET_EMBEDDING_Q = """
MATCH (c:Chunk {id: $chunk_id})
SET c.embedding = $embedding
"""

_MERGE_ENTITY_Q = """
MERGE (e:Entity {name: $name})
  ON CREATE SET e.type = $type, e.description = $description
"""

_MERGE_RELATIONSHIP_Q = """
MATCH (a:Entity {name: $source})
MATCH (b:Entity {name: $target})
MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
  ON CREATE SET r.description = $description
"""

_MERGE_MENTIONS_Q = """
MATCH (c:Chunk {id: $chunk_id})
MATCH (e:Entity {name: $entity_name})
MERGE (c)-[:MENTIONS]->(e)
"""


@task(requests=Resources(cpu="2", mem="2Gi"))
def load_graph_task(extraction_results: List[str]) -> str:
    """
    Write all chunks, entities, and relationships to Neo4j.

    Args:
        extraction_results: List of JSON strings from extract_entities_task,
                            each containing {chunk_id, source_doc, chunk_text,
                            entities, relationships}.

    Returns:
        JSON summary — {chunks_written, entities_written, relationships_written}.
    """
    model = SentenceTransformer(EMBED_MODEL)
    driver = neo4j_driver()

    chunks_written = 0
    entities_written = 0
    relationships_written = 0

    with driver:
        with driver.session() as session:
            for raw in extraction_results:
                result = json.loads(raw)
                chunk_id = result["chunk_id"]
                source_doc = result["source_doc"]
                chunk_text = result["chunk_text"]
                chunk_index = int(chunk_id.split("::")[-1])
                entities = result["entities"]
                relationships = result["relationships"]

                # Document + Chunk nodes and HAS_CHUNK edge
                session.run(
                    _MERGE_CHUNK_Q,
                    source_doc=source_doc,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                )

                # Embed and store chunk vector
                embedding = model.encode(chunk_text).tolist()
                session.run(_SET_EMBEDDING_Q, chunk_id=chunk_id, embedding=embedding)
                chunks_written += 1

                # Entity nodes
                for ent in entities:
                    session.run(
                        _MERGE_ENTITY_Q,
                        name=ent["name"],
                        type=ent["type"],
                        description=ent["description"],
                    )
                    session.run(
                        _MERGE_MENTIONS_Q,
                        chunk_id=chunk_id,
                        entity_name=ent["name"],
                    )
                    entities_written += 1

                # Relationship edges between entities
                for rel in relationships:
                    session.run(
                        _MERGE_RELATIONSHIP_Q,
                        source=rel["source"],
                        target=rel["target"],
                        rel_type=rel["type"],
                        description=rel["description"],
                    )
                    relationships_written += 1

    return json.dumps({
        "chunks_written": chunks_written,
        "entities_written": entities_written,
        "relationships_written": relationships_written,
    })


@task(requests=Resources(cpu="1", mem="500Mi"))
def create_vector_index_task(dummy: str = "") -> None:
    """
    Create the HNSW vector index on Chunk.embedding (idempotent).

    The dummy arg lets the pipeline call this after load_graph_task completes
    without needing its return value.
    """
    cypher = (
        f"CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}` IF NOT EXISTS "
        f"FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{indexConfig: {{"
        f"  `vector.dimensions`: {EMBED_DIM},"
        f"  `vector.similarity_function`: '{VECTOR_SIMILARITY}'"
        f"}}}}"
    )

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            session.run(cypher)
