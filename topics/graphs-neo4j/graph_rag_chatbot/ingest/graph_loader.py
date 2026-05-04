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
