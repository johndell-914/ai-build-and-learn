"""
ingest/pipeline.py

Task: ingest_pipeline (orchestrator)

Responsibility:
    - Entry point for the full ingest flow
    - Fans out parse_and_chunk_task in parallel (one per PDF) via asyncio.gather
    - Merges all chunk results
    - Runs extract_entities_task as a map_task (one per chunk, parallel)
    - Calls load_graph_task to write all chunks + entities + relationships to Neo4j
    - Calls create_vector_index_task to build HNSW index
    - Calls resolve_entities_task to de-duplicate entity nodes
    - Calls detect_communities_task to cluster entities
    - Calls summarize_communities_task to generate and store community summaries
    - Returns JSON stats: {node_count, relationship_count, community_count, chunk_count}

Imports from:
    ingest.chunking    import parse_and_chunk_task
    ingest.extraction  import extract_entities_task
    ingest.graph_loader import load_graph_task, create_vector_index_task
    ingest.enrichment  import resolve_entities_task, detect_communities_task,
                               summarize_communities_task
"""
