"""
ingest/pipeline.py

Full GraphRAG ingest pipeline as a single flyte 2.x task.

Accepts pre-encoded PDF bytes so the task container receives all data
as inputs (no local filesystem dependency). Steps run sequentially:
  1. parse_and_chunk   — one call per PDF
  2. extract_entities  — one call per chunk
  3. load_graph        — write chunks + entities + relationships to Neo4j
  4. create_vector_index — build HNSW index (idempotent)
  5. resolve_entities  — merge near-duplicate entity nodes
  6. detect_communities — Louvain community detection
  7. summarize_communities — Claude summaries per community
"""

import json

from config import task_env
from ingest.chunking import parse_and_chunk
from ingest.extraction import extract_entities
from ingest.graph_loader import load_graph, create_vector_index
from ingest.enrichment import resolve_entities, detect_communities, summarize_communities


@task_env.task
def ingest_pipeline(filenames: list[str], pdf_bytes_b64: list[str]) -> str:
    """
    Full GraphRAG ingest pipeline.

    Args:
        filenames:     PDF filenames — used as Document node names in Neo4j.
        pdf_bytes_b64: Base64-encoded PDF bytes, parallel to filenames.

    Returns:
        JSON summary — {communities_summarized}.
    """
    all_extraction_results = []
    for name, enc in zip(filenames, pdf_bytes_b64):
        chunks_json = parse_and_chunk(source_doc=name, pdf_bytes_b64=enc)
        for chunk in json.loads(chunks_json):
            result = extract_entities(chunk_json=json.dumps(chunk))
            all_extraction_results.append(result)

    load_graph(extraction_results=all_extraction_results)
    create_vector_index()
    resolve_entities()
    detect_communities()
    return summarize_communities()
