"""
ingest/pipeline.py

Orchestrator: ingest_pipeline

Responsibility:
    - Entry point for the full ingest flow
    - Fans out parse_and_chunk_task in parallel (one per PDF) via map_task
    - Runs extract_entities_task as a map_task (one per chunk, parallel)
    - Calls load_graph_task to write all chunks + entities + relationships to Neo4j
    - Calls create_vector_index_task to build HNSW index
    - Calls resolve_entities_task to de-duplicate entity nodes
    - Calls detect_communities_task to cluster entities
    - Calls summarize_communities_task to generate and store community summaries

Imports from:
    ingest.chunking     import parse_and_chunk_task
    ingest.extraction   import extract_entities_task
    ingest.graph_loader import load_graph_task, create_vector_index_task
    ingest.enrichment   import resolve_entities_task, detect_communities_task,
                                summarize_communities_task
"""

import base64
import json
from pathlib import Path
from typing import List

from flytekit import workflow, map_task

from ingest.chunking import parse_and_chunk_task
from ingest.extraction import extract_entities_task
from ingest.graph_loader import load_graph_task, create_vector_index_task
from ingest.enrichment import (
    resolve_entities_task,
    detect_communities_task,
    summarize_communities_task,
)


def _load_pdfs(data_dir: str) -> tuple[List[str], List[str]]:
    """Return parallel lists of (source_doc names, base64-encoded PDF bytes)."""
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    names = [p.name for p in pdf_paths]
    encoded = [base64.b64encode(p.read_bytes()).decode() for p in pdf_paths]
    return names, encoded


@workflow
def ingest_pipeline(data_dir: str = "data") -> str:
    """
    Full GraphRAG ingest pipeline.

    Reads all PDFs from data_dir, chunks them, extracts entities and
    relationships via Claude, loads everything into Neo4j, builds the
    HNSW vector index, resolves duplicate entities, detects communities,
    and generates community summaries.

    Args:
        data_dir: Path to directory containing source PDFs.

    Returns:
        JSON summary string from summarize_communities_task.
    """
    names, encoded = _load_pdfs(data_dir)

    # Fan out: one parse_and_chunk_task per PDF (parallel)
    chunk_lists: List[str] = map_task(parse_and_chunk_task)(
        source_doc=names,
        pdf_bytes_b64=encoded,
    )

    # Flatten list-of-JSON-arrays into list-of-JSON-objects for map_task
    # Each chunk_list is a JSON array; we need individual chunk JSON strings
    flat_chunks = _flatten_chunks(chunk_lists=chunk_lists)

    # Fan out: one extract_entities_task per chunk (parallel)
    extraction_results: List[str] = map_task(extract_entities_task)(
        chunk_json=flat_chunks,
    )

    # Write all graph data to Neo4j
    load_summary = load_graph_task(extraction_results=extraction_results)

    # Build HNSW vector index (idempotent)
    create_vector_index_task(dummy=load_summary)

    # Enrich: entity resolution → community detection → community summaries
    resolve_summary = resolve_entities_task()
    detect_summary = detect_communities_task(resolve_summary=resolve_summary)
    final_summary = summarize_communities_task(detect_summary=detect_summary)

    return final_summary


from flytekit import task, Resources


@task(requests=Resources(cpu="1", mem="500Mi"))
def _flatten_chunks(chunk_lists: List[str]) -> List[str]:
    """
    Convert a list of JSON arrays (one per PDF) into a flat list of
    individual chunk JSON strings (one per chunk) for map_task input.
    """
    flat = []
    for chunk_list_json in chunk_lists:
        chunks = json.loads(chunk_list_json)
        for chunk in chunks:
            flat.append(json.dumps(chunk))
    return flat
