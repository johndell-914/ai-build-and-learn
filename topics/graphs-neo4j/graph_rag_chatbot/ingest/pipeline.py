"""
ingest/pipeline.py

Full GraphRAG ingest pipeline as flyte 2.x tasks.

ingest_pipeline (async orchestrator)
    Fans out process_pdf in parallel — one Union task per PDF — then runs
    the sequential graph operations once all PDFs are processed.

process_pdf (per-PDF task)
    Parses and chunks one PDF, then calls Claude to extract entities and
    relationships from each chunk. Returns a list of extraction-result JSON strings.
"""

import asyncio
import json

from config import task_env
from ingest.chunking import parse_and_chunk
from ingest.extraction import extract_entities
from ingest.graph_loader import load_graph, create_vector_index
from ingest.enrichment import resolve_entities, detect_communities, summarize_communities


@task_env.task
async def process_pdf(source_doc: str, pdf_bytes_b64: str) -> list[str]:
    """
    Parse, chunk, and extract entities for a single PDF.

    Args:
        source_doc:     Filename used as the Document node name in Neo4j.
        pdf_bytes_b64:  Base64-encoded PDF bytes.

    Returns:
        List of extraction-result JSON strings — one per chunk.
    """
    chunks_json = parse_and_chunk(source_doc=source_doc, pdf_bytes_b64=pdf_bytes_b64)
    results = []
    for chunk in json.loads(chunks_json):
        results.append(extract_entities(chunk_json=json.dumps(chunk)))
    return results


@task_env.task
async def ingest_pipeline(filenames: list[str], pdf_bytes_b64: list[str]) -> str:
    """
    Full GraphRAG ingest pipeline.

    Fans out process_pdf in parallel (one Union task per PDF), then runs
    load_graph → create_vector_index → resolve_entities →
    detect_communities → summarize_communities sequentially.

    Args:
        filenames:     PDF filenames — used as Document node names in Neo4j.
        pdf_bytes_b64: Base64-encoded PDF bytes, parallel to filenames.

    Returns:
        JSON summary — {communities_summarized}.
    """
    # Parallel fan-out — one process_pdf task per PDF
    per_pdf_results = await asyncio.gather(*[
        process_pdf(source_doc=name, pdf_bytes_b64=enc)
        for name, enc in zip(filenames, pdf_bytes_b64)
    ])

    # Flatten list-of-lists into a single extraction result list
    all_extraction_results = [r for pdf_results in per_pdf_results for r in pdf_results]

    # Sequential graph operations
    load_graph(extraction_results=all_extraction_results)
    create_vector_index()
    resolve_entities()
    detect_communities()
    return summarize_communities()
