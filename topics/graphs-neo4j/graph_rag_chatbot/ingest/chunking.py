"""
ingest/chunking.py

Task: parse_and_chunk_task

Responsibility:
    - Decode base64 PDF bytes
    - Extract full text with PyMuPDF
    - Split into overlapping chunks with RecursiveCharacterTextSplitter
    - Return JSON list of {source_doc, chunk_index, chunk_text}

Cached per PDF input — re-ingesting an unchanged PDF is a free cache hit.
One task dispatched per PDF, all running in parallel inside ingest_pipeline.
"""

import base64
import json

import fitz  # PyMuPDF
from flytekit import task, Resources
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


@task(
    cache=True,
    cache_version="1",
    requests=Resources(cpu="1", mem="1Gi"),
)
def parse_and_chunk_task(source_doc: str, pdf_bytes_b64: str) -> str:
    """
    Parse a PDF and split into overlapping text chunks.

    Args:
        source_doc:     Filename used as the Document node name in Neo4j.
        pdf_bytes_b64:  Base64-encoded PDF bytes.

    Returns:
        JSON string — list of {source_doc, chunk_index, chunk_text}.
    """
    raw = base64.b64decode(pdf_bytes_b64)

    with fitz.open(stream=raw, filetype="pdf") as doc:
        full_text = "\n".join(page.get_text() for page in doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(full_text)

    chunks = [
        {"source_doc": source_doc, "chunk_index": i, "chunk_text": text}
        for i, text in enumerate(texts)
    ]

    return json.dumps(chunks)
