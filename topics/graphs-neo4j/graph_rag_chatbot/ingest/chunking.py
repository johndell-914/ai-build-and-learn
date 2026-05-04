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
