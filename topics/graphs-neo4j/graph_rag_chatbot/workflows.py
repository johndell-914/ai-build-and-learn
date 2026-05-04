"""
workflows.py — Entry point for Flyte task registration.

Imports ingest_pipeline and query_pipeline so flyte deploy bundles
all task modules transitively. app.py imports only from this file.

    ingest_pipeline  ←  ingest/pipeline.py
    query_pipeline   ←  query/pipeline.py
"""

from ingest import ingest_pipeline
from query import query_pipeline

__all__ = ["ingest_pipeline", "query_pipeline"]
