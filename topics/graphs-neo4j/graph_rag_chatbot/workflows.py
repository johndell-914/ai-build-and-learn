"""
workflows.py — Union / Flyte entry point for graph_rag_chatbot.

Register both pipelines with:

    union register workflows.py --project dellenbaugh --domain development

Run ingest (one-time, loads all PDFs into Neo4j):

    union run --remote workflows.py ingest_pipeline --data_dir data

Run a single query (returns JSON with answer + metadata):

    union run --remote workflows.py query_pipeline \
        --question "What is the return window for outdoor gear?"

Pipelines
---------
ingest_pipeline
    Reads all PDFs from data_dir, chunks them, extracts entities and
    relationships via Claude, writes the full graph to Neo4j AuraDB,
    builds the HNSW vector index, resolves duplicate entities, detects
    communities with Louvain, and generates community summaries.

    Run once after adding or updating source documents.

query_pipeline
    Routes the question to the best retrieval mode (hybrid / entity /
    community), fetches relevant context from Neo4j, and generates a
    grounded answer via Claude.

    Called per user question from app.py.
"""

from ingest import ingest_pipeline
from query import query_pipeline

__all__ = ["ingest_pipeline", "query_pipeline"]
