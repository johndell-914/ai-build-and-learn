"""
config.py

Central constants and connection helpers for graph_rag_chatbot.
Loaded by every ingest and query task.

Secret resolution order:
  1. Flyte secrets (when running on Union cluster)
  2. Environment variables / .env (local development)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _secret(key: str) -> str:
    """Return secret from Flyte context when on cluster, env var otherwise."""
    try:
        from flytekit import current_context
        return current_context().secrets.get(key=key)
    except Exception:
        return os.environ[key]


def neo4j_driver():
    """Return an open Neo4j driver. Caller must close it (use with driver: ...)."""
    from neo4j import GraphDatabase
    uri = _secret("NEO4J_URI")
    user = _secret("NEO4J_USERNAME")
    password = _secret("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def anthropic_client():
    """Return an Anthropic client."""
    from anthropic import Anthropic
    return Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # gte-small compatible, 384D
EMBED_DIM = 384

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150     # characters

# ---------------------------------------------------------------------------
# Graph schema — Everstorm ontology
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "PRODUCT",
    "POLICY",
    "PROGRAM",
    "TIER",
    "BENEFIT",
    "CONDITION",
    "PROCESS",
]

RELATIONSHIP_TYPES = [
    "HAS_POLICY",
    "QUALIFIES_FOR",
    "REQUIRES",
    "APPLIES_TO",
    "PART_OF",
    "COVERS",
]

# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------

VECTOR_INDEX_NAME = "chunk-embeddings"
VECTOR_SIMILARITY = "cosine"

# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

ENTITY_MERGE_THRESHOLD = 0.95   # cosine similarity above which two entities are merged

# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

LOUVAIN_RESOLUTION = 1.0        # higher = more, smaller communities
