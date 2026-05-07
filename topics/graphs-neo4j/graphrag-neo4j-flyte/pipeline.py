"""Flyte 2 Graph RAG pipeline: toy AI-paper graph → Neo4j with vector index.

V1 scope: prove the plumbing works end-to-end. Data is a hardcoded toy set of
~20 AI/ML papers with fake authors and plausible-looking citation edges. Once
the pipeline is green we swap this seed for a real dataset (arXiv via
Semantic Scholar, OpenAlex, etc).

The pipeline writes:
  - (Paper {id, title, abstract, year, embedding})  embedding = 384-dim bge-small
  - (Author {name})
  - (Category {code})
  - (:Paper)-[:AUTHORED_BY]->(:Author)
  - (:Paper)-[:IN_CATEGORY]->(:Category)
  - (:Paper)-[:CITES]->(:Paper)
  - VECTOR INDEX paper_embedding_idx ON (:Paper.embedding)  cosine, 384 dims

Usage:
    flyte run --local --tui pipeline.py graphrag_pipeline
    flyte run pipeline.py graphrag_pipeline                 # remote (devbox)
"""

from __future__ import annotations

import logging
from typing import Any

import flyte
import flyte.report

from config import (
    NEO4J_HTTP_URL,
    NEO4J_PASSWORD,
    NEO4J_USER,
    pipeline_env,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = pipeline_env

EMBEDDING_DIM = 384  # bge-small-en-v1.5


# ──────────────────────────────────────────────────────────────────────────────
# Toy seed data — swap for arXiv/Semantic Scholar later.
# Each paper has (id, title, abstract, year, authors, categories, cites).
# ──────────────────────────────────────────────────────────────────────────────

TOY_PAPERS: list[dict[str, Any]] = [
    {
        "id": "p001", "year": 2017, "categories": ["cs.CL", "cs.LG"],
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer, a model architecture eschewing recurrence and "
                    "instead relying entirely on an attention mechanism to draw global dependencies "
                    "between input and output.",
        "authors": ["A. Vaswani", "N. Shazeer", "N. Parmar"],
        "cites": [],
    },
    {
        "id": "p002", "year": 2018, "categories": ["cs.CL"],
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce a new language representation model called BERT, which stands for "
                    "Bidirectional Encoder Representations from Transformers.",
        "authors": ["J. Devlin", "M. Chang", "K. Lee"],
        "cites": ["p001"],
    },
    {
        "id": "p003", "year": 2020, "categories": ["cs.CL", "cs.LG"],
        "title": "Language Models are Few-Shot Learners",
        "abstract": "We train GPT-3, an autoregressive language model with 175 billion parameters, "
                    "and test its performance in the few-shot setting.",
        "authors": ["T. Brown", "B. Mann", "N. Ryder"],
        "cites": ["p001", "p002"],
    },
    {
        "id": "p004", "year": 2020, "categories": ["cs.CL", "cs.IR"],
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "abstract": "We introduce RAG models which combine pre-trained parametric and non-parametric "
                    "memory for language generation.",
        "authors": ["P. Lewis", "E. Perez", "A. Piktus"],
        "cites": ["p001", "p002"],
    },
    {
        "id": "p005", "year": 2020, "categories": ["cs.CL", "cs.IR"],
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "abstract": "Open-domain question answering relies on efficient passage retrieval to select "
                    "candidate contexts. We show that retrieval can be practically implemented using "
                    "dense representations alone.",
        "authors": ["V. Karpukhin", "B. Oguz", "S. Min"],
        "cites": ["p002"],
    },
    {
        "id": "p006", "year": 2021, "categories": ["cs.CL"],
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "abstract": "We present Sentence-BERT, a modification of BERT that uses siamese network "
                    "structures to derive semantically meaningful sentence embeddings.",
        "authors": ["N. Reimers", "I. Gurevych"],
        "cites": ["p002"],
    },
    {
        "id": "p007", "year": 2022, "categories": ["cs.LG"],
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "abstract": "We explore how generating a chain of thought, a series of intermediate reasoning "
                    "steps, significantly improves the ability of large language models to perform "
                    "complex reasoning.",
        "authors": ["J. Wei", "X. Wang", "D. Schuurmans"],
        "cites": ["p003"],
    },
    {
        "id": "p008", "year": 2022, "categories": ["cs.LG", "cs.CL"],
        "title": "Training Language Models to Follow Instructions with Human Feedback",
        "abstract": "We show how RLHF can align language models with user intent, producing the "
                    "InstructGPT family of models.",
        "authors": ["L. Ouyang", "J. Wu", "X. Jiang"],
        "cites": ["p003"],
    },
    {
        "id": "p009", "year": 2022, "categories": ["cs.CL", "cs.IR"],
        "title": "Atlas: Few-shot Learning with Retrieval Augmented Language Models",
        "abstract": "We introduce Atlas, a carefully designed and pre-trained retrieval augmented "
                    "language model able to learn knowledge-intensive tasks with very few training examples.",
        "authors": ["G. Izacard", "P. Lewis", "M. Lomeli"],
        "cites": ["p004", "p005"],
    },
    {
        "id": "p010", "year": 2023, "categories": ["cs.CL"],
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "abstract": "We develop and release Llama 2, a collection of pretrained and fine-tuned large "
                    "language models ranging in scale from 7B to 70B parameters.",
        "authors": ["H. Touvron", "L. Martin", "K. Stone"],
        "cites": ["p001", "p008"],
    },
    {
        "id": "p011", "year": 2023, "categories": ["cs.LG"],
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "abstract": "We introduce DPO, a new parameterization of the reward model in RLHF that enables "
                    "extracting the corresponding optimal policy in closed form.",
        "authors": ["R. Rafailov", "A. Sharma", "E. Mitchell"],
        "cites": ["p008"],
    },
    {
        "id": "p012", "year": 2023, "categories": ["cs.CL", "cs.IR"],
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "abstract": "We analyze the performance of language models on tasks that require identifying "
                    "relevant information in their input contexts and find that performance is highest "
                    "when relevant information appears at the beginning or end.",
        "authors": ["N. Liu", "K. Lin", "J. Hewitt"],
        "cites": ["p003", "p010"],
    },
    {
        "id": "p013", "year": 2023, "categories": ["cs.CL", "cs.IR"],
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "abstract": "We introduce a new framework called Self-Reflective Retrieval-Augmented Generation "
                    "that enhances an LLM's quality and factuality through retrieval and self-reflection.",
        "authors": ["A. Asai", "Z. Wu", "Y. Wang"],
        "cites": ["p004", "p009", "p012"],
    },
    {
        "id": "p014", "year": 2024, "categories": ["cs.LG", "cs.CL"],
        "title": "Mixtral of Experts",
        "abstract": "We introduce Mixtral 8x7B, a Sparse Mixture of Experts language model with the "
                    "same architecture as Mistral 7B, with the difference that each layer is composed "
                    "of 8 feedforward blocks.",
        "authors": ["A. Jiang", "A. Sablayrolles", "A. Roux"],
        "cites": ["p010"],
    },
    {
        "id": "p015", "year": 2024, "categories": ["cs.IR", "cs.CL"],
        "title": "GraphRAG: Unlocking LLM Discovery on Narrative Private Data",
        "abstract": "We introduce a graph-based approach to RAG that uses LLM-generated knowledge "
                    "graphs to structure and summarize narrative documents for question answering.",
        "authors": ["D. Edge", "H. Trinh", "N. Cheng"],
        "cites": ["p004", "p013"],
    },
    {
        "id": "p016", "year": 2024, "categories": ["cs.CL"],
        "title": "Gemma: Open Models Based on Gemini Research and Technology",
        "abstract": "We introduce Gemma, a family of lightweight, state-of-the-art open models built "
                    "from the same research and technology used to create the Gemini models.",
        "authors": ["Gemma Team", "T. Mesnard", "C. Hardin"],
        "cites": ["p001", "p010"],
    },
    {
        "id": "p017", "year": 2024, "categories": ["cs.LG"],
        "title": "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits",
        "abstract": "We introduce a 1-bit LLM variant, BitNet b1.58, that matches full-precision "
                    "Transformer LLMs with the same model size and training tokens.",
        "authors": ["S. Ma", "H. Wang", "L. Ma"],
        "cites": ["p001", "p014"],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — produce the seed corpus.
# Returns the list as-is so a future swap (arXiv fetch) is a one-task change.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def fetch_papers() -> list[dict[str, Any]]:
    """Return the toy paper corpus. Replace with a real fetcher later.

    No `cache="auto"` here: the data lives in a module-level constant, so
    Flyte's auto-hash of the function body wouldn't change when we edit the
    seed list. Cheap task anyway.
    """
    n_papers = len(TOY_PAPERS)
    n_authors = len({a for p in TOY_PAPERS for a in p["authors"]})
    n_cats = len({c for p in TOY_PAPERS for c in p["categories"]})
    n_cites = sum(len(p["cites"]) for p in TOY_PAPERS)

    log.info(f"Toy corpus: {n_papers} papers, {n_authors} authors, "
             f"{n_cats} categories, {n_cites} citations")
    await flyte.report.replace.aio(
        f"<h2>Fetched papers (toy corpus)</h2>"
        f"<ul>"
        f"<li><b>Papers:</b> {n_papers}</li>"
        f"<li><b>Authors:</b> {n_authors}</li>"
        f"<li><b>Categories:</b> {n_cats}</li>"
        f"<li><b>Citation edges:</b> {n_cites}</li>"
        f"</ul>"
    )
    await flyte.report.flush.aio()
    return TOY_PAPERS


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — embed each paper's abstract with bge-small.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def embed_papers(
    papers: list[dict[str, Any]],
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> list[dict[str, Any]]:
    """Add an `embedding` (list[float], len=384) field to each paper."""
    from sentence_transformers import SentenceTransformer

    log.info(f"Embedding {len(papers)} abstracts with {embedding_model}")
    encoder = SentenceTransformer(embedding_model)
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    # bge-small expects L2-normalized vectors paired with cosine similarity.
    vectors = encoder.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    enriched = [{**p, "embedding": v} for p, v in zip(papers, vectors)]
    dim = len(vectors[0]) if vectors else 0
    log.info(f"Produced {len(enriched)} embeddings of dim {dim}")

    await flyte.report.replace.aio(
        f"<h2>Embedded abstracts</h2>"
        f"<p><b>Model:</b> {embedding_model}</p>"
        f"<p><b>Dimensions:</b> {dim}</p>"
        f"<p><b>Vectors:</b> {len(enriched)}</p>"
    )
    await flyte.report.flush.aio()
    return enriched


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — write nodes/edges/vector-index to Neo4j over the HTTP Cypher API.
#
# Why HTTP and not Bolt: Flyte 2 fronts apps with Knative Serving's queue-proxy,
# which is HTTP-only. Bolt (TCP/7687) doesn't pass through. Neo4j's HTTP API on
# 7474 supports the full Cypher surface, including vector index queries.
# ──────────────────────────────────────────────────────────────────────────────

def _run_cypher(
    client: "httpx.Client",
    cypher: str,
    params: dict[str, Any] | None = None,
) -> list[list[Any]]:
    """POST one Cypher statement to /db/neo4j/tx/commit and return the rows.

    Returns the `data[*].row` arrays from the first result. Raises if Neo4j
    reports any error in the response payload (HTTP 200 with `errors[]` set).
    """
    payload = {"statements": [{"statement": cypher, "parameters": params or {}}]}
    resp = client.post("/db/neo4j/tx/commit", json=payload)
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"Neo4j error: {body['errors']}")
    results = body.get("results", [])
    if not results:
        return []
    return [row["row"] for row in results[0].get("data", [])]


@env.task(report=True)
async def load_neo4j(
    papers: list[dict[str, Any]],
    http_url: str = NEO4J_HTTP_URL,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    wipe_first: bool = True,
) -> dict[str, int]:
    """MERGE nodes + edges into Neo4j and (re)create the vector index."""
    import httpx

    log.info(f"Connecting to Neo4j HTTP API at {http_url} as {user}")
    # The Knative service URL points to port 80 → queue-proxy → user port 7474.
    # Basic auth header on every request.
    with httpx.Client(
        base_url=http_url,
        auth=(user, password),
        timeout=30.0,
    ) as client:
        # Smoke-check: discovery doc at GET /. Confirms auth, version, routing.
        info = client.get("/").raise_for_status().json()
        log.info(f"Neo4j {info.get('neo4j_version')} {info.get('neo4j_edition')}")

        if wipe_first:
            _run_cypher(client, "MATCH (n) DETACH DELETE n")
            _run_cypher(client, "DROP INDEX paper_embedding_idx IF EXISTS")
            log.info("Wiped existing graph + vector index")

        # Constraints make MERGE-by-key fast and prevent dupes.
        _run_cypher(client, "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                            "FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        _run_cypher(client, "CREATE CONSTRAINT author_name IF NOT EXISTS "
                            "FOR (a:Author) REQUIRE a.name IS UNIQUE")
        _run_cypher(client, "CREATE CONSTRAINT category_code IF NOT EXISTS "
                            "FOR (c:Category) REQUIRE c.code IS UNIQUE")

        # Upsert papers + author + category nodes & edges.
        for p in papers:
            _run_cypher(client, """
                MERGE (p:Paper {id: $id})
                SET p.title = $title,
                    p.abstract = $abstract,
                    p.year = $year,
                    p.embedding = $embedding
            """, {
                "id": p["id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "year": p["year"],
                "embedding": p["embedding"],
            })
            for author in p["authors"]:
                _run_cypher(client, """
                    MERGE (a:Author {name: $name})
                    WITH a
                    MATCH (p:Paper {id: $pid})
                    MERGE (p)-[:AUTHORED_BY]->(a)
                """, {"name": author, "pid": p["id"]})
            for cat in p["categories"]:
                _run_cypher(client, """
                    MERGE (c:Category {code: $code})
                    WITH c
                    MATCH (p:Paper {id: $pid})
                    MERGE (p)-[:IN_CATEGORY]->(c)
                """, {"code": cat, "pid": p["id"]})

        # Citation edges in a second pass so all paper nodes exist.
        for p in papers:
            for cited in p["cites"]:
                _run_cypher(client, """
                    MATCH (src:Paper {id: $src}), (dst:Paper {id: $dst})
                    MERGE (src)-[:CITES]->(dst)
                """, {"src": p["id"], "dst": cited})

        # Native vector index — Neo4j 5.11+. Cosine matches the L2-normalized
        # bge-small embeddings produced upstream.
        _run_cypher(client, f"""
            CREATE VECTOR INDEX paper_embedding_idx IF NOT EXISTS
            FOR (p:Paper) ON (p.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

        counts = {
            "papers": _run_cypher(client, "MATCH (p:Paper) RETURN count(p)")[0][0],
            "authors": _run_cypher(client, "MATCH (a:Author) RETURN count(a)")[0][0],
            "categories": _run_cypher(client, "MATCH (c:Category) RETURN count(c)")[0][0],
            "cites_edges": _run_cypher(
                client, "MATCH ()-[r:CITES]->() RETURN count(r)"
            )[0][0],
            "authored_edges": _run_cypher(
                client, "MATCH ()-[r:AUTHORED_BY]->() RETURN count(r)"
            )[0][0],
        }

    log.info(f"Loaded into Neo4j: {counts}")
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counts.items())
    await flyte.report.replace.aio(
        f"<h2>Loaded into Neo4j</h2>"
        f"<p><b>HTTP URL:</b> <code>{http_url}</code></p>"
        f"<table border=1 cellpadding=4>"
        f"<tr><th>Entity</th><th>Count</th></tr>{rows}</table>"
        f"<p>Vector index <code>paper_embedding_idx</code> "
        f"({EMBEDDING_DIM} dims, cosine) created.</p>"
    )
    await flyte.report.flush.aio()
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def graphrag_pipeline(
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    http_url: str = NEO4J_HTTP_URL,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    wipe_first: bool = True,
) -> dict[str, int]:
    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline</h2><p>Step 1/3 — fetching papers…</p>"
    )
    await flyte.report.flush.aio()
    papers = await fetch_papers()

    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline</h2><p>Step 2/3 — embedding abstracts…</p>"
    )
    await flyte.report.flush.aio()
    embedded = await embed_papers(papers, embedding_model)

    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline</h2><p>Step 3/3 — loading Neo4j…</p>"
    )
    await flyte.report.flush.aio()
    counts = await load_neo4j(embedded, http_url, user, password, wipe_first)

    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counts.items())
    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline complete</h2>"
        f"<table border=1 cellpadding=4>"
        f"<tr><th>Entity</th><th>Count</th></tr>{rows}</table>"
    )
    await flyte.report.flush.aio()
    return counts


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(graphrag_pipeline)
    print(f"Pipeline run: {run.name}")
    print(f"  {run.url}")
