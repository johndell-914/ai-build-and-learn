"""
ingest/enrichment.py

Tasks: resolve_entities_task, detect_communities_task, summarize_communities_task

Responsibility:
    resolve_entities_task:
        - Query all Entity nodes from Neo4j
        - Embed entity names with gte-small
        - Merge entity nodes with cosine similarity > 0.95 (de-duplication)
        - Handles cases like "Everstorm", "Everstorm Outfitters", "the Company"

    detect_communities_task:
        - Query all Entity nodes and RELATIONSHIP edges from Neo4j
        - Build a NetworkX graph from the entity relationship data
        - Run Louvain community detection via python-louvain
        - Assign community IDs back to Entity nodes in Neo4j
        - Note: uses Python Louvain — GDS not available on AuraDB Free tier

    summarize_communities_task:
        - Query each community's member entities and relationships from Neo4j
        - Call Claude Sonnet to generate a summary paragraph per community
        - Create Community nodes in Neo4j with the summary text
        - Create BELONGS_TO edges (Entity → Community)
"""

import json
from collections import defaultdict
from itertools import combinations

import networkx as nx
import community as louvain_community  # python-louvain
import numpy as np
from flytekit import task, Resources
from sentence_transformers import SentenceTransformer

from config import (
    CLAUDE_MODEL,
    EMBED_MODEL,
    ENTITY_MERGE_THRESHOLD,
    LOUVAIN_RESOLUTION,
    anthropic_client,
    neo4j_driver,
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@task(requests=Resources(cpu="2", mem="2Gi"))
def resolve_entities_task() -> str:
    """
    Merge near-duplicate Entity nodes in Neo4j using embedding cosine similarity.

    Two entities whose name embeddings exceed ENTITY_MERGE_THRESHOLD are merged:
    the lower-degree node's MENTIONS and RELATED edges are repointed to the
    higher-degree node, then the duplicate is deleted.

    Returns:
        JSON summary — {merges_performed}.
    """
    driver = neo4j_driver()
    model = SentenceTransformer(EMBED_MODEL)

    with driver:
        with driver.session() as session:
            rows = session.run("MATCH (e:Entity) RETURN e.name AS name").data()

    names = [r["name"] for r in rows]
    if len(names) < 2:
        return json.dumps({"merges_performed": 0})

    embeddings = model.encode(names)
    name_to_vec = {name: embeddings[i] for i, name in enumerate(names)}

    # Find pairs above threshold
    merge_pairs = []
    for a, b in combinations(names, 2):
        if _cosine_sim(name_to_vec[a], name_to_vec[b]) >= ENTITY_MERGE_THRESHOLD:
            merge_pairs.append((a, b))

    merges_performed = 0
    driver2 = neo4j_driver()
    with driver2:
        with driver2.session() as session:
            for keep_name, drop_name in merge_pairs:
                # Repoint MENTIONS edges from duplicate to canonical node
                session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(drop:Entity {name: $drop})
                    MATCH (keep:Entity {name: $keep})
                    MERGE (c)-[:MENTIONS]->(keep)
                    """,
                    drop=drop_name,
                    keep=keep_name,
                )
                # Repoint outgoing RELATED edges
                session.run(
                    """
                    MATCH (drop:Entity {name: $drop})-[r:RELATED]->(other:Entity)
                    MATCH (keep:Entity {name: $keep})
                    MERGE (keep)-[:RELATED {type: r.type, description: r.description}]->(other)
                    """,
                    drop=drop_name,
                    keep=keep_name,
                )
                # Repoint incoming RELATED edges
                session.run(
                    """
                    MATCH (other:Entity)-[r:RELATED]->(drop:Entity {name: $drop})
                    MATCH (keep:Entity {name: $keep})
                    MERGE (other)-[:RELATED {type: r.type, description: r.description}]->(keep)
                    """,
                    drop=drop_name,
                    keep=keep_name,
                )
                # Delete duplicate
                session.run(
                    "MATCH (e:Entity {name: $drop}) DETACH DELETE e",
                    drop=drop_name,
                )
                merges_performed += 1

    return json.dumps({"merges_performed": merges_performed})


@task(requests=Resources(cpu="2", mem="2Gi"))
def detect_communities_task(resolve_summary: str) -> str:
    """
    Run Louvain community detection over the entity graph and write community
    IDs back to Neo4j Entity nodes.

    Args:
        resolve_summary: JSON from resolve_entities_task (used for sequencing only).

    Returns:
        JSON summary — {communities_found, entities_assigned}.
    """
    driver = neo4j_driver()

    with driver:
        with driver.session() as session:
            entity_rows = session.run("MATCH (e:Entity) RETURN e.name AS name").data()
            rel_rows = session.run(
                "MATCH (a:Entity)-[:RELATED]->(b:Entity) "
                "RETURN a.name AS source, b.name AS target"
            ).data()

    G = nx.Graph()
    G.add_nodes_from(r["name"] for r in entity_rows)
    G.add_edges_from((r["source"], r["target"]) for r in rel_rows)

    partition = louvain_community.best_partition(G, resolution=LOUVAIN_RESOLUTION)

    # Write community_id back to each Entity node
    driver2 = neo4j_driver()
    with driver2:
        with driver2.session() as session:
            for entity_name, community_id in partition.items():
                session.run(
                    "MATCH (e:Entity {name: $name}) SET e.community_id = $cid",
                    name=entity_name,
                    cid=community_id,
                )

    communities_found = len(set(partition.values()))
    return json.dumps({
        "communities_found": communities_found,
        "entities_assigned": len(partition),
    })


@task(requests=Resources(cpu="1", mem="1Gi"))
def summarize_communities_task(detect_summary: str) -> str:
    """
    Generate a natural-language summary for each community and store it in Neo4j.

    Queries each community's entities and relationships, calls Claude to write
    a summary paragraph, then creates a Community node with BELONGS_TO edges.

    Args:
        detect_summary: JSON from detect_communities_task (used for sequencing only).

    Returns:
        JSON summary — {communities_summarized}.
    """
    driver = neo4j_driver()

    with driver:
        with driver.session() as session:
            rows = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "RETURN e.name AS name, e.type AS type, "
                "       e.description AS description, e.community_id AS cid"
            ).data()
            rel_rows = session.run(
                "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
                "WHERE a.community_id IS NOT NULL "
                "RETURN a.name AS source, b.name AS target, "
                "       r.type AS rel_type, r.description AS description, "
                "       a.community_id AS cid"
            ).data()

    # Group by community
    community_entities: dict = defaultdict(list)
    community_rels: dict = defaultdict(list)

    for r in rows:
        community_entities[r["cid"]].append(r)
    for r in rel_rows:
        community_rels[r["cid"]].append(r)

    client = anthropic_client()
    communities_summarized = 0

    driver2 = neo4j_driver()
    with driver2:
        with driver2.session() as session:
            for cid, entities in community_entities.items():
                rels = community_rels.get(cid, [])

                entity_lines = "\n".join(
                    f"- {e['name']} ({e['type']}): {e['description']}"
                    for e in entities
                )
                rel_lines = "\n".join(
                    f"- {r['source']} --[{r['rel_type']}]--> {r['target']}: {r['description']}"
                    for r in rels
                ) or "None"

                prompt = (
                    "You are summarizing a cluster of related concepts from Everstorm Outfitters "
                    "policy and product documents. Write a concise 2-3 sentence summary that "
                    "describes what this group of entities is about and how they relate to each other.\n\n"
                    f"Entities:\n{entity_lines}\n\n"
                    f"Relationships:\n{rel_lines}"
                )

                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary_text = response.content[0].text.strip()

                # Create Community node and BELONGS_TO edges
                session.run(
                    """
                    MERGE (comm:Community {id: $cid})
                    SET comm.summary = $summary
                    """,
                    cid=cid,
                    summary=summary_text,
                )
                for ent in entities:
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (comm:Community {id: $cid})
                        MERGE (e)-[:BELONGS_TO]->(comm)
                        """,
                        name=ent["name"],
                        cid=cid,
                    )
                communities_summarized += 1

    return json.dumps({"communities_summarized": communities_summarized})
