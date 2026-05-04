"""
ingest/enrichment.py

Tasks: resolve_entities_task, detect_communities_task, summarize_communities_task

Responsibility:
    resolve_entities_task:
        - Query all Entity nodes from Neo4j
        - Embed entity names with gte-small
        - Merge entity nodes with cosine similarity > 0.95 (de-duplication)
        - Handles cases like "Everstorm", "Everstorm Outfitters", "the Company"
        - Uses neo4j-graphrag EntityResolver component

    detect_communities_task:
        - Query all Entity nodes and RELATIONSHIP edges from Neo4j
        - Build a NetworkX graph from the entity relationship data
        - Run Louvain community detection via python-louvain
        - Assign community IDs back to Entity nodes in Neo4j
        - Note: uses Python workaround — GDS library not available on AuraDB Free

    summarize_communities_task:
        - Query each community's member entities and relationships from Neo4j
        - Call Claude Sonnet to generate a summary paragraph per community
        - Create Community nodes in Neo4j with the summary text
        - Create BELONGS_TO edges (Entity → Community)
"""
