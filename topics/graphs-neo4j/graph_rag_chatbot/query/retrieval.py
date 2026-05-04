"""
query/retrieval.py

Tasks: hybrid_retrieve_task, entity_retrieve_task, community_retrieve_task

Responsibility:
    hybrid_retrieve_task (Mode A):
        - Embed query with gte-small
        - Vector search finds top-k Chunk nodes via HNSW index
        - Cypher traversal expands outward: Chunk → Entity → related Entities
        - Returns chunks + entity context as JSON
        - Uses neo4j-graphrag VectorCypherRetriever

    entity_retrieve_task (Mode B):
        - Extract named entities from the question (Claude or spaCy NER)
        - Direct MATCH on Entity nodes by name
        - Retrieve entity neighborhood subgraph (1-2 hops)
        - Optionally find shortest path between two mentioned entities
        - Returns entity subgraph context as JSON

    community_retrieve_task (Mode C):
        - Embed question with gte-small
        - Find most relevant Community node by embedding similarity
        - Return community summary + member entity names
        - Returns community context as JSON
"""
