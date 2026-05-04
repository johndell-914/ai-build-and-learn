"""
query/generation.py

Task: generate_task

Responsibility:
    - Receive the user question + retrieved context (chunks, entities, or community summary)
    - Build a RAG prompt tailored to the retrieval mode used:
        hybrid context   → cite source docs and entity relationships
        entity context   → explain how entities are connected
        community context → synthesize across the community summary
    - Call Claude Sonnet to generate a grounded answer
    - Return JSON: {answer, sources, retrieval_mode, entities_used}
"""
