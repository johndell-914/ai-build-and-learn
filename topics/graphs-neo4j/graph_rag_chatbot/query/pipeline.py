"""
query/pipeline.py

Task: query_pipeline (orchestrator)

Responsibility:
    - Entry point for the full query flow
    - Calls route_query_task to determine retrieval mode (A / B / C)
    - Dispatches the appropriate retrieval task based on mode:
        "hybrid"    → hybrid_retrieve_task
        "entity"    → entity_retrieve_task
        "community" → community_retrieve_task
    - Passes retrieved context to generate_task
    - Returns JSON: {answer, sources, retrieval_mode, entities_used}

Imports from:
    query.routing   import route_query_task
    query.retrieval import hybrid_retrieve_task, entity_retrieve_task,
                           community_retrieve_task
    query.generation import generate_task
"""
