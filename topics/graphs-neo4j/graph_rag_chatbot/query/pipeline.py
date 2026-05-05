"""
query/pipeline.py

Orchestrator: query_pipeline

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

from flytekit import workflow, conditional

from query.routing import route_query_task
from query.retrieval import hybrid_retrieve_task, entity_retrieve_task, community_retrieve_task
from query.generation import generate_task


@workflow
def query_pipeline(question: str) -> str:
    """
    Full GraphRAG query pipeline.

    Routes the question to the best retrieval mode, fetches relevant context
    from Neo4j, and generates a grounded answer via Claude.

    Args:
        question: The user's natural-language question.

    Returns:
        JSON — {answer, sources, retrieval_mode, entities_used}.
    """
    mode = route_query_task(question=question)

    context = (
        conditional("retrieval_branch")
        .if_(mode == "hybrid").then(hybrid_retrieve_task(question=question))
        .elif_(mode == "entity").then(entity_retrieve_task(question=question))
        .else_().then(community_retrieve_task(question=question))
    )

    return generate_task(question=question, context_json=context)
