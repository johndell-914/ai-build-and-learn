"""
query/pipeline.py — query_pipeline orchestrator

Routes question → retrieves context → generates answer.
Each step is a separate Union task visible in the console.
"""

from config import task_env
from query.routing import route_query
from query.retrieval import hybrid_retrieve, entity_retrieve, community_retrieve
from query.generation import generate


@task_env.task
async def query_pipeline(question: str) -> str:
    """
    Full GraphRAG query pipeline.

    Union shows this as:
      query_pipeline
        ├── route_query
        ├── [hybrid_retrieve | entity_retrieve | community_retrieve]
        └── generate

    Args:
        question: The user's natural-language question.

    Returns:
        JSON — {answer, sources, retrieval_mode, entities_used}.
    """
    mode = await route_query(question=question)

    if mode == "entity":
        context = await entity_retrieve(question=question)
    elif mode == "community":
        context = await community_retrieve(question=question)
    else:
        context = await hybrid_retrieve(question=question)

    return await generate(question=question, context_json=context)
