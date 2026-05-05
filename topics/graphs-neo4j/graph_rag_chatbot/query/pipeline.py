"""
query/pipeline.py

Full GraphRAG query pipeline as a single flyte 2.x task.

Routes the question → retrieves context → generates an answer.
"""

from config import task_env
from query.routing import route_query
from query.retrieval import hybrid_retrieve, entity_retrieve, community_retrieve
from query.generation import generate


@task_env.task
def query_pipeline(question: str) -> str:
    """
    Full GraphRAG query pipeline.

    Args:
        question: The user's natural-language question.

    Returns:
        JSON — {answer, sources, retrieval_mode, entities_used}.
    """
    mode = route_query(question=question)

    if mode == "entity":
        context = entity_retrieve(question=question)
    elif mode == "community":
        context = community_retrieve(question=question)
    else:
        context = hybrid_retrieve(question=question)

    return generate(question=question, context_json=context)
