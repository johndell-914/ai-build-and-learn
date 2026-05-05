"""
query/routing.py

Task: route_query_task

Responsibility:
    - Receive the user's question
    - Call Claude Sonnet to classify the question into one of three retrieval modes:
        Mode A — HYBRID:    factual / specific questions
                            → vector search + graph expansion
        Mode B — ENTITY:    relationship / named-entity questions
                            → direct entity lookup + neighborhood traversal
        Mode C — COMMUNITY: thematic / global questions
                            → community summary search
    - Return the selected mode as a string: "hybrid" | "entity" | "community"

Examples:
    "What is the return window?"                       → hybrid
    "What benefits do Elite members get on returns?"   → entity
    "What programs does Everstorm have for customers?" → community
"""

from flytekit import task, Resources

from config import CLAUDE_MODEL, anthropic_client

_ROUTE_TOOL = {
    "name": "classify_query",
    "description": "Classify a user question into the best retrieval mode.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["hybrid", "entity", "community"],
                "description": (
                    "hybrid    — factual or specific questions best answered by "
                    "vector search over document chunks plus nearby entity context.\n"
                    "entity    — questions about named entities, how things relate, "
                    "or multi-hop connections between specific concepts.\n"
                    "community — broad thematic questions about programs, categories, "
                    "or overarching topics that span many documents."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining why this mode was chosen.",
            },
        },
        "required": ["mode", "reasoning"],
    },
}

_SYSTEM_PROMPT = (
    "You are a query router for a GraphRAG system built on Everstorm Outfitters "
    "policy and product documents. "
    "Given a user question, choose the retrieval mode that will produce the best answer:\n"
    "  hybrid    — specific facts, definitions, rules, dates, or numbers\n"
    "  entity    — named entities, relationships between things, multi-hop connections\n"
    "  community — broad themes, program overviews, 'what does Everstorm offer' style questions"
)


@task(requests=Resources(cpu="1", mem="500Mi"))
def route_query_task(question: str) -> str:
    """
    Classify the user question into a retrieval mode.

    Args:
        question: The user's natural-language question.

    Returns:
        One of "hybrid", "entity", or "community".
    """
    client = anthropic_client()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        tools=[_ROUTE_TOOL],
        tool_choice={"type": "tool", "name": "classify_query"},
        messages=[{"role": "user", "content": question}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input["mode"]
