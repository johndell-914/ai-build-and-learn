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
