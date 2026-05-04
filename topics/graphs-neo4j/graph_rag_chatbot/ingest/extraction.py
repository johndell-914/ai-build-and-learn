"""
ingest/extraction.py

Task: extract_entities_task

Responsibility:
    - Receive a single chunk of text
    - Call Claude Sonnet via tool use (structured output) to extract:
        entities:      [{name, type, description}]
        relationships: [{source, target, type, description}]
    - Constrain extraction to Everstorm ontology:
        Entity types:       PRODUCT, POLICY, PROGRAM, TIER, BENEFIT, CONDITION, PROCESS
        Relationship types: HAS_POLICY, QUALIFIES_FOR, REQUIRES, APPLIES_TO, PART_OF, COVERS
    - Return JSON: {chunk_id, entities, relationships}

Runs as a map_task in ingest_pipeline — one Claude call per chunk, fully parallel.
"""

import json

from flytekit import task, Resources

from config import CLAUDE_MODEL, ENTITY_TYPES, RELATIONSHIP_TYPES, anthropic_client

_EXTRACT_TOOL = {
    "name": "extract_graph",
    "description": (
        "Extract entities and relationships from a text chunk using the Everstorm ontology."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "type":        {"type": "string", "enum": ENTITY_TYPES},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "type", "description"],
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source":      {"type": "string"},
                        "target":      {"type": "string"},
                        "type":        {"type": "string", "enum": RELATIONSHIP_TYPES},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "target", "type", "description"],
                },
            },
        },
        "required": ["entities", "relationships"],
    },
}

_SYSTEM_PROMPT = (
    "You are a knowledge-graph extraction assistant for Everstorm Outfitters, "
    "an outdoor gear and apparel company. "
    "Extract only entities and relationships that appear explicitly in the provided text. "
    "Do not infer or hallucinate facts not present in the chunk. "
    f"Entity types: {', '.join(ENTITY_TYPES)}. "
    f"Relationship types: {', '.join(RELATIONSHIP_TYPES)}."
)


@task(
    cache=True,
    cache_version="1",
    requests=Resources(cpu="1", mem="500Mi"),
)
def extract_entities_task(chunk_json: str) -> str:
    """
    Extract entities and relationships from a single text chunk via Claude tool use.

    Args:
        chunk_json: JSON string with {source_doc, chunk_index, chunk_text}.

    Returns:
        JSON string — {chunk_id, source_doc, chunk_text, entities, relationships}.
    """
    chunk = json.loads(chunk_json)
    source_doc = chunk["source_doc"]
    chunk_index = chunk["chunk_index"]
    chunk_text = chunk["chunk_text"]
    chunk_id = f"{source_doc}::{chunk_index}"

    client = anthropic_client()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_graph"},
        messages=[
            {
                "role": "user",
                "content": f"Extract entities and relationships from this text:\n\n{chunk_text}",
            }
        ],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    extracted = tool_block.input

    result = {
        "chunk_id": chunk_id,
        "source_doc": source_doc,
        "chunk_text": chunk_text,
        "entities": extracted.get("entities", []),
        "relationships": extracted.get("relationships", []),
    }

    return json.dumps(result)
