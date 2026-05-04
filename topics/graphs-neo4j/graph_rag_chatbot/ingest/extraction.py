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
