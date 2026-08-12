"""Typed persistent data model for the full future product.

Phase 1 populates the media/frame/validation/run models; the caption and
evidence models define the schema that later phases (shots, characters,
events, captions) will fill in. Everything that is written to disk as JSON
goes through these Pydantic models.
"""
