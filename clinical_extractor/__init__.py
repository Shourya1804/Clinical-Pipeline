"""
clinical_extractor
==================

A small, dependency-light pipeline that turns a fine-tuned ClinicalBERT-style
NER model into a usable medical-entity extractor for long clinical notes.

It implements the four things a base model does NOT give you out of the box:

  1. chunking.py   - sliding-window chunking so notes longer than 512 tokens
                     are not silently truncated.
  2. pipeline.py   - the extractor: runs the HF token-classification pipeline
                     on each chunk and reconciles entities back to ORIGINAL
                     character offsets (sub-word stitching handled by the
                     pipeline's aggregation_strategy).
  3. negation.py   - a NegEx-style rule layer that decides whether each entity
                     is affirmed, negated, or uncertain.
  4. cli.py        - an end-to-end runner.

See README.md for the design notes and the corrections to the original guide.
"""

from .chunking import sliding_window_chunks, Chunk
from .negation import NegEx, Assertion
from .pipeline import ClinicalExtractor, Entity

__all__ = [
    "sliding_window_chunks",
    "Chunk",
    "NegEx",
    "Assertion",
    "ClinicalExtractor",
    "Entity",
]
