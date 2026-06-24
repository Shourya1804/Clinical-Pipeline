"""
clinical_extractor
==================

A small, dependency-light pipeline that turns a fine-tuned ClinicalBERT-style
NER model into a usable medical-entity extractor for long clinical notes.

Modules:
  1. chunking.py   - sliding-window chunking so notes longer than 512 tokens
                     are not silently truncated.
  2. pipeline.py   - the extractor: runs the HF token-classification pipeline
                     on each chunk and reconciles entities back to ORIGINAL
                     character offsets.
  3. negation.py   - a NegEx-style rule layer (affirmed / negated / possible).
  4. linking.py    - map entities to RxNorm / SNOMED codes (optional, network).
  5. cli.py        - single-note and batch runner.

See README.md for design notes and corrections to the original guide.
"""

from .chunking import sliding_window_chunks, Chunk
from .negation import NegEx, Assertion
from .pipeline import ClinicalExtractor, Entity
from .linking import TerminologyLinker

__all__ = [
    "sliding_window_chunks",
    "Chunk",
    "NegEx",
    "Assertion",
    "ClinicalExtractor",
    "Entity",
    "TerminologyLinker",
]
