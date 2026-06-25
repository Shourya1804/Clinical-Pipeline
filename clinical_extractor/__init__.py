"""
clinical_extractor
==================

A small, dependency-light pipeline that turns a fine-tuned ClinicalBERT-style
NER model into a usable medical-entity extractor for long clinical notes.

Modules:
  1. chunking.py   - sliding-window chunking (handles the 512-token limit).
  2. pipeline.py   - the extractor: HF NER per chunk, reconciled to offsets.
  3. negation.py   - NegEx-style rule layer (affirmed / negated / possible).
  4. linking.py    - map entities to RxNorm / SNOMED codes (optional, network).
  5. deid.py       - best-effort PHI redaction (regex + de-id model).
  6. ingest.py     - multi-format, streaming document loading (scales to millions).
  7. cli.py        - single-note and streamed batch runner.

See README.md for design notes and corrections to the original guide.
"""

from .chunking import sliding_window_chunks, Chunk
from .negation import NegEx, Assertion
from .pipeline import ClinicalExtractor, Entity
from .linking import TerminologyLinker
from .deid import Deidentifier, Redaction
from .ingest import iter_documents, extract_text, supported_extensions

__all__ = [
    "sliding_window_chunks",
    "Chunk",
    "NegEx",
    "Assertion",
    "ClinicalExtractor",
    "Entity",
    "TerminologyLinker",
    "Deidentifier",
    "Redaction",
    "iter_documents",
    "extract_text",
    "supported_extensions",
]
