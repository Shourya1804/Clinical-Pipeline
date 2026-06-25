"""
Custom dictionary (gazetteer) + abbreviation expander.

This is how you "teach" the extractor new terms WITHOUT training a model and
WITHOUT labeled data: you add lines to a CSV, and the pipeline matches those
terms/abbreviations directly, in addition to whatever the model finds.

It is not machine learning - it's a lookup list you control. That makes it
instant, transparent, and auditable (you can see exactly which term was added
and why), which is ideal for a regulated clinical setting.

CSV format (header row required; extra columns ignored, missing ones optional):

    term,label,expansion,code,code_system
    SOB,Sign_symptom,shortness of breath,,
    HTN,Disease_disorder,hypertension,,
    HCTZ,Medication,hydrochlorothiazide,5487,RXNORM

  term         the text to match (a word, phrase, or abbreviation). Required.
  label        entity type to assign (default: CUSTOM).
  expansion    full meaning of an abbreviation; shown as the concept name.
  code         optional code to attach (e.g. an RxNorm/SNOMED id you trust).
  code_system  optional code system name for that code.

Blank lines and lines starting with # are ignored. Matching is whole-word and
case-insensitive by default; multi-word terms tolerate extra whitespace.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import List, Optional

from .pipeline import Entity


@dataclass
class _Entry:
    term: str
    label: str
    expansion: str
    code: Optional[str]
    code_system: Optional[str]
    regex: "re.Pattern"


def _compile(term: str, case_sensitive: bool = False) -> "re.Pattern":
    parts = [re.escape(w) for w in term.split()]
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    return re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)


class Gazetteer:
    """Matches user-supplied terms/abbreviations in text and emits Entities."""

    def __init__(self, entries: Optional[List[_Entry]] = None):
        self.entries: List[_Entry] = entries or []

    # ------------------------------------------------------------------ #
    @classmethod
    def from_rows(cls, rows) -> "Gazetteer":
        entries = []
        for row in rows:
            term = (row.get("term") or "").strip()
            if not term or term.startswith("#"):
                continue
            label = (row.get("label") or "CUSTOM").strip() or "CUSTOM"
            expansion = (row.get("expansion") or "").strip()
            code = (row.get("code") or "").strip() or None
            code_system = (row.get("code_system") or "").strip() or None
            cs = str(row.get("case_sensitive") or "").strip().lower() in ("1", "yes", "true")
            entries.append(_Entry(term, label, expansion, code, code_system,
                                  _compile(term, cs)))
        return cls(entries)

    @classmethod
    def from_file(cls, path: str) -> "Gazetteer":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Strip whole-line comments so DictReader doesn't choke on them.
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        # normalise header names to lowercase
        rows = []
        for raw in reader:
            rows.append({(k or "").strip().lower(): (v or "") for k, v in raw.items()})
        return cls.from_rows(rows)

    @classmethod
    def from_file_or_empty(cls, path: Optional[str]) -> "Gazetteer":
        import os
        if path and os.path.exists(path):
            return cls.from_file(path)
        return cls([])

    def __len__(self):
        return len(self.entries)

    # ------------------------------------------------------------------ #
    def find(self, text: str) -> List[Entity]:
        """Return dictionary matches in `text` as Entity objects.

        Overlapping dictionary matches are reduced to the longest one, so
        'congestive heart failure' beats a nested 'heart failure'.
        """
        hits: List[Entity] = []
        for e in self.entries:
            for m in e.regex.finditer(text):
                hits.append(Entity(
                    text=text[m.start():m.end()],
                    label=e.label,
                    start=m.start(),
                    end=m.end(),
                    score=1.0,
                    assertion="affirmed",      # NegEx in the pipeline refines this
                    code=e.code,
                    code_system=e.code_system,
                    code_name=e.expansion or e.term,
                    link_score=None,
                    source="dictionary",
                ))
        return self._longest_only(hits)

    @staticmethod
    def _longest_only(hits: List[Entity]) -> List[Entity]:
        if not hits:
            return []
        ordered = sorted(hits, key=lambda h: (h.start, -(h.end - h.start)))
        kept: List[Entity] = []
        for h in ordered:
            if any(h.start < k.end and k.start < h.end for k in kept):
                continue   # overlaps an already-kept (longer/earlier) match
            kept.append(h)
        return kept
