"""
Best-effort de-identification (PHI redaction) for clinical notes.

IMPORTANT - read this
---------------------
This reduces risk; it does NOT make you HIPAA-compliant on its own. Automated
de-identification always misses some PHI and over-redacts other text. HIPAA
"de-identified" status requires either Safe Harbor (ALL 18 identifier types
removed AND no residual re-identification risk) or Expert Determination by a
qualified statistician. Treat this as a strong first pass that a human must
still review. You remain responsible for the data.

Two layers
----------
1. Regex backstops for STRUCTURED identifiers that models often miss or split:
   SSN, email, URL, IP, phone/fax, medical record numbers, dates, ages > 89.
2. A contextual model pass (default: obi/deid_roberta_i2b2, trained on the i2b2
   2014 de-id challenge) for NAMES, LOCATIONS, HOSPITALS, IDs, etc.

The model pass is optional and injectable (`ner_fn`) so this module's logic is
unit-tested with NO model download. Long notes are chunked with the same
sliding-window strategy as extraction, so nothing past 512 tokens is missed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .chunking import sliding_window_chunks


@dataclass
class Redaction:
    start: int
    end: int
    tag: str          # NAME, DATE, SSN, PHONE, ...
    text: str         # the original PHI (kept in memory only, never written out)


# --- Regex backstops -------------------------------------------------------- #
# Order matters: more specific patterns first so they win span conflicts.
_MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
           r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
           r"Nov(?:ember)?|Dec(?:ember)?)")

REGEX_PATTERNS: List[Tuple[str, str]] = [
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("EMAIL", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("URL", r"\bhttps?://\S+\b"),
    ("IP", r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    # MRN / record numbers (redact the whole phrase incl. the number)
    ("ID", r"\b(?:MRN|MR#|medical record(?:\s*(?:number|no\.?|#))?|record\s*(?:number|no\.?|#))\s*[:#]?\s*\d{3,}\b"),
    ("ID", r"\b(?:account|acct|patient\s*id|encounter)\s*[:#]?\s*\d{3,}\b"),
    # Dates: numeric and month-name forms
    ("DATE", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    ("DATE", r"\b\d{4}-\d{2}-\d{2}\b"),
    ("DATE", r"\b" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b"),
    ("DATE", r"\b\d{1,2}\s+" + _MONTHS + r"\.?\s+\d{4}\b"),
    # Phone / fax
    ("PHONE", r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    # ZIP only in address context ", ST 12345"
    ("LOCATION", r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"),
    # Age > 89 (HIPAA requires redacting ages over 89)
    ("AGE", r"\b(?:9\d|1\d\d)\s*(?:[-\s]?year[s]?[-\s]?old|y/?o|yo)\b"),
    ("AGE", r"\bage[d]?\s*[:]?\s*(?:9\d|1\d\d)\b"),
]

_COMPILED = [(tag, re.compile(pat, re.IGNORECASE)) for tag, pat in REGEX_PATTERNS]

# Map the de-id model's entity groups onto our generic tags.
MODEL_TAG_MAP = {
    "PATIENT": "NAME", "STAFF": "NAME", "DOCTOR": "NAME", "USERNAME": "NAME",
    "HCW": "NAME", "PER": "NAME", "PERSON": "NAME", "NAME": "NAME",
    "HOSP": "LOCATION", "HOSPITAL": "LOCATION", "LOC": "LOCATION",
    "LOCATION": "LOCATION", "ORG": "LOCATION", "PATORG": "LOCATION",
    "STREET": "LOCATION", "CITY": "LOCATION", "STATE": "LOCATION",
    "ZIP": "LOCATION", "COUNTRY": "LOCATION",
    "DATE": "DATE", "AGE": "AGE", "PHONE": "PHONE", "FAX": "PHONE",
    "EMAIL": "EMAIL", "URL": "URL", "ID": "ID", "IDNUM": "ID",
    "MEDICALRECORD": "ID", "BIOID": "ID", "DEVICE": "ID",
    "OTHERPHI": "OTHER",
}


class Deidentifier:
    def __init__(
        self,
        use_model: bool = True,
        model_name: str = "obi/deid_roberta_i2b2",
        ner_fn: Optional[Callable[[str], List[dict]]] = None,
        max_tokens: int = 400,
        stride: int = 50,
        keep_tags=None,
    ):
        # Tags listed here are NEVER redacted (left in the text as-is).
        # e.g. keep_tags={"AGE", "LOCATION"}. Note: keeping these REDUCES the
        # privacy protection (see README - ages > 89 and sub-state geography are
        # HIPAA Safe Harbor identifiers).
        self.keep_tags = set(t.upper() for t in (keep_tags or []))
        self.use_model = use_model
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.stride = stride
        self._ner_fn = ner_fn          # injectable for tests / custom models
        self._pipe = None
        self._tok = None

    # ------------------------------------------------------------------ #
    def deidentify(self, text: str, replacement: str = "[{tag}]") -> Tuple[str, List[Redaction]]:
        """Return (clean_text, redactions). PHI values stay in `redactions`
        only (in memory); they are never returned in the clean text."""
        spans: List[Redaction] = []

        # Layer 1: regex backstops
        for tag, rx in _COMPILED:
            if tag in self.keep_tags:
                continue
            for m in rx.finditer(text):
                spans.append(Redaction(m.start(), m.end(), tag, m.group(0)))

        # Layer 2: contextual model
        if self.use_model:
            for hit in self._run_model(text):
                grp = (hit.get("entity_group") or hit.get("entity") or "OTHER").upper()
                tag = MODEL_TAG_MAP.get(grp, "OTHER")
                if tag in self.keep_tags:
                    continue
                s, e = int(hit["start"]), int(hit["end"])
                if e > s:
                    spans.append(Redaction(s, e, tag, text[s:e]))

        spans = self._merge(spans)
        clean = self._apply(text, spans, replacement)
        return clean, spans

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge(spans: List[Redaction]) -> List[Redaction]:
        """Resolve overlaps: keep the widest span; on a tie keep the first."""
        if not spans:
            return []
        ordered = sorted(spans, key=lambda r: (r.start, -(r.end - r.start)))
        kept: List[Redaction] = []
        for r in ordered:
            if kept and r.start < kept[-1].end:
                # overlaps the last kept span; extend if this one reaches further
                if r.end > kept[-1].end:
                    kept[-1].end = r.end
                continue
            kept.append(r)
        return kept

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply(text: str, spans: List[Redaction], replacement: str) -> str:
        """Replace spans right-to-left so earlier offsets stay valid."""
        out = text
        for r in sorted(spans, key=lambda x: x.start, reverse=True):
            out = out[:r.start] + replacement.format(tag=r.tag) + out[r.end:]
        return out

    # ------------------------------------------------------------------ #
    def _run_model(self, text: str) -> List[dict]:
        if self._ner_fn is not None:
            try:
                return self._ner_fn(text)
            except Exception:
                return []
        try:
            self._ensure_pipe()
        except Exception:
            return []   # failure-soft: fall back to regex-only

        results: List[dict] = []
        for chunk in sliding_window_chunks(text, self._tok, self.max_tokens, self.stride):
            try:
                for grp in self._pipe(chunk.text):
                    results.append({
                        "entity_group": grp.get("entity_group", grp.get("entity")),
                        "start": chunk.char_start + int(grp["start"]),
                        "end": chunk.char_start + int(grp["end"]),
                    })
            except Exception:
                continue
        return results

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                                   pipeline)
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._pipe = pipeline("token-classification", model=model,
                              tokenizer=self._tok, aggregation_strategy="simple",
                              device=device)
