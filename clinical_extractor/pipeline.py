"""
The extractor.

Flow
----
note -> sliding_window_chunks -> HF token-classification pipeline (per chunk)
     -> shift spans back to original offsets -> reconcile duplicates
     -> (optional) add custom-dictionary matches (dictionary wins overlaps)
     -> attach NegEx assertion -> list[Entity]

Sub-word stitching is handled by the pipeline's aggregation_strategy; we keep
our own reconciliation for the cross-chunk overlap. A Gazetteer (custom term
dictionary) can be supplied to add user-controlled terms/abbreviations.

Device: CPU by default; CUDA used automatically if present. Batch size 1.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional

from .chunking import sliding_window_chunks
from .negation import NegEx, Assertion


@dataclass
class Entity:
    text: str
    label: str
    start: int          # char offset in the ORIGINAL note
    end: int
    score: float
    assertion: str      # affirmed | negated | possible
    # Terminology linking (filled in by linking.TerminologyLinker; optional).
    code: Optional[str] = None
    code_system: Optional[str] = None
    code_name: Optional[str] = None
    link_score: Optional[float] = None
    source: str = "model"   # "model" or "dictionary"

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_MODEL = "d4data/biomedical-ner-all"


class ClinicalExtractor:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_tokens: int = 400,
        stride: int = 50,
        aggregation_strategy: str = "max",
        device: Optional[int] = None,
        run_negation: bool = True,
        min_score: float = 0.0,
        gazetteer=None,
    ):
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            pipeline,
        )

        try:
            import torch
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
        except Exception:
            device = -1 if device is None else device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline(
            "token-classification",
            model=model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

        self.max_tokens = max_tokens
        self.stride = stride
        self.min_score = min_score
        self.run_negation = run_negation
        self.negex = NegEx() if run_negation else None
        self.gazetteer = gazetteer

    # ------------------------------------------------------------------ #
    def extract(self, note: str) -> List[Entity]:
        chunks = sliding_window_chunks(
            note, self.tokenizer, self.max_tokens, self.stride
        )

        raw: List[Entity] = []
        for chunk in chunks:                     # batch size 1 on purpose
            for grp in self.ner(chunk.text):
                score = float(grp.get("score", 0.0))
                if score < self.min_score:
                    continue
                start = chunk.char_start + int(grp["start"])
                end = chunk.char_start + int(grp["end"])
                raw.append(
                    Entity(
                        text=note[start:end],
                        label=grp.get("entity_group", grp.get("entity", "ENTITY")),
                        start=start,
                        end=end,
                        score=score,
                        assertion=Assertion.AFFIRMED.value,
                    )
                )

        merged = self._reconcile(raw)

        # Custom dictionary terms override the model on any overlap.
        if self.gazetteer is not None and len(self.gazetteer):
            merged = self._apply_gazetteer(merged, self.gazetteer.find(note))

        if self.run_negation:
            for ent in merged:
                ent.assertion = self._assert(note, ent).value

        merged.sort(key=lambda e: e.start)
        return merged

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconcile(entities: List[Entity]) -> List[Entity]:
        """Drop duplicate entities from overlapping windows (same label)."""
        if not entities:
            return []
        ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        kept: List[Entity] = []
        for ent in ordered:
            dup = False
            for k in kept:
                overlap = min(ent.end, k.end) - max(ent.start, k.start)
                if overlap > 0 and ent.label == k.label:
                    if ent.score > k.score:
                        k.text, k.start, k.end, k.score = (
                            ent.text, ent.start, ent.end, ent.score
                        )
                    dup = True
                    break
            if not dup:
                kept.append(ent)
        return kept

    @staticmethod
    def _apply_gazetteer(model_ents: List[Entity],
                         gaz_ents: List[Entity]) -> List[Entity]:
        """Add dictionary matches; on any character overlap the dictionary
        entity wins and the conflicting model entity is dropped (the user's
        explicit term is treated as authoritative)."""
        if not gaz_ents:
            return model_ents
        kept = []
        for m in model_ents:
            if any(m.start < g.end and g.start < m.end for g in gaz_ents):
                continue
            kept.append(m)
        kept.extend(gaz_ents)
        return kept

    # ------------------------------------------------------------------ #
    def _assert(self, note: str, ent: Entity) -> Assertion:
        """Find the sentence containing the entity and run NegEx on it."""
        left = max(
            note.rfind(".", 0, ent.start),
            note.rfind(";", 0, ent.start),
            note.rfind("\n", 0, ent.start),
        )
        sent_start = left + 1 if left != -1 else 0
        right_candidates = [
            i for i in (
                note.find(".", ent.end),
                note.find(";", ent.end),
                note.find("\n", ent.end),
            ) if i != -1
        ]
        sent_end = min(right_candidates) if right_candidates else len(note)

        sentence = note[sent_start:sent_end]
        return self.negex.classify(
            sentence, ent.start - sent_start, ent.end - sent_start
        )
