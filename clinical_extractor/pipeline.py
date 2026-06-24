"""
The extractor.

Flow
----
note -> sliding_window_chunks -> HF token-classification pipeline (per chunk)
     -> shift spans back to original offsets -> reconcile duplicates
     -> attach NegEx assertion -> list[Entity]

Sub-word stitching note
-----------------------
The original guide tells you to hand-write code that glues "Hydro ##chloro
##thia ##zide" back together. You no longer have to: the HF pipeline's
`aggregation_strategy` does exactly that. We pass aggregation_strategy="max"
(or "first"/"average") and the pipeline returns whole-word entity groups with
correct character offsets. We keep our own reconciliation for the *cross-chunk*
overlap, which the pipeline cannot know about.

Device
------
Defaults to CPU. If a CUDA GPU is present we use it automatically. Batch size is
1 by design for the 16 GB-laptop case; bump it only when you have a GPU.
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
    code: Optional[str] = None          # e.g. RxNorm RXCUI or SNOMED code
    code_system: Optional[str] = None   # "RXNORM" | "SNOMEDCT_US" | ...
    code_name: Optional[str] = None     # canonical concept name
    link_score: Optional[float] = None  # match confidence if provided

    def as_dict(self) -> dict:
        return asdict(self)


# A reasonable default that actually exists on the Hub and is fine-tuned for
# clinical/biomedical NER. Swap via the `model_name` arg for an n2c2/i2b2 model.
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
    ):
        # Imports are local so that importing this module (e.g. for unit tests of
        # chunking/negation) does NOT require torch/transformers to be installed.
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
                # Shift the chunk-local offsets back to the original note.
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

        if self.run_negation:
            for ent in merged:
                ent.assertion = self._assert(note, ent).value

        merged.sort(key=lambda e: e.start)
        return merged

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconcile(entities: List[Entity]) -> List[Entity]:
        """Drop duplicate entities produced by the overlapping windows.

        Two hits are the "same" if their character spans overlap AND they carry
        the same label. We keep the higher-scoring one. This is what stops a
        disease sitting on a chunk boundary from being counted twice.
        """
        if not entities:
            return []
        ordered = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        kept: List[Entity] = []
        for ent in ordered:
            dup = False
            for k in kept:
                overlap = min(ent.end, k.end) - max(ent.start, k.start)
                if overlap > 0 and ent.label == k.label:
                    # same finding seen twice -> keep the better score
                    if ent.score > k.score:
                        k.text, k.start, k.end, k.score = (
                            ent.text, ent.start, ent.end, ent.score
                        )
                    dup = True
                    break
            if not dup:
                kept.append(ent)
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
