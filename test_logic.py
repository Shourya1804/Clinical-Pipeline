"""
Logic tests that DO NOT require torch/transformers or any model download.

They validate the parts of the pipeline that are pure Python:
  - sliding-window chunking (via a tiny stub tokenizer)
  - cross-chunk entity reconciliation
  - NegEx assertion classification

Run:  python test_logic.py
"""

import re
import sys

from clinical_extractor.chunking import sliding_window_chunks
from clinical_extractor.negation import NegEx, Assertion
from clinical_extractor.pipeline import ClinicalExtractor, Entity


class StubTokenizer:
    """Whitespace tokenizer that returns offset mappings like a fast tokenizer.

    Enough to exercise chunking without pulling in transformers.
    """
    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def test_chunking():
    print("chunking:")
    tok = StubTokenizer()
    text = " ".join(f"w{i}" for i in range(1000))  # 1000 tokens

    ok = True
    chunks = sliding_window_chunks(text, tok, max_tokens=400, stride=50)
    ok &= _check("long note is split into multiple chunks", len(chunks) > 1)
    # step = 350, so chunks start at token 0,350,700,... -> 3 chunks
    ok &= _check("expected chunk count (1000 tok, step 350)", len(chunks) == 3)
    ok &= _check("first chunk starts at offset 0", chunks[0].char_start == 0)
    ok &= _check("last chunk reaches end of note",
                 chunks[-1].char_end == len(text))
    # overlap: chunk[1] should start before chunk[0] ends
    ok &= _check("consecutive chunks overlap",
                 chunks[1].char_start < chunks[0].char_end)

    short = "patient denies chest pain"
    sc = sliding_window_chunks(short, tok, max_tokens=400, stride=50)
    ok &= _check("short note stays a single chunk", len(sc) == 1)

    ok &= _check("empty note -> no chunks",
                 sliding_window_chunks("", tok) == [])
    return ok


def test_reconcile():
    print("reconcile:")
    ok = True
    # same label, overlapping spans, different score -> keep higher score, dedupe
    ents = [
        Entity("pneumonia", "DISEASE", 100, 109, 0.80, "affirmed"),
        Entity("pneumonia", "DISEASE", 100, 109, 0.95, "affirmed"),  # dup, better
        Entity("diabetes", "DISEASE", 200, 208, 0.90, "affirmed"),
    ]
    merged = ClinicalExtractor._reconcile(ents)
    ok &= _check("duplicate collapsed to one", len(merged) == 2)
    pn = [e for e in merged if e.text == "pneumonia"][0]
    ok &= _check("kept the higher-scoring duplicate", abs(pn.score - 0.95) < 1e-9)

    # overlapping spans but DIFFERENT labels -> both kept
    ents2 = [
        Entity("kidney", "ANATOMY", 10, 16, 0.7, "affirmed"),
        Entity("kidney injury", "DISEASE", 10, 23, 0.8, "affirmed"),
    ]
    ok &= _check("different labels are not merged",
                 len(ClinicalExtractor._reconcile(ents2)) == 2)
    return ok


def test_negex():
    print("negex:")
    nx = NegEx()
    ok = True

    def assertion(sentence, target):
        i = sentence.lower().index(target.lower())
        return nx.classify(sentence, i, i + len(target))

    ok &= _check("'no evidence of pneumonia' -> negated",
                 assertion("There was no evidence of pneumonia on x-ray",
                           "pneumonia") == Assertion.NEGATED)
    ok &= _check("'denies fever' -> negated",
                 assertion("He denies fever", "fever") == Assertion.NEGATED)
    ok &= _check("'rule out PE' -> possible",
                 assertion("asked to rule out pulmonary embolism today",
                           "pulmonary embolism") == Assertion.POSSIBLE)
    ok &= _check("'possible CHF' -> possible",
                 assertion("Possible congestive heart failure exacerbation",
                           "congestive heart failure") == Assertion.POSSIBLE)
    ok &= _check("'AKI is unlikely' -> negated",
                 assertion("Acute kidney injury is unlikely",
                           "kidney injury") == Assertion.NEGATED)
    ok &= _check("plain mention -> affirmed",
                 assertion("The troponin was elevated consistent with NSTEMI",
                           "troponin") in (Assertion.AFFIRMED, Assertion.POSSIBLE))
    ok &= _check("pseudo-negation 'no increase' does not negate target",
                 assertion("no increase in chest pain reported here",
                           "chest pain") != Assertion.NEGATED)
    ok &= _check("termination: 'no fever but has cough' -> cough affirmed",
                 assertion("no fever but has cough", "cough") == Assertion.AFFIRMED)
    return ok


def main():
    print("Running logic tests (no model download)\n")
    results = [test_chunking(), test_reconcile(), test_negex()]
    print()
    if all(results):
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
