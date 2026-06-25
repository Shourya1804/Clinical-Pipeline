"""
Logic tests that DO NOT require torch/transformers or any model download.

They validate the parts of the pipeline that are pure Python:
  - sliding-window chunking (via a tiny stub tokenizer)
  - cross-chunk entity reconciliation
  - NegEx assertion classification
  - terminology linking (RxNorm / UMLS) with an injected fetch

Run:  python test_logic.py
"""

import re
import sys

from clinical_extractor.chunking import sliding_window_chunks
from clinical_extractor.negation import NegEx, Assertion
from clinical_extractor.pipeline import ClinicalExtractor, Entity
from clinical_extractor.linking import TerminologyLinker


class StubTokenizer:
    """Whitespace tokenizer that returns offset mappings like a fast tokenizer."""
    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _check(name, cond):
    print("  [{}] {}".format("PASS" if cond else "FAIL", name))
    return bool(cond)


def test_chunking():
    print("chunking:")
    tok = StubTokenizer()
    text = " ".join("w{}".format(i) for i in range(1000))  # 1000 tokens

    ok = True
    chunks = sliding_window_chunks(text, tok, max_tokens=400, stride=50)
    ok &= _check("long note is split into multiple chunks", len(chunks) > 1)
    ok &= _check("expected chunk count (1000 tok, step 350)", len(chunks) == 3)
    ok &= _check("first chunk starts at offset 0", chunks[0].char_start == 0)
    ok &= _check("last chunk reaches end of note", chunks[-1].char_end == len(text))
    ok &= _check("consecutive chunks overlap",
                 chunks[1].char_start < chunks[0].char_end)

    short = "patient denies chest pain"
    sc = sliding_window_chunks(short, tok, max_tokens=400, stride=50)
    ok &= _check("short note stays a single chunk", len(sc) == 1)
    ok &= _check("empty note -> no chunks", sliding_window_chunks("", tok) == [])
    return ok


def test_reconcile():
    print("reconcile:")
    ok = True
    ents = [
        Entity("pneumonia", "DISEASE", 100, 109, 0.80, "affirmed"),
        Entity("pneumonia", "DISEASE", 100, 109, 0.95, "affirmed"),
        Entity("diabetes", "DISEASE", 200, 208, 0.90, "affirmed"),
    ]
    merged = ClinicalExtractor._reconcile(ents)
    ok &= _check("duplicate collapsed to one", len(merged) == 2)
    pn = [e for e in merged if e.text == "pneumonia"][0]
    ok &= _check("kept the higher-scoring duplicate", abs(pn.score - 0.95) < 1e-9)

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
    ok &= _check("plain mention -> affirmed/possible",
                 assertion("The troponin was elevated consistent with NSTEMI",
                           "troponin") in (Assertion.AFFIRMED, Assertion.POSSIBLE))
    ok &= _check("pseudo-negation 'no increase' does not negate target",
                 assertion("no increase in chest pain reported here",
                           "chest pain") != Assertion.NEGATED)
    ok &= _check("termination: 'no fever but has cough' -> cough affirmed",
                 assertion("no fever but has cough", "cough") == Assertion.AFFIRMED)
    return ok


def _fake_fetch_factory(counter):
    """Stub fetch_json that mimics RxNav/UMLS responses, no network."""
    def fake_fetch(url):
        counter[0] += 1
        if "approximateTerm" in url:
            return {"approximateGroup": {"candidate": [
                {"rxcui": "6809", "score": "100"}]}}
        if "/properties.json" in url:
            return {"properties": {"name": "Metformin", "tty": "IN"}}
        if "/search/current" in url:
            return {"result": {"results": [
                {"ui": "233604007", "name": "Pneumonia",
                 "rootSource": "SNOMEDCT_US"}]}}
        return None
    return fake_fetch


def test_linking():
    print("linking:")
    ok = True

    no_key = TerminologyLinker(use_rxnorm=True, umls_api_key=None, cache_path=None)
    ok &= _check("medication label -> RXNORM",
                 no_key.backend_for("Medication") == "RXNORM")
    ok &= _check("problem label with no UMLS key -> None",
                 no_key.backend_for("Disease_disorder") is None)

    with_key = TerminologyLinker(umls_api_key="DUMMY", cache_path=None)
    ok &= _check("problem label with key -> SNOMEDCT_US",
                 with_key.backend_for("Disease_disorder") == "SNOMEDCT_US")

    counter = [0]
    rx = TerminologyLinker(umls_api_key=None, cache_path=None,
                           fetch_json=_fake_fetch_factory(counter))
    hit = rx.lookup("metformin", "RXNORM")
    ok &= _check("RxNorm returns RXCUI", bool(hit) and hit["code"] == "6809")
    ok &= _check("RxNorm returns concept name",
                 bool(hit) and hit["code_name"] == "Metformin")

    calls_before = counter[0]
    rx.lookup("metformin", "RXNORM")
    ok &= _check("cache prevents repeat network calls", counter[0] == calls_before)

    umls = TerminologyLinker(umls_api_key="DUMMY", cache_path=None,
                             fetch_json=_fake_fetch_factory([0]))
    sn = umls.lookup("pneumonia", "SNOMEDCT_US")
    ok &= _check("SNOMED returns code", bool(sn) and sn["code"] == "233604007")

    ents = [Entity("metformin", "Medication", 0, 9, 0.9, "affirmed")]
    linker = TerminologyLinker(umls_api_key=None, cache_path=None,
                               fetch_json=_fake_fetch_factory([0]))
    linker.link(ents)
    ok &= _check("link() attaches code to entity", ents[0].code == "6809")
    ok &= _check("link() sets code system", ents[0].code_system == "RXNORM")

    dead = TerminologyLinker(umls_api_key=None, cache_path=None,
                             fetch_json=lambda u: None)
    ents2 = [Entity("aspirin", "Medication", 0, 7, 0.9, "affirmed")]
    dead.link(ents2)
    ok &= _check("dead network -> entity has no code (no crash)",
                 ents2[0].code is None)
    return ok




def test_deid():
    print("deid:")
    from clinical_extractor.deid import Deidentifier
    ok = True

    # regex-only (no model) backstops
    d = Deidentifier(use_model=False)

    clean, reds = d.deidentify("SSN 123-45-6789 on file")
    ok &= _check("SSN redacted", "[SSN]" in clean and "123-45-6789" not in clean)

    clean, _ = d.deidentify("email john.doe@hospital.org please")
    ok &= _check("email redacted", "[EMAIL]" in clean and "john.doe" not in clean)

    clean, _ = d.deidentify("call 415-555-0182 today")
    ok &= _check("phone redacted", "[PHONE]" in clean and "0182" not in clean)

    clean, _ = d.deidentify("seen on 03/14/2025 in clinic")
    ok &= _check("numeric date redacted", "[DATE]" in clean and "03/14/2025" not in clean)

    clean, _ = d.deidentify("admitted January 7, 2024 overnight")
    ok &= _check("month-name date redacted", "[DATE]" in clean and "2024" not in clean)

    clean, _ = d.deidentify("MRN: 00984321 active")
    ok &= _check("MRN redacted", "[ID]" in clean and "00984321" not in clean)

    clean, _ = d.deidentify("a 94-year-old male")
    ok &= _check("age over 89 redacted", "[AGE]" in clean and "94" not in clean)

    clean, _ = d.deidentify("a 62-year-old male")
    ok &= _check("age under 90 NOT redacted", "62" in clean)

    # model pass via injected ner_fn (no download): redact a name
    def fake_ner(text):
        i = text.index("Jane Roe")
        return [{"entity_group": "PATIENT", "start": i, "end": i + len("Jane Roe")}]
    dm = Deidentifier(use_model=True, ner_fn=fake_ner)
    clean, _ = dm.deidentify("Patient Jane Roe presents with cough")
    ok &= _check("model name redacted -> [NAME]",
                 "[NAME]" in clean and "Jane Roe" not in clean)

    # overlap merge: regex date inside a model DATE span shouldn't double-wrap
    def fake_date(text):
        i = text.index("03/14/2025")
        return [{"entity_group": "DATE", "start": i, "end": i + 10}]
    dd = Deidentifier(use_model=True, ner_fn=fake_date)
    clean, reds = dd.deidentify("seen on 03/14/2025 ok")
    ok &= _check("overlapping spans merged to one redaction", clean.count("[DATE]") == 1)

    # failure-soft: model raising shouldn't crash; regex still works
    def boom(text):
        raise RuntimeError("model down")
    db = Deidentifier(use_model=True, ner_fn=boom)
    try:
        clean, _ = db.deidentify("SSN 123-45-6789")
        crashed = False
    except Exception:
        crashed = True
    ok &= _check("model error is failure-soft (regex still applies)",
                 (not crashed) and "[SSN]" in clean)

    # keep_tags: AGE and LOCATION left in place; other PHI still redacted
    keep = Deidentifier(use_model=False, keep_tags={"AGE", "LOCATION"})
    clean, _ = keep.deidentify("a 94-year-old male, SSN 123-45-6789, TX 75001")
    ok &= _check("kept AGE stays in text", "94" in clean and "[AGE]" not in clean)
    ok &= _check("kept LOCATION stays in text",
                 "75001" in clean and "[LOCATION]" not in clean)
    ok &= _check("non-kept SSN still redacted",
                 "[SSN]" in clean and "123-45-6789" not in clean)

    # keep_tags also filters the model layer (NAME kept here)
    def fake_name(text):
        i = text.index("Jane Roe")
        return [{"entity_group": "PATIENT", "start": i, "end": i + len("Jane Roe")}]
    keepname = Deidentifier(use_model=True, ner_fn=fake_name, keep_tags={"NAME"})
    clean, _ = keepname.deidentify("Patient Jane Roe here")
    ok &= _check("keep_tags filters model layer (NAME kept)",
                 "Jane Roe" in clean and "[NAME]" not in clean)
    return ok


def main():
    print("Running logic tests (no model download)\n")
    results = [test_chunking(), test_reconcile(), test_negex(),
               test_linking(), test_deid()]
    print()
    if all(results):
        print("ALL TESTS PASSED")
        return 0
    print("SOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
