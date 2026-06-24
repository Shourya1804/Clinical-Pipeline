# Clinical NER pipeline (HuggingFace + ClinicalBERT)

Extracts medical entities from clinical notes, decides whether each finding is
actually present (negation/uncertainty), and optionally links them to standard
codes (RxNorm for drugs, SNOMED CT for problems). Runs locally with a browser
UI or a batch command line.

> **Research/engineering tool only. NOT validated for clinical use.** Do not use
> it to make patient-care decisions. Notes are processed on your machine and are
> never uploaded. If you enable code linking, only the short matched terms (e.g.
> "metformin") — never the full note — are sent to the public NLM APIs. With PHI,
> check your institution's policy first, or keep linking off to stay fully local.

## What's in the box

```
clinical_extractor/
  chunking.py   sliding-window chunker (keeps original char offsets)
  pipeline.py   ClinicalExtractor — HF NER per chunk, reconcile, assert
  negation.py   NegEx-style affirmed / negated / possible classifier
  linking.py    RxNorm + UMLS/SNOMED code linking (optional, cached)
  deid.py       best-effort PHI redaction (regex + de-id model)
  cli.py        single-note and batch runner
app.py          local web app (paste a note -> highlighted entities + table)
sample_note.txt example discharge summary
test_logic.py   40 unit tests that run with NO model download
requirements.txt
.env.example    where to put a UMLS API key (optional)
```

## Setup (once)

```bash
pip install -r requirements.txt
```

First extraction downloads the NER model (~400 MB); later runs are fast.

## Option A — the web app (easiest)

```bash
python app.py
# open http://127.0.0.1:5000 in your browser
```

Paste a note, click **Extract entities**. Findings are color-coded
(green = affirmed, red = negated, amber = possible) and listed in a table with
any matched codes. Tick "link codes" to enable RxNorm/SNOMED lookup.

## Option B — the command line

```bash
# one note -> table
python -m clinical_extractor.cli --input sample_note.txt

# one note with code linking, save JSON
python -m clinical_extractor.cli --input sample_note.txt --link --json out.json

# BATCH a whole folder of .txt notes -> one CSV row per entity
python -m clinical_extractor.cli --input-dir notes/ --csv results.csv --link
```

The CSV has one row per entity: source file, text, label, assertion, score,
character span, and code/code_system/code_name when linking is on.

## Option C — from Python

```python
from clinical_extractor import ClinicalExtractor, TerminologyLinker

ext = ClinicalExtractor()                 # CPU by default; auto-GPU if present
ents = ext.extract(open("sample_note.txt", encoding="utf-8").read())
TerminologyLinker().link(ents)            # optional; RxNorm needs no key
for e in ents:
    print(e.text, e.label, e.assertion, e.code_system, e.code)
```

## Code linking: RxNorm and SNOMED

- **RxNorm** (medications) works with **no key** via the public RxNav API.
- **SNOMED CT** (problems/symptoms/procedures) needs a **free UMLS API key**.
  Sign up at https://uts.nlm.nih.gov/uts/signup-login, copy your key from the
  UTS "My Profile" page, then either:
  - set an environment variable: `set UMLS_API_KEY=your-key` (Windows) /
    `export UMLS_API_KEY=your-key` (mac/Linux), or
  - pass `--umls-key your-key` on the CLI.

Without a key you still get RxNorm linking; SNOMED is simply skipped. Lookups
are cached in `.cache/linking.json` so repeats and batch runs stay fast.

## De-identification (PHI redaction)

> **This does NOT make you legally compliant by itself.** Automated
> de-identification always misses some PHI and over-redacts other text. Under
> HIPAA, de-identified status requires Safe Harbor (all 18 identifier types
> removed, no residual re-identification risk) or Expert Determination by a
> qualified statistician. Treat this as a strong first pass that a human must
> review. You remain responsible for the data.

Two layers run together:

- **Regex backstops** for structured identifiers: SSN, email, URL, IP,
  phone/fax, medical record / account numbers, dates, and ages over 89.
- **A contextual model** (`obi/deid_roberta_i2b2`, trained on the i2b2 2014
  de-id challenge) for names, locations, hospitals, and IDs the regex misses.

PHI is replaced with typed tags like `[NAME]`, `[DATE]`, `[ID]`. The original
values are kept only in memory (the returned redaction list) and are never
written into the redacted text or to disk.

```bash
# de-identify, then extract (web app: tick "de-identify first")
python -m clinical_extractor.cli --input sample_note.txt --deid

# just redact a folder of notes into a new folder, no extraction
python -m clinical_extractor.cli --input-dir notes/ --deid-only --deid-out redacted/

# regex-only (skip the de-id model download)
python -m clinical_extractor.cli --input sample_note.txt --deid --no-deid-model
```

```python
from clinical_extractor import Deidentifier
clean, redactions = Deidentifier().deidentify(open("sample_note.txt").read())
print(clean)                      # PHI replaced with [TAG]
print(len(redactions), "items redacted")
```

## Run the tests (no download, seconds)

```bash
python test_logic.py        # 40 checks: chunking, reconcile, NegEx, linking, de-id
```

## How it maps to the original guide — and where the guide is wrong

The guide's architecture is right; a few specifics are not:

**Base `Bio_ClinicalBERT` is masked-LM only** — correct, so we don't use it. We
load a token-classification fine-tune (`d4data/biomedical-ner-all` by default;
swap with `--model` for an n2c2/i2b2 model such as
`samrawal/bert-base-uncased_clinical-ner`).

**You do NOT need to hand-stitch sub-words.** The HF pipeline's
`aggregation_strategy="max"` rejoins `Hydro ##chloro ##thia ##zide` with correct
offsets. Hand-stitching is reinventing a solved problem.

**The 512-token wall and sliding window are real.** Implemented in
`chunking.py`. We chunk on tokens but carry each window's original character
offset, so reconciliation is simple interval math. The "lose 70%" figure is
illustrative — the amount depends on note length; the silent-truncation risk is
the real point.

**NegEx is necessary.** "Rule out X" must not become an affirmed diagnosis. One
nuance the guide glosses over: `rule out X` means X is being *considered* (we
mark `possible`); `X ruled out` / `is unlikely` means *negated*.

**Hardware advice is sound.** Batch size 1 on a 16 GB CPU laptop; ONNX Runtime
speeds up CPU inference (see commented optional deps in `requirements.txt`);
Google Colab's free T4 is the right escape hatch for development.

## Limitations

Public clinical-NER checkpoints are not validated for clinical decision-making.
String-matching model output to a terminology can mislink — treat codes as
suggestions for human review, not ground truth. The included de-identifier is best-effort and is NOT a compliance guarantee.
This tool has no audit trail or access control; add those, and have a human
review redactions, before anything resembling production use.
