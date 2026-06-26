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
  ingest.py     multi-format streaming loader (scales to millions)
  gazetteer.py  custom term/abbreviation dictionary (no training)
  cli.py        single-note and streamed batch runner
dictionary.csv  starter custom-term dictionary (edit to add your own)
app.py          local web app (paste a note -> highlighted entities + table)
sample_note.txt example discharge summary
test_logic.py   62 unit tests that run with NO model download
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

You can either **paste one note** or **upload one or more files** (any
supported text format). For a single note the findings are highlighted in place
(green = affirmed, red = negated, amber = possible). For uploaded files you get a
per-file summary, a combined results table, and a **Download CSV** button. Tick
"de-identify first" and/or "link codes" as needed. For very large jobs (tens of
thousands of files) use the streamed command line instead.

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

**Keeping some categories.** You can tell the de-identifier to leave certain
categories in place with `--keep` (CLI) or `keep_tags=` (Python). GENDER is
never targeted, so it always stays. The web app keeps AGE and LOCATION by
default. Keeping AGE (especially > 89) and LOCATION REDUCES privacy protection
and means the output is no longer HIPAA Safe Harbor de-identified.

```bash
# keep age and location, redact everything else
python -m clinical_extractor.cli --input sample_note.txt --deid --keep AGE,LOCATION
```

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

## Large batches & file formats

The batch runner is **streamed**: it reads and processes one document at a time
and writes results incrementally, so memory stays flat whether you have 100
files or 1,000,000. Use `--csv` or `--jsonl` for output (a single `--json`
array loads in memory and is only for small runs).

Supported input formats (run `--list-formats`): `.txt .text .md .markdown .log
.csv .tsv .json .ndjson .jsonl .rtf .htm .html` with no extra dependency, and
`.docx` if you `pip install python-docx`. HTML/RTF are stripped to plain text.
PDFs are images of text — run OCR first; they are not accepted directly.

```bash
# stream a whole tree of mixed formats to CSV, resumable
python -m clinical_extractor.cli --input-dir notes/ --csv results.csv --deid --resume
```

`--resume` writes a `results.csv.done` ledger and skips files already in it, so
an interrupted million-file run continues where it stopped. Unreadable files are
logged and skipped, never fatal.

**A note on "ready for a million records":** the pipeline is now memory-safe and
resumable for that scale, but on a CPU it would still take *days* — the model
runs one note at a time. For real million-scale throughput you want a GPU (or
several) and parallel workers; see the cloud architecture options discussed in
the project notes. The code is the same; only where you run it changes.

## Teaching it new terms (no training, no reviewers)

The model is frozen - running notes through it does NOT train it. To make it
recognise YOUR terms, abbreviations, and newly-released drugs, add them to a
**custom dictionary** (`dictionary.csv`). The pipeline matches those terms
directly, in addition to the model, and a dictionary match overrides the model
on any overlap. This is a lookup list you control: instant, transparent, and
auditable - the right pattern for a regulated setting, and it needs no labeled
data or reviewers.

```
term,label,expansion,code,code_system
SOB,Sign_symptom,shortness of breath,,
HTN,Disease_disorder,hypertension,,
HCTZ,Medication,hydrochlorothiazide,5487,RXNORM
```

Only `term` is required. `expansion` is shown as the concept name (great for
abbreviations). Matching is whole-word and case-insensitive; multi-word terms
are fine. Add a line, save, re-run - that is the whole "learning" loop.

```bash
# CLI: use a dictionary
python -m clinical_extractor.cli --input note.txt --dictionary dictionary.csv

# Web app: it auto-loads dictionary.csv sitting next to app.py
```

**What this does and does not do.** It grows the system's *vocabulary*. It does
NOT improve the model's judgement on ambiguous text - that still needs labeled
corrections and fine-tuning (see the note in "Limitations"). For new drugs/codes
this dictionary plus the RxNorm/SNOMED linking covers most of what you need.

## Run the tests (no download, seconds)

```bash
python test_logic.py        # 62 checks: chunking, reconcile, NegEx, linking, de-id, ingest, gazetteer
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
