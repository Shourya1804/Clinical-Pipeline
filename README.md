# Clinical NER pipeline (HuggingFace + ClinicalBERT)

A working extraction engine for long clinical notes. It does the four things a
base model does **not** do for you: chunk past the 512-token limit, stitch
sub-words, reconcile entities across overlapping windows, and decide whether
each finding is actually present (negation/uncertainty).

```
clinical_extractor/
  chunking.py   sliding-window chunker (keeps original char offsets)
  pipeline.py   ClinicalExtractor — runs HF NER per chunk, reconciles, asserts
  negation.py   NegEx-style affirmed / negated / possible classifier
  cli.py        end-to-end command-line runner
requirements.txt
sample_note.txt  a realistic discharge summary to try
test_logic.py    18 unit tests that run with NO model download
```

## Quick start

```bash
pip install -r requirements.txt

# pretty table
python -m clinical_extractor.cli --input sample_note.txt

# JSON, drop low-confidence hits, use a clinical i2b2 model
python -m clinical_extractor.cli --input sample_note.txt --json out.json \
    --min-score 0.5 --model samrawal/bert-base-uncased_clinical-ner
```

Each entity comes back as: text, label, original-note span, score, and an
**assertion** (`affirmed` / `negated` / `possible`).

```python
from clinical_extractor import ClinicalExtractor
ext = ClinicalExtractor()                 # CPU by default; auto-GPU if present
for e in ext.extract(open("sample_note.txt").read()):
    print(e.text, e.label, e.assertion, e.score)
```

## Run the logic tests (no download, seconds)

```bash
python test_logic.py        # 18 checks: chunking, reconciliation, NegEx
```

## How it maps to the guide — and where the guide is wrong

The guide's architecture is right. A few specifics are worth correcting before
you build on them:

**Base `Bio_ClinicalBERT` is masked-LM only — correct, so we don't use it.**
You need a token-classification *head*. The default here is
`d4data/biomedical-ner-all`; for i2b2-style problem/test/treatment labels swap
in `samrawal/bert-base-uncased_clinical-ner` via `--model`. Point `--model` at
any n2c2/i2b2 fine-tune you prefer.

**You do NOT need to hand-stitch sub-words.** The guide says to write code to
glue `Hydro ##chloro ##thia ##zide` back together. The HF pipeline does this
for you through `aggregation_strategy` (we use `"max"`), returning whole-word
groups with correct offsets. Hand-stitching is reinventing a solved problem.

**The 512-token wall and sliding window are real and correctly described.**
Implemented in `chunking.py`. One improvement: we chunk on tokens but carry the
**original character offset** of each window, so reconciliation is simple
interval math instead of fuzzy text re-matching. Overlap default is 50 tokens.

**"Silently lose 70%" is illustrative, not a constant.** How much you lose
depends on note length; the point — that truncation is silent — stands.

**NegEx integration is correct and necessary.** "Rule out pneumonia" must not
become an affirmed diagnosis. One clinical nuance the guide glosses over:
`rule out X` means X is being *considered* (we mark it `possible`), whereas
`X ruled out` / `X is unlikely` means *negated*. Both are handled.

**Hardware advice is sound.** Batch size 1 on a 16 GB CPU laptop (the runner
processes one chunk at a time). ONNX Runtime genuinely speeds up CPU inference —
see the commented optional deps in `requirements.txt`; convert with
`optimum-cli export onnx --model <id> onnx_dir/` and load via
`optimum.onnxruntime.ORTModelForTokenClassification`. Google Colab's free T4 is
the right escape hatch for development.

## Important caveat

These models are research tools. Public clinical-NER checkpoints are **not**
validated for clinical decision-making, and entity linking to SNOMED/RxNorm
(the guide's next step) needs a dedicated terminology service (e.g. UMLS
MetaMap / QuickUMLS / a SNOMED API) — string-matching model output to a
terminology is unsafe on its own. Treat output as a draft for human review.
