"""
Command-line runner.

Examples
--------
    # extract from a file, pretty table to stdout
    python -m clinical_extractor.cli --input note.txt

    # JSON out, drop low-confidence hits, use a clinical i2b2 model
    python -m clinical_extractor.cli --input note.txt --json out.json \
        --min-score 0.5 --model samrawal/bert-base-uncased_clinical-ner
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import ClinicalExtractor, DEFAULT_MODEL


def main(argv=None):
    p = argparse.ArgumentParser(description="ClinicalBERT medical-entity extractor")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="path to a clinical note (.txt)")
    src.add_argument("--text", help="raw note text passed on the command line")

    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--no-negation", action="store_true",
                   help="skip the NegEx assertion layer")
    p.add_argument("--json", dest="json_out", help="write results to this JSON file")
    args = p.parse_args(argv)

    note = open(args.input, encoding="utf-8").read() if args.input else args.text

    extractor = ClinicalExtractor(
        model_name=args.model,
        max_tokens=args.max_tokens,
        stride=args.stride,
        min_score=args.min_score,
        run_negation=not args.no_negation,
    )
    entities = extractor.extract(note)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([e.as_dict() for e in entities], f, indent=2)
        print(f"Wrote {len(entities)} entities -> {args.json_out}")
    else:
        _print_table(entities)

    return 0


def _print_table(entities):
    if not entities:
        print("No entities found.")
        return
    w = max(len(e.text) for e in entities)
    w = min(max(w, 12), 40)
    print(f"{'ENTITY':<{w}}  {'LABEL':<14}  {'ASSERTION':<9}  SCORE  SPAN")
    print("-" * (w + 45))
    for e in entities:
        txt = (e.text[:w - 1] + "…") if len(e.text) > w else e.text
        print(f"{txt:<{w}}  {e.label:<14}  {e.assertion:<9}  "
              f"{e.score:0.2f}   [{e.start}:{e.end}]")


if __name__ == "__main__":
    sys.exit(main())
