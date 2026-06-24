"""
Command-line runner: single note or a whole folder (batch).

Examples
--------
    # one note -> pretty table
    python -m clinical_extractor.cli --input sample_note.txt

    # one note, with RxNorm + SNOMED linking, JSON out
    python -m clinical_extractor.cli --input sample_note.txt --link --json out.json

    # BATCH: every .txt in notes/ -> one CSV row per entity
    python -m clinical_extractor.cli --input-dir notes/ --csv results.csv --link

Linking needs network. RxNorm works with no key; SNOMED needs a UMLS key,
passed via --umls-key or the UMLS_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

from .pipeline import ClinicalExtractor, DEFAULT_MODEL


def build_parser():
    p = argparse.ArgumentParser(description="ClinicalBERT medical-entity extractor")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="path to a single clinical note (.txt)")
    src.add_argument("--text", help="raw note text on the command line")
    src.add_argument("--input-dir", help="folder of .txt notes to batch-process")

    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--no-negation", action="store_true", help="skip NegEx")

    p.add_argument("--link", action="store_true",
                   help="link entities to RxNorm / SNOMED codes (uses network)")
    p.add_argument("--umls-key", default=os.environ.get("UMLS_API_KEY"),
                   help="UMLS API key for SNOMED (or set UMLS_API_KEY)")

    p.add_argument("--json", dest="json_out", help="write results to JSON")
    p.add_argument("--csv", dest="csv_out", help="write results to CSV")
    return p


def _load_docs(args):
    docs = []
    if args.input_dir:
        paths = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
        if not paths:
            print("No .txt files found in " + args.input_dir, file=sys.stderr)
            return None
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                docs.append((os.path.basename(path), fh.read()))
    elif args.input:
        with open(args.input, encoding="utf-8") as fh:
            docs.append((os.path.basename(args.input), fh.read()))
    else:
        docs.append(("<text>", args.text))
    return docs


def main(argv=None):
    args = build_parser().parse_args(argv)

    extractor = ClinicalExtractor(
        model_name=args.model,
        max_tokens=args.max_tokens,
        stride=args.stride,
        min_score=args.min_score,
        run_negation=not args.no_negation,
    )

    linker = None
    if args.link:
        from .linking import TerminologyLinker
        linker = TerminologyLinker(umls_api_key=args.umls_key)

    docs = _load_docs(args)
    if docs is None:
        return 1

    all_rows = []
    for name, note in docs:
        entities = extractor.extract(note)
        if linker:
            linker.link(entities)
        for e in entities:
            all_rows.append({"source": name, **e.as_dict()})

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        print("Wrote " + str(len(all_rows)) + " entities -> " + args.json_out)
    if args.csv_out:
        _write_csv(all_rows, args.csv_out)
        print("Wrote " + str(len(all_rows)) + " entities -> " + args.csv_out)
    if not args.json_out and not args.csv_out:
        _print_table(all_rows, multi=bool(args.input_dir))

    return 0


def _write_csv(rows, path):
    fields = ["source", "text", "label", "assertion", "score",
              "start", "end", "code", "code_system", "code_name", "link_score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _print_table(rows, multi=False):
    if not rows:
        print("No entities found.")
        return
    w = min(max((len(r["text"]) for r in rows), default=12), 32)
    hdr = "{:<{w}}  {:<16}  {:<8}  SCORE  CODE".format("ENTITY", "LABEL", "ASSERT", w=w)
    if multi:
        hdr = "{:<18}  ".format("SOURCE") + hdr
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        txt = (r["text"][:w - 1] + "…") if len(r["text"]) > w else r["text"]
        code = (str(r.get("code_system") or "") + ":" + str(r.get("code") or "")).strip(":")
        line = "{:<{w}}  {:<16}  {:<8}  {:0.2f}   {}".format(
            txt, r["label"], r["assertion"], r["score"], code, w=w)
        if multi:
            line = "{:<18}  ".format(r["source"]) + line
        print(line)


if __name__ == "__main__":
    sys.exit(main())
