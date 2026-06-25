"""
Command-line runner: single note, or a whole folder, STREAMED for scale.

Built to survive very large batches (hundreds of thousands to millions of
files): documents are read and processed ONE AT A TIME and results are written
incrementally, so memory stays flat. A --resume checkpoint lets an interrupted
run pick up where it left off.

Examples
--------
    # one note (any text format: .txt .md .csv .json .html .rtf .docx ...)
    python -m clinical_extractor.cli --input note.txt

    # de-identify first, then extract, with code linking, JSON out
    python -m clinical_extractor.cli --input note.txt --deid --link --json out.json

    # BIG BATCH: every supported file under notes/ -> streamed CSV, resumable
    python -m clinical_extractor.cli --input-dir notes/ --csv results.csv --deid --resume

    # one JSON object per line (best structured format for huge runs)
    python -m clinical_extractor.cli --input-dir notes/ --jsonl results.jsonl --resume

Linking needs network. RxNorm works with no key; SNOMED needs a UMLS key
(--umls-key or UMLS_API_KEY). De-identification is best-effort, not a
compliance guarantee (see README).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from .pipeline import ClinicalExtractor, DEFAULT_MODEL
from .ingest import iter_documents, extract_text, supported_extensions

CSV_FIELDS = ["source", "text", "label", "assertion", "score",
              "start", "end", "code", "code_system", "code_name", "link_score"]


def build_parser():
    p = argparse.ArgumentParser(description="ClinicalBERT medical-entity extractor")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", help="path to a single note (any supported text format)")
    src.add_argument("--text", help="raw note text on the command line")
    src.add_argument("--input-dir", help="folder of notes to batch-process (streamed)")

    p.add_argument("--no-recursive", action="store_true",
                   help="with --input-dir, do NOT descend into subfolders")
    p.add_argument("--ext", default="",
                   help="comma-separated extensions to include (default: all supported)")

    p.add_argument("--dictionary", default="",
                   help="path to a custom term/abbreviation CSV (gazetteer)")

    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--no-negation", action="store_true", help="skip NegEx")

    p.add_argument("--deid", action="store_true",
                   help="redact PHI before extraction (best-effort, not a guarantee)")
    p.add_argument("--deid-only", action="store_true",
                   help="just de-identify; write redacted notes, no extraction")
    p.add_argument("--deid-out", help="folder to write redacted .txt notes into")
    p.add_argument("--no-deid-model", action="store_true",
                   help="de-id with regex only (skip the de-id model download)")
    p.add_argument("--keep", default="",
                   help="comma-separated PHI categories to LEAVE in (e.g. AGE,LOCATION)")

    p.add_argument("--link", action="store_true",
                   help="link entities to RxNorm / SNOMED codes (uses network)")
    p.add_argument("--umls-key", default=os.environ.get("UMLS_API_KEY"),
                   help="UMLS API key for SNOMED (or set UMLS_API_KEY)")

    p.add_argument("--csv", dest="csv_out", help="stream results to this CSV")
    p.add_argument("--jsonl", dest="jsonl_out", help="stream one JSON object per line")
    p.add_argument("--json", dest="json_out",
                   help="write a single JSON array (loads in memory; small runs only)")
    p.add_argument("--resume", action="store_true",
                   help="skip files already recorded in <output>.done (resumable)")
    p.add_argument("--progress-every", type=int, default=100,
                   help="print a progress line every N documents")
    p.add_argument("--list-formats", action="store_true",
                   help="print supported input extensions and exit")
    return p


def _sources(args):
    """Yield (name, text) lazily from whatever input was given."""
    if args.input_dir:
        exts = [e.strip() for e in args.ext.split(",") if e.strip()] or None
        def warn(path, exc):
            print("  ! skipped " + path + ": " + str(exc), file=sys.stderr)
        yield from iter_documents(args.input_dir, recursive=not args.no_recursive,
                                  exts=exts, on_error=warn)
    elif args.input:
        yield (os.path.basename(args.input), extract_text(args.input))
    else:
        yield ("<text>", args.text)


def _load_done(ledger_path):
    done = set()
    if ledger_path and os.path.exists(ledger_path):
        with open(ledger_path, encoding="utf-8") as f:
            done = set(line.rstrip("\n") for line in f)
    return done


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_formats:
        print("Supported input formats: " + ", ".join(supported_extensions()))
        return 0

    if not (args.input or args.text or args.input_dir):
        print("error: one of --input, --text, or --input-dir is required",
              file=sys.stderr)
        return 2

    # Build heavy objects once.
    deider = None
    if args.deid or args.deid_only:
        from .deid import Deidentifier
        keep = [t.strip() for t in args.keep.split(",") if t.strip()]
        deider = Deidentifier(use_model=not args.no_deid_model, keep_tags=keep)

    gazetteer = None
    if args.dictionary:
        from .gazetteer import Gazetteer
        gazetteer = Gazetteer.from_file(args.dictionary)
        print("Loaded {} custom dictionary terms".format(len(gazetteer)),
              file=sys.stderr)

    extractor = None
    if not args.deid_only:
        extractor = ClinicalExtractor(
            model_name=args.model, max_tokens=args.max_tokens,
            stride=args.stride, min_score=args.min_score,
            run_negation=not args.no_negation, gazetteer=gazetteer)

    linker = None
    if args.link:
        from .linking import TerminologyLinker
        linker = TerminologyLinker(umls_api_key=args.umls_key)

    # Resume ledger lives next to the chosen output.
    out_path = args.csv_out or args.jsonl_out or args.json_out
    ledger = (out_path + ".done") if (out_path and args.resume) else None
    done = _load_done(ledger) if args.resume else set()

    # Streaming output writers.
    writer, csv_fh, jsonl_fh = None, None, None
    json_rows = [] if args.json_out else None
    if args.csv_out:
        new = not (args.resume and os.path.exists(args.csv_out))
        csv_fh = open(args.csv_out, "a" if not new else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
    if args.jsonl_out:
        new = not (args.resume and os.path.exists(args.jsonl_out))
        jsonl_fh = open(args.jsonl_out, "a" if not new else "w", encoding="utf-8")

    ledger_fh = open(ledger, "a", encoding="utf-8") if ledger else None

    n_docs, n_ents, skipped, t0 = 0, 0, 0, time.time()
    table_rows = []  # only used when no file output (small/interactive)

    for name, note in _sources(args):
        if args.resume and name in done:
            skipped += 1
            continue

        text = note
        if deider:
            clean, reds = deider.deidentify(note)
            text = clean
            if args.deid_out:
                os.makedirs(args.deid_out, exist_ok=True)
                with open(os.path.join(args.deid_out, name), "w", encoding="utf-8") as f:
                    f.write(clean)

        if extractor is not None:
            entities = extractor.extract(text)
            if linker:
                linker.link(entities)
            for e in entities:
                row = {"source": name, **e.as_dict()}
                n_ents += 1
                if writer:
                    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
                elif jsonl_fh:
                    jsonl_fh.write(json.dumps(row) + "\n")
                elif json_rows is not None:
                    json_rows.append(row)
                else:
                    table_rows.append(row)

        if ledger_fh:
            ledger_fh.write(name + "\n")
            ledger_fh.flush()
        n_docs += 1
        if n_docs % args.progress_every == 0:
            rate = n_docs / max(time.time() - t0, 1e-6)
            print("  processed {} docs, {} entities ({:.1f} docs/s)".format(
                n_docs, n_ents, rate), file=sys.stderr)

    for fh in (csv_fh, jsonl_fh, ledger_fh):
        if fh:
            fh.close()
    if json_rows is not None:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(json_rows, f, indent=2)

    # Final report.
    if out_path:
        msg = "Done: {} docs, {} entities -> {}".format(n_docs, n_ents, out_path)
        if args.resume and skipped:
            msg += " (skipped {} already done)".format(skipped)
        print(msg)
    elif args.deid_only:
        print("Done: de-identified {} docs".format(n_docs))
    else:
        _print_table(table_rows, multi=bool(args.input_dir))
    return 0


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
