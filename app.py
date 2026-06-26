"""
Local web app for the clinical extractor.

Two ways to use it, both fully local (nothing is uploaded to the internet):
  * Paste a single note in the text box, or
  * Choose multiple text files to process as a batch.

Everything runs on YOUR machine. "De-identify first" redacts PHI before
extraction (best-effort, not a compliance guarantee). "Link codes" sends only
the short matched terms (not the note) to the public RxNorm/UMLS APIs.

For very large jobs (tens of thousands of files+) use the streamed command line
(`python -m clinical_extractor.cli --input-dir ...`); the browser is best for
interactive review of up to a few hundred files at a time.

Run:
    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:5000

Models (~400 MB extraction + de-id) download on first use only.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import os
import tempfile

from flask import Flask, request

from clinical_extractor import (ClinicalExtractor, TerminologyLinker,
                                Deidentifier, Gazetteer, extract_text,
                                supported_extensions)

app = Flask(__name__)

_extractor = None
_deider = None

_DICT_PATH = os.path.join(os.path.dirname(__file__), "dictionary.csv")
KEEP_TAGS = {"AGE", "LOCATION"}   # gender is never targeted; these are kept too


def get_extractor():
    global _extractor
    if _extractor is None:
        gaz = Gazetteer.from_file_or_empty(_DICT_PATH)
        _extractor = ClinicalExtractor(gazetteer=gaz if len(gaz) else None)
    return _extractor


def get_deider():
    global _deider
    if _deider is None:
        _deider = Deidentifier(keep_tags=KEEP_TAGS)
    return _deider


ASSERT_COLORS = {"affirmed": "#1a7f37", "negated": "#cf222e", "possible": "#bf8700"}
ACCEPT = ",".join(supported_extensions())

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Clinical Extractor</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px; margin: 24px auto;
         padding: 0 16px; color: #1f2328; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 22px; }}
  textarea {{ width: 100%; height: 150px; font-family: ui-monospace, monospace;
             font-size: 13px; padding: 10px; box-sizing: border-box; }}
  .row {{ display: flex; gap: 16px; align-items: center; margin: 10px 0; flex-wrap: wrap; }}
  button {{ background: #1f6feb; color: #fff; border: 0; padding: 9px 16px;
           border-radius: 6px; font-size: 14px; cursor: pointer; }}
  .note {{ white-space: pre-wrap; line-height: 1.7; border: 1px solid #d0d7de;
          padding: 14px; border-radius: 8px; background: #f6f8fa; }}
  mark {{ padding: 1px 3px; border-radius: 4px; color: #fff; font-weight: 600; }}
  .redact {{ background: #1f2328; color: #fff; padding: 1px 4px; border-radius: 4px;
            font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  .legend span {{ font-weight: 600; margin-right: 14px; }}
  .disclaimer {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 10px 14px;
                border-radius: 8px; font-size: 13px; margin-bottom: 16px; }}
  .muted {{ color: #656d76; }} .dl {{ font-weight: 600; }}
  fieldset {{ border: 1px solid #d0d7de; border-radius: 8px; margin: 0 0 8px; }}
  legend {{ padding: 0 6px; color: #656d76; font-size: 12px; }}
</style></head>
<body>
  <h1>Clinical Entity Extractor</h1>
  <div class="disclaimer">
    Research/engineering tool only &mdash; <b>not validated for clinical use</b>,
    and de-identification is <b>best-effort, not a HIPAA guarantee</b>. Files are
    processed on this machine and never uploaded to the internet.
  </div>
  <form method="post" action="/extract" enctype="multipart/form-data">
    <fieldset><legend>Option 1 &mdash; paste one note</legend>
      <textarea name="note" placeholder="Paste a clinical note here...">{note}</textarea>
    </fieldset>
    <fieldset><legend>Option 2 &mdash; upload one or more files</legend>
      <input type="file" name="files" multiple accept="{accept}">
      <div class="muted" style="font-size:12px;margin-top:6px">
        Accepts: {accept}</div>
    </fieldset>
    <div class="row">
      <button type="submit">Run</button>
      <label><input type="checkbox" name="deid" {deid_checked}> de-identify first</label>
      <label><input type="checkbox" name="link" {link_checked}> link codes (RxNorm / SNOMED)</label>
      <span class="muted">first run downloads model(s)</span>
    </div>
  </form>
  {results}
</body></html>"""


def render(note="", deid_checked="", link_checked="", results=""):
    return PAGE.format(note=html.escape(note), accept=ACCEPT,
                       deid_checked=deid_checked, link_checked=link_checked,
                       results=results)


@app.route("/")
def index():
    return render(note=_sample())


def _process(text, do_deid, do_link):
    """Run de-id (optional) + extract (+ optional link) on one note's text."""
    redactions = 0
    if do_deid:
        text, reds = get_deider().deidentify(text)
        redactions = len(reds)
    entities = get_extractor().extract(text)
    if do_link:
        TerminologyLinker(umls_api_key=os.environ.get("UMLS_API_KEY")).link(entities)
    return text, entities, redactions


@app.route("/extract", methods=["POST"])
def extract():
    note = request.form.get("note", "")
    do_deid = request.form.get("deid") == "on"
    do_link = request.form.get("link") == "on"
    dc = "checked" if do_deid else ""
    lc = "checked" if do_link else ""

    uploads = [f for f in request.files.getlist("files") if f and f.filename]

    # Batch path: one or more uploaded files.
    if uploads:
        results = _run_batch(uploads, do_deid, do_link)
        return render(note=note, deid_checked=dc, link_checked=lc, results=results)

    # Single pasted note.
    if not note.strip():
        return render(note=note, deid_checked=dc, link_checked=lc,
                      results="<p class='muted'>Paste a note or choose files, then Run.</p>")
    text, entities, reds = _process(note, do_deid, do_link)
    results = ""
    if do_deid:
        results += _deid_summary(reds)
    results += _highlight(text, entities) + _table(entities)
    return render(note=note, deid_checked=dc, link_checked=lc, results=results)


def _run_batch(uploads, do_deid, do_link):
    rows = []          # (source, entity)
    per_file = []      # (name, n_entities, n_redactions)
    for f in uploads:
        name = os.path.basename(f.filename)
        try:
            suffix = os.path.splitext(name)[1] or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                f.save(tmp.name)
                tmp_path = tmp.name
            text = extract_text(tmp_path)
            os.unlink(tmp_path)
        except Exception as exc:
            per_file.append((name, "error: " + html.escape(str(exc)), 0))
            continue
        _clean, entities, reds = _process(text, do_deid, do_link)
        per_file.append((name, len(entities), reds))
        for e in entities:
            rows.append((name, e))

    out = "<h2>Batch results <span class='muted'>(" + str(len(uploads)) + \
          " files, " + str(len(rows)) + " entities)</span></h2>"
    out += _csv_download(rows)
    # per-file summary
    out += "<table><tr><th>File</th><th>Entities</th><th>Redactions</th></tr>"
    for name, n, reds in per_file:
        out += "<tr><td>" + html.escape(name) + "</td><td>" + str(n) + \
               "</td><td>" + str(reds) + "</td></tr>"
    out += "</table>"
    out += _batch_table(rows)
    return out


def _csv_download(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["source", "text", "label", "assertion", "score",
                "code", "code_system", "code_name"])
    for src, e in rows:
        w.writerow([src, e.text, e.label, e.assertion, "{:.3f}".format(e.score),
                    e.code or "", e.code_system or "", e.code_name or ""])
    b64 = base64.b64encode(buf.getvalue().encode("utf-8")).decode("ascii")
    return ('<p><a class="dl" download="extraction_results.csv" '
            'href="data:text/csv;base64,' + b64 + '">&#11015; Download all results (CSV)</a></p>')


def _batch_table(rows):
    if not rows:
        return "<p>No entities found.</p>"
    head = ("<tr><th>File</th><th>Entity</th><th>Label</th><th>Assertion</th>"
            "<th>Score</th><th>Code</th><th>Concept</th></tr>")
    body = []
    for src, e in rows[:2000]:    # cap on-screen rows; full set is in the CSV
        code = (str(e.code_system) + ":" + str(e.code)) if e.code else ""
        body.append("<tr><td>" + html.escape(src) + "</td><td>" + html.escape(e.text)
                    + "</td><td>" + html.escape(e.label) + "</td><td>" + e.assertion
                    + "</td><td>" + "{:.2f}".format(e.score) + "</td><td>"
                    + html.escape(code) + "</td><td>" + html.escape(e.code_name or "")
                    + "</td></tr>")
    note = ""
    if len(rows) > 2000:
        note = "<p class='muted'>Showing first 2000 of " + str(len(rows)) + \
               " rows; download the CSV for all.</p>"
    return note + "<table>" + head + "".join(body) + "</table>"


def _deid_summary(n):
    return ("<h2>De-identified <span class='muted'>(" + str(n)
            + " redactions before extraction)</span></h2>")


def _highlight(note, entities):
    out, cursor = [], 0
    for e in sorted(entities, key=lambda x: x.start):
        if e.start < cursor:
            continue
        out.append(html.escape(note[cursor:e.start]))
        color = ASSERT_COLORS.get(e.assertion, "#57606a")
        title = e.label + " / " + e.assertion
        if e.code:
            title += " / " + str(e.code_system) + ":" + str(e.code)
        out.append('<mark style="background:' + color + '" title="'
                   + html.escape(title) + '">' + html.escape(note[e.start:e.end])
                   + '</mark>')
        cursor = e.end
    out.append(html.escape(note[cursor:]))
    legend = ('<div class="legend" style="margin:14px 0">'
              '<span style="color:#1a7f37">affirmed</span>'
              '<span style="color:#cf222e">negated</span>'
              '<span style="color:#bf8700">possible</span></div>')
    return "<h2>Extracted entities</h2>" + legend + '<div class="note">' + "".join(out) + "</div>"


def _table(entities):
    if not entities:
        return "<p>No entities found.</p>"
    rows = ["<tr><th>Entity</th><th>Label</th><th>Assertion</th><th>Score</th>"
            "<th>Code</th><th>Concept</th></tr>"]
    for e in sorted(entities, key=lambda x: x.start):
        code = (str(e.code_system) + ":" + str(e.code)) if e.code else ""
        rows.append(
            "<tr><td>" + html.escape(e.text) + "</td><td>" + html.escape(e.label)
            + "</td><td>" + e.assertion + "</td><td>" + "{:.2f}".format(e.score)
            + "</td><td>" + html.escape(code) + "</td><td>"
            + html.escape(e.code_name or "") + "</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def _sample():
    try:
        with open(os.path.join(os.path.dirname(__file__), "sample_note.txt")) as f:
            return f.read()
    except Exception:
        return ""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
