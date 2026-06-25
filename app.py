"""
Local web app for the clinical extractor.

Paste a note in your browser, get color-coded entities + a table with codes.
Everything runs on YOUR machine; the note is never uploaded anywhere.

  * "De-identify first" redacts PHI (names, dates, IDs, ...) BEFORE extraction.
    Best-effort only - not a compliance guarantee (see README).
  * "Link codes" sends only the short matched terms (not the note) to the public
    RxNorm/UMLS APIs. Leave both off to stay fully offline.

Run:
    pip install -r requirements.txt
    python app.py
    # then open http://127.0.0.1:5000 in your browser

Models (~400 MB extraction + de-id) download on first use only.
"""

from __future__ import annotations

import html
import os

from flask import Flask, request

from clinical_extractor import ClinicalExtractor, TerminologyLinker, Deidentifier

app = Flask(__name__)

_extractor = None
_deider = None


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ClinicalExtractor()
    return _extractor


# Categories to LEAVE in the text during de-identification (per user policy).
# GENDER is never targeted by the model/regex, so it is kept automatically.
KEEP_TAGS = {"AGE", "LOCATION"}


def get_deider():
    global _deider
    if _deider is None:
        _deider = Deidentifier(keep_tags=KEEP_TAGS)
    return _deider


ASSERT_COLORS = {
    "affirmed": "#1a7f37",
    "negated": "#cf222e",
    "possible": "#bf8700",
}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Clinical Extractor</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 24px auto;
         padding: 0 16px; color: #1f2328; }}
  h1 {{ font-size: 20px; }}
  textarea {{ width: 100%; height: 180px; font-family: ui-monospace, monospace;
             font-size: 13px; padding: 10px; box-sizing: border-box; }}
  .row {{ display: flex; gap: 16px; align-items: center; margin: 10px 0;
         flex-wrap: wrap; }}
  button {{ background: #1f6feb; color: #fff; border: 0; padding: 9px 16px;
           border-radius: 6px; font-size: 14px; cursor: pointer; }}
  .note {{ white-space: pre-wrap; line-height: 1.7; border: 1px solid #d0d7de;
          padding: 14px; border-radius: 8px; background: #f6f8fa; }}
  mark {{ padding: 1px 3px; border-radius: 4px; color: #fff; font-weight: 600; }}
  .redact {{ background: #1f2328; color: #fff; padding: 1px 4px; border-radius: 4px;
            font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  .legend span {{ font-weight: 600; margin-right: 14px; }}
  .disclaimer {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 10px 14px;
                border-radius: 8px; font-size: 13px; margin-bottom: 18px; }}
  .muted {{ color: #656d76; }}
  h2 {{ font-size: 15px; margin-top: 22px; }}
</style></head>
<body>
  <h1>Clinical Entity Extractor</h1>
  <div class="disclaimer">
    Research/engineering tool only &mdash; <b>not validated for clinical use</b>,
    and de-identification is <b>best-effort, not a HIPAA guarantee</b>. A human
    must review redactions before any data leaves your control. The note stays on
    this machine; code linking sends only matched terms to public NLM APIs.
  </div>
  <form method="post" action="/extract">
    <textarea name="note" placeholder="Paste a clinical note here...">{note}</textarea>
    <div class="row">
      <button type="submit">Run</button>
      <label><input type="checkbox" name="deid" {deid_checked}> de-identify first (redact PHI)</label>
      <label><input type="checkbox" name="link" {link_checked}> link codes (RxNorm / SNOMED)</label>
      <span class="muted">first run downloads model(s)</span>
    </div>
  </form>
  {results}
</body></html>"""


def render(note="", deid_checked="", link_checked="", results=""):
    return PAGE.format(note=html.escape(note), deid_checked=deid_checked,
                       link_checked=link_checked, results=results)


@app.route("/")
def index():
    return render(note=_sample())


@app.route("/extract", methods=["POST"])
def extract():
    note = request.form.get("note", "")
    do_deid = request.form.get("deid") == "on"
    do_link = request.form.get("link") == "on"
    dc = "checked" if do_deid else ""
    lc = "checked" if do_link else ""
    if not note.strip():
        return render(note=note, deid_checked=dc, link_checked=lc)

    results = ""
    text = note
    if do_deid:
        clean, reds = get_deider().deidentify(note)
        results += _deid_block(clean, reds)
        text = clean      # extract on the redacted text

    entities = get_extractor().extract(text)
    if do_link:
        TerminologyLinker(umls_api_key=os.environ.get("UMLS_API_KEY")).link(entities)

    results += _highlight(text, entities) + _table(entities)
    return render(note=note, deid_checked=dc, link_checked=lc, results=results)


def _deid_block(clean, reds):
    counts = {}
    for r in reds:
        counts[r.tag] = counts.get(r.tag, 0) + 1
    summary = ", ".join("{} {}".format(v, k) for k, v in sorted(counts.items())) or "none"
    shown = html.escape(clean).replace("[NAME]", '<span class="redact">[NAME]</span>')
    for tag in ("DATE", "SSN", "PHONE", "EMAIL", "URL", "IP", "ID", "AGE",
                "LOCATION", "OTHER"):
        shown = shown.replace("[" + tag + "]",
                              '<span class="redact">[' + tag + ']</span>')
    return ("<h2>De-identified note <span class='muted'>(" + str(len(reds))
            + " redactions: " + html.escape(summary) + ")</span></h2>"
            "<div class='note'>" + shown + "</div>")


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
