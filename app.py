"""
Local web app for the clinical extractor.

Paste a note in your browser, get color-coded entities + a table with codes.
Everything runs on YOUR machine; the note is never uploaded anywhere. (If you
tick "link codes", only the short matched terms - not the note - are sent to the
public RxNorm/UMLS APIs. Leave it off to stay fully offline.)

Run:
    pip install -r requirements.txt
    python app.py
    # then open http://127.0.0.1:5000 in your browser

The model (~400 MB) downloads on the first extraction only.
"""

from __future__ import annotations

import html
import os

from flask import Flask, request

from clinical_extractor import ClinicalExtractor, TerminologyLinker

app = Flask(__name__)

# Lazy singletons so startup is instant; the model loads on first /extract.
_extractor = None


def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ClinicalExtractor()
    return _extractor


ASSERT_COLORS = {
    "affirmed": "#1a7f37",   # green
    "negated": "#cf222e",    # red
    "possible": "#bf8700",   # amber
}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Clinical Extractor</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 24px auto;
         padding: 0 16px; color: #1f2328; }}
  h1 {{ font-size: 20px; }}
  textarea {{ width: 100%; height: 180px; font-family: ui-monospace, monospace;
             font-size: 13px; padding: 10px; box-sizing: border-box; }}
  .row {{ display: flex; gap: 16px; align-items: center; margin: 10px 0; }}
  button {{ background: #1f6feb; color: #fff; border: 0; padding: 9px 16px;
           border-radius: 6px; font-size: 14px; cursor: pointer; }}
  .note {{ white-space: pre-wrap; line-height: 1.7; border: 1px solid #d0d7de;
          padding: 14px; border-radius: 8px; background: #f6f8fa; }}
  mark {{ padding: 1px 3px; border-radius: 4px; color: #fff; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  .legend span {{ font-weight: 600; margin-right: 14px; }}
  .disclaimer {{ background: #fff8c5; border: 1px solid #d4a72c; padding: 10px 14px;
                border-radius: 8px; font-size: 13px; margin-bottom: 18px; }}
  .muted {{ color: #656d76; }}
</style></head>
<body>
  <h1>Clinical Entity Extractor</h1>
  <div class="disclaimer">
    Research/engineering tool only &mdash; <b>not validated for clinical use</b>.
    Do not rely on this for patient-care decisions. The note stays on this
    machine; code linking sends only matched terms to public NLM APIs.
  </div>
  <form method="post" action="/extract">
    <textarea name="note" placeholder="Paste a clinical note here...">{note}</textarea>
    <div class="row">
      <button type="submit">Extract entities</button>
      <label><input type="checkbox" name="link" {link_checked}> link codes (RxNorm / SNOMED, uses network)</label>
      <span class="muted">first run downloads the model (~400 MB)</span>
    </div>
  </form>
  {results}
</body></html>"""


def render(note="", link_checked="", results=""):
    return PAGE.format(note=html.escape(note), link_checked=link_checked, results=results)


@app.route("/")
def index():
    return render(note=_sample())


@app.route("/extract", methods=["POST"])
def extract():
    note = request.form.get("note", "")
    do_link = request.form.get("link") == "on"
    if not note.strip():
        return render(note=note, link_checked="checked" if do_link else "")

    entities = get_extractor().extract(note)
    if do_link:
        TerminologyLinker(umls_api_key=os.environ.get("UMLS_API_KEY")).link(entities)

    results = _highlight(note, entities) + _table(entities)
    return render(note=note, link_checked="checked" if do_link else "", results=results)


def _highlight(note, entities):
    """Rebuild the note with <mark> spans around each entity."""
    out, cursor = [], 0
    for e in sorted(entities, key=lambda x: x.start):
        if e.start < cursor:      # overlap guard
            continue
        out.append(html.escape(note[cursor:e.start]))
        color = ASSERT_COLORS.get(e.assertion, "#57606a")
        title = f"{e.label} / {e.assertion}"
        if e.code:
            title += f" / {e.code_system}:{e.code}"
        out.append(f'<mark style="background:{color}" title="{html.escape(title)}">'
                   f'{html.escape(note[e.start:e.end])}</mark>')
        cursor = e.end
    out.append(html.escape(note[cursor:]))
    legend = ('<div class="legend" style="margin:14px 0">'
              '<span style="color:#1a7f37">affirmed</span>'
              '<span style="color:#cf222e">negated</span>'
              '<span style="color:#bf8700">possible</span></div>')
    return legend + '<div class="note">' + "".join(out) + "</div>"


def _table(entities):
    if not entities:
        return "<p>No entities found.</p>"
    rows = ["<tr><th>Entity</th><th>Label</th><th>Assertion</th><th>Score</th>"
            "<th>Code</th><th>Concept</th></tr>"]
    for e in sorted(entities, key=lambda x: x.start):
        code = f"{e.code_system}:{e.code}" if e.code else ""
        rows.append(
            f"<tr><td>{html.escape(e.text)}</td><td>{html.escape(e.label)}</td>"
            f"<td>{e.assertion}</td><td>{e.score:.2f}</td>"
            f"<td>{html.escape(code)}</td><td>{html.escape(e.code_name or '')}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _sample():
    try:
        with open(os.path.join(os.path.dirname(__file__), "sample_note.txt")) as f:
            return f.read()
    except Exception:
        return ""


if __name__ == "__main__":
    # host=127.0.0.1 keeps it local-only (not exposed to your network).
    app.run(host="127.0.0.1", port=5000, debug=False)
