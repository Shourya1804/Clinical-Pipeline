"""
Terminology linking: map extracted entities to standard medical codes.

Two backends, used together:

  * RxNorm  - free, no API key. Best for MEDICATIONS. Uses the public RxNav
              REST service (approximateTerm -> properties).
  * UMLS    - requires a free NLM/UTS API key. Gives SNOMED CT codes (and any
              other UMLS source) for PROBLEMS / SYMPTOMS / PROCEDURES.

Design notes
------------
* Network is optional and failure-soft. If a service is unreachable or no key
  is set, the entity simply comes back without a code instead of crashing.
* Results are cached on disk (.cache/linking.json) so repeat terms don't hammer
  the public APIs and batch runs stay fast.
* `fetch_json` is injectable so the logic can be unit-tested with NO network.

Privacy
-------
RxNorm/UMLS lookups send the matched ENTITY TEXT (e.g. "metformin",
"pneumonia") to NLM servers - never the full note. Still, if your notes contain
PHI, review your institution's policy before enabling online linking. Set
`use_rxnorm=False` and omit the UMLS key to keep everything fully local.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

# Which backend to use for a given NER label. Model label sets differ, so we
# match on substrings (lower-cased) rather than exact strings.
MED_HINTS = ("medic", "drug", "chemical", "substance", "rxnorm")
PROBLEM_HINTS = (
    "disease", "disorder", "problem", "sign", "symptom", "finding",
    "procedure", "diagnos", "condition", "syndrome", "injury", "neoplasm",
)


def _default_fetch_json(url: str, timeout: float = 8.0) -> Optional[dict]:
    """GET a URL and parse JSON. Returns None on any failure (failure-soft)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clinical-extractor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class TerminologyLinker:
    def __init__(
        self,
        umls_api_key: Optional[str] = None,
        use_rxnorm: bool = True,
        umls_sab: str = "SNOMEDCT_US",
        cache_path: Optional[str] = ".cache/linking.json",
        timeout: float = 8.0,
        fetch_json: Optional[Callable[[str], Optional[dict]]] = None,
    ):
        # Key can come from the constructor or the environment.
        self.umls_api_key = umls_api_key or os.environ.get("UMLS_API_KEY")
        self.use_rxnorm = use_rxnorm
        self.umls_sab = umls_sab
        self.timeout = timeout
        self._fetch = fetch_json or (lambda u: _default_fetch_json(u, timeout))

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, Optional[dict]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text("utf-8"))
            except Exception:
                self._cache = {}

    # ------------------------------------------------------------------ #
    def backend_for(self, label: str) -> Optional[str]:
        """Decide which terminology to query for an entity label."""
        low = (label or "").lower()
        if any(h in low for h in MED_HINTS):
            return "RXNORM" if self.use_rxnorm else None
        if any(h in low for h in PROBLEM_HINTS):
            return self.umls_sab if self.umls_api_key else None
        # Unknown label: try UMLS (broad) if we have a key, else RxNorm.
        if self.umls_api_key:
            return self.umls_sab
        return "RXNORM" if self.use_rxnorm else None

    # ------------------------------------------------------------------ #
    def lookup(self, term: str, system: str) -> Optional[dict]:
        """Return {code, code_name, system, score} for a term, or None."""
        term = (term or "").strip()
        if not term:
            return None
        key = f"{system}::{term.lower()}"
        if key in self._cache:
            return self._cache[key]

        if system == "RXNORM":
            result = self._rxnorm(term)
        else:
            result = self._umls(term, system)

        self._cache[key] = result
        return result

    # ------------------------------------------------------------------ #
    def link(self, entities: List, only_assertions=None) -> List:
        """Attach codes to a list of Entity objects (in place) and return it.

        only_assertions: optional iterable, e.g. {"affirmed", "possible"}, to
        skip linking negated findings. Default links everything.
        """
        for ent in entities:
            if only_assertions is not None and ent.assertion not in only_assertions:
                continue
            system = self.backend_for(ent.label)
            if not system:
                continue
            hit = self.lookup(ent.text, system)
            if hit:
                ent.code = hit.get("code")
                ent.code_system = hit.get("system")
                ent.code_name = hit.get("code_name")
                ent.link_score = hit.get("score")
        self.save()
        return entities

    # ------------------------------------------------------------------ #
    def _rxnorm(self, term: str) -> Optional[dict]:
        q = urllib.parse.quote(term)
        approx = self._fetch(
            f"{RXNAV_BASE}/approximateTerm.json?term={q}&maxEntries=1"
        )
        try:
            cand = approx["approximateGroup"]["candidate"]
            cand = cand[0] if isinstance(cand, list) else cand
            rxcui = cand["rxcui"]
            score = float(cand.get("score", 0))
        except (TypeError, KeyError, IndexError, ValueError):
            return None
        if not rxcui:
            return None

        name = cand.get("name")
        if not name:
            props = self._fetch(f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json")
            try:
                name = props["properties"]["name"]
            except (TypeError, KeyError):
                name = None
        return {"code": rxcui, "code_name": name, "system": "RXNORM", "score": score}

    # ------------------------------------------------------------------ #
    def _umls(self, term: str, sab: str) -> Optional[dict]:
        if not self.umls_api_key:
            return None
        q = urllib.parse.quote(term)
        data = self._fetch(
            f"{UMLS_BASE}/search/current?string={q}&sabs={sab}"
            f"&returnIdType=code&pageSize=1&apiKey={self.umls_api_key}"
        )
        try:
            results = data["result"]["results"]
            if not results or results[0].get("ui") in (None, "NONE"):
                return None
            top = results[0]
        except (TypeError, KeyError, IndexError):
            return None
        return {
            "code": top.get("ui"),
            "code_name": top.get("name"),
            "system": top.get("rootSource", sab),
            "score": None,
        }

    # ------------------------------------------------------------------ #
    def save(self):
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), "utf-8")
        except Exception:
            pass
