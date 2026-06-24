"""
NegEx-style assertion detection.

Why this exists
---------------
NER models find *mentions*. They do not tell you whether the finding is
actually present. "Rule out pneumonia", "no evidence of stroke", "denies chest
pain" all yield a happily-extracted entity that is in fact NEGATED or UNCERTAIN.

This is a compact, dependency-free implementation of the NegEx algorithm
(Chapman et al., 2001). For each entity we look at the words around it inside
the same sentence:

  - a PRE-negation trigger before the entity   ("no", "denies", "without")  -> Negated
  - a POST-negation trigger after the entity    ("is ruled out", "unlikely") -> Negated
  - an UNCERTAINTY trigger                       ("possible", "rule out")     -> Possible
  - PSEUDO-negation phrases                      ("no increase", "not only")  -> ignored
  - TERMINATION terms ("but", "however", "except") stop the scope of a trigger.

It is deliberately rule-based and fast: this runs on CPU in microseconds, which
is exactly where you want determinism in a clinical setting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class Assertion(str, Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    POSSIBLE = "possible"


# Phrases that LOOK like negation but are not. Checked first and removed.
PSEUDO_NEGATIONS = [
    "no increase", "no change", "no significant change", "not only", "not necessarily",
    "no further", "without difficulty", "not drain", "gram negative", "not extend",
    "not cause", "gram-negative",
]

# Negation that appears BEFORE the target. Scope runs forward.
PRE_NEGATIONS = [
    "no", "denies", "denied", "deny", "without", "absent", "negative for",
    "no evidence of", "no sign of", "no signs of", "no suspicion of",
    "no findings of", "not", "free of", "fails to reveal", "rules out",
    "ruled out", "no complaints of", "resolved",
    "never had", "no history of", "without any evidence of", "unremarkable for",
]

# Negation that appears AFTER the target. Scope runs backward.
POST_NEGATIONS = [
    "is ruled out", "are ruled out", "was ruled out", "were ruled out",
    "unlikely", "is negative", "are negative", "not seen", "is excluded",
    "have been ruled out", "has been ruled out", "declined",
]

# Uncertainty / hedging -> POSSIBLE rather than a hard negation.
UNCERTAINTY = [
    "possible", "possibly", "probable", "probably", "likely", "questionable",
    "question of", "suspected", "suspicious for", "concern for", "concerning for",
    "cannot exclude", "cannot rule out", "differential", "versus", "vs",
    "rule out", "r/o", "ro", "may represent", "could be", "evaluate for",
    "consistent with", "presumed",
]

# Words that END the scope of a trigger.
TERMINATIONS = [
    "but", "however", "nevertheless", "yet", "though", "although", "still",
    "aside from", "except", "apart from", "secondary to", "as well as",
    "cause for", "causes for", "etiology for", "origin for", "source for",
    "reason for", "trigger event for",
]


def _split_sentences(text: str) -> List[Tuple[int, int]]:
    """Very small sentence splitter -> list of (start, end) char spans.

    Clinical notes are messy (newlines, semicolons, lists), so we split on
    sentence punctuation AND newlines rather than relying on a heavy NLP model.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.;\n\r]+", text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Lowercased word tokens with their char offsets."""
    return [(m.group(0).lower(), m.start(), m.end())
            for m in re.finditer(r"[A-Za-z0-9/']+", text)]


def _phrase_hits(tokens: List[Tuple[str, int, int]], phrases: List[str]):
    """Yield (start_token_idx, end_token_idx_exclusive) for each phrase match.

    Matches multi-word phrases against the token stream (case-insensitive).
    """
    words = [t[0] for t in tokens]
    # Longest phrases first so "no evidence of" wins over "no".
    for phrase in sorted(phrases, key=lambda p: -len(p.split())):
        parts = phrase.split()
        L = len(parts)
        for i in range(len(words) - L + 1):
            if words[i:i + L] == parts:
                yield (i, i + L, phrase)


class NegEx:
    """Assertion classifier. Stateless; one instance can be reused."""

    def __init__(self, scope: int = 6):
        # How many tokens forward/backward a trigger reaches before TERMINATION.
        self.scope = scope

    def classify(self, sentence: str, ent_start: int, ent_end: int) -> Assertion:
        """
        Classify one entity given the sentence it lives in.

        sentence  : the sentence text
        ent_start : entity start offset RELATIVE TO THE SENTENCE
        ent_end   : entity end offset relative to the sentence
        """
        tokens = _tokenize_with_spans(sentence)
        if not tokens:
            return Assertion.AFFIRMED

        # Map the entity's char span to a token index range.
        ent_tok_start = None
        ent_tok_end = None
        for i, (_, s, e) in enumerate(tokens):
            if e > ent_start and s < ent_end:
                if ent_tok_start is None:
                    ent_tok_start = i
                ent_tok_end = i
        if ent_tok_start is None:
            return Assertion.AFFIRMED

        words = [t[0] for t in tokens]

        # Remove pseudo-negation regions so they cannot trigger.
        masked = set()
        for s, e, _ in _phrase_hits(tokens, PSEUDO_NEGATIONS):
            masked.update(range(s, e))

        termination_idx = {i for i, w in enumerate(words)
                           for s, e, _ in _phrase_hits(tokens, TERMINATIONS)
                           if s <= i < e}

        def scope_clear(a: int, b: int) -> bool:
            """True if no termination term sits between token idx a and b."""
            lo, hi = min(a, b), max(a, b)
            return not any(lo < t < hi for t in termination_idx)

        result = Assertion.AFFIRMED

        # PRE negations: trigger before entity, entity within forward scope.
        for s, e, _ in _phrase_hits(tokens, PRE_NEGATIONS):
            if s in masked:
                continue
            if e <= ent_tok_start and (ent_tok_start - e) <= self.scope \
                    and scope_clear(e, ent_tok_start):
                return Assertion.NEGATED

        # POST negations: trigger after entity, within backward scope.
        for s, e, _ in _phrase_hits(tokens, POST_NEGATIONS):
            if s in masked:
                continue
            if s >= ent_tok_end and (s - ent_tok_end) <= self.scope \
                    and scope_clear(ent_tok_end, s):
                return Assertion.NEGATED

        # UNCERTAINTY: hedging either side -> possible.
        for s, e, _ in _phrase_hits(tokens, UNCERTAINTY):
            if s in masked:
                continue
            near_before = e <= ent_tok_start and (ent_tok_start - e) <= self.scope \
                and scope_clear(e, ent_tok_start)
            near_after = s >= ent_tok_end and (s - ent_tok_end) <= self.scope \
                and scope_clear(ent_tok_end, s)
            if near_before or near_after:
                result = Assertion.POSSIBLE

        return result
