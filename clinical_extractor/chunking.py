"""
Sliding-window chunking for long clinical notes.

Why this exists
---------------
BERT-family models (including Bio_ClinicalBERT) have a hard 512-token limit.
A discharge summary is routinely 1,500-3,000 tokens. If you feed the raw note
straight into the pipeline it is silently truncated and you lose everything
past the limit.

The trick that makes reconciliation painless: we chunk on *tokens* but we carry
the *original character offset* of every chunk. That way every entity the model
finds can be mapped straight back to a span in the ORIGINAL note, and de-duping
overlapping hits becomes simple interval math (see pipeline.reconcile).

We use the model's own fast tokenizer with return_offsets_mapping so the window
boundaries always land on real token edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """A slice of the original note plus where it came from."""
    text: str          # the substring fed to the model
    char_start: int    # offset of this chunk in the ORIGINAL note
    char_end: int
    index: int         # 0-based chunk number


def sliding_window_chunks(
    text: str,
    tokenizer,
    max_tokens: int = 400,
    stride: int = 50,
) -> List[Chunk]:
    """
    Split `text` into overlapping windows of at most `max_tokens` tokens.

    Parameters
    ----------
    text : str
        The full clinical note.
    tokenizer : a HuggingFace *fast* tokenizer (PreTrainedTokenizerFast)
        Needed for offset mapping. AutoTokenizer.from_pretrained(..., use_fast=True).
    max_tokens : int
        Content tokens per window. Keep <= 510 to leave room for [CLS]/[SEP].
        400 is a sane default that leaves headroom for the special tokens the
        pipeline adds back on each chunk.
    stride : int
        Number of tokens of overlap between consecutive windows. The overlap is
        what lets an entity that straddles a boundary still be captured whole in
        at least one window.

    Returns
    -------
    list[Chunk]
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if stride < 0 or stride >= max_tokens:
        raise ValueError("stride must be >= 0 and < max_tokens")

    if not text.strip():
        return []

    # Encode WITHOUT special tokens; we want raw content tokens + their char spans.
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoding["offset_mapping"]
    n = len(offsets)

    # Short note: one chunk, no windowing needed.
    if n <= max_tokens:
        return [Chunk(text=text, char_start=0, char_end=len(text), index=0)]

    chunks: List[Chunk] = []
    step = max_tokens - stride
    idx = 0
    start_tok = 0
    while start_tok < n:
        end_tok = min(start_tok + max_tokens, n)

        # Map token window -> character window in the original string.
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]

        chunks.append(
            Chunk(
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                index=idx,
            )
        )
        idx += 1

        if end_tok == n:
            break
        start_tok += step

    return chunks
