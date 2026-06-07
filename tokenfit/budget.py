"""Budgeting — pack the highest-value chunks into a token window. (Phase 1)

Greedy fit by relevance order (chunks arrive already ranked). Phase 2 adds
summarization of oversized chunks instead of dropping them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenfit.ingest import Chunk

if TYPE_CHECKING:  # only for type hints; avoids pulling the inference SDK
    from tokenfit.models import TokenfitModel

_HEADER = "### FILE: {label}\n{text}"


def fit_to_budget(chunks: list[Chunk], model: "TokenfitModel", budget: int) -> str:
    """Concatenate ranked chunks (with file citations) up to `budget` tokens.

    Each chunk carries a `FILE: path@offset` header so the model can cite sources.
    A chunk that doesn't fit is skipped (a later, smaller chunk may still fit).
    """
    parts: list[str] = []
    used = 0
    for c in chunks:
        block = _HEADER.format(label=c.label, text=c.text)
        cost = model.count_tokens(block) + 2  # +2 for the joining newlines
        if used + cost > budget:
            continue
        parts.append(block)
        used += cost
    return "\n\n".join(parts)
