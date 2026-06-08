"""Retrieval — hybrid semantic + BM25 keyword search over the persisted index.

Phase 2: a dense ranking (bge-small cosine) and a sparse ranking (BM25 over a
code-aware tokenization) are fused with Reciprocal Rank Fusion. Dense catches
paraphrase/meaning; sparse nails exact identifiers (function/var names, error
strings) that embeddings blur. RRF needs no score normalization, so it's robust
to the different scales of cosine [-1, 1] vs BM25 [0, inf).

If `rank_bm25` isn't installed we fall back to Phase-1 semantic-only retrieval,
so the public signature and behavior degrade gracefully.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from tokenfit.index import _CHUNKS_FILE, embed_texts, load_index
from tokenfit.ingest import Chunk

# Insert a split point between a lower/digit and an upper char so camelCase and
# PascalCase break into their parts (flipH -> flip h, moveToward -> move toward).
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(text: str) -> list[str]:
    """Code-aware tokenizer for BM25.

    Splits camelCase, treats snake_case/punctuation as separators, lowercases.
    `JUMP_VELOCITY`/`move_toward`/`flipH` all decompose to their word parts so a
    natural-language query ("velocity", "toward") matches the identifier.
    """
    return re.findall(r"[a-z0-9]+", _CAMEL.sub(" ", text).lower())


@lru_cache(maxsize=4)
def _bm25_cached(persist_dir: str, _mtime: float):
    """Build (and process-cache) a BM25 index from the persisted chunks.

    Keyed by the chunks-file mtime so a `--rebuild` in the same process is picked
    up instead of serving a stale index. Caller guarantees rank_bm25 is importable.
    """
    from rank_bm25 import BM25Okapi

    _, chunks = load_index(persist_dir)
    return BM25Okapi([_tokenize(c.text) for c in chunks])


def _get_bm25(persist_dir: str | Path):
    """Return a cached BM25 index, or None if rank_bm25 isn't installed."""
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        return None
    mtime = (Path(persist_dir) / _CHUNKS_FILE).stat().st_mtime
    return _bm25_cached(str(persist_dir), mtime)


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Map scores -> 0-based rank per index (rank 0 = highest score).

    Tied scores share an averaged rank, so a non-informative ranking (e.g. all
    scores equal) contributes equally to every doc under RRF instead of biasing
    the result by array order. For distinct scores this is plain ordinal rank.
    """
    n = len(scores)
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ranks_sorted = np.arange(n, dtype=np.float64)
    # Average the ordinal positions within each run of equal scores.
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > 1:
            ranks_sorted[a:b] = (a + b - 1) / 2.0
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks_sorted
    return out


def _rrf(sem_scores: np.ndarray, kw_scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion -> indices sorted by descending fused score.

    Each list contributes 1/(k + rank); k=60 is the standard damping constant
    that keeps any single high rank from dominating the blend.
    """
    fused = 1.0 / (k + _ranks(sem_scores)) + 1.0 / (k + _ranks(kw_scores))
    return np.argsort(-fused)


def retrieve(
    query: str,
    persist_dir: str | Path,
    top_k: int = 12,
    hybrid: bool = True,
) -> list[Chunk]:
    """Return the top_k most relevant chunks for `query`.

    With `hybrid=True` (default) this fuses semantic and BM25 rankings; set
    `hybrid=False` (or omit rank_bm25) for pure semantic search.
    """
    vecs, chunks = load_index(persist_dir)
    if len(chunks) == 0:
        return []

    q = embed_texts([query])[0]  # already L2-normalized
    sem_scores = vecs @ q  # cosine similarity (both normalized)

    bm25 = _get_bm25(persist_dir) if hybrid else None
    if bm25 is None:
        order = np.argsort(-sem_scores)
    else:
        kw_scores = np.asarray(bm25.get_scores(_tokenize(query)), dtype=np.float32)
        order = _rrf(sem_scores, kw_scores)

    return [chunks[i] for i in order[:top_k]]
