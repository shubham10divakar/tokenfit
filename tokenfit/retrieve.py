"""Retrieval — semantic search over the persisted index. (Phase 1)

Phase 2 will add BM25 hybrid + a cross-encoder rerank; the signature stays the same.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenfit.index import embed_texts, load_index
from tokenfit.ingest import Chunk


def retrieve(query: str, persist_dir: str | Path, top_k: int = 12) -> list[Chunk]:
    """Return the top_k most semantically similar chunks to `query`."""
    vecs, chunks = load_index(persist_dir)
    if len(chunks) == 0:
        return []
    q = embed_texts([query])[0]  # already L2-normalized
    scores = vecs @ q  # cosine similarity (both normalized)
    k = min(top_k, len(chunks))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]  # sort the k by descending score
    return [chunks[i] for i in top]
