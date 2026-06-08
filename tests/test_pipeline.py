"""Dependency-free regression test for the Phase 1 pipeline.

Uses a fake keyword-based embedder so it runs without torch / sentence-transformers /
network, yet still exercises chunk -> index -> retrieve -> budget end to end.

Run:  python -m tests.test_pipeline   (or: pytest tests/)
"""

from __future__ import annotations

import tempfile

import numpy as np

from tokenfit import budget, index, retrieve
from tokenfit.ingest import Document, chunk_documents

_VOCAB = ["auth", "login", "token", "database", "persist", "config", "test", "endpoint"]


def _fake_embed(texts, model=index.EMBED_MODEL):
    out = []
    for t in texts:
        tl = t.lower()
        v = np.array([tl.count(w) for w in _VOCAB], dtype=np.float32)
        n = np.linalg.norm(v)
        out.append(v / n if n else v)
    return np.vstack(out).astype(np.float32)


class _FakeModel:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def test_pipeline(monkeypatch=None):
    # patch the embedder in both modules that reference it
    index.embed_texts = _fake_embed
    retrieve.embed_texts = _fake_embed

    docs = [
        Document("auth.py", "def login(user):\n    # validate auth token\n    return token"),
        Document("db.py", "def save(rec):\n    # persist to database\n    database.write(rec)"),
        Document("conf.py", "CONFIG = {}\n# load config values here"),
    ]
    chunks = chunk_documents(docs, target_chars=400)
    assert len(chunks) == 3

    with tempfile.TemporaryDirectory() as d:
        index.build_index(chunks, d)
        assert index.index_exists(d)

        hits = retrieve.retrieve("how does login auth token work", d, top_k=3)
        assert hits[0].doc_path == "auth.py"  # semantic ranking works

        packed = budget.fit_to_budget(hits, _FakeModel(), budget=40)
        assert _FakeModel().count_tokens(packed) <= 40  # budget respected
        assert "### FILE: auth.py@" in packed  # citations present


def test_hybrid_bm25_keyword(monkeypatch=None):
    """BM25 should surface an exact identifier the (fake) embedder is blind to.

    The fake embedder only knows _VOCAB, so a rare identifier like
    `frobnicate_widget` yields a zero semantic vector for every chunk — semantic
    ranking is a coin flip. Only the keyword path can pick the right chunk, so a
    correct top-1 proves BM25 is actually contributing to the fused result.
    """
    try:
        import rank_bm25  # noqa: F401
    except ImportError:
        print("SKIP test_hybrid_bm25_keyword (rank_bm25 not installed)")
        return

    index.embed_texts = _fake_embed
    retrieve.embed_texts = _fake_embed
    retrieve._bm25_cached.cache_clear()  # don't reuse a prior tempdir's index

    docs = [
        Document("a.py", "def helper():\n    return 1"),
        Document("b.py", "def frobnicate_widget():\n    # rare identifier\n    return 2"),
        Document("c.py", "def other():\n    return 3"),
    ]
    chunks = chunk_documents(docs, target_chars=400)

    with tempfile.TemporaryDirectory() as d:
        index.build_index(chunks, d)
        hits = retrieve.retrieve("frobnicate_widget", d, top_k=3, hybrid=True)
        assert hits[0].doc_path == "b.py"  # keyword match wins

        # With hybrid off, the blind embedder has no way to find it.
        sem = retrieve.retrieve("frobnicate_widget", d, top_k=3, hybrid=False)
        assert any(h.doc_path == "b.py" for h in sem)  # present, but not reliably first


if __name__ == "__main__":
    test_pipeline()
    test_hybrid_bm25_keyword()
    print("PASSED")
