"""Public API — the one function users call.

    from tokenfit import pack
    prompt = pack.build("how does auth work?", repo="./myrepo", budget=8000)

`build_naive` is the truncation baseline; `build` is the retrieved pipeline
(ingest -> chunk -> index -> retrieve -> budget). The eval harness compares them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tokenfit import index as _index
from tokenfit.budget import fit_to_budget
from tokenfit.ingest import chunk_documents, load_corpus
from tokenfit.models import TokenfitModel
from tokenfit.retrieve import retrieve

_CACHE_ROOT = Path.home() / ".cache" / "tokenfit"


def build_naive(query: str, repo: str | Path, model: TokenfitModel, budget: int) -> str:
    """Baseline: concatenate priority-sorted files, hard-truncate to budget.

    This is the bar the retrieved path must beat.
    """
    docs = load_corpus(repo)
    blob = "\n\n".join(f"### FILE: {d.path}\n{d.text}" for d in docs)
    return model.truncate_to(blob, budget)


def _persist_dir(repo: Path) -> Path:
    key = hashlib.sha1(str(repo.resolve()).encode()).hexdigest()[:12]
    return _CACHE_ROOT / key


def build(
    query: str,
    repo: str | Path,
    budget: int = 8000,
    model: TokenfitModel | None = None,
    top_k: int = 12,
    rebuild: bool = False,
) -> str:
    """Retrieved context: select the most relevant chunks within the token budget.

    The index is built once per repo and cached under ~/.cache/tokenfit; pass
    rebuild=True after the repo changes.
    """
    repo = Path(repo)
    model = model or TokenfitModel()
    persist = _persist_dir(repo)

    if rebuild or not _index.index_exists(persist):
        chunks = chunk_documents(load_corpus(repo))
        if not chunks:
            return ""
        _index.build_index(chunks, persist)

    ranked = retrieve(query, persist, top_k=top_k)
    return fit_to_budget(ranked, model, budget)
