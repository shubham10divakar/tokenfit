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


def ensure_index(repo: str | Path, rebuild: bool = False) -> Path:
    """Build the per-repo index if missing (or rebuild=True) and return its path.

    Cached under ~/.cache/tokenfit keyed by the repo's absolute path, so it's built
    once and reused across queries.
    """
    repo = Path(repo)
    persist = _persist_dir(repo)
    if rebuild or not _index.index_exists(persist):
        chunks = chunk_documents(load_corpus(repo))
        if not chunks:
            raise ValueError(f"No indexable files found under {repo}")
        _index.build_index(chunks, persist)
    return persist


def build(
    query: str,
    repo: str | Path,
    budget: int = 8000,
    model: TokenfitModel | None = None,
    top_k: int = 12,
    rebuild: bool = False,
) -> str:
    """Retrieved context: select the most relevant chunks within the token budget.

    Pass rebuild=True after the repo changes to refresh the cached index.
    """
    model = model or TokenfitModel()
    persist = ensure_index(repo, rebuild=rebuild)
    ranked = retrieve(query, persist, top_k=top_k)
    return fit_to_budget(ranked, model, budget)
