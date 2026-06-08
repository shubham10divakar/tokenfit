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
from tokenfit.ingest import DEFAULT_GLOBS, chunk_documents, load_corpus
from tokenfit.models import TokenfitModel
from tokenfit.retrieve import retrieve

_CACHE_ROOT = Path.home() / ".cache" / "tokenfit"


def build_naive(
    query: str,
    repo: str | Path,
    model: TokenfitModel,
    budget: int,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
) -> str:
    """Baseline: concatenate priority-sorted files, hard-truncate to budget.

    This is the bar the retrieved path must beat.
    """
    docs = load_corpus(repo, globs)
    blob = "\n\n".join(f"### FILE: {d.path}\n{d.text}" for d in docs)
    return model.truncate_to(blob, budget)


def _persist_dir(repo: Path, globs: tuple[str, ...]) -> Path:
    # Fold the glob set into the key so different --include sets cache separately.
    sig = f"{repo.resolve()}|{','.join(sorted(globs))}"
    key = hashlib.sha1(sig.encode()).hexdigest()[:12]
    return _CACHE_ROOT / key


def ensure_index(
    repo: str | Path,
    rebuild: bool = False,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
) -> Path:
    """Build the per-repo index if missing (or rebuild=True) and return its path.

    Cached under ~/.cache/tokenfit keyed by the repo path + glob set, so it's built
    once and reused across queries (and auto-rebuilds if you change --include).
    """
    repo = Path(repo)
    persist = _persist_dir(repo, globs)
    if rebuild or not _index.index_exists(persist):
        chunks = chunk_documents(load_corpus(repo, globs))
        if not chunks:
            raise ValueError(
                f"No indexable files found under {repo}. "
                f"Try --include to add file types (e.g. --include '*.gd')."
            )
        _index.build_index(chunks, persist)
    return persist


def build(
    query: str,
    repo: str | Path,
    budget: int = 8000,
    model: TokenfitModel | None = None,
    top_k: int = 12,
    rebuild: bool = False,
    globs: tuple[str, ...] = DEFAULT_GLOBS,
    hybrid: bool = True,
) -> str:
    """Retrieved context: select the most relevant chunks within the token budget.

    Pass rebuild=True after the repo changes to refresh the cached index, or a
    custom `globs` tuple to control which file types are ingested. `hybrid=True`
    fuses semantic + BM25 keyword search (Phase 2); set False for semantic-only.
    """
    model = model or TokenfitModel()
    persist = ensure_index(repo, rebuild=rebuild, globs=globs)
    ranked = retrieve(query, persist, top_k=top_k, hybrid=hybrid)
    return fit_to_budget(ranked, model, budget)
