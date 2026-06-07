"""Indexing — embed chunks and persist a lightweight numpy vector store.

Phase 1 is semantic-only and single-repo, so we skip a vector DB service: embeddings
go to a `.npz` file with a JSON sidecar of the chunk text/citations. (Chroma/BM25 are
Phase 2/3 swaps — see plan.md.)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from tokenfit.ingest import Chunk

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_VECTORS_FILE = "vectors.npz"
_CHUNKS_FILE = "chunks.json"


@lru_cache(maxsize=2)
def _embedder(model: str = EMBED_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """Return L2-normalized embeddings so dot product == cosine similarity."""
    vecs = _embedder(model).encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype=np.float32)


def build_index(chunks: list[Chunk], persist_dir: str | Path, model: str = EMBED_MODEL) -> None:
    """Embed chunks and persist vectors + chunk metadata to `persist_dir`."""
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    vecs = embed_texts([c.text for c in chunks], model)
    np.savez_compressed(persist_dir / _VECTORS_FILE, vectors=vecs)
    (persist_dir / _CHUNKS_FILE).write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )


def index_exists(persist_dir: str | Path) -> bool:
    persist_dir = Path(persist_dir)
    return (persist_dir / _VECTORS_FILE).exists() and (persist_dir / _CHUNKS_FILE).exists()


def load_index(persist_dir: str | Path) -> tuple[np.ndarray, list[Chunk]]:
    persist_dir = Path(persist_dir)
    vecs = np.load(persist_dir / _VECTORS_FILE)["vectors"]
    raw = json.loads((persist_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in raw]
    return vecs, chunks
