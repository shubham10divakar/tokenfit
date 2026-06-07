"""Corpus loading + chunking.

Phase 0 only needs `load_corpus` (to build the naive baseline by concatenating
files until the budget is hit). Chunking is stubbed for Phase 1 retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# File types we treat as "context" for a coding agent.
DEFAULT_GLOBS = ("*.md", "*.py", "*.txt", "*.toml", "*.cfg", "*.yaml", "*.yml")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache"}


@dataclass
class Document:
    path: str  # relative path, used as a citation handle
    text: str


def load_corpus(root: str | Path, globs: tuple[str, ...] = DEFAULT_GLOBS) -> list[Document]:
    """Load all matching files under `root` into Documents.

    Priority files (AGENTS.md, SKILL.md) are sorted first so naive truncation
    keeps them when the budget is tight.
    """
    root = Path(root)
    docs: list[Document] = []
    for pattern in globs:
        for fp in root.rglob(pattern):
            if any(part in SKIP_DIRS for part in fp.parts):
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeError):
                continue
            docs.append(Document(path=str(fp.relative_to(root)), text=text))

    def priority(d: Document) -> int:
        name = Path(d.path).name.lower()
        if name == "agents.md":
            return 0
        if name == "skill.md" or name.endswith(".skill.md"):
            return 1
        if name.endswith(".md"):
            return 2
        return 3

    docs.sort(key=priority)
    return docs


# --- chunking -------------------------------------------------------------
@dataclass
class Chunk:
    doc_path: str
    text: str
    start: int  # char offset within the document (citation handle)

    @property
    def label(self) -> str:
        return f"{self.doc_path}@{self.start}"


# Lines that mark a natural boundary in source code (split BEFORE them).
_CODE_BOUNDARY_PREFIXES = ("def ", "async def ", "class ", "@")


def _split_on_boundaries(text: str, is_boundary) -> list[tuple[int, str]]:
    """Split text into (start_offset, section) pieces at lines where
    is_boundary(line) is True. The boundary line begins the next section.
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    offset = 0
    for line in lines:
        if is_boundary(line) and buf:
            sections.append((buf_start, "".join(buf)))
            buf = []
            buf_start = offset
        buf.append(line)
        offset += len(line)
    if buf:
        sections.append((buf_start, "".join(buf)))
    return sections or [(0, text)]


def _sections_for(doc: Document) -> list[tuple[int, str]]:
    name = doc.path.lower()
    if name.endswith(".md"):
        return _split_on_boundaries(doc.text, lambda ln: ln.lstrip().startswith("#"))
    if name.endswith(".py"):
        return _split_on_boundaries(
            doc.text, lambda ln: ln.lstrip().startswith(_CODE_BOUNDARY_PREFIXES)
        )
    return [(0, doc.text)]


def _window(start: int, text: str, target: int, overlap: int) -> list[tuple[int, str]]:
    """Slide a window over an oversized section, breaking on newlines when possible."""
    if len(text) <= target:
        return [(start, text)]
    out: list[tuple[int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + target, n)
        if end < n:
            nl = text.rfind("\n", i + overlap, end)  # prefer a clean line break
            if nl != -1:
                end = nl + 1
        out.append((start + i, text[i:end]))
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return out


def chunk_documents(
    docs: list[Document], target_chars: int = 1200, overlap: int = 150
) -> list[Chunk]:
    """Split documents into retrievable chunks.

    Markdown splits on headers, Python on def/class/decorator boundaries, everything
    else as a plain stream; oversized sections are windowed with overlap.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        for sec_start, sec_text in _sections_for(doc):
            if not sec_text.strip():
                continue
            for win_start, win_text in _window(sec_start, sec_text, target_chars, overlap):
                if win_text.strip():
                    chunks.append(Chunk(doc.path, win_text, win_start))
    return chunks
