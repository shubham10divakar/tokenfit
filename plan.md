# Plan: Context-Selection Pre-Processor

> **Companion to** [`idea.md`](./idea.md)
> **Last updated:** 2026-06-07
> **Working name:** `tokenfit`

---

## One-sentence definition

A Python library that, given a developer's query + their project corpus (md files,
code, vector DB), returns the **optimal token-budgeted context** to feed a free/small
HuggingFace model — so a 7B model with an 8k window answers as if it read everything.

## Where it sits in the stack

```
FRAMEWORK (tiny-agents / smolagents)   ← agent loop, tool use (we REUSE this)
  └─ tokenfit (this project)       ← picks the right ~2k tokens  ◀ WE BUILD THIS
       └─ MODEL (Qwen-Coder via HF key) ← raw inference, no memory
```

It is a **pre-processor**: it runs first, builds the optimal prompt, then hands it to
the framework/model. We do NOT trust a weak model to call a retrieval tool correctly.

---

## Internal pipeline

```
query
  │
  ▼
1. INGEST   load AGENTS.md / SKILL.md / docs / code → chunk
2. INDEX    embed chunks (local) + BM25 keyword index → persist (Chroma)
3. RETRIEVE hybrid (semantic + keyword) → rerank
4. BUDGET   tokenizer-aware fit to N tokens, compress/drop overflow
  │
  ▼
final prompt → HF model
```

---

## Tech defaults

| Concern        | Choice                                   | Why                              |
|----------------|------------------------------------------|----------------------------------|
| Embeddings     | `BAAI/bge-small-en-v1.5` (local)         | Free, fast, no API cost          |
| Vector store   | numpy `.npz` store (Phase 1) → Chroma later | Single-repo scale needs no DB service; Chroma is a Phase 3 swap |
| Keyword search | `rank_bm25`                              | Catches exact identifiers        |
| Tokenizer      | `transformers` AutoTokenizer of target   | Accurate per-model budgeting     |
| Inference      | `huggingface_hub.InferenceClient`        | One HF key, any provider         |
| Code chunking  | Header/function split (tree-sitter later)| Keeps chunks semantically whole  |

---

## Phased build plan

### Phase 0 — Eval harness FIRST (= the validation experiment)
- 1 test repo + AGENTS.md + ~3 SKILL.md + 10 dev questions.
- Compare **naive truncation** vs **retrieved context** answer quality.
- Becomes the permanent regression test.
- **Decision rule:** retrieved beats naive → proceed; roughly equal → pivot.

### Phase 1 — Minimal pre-processor
- ingest → index → retrieve → budget, semantic-only.
- Goal: beat naive truncation on the Phase 0 eval.

### Phase 2 — Make it good for small models
- Hybrid (BM25) + reranking + summarization fallback for oversized chunks.
- Progressive disclosure (SKILL.md style: metadata first, full content on demand).

### Phase 3 — Package + integrate
- `pip install`-able. Clean API: `ctx = pack.build(query, budget=8000)`.
- Adapters for tiny-agents / smolagents.

---

## Package shape (as built)

```
tokenfit/
├── ingest.py      # ✅ load_corpus + chunk_documents (md-by-header, py-by-def/class)
├── index.py       # ✅ bge-small embeddings → persisted numpy .npz store
├── retrieve.py    # ✅ cosine top-k semantic search  (BM25/rerank = Phase 2)
├── budget.py      # ✅ tokenizer-aware greedy fit + file citations
├── pack.py        # ✅ build_naive (baseline) + build (retrieved, cached per repo)
├── models.py      # ✅ HF InferenceClient + tokenizer for budgeting
└── eval/
    ├── harness.py # ✅ naive vs retrieved comparison runner
    └── dataset/   # ✅ questions.yaml (10 starter dev questions)
tests/
└── test_pipeline.py  # ✅ dep-free regression test (fake embedder), passes
```

## Progress log

- **2026-06-07 — Phase 0 done.** Scaffolded package + eval harness + naive baseline.
  Verified `load_corpus` (14 docs) and compilation.
- **2026-06-07 — Phase 1 done.** Implemented chunk → embed → retrieve → budget and
  wired `pack.build` with per-repo index caching. Decoupled `budget.py` from the
  inference SDK (TYPE_CHECKING) so pure-logic modules don't need `huggingface_hub`.
  Added `tests/test_pipeline.py` (fake embedder, no torch/network) — **PASSED**:
  auth query ranks the auth chunk first, budget respected, citations present.
- **Deviation from plan:** Phase 1 uses a persisted numpy `.npz` store instead of
  Chroma (lighter, no service, fine at single-repo scale). Chroma = Phase 3 swap.
- **Not yet run:** the live validation (needs `pip install -r requirements.txt`,
  `HF_TOKEN`, and a real test repo). Embeddings/model not installed in env yet.

---

## Current status / next action

- [x] Scaffold project + Phase 0 eval harness
- [x] Phase 1: implement retrieval (chunk → embed → retrieve → budget), `pack.build` wired
- [ ] `pip install -r requirements.txt` + set `HF_TOKEN`
- [ ] Point harness at a real test repo + tailor the 10 questions
- [ ] Run BOTH modes and compare: `--mode naive` vs `--mode retrieved`
- [ ] Phase 2: hybrid BM25 + rerank + summarization (only if Phase 0 validates)
