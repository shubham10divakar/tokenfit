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
- **2026-06-07 — packaged & shipped.** Renamed to `tokenfit`, added `setup.py` +
  `pyproject.toml` + MIT license, built wheel/sdist, pushed to GitHub. PyPI name reserved.
- **2026-06-07 — CLI added (v0.2.0).** `tokenfit` command with subcommands
  `ask` (context → model → answer), `context` (print context only), `index`, `eval`.
  Lazy imports so `tokenfit --help` works before heavy deps install. (Phase 3 start.)
- **2026-06-07 — hardening (v0.2.1–0.2.3).** `tokenfit auth` (token check + `--ping`);
  broadened file-type coverage to ~20 languages + `--include` flag (default globs were
  Python/markdown-only, so non-Python repos indexed almost nothing); skip vendored/build
  dirs, lockfiles, oversized files; glob set folded into the cache key; `--version` flag;
  silenced the Windows symlink warning.

### ✅ First live validation — 2026-06-07 (Godot game repo, free Qwen2.5-Coder-7B)
- **It works end to end.** A free 7B model gave an accurate, code-grounded answer about
  the project via one CLI command. See [`EXAMPLES.md`](./EXAMPLES.md) for the full Q&A log.
- **Best result:** "How does player movement and jumping work?" → cited the real file
  (`player.gd`) and real symbols (`SPEED`, `JUMP_VELOCITY`, `flip_h`, `move_toward`,
  input actions, animation gating). 4151 ctx tokens. Genuinely useful, not generic.
- **Proves:** the pipeline produces high-quality grounded answers on a free model.
- **Does NOT yet prove:** that retrieval *beats naive truncation*. This repo's code fits
  inside the 8k budget, so naive would also work. The retrieval advantage only shows on
  repos **larger than the budget** — that comparison is the next validation step.
- **Scoping insight:** tokenfit fits **localized** questions ("how does X work / where is
  Y") well, but not **global/aggregate** ones ("list all unused assets") — those need a
  whole-project reference graph, a different mechanism (out of scope for the retrieval core).

### 🏆 DIFFERENTIATOR PROVEN — 2026-06-07 (`psf/requests`, ~150k tokens, free 7B)
- Ran `tokenfit eval --repo ./requests --compare` (10 questions, 8k budget).
- **Retrieved won ~9/10** vs naive truncation, using **~2000 tokens vs naive's 8000**
  (~4× cheaper). Full transcripts in [`EXAMPLES.md`](./EXAMPLES.md).
- Naive filled the entire budget with `HISTORY.md` and never reached source code →
  "context doesn't provide info", quoted the changelog, once answered in Chinese, once
  hallucinated a non-existent class. Retrieved cited the right module every time.
- **This is the "proceed" signal.** The core thesis (retrieval makes free/small models
  punch above their weight on big repos) is validated: better answers AND lower cost.
- **v0.2.5:** results now write to `./tokenfit-results/` (was site-packages); silenced
  the long-sequence tokenizer warning.

---

## Current status / next action

- [x] Scaffold project + Phase 0 eval harness
- [x] Phase 1: implement retrieval (chunk → embed → retrieve → budget), `pack.build` wired
- [x] Package + CLI + hardening (v0.2.3 on GitHub)
- [x] First live run: free 7B gives accurate grounded answers ([`EXAMPLES.md`](./EXAMPLES.md))
- [x] **Differentiator PROVEN:** retrieved beat naive ~9/10 on `psf/requests` (~150k tokens),
      at ~4× fewer tokens. Core thesis validated.
- [ ] Phase 2: hybrid BM25 + rerank + summarization (now polish, not necessity)
- [ ] Consider: first PyPI release (badges go live), since the thesis holds
