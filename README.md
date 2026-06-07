# tokenfit

> **Fit your whole repo into any small model's token window.**

`tokenfit` is a **context-selection pre-processor** for free / small LLMs. Point it at
your project's markdown + code, ask a question, and it returns the *most relevant* slice
of your codebase — packed to fit a tight token budget — so a 7B model with an 8k window
answers as if it read the whole repo.

[![PyPI](https://img.shields.io/badge/pypi-tokenfit-blue)](https://pypi.org/project/tokenfit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## Why

GitHub Copilot moved to usage-based token billing (June 2026), pushing developers toward
cheap open-source models on HuggingFace. But free/small models have **tiny context
windows** — dump your whole repo at them and they choke or truncate.

Existing tools (`tiny-agents`, `AGENTS.md`, `SKILL.md`) inject context *raw*. tokenfit is
the missing **retrieval layer** that makes those models punch above their weight. It's a
pre-processor: it builds the optimal prompt, then hands it to your model or agent
framework — it does **not** trust a weak model to call a retrieval tool correctly.

## How it works

```
query
  │
  ▼
1. INGEST    load AGENTS.md / SKILL.md / docs / code  →  chunk
2. INDEX     embed chunks (BAAI/bge-small, local)     →  persist
3. RETRIEVE  cosine top-k semantic search
4. BUDGET    tokenizer-aware fit to N tokens + citations
  │
  ▼
optimal prompt  →  any HuggingFace model
```

## Install

```bash
pip install tokenfit
```

Set a HuggingFace token with **"Make calls to Inference Providers"** permission:

```bash
export HF_TOKEN=hf_your_token_here      # bash
$env:HF_TOKEN = "hf_your_token_here"    # PowerShell
```

## Quickstart (CLI)

The fastest way — no Python required:

```bash
# Ask a question: tokenfit retrieves the right context AND gets the model's answer
tokenfit ask "How does the auth flow work?" --repo ./my-project

# Just print the selected context (no model call, pipe it anywhere)
tokenfit context "auth flow" --repo ./my-project

# Pre-build / refresh the index for a repo
tokenfit index --repo ./my-project --rebuild
```

Useful flags: `--budget 8000` (token budget), `--top-k 12` (chunks retrieved),
`--model Qwen/Qwen2.5-Coder-7B-Instruct` (any HF model), `--rebuild` (re-index).
Progress prints to stderr, so the answer/context on stdout stays clean for piping.

## Quickstart (Python)

```python
from tokenfit import pack
from tokenfit.models import TokenfitModel

# Select the best ~8k tokens of context for a question
context = pack.build(
    query="How does the auth flow work?",
    repo="./my-project",
    budget=8000,
)

# Feed it to any small HF model
model = TokenfitModel(model="Qwen/Qwen2.5-Coder-7B-Instruct")
answer = model.chat(
    system="You are a coding assistant for THIS project. Use only the provided context.",
    user=f"{context}\n\nQUESTION: How does the auth flow work?",
)
print(answer)
```

## Validation harness

tokenfit ships with an eval harness that compares **naive truncation** vs **retrieved
context** on your own repo — the experiment that proves the approach is worth it:

```bash
tokenfit eval --repo ./my-project --mode naive
tokenfit eval --repo ./my-project --mode retrieved
```

Each run writes a graded comparison sheet to `tokenfit/eval/results/`. Score the answers
1–5 and compare. Edit `tokenfit/eval/dataset/questions.yaml` to fit your project.

## Roadmap

- [x] **Phase 0** — eval harness + naive baseline
- [x] **Phase 1** — semantic retrieval (chunk → embed → retrieve → budget)
- [ ] **Phase 2** — hybrid BM25 + rerank + summarization for oversized chunks
- [ ] **Phase 3** — `tiny-agents` / `smolagents` adapters, optional Chroma backend

See [`idea.md`](./idea.md) for the rationale and [`plan.md`](./plan.md) for the full plan.

## Development

```bash
git clone https://github.com/shubham10divakar/tokenfit
cd tokenfit
pip install -e ".[dev]"
python -m tests.test_pipeline   # dep-free regression test
```

## License

MIT — see [LICENSE](./LICENSE).
