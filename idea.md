# Idea: HuggingFace-First Agentic Dev Framework

> **Status:** Phase 0 + Phase 1 built (retrieval engine implemented & tested). Awaiting the live validation run.
> **Last updated:** 2026-06-07
> **Build tracking:** see [`plan.md`](./plan.md) for phases, package shape, and progress log.

---

## 1. The Origin / Context

GitHub Copilot moved to **usage-based (token) billing on June 1, 2026**, replacing flat
pricing. Developers are unhappy (burning a month of credits in hours) and looking for
self-hostable / cheaper open-source alternatives.

**Opportunity:** give developers a way to get Copilot-like, project-aware help using
**free or cheap open-source models from HuggingFace**, driven by their own markdown docs.

---

## 2. The Core Idea (as originally conceived)

A **pip-installable agentic framework** that:

- Takes the user's **HuggingFace API key** (free tier or paid inference).
- Reads the user's **markdown files** — architecture, skills, conventions, etc.
- Lets **open-source models** understand and use those docs (via RAG or rule-based
  injection) to help with **development tasks**.
- Batteries-included: `pip install` → point at your docs → working dev assistant.

Essentially: **"CLAUDE.md as a product"**, HuggingFace-first, for the Copilot refugee crowd.

---

## 3. Reality Check — What Already Exists

The research showed the "assemble markdown + key + agent loop" concept is **largely solved**.

| Your concept | Already exists as |
|---|---|
| Agent loop + HF key + md context | **HuggingFace `tiny-agents`** (`pip install huggingface_hub[mcp]`, `agent.json` + `AGENTS.md`/`PROMPT.md`, run with one HF token) |
| Lightweight agent library | **smolagents** (HuggingFace), **OpenAI Agents SDK**, Google ADK |
| Architecture/context md standard | **AGENTS.md** — 60k+ projects, Linux Foundation stewarded |
| Skills md standard | **SKILL.md** — supported by Claude Code, Codex, Gemini CLI, Copilot, Cursor, Cline, OpenCode |
| HF skills ecosystem | **`huggingface/skills`** repo, Microsoft **SkillOpt** (auto-optimizes skill.md) |
| Full OSS Claude Code alternative, any provider incl. local | **OpenCode** (terminal, supports SKILL.md) |

**Verdict:** Rebuilding the framework itself = reinventing `tiny-agents`. The standards
(AGENTS.md, SKILL.md) have already won. Don't fight them — build on them.

---

## 4. The Real Remaining Gaps

What is **NOT** solved by the existing tools:

1. **Retrieval for small / limited-context models (PRIMARY GAP).**
   tiny-agents and AGENTS.md inject files *raw*. A 7B free model with ~8k context chokes
   the moment architecture md + skills + relevant code exceed the window. Frontier models
   hide this with huge context; **free models can't.** Proper RAG/retrieval tuned for
   code + markdown on small models is genuinely unsolved.

2. **Opinionated, zero-config, Python-native product.**
   OpenCode is TS/terminal-first. A truly batteries-included *Python* one could exist —
   but this is a thin / weak differentiator on its own.

3. **Free-tier reliability / routing layer.**
   Fallback and routing across free HF providers when rate-limited. Annoying and unsolved,
   but a feature, not a product.

---

## 5. The Focus Decision

**Build the retrieval layer that makes small/free models punch above their weight on a
real codebase. Use `tiny-agents` / `smolagents` as the engine — do NOT rebuild it.**

The one job:

> *Given a repo + docs far bigger than a small model's context window, feed the model
> exactly the right ~2,000 tokens so it answers as if it had read everything.*

### Why this specifically
- **Real pain:** free models fail exactly here; frontier models mask it with big context.
- **Defensible:** anyone can wrap an agent loop; good code+markdown retrieval for small
  context is real engineering.
- **Compounds the standards:** stay AGENTS.md / SKILL.md compatible — be the *retrieval
  layer* that makes them work on cheap models.

### What to explicitly NOT do
- ❌ Build a new agent runtime → use tiny-agents / smolagents.
- ❌ Invent a new md format → AGENTS.md + SKILL.md already won.
- ❌ "Support every model" → nail free-tier small models; that's the wedge.

---

## 6. Validation Experiment (do BEFORE building a product)

Run this week to confirm the gap is real:

1. Take a real medium repo + its `AGENTS.md` + ~3 `SKILL.md` files.
2. Write 10 realistic dev questions
   (e.g. "how does auth flow work?", "add an endpoint following our conventions").
3. Run a free model (`Qwen2.5-Coder-7B` on HF) two ways:
   - **Naive:** dump as much md/code as fits, truncate.
   - **Retrieved:** hand-picked relevant chunks only.
4. Compare answer quality.

**Decision rule:**
- Retrieved meaningfully beats naive → **real product, proceed.**
- Roughly equal → **gap isn't real, pivot.**

Gives a months-of-building decision in a few days.

---

## 7. Open Questions / Next Steps

- [x] Design the chunking + retrieval strategy for code + markdown → **built in Phase 1.**
- [x] Build the validation benchmark (Section 6) as runnable Python → **`tokenfit.eval.harness`.**
- [ ] Run the validation benchmark live (needs `pip install` + `HF_TOKEN` + a test repo).
- [ ] Confirm which free HF models are realistic baselines (context size, rate limits).
- [ ] Decide: standalone product vs. a retrieval plugin for tiny-agents/smolagents.

---

## Appendix: Key References

- GitHub Copilot usage-based billing — <https://github.blog/news-insights/company-news/github-copilot-is-moving-to-usage-based-billing/>
- AGENTS.md — <https://agents.md/>
- SKILL.md open standard — <https://www.agensi.io/learn/agent-skills-open-standard>
- tiny-agents — <https://huggingface.co/docs/huggingface.js/en/tiny-agents/README>
- smolagents — <https://github.com/huggingface/smolagents>
- huggingface/skills — <https://github.com/huggingface/skills>
- OpenCode skills — <https://www.agensi.io/learn/opencode-skills-guide>
