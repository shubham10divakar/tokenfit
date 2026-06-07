"""Command-line interface for tokenfit.

    tokenfit ask "How does the auth flow work?" --repo ./my-project
    tokenfit context "auth flow" --repo ./my-project          # print context only
    tokenfit index --repo ./my-project --rebuild              # (re)build the index
    tokenfit eval --repo ./my-project --mode retrieved        # naive-vs-retrieved sheet

Heavy imports (HuggingFace, embeddings) are deferred into each command so that
`tokenfit --help` and argument parsing work even before the optional deps install.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Duplicated as a literal (not imported from .models) to keep --help dependency-free.
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

DEFAULT_SYSTEM = (
    "You are a coding assistant for THIS project. Use ONLY the provided project "
    "context to answer. If the context is insufficient, say so explicitly rather "
    "than guessing."
)


def _status(msg: str) -> None:
    """Progress goes to stderr so stdout stays clean (pipeable answer/context)."""
    print(f"[tokenfit] {msg}", file=sys.stderr)


# --- commands -------------------------------------------------------------
def cmd_context(args: argparse.Namespace) -> None:
    from tokenfit import pack
    from tokenfit.models import TokenfitModel

    model = TokenfitModel(model=args.model)
    ctx = pack.build(
        args.query, repo=args.repo, budget=args.budget,
        model=model, top_k=args.top_k, rebuild=args.rebuild,
    )
    _status(f"{model.count_tokens(ctx)} tokens selected from {args.repo}")
    print(ctx)


def cmd_ask(args: argparse.Namespace) -> None:
    from tokenfit import pack
    from tokenfit.models import TokenfitModel

    model = TokenfitModel(model=args.model)
    _status(f"building context (budget {args.budget}) from {args.repo} ...")
    ctx = pack.build(
        args.query, repo=args.repo, budget=args.budget,
        model=model, top_k=args.top_k, rebuild=args.rebuild,
    )
    _status(f"{model.count_tokens(ctx)} ctx tokens; asking {args.model} ...")
    user = f"PROJECT CONTEXT:\n{ctx}\n\n---\n\nQUESTION: {args.query}"
    print(model.chat(args.system, user, max_new_tokens=args.max_new_tokens))


def cmd_index(args: argparse.Namespace) -> None:
    from tokenfit import index as _index
    from tokenfit import pack

    persist = pack.ensure_index(args.repo, rebuild=args.rebuild)
    _, chunks = _index.load_index(persist)
    _status(f"index ready ({len(chunks)} chunks) at {persist}")


def cmd_eval(args: argparse.Namespace) -> None:
    from tokenfit.eval.harness import DEFAULT_QUESTIONS, run

    questions = Path(args.questions) if args.questions else DEFAULT_QUESTIONS
    run(args.repo, args.mode, args.budget, args.model, questions)


# --- parser ---------------------------------------------------------------
def _add_retrieval_opts(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--repo", default=".", help="project root (default: current dir)")
    sp.add_argument("--budget", type=int, default=8000, help="context token budget")
    sp.add_argument("--top-k", type=int, default=12, dest="top_k", help="chunks to retrieve")
    sp.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id")
    sp.add_argument("--rebuild", action="store_true", help="force re-index before running")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenfit",
        description="Fit your whole repo into a small model's token window.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="retrieve context AND get a model answer")
    ask.add_argument("query", help="your question about the project")
    _add_retrieval_opts(ask)
    ask.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt override")
    ask.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    ask.set_defaults(func=cmd_ask)

    ctx = sub.add_parser("context", help="print selected context only (no model call)")
    ctx.add_argument("query", help="query to retrieve context for")
    _add_retrieval_opts(ctx)
    ctx.set_defaults(func=cmd_context)

    idx = sub.add_parser("index", help="build or refresh the index for a repo")
    idx.add_argument("--repo", default=".", help="project root (default: current dir)")
    idx.add_argument("--rebuild", action="store_true", help="force a full rebuild")
    idx.set_defaults(func=cmd_index)

    ev = sub.add_parser("eval", help="run the naive-vs-retrieved eval harness")
    ev.add_argument("--repo", required=True, help="path to the test repo")
    ev.add_argument("--mode", choices=["naive", "retrieved"], default="retrieved")
    ev.add_argument("--budget", type=int, default=8000)
    ev.add_argument("--model", default=DEFAULT_MODEL)
    ev.add_argument("--questions", default=None, help="path to a questions.yaml")
    ev.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"[tokenfit] error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
