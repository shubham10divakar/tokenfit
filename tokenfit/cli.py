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

from tokenfit import __version__

# Duplicated as literals (not imported from .models) to keep --help dependency-free.
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"

DEFAULT_SYSTEM = (
    "You are a coding assistant for THIS project. Use ONLY the provided project "
    "context to answer. If the context is insufficient, say so explicitly rather "
    "than guessing."
)


def _status(msg: str) -> None:
    """Progress goes to stderr so stdout stays clean (pipeable answer/context)."""
    print(f"[tokenfit] {msg}", file=sys.stderr)


def _globs(args: argparse.Namespace) -> tuple[str, ...]:
    """Default file types plus any --include patterns (union, de-duped)."""
    from tokenfit.ingest import DEFAULT_GLOBS

    inc = tuple(getattr(args, "include", None) or ())
    return tuple(dict.fromkeys(DEFAULT_GLOBS + inc)) if inc else DEFAULT_GLOBS


# --- commands -------------------------------------------------------------
def cmd_context(args: argparse.Namespace) -> None:
    from tokenfit import pack

    model = _model(args)
    ctx = pack.build(
        args.query, repo=args.repo, budget=args.budget,
        model=model, top_k=args.top_k, rebuild=args.rebuild, globs=_globs(args),
    )
    _status(f"{model.count_tokens(ctx)} tokens selected from {args.repo}")
    print(ctx)


def cmd_ask(args: argparse.Namespace) -> None:
    from tokenfit import pack

    model = _model(args)
    _status(f"building context (budget {args.budget}) from {args.repo} ...")
    ctx = pack.build(
        args.query, repo=args.repo, budget=args.budget,
        model=model, top_k=args.top_k, rebuild=args.rebuild, globs=_globs(args),
    )
    _status(f"{model.count_tokens(ctx)} ctx tokens; asking {model.model} "
            f"via {model.backend} ...")
    user = f"PROJECT CONTEXT:\n{ctx}\n\n---\n\nQUESTION: {args.query}"
    print(model.chat(args.system, user, max_new_tokens=args.max_new_tokens))


def cmd_index(args: argparse.Namespace) -> None:
    from tokenfit import index as _index
    from tokenfit import pack

    persist = pack.ensure_index(args.repo, rebuild=args.rebuild, globs=_globs(args))
    _, chunks = _index.load_index(persist)
    _status(f"index ready ({len(chunks)} chunks) at {persist}")


def cmd_eval(args: argparse.Namespace) -> None:
    from tokenfit.eval.harness import DEFAULT_QUESTIONS, run, run_compare

    questions = Path(args.questions) if args.questions else DEFAULT_QUESTIONS
    if args.compare:
        run_compare(args.repo, args.budget, args.model, questions,
                    backend=args.backend, ollama_host=args.ollama_host)
    else:
        run(args.repo, args.mode, args.budget, args.model, questions,
            backend=args.backend, ollama_host=args.ollama_host)


def cmd_auth(args: argparse.Namespace) -> None:
    """Verify the chosen backend is ready before running `ask`/`eval`.

    hf     -> check the HuggingFace token is set and valid.
    ollama -> check the local server is up and the model is pulled.
    """
    import os

    from tokenfit.models import resolve_backend

    if resolve_backend(args.backend) == "ollama":
        _auth_ollama(args)
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        _status("no token found in HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")
        _status("get one at https://huggingface.co/settings/tokens "
                '(enable "Make calls to Inference Providers")')
        _status('then:  $env:HF_TOKEN = "hf_..."')
        sys.exit(1)

    src = "HF_TOKEN" if os.environ.get("HF_TOKEN") else "HUGGINGFACEHUB_API_TOKEN"
    masked = f"{token[:6]}…{token[-4:]}" if len(token) > 12 else "set"
    _status(f"token found in {src} ({masked})")

    from huggingface_hub import HfApi

    try:
        info = HfApi().whoami(token=token)
    except Exception as e:  # invalid / revoked / network
        _status(f"token is INVALID or could not be verified: {e}")
        sys.exit(1)

    name = info.get("fullname") or info.get("name") or "unknown"
    role = (info.get("auth", {}).get("accessToken", {}).get("role")) or "unknown"
    print(f"authenticated as: {name} (token role: {role})")

    if not args.ping:
        _status("identity OK. Re-run with --ping to verify inference access.")
        return

    model = _model(args)
    _status(f"pinging inference on {model.model} ...")
    try:
        model.chat("ping", "Reply with: ok", max_new_tokens=1)
    except Exception as e:
        _status(f"inference FAILED on {model.model}: {e}")
        _status('token likely lacks the "Make calls to Inference Providers" permission.')
        sys.exit(1)
    print(f"inference OK on {model.model}")


def _auth_ollama(args: argparse.Namespace) -> None:
    """Check the local Ollama server is up and the target model is pulled."""
    from tokenfit.models import DEFAULT_OLLAMA_MODEL, ollama_tags

    host = args.ollama_host or "http://localhost:11434"
    want = args.model or DEFAULT_OLLAMA_MODEL
    try:
        tags = ollama_tags(host)
    except RuntimeError as e:
        _status(str(e))
        sys.exit(1)

    _status(f"Ollama is up at {host} ({len(tags)} model(s) installed)")
    # Match with or without an explicit ":tag" (ollama lists "name:tag").
    if want in tags or any(t.split(":")[0] == want.split(":")[0] for t in tags):
        print(f"ready: '{want}' is available for local inference")
    else:
        _status(f"model '{want}' is NOT pulled. Get it with:  ollama pull {want}")
        if tags:
            _status(f"installed: {', '.join(tags)}")
        sys.exit(1)


# --- parser ---------------------------------------------------------------
def _add_model_opts(sp: argparse.ArgumentParser) -> None:
    """Backend + model selection, shared by commands that call a model."""
    sp.add_argument("--backend", choices=["hf", "ollama"], default=None,
                    help="hf = hosted HuggingFace (default); ollama = local & free. "
                         "Can also be set once via the TOKENFIT_BACKEND env var.")
    sp.add_argument("--model", default=None,
                    help=f"model id (default: {DEFAULT_MODEL} for hf, "
                         f"{DEFAULT_OLLAMA_MODEL} for ollama)")
    sp.add_argument("--ollama-host", default=None, dest="ollama_host",
                    help="Ollama base URL (default: http://localhost:11434)")


def _model(args: argparse.Namespace):
    """Build a TokenfitModel from the shared backend/model flags."""
    from tokenfit.models import TokenfitModel

    return TokenfitModel(
        model=args.model,
        backend=getattr(args, "backend", None),
        ollama_host=getattr(args, "ollama_host", None),
    )


def _add_retrieval_opts(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--repo", default=".", help="project root (default: current dir)")
    sp.add_argument("--budget", type=int, default=8000, help="context token budget")
    sp.add_argument("--top-k", type=int, default=12, dest="top_k", help="chunks to retrieve")
    sp.add_argument("--rebuild", action="store_true", help="force re-index before running")
    sp.add_argument("--include", nargs="+", metavar="GLOB",
                    help="extra file globs to index, e.g. --include '*.gd' '*.cs'")
    _add_model_opts(sp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenfit",
        description="Fit your whole repo into a small model's token window.",
    )
    p.add_argument("--version", action="version", version=f"tokenfit {__version__}")
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
    idx.add_argument("--include", nargs="+", metavar="GLOB",
                     help="extra file globs to index, e.g. --include '*.gd' '*.cs'")
    idx.set_defaults(func=cmd_index)

    ev = sub.add_parser("eval", help="run the naive-vs-retrieved eval harness")
    ev.add_argument("--repo", required=True, help="path to the test repo")
    ev.add_argument("--mode", choices=["naive", "retrieved"], default="retrieved")
    ev.add_argument("--compare", action="store_true",
                    help="run naive AND retrieved into one side-by-side sheet")
    ev.add_argument("--budget", type=int, default=8000)
    ev.add_argument("--questions", default=None, help="path to a questions.yaml")
    _add_model_opts(ev)
    ev.set_defaults(func=cmd_eval)

    au = sub.add_parser("auth", help="check the chosen backend is ready (hf token / local ollama)")
    au.add_argument("--ping", action="store_true",
                    help="(hf) also make a 1-token inference call to verify access")
    _add_model_opts(au)
    au.set_defaults(func=cmd_auth)

    return p


def main(argv: list[str] | None = None) -> None:
    import os

    # Quiet the noisy Windows symlink advisory from huggingface_hub downloads.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"[tokenfit] error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
