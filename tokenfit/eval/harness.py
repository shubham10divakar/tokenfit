"""Phase 0 eval harness — the validation experiment.

Runs a set of dev questions against a target repo two ways and saves the answers
side by side so you can judge whether retrieval beats naive truncation.

    naive     : concatenate priority-sorted files, hard-truncate to budget
    retrieved : tokenfit's selected context  (Phase 1 — not yet wired)

Usage:
    # 1. set your key
    $env:HF_TOKEN = "hf_..."          # PowerShell
    # 2. run the naive baseline against a repo
    python -m tokenfit.eval.harness --repo ../some-repo --mode naive

Output: tokenfit/eval/results/<timestamp>.md  (a human-graded comparison sheet)
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import yaml

from tokenfit import pack
from tokenfit.models import TokenfitModel

HERE = Path(__file__).parent
DEFAULT_QUESTIONS = HERE / "dataset" / "questions.yaml"
RESULTS_DIR = HERE / "results"

SYSTEM_PROMPT = (
    "You are a coding assistant for THIS project. Use ONLY the provided project "
    "context to answer. If the context is insufficient, say so explicitly rather "
    "than guessing."
)


def load_questions(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(data.get("questions", []))


def run(repo: str, mode: str, budget: int, model_name: str, questions_path: Path) -> Path:
    model = TokenfitModel(model=model_name)
    questions = load_questions(questions_path)
    if not questions:
        raise SystemExit(f"No questions found in {questions_path}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}-{mode}.md"

    lines = [
        f"# Eval run — {mode}",
        "",
        f"- repo: `{repo}`",
        f"- model: `{model_name}`",
        f"- budget: {budget} tokens",
        f"- timestamp: {stamp}",
        "",
        "Grade each answer 1-5 for correctness/usefulness in the `Score` blanks.",
        "",
        "---",
        "",
    ]

    for i, q in enumerate(questions, 1):
        if mode == "naive":
            context = pack.build_naive(q, repo, model, budget)
        else:
            context = pack.build(q, repo, budget)  # Phase 1

        user = f"PROJECT CONTEXT:\n{context}\n\n---\n\nQUESTION: {q}"
        ctx_tokens = model.count_tokens(context)
        try:
            answer = model.chat(SYSTEM_PROMPT, user)
        except Exception as e:  # surface API/key errors per-question, keep going
            answer = f"[ERROR calling model: {e}]"

        lines += [
            f"## Q{i}. {q}",
            f"_context tokens: {ctx_tokens}_",
            "",
            "**Answer:**",
            "",
            answer,
            "",
            "**Score (1-5):** ____   **Notes:** ",
            "",
            "---",
            "",
        ]
        print(f"[{i}/{len(questions)}] done ({ctx_tokens} ctx tokens)")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="tokenfit Phase 0 eval harness")
    p.add_argument("--repo", required=True, help="path to the test repo")
    p.add_argument("--mode", choices=["naive", "retrieved"], default="naive")
    p.add_argument("--budget", type=int, default=8000, help="context token budget")
    p.add_argument("--model", default=TokenfitModel.model)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = p.parse_args()
    run(args.repo, args.mode, args.budget, args.model, args.questions)


if __name__ == "__main__":
    main()
