"""Thin wrapper around HuggingFace inference + a tokenizer for budgeting.

One HF token drives any model on any Inference Provider. Set it via the HF_TOKEN
(or HUGGINGFACEHUB_API_TOKEN) environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from huggingface_hub import InferenceClient

# Free / small default. Override per call or via TokenfitModel(model=...).
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


@lru_cache(maxsize=8)
def _tokenizer(model: str):
    """Load the model's tokenizer for accurate token counting.

    Falls back to a rough chars/4 estimate if transformers (or the tokenizer
    download) is unavailable, so the harness still runs offline.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        # We count tokens on whole-repo blobs (for the naive baseline); raise the
        # cap so transformers doesn't warn about sequences over the model length.
        tok.model_max_length = int(1e9)
        return tok
    except Exception:  # pragma: no cover - offline / no transformers
        return None


@dataclass
class TokenfitModel:
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        self.client = InferenceClient(model=self.model, token=_token())

    # --- token accounting -------------------------------------------------
    def count_tokens(self, text: str) -> int:
        tok = _tokenizer(self.model)
        if tok is None:
            return max(1, len(text) // 4)  # rough fallback
        return len(tok.encode(text))

    def truncate_to(self, text: str, max_tokens: int) -> str:
        """Hard-truncate text to a token budget (used by the naive baseline)."""
        tok = _tokenizer(self.model)
        if tok is None:
            return text[: max_tokens * 4]
        ids = tok.encode(text)
        if len(ids) <= max_tokens:
            return text
        return tok.decode(ids[:max_tokens])

    # --- inference --------------------------------------------------------
    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        resp = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content
