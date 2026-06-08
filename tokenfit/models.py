"""Inference + tokenizer wrapper, with a pluggable backend.

Two backends:

- ``hf``     (default) — HuggingFace Inference Providers. One HF token drives any
               model; set it via HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN). Metered:
               hosted GPUs cost money past your small monthly free credit.
- ``ollama`` — a model running LOCALLY via Ollama. Free, unlimited, offline, and
               private (code never leaves your machine). This is the deployment
               tokenfit is really built for: tight context -> a small local model.

Pick the backend with ``TokenfitModel(backend=...)``, the ``--backend`` CLI flag,
or the ``TOKENFIT_BACKEND`` env var (set once, forget). The model id defaults
sensibly per backend, so ``--backend ollama`` works with no other config.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

# Per-backend defaults. The two Qwen ids are the same model, so budgeting with
# the HF Qwen tokenizer stays accurate even when inference runs through Ollama.
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")


def resolve_backend(explicit: str | None = None) -> str:
    """Backend precedence: explicit arg > TOKENFIT_BACKEND env > 'hf'."""
    return (explicit or os.environ.get("TOKENFIT_BACKEND") or "hf").lower()


def ollama_tags(host: str = DEFAULT_OLLAMA_HOST) -> list[str]:
    """Return the model names Ollama has pulled locally.

    Raises RuntimeError with install/run guidance if the server is unreachable.
    """
    req = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"could not reach Ollama at {host} ({e.reason}). "
            f"Install it from https://ollama.com and start the app."
        ) from e
    return [m.get("name", "") for m in data.get("models", [])]


@lru_cache(maxsize=8)
def _tokenizer(model: str):
    """Load a model's tokenizer for accurate token counting.

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
    """A model handle that counts tokens and answers chat prompts.

    `model` / `backend` / `ollama_host` all default sensibly: leave them None and
    the backend is read from TOKENFIT_BACKEND (else 'hf'), with the matching
    default model id picked automatically.
    """

    model: str | None = None
    backend: str | None = None
    ollama_host: str | None = None

    def __post_init__(self) -> None:
        self.backend = resolve_backend(self.backend)
        if self.backend == "ollama":
            self.model = self.model or DEFAULT_OLLAMA_MODEL
            self.ollama_host = (
                self.ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
            ).rstrip("/")
            # Local model ids (e.g. "qwen2.5-coder:7b") aren't HF tokenizer ids, so
            # budget against the equivalent HF tokenizer (same family by default).
            self.tokenizer_id = DEFAULT_MODEL
            self.client = None
        elif self.backend == "hf":
            from huggingface_hub import InferenceClient

            self.model = self.model or DEFAULT_MODEL
            self.tokenizer_id = self.model
            self.client = InferenceClient(model=self.model, token=_token())
        else:
            raise ValueError(
                f"unknown backend {self.backend!r}; use 'hf' (hosted) or 'ollama' (local)"
            )

    # --- token accounting -------------------------------------------------
    def count_tokens(self, text: str) -> int:
        tok = _tokenizer(self.tokenizer_id)
        if tok is None:
            return max(1, len(text) // 4)  # rough fallback
        return len(tok.encode(text))

    def truncate_to(self, text: str, max_tokens: int) -> str:
        """Hard-truncate text to a token budget (used by the naive baseline)."""
        tok = _tokenizer(self.tokenizer_id)
        if tok is None:
            return text[: max_tokens * 4]
        ids = tok.encode(text)
        if len(ids) <= max_tokens:
            return text
        return tok.decode(ids[:max_tokens])

    # --- inference --------------------------------------------------------
    def chat(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        if self.backend == "ollama":
            return self._ollama_chat(system, user, max_new_tokens)
        resp = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content

    def _ollama_chat(self, system: str, user: str, max_new_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        }
        req = urllib.request.Request(
            f"{self.ollama_host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 404:  # Ollama returns 404 when the model isn't pulled
                raise RuntimeError(
                    f"Ollama has no model '{self.model}'. Pull it first:  "
                    f"ollama pull {self.model}"
                ) from e
            raise RuntimeError(f"Ollama error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"could not reach Ollama at {self.ollama_host} ({e.reason}). "
                f"Install it from https://ollama.com, start the app, then:  "
                f"ollama pull {self.model}"
            ) from e
        return data["message"]["content"]
