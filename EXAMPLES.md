# tokenfit — Example Runs

Real, unedited output from `tokenfit`, run with the free **`Qwen2.5-Coder-7B-Instruct`**
model on HuggingFace (8000-token budget). Token counts are the actual selected-context
size reported by tokenfit.

---

## 🏆 The headline result: retrieved vs naive on a large repo (`psf/requests`)

This is the test that proves tokenfit earns its existence. `requests` is **~150,000
tokens** of source + tests — ~19× larger than the 8000-token budget. So "naive" (just
concatenate files and truncate) is *forced* to throw away ~95% of the repo.

Run with `tokenfit eval --repo ./requests --compare`:

| # | Question | 🟥 Naive (8000 tok) | 🟩 Retrieved (~2000 tok) | Winner |
|---|----------|--------------------|--------------------------|:------:|
| 1 | connection pooling | "context doesn't provide info" | explains `Session` pooling | 🟩 |
| 2 | digest auth | quotes the changelog | cites `auth.py`, `handle_401` flow | 🟩 |
| 3 | redirects | **answered in Chinese**, changelog only | cites `resolve_redirects` (`sessions.py`) | 🟩 |
| 4 | `HTTPAdapter.send` | vague guess | `send → adapter.send → urlopen`, with code | 🟩 |
| 5 | `PreparedRequest` | "HISTORY.md doesn't say" | `prepare_request` flow | 🟩 |
| 6 | cookies | changelog generics | cites `extract_cookies_to_jar`, with code | 🟩 |
| 7 | streaming | changelog bug notes | real `iter_content` behavior | 🟩 |
| 8 | hooks | "no information in context" | `dispatch_hook`, with code snippet | 🟩 |
| 9 | `api.py` mapping | **hallucinated a fake class** | correct (create → request → close) | 🟩 |
| 10 | exceptions | solid hierarchy | also good (different angle) | ⚪ tie |

**Result: retrieved wins ~9/10 — and does it with ~4× fewer tokens.**

### Why naive collapses
Naive filled all 8000 tokens with `HISTORY.md` (the changelog) and **never reached a
single source file**. So it answered "the context doesn't provide info," quoted version
notes instead of code, once **answered in Chinese**, and once **invented a class that
doesn't exist** (`SessionRequestMethods`). Retrieval semantically skipped the changelog
and fetched the right module every time.

### The double win
Retrieval was **better AND ~4× cheaper** — ~2000 tokens vs naive's 8000. For free/small
models that's the whole point: a tighter prompt that yields a *more* accurate answer.

### Side-by-side sample (Q3 — redirects)

> **🟥 Naive (8000 tok):** *"In Request版本2.20.0及更新版本中，重定向解析和跟随的行为有所改善…"*
> (drowning in the changelog, it switched to Chinese and never found the code)
>
> **🟩 Retrieved (2617 tok):** *"…redirects are resolved through `resolve_redirects` from
> `src/requests/sessions.py`. (1) a `hist` list tracks response history; (2) the target is
> extracted via `get_redirect_target` from the `Location` header; (3) a `while url` loop
> cycles through redirects; … (7) `TooManyRedirects` is raised past `max_redirects`…"*

---

## More runs against a Godot game project

The runs below show tokenfit's behavior on a smaller repo and the kinds of questions it
suits.

---

## ✅ Example 1 — Localized "how does it work" (the sweet spot)

**Command**
```bash
tokenfit ask "How does player movement and jumping work?" --repo .\
```

**Context selected:** 4151 tokens

**Answer**
> In the player's script (`player.gd`), movement and jumping are handled as follows:
>
> **Movement:**
> Player movement is controlled based on horizontal input. When the left ("move_left") or
> right ("move_right") axis is pressed, the player moves horizontally at a constant speed
> (`SPEED`). The `direction` variable tracks whether the player is facing left (`dir < 0`)
> or right (`dir > 0`). The sprite flips horizontally accordingly using the `flip_h`
> property. If no movement keys are pressed, the player smoothly decelerates back to zero
> velocity using the `move_toward` function. The walking animation plays when the player
> has horizontal input and is on the floor, but not when shooting or throwing bombs. When
> the player stands still, the `idle` animation is played.
>
> **Jumping:**
> When the player presses the jump button ("jump") while grounded, their vertical velocity
> is set to `JUMP_VELOCITY`, causing them to jump upwards. Once airborne, the `jump`
> animation plays so the visual state matches the physical state. These controls are
> designed for both touch and keyboard interaction, with the `_hint()` method adjusting
> the textual instructions based on device capabilities.

**Why it's good:** cites the real file (`player.gd`) and real symbols (`SPEED`,
`JUMP_VELOCITY`, `direction`, `flip_h`, `move_toward`, the input actions, the animation
gating, `_hint()`). Specific and grounded — not generic LLM filler. ✅

---

## ⚠️ Example 2 — Global / aggregate question (wrong tool for the job)

**Command**
```bash
tokenfit ask "what are the assets in the game which are unused?" --repo .\
```

**Context selected:** 3576 tokens

**Answer**
> To determine which assets in the game are unused, we would need additional information
> about what files and assets are actually being used by the game code at runtime. Without
> this data, it's impossible to identify unused assets.

**Why it's "no":** "unused assets" needs a *whole-project reference graph* (every asset ×
every reference), not the top-k most relevant chunks. Also, binary assets (`.png`, `.wav`)
aren't ingested at all. The model correctly declined instead of hallucinating a list.
**Lesson:** tokenfit is for **localized** questions, not **global/aggregate** analysis.

---

## ⚠️ Example 3 — Asking about something that doesn't exist

**Command**
```bash
tokenfit ask "How does the auth flow work in the repo?" --repo .\
```

**Answer**
> There is no mention of an authentication flow in the provided project context files. The
> context focuses on how to set up and run a Godot Engine project related to RPG platformer
> game assets... it does not include details about any authentication processes.

**Why it's good:** a Godot game has no auth — and the model said so plainly instead of
inventing one. Faithful grounding on the actual repo. ✅

---

## Takeaways

| Question type | tokenfit fit | Example |
|---|---|---|
| **Localized** ("how does X work", "where is Y") | ✅ Sweet spot | player movement (Ex. 1) |
| **Absent** ("how does X work" when X isn't there) | ✅ Says so honestly | auth flow (Ex. 3) |
| **Global/aggregate** ("list all", "what's unused") | ❌ Wrong approach | unused assets (Ex. 2) |

**Status:** the pipeline produces accurate, grounded answers on a free 7B model. The
remaining validation is proving retrieval **beats naive truncation** on a repo *larger
than the token budget* — see [`plan.md`](./plan.md).
