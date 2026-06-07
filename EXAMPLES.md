# tokenfit — Example Runs

Real output from `tokenfit ask`, run against a **Godot game project** using the free
**`Qwen2.5-Coder-7B-Instruct`** model on HuggingFace (8000-token budget).

> These are unedited results from the first live validation on 2026-06-07. Token counts
> are the actual selected-context size reported by tokenfit.

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
