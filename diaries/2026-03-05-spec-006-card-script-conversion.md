# March 5, 2026 — Spec 006: Card Script Conversion

**TL;DR:** I spent the evening writing and refining the feature spec for
converting Forge card scripts into an LLM-friendly text format (spec 006),
then drove the planning phase through several design decisions about how the
Java conversion should work. By the end I had a plan, a data model, and a
quickstart doc — all committed or nearly so.

The session opened with committing some prior work, then invoking
`/speckit.clarify` on spec 006. A quick clarification surfaced something worth
correcting: FR-014 had called for substituting CARDNAME placeholders with the
actual card name, but I rejected that. The placeholder should stay as-is — it
is semantically meaningful to the model as a self-reference token, and
replacing it would destroy that signal. I also clarified that the batch output
should include every card type without exception: tokens, emblems, dungeons,
and the rest. My reasoning was that I would rather filter them out at training
time, where I have full control, than encode exclusion decisions into the
converter. Both changes landed in the second commit of the day.

After a `/clear`, I kicked off `/speckit.plan`. During the planning phase I
raised a point that shaped the whole architecture: printing and formatting
should happen inside the Java stub rather than in Python. The main motivation
was reuse — Forge already has classes that know how to display costs, keywords,
and ability text, and reimplementing that in Python would be both more code and
less accurate. Script parsing is deceptively hard, especially for ability costs,
which can be almost anything and never appear in a human-readable form in the
raw `.txt` scripts. Leaning on Forge's own `Cost`, `Keyword`, and
`CardRules.Reader` classes was the obvious call once I named the problem
clearly.

Later in the evening I noticed the planned data model was missing the type of
multi-face card. Split, flip, MDFC, adventure — these are meaningfully
different layouts, and the model might benefit from knowing which it is dealing
with. I flagged this and Claude added a `layout` field to `MultiCard`, derived
from Forge's `CardSplitType`. I preferred the second proposed representation
for the output format: a bare `layout: transform` line before the first face,
with nothing for single-face cards. It's clean and unambiguous.

I also locked down a casing rule I wanted to be explicit about: all symbols
wrapped in braces — mana, energy, tap, untap, and anything else rendered as
`{something}` — should stay uppercase throughout the output. Lowercasing
everything except a short allowlist was the right default, but I wanted the
symbol rule stated clearly rather than inferred. After that, I asked Claude to
propagate the design changes back into `plan.md`, `quickstart.md`, and
`research.md` for consistency, and committed the result.

The day's two git commits captured the spec rewrite itself and a targeted
clarification pass. The planning artifacts — plan, data model, quickstart —
reflect the architectural choices made during the evening session and set the
stage for task generation and implementation.

*Note: reconstructed from prompt history + git log; full session transcripts
were auto-deleted after 30 days.*
