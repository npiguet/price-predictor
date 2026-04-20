# Wednesday, 9 April 2026 — Match outcome spec clarifications

**TL;DR:** The whole day went into specifying feature 012 — the sealed
training data generation pipeline. I created the spec from the
sealed-deck-picker document and then ran three rounds of `/speckit.clarify`
to pin down every ambiguity before planning.

The day started with `/speckit.specify` pulling "Phase 0 — Training dataset
generation" out of the old monolithic sealed-deck-picker spec and turning it
into its own feature (012-sealed-training-data on a fresh branch). The
generated spec was already fairly complete, but two gaps surfaced during my
manual review: the nonland-for-nonland constraint on swap methods 2 and 3
was missing, and method 4's rule about targeting exactly 23 non-land cards
(excluding basics) was glossed over. I flagged those explicitly and Claude
fixed them.

The more interesting correction came when Claude asked about the "(including
colorless)" note in the land rebalancing rule. Its instinct was to treat
generic mana costs ({1}, {2}) the same as explicit colorless pips when
deciding how many Wastes to include. I pushed back: there is a sixth basic
land — Wastes — and only explicit {C} pips on cards should count toward it.
Generic costs can be paid by any land and carry no color signal at all.
That's standard sealed practice and it matters for any future deck builder
that consumes this output.

The clarification rounds also settled a handful of structural questions. For
set eligibility the decision was to let Forge itself determine which sets are
legal for sealed pool generation, rather than maintaining a hardcoded list or
parsing AllPrintings.json booster metadata — if Forge can generate a pool,
the set is in. Cross-set matches were ruled out: both players always come
from the same set, which keeps the power-level signal clean. The append-only
behavior of match-outcomes.txt was confirmed explicitly, since a 100k-match
accumulation is meant to span multiple multi-hour sessions.

Two output-format questions resolved in the direction of keeping the file
lean. I chose not to add a set code field (field 5) and not to embed deck
construction method IDs in the records. The method distribution will be
trusted to the random weights and validated via code rather than logged per
match. The training model only needs deck composition and outcome — not the
provenance of how either deck was built.

One genuine piece of domain knowledge I added manually: when a Forge AI game
runs too long, the worker JVM crashes rather than hanging indefinitely. That
means no explicit game timeout is needed in the spec — the existing
crash-restart loop already handles it. I had Claude remove the "stuck game"
edge case and save that fact to memory so it doesn't get re-asked later.

I also added graceful shutdown to the spec after noticing it was missing:
when the supervisor receives Ctrl-C it must terminate all worker subprocesses
before exiting, leaving no orphaned JVMs behind.

By end of day the spec had gone through eleven answered clarification
questions across three sessions, all categories were Clear, and the next step
was `/speckit.plan`.
