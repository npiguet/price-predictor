# May 31, 2026 — Draft agent spec deep dive

**TL;DR:** I spent the day designing the draft agent spec in detail,
working through the token representation, recency encoding, file format,
and training setup with Claude. The spec ended up in `specs/018-draft-agent/`
with several rounds of committed refinements.

The session started from a design-notes handoff I had prepared in a
separate Claude web UI conversation — a summary of a discussion about
building a Transformers-based draft agent. The document already covered
the high-level architecture decisions, but it had been written without
full project context, so some of the caveats Claude raised were easy to
dismiss. The main one: Forge's piloting quirks are the accepted objective,
so any framing of Forge's tendencies as "bias to subtract out" was
wrong from the start.

Claude wrote an initial draft of the spec, and I directed a series of
structural refinements from there. The first order of business was deciding
that the draft agent should live in its own `draft` Python module rather
than being bolted into `sealed`. That was a clean call — `draft` imports
from `sealed` and `price_predictor`, never the reverse, matching the
existing cross-package dependency discipline.

Most of the day was spent on Section 1: the token representation. I pushed
on several things that had been left vague. The PACK set turned out to need
careful handling: it should be deduped (one token per distinct card name,
since picking is an action over names and summing softmax mass across
duplicate copies would spuriously inflate their probability), while POOL
stays a multiset (a second copy genuinely changes the build), and
PASSED/TAKEN use one token per *card instance* rather than per name (so
that multiplicity — how many opponents took the same card — survives as an
open-color signal). The governing principle that fell out: dedup follows
from the role of each set.

The recency encoding also took several rounds. I initially pushed back on
having a learned lookup table for `picks_ago` rather than a plain scalar,
and Claude walked through the mechanics of why the lookup table is
preferable — a scalar forces a monotone contribution along one direction,
while a table lets different values point in different directions in the
residual stream, which is what you need to represent the wheel (a card that
comes back 8 picks later is a qualitatively different signal from one that
just hasn't been passed yet). I also got a clear explanation of how the
transformer's variable sequence length works: no weight matrix has an `L`
dimension; learned weights only operate on the `d_model` axis, and `L`
appears only in the activations. That grounded why padding per-batch
(not per-dataset) is correct.

The `picks_ago` definition went through two revisions. The first version
used "age since first observed," which created ambiguity for wheeled cards.
The cleaner definition that we landed on: `pick_ago` is picks since the
card was last in the seat's pack, frozen at the pack boundary. This gives
fresh PACK cards `pick_ago = 0`, wheeled cards `pick_ago ≈ 8` (the wheel
signal), and just-passed cards `pick_ago = 1` counting up — and a card
re-passed after a wheel resets to 1 automatically without any special
casing.

I also pushed to add a round-end PASSED→TAKEN flush: once a pack is
exhausted, any card still sitting in PASSED is provably in someone's pool,
so the entire PASSED set moves to TAKEN at each pack boundary. That turned
PASSED into a clean within-pack transient and gave TAKEN a richer role as
the complete known-taken record across rounds.

One thread I challenged was the CONTEXT token. I dropped `seat_position`
(the pod is a symmetric ring, absolute seat number carries no information),
`set_code` (format-agnostic means the model reads the format from the cards
themselves, not an ID — important for Chaos draft), and `skill` (too
artificial; degraded seats are used for opponent diversity and critic
coverage, not as a model input). The CONTEXT token ended up as just
`pack_number` and `pick_number`.

On the training setup, I questioned whether the critic should also be
trained on degraded seats, and Claude agreed: imitation must stay
full-skill-only (CE just copies, feeding random picks in would train the
model to replicate them), but the critic benefits from coverage of bad
states so it doesn't extrapolate wildly there. The mild continuation bias
this introduces is accepted and left for gen-2 on-policy training to fix.

The file format moved from a row-per-pick CSV approach to a JSONL format
with one self-contained record per draft, structured around the physical
booster as the first-class entity. The initial contents of each booster
are derivable from the ordered pick list, and most other per-record fields
(pool size, pod size, pack count, pod-relative reward) are also derivable
and were dropped. The format supports Chaos draft out of the box since
`set_code` is stored per booster rather than per draft.

Toward the end of the day I ran `/speckit.specify` to generate the formal
`spec.md` from the normative prose spec, which landed in
`specs/018-draft-agent/`. A few late clarifications came out of that pass:
the spec had hardcoded "15 cards" as pack size in several places, which
is only true for some sets; that was genericized to `3 × pack_size`. The
type one-hot as the sole differentiator of multiset membership was also
made explicit in both the formal spec and the parent normative spec.
