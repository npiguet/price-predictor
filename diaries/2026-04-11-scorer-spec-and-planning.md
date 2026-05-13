# Saturday, April 11 2026 — Scorer spec and planning

**TL;DR:** A long evening pushing the sealed pipeline from rough ideas to
fully-planned specs. Phase 0 (training data generation) got a new spec and
three rounds of clarification; Phase 1 (deck scorer) went from spec through
implementation planning.

The evening started with a design discussion about the 2026-03-28-sealed-deck-picker.md
Phase 1 architecture before any code was written. The first question was
whether the input projection layer in the spec — a linear layer to map from
544 dimensions to some smaller d_model — was actually needed. The conclusion
was no: the 512 embedding dims and 32 deterministic dims are already
well-structured, and blending them with a projection before the model has seen
any other cards in the deck provides no benefit. So the projection was dropped
and d_model was fixed at 544 directly.

The 32 deterministic features took some iteration to pin down. Several
domain details came up that weren't captured in the original sketch. The
Wastes basic land produces colorless mana (a sixth basic land type), so the
mana-produced feature vector needed six slots, not five. Devoid is a keyword
that overrides color identity regardless of mana cost pips — any card with
devoid is colorless. Only activated ability lines (not static or triggered
ones) should be scanned for `add {X}` mana production. The `X` pip in a
mana cost is stored as an integer count rather than a boolean because some
cards have more than one. The correction about scanning only activated
abilities was particularly important: triggered abilities like "at the
beginning of your upkeep, add {G}" would inflate mana production numbers
and teach the model that the card is a land-like fixer when it isn't.

Feature normalization came up next. The issue is a scale mismatch: the 512
embedding dimensions are already at a reasonable scale from the pretrained
encoder, but the 32 deterministic features include integers that can reach
16+ (mana value) sitting next to binary flags. That mismatch distorts the
attention dot products. The decision was to store raw values in the .npz
files and compute mean/std across the corpus at training startup, then store
those stats as registered buffers on the model so they're part of the
checkpoint and inference sees the same normalization.

After that the work shifted to writing the actual specs. Spec 012 covered
Phase 0 — the supervisor/worker system that generates sealed pools, builds
decks using four construction strategies, plays Forge AI games, and appends
results to match-outcomes.txt. The deck construction strategies are weighted
40/30/20/10% from optimal down to fully random, and the weights are
hardcoded constants rather than CLI arguments. The land rebalancing uses six
basic land types with Wastes receiving only explicit {C} pips, not generic
mana costs. Set eligibility is determined by what Forge can actually generate
sealed boosters for rather than a data-driven filter or a hardcode list.
The graceful shutdown requirement was added mid-session — the supervisor
must terminate all Java worker subprocesses when killed, leaving no orphans.

Spec 013 covered Phase 1 — the Set Transformer deck scorer itself. The scope
turned out to be wider than expected because the encode-cards command needed
to be extended from 512 to 544 dimensions as a prerequisite, and the
evaluation story (playing the scorer's greedy decks against Forge's own
builder) required clarifying which side handles each step of the evaluation
loop. A key clarification was how to re-encode when upgrading from old 512-dim
.npz files: the answer was to use the existing `--clean` flag rather than
building auto-detection of stale files.

The session closed with the full planning artifacts for spec 013: research.md,
data-model.md, plan.md, CLI and file-format contracts, and a 32-task tasks.md
organized by user story.
