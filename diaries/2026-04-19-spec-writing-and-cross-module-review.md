# April 19, 2026 — Spec writing and cross-module review

**TL;DR:** I reorganized the sealed-deck-picker spec and wrote
the missing price-predictor overview doc, then ran a cross-module
code review that turned into a substantial refactor.

The day started with the sealed-deck-picker spec. I thought the
document's organization was off — disconnected sections that
should belong together. Claude read it and diagnosed the same
problems I felt: the Card Representation section was a grab-bag
of unrelated concerns, the evaluation section was stranded after
Phase 3 instead of being part of the training loop, and the
reader hit implementation tables before getting any sense of
what the phases were. We worked out a new structure together.
One thing that wasn't in the document at all was self-play: I
noticed the concept existed as scattered fragments (a "Future
improvement" aside, some iteration language in the completion
criteria) but was never described as a coherent loop. I decided
it deserved its own phase — Phase 3, with the old Phase 3
renumbered to 4. The self-play phase would add a fifth deck-
building method (the scorer-guided builder from Phase 2) to the
training data generation pipeline, and the external baseline
evaluation and training completion criteria would move there as
the decision point and exit condition for that loop.

For the evaluation section, I confirmed it's designed to run
periodically during training but manually — not automated, not
periodic-during-an-epoch. It's a decision point: do we need more
epochs? Different data? Time to stop? So it naturally anchors
the self-play loop's exit condition.

Next I asked Claude to create a human-readable overview document
for the price-predictor module, analogous to sealed-deck-picker.
The first draft had errors in the property-line and ability-line
prefix tables. Claude had written them from the parser's regex
patterns, which are more permissive than what the Java converter
actually produces. Several prefixes listed (`etb:`, `ltb:`,
`mana[N]:`, `mode[N]:`) don't exist in any real converted card
file, and two format descriptions were wrong (`chapter X:` should
be `chapter:` with the roman numeral as a value, not part of the
key; `level X-Y:` should be `level[N]:` with bracket numbering).
I caught these and asked Claude to verify against the actual
corpus, which confirmed the issues. The distinction matters
because the parser regex handles patterns the converter never
emits — they're either planned but unimplemented or defensive
over-matching. The corrected document went in.

The bigger work of the day was a cross-module code review. I
asked Claude to look at both Python modules and find
opportunities for deduplication and re-use of domain concepts.
The review surfaced 10 findings. After I pointed out that the
review hadn't adequately distinguished between the two card text
formats — raw Forge card scripts vs. the converted tokenizer-
ready format — Claude re-verified each finding against that
distinction. None of the "consolidate these parsers" suggestions
accidentally conflated the two formats, but the clarification
made the framing sharper.

The conversation then surfaced a deeper structural question:
should that two-format distinction be materialized in the type
system? I asked whether it would make sense to do that, and
Claude proposed two options — `NewType` aliases (lightweight,
zero runtime cost) or frozen dataclasses (more ceremony but
enables format-specific methods). The recommendation was a
`ConvertedCardText` dataclass for the converted format, since
it would be a natural home for methods like
`without_name_line()`, `mana_cost_line()`, and
`activated_lines()` that were scattered across several modules,
and a `ForgeCardScript` NewType for the raw format since only
`check_convert` uses it on the Python side. I agreed, and this
became Finding #0 — the architectural groundwork that would
unblock findings #2, #3, and #4.

After planning, Claude executed all 12 commits of the plan:
renaming the `TrainingExample` name clash in the sealed
infrastructure, collapsing the three separate basic-land name
sets into one, introducing `ConvertedCardText` and propagating
it through both modules, unifying the three P/T parsing
implementations into a single `parse_combat_stat` helper, adding
`is_land()`, `has_devoid()`, and `is_colorless()` domain methods
to `Card`, extracting mana-production parsing into the shared
parser, sharing torch checkpoint serialization between the
transformer and scorer stores, lifting `add_dataclass_arg` out
of the sealed CLI into a shared infrastructure module, wrapping
one-shot Java subprocess invocations in a `run_forge_worker`
helper, moving `sanitize_card_name` to a shared location with an
injectable corrections parameter, adding `make_printing_data` and
`make_card` factory helpers in conftest, and retying
`TransformerTrainingDataset` to accept `list[TrainingSample]`
directly instead of a list of raw tuples.

After the implementation, lint across the whole codebase flagged
pre-existing issues from earlier sessions. I told Claude to fix
everything, noting that anything coming up had been caused by its
own edits, whether in this session or a prior one. That included
an orphaned integration test referencing a `run_eval` function
that had been deleted in an earlier commit without updating the
test.
