# April 20, 2026 — Self-play spec and diary tooling

**TL;DR:** I designed the Python-to-Java bridge for Phase 3 self-play
training and worked through the full speckit workflow to produce a
spec, clarifications, and a planning document for it. I also built
the `/diary` command and iterated on its format through the day.

The morning started with something unrelated to the sealed pipeline:
I realized that all the context about decisions made, experiments run,
and courses corrected existed only in session transcripts that would
eventually be deleted. The git history records *what* changed; the
transcripts hold *why*. I asked Claude to build a `/diary` slash command
that would read the JSONL transcripts for a given date, extract the
narrative content via a Python pre-processing step, and produce a
first-person prose entry. The first attempt used Haiku and produced
something that read like a commit message with bullet points — not
what I wanted. I switched to Sonnet and tightened the prompt to
forbid headers, ban enumerated findings, and not pad thin days with
filler. Over several iterations I also added: 80-character line
wrapping, a TL;DR paragraph at the top, a per-file title with a
five-word summary, and explicit rules against inferring emotions or
judgements that I hadn't actually expressed. Claude also ran the
command in parallel across all 18 transcript-covered days before April
19, then reconstructed 19 additional entries from the older
`history.jsonl` prompt log — which is how I learned that Claude Code
silently deletes session transcripts after 30 days.

The bigger work of the day was designing the self-play loop for
feature 014. I asked Claude to read `sealed-deck-picker.md` and find
what was missing from the point of view of someone trying to actually
run Phase 3. The analysis came back with a clear structural problem:
the existing `match-outcomes` flow is entirely self-contained in Java
— pool generation, deck building, and game simulation all happen
within a single JVM. There was no mechanism for the Python scorer to
inject its own decks into a Java-played match. The `ValidationWorkerMain`
class could accept pre-built decks, but it was wired only to
`evaluate-scorer`, not to `match-outcomes`.

My proposed solution was to use a flat text file as the interface
between the two languages: Python generates pools, builds decks with
`GreedyDeckBuilder`, and writes them to `generated-decks.txt` (one
deck per line). Java reads that file and, when `--generated-decks-path`
is specified, picks deck A from it and generates deck B through one
of five methods — the fifth being another random deck from the same
file. Claude flagged an important cross-set problem I hadn't fully
thought through: without a set-code prefix on each line, deck A and
deck B's pool could come from different sets with wildly different
power levels, and the set-level mismatch would dominate the training
signal over any actual deck-building quality difference. The fix was
to include the set code on each line and have Java filter method-5
picks to the same set as deck A. For methods 1–4, Java generates a
fresh pool from that same set.

A second domain insight came up when I considered whether to allow a
deck to play against an identical copy of itself. Claude's first
recommendation was to allow it — at ~46 decks per set across 10,000
generated decks, the chance of picking the same line is only ~2%.
I pushed back: a best-of-N between two identical decks produces a
winner and a loser entirely from RNG, with no information about which
deck is better. That is a confusing training signal, not a neutral
one, so I decided to exclude mirror matches and re-roll instead.

The same-set constraint also applied to `evaluate-scorer`, which had
"RVR" (Ravnica Remastered) hardcoded as the evaluation set. I decided
that evaluation should use a random sealed-legal set by default, with
an optional `--set` flag for targeted testing.

Once the design was settled, I updated `sealed-deck-picker.md` with
the new Phase 3 section, then ran the speckit workflow: `speckit.specify`
to create `specs/014-self-play-refinement/spec.md`, followed by two
`speckit.clarify` passes that resolved questions about default output
paths, method weight definitions, and the mirror-match exclusion rule,
and finally `speckit.plan` to produce `research.md`, `plan.md`,
`data-model.md`, and `quickstart.md`. One notable decision from the
planning phase: backward compatibility for the old `pools.txt` format
(which lacked the `SET_CODE;` prefix) was explicitly dropped, since
pool files are cheap to regenerate and were mostly unused anyway.

I also ran `/speckit.constitution` to add Principle VII (Codebase-Aware
Planning), which requires any planning phase to include a deliberate
survey of existing code before producing a design — specifically to
prevent pigeonholing, duplication, and domain concept rot. The
survey findings go into `research.md` under a `## Codebase Survey`
section rather than cluttering `plan.md`.
