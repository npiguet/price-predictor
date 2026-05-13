# May 1, 2026 — Build-decks SA tuning overhaul

**TL;DR:** I spent the day profiling and fixing a severe performance problem in the
sealed deck builder, then tuned the simulated annealing parameters based on actual
12-pool experiments, and finished by landing two new features: color-pair seeded
restarts and side-A/side-B deck file flags for self-play match-outcomes.

The day started with a concrete complaint: `build-decks` with `--sa-temperature 0.8`
and `--restarts 4` was taking upwards of five minutes per deck. Claude did a code
review and found the root cause quickly — `_score_many` was making roughly 900,000
separate tiny GPU kernel launches per deck because each candidate was built one
`torch.tensor()` call at a time. The fix was to build the index and mask matrices
on CPU as NumPy arrays and transfer them in a single operation. That one change,
combined with a few related fixes (pre-normalizing the pool once, removing a
redundant `_score_one` bootstrap, halving the GPU-to-CPU sync count per iteration,
and adding an early-stop once simulated annealing cools into greedy territory),
dropped the wall-clock from about 29 minutes to about 7m30s — a 4x speedup.

After that I told Claude to proceed with the rest of the plan, which introduced a
`Move` dataclass with a `MoveKind` enum to replace the stringly-typed op tuples, and
batched all restarts together in lockstep so they share a single forward pass per
iteration rather than running sequentially. That brought the time down another 30
seconds, to 7 minutes even.

The next phase was empirical: I ran the same 12-pool set under different SA
parameter combinations and reported back. The key finding was that `--sa-temperature
0.8 --sa-cooling 0.95` (the original default) produced the absolute best decks but
at 6x the cost of pure greedy. At `cooling=0.9` all the wins collapsed to within
noise. At `cooling=0.8` there was a regression: one deck dropped from 2.1 to 1.6,
which is the classic SA failure mode of fleeing a good basin without finding a
better one. Claude diagnosed this clearly from first principles — the exploration
window is long enough to wander away from a local optimum but too short to find a
new one, and best-score tracking doesn't save you if the wander never crosses the
better basin.

The sweet spot turned out to be `cooling=0.85`: nearly identical deck quality to
`cooling=0.95`, but at 2m15s instead of 7 minutes. I set that as the new CLI
default.

I then wanted to try color-pair seeded initialization — instead of N random restarts,
start one restart for each of the 10 two-color combinations, each seeded with the
best on-color spells from the pool. The feature landed cleanly in code. But when I
ran it, something was wrong: 12 pools took close to 30 minutes, a 13x regression.
Claude initially hypothesized that the color-pair inits land in a fertile region of
the scorer's loss surface and the hill-climb keeps finding tiny micro-improvements,
so the early-stop never fires. I wasn't sure that was the explanation, because the
generated decks ended up almost identical between color-pair and random-restart runs.
Claude refined: the issue is that the early-stop tracks `current_score` movement
(not `best_score` improvement), so oscillating micro-moves satisfy the "is there an
improvement" check indefinitely. The right fix is a patience counter on `best_score`
specifically. That hasn't been implemented yet — at the end of the session,
color-pair init is present in the code but marked as not recommended until the
patience fix lands.

I asked Claude to record all of this in `experiments/2026-04-25-sa-deck-builder-tuning.md`.
There was one false start where Claude tried to add a "Greedy vs SA" section that
didn't match what I wanted, so I reverted it and deleted the open questions section
myself.

The second major feature of the day was replacing the single `--generated-decks-path`
flag in `match-outcomes` with a proper side-A / side-B flag split:
`--side-a-decks`, `--side-b-decks`, and `--side-b-decks-weight`. Related to this,
the `--self-play-label` flag was removed from match-outcomes entirely, because the
label now travels with the deck in the first column of the generated-decks file
(written by `build-decks --label`). This required changes across the Python
supervisor, the connector, and the Java `MatchGenerator`. A post-implementation
code review flagged several cleanup items: Javadoc pointing at a deleted class,
an overly long `main()` in `MatchWorkerMain`, repeated forge-build and file-
materialize patterns that deserved named helpers, and the side-B weight default
defined in four separate places. All of those were fixed.

One detail that almost got lost in the refactor was the table in the spec
documenting the 5 deck generation methods and their weights. Claude had dropped it
when rewriting the match-outcomes spec section. I noticed and asked for it back, and
it was restored in adapted form for the new flag matrix.

The working tree was clean by the end of the session, with six new commits from
`d378127` through `1413832`.
