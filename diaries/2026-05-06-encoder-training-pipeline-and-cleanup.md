# May 6, 2026 — Encoder training pipeline and data cleanup

**TL;DR:** I shipped the full spec-016 sealed encoder training pipeline in one
session — 60 tasks across Java and Python — then spent the afternoon cleaning
up what the match-worker was actually recording as "played" and rethinking the
score function for the regression head.

The session opened with `speckit.implement` running against spec-016, which
decomposed the work into eight phases: a baseline check, the sealed encoder
model and store, Java-side per-game card tracking, the full `train-encoder`
Python pipeline, default-flip for `--encoder-checkpoint`, a `build-vocab`
subcommand, shrinkage verification, and a polish pass. Claude worked through
all 60 tasks systematically, and by the time the dust settled the test count
had gone from 786 to 855 fast Python tests, 259 Java tests, 16 integration
tests — all green, ruff clean.

One design issue Claude caught mid-stream was around strict checkpoint loading:
loading an encoder-only state dict into a model that also has a regression head
would fail on unexpected keys if done naively. Claude switched to
prefix-based validation instead of full strict-load, which correctly catches a
leaked regression head without choking on intentionally-absent ones.

After the main feature landed and I committed it, I asked for a
`scripts/print_card_winrates.py` table to inspect what the pipeline was
actually learning from. The initial version was slow — 32 seconds to look up
mana costs for 24k cards — because it was reading each card file sequentially.
A `/review` pass with a performance focus identified the bottleneck and Claude
rewrote it with a `ThreadPoolExecutor` (32 workers) plus early-break on the
first `mana cost:` line, bringing it down to 5 seconds — about a 6x speedup.

Then I redesigned the score function. The original was
`wins_when_played / wins_in_deck`, treating the score as a pure win rate. I
switched it to `(wins_played - losses_played) / (wins_in_deck + losses_in_deck)`,
bounded between -1 and +1, which makes negative influence (overcosted cards,
bad effects) visible and weights the magnitude by how often the card actually
gets played rather than just sitting in the deck. I also added Bayesian
shrinkage — dividing by `(wd + ld + k)` with k=20 — as a separate column so
low-observation cards don't hijack the sort order. When I asked Claude to check
whether the adjusted scores were already Gaussian in shape (using `./tmp.txt`
from a live run), it ran a quick distribution analysis and confirmed: mean near
zero, std ~0.063, skew ~0.026, W=0.990. That result actually changed the
design decision about what to use as a regression target: since the adjusted
score is already nearly Gaussian and preserves a meaningful zero (neutral
influence), using it directly as the regression label is simpler than computing
quantiles and loses less information.

The other major thread of the day was figuring out what the card collector was
accidentally capturing. Investigating why "Island" and "Undercity" appeared in
the played-cards list led to two distinct bug classes: manifested cards entering
face-down fool the basic-land filter because `card.getType()` returns the
face-down type rather than the underlying card type; and dungeons, effects like
"The Ring," and emblems slip through because `card.isToken()` is false for
`GamePieceType.EFFECT` and `GamePieceType.DUNGEON`. The second class got fixed
by adding a `gamePieceType == CARD` guard to `shouldRecord`. For manifested
cards specifically, I chose a nuanced rule: a card played face-down via its
own morph/disguise keyword counts (the cast event fires), but a card manifested
face-down by another card's effect does not count until flipped — because from
the deck-evaluation standpoint, manifesting someone else's Island is not the
same as casting your own Willbender for its morph cost. That got implemented
as a `!card.isFaceDown()` guard on the zone-change branch only, with a comment
noting the unhandled case (face-up reveals of externally-manifested cards).

By the end of the session I had three commits and a much cleaner picture of
what the data collection actually means. The 1M-game estimate for stable scores
— about three weeks at current throughput — made it clear that the data
collection side is the long pole.
