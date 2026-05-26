# May 13, 2026 — Encoder data pipeline overhaul

**TL;DR:** I implemented the multi-head + MLM rewrite of `train-encoder`,
then spent the rest of the day cleaning the `cards-played.txt` dataset
and profiling and fixing a series of performance bottlenecks that brought
the per-epoch time from ~2m40s down to ~15s.

The day started with `/speckit.implement` on spec 016, which went cleanly
— 877 fast tests and 13 integration tests all green after the commit. The
implementation was substantial: nine regression heads with Bayesian
shrinkage, an MLM auxiliary loss, stratified card splitting, a 23-column
`cards-win-rates.txt` snapshot, and the `[MASK]` token now seeded in
the vocabulary. One wrinkle came up in the integration test for the
shrinkage behaviour: a high-N card with very few co-occurrences in a
specific color slot caused the per-color stability check to fail, since
total observation count and per-color slice count can diverge
dramatically. The fix was to restrict the "high-N stability" assertion to
the four non-color heads.

From there I shifted to dataset quality. My `cards-played.txt` had been
accumulating noise from early bugs — Forge internal triggered-ability
placeholder names (things like `Lightning Bolt (3)'s Effect`), dungeon and
Monarch token names, and Universes Beyond flavor-name reskins where Forge
logs the videogame character name (`Cloud Strife`, `Squall Leonhart`)
rather than the printed MTG card name (`Najeela, the Blade-Blossom`,
`Danitha Capashen, Paragon`). Checking all 607 unplayed=0 entries against
`AllPrintings.json` confirmed that 342 of them were genuine junk and none
of the junk names matched a real printed card. The remaining 265 were
legitimate — split-card halves, adventure face names, MKM room names —
where Forge logs whichever face resolved rather than the combined printed
name.

The fix I decided on was to build a canonical-name resolver from MTGJSON's
`faceName` and `flavorName` fields, plus a drop list for actual junk
patterns. I also fixed `PlayedCardCollector.java` in the forge-connector
to call `getOracleName()` instead of `getName()`, so future self-play runs
emit canonical names directly. After a few iterations to handle edge cases
(identity entries winning over face-name entries for cards like "Lightning
Bolt" which appear as both a canonical name and a saga face name; token
names in MTGJSON not being in the main card walk; a stray space in Forge's
filename for "Bespoke Bō"), the cleaned file resolved every card. Two
genuine gaps remained — `Cecily, Haunted Mage` and `Tadeas, Juniper
Ascendant`, both real Secret Lair cards Forge has no script for — and I
changed `train-encoder` to report those as a warning and drop them rather
than aborting.

The performance review came next. The symptom was a ~4.5-minute warm-up
before epoch 1 and ~2m40s per epoch. Claude flagged seven findings. I
pushed back on two of them based on things discussed in prior sessions: on
the MLM head projection, Claude said applying the `Linear(256→5000)` over
all positions before masking was wasteful, and I asked which was correct
given an earlier claim about CPU gather costs negating GPU savings.
Claude walked through the current code and confirmed there is no CPU gather
— both the mask and the gather are on-GPU already, so moving the masking
before the projection is strictly cheaper in both compute and memory
(eliminating a ~780 MB/batch intermediate). On length-bucketed batching I
asked whether the transformer actually requires a fixed input length; it
doesn't, only the position table has a ceiling, so per-batch dynamic
padding is safe. I told Claude to do all seven findings.

The measured result: warm-up ~4m32s → ~2m55s, per-epoch ~2m40s → ~15s
(~10×). The warm-up is now dominated by the single streaming pass over
the 700 MB `cards-played.txt` at ~2m51s.

After that I questioned whether parallelizing the card-file reads would
help the warm-up further, recalling that a similar trick had saved two
minutes on a different script. I asked for a profile before accepting any
claims. The cProfile output was clear: ~88 of ~100 warm-up seconds were
inside the per-game/per-card Python loop, not in I/O. The 28k card files
total ~56 MB and read in ~2.7s warm-cache; parallelizing them would save
~2.5s. The actual bottleneck was a `dict.setdefault(name, CardCounters())`
call that constructs a throwaway `CardCounters` — with four nested color
dicts — on every iteration of a ~38M-iteration loop, since Python evaluates
the default argument unconditionally. Replacing it with `get`-then-create
and switching the per-color counters from `dict[str,int]` to a fixed
`list[int]` indexed by WUBRG position brought `_aggregate` from ~96s to
~82s — a real ~15% win, though the remaining ~82s is the irreducible Python
loop overhead.

The training run itself, when I let it go for 63 epochs, looked healthy
by all the new diagnostics: `val corr` on `score_play` reached +0.57,
`played_rate` +0.64, MLM perplexity fell from 1487 to ~2.0, and top-1
masked-token accuracy climbed from 11% to ~82%. The color-lift heads were
weakest (U peaking around +0.50, the others lower), which Claude explained
as expected given sparse per-color slices and the ×1/5 loss prefactor.
