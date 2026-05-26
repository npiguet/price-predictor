# May 17, 2026 — Gen3 eval results and experiment housekeeping

**TL;DR:** The gen3-256 scorer came back at 64% game win rate across
576 BO7 matches (47/48 pools positive), then a second run confirmed
~+23 pp mean delta against forge-best. The day was also heavy on
documentation cleanup and a long discussion about Bradley-Terry loss
design.

The day started with a practical setback: my computer had rebooted
overnight and killed a running `evaluate-scorer` job. It turned out
all 576 matches had actually completed before the crash — only the
aggregation step was lost. Claude added a `--resume` flag to
`evaluate-scorer` that skips pool and deck generation when existing
worker shard files are found, then aggregates immediately. Running it
recovered the result: scorer 64.0% / forge 36.0% per game, which
is 76.2% / 23.8% per match. I noted "results look pretty good" at
that point.

A second independent 24-pool run came in at 58.4% per game (17.3 pp
mean delta vs 28.7 pp in run 1). Across both runs together, the
scorer won 47 of 48 pools, with an aggregate ~+23 pp mean delta and
roughly a 61/39 game split. The 11.4 pp run-to-run swing in mean
delta is the important calibration point: a single 24-pool number
carries about a 5 pp uncertainty band, so the true edge is closer
to +20 pp than to +30 pp.

I also pointed out that the magnitude of the win-rate delta per pool
correlates with the score delta between the two decks on that pool.
Claude computed Pearson r = +0.52, Spearman = +0.53, OLS slope
roughly +7.8 pp of win-rate edge per 1.0 of score gap across the
48 pools. Run 2 alone was stronger (r = +0.61) because it had a
wider spread of score gaps.

On the reporting side, I asked Claude to expand the eval summary
table to show both per-game and per-match win rates side by side.
Pool 8 of run 1 turned out to be the validating example: the scorer
was ahead per game at +2.0% but behind per match at -12.5%, which
the old per-game-only table had hidden. All this went into the gen3
experiment report file and was committed.

The first session was devoted to restructuring
`experiments/2026-05-13-gen3-initial-training.md`. My complaint was
that it read chronologically rather than as a report. Claude proposed
a specific restructuring: moving evaluation variance and the
score-delta section under "Training results", merging the scattered
deck-building observations into one unified section, and removing
phrases like "the narrative in the previous subsection needs
softening" that carried chronological baggage. I pushed back on the
section order twice before it landed right.

I also corrected two framing mistakes in the writeup. First, I said
I was going to use both the 128-d and 256-d gen3 checkpoints for
further self-play, not deploy the winner and discard the rest — the
point is that different "personalities" as deck-builders feed better
diversity into the next training round. Second, I noted a third
confounder in the gen2-to-gen3 val_acc jump that had been omitted:
corpus composition shift (matchup-mix change), on top of encoder
swap and corpus size.

The second session covered the gen4 encoder hparam sweep. I had
started pretraining seven runs at 512d and asked Claude to write the
experiment file. After Claude drafted it, I pushed back on the
conclusion "512d doesn't help the picker" — we hadn't tried 512d
with the picker yet, and I wanted the writeup softened to "unlikely
to help much" with the note that it was still worth taking 512d
through scorer and match-play evaluation. Claude also initially
included a corpus-shift caveat between gen3 and gen4 that I
corrected: gen4 trains on exactly the same corpus as gen3. That
made the gen4 vs gen3 encoder comparison a clean A/B.

I then pointed at something from the gen3 long-train logs: 128d had
`val_reg` = 0.0187 and 256d had `val_reg` = 0.0199 — the regression
term actually favoured 128d — yet 256d won decisively at the picker
stage. Claude decomposed the 0.0117 `val_loss` gap: MLM contribution
was +0.0128 in favour of 256d and the regression contribution was
-0.0012 in favour of 128d, making MLM responsible for approximately
110% of the encoder-stage advantage. That became the key evidence for
the working hypothesis: the picker is reading text-representation
quality (which MLM accuracy proxies) rather than regression-head fit.

The day also included a long discussion about Bradley-Terry loss
design. Starting from a margin-weighting idea in the future-
experiments file, I raised the point that not all 4-0 wins carry the
same information: beating forge-best with a 4-0 is strong evidence
the model is good; beating random with a 4-0 tells you little. I
sketched that the gradient should pull differently on each side
depending on prior expectations of that deck's strength. Claude
formalized this as asymmetric per-side gradients via PyTorch
`stop_gradient`, connected it to Glicko rating deviation and
TrueSkill, and confirmed via web search that the neural side of
this bridge has no established name. Claude assessed margin weighting
as worth shipping (it adds strictly new information BCE can't see),
while the asymmetric per-side extension is lower priority — BCE
already partly captures opponent-strength asymmetry once the model
is calibrated, and the encoder swap has been the dominant lever.

Later in the day, the in-play head-to-head data from 520 early gen4
matches showed the three builder methods were far from near-mirror:
gen3-256 vs forge-best runs 67/33, gen3-128 vs forge-best 62/38,
gen3-256 vs gen3-128 62/38. I pointed out that gen3-256 was
substantially better than gen3-128 in match play despite the
encoder-stage numbers showing only a 1.67 pp val_acc gap. Claude
explained this through the interplay of representation quality and
the way per-card scores drive greedy deck building — val_acc tests
pairwise prediction on finished decks while deck construction queries
per-card scores at much finer granularity. I decided to test all
three widths (128d, 256d, 512d) for gen4 at the scorer and match-
play stage rather than reading the encoder metrics alone.

I also asked whether retraining the encoder with the new ~100-150K
rows of `cards-played.txt` data was worthwhile. Claude said yes at
essentially no extra cost since the three encoders were already being
retrained, but expected only ~0.3-0.7 pp val_acc from the data
growth alone. The main caution was self-play sampling bias: the new
rows come exclusively from the three methods currently in rotation,
so long-tail cards those builders avoid get no new data.

The day wrapped with a batch of housekeeping commits: prefixing all
files in `experiments/` and `specs/` with their first-git-commit
date in YYYY-MM-DD format, fixing a race condition in
`print_card_winrates.py` where a live match-outcomes worker could
append new rows between two sequential `iter_rows()` calls (causing
a `KeyError` on names seen only in the second pass), and folding
additional gen4 512d sweep runs into the experiment file as they
completed.
