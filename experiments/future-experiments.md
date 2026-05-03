# Future experiments

A holding pen for ideas surfaced during gen-2 work that aren't worth chasing
right now (typically because the binding constraint is data quality / volume,
not the model or the optimizer). Each entry: what it is, why it might help,
estimated magnitude of the effect, cost to try, and dependencies that might
unblock or favor it.

## Margin-weighted Bradley-Terry training loss

### Idea

Today every match contributes one training pair `(winner, loser)` with the
loss `BCE(score_winner − score_loser, 1)`. A 4-0 match and a 4-3 match
contribute the same gradient signal even though their information content
about deck quality is very different.

The proposal: weight each pair's contribution to the loss by the absolute
match margin (`|wins_a − wins_b|`, ∈ {1, 2, 3, 4} for Bo7). A 4-0 match
contributes 4× the gradient of a 4-3 match.

```python
# From: BCE(score_winner − score_loser, 1).mean()
# To:   (BCE(score_winner − score_loser, 1, reduction="none") * |margin|).mean()
```

Three implementation shapes, in increasing intrusiveness:

1. **Margin-weighted BCE** (above): smallest change. Same gradient
   direction, magnitude scales with margin. ~10 lines: extend
   `MatchTrainingExample` with a `margin` field, pipe through the
   collator, multiply in `_pairwise_bce`.
2. **Per-game pairs**: split each match into N pairs, one per game.
   Equivalent to margin-weighting in expectation but with more variance.
   Probably worse than #1 for our dataset size.
3. **Margin regression**: replace BCE with MSE/Huber on
   `predicted_margin = score_a − score_b` vs `actual_margin`. Most
   theoretically principled but biggest change — different loss, model
   now trained to predict a value with units, val_acc metric needs to be
   derived from sign agreement.

Recommendation if revived: start with #1 (linear `|margin|` weighting),
fall back to log-dampened (`log(1 + |margin|)`) if training proves
unstable.

### Why it might help

A 4-0 result strongly suggests the per-game `p` is at least 0.65 (≈ 18%
likelihood at p=0.65 vs 6% at p=0.50). A 4-3 result barely distinguishes
between p = 0.50 and p = 0.55. The current binary loss treats both the
same — throwing away the more confident pairs' extra information.

The mechanism is two-fold: more total signal per match (richer label),
and naturally down-weighting the noisiest matches (close ones at
p ≈ 0.50–0.55 which the Bo7 label-noise math already flagged as the
biggest contributor to our val_acc ceiling).

### Estimated magnitude

**+1 to +3 pp on val_acc.** The ceiling argument from
`gen2-initial-training.md` puts the model-imperfection bucket at 2–8 pp
above our current 0.70 (under the oracle ceiling of 0.72–0.78). Margin
weighting attacks that bucket by giving the model a richer signal per
match, but doesn't move the irreducible-Bo7-noise component.

May also improve val_loss without moving val_acc much — the model becomes
more confident-and-correct on decisive matches without flipping
borderline predictions. That's still useful (sharper score calibration
helps `evaluate-scorer`'s deck-vs-deck ranking, the metric we actually
deploy on) but it won't show up in the headline metric.

### Cost

Low. ~10 lines of code change for option #1, plus one new field in
`MatchTrainingExample` and the collator. No new CLI flags strictly
required (could ship as the new default, falling back to binary if a
flag like `--margin-weighting linear|log|none` is set to `none`). Test
cost is small.

### Industry-practices precedent

Mature technique with decades of use across three traditions:

- **Sports rating systems.** Glicko and Glicko-2 (chess.com, USCF) and
  TrueSkill (Halo matchmaking) weight rating updates by outcome
  precision. Massey and Sagarin ratings use margin-of-victory directly
  in college football. **Pythagorean expectation** in baseball (run
  differential outpredicts W-L record) is the same insight at the
  season level. The NCAA banned MOV from BCS rankings in 2002, but for
  an incentive reason that doesn't apply to bo-N matches: NCAA didn't
  want to encourage running up the score.

- **Learning-to-rank.** **LambdaRank** (Burges, NIPS 2006) and its
  descendants LambdaMART / LightGBM-Rank are the dominant practical
  learning-to-rank objective at Bing/Yahoo and modern search-recsys
  systems. The core idea is *exactly* what's proposed here: take the
  pairwise BT/BCE loss and multiply each pair's gradient by the size
  of the metric change that swapping the pair produces. In LambdaRank
  that's the change in NDCG; for sealed it'd be the match margin.

- **RLHF reward modeling.** BT pairwise loss is the standard objective
  for reward models. The "graded preferences" research line (5- or
  7-point Likert scales replacing binary preferences in DPO/IPO/KTO
  variants) generally shows modest reward-model accuracy gains —
  similar magnitude to the +1 to +3 pp we'd expect.

Empirical effect sizes across these domains land in "useful but not
transformative" territory, which matches the size estimate above.

### Dependencies / when to revisit

Worth doing once the data-side levers are exhausted and we want to
squeeze the last bit out of model-side. **Stackable with all other
interventions** (Phase B encoder fine-tuning, architecture changes,
more matches, longer Bo-N) — the margin signal is orthogonal to
everything else.

Likely target: gen-3 model. By then we'll have more data
(reducing the noise floor in absolute terms), at which point the
per-pair margin weighting captures larger absolute information gains
per match.

## Color-restricted deck builder personalities

### Idea

Two new `build-decks` modes that lock the SA search inside a fixed
set of 2 or 3 colors. Intended as gen-3 personalities for self-play
shape diversity, and as a diagnostic for the gen-2 4-5-color drift.

Flag shape (sketch):

- `--restrict-spell-colors`, valid only with `--restarts color-pairs`
  or the new `--restarts color-slices`. Filters `spells_remaining` (and
  the SA swap candidates by extension) to cards whose nonland mana cost
  is a subset of the restart's color set, plus *truly* colorless spells.
  **Devoid cards do not count as colorless** — they have colored mana
  symbols in their cost even though devoid suppresses color elsewhere.
  Today `--restarts color-pairs` only filters the *initial* 23 spells;
  the SA can drift outside the pair via swaps. This flag closes that
  drift.

- `--restarts color-slices`: new restart strategy enumerating all
  C(5,3) = 10 three-color combinations (5 shards + 5 wedges). Same
  per-restart flow as `color-pairs` (initial 23 filtered to subset
  cards), with `--restrict-spell-colors` locking the search inside
  the slice for the duration. "Two colors + splash" is a well-known
  sealed archetype and a natural separate personality.

Two gen-3 personalities fall out: `gen3-pair` (2 colors) and
`gen3-slice` (3 colors). Shipped alongside the unconstrained
gen-3 model, giving self-play three deck-shape variants from
the same scorer.

### Why it might help

Diagnostic + diversity, not direct strength.

1. **Disambiguates scorer-vs-search responsibility for the color drift.**
   gen2a's training data showed 2-color decks winning 35pp more than
   5-color (`Win rate by method by deck color count` table). gen2a still
   built 4-5 colors 44% of the time. Two competing explanations: the
   scorer genuinely thinks 4-color is the best deck for the pool (it's
   right or wrong, but the search is faithful to it), or the scorer
   knows 2-color is better but the SA landscape has 4-color local
   optima the search settles into. A constrained 2-color deck plays
   forge-best — if win rate jumps up, the search was the problem; if
   it stays flat or drops, the scorer was calibrated and we need to
   look at the training distribution instead.

2. **Self-play deck-shape diversity.** The gen-2 family produced
   multiple training variants (gen2a, gen2b1, gen2ba) that all built
   similar shapes (~45% 3-color, ~36% 4-color). Forcing the next gen's
   self-play matches to include `gen3-pair` and `gen3-slice` decks
   gives the next scorer deck shapes the family has under-represented
   in its training corpus.

### Estimated magnitude

**Likely negative on the scorer's reported deck score** — the
constrained search space is a subset of the unconstrained one, so the
unconstrained search's chosen deck always scores ≥ the constrained
one's by definition. The user's standing prior (from running the
existing `--restarts color-pairs`): unconstrained outputs are often
identical or near-identical to color-pair-init decks, suggesting the
search drifts away from 2-color almost immediately when allowed to.

The interesting metric is *win rate vs forge-best*, not the score:

- Win rate up while score is down → scorer miscalibrated; SA stuck in
  4-color local optima despite better deck existing nearby.
- Win rate down with score down → scorer is calibrated to the pools;
  4-color is genuinely the right play and the 2-color training signal
  comes from a confound (forge-best dominates 2-color cells, n=14988
  of 18166 in `match-outcomes-all.txt`).
- Win rate flat → underpowered; either the constrained deck is roughly
  as good in expectation or n is too small to tell.

### Cost

Low. `--restarts color-pairs` already builds the on-color spell list
at init; reusing that mask in `spells_remaining` is ~15 lines + the
CLI flag. The slice variant clones the pair logic with a 10-triple
enumeration. The "devoid doesn't count as colorless" rule is one extra
predicate (`mana_cost.color_count == 0` instead of `Card.is_colorless()`,
which currently treats devoid as colorless).

### Dependencies / when to revisit

Schedule for gen-3 training. Independent of the data-side levers from
`gen2-unfrozen-embeddings.md` and stackable with margin-weighted loss
(above). Worth doing even if gen-3 ends up matching gen-2 in raw win
rate — the diagnostic answer (scorer vs search) is high-value
regardless of whether the personalities are competitive.

## Per-card winnability as encoder pretraining target

### Idea

Replace (or augment) the price-prediction pretraining with a per-card
"winnability" target derived from per-game cards-played records. The
label is a *credit-assignment* score: of the games the card's deck
won, how often did the card actually contribute (= get played)?

For each card, label = `wins-when-played / wins-when-in-deck` where:

- the **numerator** is the number of games where this card was
  played by the winning side (entered the battlefield or stack);
- the **denominator** is the number of games where this card was in
  the winning side's deck — whether or not it was played that game.

Equivalently: among games the deck won while this card was in it,
what fraction of those wins did this card show up for? A card that's
always cast in the winning games scores ~1.0; a 7-mana bomb the deck
never gets to mana for scores low; a card that's stuck in losing decks
doesn't drag its own score down because losing games don't enter the
denominator.

The denominator choice matters; two alternatives we explicitly reject:

- **`wins-when-played / games-when-played`.** Biases toward late-game
  splash bombs that win when they finally hit but rarely get cast at
  all (denominator only counts the rare games where the bomb
  resolves). The card looks great by win rate but contributes to few
  actual games.

- **`wins-when-played / in-deck-games`** (every game the card was in
  someone's deck, won or lost). Penalizes cards for being randomly
  included in losing decks — losses where the card was a non-factor
  still drag the ratio down. We want a *credit-assignment* signal
  within winning decks, not an attribution-of-blame across both.

The chosen `wins-when-played / wins-when-in-deck` ratio answers
"when this card was in a winning deck, did it contribute to the win?"
Combined with appearance-count weighting (below) it produces a
defensible per-card quality score without forcing each card to also
explain its losses.

### Why it might help

Direct response to the deterministic-feature-reliance hypothesis
(`deterministic-feature-reliance.md`). If that diagnostic confirms the
scorer leans on the 32 hand-features and ignores the 512 transformer
dims, the question is "what would actually load card-quality
information into the transformer dims?" Match-outcome gradients can't
do it through the noisy Bo7 path. A per-card auxiliary loss with a
label that *is* card quality, scaled across thousands of games, can.

The mechanism is straightforward: a regression head predicting
winnability forces the encoder to allocate dimensions to "what makes
this card good" — abilities, P/T-vs-cost ratios, evasion, removal
modes. Because the label is dense (one number per card, not pairwise),
the gradient signal per epoch is much higher than match outcomes can
provide.

Also addresses the "Forge has a card rating but won't share its
methodology" branch of the embedding investigation: this *is* a
methodology, it produces a defensible per-card rating, and it doesn't
depend on Forge's hand-curated draft values.

### Acquiring labels

The numerator requires per-game cards-played data, which the existing
`match-outcomes.txt` doesn't contain (it only records deck composition
at the match level). We need a new sidecar file written by the Java
worker.

**File**: `output/sealed/cards-played.txt` by default. Collected
automatically during any `python -m sealed match-outcomes` run — no
opt-in flag, this becomes a standard output of self-play. One line is
appended after every *game* (versus once per *match* for
`match-outcomes.txt`, so a Bo7 produces 4-7 lines here for every 1
line in `match-outcomes.txt`). Line buffering and append semantics
match `match-outcomes.txt` so an interrupted run keeps everything
written so far.

**Columns**:

```
set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter
```

- `cards_played_X` / `cards_not_played_X` — pipe-separated card-name
  lists that together reconstruct side X's full deck for that game.
  Splitting the deck this way per game lets the aggregator compute
  both halves of the metric in one pass without joining.
- `winner` — `A` or `B`.
- `starter` — which side started this game (derivable from the
  existing `play` field in `match-outcomes.txt`, but cleaner to write
  per-line for joinless aggregation).

This requires Java-side instrumentation in `MatchWorkerMain` (or
wherever Forge surfaces "card entered battlefield / stack" events
during AI-vs-AI play) plus a Python supervisor change to wire the
output file alongside the existing `match-outcomes.txt` append. The
two files share a row ordering by match (a Bo7 produces 4-7 game
lines in `cards-played.txt` per 1 line in `match-outcomes.txt`), so
they can be joined when needed.

Aggregator pseudocode, per game line:

```
if winner == A:
    for c in cards_played_A:     wins_when_played[c] += 1
    for c in cards_played_A + cards_not_played_A: wins_when_in_deck[c] += 1
if winner == B:
    for c in cards_played_B:     wins_when_played[c] += 1
    for c in cards_played_B + cards_not_played_B: wins_when_in_deck[c] += 1
```

Losing-side cards contribute nothing — that's the whole point of the
denominator choice. After processing the corpus,
`label[c] = wins_when_played[c] / wins_when_in_deck[c]` per card.

Low-n shrinkage handles the inevitable long tail of cards seen only a
handful of times: Bayesian shrinkage toward 0.5 with a prior weight
`k`, so `label = (wins_when_played + k/2) / (wins_when_in_deck + k)`,
typical `k = 10-30`. Or sample-weight the regression loss by
`wins_when_in_deck / (wins_when_in_deck + k)` so high-n cards
contribute more gradient than low-n ones. Either is a one-liner; pick
after looking at the per-card win-appearance distribution from the
first few thousand games.

### Estimated magnitude

Conditional on the deterministic-feature-reliance diagnostic
confirming the hypothesis: this is the most plausible candidate for
moving the within-bucket win rate against forge-best, since it's the
only intervention that directly trains card-quality knowledge into the
encoder.

If that hypothesis is wrong (transformer dims already carry useful
signal): smaller upside, but still likely positive — the encoder gets
a denser, less noisy label than match outcomes provide, similar
magnitude to the multi-task pretraining bullet in "See also" below.

Estimated effect on win rate against forge-best: **+3 to +10 pp** if
the hypothesis is right; +0 to +3 pp otherwise. Wide range because
this depends entirely on whether the encoder is the binding constraint.

### Cost

Medium. ~100-200 lines of Java in `MatchWorkerMain` (plus whatever
Forge-side hooks expose card-entered-battlefield events per game) to
write the new per-game sidecar file, ~30-50 lines of Python for the
aggregator, ~50 lines for the regression head. Plus the cost of
regenerating the corpus with instrumentation on — though existing
matches are not lost, only matches played going forward will have
the per-game data. After ~50-100k instrumented matches every
sealed-legal playable should have enough sample to produce a usable
label.

Roughly 1-2 weeks of implementation + days to weeks of compute.

The training-time cost of adding the auxiliary loss to the encoder
itself is modest — one extra forward + one extra loss term per batch.

### Dependencies / when to revisit

Run after `deterministic-feature-reliance.md` Test 1a/1b. If those
confirm the encoder is underused, this is the most concrete next step.
If they show the encoder is already pulling its weight, revisit only
once the data-side label-noise levers in `gen2-unfrozen-embeddings.md`
have been pulled and the model-imperfection bucket reopens.

## See also (deferred items already documented elsewhere)

- **Multi-restart + multi-temperature ensemble for deck building** —
  see `sa-deck-builder-tuning.md` open questions §3, §4. The cheap
  win for SA-built deck variance.
- **Color-pair seeded init for the deck builder** — same doc, open
  question §4. Directly addresses "random init contains all colors";
  orthogonal to and stackable with SA.
- **`cooling` parameter sweep at fixed `T=0.8`** — same doc, open
  question §2.
- **Pre-train the encoder on a closer auxiliary task** — surfaced
  during the gen-2 Phase B work but not formally documented.
  Candidates: masked card-text prediction (BERT-style, self-supervised),
  multi-task pre-training (price + type classification + keyword
  multi-label), format-playability classification (binary: appears in
  pro decks of any format vs doesn't). Each gives the encoder MTG-text
  understanding via dense, low-noise signal without the price-task
  contamination that the gen-2 Phase B runs revealed. High implementation
  cost — each is a substantial new pipeline. Worth revisiting only if
  Phase B becomes interesting again under a new label-noise regime.
  See also the per-card-winnability section above for one specific
  auxiliary target.
- **Iterative feature distillation** (Phase B encoder → re-cache .npz
  → fresh Phase A scorer → repeat). Surfaced during gen-2 Phase B as a
  test of "are Phase B's gains real or co-adaptation?". Premature until
  Phase B actually helps val_acc on a future cleaner-label corpus.
- **Shallower scorer architecture under Phase B.** Hypothesis: with
  encoder fine-tuning sharing the per-card-feature work, a 2- or
  3-layer scorer might generalize better than 6 layers. Untested
  because Phase B didn't move val_acc; would be revisited if a future
  Phase B does.
