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

Specced separately at `specs/card-winnability-pretraining.md`.
Worth pursuing once `deterministic-feature-reliance.md` Test 1a/1b
confirms the encoder is underused; the spec defines the
`output/sealed/cards-played.txt` per-game sidecar, the
`wins_when_played / wins_when_in_deck` label, low-n regularization,
and the auxiliary regression-head training integration.

## Masked-token auxiliary loss for the sealed encoder

### Idea

The card-winnability spec trains the sealed encoder against two
regression heads: net winning influence and played rate, both per-card
scalars derived from per-game play counts. The proposal: add a third
training-only head that performs masked-token reconstruction over each
card's tokenized text — the BERT-style MLM objective, run jointly with
the two regression heads on the same encoder.

Randomly mask ~15% of input tokens (replace with a `[MASK]` token), feed
the corrupted sequence through the same token + card encoder used by the
regression heads, and project the contextualized token outputs back to
vocab logits at each masked position. Cross-entropy against the original
token, summed (with a small weight) into the existing two-MSE loss:

```
loss = MSE(score) + MSE(played_rate) + w * CE(masked_token, true_token)
```

The MLM head is discarded after training, like the regression heads.
The encoder artifact is unchanged in shape.

### Why it might help

The two regression heads deliver gradient only through per-card
aggregate labels. A card with few in-deck observations contributes one
noisy scalar per head, and the shrinkage prior pulls it aggressively
toward neutral — the encoder learns very little about that card's text.
Most of the corpus is in the long tail.

MLM gives every card dense, per-token training signal regardless of
play-count. The encoder is forced to encode each token so neighbors are
predictable from context, which is exactly the contextual understanding
the regression heads need downstream to read card text into a "good
card" signal. Tail cards that the regression loss can't usefully train
the encoder on still contribute meaningful gradient through MLM.

It's also a regularizer against the encoder collapsing onto a few
regression-label-specific dimensions — those dimensions still need to
support token reconstruction across the full corpus.

Distinct from the "pre-train then fine-tune" framing in the See-also
bullet below: MLM here is a *joint* auxiliary loss during the
winnability training run, not a separate pretraining stage. Joint
training avoids the catastrophic-forgetting risk of fine-tuning, at the
cost of one extra hyperparameter (`w`).

### Estimated magnitude

Hard to ballpark without running it. The size of the win depends on
whether the encoder's binding constraint is *tail-card under-training*
(MLM helps a lot) or *the regression labels themselves are too noisy
even on high-observation cards* (MLM doesn't help; the regression heads
are at their floor). Diagnostics from the winnability run — val loss
broken down by per-card observation count — would say which.

### Cost

Moderate. New head (linear projection from token-level encoder outputs
back to vocab size), masking augmentation in the dataset, one new
loss-weight CLI flag. No new data, no inference-time cost.

### Dependencies / when to revisit

Conditional on the per-card winnability encoder
(`specs/card-winnability-pretraining.md`) being implemented and
producing diagnostics that point at tail-card underfitting. Stackable
with margin-weighted scorer loss and color-restricted personalities —
operates entirely inside encoder training, not the scorer. Most natural
addition once the basic two-head encoder is in service and the
long-tail-card signal becomes the binding constraint.

## Play/draw split of the winning-influence head

### Idea

The card-winnability spec's head 1 (net winning influence) sums over all
games regardless of whether the card's owner started first. Sealed has a
real and well-understood structural asymmetry between the play and the
draw — tempo cards (one-drops, hasty creatures, curve plays) gain value
on the play; reactive cards (sweepers, expensive removal, card draw)
gain value on the draw. The current head averages these into one
scalar, so a card that is +0.30 on the play and -0.10 on the draw and a
card that is +0.10 on both both land at head 1 ≈ +0.10. The encoder
sees identical labels for two MTG-distinct phenotypes.

The proposal: replace head 1 with two parallel scalar heads, each a
single linear projection + tanh on the shared encoder.

```
head_1_play = (W_played@play - L_played@play) / (in_deck@play)
head_1_draw = (W_played@draw - L_played@draw) / (in_deck@draw)
```

`starter` is already a column in `cards-played.txt`. Aggregation
becomes "increment one of 8 counters per card per game instead of 4."
Bayesian shrinkage applies independently to each half. Played rate
(head 2) stays unsplit — whether a card gets cast is dominated by mana
cost and draws, both ≈ insensitive to play/draw, so a head-2 split
mostly buys √2 noise for negligible signal.

### Why it might help

The encoder is forced to encode the tempo↔reactivity axis explicitly.
Today the gradient on `(+0.30, -0.10)` and `(+0.10, +0.10)` is
identical and tells the encoder nothing about the difference; with the
split, the gradients are `(+0.30, -0.10)` and `(+0.10, +0.10)` —
opposite *vectors*, very different positions in embedding space. The
downstream scorer can then balance tempo and reactive cards in a deck
intelligently (sealed decks deliberately mix both because game-1
play/draw is unknown).

### Drop the original head 1, don't keep it alongside

`head_1 = (n_play/n_total) * head_1_play + (n_draw/n_total) * head_1_draw`
with `n_play ≈ n_draw` for every card (starter assignment is
approximately uniform across games). So head 1 carries no information
the two split heads don't already carry. Supervising on all three would
add no signal *and* triple the gradient pressure on winning-influence
relative to played rate, breaking the loss balance the spec's "Why two
heads" section relies on.

### Estimated magnitude

Hard to ballpark in isolation. Effect size depends on how much of head
1's variance across cards is currently driven by the play/draw axis
versus other factors (raw card power, color, curve position). Ballpark
intuition: probably a small-to-moderate gain on its own, larger when
stacked with the MLM auxiliary loss because tail cards with limited
data on each split half lean harder on encoder priors.

### Cost

Very low. ~30 lines: 4 extra counters in aggregation, two heads
instead of one, two extra columns in `cards-win-rates.txt`. No new
data, no inference-time cost, no new CLI flags strictly required.

The √2 per-cell noise increase from halving the data per label is
absorbed by the existing shrinkage prior.

### Bonus diagnostic

`cards-win-rates.txt` gains two columns instead of one. Scrolling it
becomes directly human-readable: which cards in a set are
tempo-positive, which are catch-up-positive. A sealed-format-experienced
reader can sanity-check that the labels are picking up real MTG
structure rather than artifacts of the sampling distribution.

### Dependencies / when to revisit

Conditional on the per-card winnability encoder being implemented.
Stackable with the MLM auxiliary loss (above) and complementary to it —
play/draw split sharpens the regression signal, MLM densifies the
gradient on tail cards. Most natural addition once the basic two-head
encoder is in service and a first round of training has confirmed head
1 is doing meaningful work to begin with.

## Cast-lift as a third regression head

### Idea

The card-winnability spec's head 1 (net winning influence) sums over
all in-deck observations regardless of whether the card was actually
cast that game. This folds two distinct effects together: (a) "does
casting this card change the outcome?" and (b) "does this card tend to
land in winning decks?". A card that's just along for the ride in
strong decks and a card that genuinely swings games when cast can
arrive at the same head 1 value.

The proposal: add a third regression head supervised against
**cast-lift**:

```
p_play  = W_played      / (W_played      + L_played)        # winrate when cast
p_dead  = W_not_played  / (W_not_played  + L_not_played)    # winrate when in deck but not cast
lift    = p_play - p_dead                                   ∈ [-1, +1]
```

The four counters are already computable from `cards-played.txt` —
each side of each game contributes `(played | not_played) ×
(winner | loser)` to one of four buckets per card. No new data
collection; aggregation pass extends to populate four counters per
card instead of two.

Architecturally, head 3 is a single linear projection + tanh on the
shared encoder, mirroring head 1.

```
loss = MSE(score) + MSE(played_rate) + MSE(lift)
```

### Why it might help

Three example cards, all 100 in-deck observations, cleanly distinct on
the lift axis but partially confused on (head 1, head 2):

| Case | n_cast | p_play | p_dead | head 1 | head 2 | lift |
|------|-------:|-------:|-------:|-------:|-------:|-----:|
| Workhorse 2-drop  | 80 | 0.60 | 0.50 | +0.16 | 0.80 | +0.10 |
| 6-drop bomb       | 30 | 0.70 | 0.50 | +0.12 | 0.30 | +0.20 |
| Auto-include drag | 60 | 0.55 | 0.55 | +0.10 | 0.60 | 0.00  |

The third row is the interesting one: head 1 looks like a useful card
(+0.10), but the deck wins 55% whether or not the card hits the table.
The card isn't doing anything — it's systematically landing in
slightly-better-than-random decks (build-method × card-strength
interaction: forge-best favors it over `random`). Head 1 attributes
that lift to the card. The lift metric correctly says zero.

The 6-drop bomb is the symmetric case: its head 1 is *attenuated*
(+0.12) because two-thirds of its in-deck appearances are dead games,
even though every individual cast swings the outcome by 20pp.

### Why all three heads, not two

Heads 1, 2, and lift correspond to three genuinely independent
quantities. The four raw counters have three degrees of freedom after
factoring out total scale, so three independent labels are needed for
full coverage. Algebraically:

```
head_2 = played_rate
head_1 = head_2 * (2 * p_play - 1)
lift   = p_play - p_dead
```

From `(head_1, head_2)` you recover `p_play` but not `p_dead`. From
`(head_2, lift)` you recover the gap but not the absolute level. All
three are needed; none is a linear combination of the other two.

This is the opposite of the play/draw split's situation, where the
original head 1 *was* a linear combination of the two split heads and
got dropped.

### Downforce is not a separate head

Defining `downforce = p_play_lose − p_dead_lose` (the loss-side
analogue) gives `−lift` exactly. The signed lift metric already
covers both directions: a positive value means casting helps,
negative means casting hurts (a "trap card" — looks fine but
backfires more often than it helps when resolved).

### Estimated magnitude

Hard to ballpark without running it. The size of the gain depends on
how many cards in the corpus are "auto-include drag" cases versus
"6-drop bombs" — i.e., how often head 1 misattributes deck winrate to
card contribution. The user's earlier point about random pools largely
de-confounding teammate quality applies here too: the lift metric's
biggest absolute wins come from the build-method-induced confound,
which is real but bounded.

### Cost

Low. Aggregation pass goes from 4 counters per card to 4 (same number,
different bucketing — wins/losses split by played/not-played instead
of just summed). One extra regression head, one extra column in
`cards-win-rates.txt`. Same shrinkage logic, applied to the new label.
~30-50 lines.

### Caveat

Lift labels degenerate for cards with extreme played rates. A card
that's cast nearly every time it's drawn (head 2 ≈ 1) has almost no
`not_played` observations, so `p_dead` becomes too noisy to estimate.
Bayesian shrinkage with the same `--shrinkage-k` knob handles this
correctly — the prior pulls those cases toward 0 lift — but the
encoder gets little usable gradient on those cards from this head.
Symmetric problem at head 2 ≈ 0. The middle of the head-2 range is
where lift carries the most signal.

### Dependencies / when to revisit

Conditional on the per-card winnability encoder being implemented.
Stackable with the play/draw split (above), the MLM auxiliary loss
(above), and margin-weighted scorer loss — orthogonal to all three.
Most natural addition once the basic encoder is in service and a first
training round confirms head 1 is the primary signal but is suspected
of conflating ride-along effects with casting effects.

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
  See also `specs/card-winnability-pretraining.md` for one specific
  auxiliary target with a defined data-collection format.
- **Iterative feature distillation** (Phase B encoder → re-cache .npz
  → fresh Phase A scorer → repeat). Surfaced during gen-2 Phase B as a
  test of "are Phase B's gains real or co-adaptation?". Premature until
  Phase B actually helps val_acc on a future cleaner-label corpus.
- **Shallower scorer architecture under Phase B.** Hypothesis: with
  encoder fine-tuning sharing the per-card-feature work, a 2- or
  3-layer scorer might generalize better than 6 layers. Untested
  because Phase B didn't move val_acc; would be revisited if a future
  Phase B does.
