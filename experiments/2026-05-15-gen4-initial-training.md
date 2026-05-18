# Gen-4 Initial Training

*The gen-4 model keeps gen-3's overall pipeline (sealed-trained encoder
+ Set Transformer scorer per specs 010 / 013) but reopens two
optimisation knobs: encoder width (256d → 512d) and a new margin-weighted
Bradley-Terry loss for the scorer. This file rolls up every gen-4 stage
— encoder hparam sweep, scorer training, and (later) deck-building +
match-play evaluation — into one report. Each stage references its
source logs and checkpoints so a later reader can navigate back to the
artifacts.*

## Encoder hyperparameter sweep

*Runs: 2026-05-15 → 2026-05-17. Dataset: the **same** cleaned cards-played
corpus gen3 trained on, just relocated/renamed
(`mtg-models-data/sealed/training-data/matches-bo1/cards-played-bo1-embedding.cleaned.txt`,
974,028-game corpus → 27,983 cards after dropping the 2 with no converted
`.txt`; 22,387 train / 5,596 val, card-disjoint — identical card universe,
split, and per-card labels as gen3). Spec 016 multi-head + MLM encoder. All
runs: 6 layers, `d_model=512`, `--pool-mode attn`, `mlm_weight=0.1`, AdamW
`lr=1e-4`, 5%-warmup-then-constant, 250-epoch cap, `--patience 20`. Sweep
dimensions: `n_heads × n_pool_queries`.*

### TL;DR

- **512d's effect on the picker is genuinely unknown — and the gen3
  precedent is a strong reason not to read too much into encoder
  metrics.** Same corpus and labels as gen3, so the comparison is a
  clean A/B. Per-head `val corr` at 512d is within ±0.01–0.02 of the
  matching-architecture 256d gen3 runs on every head; `val_reg` is flat
  (in fact fractionally *worse*) on the encoder's own held-out cards.
  **But:** in gen3, the 250-epoch 128d and 256d encoders also had
  near-identical `val corr` and near-identical `val_reg` (128d's was
  fractionally *better*, 0.0187 vs 0.0199), yet 256d turned out to be
  meaningfully better for the deck picker downstream. So the encoder
  metrics did *not* predict the picker outcome at the 128d → 256d
  doubling. By the same logic they may not predict it at 256d → 512d
  either. A picker-stage evaluation (encode → train scorer →
  match-play) is necessary; the encoder numbers alone are not enough
  to call 512d dead, and given the gen3 precedent are not enough to
  call 256d dead either.
- **The ~0.007 `val_loss` improvement vs gen3 is entirely the MLM term —
  and that may matter more than it looks.** `val_loss = val_reg +
  0.1·mlm_loss`, `val_reg` flat ⇒ all of the headline drop is MLM
  accuracy (86–87 % → 88–89 %). Consistent with gen3's `d_model`
  ablation showing MLM is the more capacity-hungry head. Naively the
  picker doesn't read MLM directly; but in gen3, going 128d → 256d
  similarly produced a MLM-dominated encoder gain (the regression term
  actually moved *against* 256d, see "Gen4 vs Gen3" below) and 256d
  still won the picker. So the gen4 512d MLM-only gain is *not* safe
  to dismiss as "no deck-building win" — it's the same shape of
  evidence that paid off at the picker once before.
- **`n_heads × n_pool_queries` is essentially flat at 512d across the
  2–8 range.** Eight configurations from 2h/2q up through 8h/8q land
  within 0.0025 `val_loss` of each other and within ±0.02 per-head
  `val corr`. 8h/8q is marginally best by `val_loss` (0.0575); 4h/8q
  marginally worst (0.0600). Pushing to 16h/16q replicates gen3's run
  F finding — fractionally worse on the regression heads (per-head /
  per-pool subspaces too narrow once `d_model` is split that many
  ways), so 8 in each dimension is the soft upper bound.
- **6 layers is the soft depth ceiling at 512d.** Going 6L → 8L at
  8h/8q (G4-9 vs G4-7) leaves every `val_corr` head within ±0.01,
  `val_reg` tied, and MLM slightly *worse* (88.7 % vs 89.3 %) — the
  extra two transformer blocks buy nothing the encoder metrics can
  see. Same pattern gen3's attn-pool depth ladder showed at 256d
  (saturating around 4L–6L); 512d doesn't shift that.
- **Worth taking 512d to the picker evaluation stage anyway.** The
  runtime cost is real (`pooled_dim = 512` ~doubles the scorer's input
  width versus 256d and ~4×s it versus 128d, with a roughly quadratic
  hit on the scorer body), so the bar 512d has to clear at the picker
  stage is *not just "ties 128d/256d"* — it has to deliver enough
  match-play improvement to justify the inference cost. But proving the
  null at the encoder stage isn't the same as proving the null at the
  picker stage; one scorer training + eval per encoder width would
  settle this.

### Sweep results

All runs: 512d · `--pool-mode attn` · `mlm_weight=0.1`, on the same
corpus gen3 used; 6 layers unless noted.

| run | layers | heads | pool-q | best ep | val_loss | val reg | MLM ppl / acc | score_play | score_draw | played_rate | cast_lift | color_lift W/U/B/R/G |
|-----|-------:|------:|-------:|--------:|---------:|--------:|---------------|-----------:|-----------:|------------:|----------:|----------------------|
| G4-1 | 6 | 4 | 4 | 93 | 0.0579 | 0.0195 | 1.5 / 88.6% | +0.57 | +0.56 | +0.63 | +0.59 | .42/.54/.46/.49/.48 |
| G4-2 | 6 | 2 | 2 | 110 | 0.0593 | 0.0202 | 1.5 / 88.3% | +0.58 | +0.56 | +0.62 | +0.59 | .42/.54/.45/.50/.50 |
| G4-3 | 6 | 8 | 2 | 107 | 0.0588 | 0.0196 | 1.5 / 88.6% | +0.57 | +0.58 | +0.64 | +0.61 | .43/.54/.47/.50/.50 |
| G4-4 | 6 | 2 | 8 | 147 | 0.0584 | 0.0198 | 1.5 / 89.0% | +0.56 | +0.55 | **+0.65** | +0.59 | .43/.54/.47/.50/.50 |
| G4-5 | 6 | 4 | 8 | 80 | 0.0600 | 0.0197 | 1.5 / 88.1% | +0.56 | +0.55 | +0.63 | +0.59 | .41/.54/.47/.49/.49 |
| G4-6 | 6 | 8 | 4 | 97 | 0.0590 | 0.0197 | 1.4 / 88.4% | +0.58 | +0.56 | +0.64 | +0.60 | .44/.54/.46/.50/.50 |
| **G4-7** | 6 | **8** | **8** | 137 | **0.0575** | 0.0201 | 1.5 / **89.3%** | +0.57 | +0.56 | +0.63 | +0.59 | .43/.54/.45/.50/.49 |
| G4-8 | 6 | 16 | 16 | 97 | 0.0587 | 0.0202 | 1.5 / 88.6% | +0.56 | +0.56 | +0.62 | +0.59 | .43/.54/.46/.50/.49 |
| G4-9 | **8** | 8 | 8 | 99 | 0.0583 | 0.0199 | 1.5 / 88.7% | +0.56 | +0.55 | +0.63 | +0.59 | .43/.54/.46/.50/.49 |

#### Heads × pool-queries: insensitive

- `val_loss` spread: 0.0025 (0.0575–0.0600). `val_reg` spread: 0.0007
  (0.0195–0.0202). Both inside batch-to-batch noise.
- Every signed `val corr` head spans ±0.02 across the entire table; the
  colour-head average is 0.47–0.49 everywhere.
- Best-epoch numbers vary wildly (80–147) — `--patience 20` stops kick in
  at different points depending on the loss-landscape walk, but the
  endpoint plateau is the same.
- Squinting for trends: going from 2 heads → 8 heads buys ~+0.01–0.02 on
  the colour heads on average (e.g. G4-2 W=.42 vs G4-6 W=.44); pool-query
  count alone (2 vs 8, fixing heads) moves nothing systematically.
  Pushing further to **16h/16q (G4-8)** doesn't continue the colour-head
  trend — its colour avg (~0.48) ties 8h/8q (~0.48), and `played_rate`
  and `score_play` are fractionally *lower* (+0.62 / +0.56 vs the
  middle of the table's +0.63–0.64 / +0.57–0.58). This replicates the
  gen3 256d sweep's run F finding (16h/16q strictly worse than 4h/4q
  on the regression heads), suggesting the per-head / per-pool-query
  dimension is just too narrow once `d_model` is split sixteen ways.
  None of the gaps is large enough that a single replication couldn't
  erase it.
- 8h/8q wins on `val_loss` (.0575) but ties on `val corr`. 4h/8q is the
  worst on `val_loss` (.0600) and also stops earliest (epoch 80) — likely
  a less stable optimisation trajectory rather than a real architectural
  defect (small batch-to-batch noise on the val curve trips the patience
  counter earlier).

So the same conclusion as gen3's 256d sweep: at this capacity, `n_heads`
and `n_pool_queries` are reallocations of a fixed slot-width budget, not
capacity additions, and going past 8 in either dimension starts to hurt
(too-narrow per-head / per-pool subspaces). The picker doesn't notice
the reallocation across the 2–8 range; 16/16 is the only configuration
where the regression heads show consistent mild degradation.

#### Depth: 6L is the soft ceiling

G4-9 holds 8h/8q/512d fixed and pushes layers from 6 → 8. Result:
`val_loss` 0.0583 vs G4-7's 0.0575 (slightly *worse*), `val_reg` 0.0199
vs 0.0201 (tied), MLM acc 88.7 % vs 89.3 % (slightly worse), every
`val_corr` head within ±0.01 (`score_play` +0.56 vs +0.57; everything
else identical). Adding two more transformer blocks at 512d buys
nothing on any encoder metric and may cost a hair on MLM. This is the
same pattern gen3 showed under attn pooling — depth past ~4L tightens
into the plateau, and beyond 6L the regression heads simply don't read
the extra layers. So **6 layers is the soft ceiling at 512d**; 8 is
already on the decay side. (No 1L/2L/4L runs at 512d to pin down the
lower bound, but the gen3 attn-pool depth ladder placed it at 2L; no
reason to expect 512d shifts that.)

### Gen4 vs Gen3 — clean A/B at fixed corpus

Same corpus, same labels, same val split — only `d_model` differs. The
nearest architectural twins:

| architecture | gen | d_model | val_loss | val_reg | MLM acc | score_play | played_rate | cast_lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 6L · 4h/4q · attn (gen3 H, G4-1) | gen3 | 256 | 0.0649 | 0.0194 | 86.6 % | +0.57 | +0.63 | +0.60 |
| 6L · 4h/4q · attn (gen3 H, G4-1) | gen4 | 512 | 0.0579 | 0.0195 | 88.6 % | +0.57 | +0.63 | +0.59 |
| 6L · 2h/2q · attn (gen3 N, G4-2) | gen3 | 256 | 0.0664 | 0.0192 | 86.0 % | +0.58 | +0.63 | +0.61 |
| 6L · 2h/2q · attn (gen3 N, G4-2) | gen4 | 512 | 0.0593 | 0.0202 | 88.3 % | +0.58 | +0.62 | +0.59 |

- **`val_reg` is flat.** 0.0195–0.0202 at 512d vs 0.0192–0.0194 at 256d —
  the 512d numbers are if anything *slightly worse*, well inside
  batch-to-batch noise. On this metric, the regression heads gain
  nothing from the doubled width.
- **`val corr` is flat.** Every signed head within ±0.01–0.02 of its 256d
  twin; `cast_lift` is fractionally *lower* at 512d. With identical
  labels this is a clean A/B at the encoder stage: 256d → 512d on the
  regression metrics is the flat part of the curve.
- **A precedent from gen3 against over-reading these flat numbers —
  and a hint about *why* the encoder might still matter to the picker.**
  Gen3 also produced "encoder-regression-flat" numbers between 128d and
  256d on the long 250-epoch runs (identical `val corr` per head,
  `val_reg` of 0.0187 vs 0.0199 — i.e. fractionally favouring 128d).
  Decomposing the 0.0117 `val_loss` gap that *did* exist between them:
  the regression term contributed **−0.0012** (favoured 128d), while
  the 0.1-weighted MLM term contributed **+0.0128** in favour of 256d
  (raw `mlm_loss` 0.5293 vs 0.4011; MLM accuracy 84.3 % vs 87.8 %).
  Essentially **all** of 256d's encoder-stage advantage was in MLM.
  Yet **256d turned out to be meaningfully better at the deck picker**.
  That's strong circumstantial evidence that what the picker actually
  benefits from is the richer text / rules-language representation
  that drives MLM accuracy — a signal the linear regression heads
  can't fully read out (so it doesn't show up in `val corr` /
  `val_reg`) but the Set Transformer scorer can. By that reading,
  the gen4 256d → 512d MLM-accuracy gain (~2 pp, 86–87 % → 88–89 %)
  is the most plausible mechanism through which 512d *could* still
  help the picker, even though the regression metrics are dead flat.
- **The headline `val_loss` drop (~0.007 lower at gen4) is entirely the
  MLM term.** `val_loss = val_reg + 0.1·mlm_loss`, `val_reg` flat ⇒ the
  ~0.007 drop is the 0.1-weighted MLM improvement. MLM accuracy gains
  ~2 pp (86–87 % → 88–89 %), consistent with gen3's `d_model` ablation
  showing MLM scales with width more than the regression heads do.
- **MLM gains on a side task; whether the encoder's *representation* of
  cards changes in ways the scorer can use is a separate question** —
  the scorer-stage sweep below answers that empirically.

### Implications

1. **At the encoder stage, gen3's "widening `d_model` past 128–256
   doesn't move `val corr`" pattern continues at 512d.** Doubling
   256→512 sits on the same flat part of the curve as 128→256 did at
   the encoder stage. The gen3 effective-rank probe showed both 128d
   and 256d use only ~3–9 of their dimensions in practice; doubling
   to 512d plausibly gives the optimiser more slack rather than more
   useful capacity, but the PCA probe hasn't been re-run at 512d yet,
   and the gen3 picker outcome (256d > 128d despite encoder parity)
   says "more slack" isn't necessarily the same as "no picker payoff".
2. **2h/2q stays the parsimonious default on encoder-side metrics.**
   Halving heads/queries to 2/2 ties 4/4 and 8/8 on every encoder
   metric measured here. Whether this carries over to the picker stage
   is — by the same gen3 precedent — not safe to infer.
3. **Take 512d (and 128d) to the scorer; let match-play decide.** The
   gen3 case where encoder-flat 128d vs 256d turned out *not* to be
   picker-flat is the clearest reason not to ship on encoder metrics
   alone. Plan: train scorers on top of (at minimum) 128d, 256d, and
   512d gen4 encoders, run `evaluate-scorer` vs `forge-best`, and let
   the match-play numbers rank them. The deployment-economics argument
   (smaller `pooled_dim` → smaller scorer input → ~4× faster scorer
   body, once `card_embedding_layout` reads the encoder's `pooled_dim`
   instead of hardwiring `2·d_model`) only kicks in once a smaller
   width is shown to *also* win or tie at the picker stage; otherwise
   the runtime saving buys a worse picker.

## Scorer training — picker-stage hparam sweep

*Runs: 2026-05-17 → 2026-05-18. Logs under
[`models/sealed/scorer/gen-4/`](../models/sealed/scorer/gen-4/). Dataset:
`output/sealed/match-outcomes-all.txt` (70,134 matches → 56,107 train /
14,027 val, 27,671 unique cards). Scorer architecture fixed at
6-layer / 4-head / 4-seed Set Transformer; `--lr 1e-5`, `--batch-size 64`,
`--dropout 0.2`, AdamW with max-norm-1.0 grad clipping per param group;
250-epoch cap with `--patience 20` early stop on val_acc. Three
orthogonal axes swept:*

- *encoder width — 256d (`cardsfolder-256/`, the gen-4 8h/8q sealed
  encoder retrained at `d_model=256` to match gen-3's production
  width) vs 512d (`cardsfolder-512/`, the gen-4 G4-7 8h/8q 512d
  encoder). Scorer input width = `pooled_dim + FEATURE_COUNT`, i.e.
  288 vs 544.*
- *scorer body — small (`ff1088 / mlp256`, gen-3 default) vs large
  (`ff2176 / mlp512`, doubled to keep the FF / MLP : input-width
  ratio constant when the encoder doubles).*
- *margin weighting — `--margin-weighting {none, linear, log}` (the
  flag added in commit `b2fd816`). Weighted runs use the
  `_mwlin` / `_mwlog` suffix in the checkpoint name.*

*Margin weighting scales val_loss (its magnitude is no longer
comparable across margin modes), but val_acc keeps the unweighted
"% of pairs ranked correctly" definition and stays apples-to-apples.
Early-stop selection is val-acc-driven throughout.*

### Sweep results

| run | encoder | scorer ff/mlp | margin  | best ep | best val_acc | val_loss @ best | end ep |
|-----|--------:|---------------|---------|--------:|-------------:|----------------:|-------:|
| S1     | 256d | 1088 / 256 | none    | 11 | 0.7165 | 0.5488 | 31 |
| S2     | 256d | 1088 / 256 | linear  | 14 | 0.7193 | 1.2654 | 34 |
| S3     | 256d | 1088 / 256 | log     | 21 | 0.7169 | 0.6290 | 41 |
| S4     | 512d | 1088 / 256 | none    | 16 | 0.7190 | 0.5500 | 36 |
| S5     | 512d | 2176 / 512 | none    |  6 | 0.7190 | 0.5410 | 26 |
| S6     | 512d | 2176 / 512 | linear  | 11 | 0.7189 | 1.2539 | 31 |
| **S7** | 512d | 2176 / 512 | **log** | 13 | **0.7200** | 0.6393 | 29 |

Spread top-to-bottom: 0.0035 val_acc (0.7165 → 0.7200). Every gap in
the table is under 0.5 pp — well inside the run-to-run / seed noise
the gen-3 sweep already established at this corpus size. The
discussion below threads the same caveat through every cell.

### Encoder width: 256d ties 512d at the picker

The encoder TL;DR's open question — does 512d's MLM-driven encoder gain
translate into a picker-stage win? — resolves to *no*:

- Unweighted, same scorer body: S1 (256d, 0.7165) → S4 (512d, 0.7190) ⇒
  +0.25 pp.
- Unweighted, scorer body scaled with encoder: S1 (256d/small, 0.7165)
  → S5 (512d/big, 0.7190) ⇒ +0.25 pp.
- Log-weighted, scaled scorer: S3 (256d, 0.7169) → S7 (512d, 0.7200) ⇒
  +0.31 pp.

Every cross-encoder gap sits at 0.25–0.31 pp, single-seed, on a
sweep where the overall spread is 0.35 pp. The encoder-stage TL;DR
called this question explicitly open ("encoder metrics did *not*
predict the picker outcome at the 128d → 256d doubling. By the same
logic they may not predict it at 256d → 512d either"); the
picker-stage answer is that **they did predict it this time** —
512d's encoder-stage MLM gain does not buy a picker-stage win
distinguishable from noise.

The deployment-economics arm of the encoder TL;DR now bites: 512d's
`pooled_dim = 512` doubles the scorer's input width and ~4×s its body
compute relative to 256d, and that cost has bought a sub-noise val_acc
swing. **256d is preserved as the gen-4 production encoder by
exclusion** — 512d had to beat 256d at the picker to be worth keeping,
not just tie. The gen-3 precedent that motivated the 512d sweep
(MLM gains paid off downstream at 128d → 256d even when regression
heads were flat) does not replicate at 256d → 512d.

### Scorer body: doubling FF/MLP buys nothing

S4 (512d / small ff1088 / mlp256) and S5 (512d / big ff2176 / mlp512),
both unweighted, both reach **exactly** val_acc = 0.7190. The bigger
body fits the training set faster (peak at epoch 6 vs 16) and reaches a
fractionally lower val_loss at its peak (0.5410 vs 0.5500) but
generalises to the same point. Doubling FF and MLP widths to keep
ratio with the doubled encoder input adds parameters that the
scorer does not put to work at 70k matches.

This is the scorer-side analogue of the encoder-sweep finding that 8L
loses to 6L at 512d: extra capacity past the gen-3 baseline arch is
dead at this corpus size. The binding constraint is data, not the
scorer's hidden-layer width.

### Margin weighting: log has a faint pulse, linear is mixed, both inside noise

- **256d × small scorer**: linear S2 = 0.7193 (+0.28 pp over unweighted
  S1 = 0.7165); log S3 = 0.7169 (+0.04 pp). Linear edges; log null.
- **512d × big scorer**: linear S6 = 0.7189 (−0.01 pp vs unweighted S5 =
  0.7190); log S7 = 0.7200 (+0.10 pp). Linear null; log edges.

The sign is inconsistent across encoder widths (linear wins at 256d but
flats at 512d; log wins at 512d but flats at 256d), which is the classic
shape of a treatment lost in seed noise. S7's 0.7200 is the best run in
the table by 0.07 pp over the next best (S2's 0.7193); the spec's
"+1 to +3 pp" estimate assumed exhausting the data-side lever first,
so a sub-pp swing at this corpus size is in line with the prior
estimate of "useful but not transformative".

**Default the production gen-4 scorer to unweighted.** Reopen if a
future run with more match data and exhausted data-side levers still
leaves a measurable gap to the irreducible Bo7 oracle noise floor.

### Overfitting accelerates under margin weighting

Every run overfits hard: train_acc climbs to 0.78–0.84 while val_acc
plateaus near 0.71; train_loss drops 20–30 % below its best-val-acc
value while val_loss climbs. Margin-weighted runs overfit *faster*
because the 4× gradient on 4-0 matches lets the scorer memorise those
matches first. Train- vs val-loss readings at the run's last epoch:

| run | scenario              | train_loss (end) | val_loss (end) | train/val |
|-----|-----------------------|-----------------:|---------------:|----------:|
| S1  | 256 / small / none    | 0.49 | 0.56 | 0.88 |
| S2  | 256 / small / linear  | 1.08 | 1.36 | 0.80 |
| S5  | 512 / big   / none    | 0.40 | 0.65 | 0.62 |
| S6  | 512 / big   / linear  | 0.75 | 1.74 | 0.43 |
| S7  | 512 / big   / log     | 0.41 | 0.78 | 0.53 |

Bigger scorer (S5–S7) overfits faster than smaller (S1–S2), and
margin-weighting compounds the effect. Log dampens it relative to
linear (S7 0.53 vs S6 0.43) but doesn't reverse it. None of this
moves val_acc — best-val-acc epochs are all in the first quarter of
the run, well before the train/val divergence opens up.

### Picker-stage conclusions

1. **Ship gen-4 production scorer as 256d encoder × small scorer ×
   unweighted (S1)** — the cheapest configuration, tied for val_acc
   with everything else under noise. The downstream deck-building /
   match-play eval starts from this checkpoint.
2. **Shelve the 512d encoder.** Its 2 pp MLM accuracy win at the
   encoder stage was the only non-trivial signal it had, and it does
   not transfer to the picker. The gen-3 precedent (MLM-only gain
   paid off downstream) does not replicate here.
3. **Shelve margin weighting at this corpus size.** The +0.28 pp
   linear gain at 256d/small is the biggest swing in the sweep and is
   single-seed; the +0.10 pp log gain at 512d/big is the next-best and
   is dwarfed by the seed noise floor implied by the rest of the
   table. The implementation (`b2fd816`) is preserved for the next
   gen iteration when match-data volume has substantially grown.
4. **Doubling scorer FF/MLP to match a doubled encoder input is dead.**
   Future-gen scorer arch changes should change *something
   structural* (depth, attention pattern, set-pool order) rather than
   widen FF/MLP — width is not the bottleneck.

## Open questions / next steps

- **Deck-building from S1 checkpoint.** Run `sealed build-decks` on a
  sealed-pool sweep with the gen-4 256d-encoder scorer and inspect
  pool-level deck quality and shape (colour count, curve, mana
  symbols) versus gen-3. This is the first place the gen-3 →
  gen-4 production change can show up qualitatively even if val_acc
  was flat.
- **Match-play vs `forge-best` and vs gen-3.** Same pool sweep, two
  matchups: gen-4 (S1) vs `forge-best` (the gen-3 baseline matchup,
  to compare directly to gen-3's 47/48-pool headline), and gen-4 vs
  gen-3 in self-play (the apples-to-apples test, with both scorers
  building from the same pools). The +0.25–0.31 pp val_acc gap to
  the noise-eclipsed cells should be ignored; the relevant
  comparison is whether gen-4 (S1) ties or beats gen-3's match-play
  number, since the only architectural change at the production
  width is the encoder retrain.
- **Effective-rank / PCA probe on a 512d checkpoint.** Cheap, direct
  confirmation that 512d's extra width is mostly unused — the
  mechanistic explanation for why the picker doesn't notice the
  doubling. Wasn't needed to make the gen-4 production call (the
  scorer-stage tie is sufficient) but completes the gen-3 →
  gen-4 width-scaling story.
- **Replicate S7 (512d/big/log) and S2 (256d/small/linear).** Both
  are the best-in-axis margin-weighting runs and both sit in the
  noise band; a single replicate per configuration would tell us
  whether either gain is real before fully closing the file on
  margin weighting.
