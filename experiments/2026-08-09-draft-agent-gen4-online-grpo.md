# Draft agent (gen-4) — online GRPO from the promoted gen-3 candidate

## Context — what gen-3 left us with

Gen-3 closed with one setting to carry forward and one question left open
([`2026-06-15-draft-agent-gen3-online-grpo-design.md`](2026-06-15-draft-agent-gen3-online-grpo-design.md),
*Where this leaves gen-3*).

Field at T is the training field: every ML agent samples at the rollout temperature, not
just the learner. It produced gen-3's best candidate and its tightest score distribution.

The temperature itself was unsettled. `T = 3` is the only value that holds the exploration
band of perplexity 2–3 and off-argmax 25–40 %, but it ran for 28 rounds under field at T.
`T = 2` lost the band mid-run and still produced the best candidate. The band and the
outcome disagreed.

Gen-4 warm-starts every run from the promoted gen-3 candidate
(`gen3/temperature-on-all-agents/lr1e-5_t2_20260805_221050.pt`) and uses it as the anchor as
well. Learner and anchor being the same weights turns out to be the most useful measuring
instrument in the generation, for reasons unrelated to why it was chosen (*The round-9 best
is noise*, below).

## The runs

Four runs on 2026-08-07 to 2026-08-09, all `lr 1e-5`, all from the same base, all on the mix
`gen4:3,gen3a:2,gen3c:1,gen1:1,forge-full:1` with `--anchor gen3a`. The per-label
`--agent-temp` from spec 021 phase 9 makes the learner/field temperature split expressible
for the first time, and the fourth run is the first to use it.

| Run | learner T | field T | Rounds | Duration | Best margin (round) | Final margin | Checkpoint taken |
|---|---|---|---|---|---|---|---|
| `t2all_nodecay` | 2.0 | 2.0 | 83 | 1h42m | +0.546 (r72) | +0.379 | r72 |
| `t2all_decay0.3` | 2.0 | 2.0 | 1268 | 22h39m | +0.831 (r312) | +0.468 | r312 |
| `t3all_decay0.3` | 3.0 | 3.0 | 568 | 9h54m | +0.676 (r208) | +0.206 | r58 |
| `t3learner_t2field` | 3.0 | 2.0 | 177 | 4h07m | +0.391 (r68) | +0.002 | r68 |

The checkpoint column is the round whose weights were kept and yardsticked, read from each
file's `epoch` field. Three files hold their run's best-margin checkpoint. The
`t3all_decay0.3` file does not: it holds round 58, whose margin was +0.618, and its run went
on to a better round 208. That candidate is therefore the least-trained of the four as well
as the only one not represented by its own best round.

`t2all_nodecay` ran against a field that also contained a `gen3b` seat, so its in-run margins
do not compare with the other three. Its yardstick figures do, being measured against a
common field.

All four peak and decline, as every gen-3 run did. Three end at less than half their best;
the fourth ends at zero.

Every run's rollouts were retained this time, one `-drafts.jsonl` per run — the
instrumentation gen-3 asked for and lost. They are the input every hypothesis below needs.

## The yardstick

All four checkpoints are now measured. Each was taken through two 500-draft argmax runs: one
on the fixed mix `gen4:2,gen1:1,forge-full:1`, and one head-to-head against the promoted
gen-3 incumbent on `gen4:1,gen3:1`.

Every gen-4 candidate beats every reference by a wide margin, and every one of them beats
the gen-3 incumbent it was fine-tuned from. The gen-3 incumbent's own yardstick, recomputed
on the same estimator, is included as the row to beat.

| Checkpoint | vs gen-1 | vs `forge-full` | vs gen-3 incumbent |
|---|---|---|---|
| `t2all_decay0.3` | +1.380 ± 0.046 | +1.555 ± 0.052 | +0.634 ± 0.031 |
| `t2all_nodecay` | +1.328 ± 0.049 | +1.519 ± 0.050 | +0.645 ± 0.031 |
| `t3learner_t2field` | +1.276 ± 0.052 | +1.384 ± 0.045 | +0.600 ± 0.033 |
| `t3all_decay0.3` | +1.152 ± 0.050 | +1.337 ± 0.056 | +0.467 ± 0.030 |
| *gen-3 incumbent* | *+0.824 ± 0.046* | *+0.920 ± 0.044* | — |

Each figure is a mean over pods of (mean candidate seat − mean reference seat) in that pod,
with the standard error clustered on set code. Pods are the unit because seats in a pod share
a set and a card supply; sets are the cluster because 500 drafts draw only about 170 distinct
sets, so pods repeat sets and a seat-level standard error would be too narrow by roughly
half.

The two routes to a gen-4-versus-gen-3 comparison agree. Differencing the first column
against the incumbent's own row gives +0.50, +0.56, +0.45 and +0.33; the direct head-to-head
gives +0.65, +0.63, +0.60 and +0.47. The levels differ by about 0.14 but the ordering is
identical, which is what the comparison is for.

The per-seat means that `analyze-generated-decks` prints understate all of this. Those pool
every seat regardless of pod composition, and composition moves every score in the pod: a pod
that happens to draw six gen-4 seats scores worse for everyone in it, and it contributes six
seats to gen-4's mean against two to gen-1's. On `t2all_nodecay` the printed means give +1.09
where the pod-paired estimate gives +1.33. The ordering is unaffected, so gen-3's figures
remain valid for ranking; the levels there are low for the same reason. *Why every frozen
label declines*, below, measures the effect and shows the bias scaling with the candidate's
own strength — the stronger the candidate, the more the printed mean understates it.

### What separates and what does not

| Contrast | vs gen-1 | vs gen-3 |
|---|---|---|
| `t2all_decay0.3` − `t2all_nodecay` | +0.05 ± 0.07 | −0.01 ± 0.04 |
| `t2all_nodecay` − `t3learner_t2field` | +0.05 ± 0.07 | +0.05 ± 0.05 |
| `t3learner_t2field` − `t3all_decay0.3` | +0.13 ± 0.07 | +0.13 ± 0.05 |
| `t2all_nodecay` − `t3all_decay0.3` | +0.18 ± 0.07 | +0.18 ± 0.04 |

Only the contrasts involving `t3all_decay0.3` separate, and each is the same size in both
corpora, which were drawn independently. The two `T = 2` runs and `t3learner_t2field` are a
three-way tie. The narrowest of the four, `t3learner_t2field` against `t3all_decay0.3`,
separates against gen-3 and falls just short against gen-1.

Two conclusions follow directly. `T = 3` applied to every agent is the worst configuration
tried, settling gen-3's open question against the exploration band: the band pointed at
`T = 3` and the yardstick prefers `T = 2`. And the 22-hour run is not distinguishable from
the 1h42m one. `t2all_decay0.3` trained 312 rounds to `t2all_nodecay`'s 72 and bought
+0.05 ± 0.07 for it.

The `t3all_decay0.3` conclusion carries one caveat, stated above: its yardsticked weights
are round 58, not its run's round-208 best. Its deficit could in principle be undertraining
rather than temperature. Two things argue against that reading. Its round count is within
15 rounds of `t3learner_t2field`, which beats it in both corpora. And the in-run margin that
would rank round 208 above round 58 is shown below to have no ranking power at all.

### A note on the corpora

The two `t2all_decay0.3` yardstick files are not what their names imply. Both accumulated
earlier runs' records, because `output/draft/yardstick-drafts.jsonl` was reused without being
cleared: the `v-forge` file holds 1000 records from two runs and the `v-gen3` file holds 1500
from three. The `analyze-generated-decks` output captured in those two `-yardstick-*.log`
files is therefore a mixture of checkpoints, and reports 4034 gen-4 seats where 500 drafts
can yield about 2000. Every figure in this document is computed from the correct `run_id`
only. The other six corpora hold exactly their own run.

## `deck_score` does predict winning

Every ranking in this document is a `deck_score` difference, and gen-3 carried the assumption
that such differences turn into games won without ever testing it on drafted decks. They do,
and at close to the rate the sealed pipeline showed.

`python -m draft play-draft-games`
([`../specs/2026-08-09-draft-game-evaluation.md`](../specs/2026-08-09-draft-game-evaluation.md),
feature `022-draft-game-evaluation`) was run over the `t3learner_t2field` yardstick corpus:
1001 best-of-seven matches in 2h14m on 12 workers, each pairing two seats of one recorded pod,
mirrors excluded, with `--forge-native-fraction 0.5` diverting half the `forge-full` seats to
Forge's own sealed builder. Pairing inside a pod controls set and pool quality by construction.
Bo7 match win rate over all matches a label appeared in:

| Label | matches | Bo1 | Bo7 |
|---|---|---|---|
| gen-4 | 767 | 65.4 % | 67.7 % |
| gen-1 | 576 | 43.8 % | 43.4 % |
| `forge-full` | 311 | 43.4 % | 40.5 % |
| `forge-native` | 348 | 32.2 % | 30.5 % |

The ordering is the yardstick's, on decks the scorer never saw played. Reading it against the
score gaps measured in the same corpus turns that into a rate. Game win rate is the right
column for calibration, since a Bo7 amplifies whatever per-game edge exists.

| Matchup | score gap | game win rate | games | Bo7 match win rate | matches |
|---|---|---|---|---|---|
| gen-4 over `forge-full` | +1.384 | 60.6 % | 964 | 68.0 % | 178 |
| gen-4 over gen-1 | +1.276 | 58.9 % | 2124 | 64.9 % | 393 |
| gen-1 over `forge-full` | +0.105 | 53.2 % | 449 | 58.5 % | 82 |

A line through the origin fits those three at 7.4 points of game win rate per unit of
`deck_score`. The sealed pipeline's analogue, measured on sealed pools rather than draft pods,
is about 7.8 points per unit. Two independently constructed pipelines agreeing within half a
point on the exchange rate is stronger evidence than either alone.

The third row carries almost none of that fit and should not be read as if it did. Its score
gap is +0.105 against a standard error of 0.049, so it says two roughly equal decks win roughly
equally often, which is consistent with the line without constraining its slope. The two
informative rows imply 7.0 and 7.7 points per unit on their own.

`forge-native` is the one label with no `deck_score` at all, being rebuilt from the pool at game
time, and it supplies a clean read on the deck builder. Its seats and the `forge-full` seats
are the same drafting agent on the same drafted pools, differing only in who assembles the 40
cards. This project's picker and simulated-annealing builder beats Forge's own sealed builder
68.6 % of matches head to head, on 51 matches with a cluster-robust standard error of 6.4
points. Ten points of the gap between `forge-full` and `forge-native` in the first table is
builder, not drafting.

Two of this generation's readings survive contact with games. Creature count tracks winning
inside every label — gen-4 wins 40.6 % of matches with 13 creatures or fewer and 72.3 % with 20
or more — which is the direction the composition analysis assumed. And gen-4's win rate is flat
across colours, 66.6 % to 67.6 % in each of WUBRG, where both references swing 6 to 8 points
between their best and worst colour. A miscalibrated colour prior would show as gen-4 winning
less in the colours it over-plays; it does not show, which is weak evidence for the prior being
earned rather than a tic. It is not a test of whether the prior is optimal, since gen-4 chooses
its own colours and never has to defend the choice.

What this does not do is rank the four candidates. One checkpoint was played, and not the
promotable one — `t3learner_t2field` is third of four on the yardstick, so 67.7 % is a lower
bound on what the generation reaches rather than its headline. The exchange rate is what
generalises, and it says the +0.23 that separates the best candidate from the worst is worth
about 1.7 points of game win rate: real, and small enough that separating adjacent candidates
on games alone would need thousands of matches.

The three runs sharing a field decompose the temperature effect, because they change one
temperature at a time and their checkpoints come from within 15 rounds of each other.

| Contrast | What changes | Effect vs gen-1 | Effect vs gen-3 |
|---|---|---|---|
| `t2all_nodecay` → `t3learner_t2field` | learner 2 → 3, field held at 2 | −0.05 ± 0.07 | −0.05 ± 0.05 |
| `t3learner_t2field` → `t3all_decay0.3` | field 2 → 3, learner held at 3 | −0.13 ± 0.07 | −0.13 ± 0.05 |

Raising the learner's own sampling temperature is free. Raising the field's costs, and the
cost is the whole of the `T = 3` deficit.

The mechanism is the one the gen-3 spec proposed and gen-3's results appeared to refute
(§ 8.1). A field sampling at `T` sometimes passes a card it should have kept. The learner
then trains against packs a properly-playing field would never have handed it, and what it
learns from them does not transfer. Gen-3 compared field-at-T against field-at-argmax and
found sampling better; gen-4 varies the amount of sampling and finds less of it better. Both
are consistent with an optimum in the interior, near `T = 2`.

One incidental measurement supports the first row. In `t3learner_t2field` the learner and the
anchor hold identical weights and differ only in temperature, 3.0 against 2.0. Their round-0
margin is −0.042, so running the same policy hotter barely moves the deck it produces.

## What the in-run metrics did, and did not, tell us

Four yardsticks now exist where the last generation had partial coverage, so the in-run
metrics can be scored against them rather than trusted.

### The anchor margin does not rank checkpoints

Ranking the three comparable runs by their best in-run margin gives `t2all_decay0.3`,
`t3all_decay0.3`, `t3learner_t2field`. The yardstick gives `t2all_decay0.3`,
`t3learner_t2field`, `t3all_decay0.3`. The metric that run control keys off inverts the two
runs it was asked to separate, and it does so on the one contrast that the yardstick resolves
cleanly.

The margin is not useless — every run's margin is positive and every run's candidate does
beat the field. It carries no information about *how much*.

### The LR annealing decayed once and then silently stopped

All three armed runs took exactly one decay, to 3.0e-6, and never moved again.
`t2all_decay0.3` took its decay at round 29 and sat at 3.0e-6 for every round after it.

The cause is arithmetic in `_PlateauLR.can_decay()`
(`src/draft/application/train_draft_agent.py:463`), which refuses a decay that would land
below `min_lr` rather than clamping to it:

```
return self.base_lr * self.factor ** (self.decay_count + 1) >= self.min_lr
```

With `--lr 1e-5 --lr-decay-factor 0.3 --min-lr 1e-6` the ladder is `1e-5 → 3e-6 → 9e-7`, and
`9e-7 < 1e-6`, so the second decay is refused forever. The effective floor is 3e-6; the
startup echo advertises 1.0e-06, which is unreachable. Every existing `_PlateauLR` test uses
`factor = 0.1` with a floor an exact power of ten below the base, so the ladder always lands
on the floor and the truncation never shows. `0.3` is the first factor used that does not
divide the base-to-floor ratio evenly.

Two things made it silent. `_maybe_decay` returning `None` is indistinguishable from "not
stalled yet", so nothing logs the refusal. And `--patience` was armed in none of the four
runs, so a run that can no longer anneal also cannot stop: `t2all_decay0.3` spent its last
956 rounds with no new best, no further decay and no exit.

The decay is applied to the live weights rather than to `best_*.pt`, so there is no rollback,
no moment reset and no re-warmup. That is the right convention here, since the best it would
roll back to is the noise described next.

### The round-9 best is noise, and the noise floor is measurable

In three of the four runs the learner and the anchor are the same checkpoint at the same
temperature. Generation precedes the update, so round 0's drafts come from two bit-identical
policies and the true margin at round 0 is exactly zero. Those three runs report −0.049,
+0.781 and +0.136.

That is a free calibration of the metric. Three draws of a quantity whose true value is zero
give a root-mean-square of 0.46 over a 10-draft window, so about ±0.15 once the 100-draft
window fills. The +0.781 that `t2all_decay0.3` drew is the tail of that distribution, not a
typical reading.

It is also the run that paid for it. Its margin then falls monotonically — 0.781, 0.585,
0.558, 0.412, 0.373, …, 0.218 — which is not the learner regressing but a cumulative mean
regressing away from a lucky first ten drafts. The window fills at round 9, and the
window-full guard admits it:

```
return len(self._window) >= self._maxlen
```

The guard's docstring says it prevents "an early lucky round" from pinning the run's best. It
does not. Excluding rounds 0–8 does not exclude rounds 0–8's drafts, which are 100 % of round
9's window. It filters the reporting round, not the contaminated data. The window holds no
fill-period drafts at all only from round 19.

The consequence is concrete. `t2all_decay0.3` pinned its best at round 9 at +0.218, inside one
sigma of zero. The stall counter started there, rounds 10–29 ran at 0.00–0.18 and never
cleared it, and the single available decay fired at round 29 while the policy was learning
perfectly well. The first genuine best came at round 70, more than three times the decay
patience later. The run went on to +0.831. The other three runs drew low round-9 windows
(−0.042, +0.029, −0.131) and cleared them within three rounds, so only the run with the
unlucky draw was affected — which is what a noise floor does.

### Policy loss cannot select checkpoints

The obvious alternative to the margin is the quantity being optimised. It is worse, and the
sign is the interesting part.

| Run | corr(policy_loss, margin) | corr(policy_loss, H) |
|---|---|---|
| `t2all_nodecay` | +0.250 | −0.890 |
| `t2all_decay0.3` | +0.333 | −0.845 |
| `t3all_decay0.3` | +0.202 | −0.718 |
| `t3learner_t2field` | +0.455 | −0.591 |

Positive against the margin in all four runs. Loss is a minimisation target, so a useful
selector would correlate negatively. Over `t2all_decay0.3`, `argmin(loss)` is round 1193
(margin +0.152) where `argmax(margin)` is round 312 (+0.831).

The mechanism is structural. `assign_advantages` standardises each round's rewards to mean 0
and std 1, and the loss is `−mean(A·logπ)`, so with `mean(A) = 0` it is `−Cov(A, logπ)`,
bounded by `σ_logπ`. The per-round standardisation destroys all absolute performance
information, and what remains is an entropy thermometer — the second column, and
`r(loss, mean logπ) = +0.838` over the long run. Its apparent correlation with the margin is
inherited second-hand from entropy. It also has no forward-looking content: correlation with
the margin change at +5, +10 and +20 rounds is +0.06, −0.07 and −0.12.

This is the standard property of a policy-gradient surrogate, built so its gradient is the
REINFORCE gradient with no claim on its value. It is worth recording because the question is
natural and the answer is not visible in the loss curve.

### The anchor is not a fixed reference either

The margin is defended in the spec as improvement over a fixed point (FR-021). Fixed weights
are not a fixed score. Over `t2all_decay0.3`, from the first full window to the end:

| label | r9 | final | drift |
|---|---|---|---|
| gen4 (learner) | 1.87 | 1.78 | −0.09 |
| gen3a (anchor) | 1.71 | 1.31 | −0.40 |
| gen3c | 1.69 | 0.42 | −1.27 |
| gen1 | 0.79 | 0.53 | −0.26 |
| forge-full | 1.03 | 0.76 | −0.27 |

Every label falls, the learner included. The margin rises because the field falls faster.
Both the anchor margin and any field-relative variant are therefore reporting "declined less
than the field" rather than "improved" — which is why the learner's raw windowed mean belongs
on the `progress` line beside the margin, where it already is.

The natural alternative to a single anchor is the learner against the whole frozen field,
matching the pod-relative reward the policy actually optimises. The two agree closely
(`r = +0.914`) but select different rounds, 312 against 653, and the field version is the
less noisy (standard deviation of round-to-round change 0.056 against 0.059). One caveat if
it is adopted: the logged `R` includes the learner's own other seats in its baseline, so with
`gen4:3` in an eight-seat pod about 2 of 7 baseline seats are the learner and a uniform
improvement δ registers as (5/7)δ. The non-self-referential form, learner against the frozen
labels only, is the one to use.

### The margin decomposition heuristic is refuted

Gen-3 used the split between the learner's rise and the anchor's fall to separate real
learning from field decline, and disqualified its `lr 1e-4` run on that basis. Measured from
the first full window to each run's best round:

| Run | Δ learner | Δ anchor | anchor's share | yardstick vs gen-1 |
|---|---|---|---|---|
| `t2all_nodecay` | −0.12 | −0.71 | 85 % | +1.328 |
| `t2all_decay0.3` | +0.28 | −0.33 | 54 % | +1.380 |
| `t3all_decay0.3` | +0.37 | −0.28 | 43 % | +1.152 |
| `t3learner_t2field` | +0.33 | −0.19 | 37 % | +1.276 |

The heuristic ranks the four in exactly the order the yardstick reverses. `t2all_nodecay` is
its worst run — the learner's own score fell and 85 % of its margin is the anchor collapsing,
the pattern that disqualified gen-3's `lr 1e-4` — and it ties for best on the yardstick.
`t3all_decay0.3` is its second-best and yardsticks last. Do not rank runs this way.

## Why every frozen label declines

Gen-3 attributed the field's decline to denial, and left the size of the effect open. The
yardstick corpora measure it directly, because `--agent-mix` is sampled independently per
seat and pods therefore vary in how many gen-4 seats they contain.

Gen-1 and `forge-full` are treated as one reference label throughout this section. *Which card
axis the agent actually follows*, below, shows they weight every axis of card behaviour
identically, and here they decline in step, gen-1 sitting about 0.1 above `forge-full` at every
composition. Pooling them also makes both corpus families a two-label eight-seat pod at an even
split, so the gen-4 count fixes the whole composition and the two tables are read the same way.
All four checkpoints are pooled; the two carry about 7900 and 8000 gen-4 seats.

| gen-4 seats in the pod | gen-4 | gen-1 + `forge-full` | gap |
|---|---|---|---|
| 1 | +2.77 | +1.44 | +1.33 |
| 2 | +2.49 | +1.16 | +1.32 |
| 3 | +2.33 | +1.04 | +1.30 |
| 4 | +2.17 | +0.80 | +1.37 |
| 5 | +2.05 | +0.63 | +1.43 |
| 6 | +1.89 | +0.38 | +1.52 |
| 7 | +1.62 | +0.19 | +1.43 |

| gen-4 seats in the pod | gen-4 | gen-3 | gap |
|---|---|---|---|
| 1 | +2.40 | +1.63 | +0.76 |
| 2 | +2.06 | +1.51 | +0.55 |
| 3 | +2.00 | +1.39 | +0.60 |
| 4 | +1.85 | +1.29 | +0.56 |
| 5 | +1.81 | +1.24 | +0.57 |
| 6 | +1.74 | +1.12 | +0.62 |
| 7 | +1.76 | +1.07 | +0.68 |

Both fall monotonically, and neither label escapes: crowding a pod with strong drafters takes
cards out of the packs that reach everyone in it, gen-4 seats included.

Crowding costs less against gen-3 than against the Forge-like reference — 0.095 per seat where
it was 0.205, and 0.075 for gen-4's own seats where it was 0.159. Nothing about the mechanism
differs between the two. What differs is the size of the upgrade each swapped seat represents.
Displacing a reference seat at +0.8 with gen-4 at +2.2 adds far more competition than
displacing gen-3 at +1.3 with the same seat.

Dividing the cost by the gap makes that explicit, and the two families agree to within a tenth
of each other on both rows.

| Family | seat displaced | gen-4's lead over it | cost to a rival seat | ratio | cost to a gen-4 seat | ratio |
|---|---|---|---|---|---|---|
| `v-forge` | gen-1 + `forge-full` | +1.377 | −0.205 | 0.149 | −0.159 | 0.116 |
| `v-gen3` | gen-3 | +0.587 | −0.095 | 0.162 | −0.075 | 0.128 |

A seat entering a pod costs each rival roughly a sixth of the amount by which it outclasses
the seat it replaced, and its own kind roughly an eighth. The gap between those two numbers is
the interesting part: a strong drafter is measurably more robust to a crowded pod than the
field it beats, which is why the gap column widens as the pod fills. Two families are two
points, so read the ratio as a regularity worth testing rather than a constant.

Two consequences follow. The first is the training-time decline, seen from a different angle.
During a run the mix is fixed at three learner seats, but those seats get stronger, which is
the same intervention as swapping a reference seat for a gen-4 one. `gen3a` falling 0.40 over
the long run is the expected size of the effect rather than evidence of anything else.

The second is the estimator bias flagged under *The yardstick*. Pooling seats without regard to
composition weights gen-4's mean towards crowded pods, where every seat scores worse, and
weights the reference's mean towards uncrowded ones, because a pod with more gen-4 seats has
fewer reference seats to contribute. Both displacements push the difference down.

| Family | naive per-seat difference | pod-paired | bias | predicted |
|---|---|---|---|---|
| `v-forge` | +1.184 | +1.377 | −0.193 | −0.191 |
| `v-gen3` | +0.498 | +0.587 | −0.089 | −0.084 |

The prediction is the size the two slopes imply. Writing `k` for the number of gen-4 seats in a
pod, `s₄` and `s_ref` for the two slopes, and `S` for the pod size, the size-biased weighting
shifts each mean by its covariance with the seat count, giving

```
bias ≈ s₄ · Var(k)/E[k]  +  s_ref · Var(k)/(S − E[k])
```

which lands within 0.005 of the observed bias on both families. So the bias is set by the
crowding slopes, and the slopes scale with the strength gap, so the bias scales with it too:
it is 14 % of the gap on one family and 15 % on the other. Mix balance is not what drives it.
Both families here are evenly split and their biases differ by a factor of two, and measuring
`v-forge` against gen-1 alone — a quarter of the pod against gen-4's half — gives −0.164,
no worse than the even split does.

The practical form of that: a yardstick reporting levels rather than ordering should use the
pod-paired figure. Balancing the mix does not fix it, and the stronger the candidate the more
the naive number understates it, which is the wrong direction for a measurement whose whole
job is to detect improvement.

## Deck composition

From the four `v-forge` corpora, where gen-4 shares pods with both references.

| | `t2all_nodecay` | `t2all_decay0.3` | `t3all_decay0.3` | `t3learner_t2field` | gen-1 | forge-full |
|---|---|---|---|---|---|---|
| creatures | 18.01 | 18.19 | 17.79 | 17.91 | 15.22–15.28 | 15.09–15.16 |
| avg mana value | 3.17 | 3.21 | 3.19 | 3.19 | 3.02–3.05 | 3.03–3.07 |
| rares | 1.24 | 1.24 | 1.25 | 1.33 | 1.69–1.74 | 1.86–1.95 |
| ≥ 4 basic land types | 4.5 % | 5.6 % | 4.7 % | 6.4 % | 7.0–9.8 % | 9.0–10.9 % |
| score of those wide decks | +1.30 | +1.65 | +1.47 | +1.65 | −0.03…+0.28 | +0.03…+0.14 |

Reference columns give the range across the four corpora, each measured in the same pods as
the candidate beside it.

Both gen-3 trends continue: more creatures and fewer rares, on-colour commons in place of
higher-rarity cards, at an essentially unchanged curve. Every candidate builds narrower mana
bases than the gen-1 seats sitting in the same pods, where gen-3's incumbent led its
references by 3.6 points and these lead by 1.4 to 4.1.

The wide decks are the more informative row. When gen-4 does build four or five basic land
types it scores between +1.30 and +1.65 there, where gen-1's wide decks score around zero.
Going wide is not the failure; going wide when the pool does not support it is, and none of
these four does that. Gen-4's advantage is in fact uniform across every width bucket — on
`t2all_nodecay`, +2.23 against +1.26 at two land types and +1.79 against +0.61 at three — so
the mana base is a symptom of pool quality rather than the thing that differs.

The failure mode that separated gen-3's two families is absent. Splitting each candidate's
median-minus-mean gap into the part four- and five-colour decks account for and the rest, the
drag is +0.011 to +0.022 against gen-3's incumbent at +0.015, and the residual, +0.084 to
+0.121, sits inside the +0.034 to +0.136 that gen-3 measured on its reference seats. That
residual belongs to `deck_score` itself: a sealed deck can be far worse than average more
easily than far better. Nothing about gen-4's training moved it.

## The gen-3 hypotheses under gen-4

Gen-3 asked why its two families drifted in opposite directions on colour, and offered three
explanations. All three are testable on gen-4's corpora. One reverses, one holds and turns out
to rank the candidates, and one is confirmed but common to all of them.

All of it runs on the same card-quality labels gen-3 used. `cards-win-rates.txt` is no longer
under `output/sealed/`; the snapshot lives at
`Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1\cards-win-rates.txt`, and
rerunning gen-3's two corpora against it reproduces its published table cell for cell, so the
levels below are directly comparable across generations. That scale is built from real game
outcomes rather than from anyone's pick behaviour, which is what makes it usable to judge an
agent that has drifted away from the references.

### Hypothesis 1 — lane starvation. Confirmed as a mechanism, refuted as a quality signal.

The hypothesis: a field that plays its best takes the good cards in the learner's colours, so
the learner keeps facing packs with nothing playable on-colour, takes the off-colour card
because it has to, and acquires a general taste for off-colour cards from positions where it
never had a choice.

Gen-4 varies field strength on a new axis. A field at `T = 3` misplays more than a field at
`T = 2`, so it starves the learner less, and the hypothesis predicts the `T = 3` field should
produce the least off-lane taste. It does, and by a wide margin. Off-lane rate at picks 6–10
of pack 1, with the share of those picks made while an on-colour card was still in the pack:

| Corpus | gen-4 | gen-1 | forge-full |
|---|---|---|---|
| `t2all_nodecay` | 11.9 % (49.7) | 9.5 % (46.0) | 11.7 % (57.7) |
| `t2all_decay0.3` | 12.6 % (54.2) | 8.3 % (46.3) | 11.1 % (57.3) |
| `t3all_decay0.3` | 8.6 % (34.0) | 9.9 % (49.0) | 12.9 % (58.4) |
| `t3learner_t2field` | 12.0 % (48.9) | 9.8 % (49.6) | 12.2 % (51.7) |
| *gen-3 incumbent* | *9.6 % (34.2)* | *10.8 % (45.4)* | *11.5 % (50.4)* |

The `T = 3` field candidate declines an available on-colour card in a third of its off-lane
picks, where the others decline in about half, and it goes off-lane less often than the gen-1
seats beside it. It is the most lane-disciplined agent in the table by both readings. It is
also the worst on the yardstick.

The gen-4 candidates trained against a `T = 2` field went the other way. All three go off-lane
more often than gen-1 and at least as often by choice, and all three left the incumbent's
position: gen-4 inherited a policy that declined on-colour cards a third of the time and moved
it to about half, which is where the gen-1 and `forge-full` references have always sat.

None of that is an artefact of gen-4's colour preferences. Off-lane is defined against a
seat's own eventual top-2, so an agent with an unusual lane could face packs holding more
off-lane cards and go off-lane more often without choosing anything. Measuring the supply
directly at each decision, the share of available coloured cards outside the seat's top-2 is
55–58 % for every agent in every corpus, and the ratio of off-lane picks to off-lane supply
separates the agents exactly as the rate does. The differences are behaviour.

So the mechanism is real and the reading gen-3 built on it is not. Off-lane rate rises and
falls with field strength as predicted, and it does not track deck quality: gen-3 saw its bad
candidate go off-lane most and inferred that going off-lane is the fault, while gen-4's best
candidates go off-lane most and its most disciplined one places last. The discriminator has
to come from the next two hypotheses.

### Hypothesis 2 — card power over lane fit. Holds, and it ranks the candidates.

The hypothesis: breaking colour is correct when the card is enough better than the on-colour
alternative, so a healthy policy should show a positive quality premium on its voluntary
off-lane picks and a failing one should not. Cards are scored by `shrunk_score_play`, net
winning influence on the play, which covers 98 % of drafted card slots.

| Corpus | agent | best-card rate | mean premium | share above zero |
|---|---|---|---|---|
| `t2all_decay0.3` | gen-4 | 25.1 % | +0.0318 | 68.2 % |
| `t3learner_t2field` | gen-4 | 24.4 % | +0.0287 | 66.2 % |
| `t2all_nodecay` | gen-4 | 25.3 % | +0.0260 | 65.7 % |
| `t3all_decay0.3` | gen-4 | 24.0 % | +0.0225 | 62.1 % |
| gen-3 incumbent | gen-3 | 22.9 % | +0.0240 | 63.0 % |
| gen-3, field at argmax, `T = 3` | gen-3 | 28.5 % | +0.0049 | 51.9 % |
| *references, all corpora* | *gen-1 / forge-full* | *18.8–21.7 %* | *+0.007…+0.015* | *53.4–58.7 %* |

Every gen-4 candidate breaks colour more selectively than either reference and more
selectively than gen-3's incumbent, at a larger premium. Three of the four also beat the
incumbent on the share above zero. Whatever gen-4 gained from training, it did not come at
the cost of the judgement that separated gen-3's healthy candidates from its failed one.

The last row is that failed candidate, whose wide decks averaged −1.88. It breaks colour at a
coin flip and gains nothing, and it does so while taking the pack's highest-win-rate card more
often than any other agent here. Gen-3 read that pair of facts as card evaluation intact and
pool fit lost, and gen-4 gives the reading a control: gen-4's best-card rate sits below the
failure's and its premium sits far above it, so the two columns are measuring different things
and only the premium tracks quality.

The premium ranks the four candidates, which nothing else at the pick level does. Its ordering
is `t2all_decay0.3`, `t3learner_t2field`, `t2all_nodecay`, `t3all_decay0.3`, and the share
above zero gives the same order. `t3all_decay0.3` is last on both, matching the yardstick, and
its 6.1-point deficit in share above zero against `t2all_decay0.3` carries a standard error of
1.1 points. The middle two swap places against the yardstick, so read the ordering as
identifying the loser rather than resolving the top.

This is the first pick-level measurement that agrees with the yardstick. The mechanism it
suggests is that a field sampling at `T = 3` gives worse evidence about when breaking colour
pays, and the resulting policy breaks colour slightly worse.

### Hypothesis 3 — a colour prior learned from Forge. Confirmed and stronger.

The hypothesis: Forge pilots green, black and white better than blue and red, because blue
and red lean on instants and sorceries and Forge plays those worst. `deck_score` is fitted to
Forge-piloted outcomes, so a policy trained on it should acquire a taste for those three
colours, and the taste should show when it breaks lane.

The test compares, at each off-lane pick, the colour of the card taken against the colour mix
of the off-lane cards available in that pack at that moment. Each pick contributes weight 1 to
both sides, and gold cards split their weight across their colours. Mean per off-lane pick of
(green-black-white taken − green-black-white available):

| Corpus | gen-4 | gen-1 | forge-full | learner picks, this generation |
|---|---|---|---|---|
| `t3all_decay0.3` | +2.91 pp | −0.22 | −0.10 | ~75k |
| `t3learner_t2field` | +3.61 | +0.23 | −0.78 | ~90k |
| `t2all_nodecay` | +4.09 | −0.18 | +0.51 | ~95k |
| `t2all_decay0.3` | +3.99 | −0.21 | −0.58 | ~405k |

Standard errors are 0.23–0.33 pp, so every gen-4 figure is more than twelve standard errors
from zero and no reference figure is more than two and a half. Gen-3's four candidates ran
+0.70 to +2.90 on the same measurement; all four gen-4 candidates exceed gen-3's largest.

The lean tracks cumulative training and then saturates. Ordered by learner picks the four run
+2.91, +3.61, +4.09, +3.99, and the fourth trained four times as long as the third for
nothing further. Gen-3 found the lean's size tracked training length rather than which field
a candidate trained against; gen-4 reproduces that on all three temperature configurations and
adds the ceiling, somewhere near +4 pp.

It shows in the decks, not just at the pick. Across the four corpora gen-4 plays white in
59.8–63.4 % of its decks and red in 30.8–33.4 %, where the gen-1 and `forge-full` seats in the
same pods sit at 45.9–49.9 % white and 52.6–56.0 % red. Blue follows red down, green and black
follow white up. Two generations of self-play have turned a mild preference into a
near-inversion of the reference colour distribution.

The lean is correct play against this opponent, and the same preference would be
miscalibrated against a human. Forge wins more with green, black and white; `deck_score`
measures Forge-piloted outcomes; a policy that learns which colours win those games is doing
what it was asked. The finding to carry forward is the size and the ceiling, not a problem to
fix.

### Which card axis the agent actually follows

Hypothesis 2 scores cards on `score_play` alone, and the encoder is trained on five axes
([`../specs/2026-05-03-card-winnability-pretraining.md`](../specs/2026-05-03-card-winnability-pretraining.md)),
so there is no reason the drafter's preferences should line up with that one. The test looks
at every pick rather than only the off-lane ones, and asks where the taken card sat among the
cards still in the pack under each axis, as a percentile. An agent indifferent to an axis
scores 0.5; one that always takes the pack's maximum scores 1.0.

`color_lift` is the mean of `color_lift_X` over the seat's eventual top-2 colours: how well
the taken card pairs with the colours the seat committed to. The reference columns give the
range each reference took across the five corpora, every one of them measured in the same pods
as the candidates beside it.

| Axis | `t2all_decay0.3` | `t2all_nodecay` | `t3learner_t2field` | `t3all_decay0.3` | gen-3 incumbent | gen-1 | forge-full |
|---|---|---|---|---|---|---|---|
| `score_play` | 0.703 | 0.688 | 0.684 | 0.679 | 0.662 | 0.603–0.610 | 0.593–0.610 |
| `score_draw` | 0.698 | 0.685 | 0.683 | 0.675 | 0.659 | 0.601–0.606 | 0.592–0.607 |
| `played_rate` | 0.643 | 0.653 | 0.644 | 0.636 | 0.632 | 0.587–0.589 | 0.573–0.583 |
| `cast_lift` | 0.604 | 0.588 | 0.588 | 0.581 | 0.561 | 0.523–0.529 | 0.518–0.530 |
| `color_lift` | 0.508 | 0.512 | 0.517 | 0.530 | 0.529 | 0.557–0.563 | 0.559–0.568 |

Read down a column, not across a row. A noisier or more tied label regresses towards 0.5 on
its own, so the axes are not on a common scale and `score_play` sitting above `cast_lift`
says as much about the two labels as about the agent. Comparing agents within an axis is
safe, and that is what everything below does.

The two references are the same drafter on every axis. Gen-1 and `forge-full` agree to within
0.01 on four of the five, and the one gap, `played_rate`, is 0.01 wide. Gen-1 was trained to
imitate Forge, and the imitation reproduced not just Forge's picks but the weighting Forge
puts on each axis of card behaviour. That makes the reference columns a single baseline rather
than two, and it is why the gains quoted below are stated against gen-1 alone.

Every trained agent is above both references on the four quality axes and below both on the
colour axis, so the shape of the departure is shared across two generations and four
candidates. Raw winning influence is the axis the agents follow most closely in absolute
terms, and `score_play` and `score_draw` are not independent evidence for that: they correlate
at Spearman 0.72 over the card population, and every agent tracks them equally, so they are
one finding.

The axis that separates gen-4 from gen-3 is `cast_lift`. Gen-3's incumbent led its references
by +0.032 there and gen-4 leads by +0.057 to +0.079, roughly a doubling, where the gain on
`played_rate` barely moved between the generations. `cast_lift` is the causal effect of
actually casting a card, net of the quality of the deck it sat in — it is the axis that
separates a card that swings games from one that rides along in decks that were winning
anyway. It correlates with `score_play` at 0.64, so most but not all of that movement is
shared with raw power. Gen-4 got better at taking cards that do something.

`cast_lift` is also the one axis whose ordering matches the yardstick across all four
candidates. Differencing each candidate against the gen-1 seats in its own corpus gives
+0.079, +0.065, +0.062 and +0.057, against yardstick margins of +1.380, +1.328, +1.276 and
+1.152. The pairing is what resolves it: `t2all_nodecay` and `t3learner_t2field` both sit at
0.588 in the table and separate only once each is read against the gen-1 seats it actually
drafted alongside. Four candidates ordering correctly by chance is a one-in-twenty-four event,
so this is worth another generation's data rather than a conclusion.

On the colour axis three of the four gen-4 candidates sit further below their references than
gen-3 did, and `t3all_decay0.3` is the exception in the direction its ranking would predict:
it is the least improved on every quality axis and the least degraded on this one. Do not read
any of it as gen-4 ignoring colour. The
colour-lift labels correlate negatively with `score_play` (−0.13 to −0.28) and more strongly
negatively with `played_rate` (−0.27 to −0.41), so an agent climbing the power axis is pushed
down the colour axis mechanically. Whether anything is left after that confound is not
answerable from this measurement.

That axis is also not the colour prior of Hypothesis 3, which is worth keeping separate.
Hypothesis 3 is about which of WUBRG the agent prefers. `color_lift` is about whether a given
card pairs well with the colours already committed to, and its label is constructed to cancel
the card's own quality baseline. Gen-4 has a strong preference over colours and no measurable
preference over colour synergy.

### What the hypotheses leave

Hypothesis 1's mechanism predicts each candidate's lane discipline from its field temperature
and gets the direction right, but discipline does not track deck quality, so the reading gen-3
built on it does not survive. Hypothesis 2 holds, separates a healthy policy from gen-3's
failed one, and orders the four candidates with the loser in the right place. Hypothesis 3 is
confirmed, stronger than in gen-3, and common to all four candidates, so it explains none of
the spread between them.

Two pick-level statistics therefore point the same way as the yardstick, both of them about
the quality of a card choice rather than its colour: the off-lane premium and `cast_lift`
alignment. Both say the `T = 3` field produced a policy that judges cards slightly worse. That
is consistent with the account the temperature decomposition gives — a field sampling further
from its best play gives worse evidence about which card was the right one — and it is the
first evidence for that account measured on the finished policy rather than inferred from the
training setup.

## Where this leaves gen-4

Promote `t2all_decay0.3` or `t2all_nodecay`. They are tied on both yardsticks, they lead the
generation, and each beats the gen-3 incumbent by about +0.6 head-to-head. `t2all_nodecay` is
the cheaper of the two by a factor of thirteen in wall-clock and reaches the same place, so
prefer it unless a reason to prefer the longer run appears.

Carry `T = 2` on the field into gen-5. The learner's own temperature is free to raise and
`T = 3` on the learner alone costs nothing measurable, so the exploration band can be chased
on the learner without paying for it on the field. That is the one setting gen-3 left open and
it is now settled.

Fix the three in-run defects. They did not change this generation's outcome, but they made
every run harder to read than it needed to be.

- Clamp `can_decay` to `min_lr` instead of truncating, and log when the floor is reached.
- Delay best-tracking to `2 × anchor_window` drafts, and require a new best to clear a
  `min_delta` on the order of the window's standard error. A run whose learner and anchor
  share a warm start supplies that estimate for free at round 0.
- Select on the learner against the frozen field rather than a single anchor, and keep the
  learner's raw windowed mean beside it.

Arm `--patience`. No run in this generation had a working stopping rule, and three of the four
spent most of their wall-clock past their best round.

The measurement gap gen-3 left open is closed, and the answer was the favourable one. Every
metric available to the loop is a derivative of the same frozen `deck_score`, and 1001 played
matches now say that number buys 7.4 points of game win rate per unit, agreeing with the
sealed pipeline's 7.8 on a separately built corpus. Nothing above needs re-reading as a
proxy result. What remains is that the exchange rate is shallow enough to make games an
expensive way to compare candidates: the 0.23 separating the best from the worst is worth
about 1.7 points of win rate, which is thousands of matches to resolve.

That makes cheap non-`deck_score` signals worth having anyway, and one appeared while checking
Hypothesis 2. Pick alignment against the `cards-win-rates.txt` labels is not a derivative of
`deck_score`. Both trace back to Forge-piloted games, but the scorer predicts a match outcome
from a whole deck while these labels count per-card play and win events, so an agent cannot
improve on one by construction of the other. Alignment is computable from any existing corpus
with no Forge time at all, it separated the four candidates in the same order as the
yardstick, and the off-lane premium on the same labels put the loser in the same place. Add
both to the checkpoint report, and if they keep agreeing with the yardstick over another
generation, they are a candidate run-control metric that the anchor margin has already failed
to be.

Two of the three questions the games were meant to answer are settled, and the remaining one
is the expensive one.

1. Does `deck_score` predict winning? Yes, at 7.4 points per unit. Settled.
2. Does the colour prior survive contact with games? Gen-4 wins within a point of 67 % in every
   colour, so nothing in its lean costs it games. Settled as far as an observational read can
   settle it.
3. Do the four candidates rank the same way on games as on score? Open. One checkpoint was
   played, and matching the score ordering requires separating candidates 1.7 points apart.

Sizing, so the third is armed deliberately: a head-to-head win rate needs roughly `1.96/δ²`
games for 80 % power at α = 0.05 — about 200 games to resolve 60/40, 800 for 55/45 and 2200
for 53/47, before any inflation for the clustering that comes from reusing a deck across
pairings. Separating adjacent candidates sits past the right-hand end of that scale, so the
`v-gen3` corpora — two labels in every pod, and a 0.6 gap rather than a 0.23 one — are the
place to spend the time.

## Open questions

- **Is `t3all_decay0.3`'s deficit temperature or undertraining?** Its yardsticked weights are
  round 58 and its run's best is round 208. Yardsticking round 208 would settle it, if that
  checkpoint still exists.
- **Why does the learner's own absolute score fall over a long run?** Pod crowding explains
  the frozen labels and part of the learner, but `t2all_decay0.3` ran 956 rounds past its best
  with a cumulative KL of 2.37, by far the largest displacement from a warm start in either
  generation, and ended 0.36 below its own best margin. Whether those rounds damaged the
  argmax policy is answerable by yardsticking its final snapshot against its round-312 best.
- **Where is the field-temperature optimum?** Gen-4 brackets it between 2 and 3 from above.
  Nothing has tested below 2 under field at T, and gen-3's `lr 1e-6` run is not evidence
  because it did not train.
- **Does the colour prior have a ceiling or a cost?** It saturated near +4 pp within this
  generation, and the games show no colour where gen-4 wins less. That rules out a cost it is
  already paying, not one it would pay by leaning further, and it cannot say whether a policy
  without the lean would do better. Forcing a candidate's colours at draft time and replaying
  is the test.
- **Does the builder gap hold up?** Forge's own sealed builder lost 68.6 % of matches to this
  project's on identical drafted pools, on 51 matches. That is a large effect on a small
  sample, and `--forge-native-fraction` makes replicating it nearly free on any corpus.
- **Does `cast_lift` alignment keep tracking the yardstick?** It ordered four candidates
  correctly, which four candidates do one time in twenty-four by chance. Gen-5 supplies the
  replication, and it is free to compute on corpora that already exist.
- **Does the agent read colour synergy at all?** Every trained agent scores below its
  references on `color_lift` alignment, but the colour-lift labels are anticorrelated with
  both `score_play` and `played_rate`, so climbing the power axis produces that reading on its
  own. Residualising colour lift on power before measuring alignment would separate the two,
  and would say whether there is an unused axis in the encoder's supervision.
- **Does the crowding ratio hold as a design rule?** Two corpus families put the cost to a
  rival seat at about a sixth of the strength gap the entering seat introduces. If that holds,
  the depression a training mix induces in its own frozen field is predictable from the
  learner's lead before the run starts, and `--mix` becomes a knob with a known cost. Two
  points are a coincidence away from nothing; a third family would settle it, and a gen-5
  yardstick supplies one for free.
