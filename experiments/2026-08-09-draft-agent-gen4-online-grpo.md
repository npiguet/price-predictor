# Draft agent (gen-4) — online GRPO from the promoted gen-3 candidate

## Context — what gen-3 left us with

Gen-3 closed with two settings to carry forward and one prescribed run
([`2026-06-15-draft-agent-gen3-online-grpo-design.md`](2026-06-15-draft-agent-gen3-online-grpo-design.md),
*Where this leaves gen-3*):

- **Field at T** as the training field — every ML agent samples, not just the
  learner. It produced the best candidate on the yardstick and the tightest score
  distribution, inverting the spec's expectation that a weakened field would
  transfer worse.
- **`T = 3`** as the temperature that holds the exploration band, but tested for
  only 28 rounds under field at T, so the band and the outcome disagreed and the
  question was left open.

The prescribed run was `lr 1e-5`, `T = 3`, field at T, at least 150 rounds, with
`--patience` and `--lr-decay-patience` armed.

Gen-4 warm-starts every run from the promoted gen-3 candidate
(`gen3/temperature-on-all-agents/lr1e-5_t2_20260805_221050.pt`) and uses it as the
anchor as well. That identity — learner and anchor being the same weights at the
same temperature — turns out to be the most useful measuring instrument in the whole
run, for reasons that had nothing to do with why it was chosen (*The round-9 best is
noise*, below).

## The runs

Four runs, all `lr 1e-5`, all from the same base, all on the mix
`gen4:3,gen3a:2,gen3c:1,gen1:1,forge-full:1` with `--anchor gen3a`. The per-label
`--agent-temp` from spec 021 phase 9 makes the learner/field temperature split
expressible for the first time, and the fourth run is the first to use it.

| Run | learner T | field T | Rounds | Duration | Best margin (round) | Final margin | LR decays |
|---|---|---|---|---|---|---|---|
| `t2all_nodecay` | 2.0 | 2.0 | 84 | 1h42m | +0.546 (r72) | +0.379 | — (unarmed) |
| `t2all_decay0.3` | 2.0 | 2.0 | 1269 | 22h39m | +0.831 (r312) | +0.468 | 1 |
| `t3all_decay0.3` | 3.0 | 3.0 | 569 | 9h54m | +0.676 (r208) | +0.206 | 1 |
| `t3learner_t2field_decay0.3` | 3.0 | 2.0 | 178 | 4h07m | +0.391 (r68) | +0.002 | 1 |

`t2all_nodecay` ran against a field that also contained a `gen3b` seat, so its
margins do not compare with the other three — the same discipline gen-3 applied
across its two families. The other three share a field and do compare.

All four peak and decline, as every gen-3 run did. Three of the four end at less
than half their best; the fourth ends at zero.

Every run's rollouts were retained this time, one `-drafts.jsonl` per run — the
instrumentation gen-3 asked for and lost. They are the input the unresolved gen-3
question needs, and they are also what the evaluation feature below consumes.

## The yardstick

One of the four has been taken to a yardstick: `t2all_nodecay`, via a 500-draft
argmax run on the fixed mix `gen4:2,gen1:1,forge-full:1`.

| Agent | mean | median | n |
|---|---|---|---|
| gen-4 | 2.07 | 2.21 | 2049 |
| gen-1 | 0.98 | 1.17 | 979 |
| forge-full | 0.80 | 0.99 | 972 |

That is **+1.09 mean / +1.04 median over gen-1**, against gen-3's best of +0.73 on
its own corpus. The cross-corpus comparison carries gen-3's caveat — each yardstick
draws its own random sets and the level moves with the draw, visible here in gen-1
landing at 0.98 where it sat at 1.25 in gen-3's `T = 2` corpus. Within this corpus
the three agents share pods, so the ordering is paired and safe; ranking on
`forge-full` instead of gen-1 gives the same order.

### Deck composition

From the same corpus, where all three agents share pods:

| | gen-4 | gen-1 | forge-full |
|---|---|---|---|
| creatures | 18.01 | 15.22 | 15.09 |
| avg mana value | 3.17 | 3.05 | 3.07 |
| rares | 1.24 | 1.69 | 1.95 |
| decks with ≥ 4 colours | 2.8 % | 6.8 % | 7.9 % |

Both gen-3 trends continue and strengthen: more creatures, fewer rares — on-colour
commons in place of higher-rarity cards — at an essentially unchanged curve.

Colour discipline is the headline. Gen-3's field-at-T policies built wide mana bases
5.5–6.4 % of the time against gen-1 references at 7.3–9.1 %; this gen-4 is at **2.8 %
against 6.8 %**, less than half the reference and about a third of Forge. The failure mode that
separated gen-3's two families — the field-at-argmax `T = 3` policy going wide 13.3 %
of the time with a −1.88 five-colour bucket — is absent. Continuing to train field-at-T
on a policy that was itself trained field-at-T compounds the effect rather than
saturating it.

## What the in-run metrics did, and did not, tell us

The yardstick above is the only measurement in this generation that is not a
derivative of the same frozen `deck_score`. Everything below is about how badly the
in-run machinery served the selection it exists to perform.

### The LR annealing decayed once and then silently stopped

All three armed runs took exactly one decay, to 3.0e-6, and never moved again —
`t2all_decay0.3` sat at 3.0e-6 for 1239 of its 1269 rounds.

The cause is arithmetic in `_PlateauLR.can_decay()`, which refuses a decay that would
land *below* `min_lr` rather than clamping to it:

```python
return self.base_lr * self.factor ** (self.decay_count + 1) >= self.min_lr
```

With `--lr 1e-5 --lr-decay-factor 0.3 --min-lr 1e-6` the ladder is
`1e-5 → 3e-6 → 9e-7`, and `9e-7 < 1e-6`, so the second decay is refused forever. The
effective floor is 3e-6; the startup echo advertises 1.0e-06, which is unreachable.
Exactly one decay was ever possible. Every existing `_PlateauLR` test uses
`factor = 0.1` with a floor an exact power of ten below the base, so the ladder always
lands *on* the floor and the truncation never shows — `0.3` is the first factor used
that does not divide the base-to-floor ratio evenly.

Two things made it silent. `_maybe_decay` returning `None` is indistinguishable from
"not stalled yet", so nothing logs the refusal. And `--patience` was not armed in any
of the four runs, so a run that can no longer anneal also cannot stop: `t2all_decay0.3`
spent its last 956 rounds with no new best, no further decay, and no exit.

The decay is applied to the live weights, not to `best_*.pt` — no rollback, no moment
reset, no re-warmup. That is the standard convention and the right one here, since the
best it would roll back to is the noise described next.

### The round-9 best is noise, and it is measurable

In every gen-4 run the learner (`gen4`) and the anchor (`gen3a`) are **the same
checkpoint at the same temperature**. Generation precedes the update, so round 0's
drafts are produced by two bit-identical policies and the true margin at round 0 is
exactly **0**.

`t2all_decay0.3` reports `margin +0.781` at round 0. That is a free calibration of the
metric's noise floor: ±0.78 at 10 drafts, so roughly **±0.25 at the 100-draft window**.

The margin then falls monotonically — 0.781, 0.585, 0.558, 0.412, 0.373, …, 0.218 —
which is not the learner regressing but a *cumulative* mean regressing away from a
lucky first ten drafts. The window fills at round 9 (10 rounds × 10 drafts), and the
window-full guard admits it:

```python
return len(self._window) >= self._maxlen
```

The guard's docstring says it prevents "an early lucky round" from pinning the run's
best. It does not: excluding rounds 0–8 does not exclude rounds 0–8's *drafts*, which
are 100 % of round 9's window. It filters the reporting round, not the contaminated
data. The window only holds no fill-period drafts at all from round 19.

The consequence in `t2all_decay0.3` is concrete. The best was pinned at round 9 at
+0.218 — inside one sigma of zero — the stall counter started there, the settled
process over rounds 10–29 ran at 0.00–0.18 and never cleared it, and the single
available decay fired at round 29 while the policy was learning perfectly well. The
first genuine best came at round 70, 61 rounds later, more than three times the decay
patience. The run went on to +0.831.

The two defects partially cancelled: had the floor clamped correctly, decay #2 would
have fired at round 49 and parked the LR at 1e-6 for the whole productive phase through
round 312.

### Policy loss cannot select checkpoints

The obvious alternative to the margin is the quantity being optimised. It is worse —
and the sign is the interesting part.

| Run | corr(policy_loss, margin) |
|---|---|
| `t2all_nodecay` | +0.229 |
| `t2all_decay0.3` | +0.344 |
| `t3all_decay0.3` | +0.215 |
| `t3learner_t2field_decay0.3` | +0.464 |

Positive in all four. Loss is a minimisation target, so a useful selector would
correlate *negatively* with the margin. Over `t2all_decay0.3`, `argmin(loss)` is round
1193 (margin +0.152) where `argmax(margin)` is round 312 (+0.831) — selecting on loss
picks a checkpoint 0.68 margin worse, from the bottom half of the run's distribution.

The mechanism is structural. `assign_advantages` standardises each round's rewards to
mean 0 and std 1, and the loss is `−mean(A·logπ)`, so with `mean(A) = 0` it is
`−Cov(A, logπ)`, bounded by `σ_logπ`. Two consequences: the per-round standardisation
destroys all absolute performance information, and what remains is an entropy
thermometer — `r(loss, H) = −0.852`, `r(loss, mean logπ) = +0.843` over the long run.
Its entire apparent correlation with the margin is inherited second-hand from entropy.
It also has no forward-looking content: correlation with the margin change at +5, +10
and +20 rounds is +0.06, −0.07, −0.12.

This is the standard property of a policy-gradient surrogate — built so its *gradient*
is the REINFORCE gradient, with no claim on its *value*. Worth recording because the
question is natural and the answer is not obvious from the loss curve.

### The anchor is not a fixed reference either

The margin is defended in the spec as improvement over a fixed point (FR-021). Fixed
weights are not a fixed score. Over `t2all_decay0.3`, from the first full window to the
end:

| label | r9 | final | drift |
|---|---|---|---|
| gen4 (learner) | 1.88 | 1.78 | −0.11 |
| gen3a (anchor) | 1.67 | 1.31 | −0.36 |
| gen3c | 1.62 | 0.42 | −1.21 |
| gen1 | 0.72 | 0.53 | −0.19 |
| forge-full | 1.11 | 0.76 | −0.35 |

Every label falls, the learner included, and the learner's fitted slope (−0.41 per 1000
rounds) is *steeper* than the anchor's (−0.32). The margin rises because the field falls
faster. Both the anchor margin and any field-relative variant are therefore reporting
"declined less than the field", not "improved" — a distinction no relative metric can
surface on its own, and the reason the learner's raw windowed mean belongs on the
`progress` line beside the margin, where it already is.

Since the reward the policy optimises is the pod-relative leave-one-out score against
*all* other seats, the natural alternative to a single anchor is the learner against the
whole frozen field. The two agree closely (`r = +0.914` against a mix-weighted field
margin) but select different rounds — 312 against 653 — and the field version is the
less noisy of the two (std of round-to-round change 0.056 against 0.059, and 0.037 for a
windowed leave-one-out reward). One caveat if it is adopted: the logged `R` includes the
learner's *own other seats* in its baseline, so with `gen4:3` in an eight-seat pod about
2 of 7 baseline seats are the learner and a uniform improvement δ registers as
(5/7)δ. The non-self-referential form — learner against the frozen labels only — is the
one to use.

### Composition of the margin did not predict the yardstick

Gen-3 used the split between the learner's rise and the anchor's fall to separate real
learning from field decline, and disqualified its `lr 1e-4` run on that basis. Measured
from the first full window to each run's best round:

| Run | learner | anchor | Δ learner | Δ anchor | anchor's share |
|---|---|---|---|---|---|
| `t2all_nodecay` | 1.90 → 1.77 | 1.94 → 1.23 | −0.12 | −0.71 | 85 % |
| `t2all_decay0.3` | 1.88 → 2.17 | 1.67 → 1.33 | +0.28 | −0.33 | 54 % |
| `t3all_decay0.3` | 1.71 → 2.07 | 1.68 → 1.40 | +0.37 | −0.28 | 43 % |
| `t3learner_t2field_decay0.3` | 1.67 → 2.00 | 1.80 → 1.61 | +0.33 | −0.19 | 37 % |

By that reading `t2all_nodecay` is the worst run in the generation: its learner score
*fell* and 85 % of its margin is the anchor collapsing — precisely the pattern that
disqualified gen-3's `lr 1e-4`. It is also the only run yardsticked, and it yardsticks
at +1.09 over gen-1 with the tightest colour discipline in the lineage.

That is not proof the heuristic is wrong — the other three are unyardsticked, and the
training-time learner mean is measured at `T` while the yardstick is argmax, so the two
are not the same quantity. But it is a direct counter-example to reading run health off
the margin decomposition, and it should not be used to rank the remaining three without
a yardstick behind it.

## Exploration and movement

| Run | T | ppl r0 | ppl mean | ppl last 10 | off-arg r0 | mean | last 10 | in band? |
|---|---|---|---|---|---|---|---|---|
| `t2all_nodecay` | 2.0 | 1.42 | 1.46 | 1.35 | 15.6 % | 14.8 % | 12.6 % | no — below, declines |
| `t2all_decay0.3` | 2.0 | 1.33 | 1.61 | 1.53 | 8.7 % | 18.4 % | 17.5 % | no — below throughout |
| `t3all_decay0.3` | 3.0 | 1.83 | 1.86 | 2.06 | 18.4 % | 23.3 % | 28.8 % | closest — enters late |
| `t3learner_t2field_decay0.3` | 3.0 | 1.80 | 1.84 | 1.58 | 18.2 % | 22.1 % | 17.6 % | no — sags out |

The band is perplexity 2–3 and off-argmax 25–40 %. `T = 3` on all agents is the only
configuration that reaches it, and only in the second half of the run; `T = 2` never
comes close. This reproduces gen-3's finding that the proposed `T ∈ {1.0 … 2.5}` sweep
was aimed too low.

| Run | KL(prev) median | p90 | max | KL(π₀‖πₖ) final | grad-norm mean / max |
|---|---|---|---|---|---|
| `t2all_nodecay` | 0.0125 | 0.1060 | 1.4514 | 0.516 | 8.4 / 18.5 |
| `t2all_decay0.3` | 0.0062 | 0.0439 | 0.2084 | 2.368 | 8.4 / 20.2 |
| `t3all_decay0.3` | 0.0101 | 0.0556 | 0.1980 | 1.218 | 8.2 / 14.3 |
| `t3learner_t2field_decay0.3` | 0.0062 | 0.0400 | 0.3972 | 0.783 | 7.5 / 14.0 |

Cumulative KL again runs two to three orders of magnitude above the per-round median,
and `t2all_decay0.3` reaches 2.37 — by far the largest displacement from a warm start in
either generation, over a run that ends 0.36 below its own best margin.

## Corrections to the gen-3 design doc

- **Entropy does not simply decay at fixed `T`.** Gen-3 predicted, and observed,
  entropy sagging over a run. Over gen-4's two long runs it *rose* — `t2all_decay0.3`
  from H 0.282 to 0.423, `t3all_decay0.3` from 0.602 to 0.695 — while the two short runs
  fell. The direction appears to depend on run length rather than on `T`, and a rising
  entropy alongside a declining absolute score is worth watching as a degradation signal
  rather than a health one.
- **The prescribed settling run was run, and did not settle it.** `t3all_decay0.3` is
  `lr 1e-5`, `T = 3`, field at T, 569 rounds — well past the prescribed 150. It peaks at
  +0.676 (r208) and ends at +0.206. Without a yardstick it cannot be compared with the
  `T = 2` runs, so the `T = 2` versus `T = 3` question is still open.
- **`--patience` was armed in none of the four runs**, despite the prescription. Combined
  with the annealing defect, no run had a working stopping rule.

## Where this leaves gen-4

The promotable candidate is `t2all_nodecay`, on the only evidence that exists: +1.09
mean over gen-1, +1.27 over `forge-full`, and the tightest colour discipline of any
agent measured in either generation. Three runs — including the 22-hour one and both
`T = 3` variants — are unyardsticked, and nothing in the training logs ranks them
reliably, as the section above shows.

The blocking problem is not any single defect. It is that **every metric available to
the loop is a derivative of the same frozen `deck_score`**, which is a scorer fitted to
Forge-piloted outcomes. The anchor margin, the field margin, the reward, and the
yardstick are four views of one number, and the loop has no way to ask whether that
number tracks winning games. The whole generation optimised it for 38 hours without
playing a single game.

The three cheap fixes to the in-run machinery are worth doing and are not the priority:

- clamp `can_decay` to `min_lr` instead of truncating, and log when the floor is reached;
- delay best-tracking to `2 × anchor_window` drafts and require a new best to clear a
  `min_delta` on the order of the window's standard error — round 0 supplies that estimate
  for free whenever learner and anchor share a warm start;
- select on the learner against the frozen field rather than a single anchor, and keep
  the learner's raw windowed mean beside it.

**The gen-4 action is to close the measurement gap: play the games.** The next feature
samples deck pairs out of a drafts corpus, plays them in Forge, and tallies per-agent win
rates — turning the four unranked runs into a ranking, and, for the first time, measuring
whether `deck_score` predicts winning at all. That last statistic is the one no amount of
margin-metric tuning can produce, and the sealed pipeline's analogue (score delta
predicting win-rate delta at r ≈ 0.52, ~7.8 pp per unit score) says it is worth having.

The design settled on:

- **Reuse an existing argmax yardstick corpus** rather than re-drafting. `--agent-mix` is
  sampled independently per seat, so seat position is already randomised and no agent
  systematically sits downstream of another.
- **Pair within a pod.** Both decks come from one 360-card pool, so set and pool quality
  are controlled by construction — the biggest nuisance factor removed for free.
- **BO1, play-first alternating, report game win rate.** Balanced play/draw removes the
  largest systematic bias exactly rather than in expectation, and game win rate uses every
  game where a match win rate discards information.
- **Stratify by label pair and reject mirrors**, so every matchup gets equal precision
  instead of whichever happens to be sampled most.
- **Record `draft_id` and seat indices**, so games sharing a deck can be clustered at
  analysis time. Without it a sample of games reused across pods reads as more independent
  than it is and the intervals come out too narrow.

`ValidationWorkerMain` already plays explicit `deckA;deckB` pairings from a file and
`EvaluationConnector.launch_workers` already runs a crash-restarting pool of them, so this
needs no new Java. The normative spec is
[`../specs/2026-08-09-draft-game-evaluation.md`](../specs/2026-08-09-draft-game-evaluation.md).

Sizing, so it is armed deliberately: a head-to-head win rate needs roughly `1.96/δ²` games
for 80 % power at α = 0.05 — about 200 games to resolve 60/40, 800 for 55/45, 2200 for
53/47, before any inflation for clustering. Separating adjacent generations is the
expensive case and should be budgeted as such.

## Open questions

- **`T = 2` against `T = 3` under field at T**, carried over from gen-3 and still open —
  `t3all_decay0.3` is the run that answers it and needs a yardstick.
- **Does the learner's absolute decline matter?** Every label falls over a long run, the
  learner fastest of all in `t2all_decay0.3`, yet the argmax yardstick on a *short* run is
  the best result in the lineage. Whether long runs degrade the argmax policy while the
  margin climbs is answerable by yardsticking `t2all_decay0.3`'s round-312 best against
  `t2all_nodecay`.
- **Why does every frozen label decline?** Denial explains part of it — an improving
  learner takes cards its podmates would have had — but `gen3c` falling 1.21 while `gen3a`
  falls 0.36 in the same pods is not obviously denial, and the rollouts are retained this
  time, so it is answerable from the corpus.
- **The gen-3 fit-signal question** — whether starvation generalises or the fit signal is
  too weak — remains open and now has the training rollouts it needed.
