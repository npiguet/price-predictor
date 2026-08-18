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

Four runs, all `lr 1e-5`, all from the same base, all on the mix
`gen4:3,gen3a:2,gen3c:1,gen1:1,forge-full:1` with `--anchor gen3a`. 

| Run                 | learner T | field T | Rounds | Duration | Best margin (round) | Final margin | Checkpoint taken |
|---------------------|-----------|---------|--------|----------|---------------------|--------------|------------------|
| `t2all_nodecay`     | 2.0       | 2.0     | 83     | 1h42m    | +0.546 (r72)        | +0.379       | r72              |
| `t2all_decay0.3`    | 2.0       | 2.0     | 1268   | 22h39m   | +0.831 (r312)       | +0.468       | r312             |
| `t3all_decay0.3`    | 3.0       | 3.0     | 568    | 9h54m    | +0.676 (r208)       | +0.206       | r58              |
| `t3learner_t2field` | 3.0       | 2.0     | 177    | 4h07m    | +0.391 (r68)        | +0.002       | r68              |

All four peak and decline, as every gen-3 run did.

`t3all_decay0.3` is the one run whose yardsticked checkpoint is not its best-margin round, and
the choice is deliberate. Its margin climbed steadily to round 58, setting 24 new bests along
the way, then went 150 rounds without one before a single isolated reading of +0.676 at
round 208. That last step is +0.058 over round 58, well inside the ±0.15 noise floor the
metric carries at this window size (*The round-9 best is noise*, below). Round 58 sits at the
top of a real climb; round 208 is one lucky window for the learner's seats.

## The yardstick

All four checkpoints are measured. Each was taken through two 500-draft argmax runs: one
on the fixed mix `gen4:2,gen1:1,forge-full:1`, and one head-to-head against the promoted
gen-3 incumbent on `gen4:1,gen3:1`.

Every gen-4 candidate beats every reference by a wide margin, and every one of them beats
the gen-3 incumbent it was fine-tuned from. The gen-3 incumbent's own yardstick is included as the row to beat.

| Checkpoint          | vs gen-1         | vs `forge-full`  | vs gen-3 incumbent |
|---------------------|------------------|------------------|--------------------|
| `t2all_decay0.3`    | +1.380 ± 0.046   | +1.555 ± 0.052   | +0.634 ± 0.031     |
| `t2all_nodecay`     | +1.328 ± 0.049   | +1.519 ± 0.050   | +0.645 ± 0.031     |
| `t3learner_t2field` | +1.276 ± 0.052   | +1.384 ± 0.045   | +0.600 ± 0.033     |
| `t3all_decay0.3`    | +1.152 ± 0.050   | +1.337 ± 0.056   | +0.467 ± 0.030     |
| *gen-3 incumbent*   | *+0.824 ± 0.046* | *+0.920 ± 0.044* | —                  |

Each figure is a mean over pods of (mean candidate seat − mean reference seat) in that pod,
with the standard error clustered on set code. The naive un-clustered standard error treats every
pod as its own draw. That overstates the precision here, because a corpus samples a
random set per draft and most sets end up with several pods, and pods sharing a card pool move
together. Whatever makes a candidate unusually strong or weak in one set applies to every pod
drafted from it.

The last column can be checked against the first, because the gen-4-versus-gen-3 gap is
measurable two ways. Directly, from the head-to-head corpora where the two sit in the same
pods. Indirectly, by subtracting the incumbent's margin over gen-1 from each candidate's — a
different set of drafts, and one that assumes gen-1 scores consistently across corpora.

The indirect route puts the gap at roughly 0.3 to 0.55, the direct route at roughly 0.45 to
0.65, so the direct one reads about 0.1 higher throughout. Both rank the four candidates in
the same order. Two independent sets of drafts, played against different opponents, agreeing
on the ordering and roughly on the size means neither corpus is distorting its own result.

### Only the `T = 3` field separates from the rest

`t3all_decay0.3` is the one candidate the yardstick can tell apart, trailing the other three by
0.13 to 0.23 where the error bars run 0.04 to 0.07. The three above it sit inside their own
uncertainty of each other, and both corpora give the same picture despite being drawn
independently.

Gen-3's open temperature question is settled against the exploration band. The band pointed at
`T = 3`, the only value that holds perplexity 2–3 and off-argmax 25–40 %, and running every
agent there is the worst of the four settings tried. Training longer bought nothing either:
`t2all_decay0.3` ran 312 rounds against `t2all_nodecay`'s 72 and finished level with it.

## Raising the field's temperature costs; raising the learner's is free

The whole of the `T = 3` deficit belongs to the field's temperature, and none of it to the
learner's. The three runs sharing a field decompose it, because they change one temperature at
a time and their checkpoints come from within 15 rounds of each other.

| Contrast                               | What changes                   | Effect vs gen-1 | Effect vs gen-3 |
|----------------------------------------|--------------------------------|-----------------|-----------------|
| `t2all_nodecay` → `t3learner_t2field`  | learner 2 → 3, field held at 2 | −0.05 ± 0.07    | −0.05 ± 0.05    |
| `t3learner_t2field` → `t3all_decay0.3` | field 2 → 3, learner held at 3 | −0.13 ± 0.07    | −0.13 ± 0.05    |

The mechanism is the one the gen-3 spec proposed and gen-3's results appeared to refute
(§ 8.1). A field sampling at `T` sometimes passes a card it should have kept. The learner
then trains against packs a properly-playing field would never have handed it, and what it
learns from them does not transfer. Gen-3 compared field-at-T against field-at-argmax and
found sampling better; gen-4 varies the amount of sampling and finds less of it better. Both
are consistent with an optimum in the interior, near `T = 2`.

One incidental measurement supports the first row. In `t3learner_t2field` the learner and the
anchor hold identical weights and differ only in temperature, 3.0 against 2.0. Their round-0
margin is −0.042, so running the same policy hotter barely moves the deck it produces.

## `deck_score` does predict winning

Every ranking in this document is a `deck_score` difference, and gen-3 carried the assumption
that such differences turn into games won without ever testing it on drafted decks. They do, at
close to 15 points of match win rate per unit of score. Two runs measured it independently and
agree.

`python -m draft play-draft-games`
([`../specs/2026-08-09-draft-game-evaluation.md`](../specs/2026-08-09-draft-game-evaluation.md),
feature `022-draft-game-evaluation`) was run once over each of two `v-forge` yardstick corpora,
`t3learner_t2field`'s and `t2all_decay0.3`'s — third and first on the yardstick. Each run drew
1000 best-of-seven pairings from the recorded pods, mirrors excluded, with
`--forge-native-fraction 0.5` diverting half the `forge-full` seats to Forge's own deck builder.

| Label          | `t3learner_t2field` corpus | `t2all_decay0.3` corpus |
|----------------|----------------------------|-------------------------|
| gen-4          | 73.5 % ± 1.7 (781)         | 74.8 % ± 1.7 (810)      |
| `forge-full`   | 41.0 % ± 2.9 (329)         | 39.3 % ± 3.1 (298)      |
| gen-1          | 40.1 % ± 2.2 (581)         | 36.2 % ± 2.3 (575)      |
| `forge-native` | 18.8 % ± 2.4 (309)         | 21.8 % ± 2.4 (317)      |

The ordering is the yardstick's in both runs, on decks the scorer never saw played.

Every figure in this section counts matches won, not individual games. Which deck wins a single
game depends heavily on what each player happens to draw. A match only ends when one deck has
won four games, so most of that luck averages out.

### A unit of `deck_score` buys about 15 points of match win rate

Reading each matchup's Bo7 rate against the score gap measured in the same corpus turns the
ordering into a rate. The gaps are pod-paired means from the yardstick corpus the matches were
drawn from, so each row's two labels are compared inside the same pods. `forge-native` is
absent because it has no `deck_score`, its deck being rebuilt from the pool at game time; its
matchups are the subject of the next section.

| Matchup                 | corpus              | score gap | Bo7 match win rate | matches |
|-------------------------|---------------------|-----------|--------------------|---------|
| gen-4 over `forge-full` | `t2all_decay0.3`    | +1.555    | 71.2 % ± 3.9       | 177     |
| gen-4 over `forge-full` | `t3learner_t2field` | +1.384    | 71.9 % ± 3.5       | 192     |
| gen-4 over gen-1        | `t2all_decay0.3`    | +1.380    | 72.4 % ± 2.4       | 421     |
| gen-4 over gen-1        | `t3learner_t2field` | +1.276    | 69.1 % ± 2.5       | 418     |
| gen-1 over `forge-full` | `t2all_decay0.3`    | +0.197    | 52.9 % ± 5.8       | 85      |
| gen-1 over `forge-full` | `t3learner_t2field` | +0.105    | 51.9 % ± 5.9       | 81      |

A line through the origin fits all six rows at 15.1 points of Bo7 match win rate per unit of
`deck_score`. Each corpus on its own gives 15.4 and 14.8, and the individual rows imply 13.6 to
18.1.

### Swapping the deck builder is worth about as much as a generation of drafting

Two seats can draft identically and still end up with different decks, because which 40 cards
go in is decided after the draft ends. `forge-full` and `forge-native` are that pair. Both are
Forge's drafting AI, and both work from the same drafted pools. They differ only in which
program picks the 40 cards: this project's picker and simulated-annealing builder for
`forge-full`, Forge's own builder for `forge-native`. Playing the two against each other
therefore measures the builders and nothing else.

This project's builder wins that matchup in both runs.

| Corpus              | This project's builder wins | Matches |
|---------------------|-----------------------------|---------|
| `t3learner_t2field` | 75.0 %                      | 56      |
| `t2all_decay0.3`    | 72.2 %                      | 36      |
| Pooled              | 73.9 %                      | 92      |

Gen-4 wins 70.8 % of its 839 matches against gen-1, which is what one generation of drafting
improvement is worth. That is close to the builder's 73.9 %, and 92 matches put an error bar of
about 5 points on the builder figure, so the two cannot be told apart. Changing who builds the
deck buys roughly what a generation of better drafting buys.

The builder also decides the bottom of the table. `forge-native` loses to every other label,
including `forge-full`, which drafted the same pools with the same agent. Gen-4's largest
margin over any label is against `forge-native` rather than against any drafting agent.

### Forge pads a tenth of its decks with basics rather than lowering its bar

About a tenth of `forge-native` decks are not decks. Of the 615 built across the two runs, 65
hold 21 or more lands in 40 cards, out to 33. Forge picks 22 spells if it can and fills the
rest of the deck with basic lands. Those decks win a tenth of their matches, and none at all
past 25 lands, which costs `forge-native` around four points of win rate on its own.

Two things leave Forge short of 22 spells. The first is the colour commitment: the drafter
fixes two colours early and the builder is handed the same pair, so a pool that ran dry in
those colours has too little to play. Every seat holding fewer than 22 on-colour cards built a
land-heavy deck, without exception. The second is that Forge discards cards its own AI handles
badly. These are build-around cards, flagged `RemRandomDecks`: the builder keeps one when the
partners it needs are already in the deck and drops it when they never arrived. The drafter
never reads that flag, so it takes such cards on raw power and only finds the hole at build
time. Land-heavy seats that did hold 22 on-colour cards carried more than three times the usual
share of them.

### Creature count predicts winning, and no colour costs gen-4 games

Decks with more creatures win more, in both runs and inside every label. The composition
analysis assumed that direction and the games confirm it.

| Bo7 win rate                   | ≤ 13 creatures | ≥ 20 creatures |
|--------------------------------|----------------|----------------|
| all decks, `t3learner_t2field` | 25.7 %         | 74.2 %         |
| all decks, `t2all_decay0.3`    | 26.2 %         | 70.8 %         |
| gen-4, `t3learner_t2field`     | 50.0 %         | 79.5 %         |
| gen-4, `t2all_decay0.3`        | 54.3 %         | 75.6 %         |

Gen-4's win rate is about the same in all five colours. The differences are small enough to be
sampling noise, given how many matches each colour appears in. Its colour lean therefore costs
it nothing. White is the colour it plays most, and it wins slightly more often in white than it
does overall.

These are rates conditional on a deck containing the colour. They are not evidence that gen-4
plays the five colours equally often, and it does not: white appears in about three fifths of
its decks and red in under a third. *Hypothesis 3* below measures that lean and explains where
it comes from.

What the games cannot say is whether that lean is the best one available. Testing that would
mean forcing gen-4 into colours it did not pick and seeing whether it wins more, and no run does
that.

## No in-run metric ranks the checkpoints

All four checkpoints now have a yardstick, so the metrics the training loop reports can be
checked against it rather than trusted.

### The margin decomposition heuristic is refuted

Gen-3 used the split between the learner's own score and the anchor's to discount a margin that
read healthy. Its `lr 1e-4` learner peaked near round 15 and fell back to its starting level
while the anchor kept dropping, so the margin after that was the field declining rather than the
learner improving
([`2026-06-15-draft-agent-gen3-online-grpo-design.md`](2026-06-15-draft-agent-gen3-online-grpo-design.md),
*Movement, and the learning-rate sweep*). Used to rank runs instead, the same split fails.
Measured from the first full window to each run's best round:

| Run                 | Δ learner | Δ anchor | anchor's share | yardstick vs gen-1 |
|---------------------|-----------|----------|----------------|--------------------|
| `t2all_nodecay`     | −0.12     | −0.71    | 85 %           | +1.328             |
| `t2all_decay0.3`    | +0.28     | −0.33    | 54 %           | +1.380             |
| `t3all_decay0.3`    | +0.37     | −0.28    | 43 %           | +1.152             |
| `t3learner_t2field` | +0.33     | −0.19    | 37 %           | +1.276             |

The heuristic ranks these four in exactly the order the yardstick reverses. `t2all_nodecay` is
its worst run, with the learner's own score falling and 85 % of the margin coming from the
anchor collapsing, and it ties for best on the yardstick. Do not rank runs this way.


## Crowding a pod with strong drafters costs every seat in it

Gen-3 attributed the field's decline to denial, and left the size of the effect open. The
yardstick corpora measure it directly, because `--agent-mix` is sampled independently per
seat and pods therefore vary in how many gen-4 seats they contain.

Gen-1 and `forge-full` are treated as one reference label throughout this section. The pick
alignment measured further down finds they weight every axis of card behaviour identically, and
here they decline in step, gen-1 sitting about 0.1 above `forge-full` at every composition. Pooling them also makes both corpus families a two-label eight-seat pod at an even
split, so the gen-4 count fixes the whole composition and the two tables are read the same way.
All four checkpoints are pooled; the two carry about 7900 and 8000 gen-4 seats.

| gen-4 seats in the pod | gen-4 | gen-1 + `forge-full` | gap   |
|------------------------|-------|----------------------|-------|
| 1                      | +2.77 | +1.44                | +1.33 |
| 2                      | +2.49 | +1.16                | +1.32 |
| 3                      | +2.33 | +1.04                | +1.30 |
| 4                      | +2.17 | +0.80                | +1.37 |
| 5                      | +2.05 | +0.63                | +1.43 |
| 6                      | +1.89 | +0.38                | +1.52 |
| 7                      | +1.62 | +0.19                | +1.43 |

| gen-4 seats in the pod | gen-4 | gen-3 | gap   |
|------------------------|-------|-------|-------|
| 1                      | +2.40 | +1.63 | +0.76 |
| 2                      | +2.06 | +1.51 | +0.55 |
| 3                      | +2.00 | +1.39 | +0.60 |
| 4                      | +1.85 | +1.29 | +0.56 |
| 5                      | +1.81 | +1.24 | +0.57 |
| 6                      | +1.74 | +1.12 | +0.62 |
| 7                      | +1.76 | +1.07 | +0.68 |

Both fall monotonically, and neither label escapes: crowding a pod with strong drafters takes
cards out of the packs that reach everyone in it, gen-4 seats included.

Crowding costs less against gen-3 than against the Forge-like reference — 0.095 per seat where
it was 0.205, and 0.075 for gen-4's own seats where it was 0.159. Nothing about the mechanism
differs between the two. What differs is the size of the upgrade each swapped seat represents.
Displacing a reference seat at +0.8 with gen-4 at +2.2 adds far more competition than
displacing gen-3 at +1.3 with the same seat.

Dividing the cost by the gap makes that explicit, and the two families agree to within a tenth
of each other on both rows.

| Family    | seat displaced       | gen-4's lead over it | cost to a rival seat | ratio | cost to a gen-4 seat | ratio |
|-----------|----------------------|----------------------|----------------------|-------|----------------------|-------|
| `v-forge` | gen-1 + `forge-full` | +1.377               | −0.205               | 0.149 | −0.159               | 0.116 |
| `v-gen3`  | gen-3                | +0.587               | −0.095               | 0.162 | −0.075               | 0.128 |

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

| Family    | naive per-seat difference | pod-paired | bias   | predicted |
|-----------|---------------------------|------------|--------|-----------|
| `v-forge` | +1.184                    | +1.377     | −0.193 | −0.191    |
| `v-gen3`  | +0.498                    | +0.587     | −0.089 | −0.084    |

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

## More creatures, fewer rares, narrower mana bases

From the four `v-forge` corpora, where gen-4 shares pods with both references.

|                           | `t2all_nodecay` | `t2all_decay0.3` | `t3all_decay0.3` | `t3learner_t2field` | gen-1       | forge-full  |
|---------------------------|-----------------|------------------|------------------|---------------------|-------------|-------------|
| creatures                 | 18.01           | 18.19            | 17.79            | 17.91               | 15.22–15.28 | 15.09–15.16 |
| avg mana value            | 3.17            | 3.21             | 3.19             | 3.19                | 3.02–3.05   | 3.03–3.07   |
| rares                     | 1.24            | 1.24             | 1.25             | 1.33                | 1.69–1.74   | 1.86–1.95   |
| ≥ 4 basic land types      | 4.5 %           | 5.6 %            | 4.7 %            | 6.4 %               | 7.0–9.8 %   | 9.0–10.9 %  |
| score of those wide decks | +1.30           | +1.65            | +1.47            | +1.65               | −0.03…+0.28 | +0.03…+0.14 |

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

## Gen-4's margin depends on the set, and core sets are its weakest ground

The set a pod drafts explains about a fifth of the variation in gen-4's margin over
`forge-full`, and the pattern survives both checks that could have shown it to be sampling
luck. The four `v-forge` corpora pool to 1778 pods over 181 sets, a median of ten pods each,
with each checkpoint's own mean removed first so that only set variation remains.

The between-set spread is about half the within-set spread, at an F of 3.2 on 180 and 1597
degrees of freedom. That is the effect the yardstick's clustered standard errors exist to
absorb, measured directly.

Two checks say the ranking carries information. Scoring each set on two of the four checkpoints
and again on the other two, the halves correlate at +0.49. Scoring against `forge-full` and
against gen-1, different opponents drawn from the same pods, correlates at +0.68.

Grouping sets by what kind of product they are gives the generalisation.

| Edition type | sets | pods | mean margin |
|--------------|------|------|-------------|
| Draft        | 11   | 122  | +1.63       |
| Reprint      | 15   | 166  | +1.62       |
| Online       | 23   | 215  | +1.48       |
| Expansion    | 106  | 1032 | +1.44       |
| Core         | 22   | 205  | +1.29       |
| Starter      | 3    | 33   | +1.12       |

Gen-4 gains least where the cards are simplest. Core and starter sets are built to teach the
game: flat power curves, few build-arounds, little that rewards knowing the format. There is
less for a drafter to get right, so Forge's heuristics give up less. The widest margins are on
curated products and small old expansions, where power is uneven and the themes are narrow.

Individual sets are shrunk toward the grand mean before ranking. A set's own mean is a noisy
estimate of its true margin, and the fewer pods it has the noisier it is, so ranking on raw
means would put the sets with the luckiest small samples at both ends. Shrinkage replaces each
set's mean with a weighted average of that mean and the grand mean, the weight being the share
of the set's apparent deviation that the data can attribute to the set rather than to sampling:

```
weight = τ² / (τ² + σ²/n)
```

`τ²` is the between-set variance estimated above, `σ²` the within-set variance, and `n` the
set's pod count. A set measured on many pods keeps most of its own mean. A set measured on few
is pulled most of the way back to the middle, because at that sample size its deviation is as
easily explained by luck. At the median ten pods the weight is about 0.7, so the ranking stays
mostly the sets' own means and only the thinnest samples move far. This is the empirical-Bayes
estimator, using the observed spread of set means to set how much to trust each one.

Raw means are given beside the shrunk ones to show how much work it does. Planeshift is the
clearest case: six pods, a raw margin of +3.04, and a shrunk one of +2.36.

| Set                        | Type      | Margin | Raw    | Pods |
|----------------------------|-----------|--------|--------|------|
| MB1 Mystery Booster        | Draft     | +2.37  | +2.59  | 19   |
| PLS Planeshift             | Expansion | +2.36  | +3.04  | 6    |
| KLR Kaladesh Remastered    | Online    | +2.13  | +2.39  | 12   |
| DST Darksteel              | Expansion | +2.13  | +2.63  | 6    |
| PCY Prophecy               | Expansion | +2.09  | +2.29  | 14   |
| *… 175 sets between …*     |           |        |        |      |
| 9ED Ninth Edition          | Core      | +0.91  | +0.64  | 9    |
| M11 Magic 2011             | Core      | +0.87  | +0.65  | 12   |
| KTK Khans of Tarkir        | Expansion | +0.85  | +0.63  | 12   |
| FRF Fate Reforged          | Expansion | +0.71  | +0.46  | 13   |
| 3ED Revised Edition        | Core      | +0.63  | +0.17  | 8    |

The bottom of the table is core sets and the Khans block. The core sets follow from the
edition-type reading. Khans and Fate Reforged do not, and are the one part of the ranking that
asks for an explanation rather than supplying one.

`forge-native` cannot be ranked this way at all. It carries no `deck_score`, so the only
per-set measure available is games won, and the two played corpora supply 383 gen-4 versus
`forge-native` matches spread over 142 sets. Under three matches a set puts the standard error
on a per-set win rate above 20 points, which is wider than the entire effect being measured.
Ranking sets by games needs a run that fixes the set rather than sampling it.

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

| Corpus              | gen-4          | gen-1           | forge-full      |
|---------------------|----------------|-----------------|-----------------|
| `t2all_nodecay`     | 11.9 % (49.7)  | 9.5 % (46.0)    | 11.7 % (57.7)   |
| `t2all_decay0.3`    | 12.6 % (54.2)  | 8.3 % (46.3)    | 11.1 % (57.3)   |
| `t3all_decay0.3`    | 8.6 % (34.0)   | 9.9 % (49.0)    | 12.9 % (58.4)   |
| `t3learner_t2field` | 12.0 % (48.9)  | 9.8 % (49.6)    | 12.2 % (51.7)   |
| *gen-3 incumbent*   | *9.6 % (34.2)* | *10.8 % (45.4)* | *11.5 % (50.4)* |

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

| Corpus                          | agent                | best-card rate | mean premium    | share above zero |
|---------------------------------|----------------------|----------------|-----------------|------------------|
| `t2all_decay0.3`                | gen-4                | 25.1 %         | +0.0318         | 68.2 %           |
| `t3learner_t2field`             | gen-4                | 24.4 %         | +0.0287         | 66.2 %           |
| `t2all_nodecay`                 | gen-4                | 25.3 %         | +0.0260         | 65.7 %           |
| `t3all_decay0.3`                | gen-4                | 24.0 %         | +0.0225         | 62.1 %           |
| gen-3 incumbent                 | gen-3                | 22.9 %         | +0.0240         | 63.0 %           |
| gen-3, field at argmax, `T = 3` | gen-3                | 28.5 %         | +0.0049         | 51.9 %           |
| *references, all corpora*       | *gen-1 / forge-full* | *18.8–21.7 %*  | *+0.007…+0.015* | *53.4–58.7 %*    |

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

| Corpus              | gen-4    | gen-1 | forge-full | learner picks, this generation |
|---------------------|----------|-------|------------|--------------------------------|
| `t3all_decay0.3`    | +2.91 pp | −0.22 | −0.10      | ~75k                           |
| `t3learner_t2field` | +3.61    | +0.23 | −0.78      | ~90k                           |
| `t2all_nodecay`     | +4.09    | −0.18 | +0.51      | ~95k                           |
| `t2all_decay0.3`    | +3.99    | −0.21 | −0.58      | ~405k                          |

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

### `cast_lift` is the axis gen-4 gained on

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

| Axis          | `t2all_decay0.3` | `t2all_nodecay` | `t3learner_t2field` | `t3all_decay0.3` | gen-3 incumbent | gen-1       | forge-full  |
|---------------|------------------|-----------------|---------------------|------------------|-----------------|-------------|-------------|
| `score_play`  | 0.703            | 0.688           | 0.684               | 0.679            | 0.662           | 0.603–0.610 | 0.593–0.610 |
| `score_draw`  | 0.698            | 0.685           | 0.683               | 0.675            | 0.659           | 0.601–0.606 | 0.592–0.607 |
| `played_rate` | 0.643            | 0.653           | 0.644               | 0.636            | 0.632           | 0.587–0.589 | 0.573–0.583 |
| `cast_lift`   | 0.604            | 0.588           | 0.588               | 0.581            | 0.561           | 0.523–0.529 | 0.518–0.530 |
| `color_lift`  | 0.508            | 0.512           | 0.517               | 0.530            | 0.529           | 0.557–0.563 | 0.559–0.568 |

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

### Card-choice quality tracks the yardstick, colour does not

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
generation, and each beats the gen-3 incumbent by about +0.6 head-to-head. `t2all_decay0.3` is
also the one candidate whose decks have been played in bulk, taking 74.8 % of 810 matches
against a field of gen-1 and Forge seats. `t2all_nodecay` is the cheaper of the two by a factor
of thirteen in wall-clock and reaches the same place on score, so prefer it unless a reason to
prefer the longer run appears.

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
metric available to the loop is a derivative of the same frozen `deck_score`, and 2000 played
matches across two corpora now say that number buys about 15 points of Bo7 match win rate per
unit, with no offset at zero. Nothing above needs re-reading as a proxy result. What remains is
that the exchange rate is shallow enough to make games an expensive way to compare candidates:
the 0.23 separating the best from the worst is worth about 3 points of match win rate, which is
thousands of matches to resolve.

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

1. Does `deck_score` predict winning? Yes, at about 15 points of Bo7 match win rate per unit,
   fitted on six matchups across two corpora. Settled.
2. Does the colour prior survive contact with games? Gen-4's win rate varies by 2.5 to 6.3
   points across WUBRG, inside what its sample sizes support, so no colour it plays costs it
   games. Settled as far as an observational read can settle it.
3. Do the four candidates rank the same way on games as on score? Half open. The two played so
   far finish in the yardstick's order, but 3.3 points apart on a standard error of 3.5.

Sizing, so the third is armed deliberately: a head-to-head win rate needs roughly `1.96/δ²`
matches for 80 % power at α = 0.05 — about 200 matches to resolve 60/40, 800 for 55/45 and
2200 for 53/47, before any inflation for the clustering that comes from reusing a deck across
pairings. Each run bought 1000 matches with about two hours of twelve workers, so those are
hours rather than days. Separating adjacent candidates at 3 points sits past the right-hand end of
the scale, so the `v-gen3` corpora — two labels in every pod, and a 0.6 score gap rather than a
0.23 one — are the place to spend the time.

## Open questions

- **Is `t3all_decay0.3`'s deficit temperature or training length?** Its checkpoint is round 58,
  a quarter of the rounds behind the shortest of the other three, because that is where its
  margin stopped climbing. Whether a `T = 3` field is slower rather than worse is answerable by
  running it again and stopping only on a genuine plateau.
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
- **How much of the builder gap survives removing the broken decks?** A tenth of `forge-native`
  decks are land-heavy wrecks that win almost nothing, and they are worth about four points of
  the label's average on their own. Whether Forge's builder is merely worse or mostly fine
  outside that tail is answerable by re-tallying the existing corpora with those decks excluded,
  at no Forge cost.
- **Would relaxing the 22-spell target rescue the starved seats?** Forge pads to 40 with basics
  rather than playing a 21st off-colour card or an 18th spell alongside 19 lands. A seat with
  only 18 on-colour cards would plainly rather run 20 lands and two splashes. This is Forge's
  code, so the test is a local patch to `LimitedDeckBuilder` rather than a change here.
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
