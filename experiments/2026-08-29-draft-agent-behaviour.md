# What makes the draft agent a good drafter

## The short version

- Gen-4 drafts from a sharper list, not from a better read of the table. About
  half its picks are decided by a fixed card ranking that ignores everything on
  the board, and reinforcement learning raised that share rather than lowering it.
- **It cannot read a signal.** Erasing what the other players took moves its
  picks *less* than erasing the same number of random cards. Gen-1 is the same, so
  this was never learned and never lost.
- What it does read is which cards are its own. Relabelling its pool as cards
  others took — no card changed, no count changed — moves two picks in five, and
  that alone produces its two-colour discipline.
- It has a standing colour preference. At the first pick of a draft, with nothing
  known about the table, it already favours white and green and avoids red and
  blue. Gen-1 has no such preference and if anything favours red, so this is the
  reinforcement learning's doing.
- It moved away from human pick order and toward its reward's, monotonically
  across the generations. It gives up ground on removal and card draw and gains it
  on creatures. On identical states, gen-1's pick sits higher in Forge's human
  ranking than gen-4's — it picks differently, and the direction only counts as
  better against the opponent it trained on. About a quarter of that move is
  drift: it keeps going in a checkpoint that trained two and a half times as long
  and finished no stronger.
- It prefers redundancy. A copy of a card already in the seat's pool raises that
  same card's value in the pack, most in gen-1 and least in gen-4. The reward is
  blind to duplicates, so nothing could have taught this in either direction and
  the whole effect comes from imitating Forge.
- Its learning went where its reward could see. The change from gen-1 grows with
  a card's quality and is twice as large on cards in the seat's colours, because
  a card that misses the built deck contributes nothing to the score.
- It starves the player it passes to. A seat downstream of gen-4 loses about what
  one more strong drafter joining the pod would cost it, and a seat downstream of
  gen-1 loses nothing measurable. The agent cannot see where it sits, so this is a
  consequence of its taste rather than a strategy.
- Its training barely used its training signal. Every gradient step was clipped to
  the same length, and how far the policy moved in a round is close to independent
  of how much that round had to teach.

## Method and subject

This document records an interpretability study of the gen-4 production draft
agent: which drafting strategies it uses, which it does not, and which of its
habits belong to it rather than to the two models it is built on. It is the
third of a set. The deck scorer that supplies its reward has its own study
([`2026-08-27-scorer-preferences.md`](2026-08-27-scorer-preferences.md)), and so
does the card encoder that supplies its inputs
([`2026-08-28-encoder-preferences.md`](2026-08-28-encoder-preferences.md)). The
question here is what is left once those two are accounted for.

The subject is `models/draft/agent/gen4/lr1e-5_t2all_decay0.3.pt`, the strongest
of the four gen-4 candidates on the yardstick. Its sibling
`lr1e-5_t2all_nodecay.pt` — the same base, the same settings, a different run —
supplies the noise floor wherever a measurement needs one. Its lineage is
gen-1 imitation of Forge's drafting AI, then online GRPO at gen-3 and again at
gen-4. Background:
[`2026-08-09-draft-agent-gen4-online-grpo.md`](2026-08-09-draft-agent-gen4-online-grpo.md).

The method had three phases. Thirty-eight hypotheses were brainstormed and
critiqued by four independent Opus reviewers, one each on architectural
feasibility, learning dynamics, cross-model attribution, and coverage against
human drafting technique; the survivors were consolidated into sixteen ranked
hypotheses. A probe battery then tested them, inference-only, on frozen
checkpoints. Every probe lives in
[`scripts/draft_probes/`](../scripts/draft_probes/README.md), and each section
below names the script it comes from.

Three rules run through every measurement, and each of them changed a result.

**Every number carries a gen-1 column.** Gen-1 is a distillation of Forge's
`LimitedPlayerAI`, trained by cross-entropy on `forge-full` seats alone. A
behaviour gen-1 already has is Forge's. The colour lean is the case that turns on
this: gen-1's runs the opposite way from gen-4's, so a measurement of gen-4 alone
would have attributed it to Forge.

**Only within-state logit contrasts are behavioural.** The policy head is a
single linear map on each `PACK` token followed by a softmax over that state's
tokens, so adding a constant to every logit of a state changes nothing. Every
logit quoted below is centred inside its own state.

**Edits preserve token count.** Deleting a block of tokens also changes how many
tokens the trunk averages over, which moves the logits by itself. Blocks are
blanked by substituting a corpus-mean card vector for their cards instead, and
read against a random substitution of the same size.

The behavioural corpus is the `v-forge` yardstick run of the promoted
checkpoint: 500 drafts, eight-seat pods, three packs of fifteen, with `gen4`,
`gen1` and `forge-full` seats sharing every pod, all playing their argmax. The
corpus-side analyses pool all four candidates' yardstick runs, 2,000 drafts and
16,000 seats.

Replay is exact. States are rebuilt with `draft.domain.draft_state.build_state`,
the full-record oracle the live pick tracker is pinned to, and replaying the
argmax corpus through the checkpoint that generated it reproduces 100 % of the
recorded picks. The trainers' own state walk reproduces 97.7 % of them: it
freezes a `TAKEN` card's recency at the moment the card left the pack where
`build_state` recomputes it from the current clock, and the two differ whenever a
card name recurs in a later pack. The probes use `build_state`, so the policy
being measured is the one that was deployed.

## What the code settles before any experiment runs

Four facts come from reading the state builder and the trainer, and they retired
four hypotheses before a GPU was touched.

The agent cannot see the table for the first eight picks of a pack. The `TAKEN`
block fills from two sources only: a wheel diff, which first fires when a booster
returns after `pod_size` picks, and a flush of everything unresolved at each pack
boundary. In an eight-seat pod that means `TAKEN` is empty until pick 9 of pack 1
— the window in which a human drafter reads signals hardest.

What `TAKEN` holds is the picks made *downstream*, not upstream. The wheel diff
recovers the cards taken between the seat's two sightings of one booster, which
are exactly the picks of the seats the booster passed through after leaving this
seat. After the first pack-end flush the block also holds every card the seat
ever passed, so from pack 2 on it mixes what was sent to the seat with what was
taken from it.

There is no rarity, no set code, no seat index and no pass direction anywhere in
the state. The 32 deterministic features are a land flag, colour pips, colour
flags, mana value, power, toughness, loyalty and mana production. "Gen-4 passes
on rares" cannot be a policy term; rarity reaches the model only as whatever its
text implies.

Denial is priced at exactly `1/(pod_size − 1)`, or 0.143 per unit of harm to a
rival. The reward is one terminal pod-relative scalar shared by all 45 of a
seat's picks, gen-3 and gen-4 together saw about 12,500 of them, and the learner
took its own argmax about 90 % of the time. A per-seat effect below roughly
0.014 score units is not resolvable by that estimator, and a typical
hate-draft's benefit works out near 0.004. Measuring gen-4's denial as
indistinguishable from zero is the predicted behaviour of the training setup, not
a shortcoming of the agent.

## It reads its own pool, and nothing else on the table

Erasing what the other players took moves the policy less than erasing the same
number of random cards. That is the central mechanism result, and it holds in
every pack and in every generation. `d1_channels.py` blanks one block at a time
by substituting a corpus-mean card vector for its cards, which keeps the token
count fixed, and separately blanks `k` uniformly chosen non-`PACK` tokens for a
ladder of `k`. The ladder is the magnitude law; a block's honest weight is where
it sits against the ladder at its own size.

Sizes differ enormously across the draft, so the comparison is made inside each
pack. By pack 3 the cards a seat knows others took outnumber the cards in its own
pool by nearly four to one.

| block | mean size | picks changed | placebo of the same size | ratio |
|---|---|---|---|---|
| `POOL`, pack 1 | 6.9 | 0.399 | 0.083 | 4.8 |
| `POOL`, pack 2 | 21.8 | 0.452 | 0.060 | 7.5 |
| `POOL`, pack 3 | 36.7 | 0.459 | 0.054 | 8.6 |
| `PASSED`, pack 1 | 31.8 | 0.191 | 0.252 | 0.75 |
| `PASSED`, pack 2 | 32.0 | 0.086 | 0.091 | 0.95 |
| `PASSED`, pack 3 | 32.4 | 0.054 | 0.048 | 1.12 |
| `TAKEN`, pack 1 | 12.8 | 0.033 | 0.126 | 0.26 |
| `TAKEN`, pack 2 | 77.9 | 0.159 | 0.243 | 0.65 |
| `TAKEN`, pack 3 | 142.9 | 0.183 | 0.281 | 0.65 |

Gen-4, 7,587 states. The three other checkpoints give the same picture: `POOL`
runs between 4.7 and 11.4 times its placebo, `PASSED` between 0.65 and 1.12, and
`TAKEN` between 0.23 and 0.70 without once exceeding 1.

Reading the table is the drafting skill this agent does not have. `TAKEN` holds
what the seats downstream took, which is the only direct evidence about the pod
in the state at all, and blanking it is worth less than blanking noise. `PASSED`
sits at the law, so the cards the seat declined this pack are worth nothing to it
either. Neither result is a limitation of the probe: the same probe finds `POOL`
five to nine times above the law.

What the agent does read is which cards are its own. Relabelling every `POOL`
token as `TAKEN` changes no card and no count, destroying only the claim of
ownership, and it moves as many picks as blanking the pool's card identities
outright. The reverse edit is almost free.

| edit | card identities changed | picks changed |
|---|---|---|
| pool cards blanked | 21.8 | 0.437 |
| pool relabelled as taken by others | 0 | 0.431 |
| every pool card replaced by the pool's own mean | 21.8 | 0.136 |
| taken relabelled as passed | 0 | 0.050 |
| all recency tags zeroed | 0 | 0.090 |
| pack number rewritten to 1 | 0 | 0.004 |
| pick number rewritten to 1 | 0 | 0.005 |

The pool is not read as an average. Replacing every pool card with the pool's own
mean vector preserves the average exactly and destroys the composition, and it
still moves one pick in seven. The scorer and the encoder both pool uniformly;
the draft policy does not, which sets it apart from the two models under it.

The clock token does almost nothing. Rewriting the `CONTEXT` token's pack number
or pick number changes about one pick in two hundred, so whatever temporal
structure the policy has arrives through the pool and the recency tags rather
than through the counter it was given.

## A card already in the pool is worth more, not less

The policy prefers redundancy. Putting a copy of a card into the seat's pool
raises that same card's logit in the pack, by about three times the gap between
the two near-identical cards the probe compares, and it does so in every
generation. Nothing in the reward could have taught the preference, because the
scorer prices a second copy of a card exactly like the first
([`2026-08-27-scorer-preferences.md`](2026-08-27-scorer-preferences.md),
*Duplicates are priced like distinct cards of the same quality*). Its presence is
therefore inherited from imitating Forge.

`d8_duplicates.py` measures it with a two-arm edit on one state. Two cards are
chosen from the pack, matched on colour identity and as close as possible in the
model's own logit; the median pair differs by 0.055 logits. One pool slot is then
overwritten, in one arm with a copy of the first card and in the other with a copy
of the second. The arms hold the pack, the pool's size, its recency tags and its
colour mix identical, and differ only in which of two near-equivalent cards the
seat now owns. Both cards are measured, each in the arm that copied it against the
arm that did not, so the effect is estimated twice from opposite sides.

| | one copy | two copies | three copies | placebo card |
|---|---|---|---|---|
| gen-1 | +0.269 | +0.496 | +0.684 | −0.016 |
| gen-3 | +0.168 | +0.346 | +0.500 | −0.034 |
| gen-4 | +0.174 | +0.357 | +0.503 | −0.075 |
| gen-4 sibling | +0.218 | +0.424 | +0.579 | −0.057 |

Change in the card's centred logit, over 17,109 pairs measured from both sides;
standard errors run from 0.004 to 0.010. The placebo column is a third card in
the same pack that neither arm copied, which is where the edit's own size would
show if the effect were not about the copied card.

Reinforcement learning cut the redundancy bonus without removing it. Gen-1 carries
the largest and gen-3 and gen-4 about a third less. The reduction is roughly twice
the spread between the two gen-4 siblings, so it is larger than run-to-run
variation but not by much. A policy trained on a reward blind to duplicates
drifted away from valuing them, which is what a reward saying nothing about a
behaviour produces: the behaviour decays rather than reversing.

Part of the effect is card similarity rather than card identity. The two arms
differ in which card enters the pool, so an arm that inserts a card resembling the
target raises the target's logit through the same pull the pool exerts by colour.
Splitting the pairs by how close the two cards sit in embedding space separates
the two readings.

| how close the control card sits | gen-1 | gen-3 | gen-4 | gen-4 sibling |
|---|---|---|---|---|
| cosine 0.00–0.55 | +0.335 | +0.194 | +0.237 | +0.291 |
| cosine 0.55–0.68 | +0.281 | +0.179 | +0.181 | +0.228 |
| cosine 0.68–0.83 | +0.253 | +0.163 | +0.148 | +0.195 |
| cosine 0.83–1.00 | +0.206 | +0.133 | +0.132 | +0.158 |

One copy. The effect falls as the control card approaches the target, which is
the similarity component, and it does not fall to zero at the top bucket. Exact
identity is therefore worth something beyond resemblance, though the size of that
remainder rests on extrapolating a four-point trend past its last bucket.

The bonus fades across the draft and never turns negative. Measured on one copy,
it runs at about half a logit in pack 1 in every generation and is much smaller by
pack 3, falling furthest in gen-4 and least in gen-1. Part of the fade is
arithmetic, since one overwritten slot is a much larger share of a seven-card pool
than of a thirty-seven-card one, and in gen-1 the pool's growth accounts for all
of the fade.

The direction that never appears is the one a drafter needs. Late in a draft a
second copy of a card displaces a card the seat already owns rather than one a
rival would have taken, so its marginal value should fall below that of a fresh
card of equal quality. No generation prices it there. The reward is computed on a
finished deck and says nothing about the order cards arrive in, so no gradient
ever pointed that way.

## Half of gen-4's picks are decided before it looks at the board

At pack 1 pick 1 the pool, the passed cards and the taken cards are all empty by
construction, so the policy at that state is a pure card ranking with nothing to
condition on. Reading it off 4,000 opening boosters across 165 sets gives each
model's context-free pick order directly, with no fitting. `d2_pickorder.py`
produces it.

The same order predicts about half of every pick the agent makes anywhere in the
draft. Restricting each model's own P1P1 ranking to the cards in a pack and
taking its maximum reproduces the model's actual choice at the rates below, over
27,066 states. Chance is 0.222, the mean of one over the pack size across those
states.

| | overall | pack 1 | pack 2 | pack 3 |
|---|---|---|---|---|
| gen-1 | 0.429 | 0.430 | 0.433 | 0.422 |
| gen-3 | 0.443 | 0.447 | 0.443 | 0.436 |
| gen-4 | 0.491 | 0.496 | 0.490 | 0.484 |
| gen-4 sibling | 0.488 | 0.493 | 0.495 | 0.472 |

Reinforcement learning made the fixed list matter more, not less. Gen-4 is six
points more context-free than gen-1, and the ranking itself sharpened: the spread
of the per-card P1P1 value doubles from gen-1 to gen-4.

The share is flat across the three packs. A pool three packs deep buys no more
deviation from the fixed order than an empty one does, which is the first sign
that the agent's context use is a colour term rather than a running plan.

Within a pack the share tracks how much choice there is. Gen-4 follows its fixed
order on about two thirds of first picks, on a third of picks 6 to 10, on three
quarters of pick 14, and on effectively all of pick 15, where one card is left.

## Gen-4 traded human pick order for its reward's order

The clearest thing reinforcement learning did to the pick order is move it away
from how humans draft. Forge ships a hand-written pick-order file, which is
exogenous to all three models in this stack and is itself a pick order rather
than a card rating, so it is the right yardstick. Agreement with it falls at
every generation.

| source | gen-1 | gen-3 | gen-4 | gen-4 sibling |
|---|---|---|---|---|
| Forge's human draft rank | 0.793 | 0.771 | 0.658 | 0.690 |
| the scorer's `v_swap` | 0.372 | 0.597 | 0.697 | 0.736 |
| encoder text PC1 (played rate) | 0.198 | 0.398 | 0.456 | 0.523 |
| encoder text PC2 (winnability) | 0.320 | 0.437 | 0.530 | 0.507 |

Spearman correlations of the P1P1 card scalar against each source, over the 3,770
cards ranked by all of them. Paired on the same cards, gen-4's loss against the
human order is −0.135 with a bootstrap interval of −0.152 to −0.118.

About a quarter of the move away from the human order is drift rather than
learning, which the last section of this document separates out.

The two movements are one trade. Gen-1's order is essentially the human order:
of the variance in its pick order, 0.47 is uniquely explained by the human rank
and only 0.04 uniquely by the encoder's axes. By gen-4 the uniquely-human share
has fallen to 0.15 while the uniquely-encoder share has tripled. The agent did
not lose card judgement; it swapped whose judgement it was using.

Where it gave ground is the category the scorer study predicted. Each column
below is that model's standardised P1P1 scalar minus gen-1's, averaged over the
cards of one category:

| category | n | gen-3 | gen-4 | gen-4 sibling |
|---|---|---|---|---|
| noncreature removal | 276 | −0.208 ± 0.022 | −0.440 ± 0.038 | −0.450 ± 0.035 |
| card draw | 181 | −0.247 ± 0.028 | −0.279 ± 0.046 | −0.342 ± 0.043 |
| other noncreature | 1220 | −0.130 ± 0.011 | −0.153 ± 0.018 | −0.200 ± 0.017 |
| creature | 2096 | +0.125 ± 0.008 | +0.171 ± 0.016 | +0.205 ± 0.014 |

The human BREAD rule puts removal second only to bombs. Gen-4 demotes it further
than any other category, and promotes creatures, moving removal about six tenths
of a standard deviation below creatures relative to where gen-1 had them. The
scorer prices an average creature above an average removal spell
([`2026-08-27-scorer-preferences.md`](2026-08-27-scorer-preferences.md), *The
category order is creatures, then removal*), and the policy has reproduced that
ordering in its pick order. This is the reward's taste arriving at the pick, and
it is a measurable departure from human play rather than a refinement of it.

## The colour lean is a standing prior, and Forge does not have it

Gen-4's preference for white, green and black was known from its off-lane picks,
where the seat has a pool and a lane and could be reacting to what is open. Pack
1 pick 1 removes that possibility, because nothing is open yet. The lean is
there anyway.

| model | W | U | B | R | G | colourless |
|---|---|---|---|---|---|---|
| gen-1 | +2.7 | −2.4 | +3.6 | +4.6 | −0.8 | −7.8 |
| gen-3 | +8.0 | −4.2 | +3.6 | −0.7 | +2.1 | −8.8 |
| gen-4 | +9.0 | −2.3 | +2.7 | −5.1 | +4.9 | −9.3 |
| gen-4 sibling | +12.5 | −3.8 | +0.8 | −3.5 | +3.6 | −9.6 |

Share of P1P1 argmax picks of each colour minus that colour's share of the cards
available, in percentage points, over 4,000 opening boosters. Supply is flat at
about 17 % per colour.

Gen-1 runs the other way on the two colours that separate the generations: red is
its strongest colour and green its weakest, where gen-4 has them reversed. The
preference the earlier study measured at off-lane picks is therefore a standing
prior the agent brings to an empty board, and it is the reinforcement learning's
rather than Forge's.

Reading the same states as logits instead of argmax picks says the same thing
about how far the two are apart. Gen-1's five colours span 0.28 logit units,
which is as close to indifferent as this measurement resolves. Gen-4's span 2.5,
from white at +0.89 down to red at −1.61.

One lean is inherited. Every generation discounts colourless cards by seven to
ten points, gen-1 included, so that one is Forge's.

## The policy is the same on states it would never have reached

Two confounds ruin any comparison of drafting agents read off their own corpora.
An agent with unusual colour preferences is passed different packs, so its picks
differ without its judgement differing. And the states it reaches are produced by
its own earlier picks. Running each policy on the states another policy actually
faced removes both. `d3_exchange.py` does it over 71,520 states.

Neither confound turns out to matter. How far apart two policies sit is
essentially the same whoever's states they are judged on.

| policy pair | on gen-4's states | on gen-1's states | on `forge-full`'s states |
|---|---|---|---|
| gen-1 vs gen-4 | 0.693 | 0.695 | 0.695 |
| gen-3 vs gen-4 | 0.812 | 0.815 | 0.817 |
| gen-1 vs gen-3 | 0.822 | 0.822 | 0.824 |

Argmax agreement. Each policy reproduces its own recorded picks on 0.9998 of its
own states, which is the check that the replay is exact.

Whatever gen-4 learned, it applies on states it would never have reached, so its
behaviour is a policy and not a memorised trajectory. Ruling that out leaves the
question that matters: on those foreign states, is the card gen-4 would take
better than the card that was taken?

## The two graders disagree about whether gen-4 picks better

Judged by the label its own reward was built on, gen-4 picks better than gen-1 on
identical states. Judged by human pick order, it picks worse. Both orderings are
monotone in reinforcement-learning exposure, and they point opposite ways.

| states | grader | gen-1 | gen-3 | gen-4 |
|---|---|---|---|---|
| `forge-full` | `shrunk_score_play` | +0.0009 | +0.0049 | +0.0106 |
| gen-1 | `shrunk_score_play` | 0 | +0.0042 | +0.0099 |
| gen-4 | `shrunk_score_play` | −0.0091 | −0.0055 | 0 |
| `forge-full` | human draft rank | −0.0074 | −0.0194 | −0.0280 |
| gen-1 | human draft rank | 0 | −0.0148 | −0.0247 |
| gen-4 | human draft rank | +0.0286 | +0.0132 | 0 |

Mean grade of the policy's card minus the grade of the card actually taken,
paired per state, standard errors clustered on the draft; every entry is at least
four standard errors from zero. Zeros are structural, where the policy is the one
that made the recorded pick.

`shrunk_score_play` is not an independent grader. It is the encoder's own
training label, the scorer's card values track it at Spearman 0.68, and the
reward gen-4 was trained on is built on top of both. "Gen-4 takes cards with
higher `shrunk_score_play`" restates that the training worked. The human draft
rank is exogenous to all three models, and by that grader gen-1's pick beats
gen-4's on every state set including gen-4's own.

Part of gen-1's advantage on the human grader is that gen-1 imitates the ranker
that grader came from: it agrees with Forge's own choice on 0.856 of
`forge-full`'s states against gen-4's 0.659. That does not account for all of it,
because gen-3 sits between the two on both graders while agreeing with Forge less
than gen-1 and more than gen-4.

The honest summary is that gen-4 picks differently rather than better, in a
direction one grader rewards and the other penalises. Which grader is right is
settled elsewhere and in gen-4's favour: its decks win about three matches in
four when played out under Forge's AI
([`2026-08-09-draft-agent-gen4-online-grpo.md`](2026-08-09-draft-agent-gen4-online-grpo.md),
*`deck_score` does predict winning*). What the human grader adds is that the
strength is specific to this opponent. A drafter that has moved 0.135 of Spearman
away from human pick order has not learned Magic; it has learned the game Forge's
AI plays.

## Policy movement concentrates where the pack still offers a choice

Gen-4 diverges from gen-1 most at the start of each pack and hardly at all at the
end. The objective gave no reason to expect that shape: one terminal advantage is
shared equally by all 45 of a seat's picks, and every pick carries the same weight
in the batch mean. Equal gradient did not produce equal behavioural change.

| pick in pack | cards left | `KL(gen-1 ‖ gen-4)` | disagreement | disagreement over chance |
|---|---|---|---|---|
| 1 | 14.9 | 4.22 | 0.417 | 0.447 |
| 2 | 13.9 | 4.63 | 0.445 | 0.479 |
| 4 | 11.9 | 4.62 | 0.434 | 0.474 |
| 8 | 7.9 | 3.25 | 0.342 | 0.391 |
| 11 | 4.9 | 1.89 | 0.254 | 0.319 |
| 14 | 2.0 | 0.38 | 0.101 | 0.206 |
| 15 | 1.0 | 0.02 | 0.003 | 0.075 |

Measured on gen-4's own states, pooled over the three packs. A pack that shrinks
raises agreement on its own, so the last column divides the disagreement rate by
the rate a random choice would produce.

The fall survives that control. Divergence between gen-1 and gen-4 runs at 0.44
of the random baseline over the first picks of a pack and 0.30 over the last,
correlating with pick index at −0.97. Pack number barely matters, so what governs
the size of the change is how many cards are still in the pack rather than how
far into the draft the seat is.

## Colour commitment hardens across the draft, and Forge already did that

A pool pulls the policy toward its own colours, and the pull nearly triples
between the start of the draft and its end. Two runs of the same generation
differ by more than the generations differ from each other, so the hardening is
Forge's behaviour rather than something reinforcement learning added.

`d4_commitment.py` measures the pull causally. A receiver state supplies the pack
and the clock; a donor seat from a different draft at the same pack and pick
supplies the pool. The same physical card is then scored against eight different
pools with everything else held fixed, and demeaning within each card gives a
slope that card identity, pack composition and clock all cancel out of. 14,000
receiver-donor pairs, seven clocks.

| pick in draft | gen-1 | gen-3 | gen-4 | gen-4 sibling | pool's share of the tokens |
|---|---|---|---|---|---|
| 3 | +7.0 | +6.9 | +5.8 | +6.6 | 0.053 |
| 8 | +12.8 | +13.0 | +11.7 | +14.2 | 0.102 |
| 16 | +12.5 | +12.1 | +10.4 | +14.9 | 0.158 |
| 23 | +15.1 | +15.9 | +15.0 | +18.6 | 0.147 |
| 31 | +14.7 | +14.3 | +13.1 | +18.0 | 0.171 |
| 38 | +16.4 | +17.9 | +16.9 | +20.3 | 0.161 |
| 43 | +11.2 | +15.2 | +16.2 | +17.1 | 0.177 |

Logit pull per unit of the pool's share in the card's own colour.

The hardening is probably arithmetic rather than policy. The pool's share of all
the tokens the trunk sees triples over the same span, from about a twentieth to
about a sixth, which is most of the way to the observed rise on its own. Nothing
here needs a rule that says a late colour change costs more than an early one.

The two-colour discipline visible in gen-4's finished decks is therefore
inherited. What reinforcement learning changed is which colours the agent
commits to, not how hard it commits to them.

One caveat on the design. The natural placebo, the pull from colours the card
does not have, comes out strongly negative at every clock, and that is not an
independent null: colour shares within a pool are compositional, so more of one
colour is mechanically less of the others. It confirms the sign and nothing more.
The comparisons that carry weight here are across clocks and across generations,
both of which hold the estimator fixed.

## The learning went where the reward could see it

The gen-1 to gen-4 change is nearly twice as large on cards in the seat's own
colours as on cards outside them, 4.28 against 2.39. A card that misses the built
23 contributes nothing to the score, so the gradient had nothing to say about it.
The reward's construction is what shows here: it is the scorer applied to a
*built* deck, so the builder's choice of 23 spells decides which cards the
training could ever have spoken about.

`d6_buildfilter.py` runs three checkpoints on the same 36,036 states,
centres every logit within its state, and reads the gen-4 minus gen-1 difference
against a noise floor: the difference between gen-4 and its sibling, two runs from
the same base with the same settings.

| card's rank in its pack | n | \|gen-4 − gen-1\| | \|gen-4 − sibling\| | ratio | gen-4 − gen-1 |
|---|---|---|---|---|---|
| bottom fifth | 64,173 | 2.657 | 1.484 | 1.79 | +1.014 |
| second fifth | 44,679 | 2.603 | 1.593 | 1.63 | +0.056 |
| middle fifth | 47,230 | 3.027 | 1.808 | 1.67 | −0.233 |
| fourth fifth | 49,575 | 3.583 | 1.950 | 1.84 | −0.405 |
| top fifth | 64,173 | 4.365 | 2.029 | 2.15 | −0.618 |

Rank is by Forge's human pick order among the cards in that pack.

The size of the change rises with the card's quality, and is largest on the top
fifth of a pack. Two sibling gen-4 runs differ by less than half as much
everywhere, so the change is training rather than run-to-run variation. Against
that floor the top fifth still stands out, though the ratio does not fall
monotonically down the pack.

The signed column is the pick-order divergence again, read one level down. Gen-4
raises the cards humans rank lowest in a pack and lowers the ones they rank
highest, monotonically. Grading the same states by `shrunk_score_play` instead
gives the mirror image, from −0.636 on the bottom fifth to +1.769 on the top.

Against the sibling floor the change is close to uniform along a pack, running
between 1.6 and 2.4 times the floor over the first twelve picks of every pack.
That is consistent with the falling argmax divergence of the previous section
rather than in tension with it: the weights moved by about the same amount
everywhere, and the same movement changes fewer picks once a pack is down to two
or three cards and the choice is nearly forced.

## The trunk tracks the seat's colours and its final score, and gen-1 tracks them better

The model computes a summary of the draft that the deployed policy never reads.
The trunk puts a `CONTEXT` token in front of the cards; the policy head reads only
the `PACK` positions, and gen-3 and gen-4 carry their critic head untrained. A
ridge probe on that token, fitted on a draft-disjoint split, says what the trunk
has worked out.

| picks | eventual colours, AUC | final score, R² | gen-1, final score, R² | pool's colours, R² |
|---|---|---|---|---|
| 1–5 | 0.823 | 0.103 | 0.095 | 0.533 |
| 6–10 | 0.914 | 0.259 | 0.210 | 0.536 |
| 11–15 | 0.917 | 0.356 | 0.295 | 0.519 |
| 16–22 | 0.929 | 0.351 | 0.373 | 0.488 |
| 23–30 | 0.941 | 0.449 | 0.517 | 0.548 |
| 31–38 | 0.930 | 0.539 | 0.577 | 0.515 |
| 39–45 | 0.930 | 0.618 | 0.661 | 0.502 |

Gen-4 unless the column says otherwise, 13,341 validation states; the score is the
seat's final pod-relative `deck_score`. The pool's own colour fractions are the
control, since the model can read those straight off its `POOL` tokens.

Commitment is settled early and represented well. By pick 10 the token identifies
the two colours the seat will finish in at an AUC above 0.91, and the number
barely moves for the remaining 35 picks. Whatever decides a seat's colours has
happened inside the first booster.

Reinforcement learning did not improve either probe, and gen-1 reads the final
score better than gen-4 over the whole second half of the draft. The colour
probes sit within 0.01 of each other throughout. The asymmetry has a cause:
gen-1's critic head was trained, by regression on exactly this pod-relative
reward, while gen-3 and gen-4 carry that head frozen and untrained. The value
representation in gen-4's trunk is an inheritance from the imitation phase that
the reinforcement learning let decay.

## Gen-4 starves the seat it passes to

The corpus records who sat where and which way the packs moved; the state the
model consumes records neither. That gap makes the seating a clean test bed, and
it shows an effect gen-1 does not produce. `d5_corpus.py` measures it over all
four yardstick corpora, 16,000 seats in 2,000 drafts, with standard errors
clustered on the draft.

A gen-4 seat scores 0.68 above its pod mean, and each additional gen-4 seat in
the pod costs every seat 0.176, reproducing the crowding effect the gen-4 record
established. Net of that, the seat that feeds you is what matters.

| what changes | effect on the seat's pod-relative score |
|---|---|
| upstream neighbour is gen-4 rather than `forge-full` | −0.185 ± 0.022 |
| upstream neighbour is gen-1 rather than `forge-full` | −0.006 ± 0.023 |
| downstream neighbour is gen-4 rather than `forge-full` | +0.039 ± 0.022 |
| downstream neighbour is gen-1 rather than `forge-full` | −0.008 ± 0.024 |

The seat a drafter passes to does not matter; the seat that passes to it does.
Sitting downstream of a gen-4 seat costs about as much as adding a whole extra
gen-4 seat to the pod, and the same gap appears at every pod composition from
three gen-4 seats to seven, so it is not the crowding term in disguise.

Sitting downstream of gen-1 costs nothing measurable. Gen-1 drafts well and takes
good cards, so a raw strength story does not explain the asymmetry. What
separates the two is that gen-4 shares its taste with the seat behind it: both
are scored by the same reward and both want the same cards, so a card gen-4 takes
is disproportionately the card its podmate wanted. Gen-1's different pick order
removes different cards.

The agent cannot see seat index or pass direction, so it did not learn to starve
anyone. The starvation is a consequence of what it takes. It does mean a share of
gen-4's measured margin is a property of the seating rather than of the policy,
and margins should be quoted against a stated field.

## It takes fewer build-around traps, and pays less for the ones it takes

Forge's card scripts carry a hand-written `RemRandomDecks` flag marking cards its
own deck builder refuses to play when the partners they need never arrived. Its
drafter never reads the flag and takes such cards on raw power, which is where a
tenth of Forge's decks get their land padding. A card that never makes the built
23 contributes nothing to gen-4's reward, so the training had a reason to push
against them, and it partly did.

| agent | flagged card available | took it | played it into the 40 |
|---|---|---|---|
| gen-4 | 116,757 picks | 14.35 % | 27.0 % |
| gen-1 | 58,517 | 17.52 % | 30.0 % |
| `forge-full` | 56,296 | 18.77 % | 31.7 % |

Distillation alone removed 1.3 points of the take rate and reinforcement learning
removed a further 3.2, so two thirds of the drop is the RL's. Restricting to
picks where an unflagged card was also available changes nothing.

Gen-4 has not learned to read the flag. It still takes a flagged card on one such
pick in seven, and still plays only a quarter of what it takes. What changed more
than the count is the cost: the pod-relative score falls by 0.011 per flagged card
gen-4 drafts, against 0.045 for gen-1 and `forge-full`. Gen-4 takes flagged cards
it can use.

## The training step was not driven by the training signal

Every round of every gen-4 run was gradient-clipped, so the optimiser took a
fixed-length step whatever the round contained. Pre-clip gradient norms average
between 7.5 and 8.4 against a clip of 1.0, in 100 % of the 2,093 rounds across
the four runs.

How far the policy moved in a round is close to independent of how much the round
had to teach. Correlating each round's `KL(π_k‖π_{k+1})` against its signal, with
each series standardised within its run: reward standard deviation gives +0.211 ±
0.021, the near-zero advantage fraction −0.051 ± 0.022, and the fraction of picks
with a large advantage +0.031. The one variable that predicts the step well is
the pre-clip gradient norm itself, which is geometry. The advantage standard
deviation is 1.000 by construction, because advantages are standardised within
the round, so that axis of the log carries no information at all.

The policy walks away from its warm start at a near-constant rate. Distance from
`π_0` grows almost linearly in rounds, correlating with the round number at
+0.985 in the promoted run. Every run peaks between 0.3 and 0.7 nats from its
start and then declines while continuing to walk: the promoted run kept going for
956 rounds after its best margin, finishing at 2.37 nats with the margin down by
almost half.

That is the mechanical account of why all four runs peak and decline. The step
length is set by the clip rather than by the signal, so the run is close to a
fixed-step walk, and a fixed-step walk crosses a ridge and keeps going.

## Part of the move away from human pick order is drift, not skill

A habit that grows across the generations might be what made the agent stronger,
or might be what a fixed-step walk away from the warm start produces on its own.
The two readings have the same shape. Separating them needs two checkpoints that
differ in training length and not in strength, and one pair qualifies.
`t2all_decay0.3` trained on about two and a half times the learner picks of
`t2all_nodecay`, and the two finish within each other's yardstick error bars. A
habit that moves between them is following training length rather than strength.
`d9_signatures.py` computes the same measurements on six checkpoints.

| checkpoint | margin over gen-1 | learner picks | ranking spread | colour lean | vs human order | vs `v_swap` | `POOL` blanked | `TAKEN` blanked |
|---|---|---|---|---|---|---|---|---|
| gen-1 | 0.000 | 0 | 2.50 | +0.11 | 0.793 | 0.372 | 0.458 | 0.117 |
| gen-3 | 0.824 | 110k | 3.45 | +1.29 | 0.771 | 0.597 | 0.467 | 0.112 |
| `t3all_decay0.3` | 1.152 | 185k | 4.63 | +1.91 | 0.704 | 0.684 | 0.458 | 0.109 |
| `t3learner_t2field` | 1.276 | 200k | 4.84 | +2.51 | 0.687 | 0.685 | 0.443 | 0.102 |
| `t2all_nodecay` | 1.328 | 205k | 4.42 | +2.18 | 0.690 | 0.736 | 0.447 | 0.113 |
| `t2all_decay0.3` | 1.380 | 515k | 5.06 | +2.31 | 0.658 | 0.697 | 0.432 | 0.118 |

Ranking spread is the standard deviation of the context-free pack-1-pick-1 card
scalar. Colour lean is the mean of that scalar over white and green cards minus
its mean over blue and red. The two rank correlations are against Forge's human
pick order and the scorer's card values, over the 3,773 cards seen at least five
times in the 4,000 opening boosters. The last two columns are the argmax change
when a block's card identities are blanked, measured on 22,101 states.

The correlations settle nothing on their own. Yardstick margin and training
length correlate at 0.952 over these six checkpoints, and the four gen-4 siblings
rank identically on both, so every quality measurement correlates about equally
well with either axis. Only the last pair of rows identifies anything.

Two habits keep moving after the strength stops. Between `t2all_nodecay` and
`t2all_decay0.3` the ranking spread gains a quarter of its whole gen-1-to-gen-4
range, and agreement with the human pick order gives up a quarter of its whole
loss, both for no measured strength at all. The sharpening of the pick order and
the drift away from human taste are therefore partly what the extra training did
rather than what made the agent better.

One habit is finished before that point. The colour lean moves by a twentieth of
its range across the same pair, against more than two units of movement over the
lineage, so it saturates somewhere near 200,000 learner picks. Gen-4's own record
found the same ceiling from the other direction, measuring the lean at off-lane
picks across four candidates and finding it stop near four percentage points
([`2026-08-09-draft-agent-gen4-online-grpo.md`](2026-08-09-draft-agent-gen4-online-grpo.md),
*Hypothesis 3*).

Agreement with the reward's own card values peaks before the training ends. It
rises through the lineage as far as `t2all_nodecay` and falls back at
`t2all_decay0.3`. The extra 310,000 picks moved the policy away from the card
ranking it was climbing without moving it up the yardstick, which is what the
training logs predict: the step length is set by the gradient clip rather than by
the signal, so a run keeps walking after it has stopped improving.

How the policy reads its inputs follows neither axis. Blanking the `POOL` card
identities changes between 43 % and 47 % of picks in every checkpoint, and
blanking `TAKEN` between 10 % and 12 %, with no trend against either strength or
training length. The information a policy uses is fixed by the architecture and
by the imitation phase; reinforcement learning changed what it does with that
information and not what it reads.

One measurement resists the partition. The share of picks the context-free
ranking already decides, which is not in the table above, differs between the two
siblings by less than it differs between two ways of measuring it, so the pair
contrast leaves it unresolved.

## The scorecard: three falsifications, and most of the skill is a list

Sixteen hypotheses survived the critique phase, and most of the strategies they
proposed turn out to belong to Forge or to the reward rather than to the draft
model. Three are falsified outright, one of them in the opposite direction from
the one proposed, and four are what reinforcement learning actually taught.

| hypothesis | verdict | evidence |
|---|---|---|
| H1 the policy reads its own pool and little else | verified, stronger than proposed: `TAKEN` sits below its own placebo in every pack | `d1_channels.py` |
| H2 the policy is largely a fixed pick order | verified: 49 % of picks, up from gen-1's 43 % against a chance floor of 22 % | `d2_pickorder.py` |
| H3 skill or trajectory | resolved as skill: policy distance is unchanged by whose states it is measured on | `d3_exchange.py` |
| H4 it drafts for the built 23 | verified: the gen-4 − gen-1 residual grows with card quality and is twice as large on-lane | `d6_buildfilter.py` |
| H5 colour commitment is a clock-dependent rule | refuted as a policy: it hardens, gen-1 hardens as much, and the pool's growing token share explains most of it | `d4_commitment.py` |
| H6 the colour lean is an unconditional prior | verified, and it is the RL's: gen-1's lean runs the other way | `d2_pickorder.py` |
| H7 it moved away from human pick order | verified: −0.135 Spearman, largest loss on removal | `d2_pickorder.py`, `d3_exchange.py` |
| H8 it reads the table | **falsified**: erasing what others took moves it less than erasing noise | `d1_channels.py` |
| H9 equal gradient produced equal change across picks | **falsified**: argmax divergence falls steeply across a pack, and still falls after the shrinking-pack control | `d3_exchange.py` |
| H10 skill signatures versus drift signatures | verified, and it revises H7: the ranking's sharpening and a quarter of the move away from human order keep going after the strength stops, while the colour lean saturates | `d9_signatures.py` |
| H11 denial calibrates the sensitivity floor | verified by arithmetic: the incentive is 0.143, the estimator resolves 0.014, a hate-draft is worth 0.004 | code |
| H12 the wheel and adverse selection | resolved as near-absent: zeroing the recency tags on the pack changes 2.6 % of picks | `d1_channels.py` |
| H13 duplicate indifference | **falsified**, and in the opposite direction: a copy in the pool raises the same card's logit, most in gen-1 | `d8_duplicates.py` |
| H14 what the `CONTEXT` token carries | verified with a reversal: it carries the seat's colours by pick 10 and its final score, and gen-1 carries the score better | `d7_contextprobe.py` |
| H15 the geometry it cannot see | verified: the upstream neighbour's label matters, the downstream neighbour's does not | `d5_corpus.py` |
| H16 build-around traps | verified, modestly: 3.2 points of take rate below gen-1, and the ones it takes cost it a quarter as much | `d5_corpus.py` |

What makes gen-4 a good drafter, then, is a shorter list than the hypotheses
allowed for. It has a sharper card ranking, tuned to its reward rather than to
human taste. It has a standing colour preference matched to which colours win
under Forge's piloting. It spends its learned capacity on the cards that can
reach a deck. And it knows which cards are already its own, which is enough to
produce two-colour discipline without any rule about when to commit.

What it does not have is the half of drafting that is about the other seven
players. It cannot read a signal and it does not hate-draft. The one pod-level
effect it produces, starving the seat it feeds, is a side effect of its taste
rather than anything it chose.

Two of its habits are Forge's and survive because nothing in the reward
contradicts them. It prefers a card it already owns to an equally good one it does
not, and it commits harder to its colours as the draft runs on. A reward computed
on a finished deck is silent about both, so imitation set them and reinforcement
learning only let them decay.

## Limitations

Every number is a property of these checkpoints under Forge-AI play, and the
human pick order is the only grader in the study that is exogenous to the stack.
Where the two graders disagree, this document reports the disagreement rather
than picking a winner; the case that gen-4's direction is the winning one rests
on played games measured elsewhere.

The interventions are causal on the model and not on the draft. Blanking a block
or transplanting a pool measures how much the policy's output depends on an
input, which is not the same as measuring what would happen to a draft if the
agent behaved differently. No probe here re-ran a draft with a forced pick,
because the pick side-channel has no override.

Three comparisons rest on states the model never trained on. A pool transplanted
from another draft, a blanked block, and a rewritten clock all produce states
outside the training distribution, and the placebo ladder bounds the size of that
problem without removing it.

The corpus samples a random set per draft, so per-set behaviour is not measured
here at all, and the four candidates' yardstick corpora differ in composition.
Everything comparing agents is paired inside a pod or inside a state for that
reason.

The skill-versus-drift partition rests on a single pair of checkpoints. Yardstick
margin and training length correlate at 0.952 across the six measured here, so
`t2all_nodecay` against `t2all_decay0.3` is the only contrast that separates them,
and its strength gap is not zero but merely inside the error bars. A second pair
matched the same way would turn that partition from a suggestion into a
measurement.
