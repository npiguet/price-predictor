# May 19, 2026 — One-shot picker spec and critique

**TL;DR:** I spent the day designing and refining a spec for a
one-shot deck picker that replaces simulated annealing with a
single transformer forward pass, aiming at 100x+ inference
speedup. A peer review of the spec sharpened the training risk
story substantially before I locked it in.

The day opened with a critique session on a spec that had already
landed in the repo. Claude read the spec end-to-end and came back
with a ranked list of concerns. The architectural choices — SAB
trunk over the pool, per-card shared linear head, no positional
encoding — all got clear passes. The Plackett-Luce log-probability
formulation for REINFORCE was called correct. What landed hard was
the training method: vanilla REINFORCE from random init on a
combinatorial action space of C(70,23) ≈ 10²⁰. Claude framed the
"baseline, multi-sample, entropy" mitigations in the spec as just
the standard REINFORCE setup, not actual risk reduction, which was
fair.

Before reacting to those critiques, I asked Claude to evaluate
them properly and flag which ones could be dismissed given context
the reviewer didn't have. The most important correction was on cold
start: the reviewer had claimed the gen4-512 scorer was never
trained on random decks, so gradients there might be noise. I
pushed back — random decks are ~4% of the training corpus with
clear calibrated signals (7.3% Bo7 win rate vs forge-best,
48.6% vs random). That dismantled the OOD framing. What survived
was a narrower version: only ~144 random-vs-random matches, so
the scorer's within-random-quality-band resolution might be coarse.
That's still a real question, and a pre-training probe measuring
per-pool reward std at random init is the right cheap test for it.

I also clarified that the "25k tiny GPU calls per training step"
language in the spec was misleading — the sequential sampler runs
as ~25 vectorized iterations on a `(batch_size × N_samples, N)`
tensor, not 25 per-sample Python loops. And the Gumbel top-K
alternative got cut entirely: with a variable-length walk
(spell-quota rule for nonbasic lands), the vectorization win
disappears and the sequential sampler is distributionally
equivalent at ~25 bounded iterations.

The gen4-512 vs gen4-256 match play data arrived mid-session. The
head-to-head is 55% to gen4-512 at n=151 matches, a ~2σ result on
its own, but the Bo-N growth pattern is the clean tell: gen4-512
is the only method whose win rate climbs monotonically from Bo1
(55.2%) to Bo7 (60.3%), which is the signature of per-game p > 0.5
rather than noise. I decided the "ship 256d × small" conclusion
from the scorer sweep needed to flip to 512d given the match-play
numbers. The proximate mechanism that makes most sense to me is
color discipline: 5-color drift collapses from gen3-128 at 1.4%
to gen4-512 at 0.13%, consistent with richer text representation
from the 512d encoder driving sharper color-affinity reasoning.

The inference speed problem also came up in its own conversation.
At 12s per deck on a 3060Ti, the gen4-512 + SA builder is two
orders of magnitude too slow for mobile. I worked through the
architecture options: per-card classifier (one forward, take
top-23), query-decoder with Sinkhorn/Hungarian (23 deck slots
assign themselves to cards), and autoregressive pointer (which I
had already tried without success). I decided option 1 is the
most tractable starting point. Claude walked through the training
options — supervised distillation from SA outputs, or direct
REINFORCE against the frozen scorer — and I noted that the
throughput math flips the conventional "REINFORCE is sample-
inefficient" argument: at 12s per SA-labeled pool versus ~10ms per
scorer forward, REINFORCE can reach 10,000x more samples per wall-
clock hour. Claude had initially missed this and revised the
recommendation from "supervised first, REINFORCE as fallback" to
"try REINFORCE from random init first (Option B), with supervised
warmstart (Option A) as the contingency."

The spec was written from scratch by Claude following the format of
the other date-prefixed specs in the repo. The main editing passes
after the initial draft were: adding the pick-decomposition walk
for nonbasic lands (they rank alongside spells; the spell-quota
stops the loop, not a fixed top-23 cutoff), dropping the 17-slot
cap as dead code given pools have at most ~6 nonbasic lands, making
the auxiliary head unconditional, simplifying to a single sampling
scheme, locking `--val-fraction 0.2` as the project-wide
convention, and moving all the fallback training paths (SA
warmstart, supervised distillation) into a "Contingency plans (not
part of this specification)" top-level section with explicit
framing that pivoting to them requires a new spec round.

The reviewer's response came back as a formal written critique.
After going through it, I sent a detailed reply. The pieces I
accepted: cold-start probe as a documented pre-training procedure
(not a CLI subcommand), reworked reward-hacking detection since
in-training Forge validation is infeasible in a ~2h training run.
The replacement is a three-part plan: per-epoch cross-scorer
agreement using gen3-256 as auditor of gen4-512 (if gen4 score
climbs while gen3 lags, that's gen4-specific hacking), per-epoch
distributional sanity checks on deck shape, and a single Forge
eval paid once at the end of training. The entropy schedule moved
from linear-to-zero on wall-clock to constant-until-lift-off-
confirmed, then decay tied to val-reward plateau.

The reviewer's architectural concern on compositional coordination
(dual land + splash spell requiring correlated decisions) I pushed
back on: the SAB trunk attends over the full pool, so the dual
land's logit is computed attending to the splash spell and vice
versa. The joint pattern "both worth picking" is representable as
"both get high logits." Whether REINFORCE teaches this is an
empirical question about training signal, not an architectural
capacity question. On the color-complexity-conditioned eval the
reviewer recommended, I declined it: aggregate win rate is the
deployment criterion, and if the picker beats SA in aggregate via
a different strategy (including treating splash pools differently
than SA does), that's value, not regression. Forge AI piloting
biases mean "playing better Magic" and "winning more Forge matches"
can diverge anyway.

The aux head justification in the response got rewritten around a
stronger argument than the MLM analogy the spec had used: the aux
head is structurally identical to the gen4-512 scorer (card
embeddings → transformer → scalar quality), and to predict pool-
level expected reward the aux head must internally model which
cards would be picked, which is exactly the picker's task. That's
aligned representation pressure, not loose analogy.

One sharp observation from the reviewer that I found genuinely
interesting: the aux head is structurally a critic (predicts
expected reward) but isn't used as the baseline — the empirical
per-pool mean is. Using the aux head prediction as the baseline
directly would be an actor-critic upgrade. Agreed it's worth a
follow-up spec round once the basic loop has demonstrated lift-off.
