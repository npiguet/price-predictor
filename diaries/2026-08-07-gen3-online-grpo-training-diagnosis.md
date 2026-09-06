# August 7, 2026 — Gen-3 online GRPO training diagnosis

**TL;DR:** Finished implementing the gen-3 online draft-trainer, then spent the
day running it, hunting down a stable learning rate, and correcting Claude on
two design assumptions that turned out to be wrong. By evening the gen-3
agent finally beat both gen-1 and Forge's own drafting, and a deep dive into
*why* one training regime produced better decks than another turned up a real
mechanism: the agent was losing colour discipline, not gaining raw card
judgment, and I traced it to how the shared per-draft reward assigns credit.

The day opened with the last two phases of spec 021 (best-checkpoint
tracking, opt-in patience, LR annealing, a cumulative-KL diagnostic, then
per-category pick modes) going in against tasks already broken down from the
day before. Once that was live I moved to actually running the trainer, and
the first pass was mostly a learning-rate hunt: 1e-4 climbed nicely for about
fifteen rounds and then blew up once the KL between consecutive policies hit
the 1-3 range; 1e-6 turned out to do essentially nothing, which I kept
running anyway and it became a useful control — a policy that provably never
moved gave a noise floor for the anchor margin (mean +0.064, std ~0.06) that
made the earlier +0.44 peak look real rather than lucky. 1e-5 was the one
that worked cleanly: margin climbed to +0.75-0.84 with no collapse and the
frozen anchor staying flat the whole run, which is what let me trust the
number in the first place.

Partway through I asked whether the sealed-trained one-shot picker could
replace the slow SA deck builder to speed up rounds. Claude confirmed it
transfers mechanically to 45-card pools, but I decided not to switch — the
picker's earlier scorer-score gains over the SA builder never cashed out in
actual win rate on sealed pools, and I wasn't convinced retraining a
draft-specific picker was worth the effort right now. Greedy stayed the
default.

Two corrections I pushed on ended up reshaping the trainer's design. First,
Claude had the frozen anchor sampling at the same temperature as the learner
"for a like-for-like comparison." I didn't see the point — the models that
aren't learning should be playing their best, otherwise the learner is
training against artificially weak opponents. Claude initially framed this
as only shifting the margin's zero point; I pointed out that's wrong because
drafting is a fixed pool of cards, so a bad pick by an opponent doesn't just
lower their own score, it hands the card to me. That reframed the whole
thing as a genuine training distortion, not just a measurement offset, and
led to Phase 8: only the learner samples, everyone frozen plays argmax.

Second, once decks started coming back from the yardstick, Claude described
gen-3's card-transfer effect as "amplification" that discounted the raw
margin. I disagreed — denying an opponent a card is a real draft strategy,
not a measurement artifact, and the reward already pays for it directly by
construction. Claude agreed and rewrote that section across six files as
"field-relative" rather than "amplified": the margin isn't inflated, it's
just measured against a specific mix of opponents.

By the end of the first session gen-3 was clearly ahead: +0.56 to +0.73 deck
score over gen-1 on the argmax yardstick, and gen-1 itself edging out
Forge's own drafting by a small amount — the first generation in this
lineage to actually move the needle. Also used /doctor to trim the
project's CLAUDE.md from 49k to under 11k characters and split package-level
guidance into per-package files, and recorded a standing note not to prefix
shell commands with cd since the working directory already persists.

The second session was the real investigative piece: writing up the
experiment record for six training runs, split between "field at T"
(temperature applied to every ML-piloted seat) and "field at argmax" (only
the learner samples, per the Phase-8 change). Comparing the two, I noticed
the field-at-argmax candidates had a much bigger gap between mean and median
deck score than the field-at-T ones, and asked why. That question drove
several rounds of hypothesis-and-test: lane starvation (forced off-colour
picks late in a pack) was proposed and then falsified by pack-by-pack data —
the excess off-colour picks were early and voluntary, not late and forced.
A second hypothesis, that the policy was trading colour discipline for
raw card power, was also falsified: measured on two independent quality
scales (a pick-rate proxy, then real win-rate labels from the corpus), the
field-at-argmax T=3 policy's off-colour picks carried no quality premium at
all — essentially a coin flip against the on-colour alternative it
passed on. What survived was narrower: that policy had the best raw card
evaluation of any candidate but had lost colour conditioning specifically.

I then asked whether this was really about training length or whether lane
starvation made sense as a training-time cause even though it had just been
ruled out as the deployed behavior. Claude had tested starvation against the
wrong corpus — the yardstick uses an identical field for every candidate, so
it can't show a training-time difference. Reasoned through it, starvation
does make sense during training: under field-at-argmax the opponents draft
correctly and consistently, so the learner visits genuinely lane-starved
states far more often, and because GRPO shares one reward across all 45
picks in a draft, a forced off-colour pick in an otherwise good draft gets
reinforced exactly like a chosen one. Nothing in the gradient can tell the
two apart. That reframed hypothesis 1 as falsified-as-behavior but revived
as a training-time cause, and settled on field-at-T as the training method
to carry into gen-4, with instrumentation added to gen-4's plan: give each
run its own output path so training rollouts survive for exactly this kind
of after-the-fact analysis (the ones from this campaign were already gone,
overwritten by a shared corpus file), and log the wide-mana-base rate per
round alongside the anchor margin, since the margin's own windowed-mean
selection criterion turned out to be blind to this failure mode.
