# August 9, 2026 — Gen-3 metric diagnostics, writeup style

**TL;DR:** Chased down why gen-3's LR decay and best-checkpoint selection
were behaving oddly, decided the run's results were good enough to keep
as-is, then spent most of the day rewriting the gen-3 experiment writeup
into a much plainer style — which in the process surfaced a real, confirmed
finding: the draft agent has learned a colour preference from Forge. A
separate thread explored whether Q-learning or search-based alternatives
could replace the current RL approach.

The day opened with a training-log postmortem. The LR-decay flags on a
gen-4 draft-agent run (`lr1e-5_t2all_brokendecay`) looked ignored — the
learning rate sat flat at 3e-6 for 1239 of the run's 1269 rounds. Claude
traced it to `_PlateauLR.can_decay()`: it refuses a decay that would land
below the floor instead of clamping to it, so `lr 1e-5 / factor 0.3 /
floor 1e-6` only ever permits one decay — the advertised 1e-6 floor was
never actually reachable. The unit tests never caught it because they all
use factor 0.1 against a floor that happens to land exactly on a power of
ten. I decided to keep the run's checkpoints as they were rather than
rerun anything, since I liked the result.

I then pushed on whether the round-9 "best" that triggered the round-29
decay was contaminated by the warmup window, and it turned out to be a
sharper finding than that. The learner (`gen4`) and the frozen anchor
(`gen3a`) were the same checkpoint at the same temperature at round 0, so
the true margin there was exactly zero — and the logged value was +0.781,
a free calibration of the margin's noise floor. By that yardstick, round
9's "best" of +0.218 sat well within one sigma of zero. The decay that
fired at round 29 was chasing a mirage.

That led me to ask whether "best" should be selected by policy loss
instead of the anchor margin. The correlation came out backwards — +0.34,
when a minimization loss should correlate negatively with quality if it
means anything. With rewards standardized to mean 0 within each round, the
loss reduces to a bounded quantity that reflects how confident the policy
is, not how good it is; selecting on it would have picked a materially
worse checkpoint than the one actually kept. I raised a related worry
next: the policy is trained against the leave-one-out mean of the whole
field, not the single frozen anchor used for reporting, so selection
seemed to be optimizing the wrong thing. That held up too — anchor margin
and a field-relative margin correlate at +0.914 but pick checkpoints 340
rounds apart. I decided the anchor margin was good enough to keep, given
how strongly the two numbers agree, and had that acceptance written down
explicitly in the gen-3 writeup rather than switching metrics.

I also asked why a decay doesn't roll training back to the best checkpoint
before annealing — it seemed to defeat the purpose of annealing if you
don't anneal from the best point found. Claude's answer, that annealing
shrinks the noise ball around wherever training currently sits rather than
around a "best" that in this run was itself noise, held up against the
round-29 data: rolling back there would have thrown away twenty rounds of
real progress to return to a lucky sample.

The rest of the long session went into rewriting the gen-3 online-GRPO
experiment writeup, paragraph by paragraph, on prose style. I kept
rejecting drafts that were technically correct but hard to follow — "this
paragraph is exhausting to read," "I still have no idea what that
paragraph tries to say" — and kept asking for the point to come first, one
claim per sentence, numbers in tables instead of buried in prose. Once a
rewrite finally read the way I wanted, I asked Claude to record the rules
behind it in the feature-workflow skill so they'd apply to every future
experiments doc, then kept tuning that rule list itself over several more
rounds: dropping a rule that turned out to be about correctness rather
than style, adding a rule against sprinkling random bold text, and later
adding a "plain is not casual" guard after one rewrite came out too chatty.

While rewriting the writeup I asked whether the agent's known preference
for green, black and white — since Forge plays worse with colours that
lean on instants and sorceries — shows up specifically in its off-lane
picks. The first pass came back null, but it used a biased baseline
(weighting availability by pack size rather than per decision); once
corrected, the finding reversed hard. Every one of the four gen-3
candidates takes green/black/white above supply when it breaks lane, and
neither reference agent does. The lean scales with how long a candidate
trained, not with which field it trained against, and it does not explain
which candidate drafts the more disciplined mana base — the candidate with
the biggest colour lean also has the narrowest mana base and the best
score. When Claude's first draft of the writeup framed the lean as a flaw
("the wrong criterion"), I corrected that: against the Forge AI opponent
this stack actually targets, leaning into the colours Forge plays worse is
correct play, not a defect. What is actually wrong is narrower — one
candidate applies that colour preference even to off-lane picks, where the
pool should be deciding instead, and earns no quality premium for it while
the other three do.

In a separate thread I asked whether Q-learning could replace the current
RL approach for the draft agent. Claude judged it applicable but risky: it
would fix real problems — every past corpus becomes reusable training data
instead of being discarded every ~56 steps, and the loss is bounded so it
can't diverge the way gen-2's did — but a greedy Q-policy is exactly the
"critic-only greedy actor" the gen-2 postmortem already rejected for
hunting the critic's own overestimation, and a terminal reward 45 picks
out is close to the worst case for bootstrapping. The recommendation was
to keep online GRPO as the main line and, if a value-based branch gets
built at all, use IQL rather than vanilla DQN.

I then asked for a genuinely different family of approaches, and got four:
searching at pick time and distilling the result (AlphaZero-style rollout
policy improvement), turning the problem into supervised pairwise ranking
via counterfactual paired picks, potential-based dense reward shaping that
provably preserves the same optimal policy, and a free auxiliary "will
this card make the final deck" head available from existing corpora at
zero extra cost. Revisiting gen-1's numbers along the way: 87.7% top-1
match to Forge's actual picks and 99.3% top-3 — an excellent imitator that
is nevertheless a dead heat with Forge on actual deck quality (1.48 vs
1.52 mean deck_score) — the designed ceiling of copying a demonstrator you
can't exceed by imitation alone.

I pushed back hard on the counterfactual-paired-picks idea: holding
opponent picks fixed across two simulated draft branches isn't valid,
since my own pick changes what card is even available to pass downstream.
Claude agreed the phrasing was wrong, but rebuilt the argument on cleaner
ground: sharing only the exogenous randomness — booster contents, not
opponent behaviour — means the two simulated drafts can differ by at most
one card at any point. The perturbation either gets absorbed by a seat
that wants that one card in both branches, at which point the two
branches literally re-merge, or it keeps walking forward as a single
swapped card; it can never grow to affect two cards at once. That
containment is what makes the comparison usable despite the objection
being correct about opponent picks not being strictly fixable.
