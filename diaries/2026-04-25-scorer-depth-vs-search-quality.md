# April 25, 2026 — Scorer depth and search quality

**TL;DR:** Discovered that training ran on CPU the whole time (GPU fix
was the first thing done), then spent the day studying how scorer
depth affects actual win rate vs. validation accuracy, and added
simulated annealing to the deck builder.

The day opened with a basic oversight: `train_scorer.py` had never
moved any tensors to the GPU, so every training run had been pure CPU.
Claude spotted the missing `.to(device)` calls immediately and the
fix was straightforward.

With GPU training working, I started seriously comparing architectures.
Val accuracy looked flat across 2-, 3-, and 4-layer models (all ~72%),
which made depth look pointless. The actual eval against Forge told a
completely different story: the 2-layer model lost to Forge by 20
percentage points, while the 4-layer beat it by about 7 points. Val
accuracy was simply the wrong metric for judging which model to deploy;
it scores random pairs of finished decks, not the quality of
card-by-card greedy decisions. I decided to keep pushing deeper.

The 4-layer model had a known instability: the default learning rate
of 1e-3 caused it to collapse (all-zero gradients, constant output).
I remembered having fixed this before, and it turned out the fix was
a *lower* learning rate, not higher — 1e-4 let the tiny early gradients
accumulate coherently rather than noisily bouncing around the flat
initialization region. At that rate, 4 layers ran well but overfit
aggressively after about epoch 5.

I kept going to 5 and 6 layers. Five came back at roughly the same
eval win rate as 4. Six layers jumped to a mean delta of +18.6% against
Forge across 12 pools, with 11 of 12 pools favoring the scorer. Claude
flagged this as potentially lucky (different eval runs pick different
random sets), and the next eval run confirmed it: all three models came
back at around -12%. The evaluation harness is just too noisy on 12
pools because a single unlucky set draw dominates all the pool-level
results. I decided to stop trying to rank architectures by this eval and
instead just proceed with gen-2 self-play using the 6-layer model —
the real quality signal will accumulate from thousands of match-outcomes
entries over many sets, which averages out pool luck properly.

I had also asked Claude to add a match-length breakdown to the
`analyze_winrates` script: 4-0 and 4-1 matches together account for
about 57% of the Bo7 data, with the remaining 43% being 4-2 and 4-3
games that could plausibly have gone either way. The 72% val-accuracy
ceiling is consistent with a dataset where a significant fraction of
matches are genuinely too close to call.

The gen-1 match-outcomes data (1,406 matches by that point) gave the
first honest cross-method comparison: gen-1 decks win 46% of matches
against forge-best, beating forge-3sub and forge-8sub more
comfortably. The gen-1 mirror win rate sat at 50.6% over 350 matches,
which is just sampling noise around the expected 50% — a clean sanity
check on the methodology. I noted that the 56.2% aggregate gen-1 win
rate is misleading because it includes easier opponents; the honest
number for how gen-1 compares to the best Forge strategy is that 46%.

While watching the greedy deck builder output, I noticed the decks were
consistently using many colors. Claude explained why: the greedy starts
from a random 23-card subset (almost certainly spanning all five colors)
and can only make single-card swaps, which means it can never make
the coordinated multi-card move needed to drop a color. We discussed
alternatives — genetic algorithms, beam-constructive search,
color-pair seeded starts — then I asked to just try simulated annealing
as the easiest option. Claude wired in temperature, cooling, and
max-iterations parameters, with best-deck tracking to guard against
wandering.

I tested several temperature values on a fixed set of 12 pools.
Temperature 0.8 produced the best mean score across pools (+10.8% over
pure greedy). Temperature 1.0 was sometimes better on individual pools
but catastrophically worse on at least one pool, pulling the mean down.
I decided 0.8 was the right default for gen-2. Claude pointed out that
higher scorer score is not the same as higher win rate against Forge —
the SA might be finding the most miscalibrated deck, not the best one.
I acknowledged this but framed the goal clearly: I want to extract the
scorer's own best answer as a seed for gen-2 training, and the actual
quality of gen-2 will be validated by its own match outcomes.

I also had `--print-decks` added to `build-decks`, with cards sorted
by mana value so the curve is readable at a glance.

At the end I asked Claude to write an experiment file summarizing the
SA tuning results, then reference it from the sealed-deck-picker spec.
Claude's first draft used "now supports" language and included
performance numbers in the spec document; I pushed back on both and
Claude updated the memory to carry those rules forward.
