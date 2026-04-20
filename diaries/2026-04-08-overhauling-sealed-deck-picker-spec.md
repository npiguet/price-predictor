# April 8, 2026 — Overhauling sealed deck-picker spec

**TL;DR:** I brought in a new architecture from a separate conversation and
rewrote the sealed deck-picker spec around it. The core shift: ditch absolute
win-rate regression in favour of pairwise Bradley-Terry scoring with greedy
hill-climbing search.

The day started with a transcript from a separate brainstorming session I had
done elsewhere. The direction that emerged there was compelling enough that I
wanted it locked into the spec rather than left in chat history. The old spec
had leaned toward an eight-layer transformer predicting absolute win rates;
the new one collapses that into two practical phases.

Phase 1 is a Set Transformer trained on raw pairwise game outcomes. Instead
of needing fifty-plus games per deck to get a stable win-rate estimate, every
single game produces one training example directly. The Bradley-Terry
formulation — sigmoid(score_A − score_B) with binary cross-entropy — makes
the math clean and means the scores don't have to be calibrated probabilities,
only ordinal. At 3,000 games per hour, the data generation budget becomes
tractable.

Phase 2 turned out to need no ML at all: greedy hill-climbing from the
heuristic seed deck, scoring every possible single-card swap (~1,500 per
iteration) and keeping the best. The scorer runs in microseconds per forward
pass, so many iterations fit in well under a second. The insight from the
brainstorm that I found most useful was that lands should be excluded from the
model entirely and assigned deterministically from mana-pip counts. My earlier
RL attempt had collapsed to "two lands of each color" precisely because the
policy had to learn that mapping from sparse win/loss signal — a structurally
hard credit-assignment problem that disappears once you remove lands from the
decision space.

A Phase 3 (one-shot distillation of scorer + search into a single forward
pass) was noted as optional and not planned for now.

The spec also needed a Phase 0 to describe training data generation, since
that process — generate pools, build variant decks via four weighted
strategies, play cross-pool and within-pool matchups, record
(deck_A, deck_B, winner) — was buried inside Phase 1. Extracting it made the
dependency chain legible. While doing that, Claude flagged that Method 3 for
deck generation was removing eight cards but only adding four back, leaving
undersized 19-card decks; I confirmed that was a mistake and corrected it
to eight-for-eight.

Other spec cleanups that came out of the review: the generate-pools tooling is
no longer relevant since Phase 0 generates pools on-the-fly in Java, so that
section was removed. The encode-cards step, still needed as a prerequisite
before the scorer can run, was moved to its own section between Phase 0 and
Phase 1.

The embedding training schedule is frozen-then-unfreeze, matching standard
practice when fine-tuning a pre-trained model: let the scorer head stabilize
first on the existing embedding geometry, then unfreeze at 10–100x lower
learning rate once validation loss plateaus.
