# Saturday, May 3, 2026 — Deck analysis reveals scorer blindspot

**TL;DR:** I ran the full speckit workflow to formalize feature 016 (card
winnability pretraining), making seven architectural decisions along the way.
Then I built an analysis script for generated-deck statistics, compared
generations head-to-head, and traced a clear signal that the scorer is learning
macro deck structure from the 32 deterministic features while likely ignoring
the 512 transformer embedding dimensions entirely.

The day started with plumbing for feature 016. Claude ran `speckit.specify`
from my informal description file, then two rounds of `speckit.clarify`, then
`speckit.plan`. The clarification rounds surfaced seven real decisions across
the sessions. The most consequential ones: the pool layer's per-query output
dimension was internally inconsistent in my source doc (option A, where each
query outputs `d_token / K` dims, was the only reading that kept `d_card = 512`
independent of K), basic lands get filtered out by the Java worker at write
time rather than at the aggregation stage, and `train-encoder` hard-fails if
any card in `cards-played.txt` is missing from `output/cardsfolder/`. The
second clarification round caught a train/inference mismatch I had not noticed:
`encode-cards` strips the `name:` line before tokenizing, but the spec had
not said `train-encoder` should do the same, which would have meant the
encoder memorized card names rather than card attributes. That got locked in as
option A. The other decision that came from me rather than from the multiple
choice was to dump the full per-card label map to `output/sealed/cards-win-
rates.txt` after every `train-encoder` run, sorted by raw ratio descending —
a concrete inspection path for the shrinkage behavior described in SC-005.
Claude also found prior art in `../jumpstart-tierlist` for the Java-side card
collection approach (two-event union: `GameEventCardChangeZone → Battlefield`
plus `GameEventSpellAbilityCast`, with filters for controller==owner, no
tokens, no basics), which replaced a more abstract approach in the plan.

After all that, a separate session produced `scripts/analyze_generated_decks.py`.
The original motivation was straightforward: I wanted to read out color
preferences, mana curves, type balance, and rarity distribution from the
generated-decks files, and I asked Claude what other stats might be worth
extracting. Claude suggested the per-color pip distribution and the per-label
breakdown. I pushed back on a per-set breakdown that was too granular, and we
dropped it. Once the script ran on gen1 (5013 decks) and gen2a (1500 decks),
the contrast was already striking: gen1 was green-heavy with an avg MV of 3.58
and 20 creatures per deck; gen2a shifted toward white and blue, dropped the
curve to 3.19, and traded creature slots for spells. Both are overwhelmingly
3-to-4-color.

The more revealing comparison came when I ran the same analysis on the forge-
best decks I extracted from `match-outcomes-all.txt`. Forge commits to 2 colors
99.9% of the time, carries 18.4 lands including 1.15 nonbasics, and plays
textbook sealed (15 creatures, 5 noncreature spells, avg MV 3.10). The model
has never learned to do anything like that.

I then pulled the win-rate stats from `analyze_winrates.py` with new tables
showing win rate by color count, by color presence, by creature count, and by
avg MV. The color-count table made the core finding stark: across all methods
combined, 2-color decks win 58.7% and 5-color decks win 32.5% — a 26pp drop.
That 35pp signal was even cleaner in gen2's training corpus. And yet gen2a
builds 4-to-5-color decks 44% of the time. The training data was broadcasting
exactly what to do, and the model couldn't act on it.

Claude's initial read was that the scorer might be conflating "2-color is good"
with "forge-best is good" because forge-best dominates the 2-color training
cells. I pushed back: the scorer never sees method labels, only card sets.
Claude corrected the framing: the real problem is that `build-decks` uses a
simulated annealing search evaluated on finished decks. The scorer is queried
on completed 23-card configurations, not on partial builds, so there is no
train/inference mismatch of the OOD-partial-deck variety. But that means the
scorer has the opportunity to prefer 2-color finished decks and still the SA
search lands at 4-color. The deeper issue may be that gen1's 5-color decks
won 46.6% in training (close to a coin flip), so the model learned "good 5-
color cards win, bad 5-color cards lose" rather than "5-color is structurally
bad." The mana-value breakdown added to this: the winning band is 3.4-4.0 avg
MV, yet gen2a deliberately moved to 3.19. Both moves — lower curve, more
colors — were calibrated against gen2's slightly improved win rate versus
forge-best, but within-bucket comparisons (controlling for both color count and
avg MV) show a 17pp gap to forge-best at matched deck shape. The improvement
was real but small; the score comparison within equivalent shape reveals the
model still cannot select better cards.

That led to the concern about whether the 32 deterministic features are carrying
almost all of the scorer's load. Color count, avg MV, type balance, pip
distribution — everything the score appears to have learned is a Set Transformer
aggregation over those hand-coded inputs. The 512 transformer embedding
dimensions, which are the only place card-level ability quality can live, may be
contributing almost nothing. This would also explain why Phase B embedding fine-
tuning never moved the needle: if the scorer barely reads those dimensions,
retraining them changes nothing. I asked Claude to document this hypothesis with
the diagnostic plan in `experiments/2026-05-02-deterministic-feature-reliance.md`. The
agreed test sequence is: first, mask the 512 transformer dims to zero at
inference and check if win rate drops; if it does not drop, that ends the
inquiry; if it does, retrain Phase A from scratch with those dims zeroed to ask
whether a fresh model would find them useful at all.

Toward the end of the session I mentioned my earlier jumpstart-tierlist project,
which logged per-game card plays and used wins-when-played (rather than raw win
rate or TrueSkill) to rank card quality. Claude recognized this as the missing
pretraining signal for the embeddings — a dense per-card label grounded in
actual play rather than deck composition. I had already rejected raw win rate in
the jumpstart context because it over-rewards rarely-played finishers; the same
logic applies here. This went into `experiments/2026-04-30-future-experiments.md` as a
candidate pretraining target alongside the color-restricted deck-builder
personalities (`--restarts color-pairs` and `--restarts color-slices`) that
would let me test whether the scorer's within-bucket gap to forge-best comes
from card selection or from the unconstrained search landing in a multi-color
local maximum.
