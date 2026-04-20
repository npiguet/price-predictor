# March 31, 2026 — PPO bugs found, Stage 1 completes

**TL;DR:** A cascade of PPO training bugs kept the deck-picker Stage 1
model from learning anything. Once all were fixed — most critically a
structural flaw in the transformer head — the model ran through all 40
curriculum levels in a single session.

The day started with a curriculum redesign. The old `best_run` mechanic
let the model coast on luck: since the first card pick is roughly 50/50
legal, a model could reach `best_run=10` without having learned anything.
The fix was to make episode length equal to `best_run` and require every
episode in a full batch of 32 to succeed before advancing. That way a
model at level N had to prove it could reliably pick N cards cleanly,
not just get lucky once.

With the new curriculum running, the training logs immediately revealed
that the model wasn't moving. It hovered at `best_run=1` for hundreds of
batches with no trend. My hypothesis was that the parameter count (25M)
was too large relative to the gradient signal being generated. Claude
traced through the math and found something more fundamental: the
episode runner recorded actions only for picks that did *not* terminate
the episode. Every recorded step reward was therefore +1.0. Batch
normalization of a constant array produces std=0, which makes every
advantage exactly 0.0. The policy gradient term contributed nothing to
the update — only the entropy bonus applied, which pushed the policy
toward uniform random. The model wasn't learning because it was receiving
no useful gradient signal at all.

The fix was to record the terminating pick with a −1 step reward before
breaking out of the episode loop, so the bad picks actually got a
negative gradient. After that change, the model advanced past `best_run=1`
promptly.

It stalled again at `best_run=4`, staying there for over 3,000 episodes
with no upward trend in mean reward. By that point land failures and
duplicate failures were both visible in the batch summary (another
diagnostic added that day). Claude diagnosed a structural problem in
`pool_transformer.py`: the scoring head was computing logits as
`mean_pool(encoder_output) → Linear(d_model, n_slots)`, meaning every
slot's score came from the same collapsed vector. The model could not
score slot `i` based on what card was actually at slot `i`. Booster land
cards appear at random shuffled positions each step, so there was no
stable signal for the model to learn from. Changing the head to
`Linear(d_model, 1)` applied independently to each slot's own encoder
output — keeping global attention context while making each logit a
function of its own card — was the fix that mattered.

After that architecture change the training went through levels almost
every single batch. The model completed Stage 1 (all 40 picks, zero
illegal terminations across a full batch of 32 episodes) in about 9,600
total episodes.

There was one more issue lurking: Stage 1 as designed taught the model to
avoid all lands, including in the phase where lands were supposed to be
legal. By the time `best_run` reached 40, the model was picking 40 spells
and ignoring the +1 reward for lands in Phase 2 entirely. The fix was to
drop the phase-1/phase-2 distinction and replace it with a dual budget:
+1 for a spell when `n_spell < 23`, −1 when over; +1 for a land when
`n_land < 17`, −1 when over. Under this reward, the right 23/17 split
emerged naturally. The curriculum advancement gate was also tightened to
require `mean_reward > 0.9` (roughly ±2 lands off ideal) in addition to
all episodes completing.

A final experiment: removing the explicit `is_land` flag from the
feature vector entirely, to see whether the frozen card embeddings
contained enough type information for the model to figure it out from
the embeddings alone. The model reached `best_run=36` under training,
suggesting the encoder's representations do encode land vs. spell
implicitly well enough to act on.

The second and third sessions that day were a Java review of the
forge-connector module — finding duplicated tree-traversal methods,
copy-pasted dice-outcome walkers, and a collapsed SA chain → Ability tree
→ flat text round-trip that suggested the Ability tree wasn't being used
as the single source of truth. All of that got implemented: seven
refactoring steps that removed the duplicates and consolidated the
structure, with 196 Java tests passing at the end.
