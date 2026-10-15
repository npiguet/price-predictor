# Ability effect model — gen-3 (draft)

Draft design record for the third generation of the ability effect model. It collects the gaps in
collection and in the curated corpus that gen-2 leaves open, so that gen-3's design starts from
them. Gen-2 is described in
[`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md),
called the gen-2 record below. Nothing here is decided yet.

A few terms recur. A record is one observed game event: an ability resolving, a combat, a trigger, a
legality decision. A resolution record comes in two halves joined by a link, the cost half and the
effect half. A record names the ability that acted by its provenance key, the script file, face,
trait kind and index of the Forge trait. A sidecar is the file beside each converted card that maps
those keys to the card's rendered ability lines. The random seat is a player that makes a random
legal choice at a share of its decisions instead of the Forge AI's. A probe fork is a copy of the
game, taken at the combat damage step, in which one keyword is stripped from one creature to see
what changes.

## Collection can lose or misplace events without leaving a trace

**Events dropped by the collectors are never counted.** An engine event that arrives while no
record is open is discarded, and nothing records that it happened. A state-based death was lost
this way for a whole gen-1 corpus before anyone noticed. Gen-3 should count every discarded event
by event type and engine phase, behind a debug flag, and run a short pilot with the count on.

**A death can be written into a trigger that it happened before.** A dies-trigger put on the stack
during the state-based check that follows a resolution ends the open effect half. A death in the
next pass of the same check then lands in the trigger's bracket. It is written into the trigger's
effect half when the trigger resolves, although the death came first.

**The first-strike damage step lost everything but its damage to players.** The collector
decided that an event was combat damage by checking that the phase name contains
`COMBAT_DAMAGE`. Forge names the first-strike step `COMBAT_FIRST_STRIKE_DAMAGE`, which does not
contain it. In that step only damage to a player opened a combat record, because that path reads
the event's own combat flag. Damage to creatures, the life changes and the deaths went to the
ordinary bracket, which no ability holds open during a damage step, so they reached no record.
Gen-1 and gen-2 were both collected this way.

The loss left two marks on gen-2's raw shards. A first-strike-step record exists only when a
player took damage, and it holds that damage and nothing it caused: the defending player's life
never moves. A first-strike fight between creatures alone has no record at all, so no record
shows a first striker killing its blocker before the blocker strikes back. The probe forks are
collected on another path and record the step correctly. On a sample of 40 raw shards the
first-strike step occurs in one game in ten and makes 6% of combat records.

| first-strike steps, 40 raw shards | count |
|---|---:|
| seen in any record | 361 |
| with a combat record, a player having taken damage | 282 |
| of those, with a blocked fight | 12 |
| with no combat record | 79 |

Gen-2 repairs its built corpus and leaves the raw shards alone, because it is not collected
again. `scripts/repair_first_strike_steps.py` adds each record's missing player outcomes from
the record's own damage events: the life loss, the lifelinker's gain, and poison for infect
damage. It drops a record holding a blocked fight in which either side strikes first, because
that fight's creature damage and deaths are gone. Gen-2's real first-strike knowledge therefore
covers unblocked attackers, and its creature-side knowledge of first strike comes from the probe
forks alone.

The collector now tests both damage-step names, and a collector test fixes the case. Gen-3's
collection is the first to record the first-strike step whole. The dropped-event count proposed above
would have caught this loss on the first pilot.

**Damage events on a probe fork lose their source.** On a probe fork, every `damage_dealt` event
lacks `params.cause`, and damage to a creature reports `combat: false`. Observed combats carry both
correctly. The source can be recovered from the payload's `assignment_choices`. Because the source
is missing, several attackers' identical hits produce byte-identical events, which inflates the
validator's "repeats an event" check. Gen-1 and gen-2 shards both carry the gap.

## Some records cannot be joined to the text of the ability that acted

**A charm cast through Flashback is keyed to no line.** The collector keys the chosen mode as the
Flashback keyword trait plus the mode's `option`, as in `wreck_and_rebuild.txt keyword[0]
option=0`. The converter writes `option` keys only for a charm's spell root and for saga chapter
keywords, so that key matches no line in the sidecar. Gen-2 holds one such record. Its
`build-corpus` refuses records like it as `unjoinable-key`, which is a workaround rather than a fix.

**An ability a clone gains from the card it copies cannot be keyed.** A Gigantoplasm copying
"Winter Soldier, Bucky Barnes" gains that card's "{X}: base power and toughness X/X" ability through
`AddAbilities$`. The collector's key resolution does not reach abilities granted that way, so the
records come out with an empty ability and `build-corpus` drops them. Traits a clone copies are
likewise recorded as `granted_trait`. Gen-3 should key these abilities, because the plays they
record are ones the model should learn from, as the random-seat section below explains.

**Some variant records are keyed under the card tree.** About a hundred keys from the gen-2 variant
run name files like `cardsfolder/a/a_cosmos_charger_variant_5001.txt` instead of
`variant-scripts/`. They appear to be variants of Alchemy "A-" cards. No sidecar exists at that
path, so the key resolves to no text, and the record is dropped or kept without text rather than
failing the build. The cause has not been looked into.

## The random seat can repeat a free activation that changes nothing

**The random seat never passes while an ability is still activatable.** In one gen-2 coverage game,
the random seat activated the Gigantoplasm's granted ability about 2,190 times in a single turn, at
X = 0. Each activation cost nothing and resolved to an empty change. The ability sets base power and
toughness, which sits below the creature's +2/+2 boost, so the creature stayed a 2/2 rather than
dying as a 0/0. Once such records are keyed, the corpus's per-text cap trims the repeats to a
handful. The play itself is wanted: a pointless activation, and an activation that kills the
creature, are both decisions the model should learn the outcome of. What gen-3 needs is for the
random seat to stop looping on a free, repeatable ability.

**The snapshot may hide a creature's dependence on a boost.** After the base became 0/0 under the
+2/+2 boost, the snapshot still reported `base [2,2], boosts [0,0]`. It appears to report the
printed base and fold set-base effects into the boosts, so a creature that survives only because of
a boost looks like an ordinary 2/2. This needs checking in general, not only on this card.

## Two sampling rules over-represent some questions

**What-if legality queries are sampled before they are de-duplicated.** A what-if query is the AI
asking whether an attack or block would be legal without committing to it. `--legality-rate` draws
each query before identical snapshots are merged, so a question the AI asks K times survives with
probability 1 − (1 − r)^K. Blocker what-ifs outnumber real block decisions about five to one.

**The held-out coverage round covers more than the held-out texts.** `collect-coverage
--only-cards` targets every text carried only by held-out cards. That set is larger than the
held-out texts themselves.

## The observed-mana probe counts every mana event as one mana

**The knowledge probes' observed-mana target reads a field the collector never writes.**
`labels.mana_observed` sums each `mana_produced` event's `params.amount`. The collector writes
`params.mana_by_color` instead, a map from colour to count, so every event counts as one mana
whatever it produced. On gen-2's first arm the target scored no rung at all, so no gen-2 conclusion
rests on it.

**Two questions about what the collector writes have to be settled before the fix.** A dual land's
event reads `{"W": 1, "R": 1}`, which may be one mana of either colour or the collector listing
both of the land's colours. Thousands of gen-2's events carry an empty map. The probe can count
mana correctly only once both are understood, and either may turn out to be a collector defect
rather than a probe one.

## Training could draw more of the records a weak output field needs, from the whole corpus

**Some output fields stay weak while the others improve.** The model predicts 46 output fields per
entity, such as life change, damage taken or keywords gained. Each is scored by its explained
deviance: the share of the loss of the best constant prediction that the model removes, where 0%
is no better than always guessing the field's average. On the first gen-2 arm, a few fields with a
real amount to learn barely moved between the first two epochs that scored them.

| field | loss of the constant prediction | epoch 3 | epoch 4 |
|---|---:|---:|---:|
| `keywords_gained` | 0.565 | 21% | 21% |
| `poison_delta` | 0.138 | −11% | −14% |
| `life_delta`, for comparison | 2.402 | 80% | 81% |

Gen-3 could let training respond: when a field stays weak, show more of the records that
supervise it. Gen-2 fixes each record's share of training when the corpus is built, and the
trainer only shuffles within the one shard it holds.

**The idea is established, as adaptive data mixing and adaptive loss weighting.** Large
language-model pretraining stores its data as separate sources and draws from each by a mixing
weight. Methods such as DoReMi raise a source's weight while its loss stays high against a reference
model. Multi-task training, of which these 46 fields are a case, balances the tasks' loss weights
instead, with methods such as GradNorm and uncertainty weighting.

**Four mechanisms are possible, from cheapest to most flexible.**

1. **A weight per field in the loss.** The trainer already computes every field's loss separately,
   so nothing on disk changes.
2. **A field factor in the weighted shuffle within a shard.** An epoch uses about a third of each
   shard's records, so this can raise a field's share about threefold and no further.
3. **Pools with mixing weights.** `build-corpus` writes the training stratum as several shard
   directories, one per pool, and files each record in the pool of the rarest field it supervises.
   Gen-2 already gives a legality record the rarest single mode on its board. The trainer draws each
   epoch's shards from the pools by weights a controller adjusts at each epoch boundary. A pool of a
   few hundred records can stay in memory and be mixed into every step.
4. **An index from field to records, possibly in a database.** A record would then count for every
   field it supervises, not only its rarest. The cost is reading scattered records across the whole
   corpus, where today the trainer reads whole shards in sequence.

**Five constraints apply whichever mechanism is chosen.**

- **Calibration.** A field drawn at five times its natural share teaches the model that its outcome
  is five times as common. Each record's loss then needs an importance weight, its natural share
  divided by its drawn share. Loss weighting does not shift the base rates in the same way.
- **Memorization.** A rare field may rest on few distinct texts; poison comes from a handful of
  infect and toxic abilities. Raising its share repeats those texts. The ceiling has to be on how
  often each distinct text is seen per epoch, as gen-2's reuse cap is.
- **Reproducibility.** A gen-2 run repeats bit for bit from its seed. A controller driven by losses
  does not, because GPU arithmetic varies between runs. Changing the weights only at epoch
  boundaries, and logging them, lets a rerun replay them.
- **Weak against unlearnable.** A field can be weak because the snapshot lacks what decides it. The
  controller should boost a field only while its training loss keeps falling, and pass a field that
  stays flat to collection.
- **Validation stays out of the loop.** The controller reads training loss. If the card-disjoint
  validation steered training, it would stop being an unbiased measure.

## The Comprehensive Rules could index what the corpus covers, rule by rule

**Gen-2 measures coverage by card text and by rule family, not by the rules of the game.** A rule
family is the script-level class `build-corpus` balances on, such as an ability's API type or a
trigger's mode. Whether the corpus shows the legend rule, a creature dying at toughness 0, or two
continuous effects applied in layer order is never counted. Gen-3 could go through the
Comprehensive Rules and record three things for each rule:

1. whether the records the collectors write can show it at all;
2. how many records in the corpus do show it;
3. which game situations would illustrate it, as a unit test illustrates one behaviour of code.

The counts would then feed corpus balancing and collection, and the illustrations would become a
test set.

**A triage pass comes first, because many rules can never appear in a record.** The rules run to
thousands of numbered subrules. Deck construction, mulligans, multiplayer variants and most
priority bookkeeping have no counterpart in an effect record. Sorting every rule into observable in
a record, observable only indirectly, or out of scope decides how large the rest of the work is. A
language model can make the first pass over the document, with a person checking the borderline
rules.

**Counting a rule needs a detector that recognizes it in a record, and detectors range from trivial
to impossible.**

| rule | how a record shows it | detector |
|---|---|---|
| 702.2, deathtouch | a combat with a deathtouch creature, and the probe forks that strip it | exists: the combat family and the probe keyword |
| 704.5f, a creature with toughness 0 dies | a zone change during the state-based check after a resolution | derivable from the effect half's events |
| 613, the order of continuous-effect layers | no field records which layer applied what | needs new collected fields |

A detector is also a coverage counter, so the same code would put per-rule counts in the corpus
manifest.

**Illustrating a rule needs situations built on purpose, not found in self-play.** Coverage rounds
target cards, and a rare interaction between rules may never occur in a sealed game. Two sources
could build them:

- **Interventional forks.** They already force an ability onto the stack in a copied game. A
  rule-driven fork would set up the board the rule needs first.
- **Forge's puzzle files.** Forge ships 372 puzzles under `forge-gui/res/puzzle/`, each a starting
  game state: life totals, the active player and phase, and every zone's cards with their links.
  Written for a rule, such a state is a unit test the collector could play. Whether a collecting
  worker can load one has not been checked.

**The rule tests that judge the model must stay out of its training.** If the same situations both
balance the corpus and measure the model, they stop measuring generalization. A frozen,
hand-checked set of rule scenarios would join the knowledge probes as an evaluation. Collection
targeted at a rule would feed training separately.

**Rules could also be the pools of the adaptive sampling above.** A per-rule detector gives each
record the rules it exercises, which is the same index that sampling by field needs. The two items
would share the decision between pools, an index and a database.

## The snapshot should list every line of a permanent, not only its spell and activated abilities

**The model never reads a permanent's statics, triggers or replacements.** `SnapshotBuilder.printedKeys` fills a permanent's card-slot lines from Forge's `getSpellAbilities()`, which holds the cast spell and the activated abilities. Attachment-granted lines come from the same call on each attachment. About a quarter of the permanents on the battlefield carry a line this leaves out, and the knowledge-probes record gives the counts. Keyword lines are unaffected, because the keyword flags carry them.

**What a static has already done reaches the model, and what it decides at the moment of the record does not.** An anthem's +1/+1 is in the toughness and a granted keyword is in the flags. A "can't block" static, a replacement that exiles instead of destroying, a damage doubler, or a lord whose removal shrinks its creatures are text the model needs and never sees. Pacifism shows it an enchantment attached to a creature and nothing more.

**The fix is in the collector, so gen-2 keeps its corpus and measures the cost instead.** `printedKeys` and `grantedAttached` would add the card's live triggers, static abilities and replacement effects, through `getTriggers()`, `getStaticAbilities()` and `getReplacementEffects()`. Reading Forge's live lists respects a creature that has lost its abilities, which reading the card's sidecar at training time would not. Abilities a permanent gains from another permanent's static, such as a lord granting a mana ability, still appear nowhere, and are worth listing in the same change. A board-heavy record then encodes more texts per step, so training slows somewhat.

**Gen-2's board-abilities ablation is the baseline.** It scores each gen-2 arm on legality restricted by a non-keyword static and on resolutions with a replacement on another permanent, beside controls of the same kinds. A gen-3 model that reads the new lines should close those subsets' loss toward their controls, and noise in its card slots should start to cost something there.

## Early stopping could read a smoothed loss in which every output field counts

**The card-disjoint loss moves between epochs by about as much as it improves, so stopping on its
raw value can fire on noise.** Gen-2 training keeps the epoch with the lowest card-disjoint
validation loss and stops after five epochs without a new lowest, a rule called patience. Late in
gen-2's run with 64-dimensional ability embeddings, the loss went up and down by about 0.1 from one epoch to the
next while its trend fell by less than that per epoch.

| epoch | card-disjoint loss |
|---:|---:|
| 20 | 3.9391 |
| 21 | 4.0106 |
| 22 | 3.9189 |
| 23 | 4.0816 |
| 24 | 3.7801 |
| 25 | 3.7344 |

An unusually low epoch resets the patience counter and sets a mark that ordinary epochs then fail to
beat. Five ordinary epochs in a row then end training while the trend is still falling. The kept checkpoint is
also the luckiest epoch, so its loss overstates the model.

**The first change is to stop on a smoothed loss.** Two forms are common:

- **A moving mean of the last three epochs' losses.** Patience counts epochs without a new lowest
  mean. The kept checkpoint is the middle epoch of the best window.
- **An exponential moving average of the weights.** Training keeps a running average of the
  parameters, each step moving it a small fraction toward the current weights, and validation scores
  the average. The average is itself less noisy than any single epoch's weights and is usually a
  slightly better model, so it is both what stopping reads and what the run keeps. `resume.pt` would
  have to carry it.

**The second change is to count every output field equally in the stopping loss.** The loss is a sum
over the 46 output fields, and a few large fields carry most of it: the gate, toughness, damage taken
and life. A field such as `keywords_gained` can get worse without moving the total. Each field can
instead be scored by its explained deviance on the card-disjoint stratum, the measure defined in the
section on adaptive sampling above, and stopping can read the mean over the fields. Two details need
deciding:

- **Fields with almost nothing to learn.** A field whose constant prediction already has a tiny loss,
  such as `poison_delta`, has an explained deviance that swings widely on a few records. Such fields
  need a floor on their constant-prediction loss to enter the mean, or a weight that grows with it.
- **What a point is worth.** An equal mean rewards a weak field going from 20% to 30% as much as
  life going from 80% to 90%. That is the intent, but it lets a weak field's noise steer stopping,
  which makes the smoothing above more necessary, not less.

The same per-field scores are what adaptive sampling would read on the training side, so the two
items would share their bookkeeping.

## Outcome

To be filled in once gen-3 is designed and run.
