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

## Outcome

To be filled in once gen-3 is designed and run.
