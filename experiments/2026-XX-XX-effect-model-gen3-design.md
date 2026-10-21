# Ability effect model — gen-3 (draft)

Draft design record for the third generation of the ability effect model. It collects the gaps in
collection and in the curated corpus that gen-2 leaves open, so that gen-3's design starts from
them. Gen-2 is described in
[`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md),
called the gen-2 record below.

Gen-3 collects its corpus from scratch and keeps no gen-1 or gen-2 record. Its record schema is
therefore a new version: a field can change meaning or go away, and no gen-3 reader needs to read a
gen-2 shard.

One goal is set: gen-3 rebalances its training examples rule by rule, so that no keyword or
ability counts as rare in what the model trains on. Three sections below are its means. The rules
index counts how many records show each rule of the game. Adaptive sampling draws more of the
records a weak output field needs. The turn-rule section balances each condition between met and
unmet. Beyond the schema and that goal, nothing here is decided yet.

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

## The Forge hooks recorded the AI's block plan as block legality, and four outcomes reached the wrong entities

Gen-3's hooks fix all five defects the gen-2 record describes ([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#five-collector-defects-shaped-gen-2s-labels-and-gen-3-collects-without-them)). The fixes are commits on the `effect-record-hooks` branch of `../forge` and in the connector. They change what a record says, so they are in place before gen-3's first collection.

**Block and attack legality are asked by Forge's declare steps, before anyone declares.** The hook lists every creature the player controls as a candidate, and the combat holds no blocks yet when it is asked. A creature planned for one block can no longer read as forbidden for another. A tapped or "can't block" creature is always listed as forbidden. The answer is the same whatever controls the seat, so the random seat no longer writes legality records of its own, and the AI's internal legality queries are no longer hooked. Every legality record is therefore a real declaration, one per defender or attacker per combat. `--legality-rate`, which sampled the AI's what-if queries, now has nothing to sample; removing the flag is left to the gen-3 collection work.

**A reveal from a library during a resolution names the players whose libraries it came from.** A scry already names the player who scried in the same way. A reveal from a hand, a graveyard or any other zone names nobody. So does a reveal made while paying a cost, which happens before the spell it pays for resolves. A clash names both clashing players.

**Each kind of object an effect makes gets its own note, and a token's note carries its keywords.** Bestial Menace now records a Snake, a Wolf and an Elephant. The collector writes a token's script id and keywords with the code that writes every token on the board, so the created-objects field's keyword flags have something to learn from.

**Prevented damage names the creature or player it would have been dealt to.**

**A note that names no subject marks nothing affected.** Coin flips, token creation, copies, votes and ventures no longer mark the acting card as affected.

**Two bits of the `library_events` field are still never set.** The target mapping writes the `scry` bit for both scry and surveil, and nothing writes the `surveil` or `tutor` bit. The mapping is in `src/effects/domain/effect_targets.py` and is left to the gen-3 target work.

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

## The summed validation loss disagrees with the probes over training, and gen-3 judges checkpoints by probes and rule scenarios

**After the card-disjoint loss reaches its lowest, the probes keep improving.** Each of gen-2's
four continuation runs kept its best epoch by card-disjoint loss and also probed its last epoch.
The model-level probes ask the model's own outputs targeted questions, such as whether a card is
affected, whether it dies, or whether a target is legal. They are scored mostly as AUC, the chance
that a true case ranks above a false one. In every run the last epoch has the higher card-disjoint
loss, by how the best epoch is chosen. On the same card-disjoint records, the probes rate the last
epoch as good or better in three runs of four, and on the game-disjoint records in all four. The
game-disjoint loss points the same way as the probes in the two 128-wide runs only.

| gen-2 run, best epoch → last epoch | card-disjoint loss | game-disjoint loss | probe mean, card-disjoint records | probe mean, game-disjoint records | "affected" probe, card-disjoint records |
|---|---|---|---|---|---|
| run A64 | 3.519 → 3.759 | 2.906 → 2.913 | 0.953 → 0.951 | 0.958 → 0.960 | 0.979 → 0.981 |
| run B64 | 3.506 → 3.643 | 2.880 → 2.942 | 0.956 → 0.956 | 0.963 → 0.963 | 0.978 → 0.980 |
| run A128 | 3.708 → 3.802 | 2.908 → 2.864 | 0.944 → 0.949 | 0.944 → 0.957 | 0.980 → 0.976 |
| run B128 | 3.634 → 3.689 | 2.860 → 2.846 | 0.945 → 0.948 | 0.952 → 0.955 | 0.981 → 0.982 |

**The loss rises because the model grows more confident, not because it ranks worse.** The rise sits
mostly in the gate field, whether a card was affected, and in poison, with power and toughness
changes next. In three runs of four, the probe asking the model whether a card is affected scores higher on
the same records. Log-loss charges heavily for a confident miss, and a text the model never saw
produces a few of those. The summed loss therefore mixes how well the model ranks outcomes with how well its
confidence matches its accuracy, and the second can move it the wrong way.

**Gen-3's control score is built from probes and rule scenarios.** The control score is what
stopping reads and what picks the checkpoint the run keeps. Gen-3 builds it from the model-level
knowledge probes, the damage-by-toughness and combat grids, and the frozen rule scenarios of the
rules index. Every input is a forward pass, so nothing is trained to compute it. Built per rule, the
score follows the balance training aims for rather than how often a situation happens in
self-play. The smoothing of the section above applies to it as well.

**The probe sets grow, and a selection half picks while a reporting half measures.** Choosing the
best of many epochs on a score favours the epoch whose noise happened to help, so the chosen
epoch's score overstates it by an amount that grows with the noise. More probe records shrink that
noise. Selecting and reporting on the same records would still flatter the chosen epoch, so each
probe set and the rule scenarios are split in two. One half picks the checkpoint, and the other
half reports it. With the sets enlarged, each half can be as large as gen-2's whole set.

**Each validation stratum judges the part it represents.** The in-game head predicts outcomes for
cards that already exist, so its score reads the game-disjoint stratum. The embedding `e` serves
deckbuilding on new sets and hypothetical cards, so its score reads the card-disjoint stratum,
where gate 1 also lives. A run keeping one checkpoint for both weighs the two scores explicitly.

**Two other designs are not taken:**

- **Stopping on discrimination with calibration fitted afterwards.** It would score each field by
  AUC or rank correlation and fit one temperature per field after training. It changes the measure
  without measuring the questions that matter more directly than the probes do. Calibration stays
  a reported number, because the in-game head's probabilities are used as probabilities.
- **A fixed schedule that keeps the last epoch.** It would end every run with a tenth of the
  learning rate and keep the last epoch, as gen-2's runs B64 and B128 suggest works. It avoids a
  poor measure instead of replacing it with a better one.

## The training loss puts every output field on one scale, with amount distributions that fit

**Each field's loss is divided by the loss of its constant prediction.** The training loss sums the
46 output fields, and the gate, damage taken and life carry most of it. After the division, a field
counts by the share of its constant prediction's loss that remains, 1 when it has learned
nothing. This is the training-side twin of the explained deviance the stopping section scores. A
field with almost nothing to learn, such as poison, needs a floor on its constant-prediction loss, or
its noise is amplified. Per-rule rebalancing does not make the division unnecessary: with every
rule equally represented, a yes-or-no field on every card and a life-change amount still sit on
different scales.

**Amounts get a distribution wider than Poisson.** Life, damage and card counts are scored by a
Poisson likelihood, which assumes the variance equals the mean. Life and damage swings vary more
than that, so a rare large swing is penalized as if it were nearly impossible. A negative binomial,
or a distribution over bins of the amount, fits them better. The constant prediction the division
uses is then computed with the same distribution, so the two changes combine.

**Rule-labelled copies fill the gaps the real data leaves.** Gen-2's lethality copies redraw the
amount and the toughness of a qualifying damage spell and label the death by the rule. Where the
rules index finds a rule with too few qualifying or varied real records, copies of the same kind
supply them. The combat copies proposed below are one case.

**Two designs are not taken:**

- **Learned field weights**, such as the uncertainty weighting named in the adaptive-sampling
  section. Each field's weight is trained along with the model, and a field the model finds hard
  ends up weighted down, which works against the per-rule rebalancing.
- **Label smoothing on the gate.** It would cap the gate's confidence, the main cause of the loss
  rise above. Once checkpoints are judged by probes, that confidence no longer misleads the
  choice.

## The model has to learn every rule that depends on whose turn or which step it is, and collection records which check decided each one

**The `can_play` label leaves out the casting-timing rule, so the corpus labels a sorcery playable on the opponent's turn.** A `decision` record's three verdict bits are `can_play`, `affordable` and `has_legal_target`, and `can_play` is Forge's `SpellAbility.canPlay()`. That call covers split second and a script's own restrictions: the phases it names, whose turn it requires, activation limits and zones. The default timing rule lives in a separate call, `canCastTiming`, which no hook records. Under that rule a spell is cast at sorcery speed unless it is an instant or has flash, and an activated ability with `SorcerySpeed$` or a loyalty cost needs sorcery timing too. The hook in `AiController.canPlayAndPayForFace` fires after the AI's own timing check whatever that check answered, so a sorcery the AI weighs during the opponent's turn is written as playable. The random seat records `canPlay` as well, though it applies timing to its own choice of play.

**The AI writes only candidates that already passed `canPlay`.** `AiController.canPlayAndPayFor` returns before the hook for a candidate that fails it, so every `can_play = false` in the corpus comes from the random seat.

**On the AI seat, gen-1 and gen-2 `has_legal_target` and `affordable` record the AI's choices rather than the rules.** `has_legal_target` reads whether the AI had set targets on the candidate. The AI sets targets only when its evaluation gets that far and finds a target it wants. A candidate the AI put off until later is therefore written with no legal target. A candidate evaluated earlier can keep that evaluation's targets and is written with one. `affordable` comes from `ComputerUtilCost.canPayCost`, which applies the AI's cost preferences on top of the rules. It refuses a planeswalker's last loyalty and a Ward cost the AI dislikes, and it draws from the shared random stream to decide. The random seat uses the same cost check. It counts target candidates rather than reading chosen targets, but it walks a charm's leftover chain of chosen modes instead of the modes the charm offers. Verdict-head results and probes on gen-1 and gen-2 records inherit all of these errors.

**Gen-3 computes all three verdict bits in the collector, from the rules alone, for both seats.** The AI's hook passes only the candidate, before the AI evaluates it and ahead of its `canPlay` filter. The collector's `PlayabilityVerdict` checks mana and additional costs without the AI's cost preferences. It counts each targeting clause's candidates against the clause's minimum. It judges a charm by how many of its offered modes have reachable targets. The random seat filters its own plays on the same verdict.

**Split second on the stack reaches the model only as the stack card's ability text.** The card slots' keyword flags have no `split_second`, and the knowledge probes found the trunk reads little of other cards' text.

**The fix is in the collector, and Forge's own answer is the label.** Each candidate carries a fourth verdict bit, `timing`, holding `canCastTiming`, computed by `PlayabilityVerdict` for the AI seat and the random seat alike. Recomputing the rule in Python from the snapshot would miss what Forge already applies:

- flash granted by a static, such as Leyline of Anticipation or Vedalken Orrery;
- a static that holds opponents to sorcery timing, such as Teferi, Time Raveler;
- a spell cast through another effect, which ignores its own timing.

Where Forge can name the static that decided the timing, its key goes in `responsible_static`, as for the other verdicts. The AI's hook moves ahead of its `canPlay` filter, so the AI's candidates include the ones that fail. The keyword flags gain `split_second`. `can_play` becomes Forge's whole answer, `canPlay` and `canCastTiming` together, and the verdict head gains a column for `timing` beside it.

**Casting speed is one case of a wider family, and a timing bit alone does not teach the rest.** Many texts depend on whose turn or which step it is:

- **activation restrictions**: "activate only during your turn", "only during combat", "only once each turn", "only if you control an artifact";
- **resolution branches**: a spell or ability that does more, or something else, when it resolves on its controller's turn or an opponent's;
- **conditional statics**: a permanent whose static abilities differ between its controller's turn and an opponent's, so that its keywords or its restrictions change when the turn passes.

The model needs to read each condition from the text, compare it with the board, and predict what follows. Each kind needs its own label at collection, because a single `can_play` bit or a resolution's outcome often does not say which condition decided it.

**Each candidate records which rules checks it fails, not only that it fails.** Beside `timing`, a candidate carries `failed_checks`, a list naming each check Forge found unmet: split second, casting speed, whose turn, phase or step, combat, activation limit, a script condition, the zone, a static prohibition, and the cost. Each named check carries its `responsible_static` where a static decided it. Forge groups most of these inside `SpellAbilityRestriction.canPlay`, so the hook calls the parts separately where Forge exposes them, and the implementation has to confirm how finely it can split them. The verdict head gains a column per check. "Activate only during your turn" is then supervised by its own column as well as through `can_play`.

**A resolution records which branches of its chain ran.** Forge evaluates each chain segment's `Condition$` as it resolves. The resolution payload gains `conditions_met`, one entry per segment the chain conditions, saying whether the condition held, keyed by the segment index `attributed_to` already uses. An outcome alone can be ambiguous, because a branch that ran may change nothing visible on this board. The new field says which branch ran, and the model can be trained to predict it at the acting line.

**The snapshot records whether each static line is in force.** Once the snapshot lists a permanent's statics, as the section above on statics, triggers and replacements proposes, each listed static carries `active`, Forge's answer to whether its condition holds now. A permanent with different statics on its controller's turn then shows the text of both, which one is in force, and, through the `continuous` records, what the one in force does.

**The board input says who controls, who owns, whose turn it is and who holds priority, for every player and every card.** Three roles are easy to confuse. The active player is the player whose turn it is (rule 102.1), and changes only when the turn does. The player with priority is the one allowed to act now (rule 117), which passes back and forth within a step, so an opponent holds priority while responding during the active player's turn. The acting player is the record's own notion: the controller of the ability being recorded, or the player making the decision.

Gen-2's input states each of these only relative to the acting player. The global slot says whether the acting player is the active player and whether they hold priority, a player slot says whether it is the acting player, and a card slot, a spell on the stack included, says whether the acting player controls it. To learn that an opponent's creature belongs to the active player, the model has to combine two of those flags. The snapshot records no owner at all, which matters wherever control has changed: a stolen creature, "return it to its owner's hand", a card going to its owner's graveyard.

| gen-3 input | slot | source |
|---|---|---|
| is the active player | each player | computed from the existing snapshot |
| holds priority | each player | computed from the existing snapshot |
| controller is the active player | each card | computed from the existing snapshot |
| controller holds priority | each card | computed from the existing snapshot |
| owner is the acting player | each card | new `owner` field per entity, at collection |
| owner differs from controller | each card | new `owner` field per entity, at collection |

The acting-player bits stay. Every row but the owner rows is an input-layout change computed from what the snapshot already holds, and the `owner` field is a new snapshot field the collector writes for every entity.

**Each condition is balanced between met and unmet in the curated corpus.** Forge's AI casts sorcery-speed spells on its own turn and holds most instants for the opponent's, so a turn condition is seen mostly one way. Training on that teaches the base rate rather than the condition, as Forge's lethal removal did for deaths in gen-2. The fix has three parts:

- **Curation splits each condition's budget by its value.** Gen-2's curation already splits each class over rule families and then over outcome signatures, with one water-filling allocator, a reuse cap on short cells and the shortfall reported in the manifest. For a record whose acting line, candidate or listed static carries a turn or step condition, the signature gains whether the condition held, taken from `failed_checks`, `conditions_met` or the static's `active`. Met and unmet then draw equal budgets, a short side repeats up to the reuse cap, and the manifest names every condition that stays one-sided.
- **Collection evaluates both players' candidates at every priority stop.** A decision record is a rules-level question and takes no action, so the collector can ask it for the player without priority too. The answers come from the same board under both values of "whose turn it is" for that candidate's controller, which balances whose-turn checks by construction. This needs the hook to evaluate candidates without priority correctly, because Forge's `canPlay` assumes the asking player could act; the implementation has to establish what Forge answers there.
- **Interventional forks fill the side that stays short.** A text whose resolution branch is seen on one turn only is a candidate for a fork that resolves it at a phase boundary of the other player's turn, under the existing fork rules and budgets.

**A knowledge probe measures each kind.** It takes records whose acting line, candidate or board carries a turn or step condition and edits whose turn it is and the phase, then reads the prediction the condition governs: the matching verdict column for a restriction, the predicted outcome or branch for a resolution, and the affected creature's keywords or legality for a conditional static. Further subsets cover casting speed under the same edits and a split-second spell on the stack. Gen-2 has no such columns, so its baseline is `can_play` and its outcome predictions under the same edits.

## A combat record's snapshot should be taken before the damage step deals any damage

**A combat record's snapshot already shows the damage of the first creature or player hit.** `BusBracketCollector.openCombatBracket` takes the snapshot when the step's first damage event arrives, from `onCardDamaged` or `onPlayerDamaged`. By then Forge has marked that first damage, so the creature it hit shows it as marked damage, and a player hit first shows the lower life total. In 26 of the 40 one-on-one blocks gen-2's combat grid reads from the validation strata, the blocker's marked damage equals the attacker's power. The attacker never shows damage in those records.

**The model can read part of the answer off the board.** Gen-2's input gives each card a thermometer of toughness minus marked damage, so a blocker the attacker already damaged shows whether that damage is lethal. A combat record then asks about a board on which half the step has happened, and how much of the outcome leaks depends on which damage Forge dealt first.

**The fix is to take the snapshot at the start of the damage step.** The damage-step probe forks already copy the game at the phase event, before any damage, so the observed record can take its snapshot at the same moment. The assignment and the combat shape are read at the first damage event, as now, because Forge has assigned the damage by then.

## The board input should say which attacker each blocker blocks

**A card slot says that a creature is blocking, not which attacker it blocks.** The snapshot records the pairing in each creature's combat status, `blocking` and `blocked_by`, but the card features keep only a blocking bit and a count of blockers. With two blocked attackers on the board, the model cannot tell which blocker deals damage to which attacker, so a combat's deaths depend on a pairing it has to guess. Gen-2's combat grid reads only combats with one blocked attacker for this reason.

**The pairing belongs in the trunk's input.** Two forms would carry it: each card slot gains the board position of the attacker it blocks, or the trunk's attention gains a bias between a blocker and its attacker, as it could between an aura and the permanent it enchants. Either needs no new collection, since the snapshot already holds the pairing.

## The keyword flags should say what a creature has protection from

**Gen-2's keyword flags collapse every protection into one flag.** The overlay maps `protection_from_red`, `protection_from_blue` and every other protection onto a single `protection` slot. A printed protection line still reaches the model through its text in `e`. A granted one, such as Gods Willing's "gains protection from the color of your choice", reaches it only through the flag, and the flag cannot say which colour. Gen-2's evasion grid shows the cost directly: its two protection rows, protection from the blocker's colour and protection from another colour, give the model identical inputs and so identical predictions, where the rule forbids the first block and allows the second.

**Protection is not always from a colour.** Forge's card scripts print protection on 196 cards. Colours are most of it, and the rest is a long tail, several of its qualities on a single card.

| protection from | printed protection lines |
|---|---:|
| a colour | 153 |
| a creature type, such as Goblins, Dragons or Vampires | 23 |
| a card type, such as artifacts, creatures or instants | 21 |
| multicolored, monocolored, or each colour | 9 |
| a condition, such as mana value 3 or less or a counter | 4 |
| a creature subset, such as legendary creatures or Gods | 3 |
| everything | 1 |

**Gen-3 gives each granted protection its quality.** Two forms would carry it:

- **Fixed flags for the common qualities and one open slot.** One flag per colour, per card type, for multicolored and monocolored, and for everything, plus an "other" flag for a creature type or a condition. Cheap, but the open slot cannot say which creature type.
- **Every granted keyword encoded as a line.** The overlay's granted keywords pass through the encoder as their expanded definition, "protection from Goblins", the way a printed keyword line already does. Any quality then reaches the trunk as text, including ones no flag anticipates, at the cost of an encoder pass per granted keyword.

**The rare qualities need built examples, or the flags carry nothing the model learned.** A creature with protection from everything appears on one card, and protection from most creature types on two or three. Self-play shows them in a handful of games at best. Rule 702.16 says what protection does: damage from a source with the quality is prevented, and the creature cannot be enchanted or equipped, blocked or targeted by such a source. Three sources could supply the examples, and per-rule rebalancing decides how many each quality gets:

- **Rule-labelled copies.** A copy of a real record grants a creature a protection quality and relabels the four outcomes by the rule, as the lethality copies relabel a death: the damage prevented, the block or target made illegal, the Aura put into the graveyard.
- **Variant scripts.** `collect-variants` already swaps selectors from a whitelist. A protection keyword's quality is such a selector, so variants of real cards with another quality are played for real.
- **Interventional forks.** A fork grants protection mid-game and plays the turn out, so the outcomes come from Forge rather than from a rule written beside it.

Gen-2's evasion grid becomes the measure: its protection rows extend to the other qualities, from a creature type to everything.

## Combat could get lethality copies of its own, as damage spells have in gen-2

**Gen-2's lethality copies cover damage spells only, and combat is now the weaker of the two.** A copy redraws a qualifying record's damage amount and the hit creature's toughness, and labels the death by the rule rather than by what Forge did. The gen-2 record's combat grids show where combat stands without them. Before the copies, the model separated a lethal block from a non-lethal one better than a lethal spell from a non-lethal one. With the copies and a tenth of the learning rate, the spell grid's error falls below every combat grid's. The combat boundary stays soft on every checkpoint, the attacker's death sits further from the rule than the blocker's, and a first striker is predicted to die as often as a creature without the keyword.

**A combat copy would redraw the numbers of a qualifying block and label every death by the rule.** It would set one creature's power and the other's toughness, label both creatures' deaths and their damage taken by rules 510 and 702, and could grant a damage-step keyword with the step marker set to match. The first-strike step makes 6% of combat records, and blocked fights are a part of those. A keyword copy could show a first striker surviving its blocker as often as training needs.

**Whether gen-3 adopts them depends on what the rest of gen-3 already fixes.** Four items above change the same outcomes: the snapshot taken before the damage step, the block pairing in the input, the whole first-strike step collected, and training that draws more of the records a weak field needs. Combat copies are worth their cost only for what remains after those. Three questions to settle with them:

- Which blocks qualify: single blocks only, or several blockers once the pairing is in the input, with damage assignment order and trample.
- Whether rule labels or Forge's outcome is the target. Gen-2's spell copies chose the rule. Combat copies would make the same choice, unless a rule-by-rule coverage plan settles it differently.
- How many copies, and how they share an epoch with the field-weighted sampling.

The combat grid measures the result on blocks from the validation strata, which copies never touch.

## Outcome

To be filled in once gen-3 is designed and run.
