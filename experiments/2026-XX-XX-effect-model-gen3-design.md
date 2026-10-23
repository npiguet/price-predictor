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
unmet. The rare-wording section shows that a rule's cells have to reach down to the kind of ability
that exercises it, since an activated ping is rare where a damage spell is not. Beyond the schema and
that goal, nothing here is decided yet.

A few terms recur. A record is one observed game event: an ability resolving, a combat, a trigger, a
legality decision. A resolution record comes in two halves joined by a link, the cost half and the
effect half. A record names the ability that acted by its provenance key, the script file, face,
trait kind and index of the Forge trait. A sidecar is the file beside each converted card that maps
those keys to the card's rendered ability lines. The random seat is a player that makes a random
legal choice at a share of its decisions instead of the Forge AI's. A probe fork is a copy of the
game, taken at the combat damage step, in which one keyword is stripped from one creature to see
what changes.

The record has three parts. Part 1 lists what gen-2 leaves open, why each gap matters, and how
gen-3 plans to close it. Part 2 records each step taken to close the gaps, what it looked for and
what it found, and the steps still to take. Part 3, the outcome, records whether gen-3 worked,
what still needs changing, and the improvements its results suggest.

## Part 1: what gen-2 leaves open, and how gen-3 plans to fix it

### Collection can lose or misplace events without leaving a trace

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

### The Forge hooks and collectors recorded planned events as real ones, and several outcomes reached the wrong record

Gen-3's hooks and collectors fix all nine defects the gen-2 record describes ([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#nine-collector-defects-shaped-gen-2s-labels-and-gen-3-collects-without-them)). The fixes are commits on the `effect-record-hooks` branch of `../forge` and in the connector. They change what a record says, so they are in place before gen-3's first collection.

**Block and attack legality are asked by Forge's declare steps, before anyone declares.** The hook lists every creature the player controls as a candidate, and the combat holds no blocks yet when it is asked. A creature planned for one block can no longer read as forbidden for another. A tapped or "can't block" creature is always listed as forbidden. The answer is the same whatever controls the seat, so the random seat no longer writes legality records of its own, and the AI's internal legality queries are no longer hooked. Every legality record is therefore a real declaration, one per defender or attacker per combat. `--legality-rate`, which sampled the AI's what-if queries, now has nothing to sample; removing the flag is left to the gen-3 collection work.

**A reveal from a library during a resolution names the players whose libraries it came from.** A scry already names the player who scried in the same way. A reveal from a hand, a graveyard or any other zone names nobody. So does a reveal made while paying a cost, which happens before the spell it pays for resolves. A clash names both clashing players.

**Each kind of object an effect makes gets its own note, and a token's note carries its keywords.** Bestial Menace now records a Snake, a Wolf and an Elephant. The collector writes a token's script id and keywords with the code that writes every token on the board, so the created-objects field's keyword flags have something to learn from.

**Prevented damage names the creature or player it would have been dealt to.**

**A note that names no subject marks nothing affected.** Coin flips, token creation, copies, votes and ventures no longer mark the acting card as affected.

**A combat record carries its own damage step's assignment, as the players made it.** Forge copies the assignment the moment it is complete, before prevention or redirection rewrites the damage, and holds it until the next step's assignment replaces it. The collector writes that copy whole. A double striker therefore shows its assignment in both steps, a prevented hit still shows the damage assigned, and the observed record and its probe fork read the same thing.

**A combat record stops where the stack starts.** The first ability or spell to go on the stack in a damage step writes the step's combat record, and nothing after it joins one. A combat-damage trigger's draw, token or discard is written in the trigger's own resolution record, and damage a spell deals in the window is not combat damage.

**Trigger records describe only events that happen.** The engine's query of which triggers an event would fire no longer notifies the hook, so the AI's questions about taps it is planning write nothing. A static trigger is also evaluated once per event rather than once when its event is collected and again when it runs.

**A prevention that prevented nothing writes no note.** A shield whose amount is used up no longer records a prevention of 0.

**Two bits of the `library_events` field are still never set.** The target mapping writes the `scry` bit for both scry and surveil, and nothing writes the `surveil` or `tutor` bit. The mapping is in `src/effects/domain/effect_targets.py` and is left to the gen-3 target work.

### Some records cannot be joined to the text of the ability that acted

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

### The random seat can repeat a free activation that changes nothing

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

### A random seat in more matches is an option, to be decided by what it adds per rule

**Gen-3 could put a random seat in far more matches than gen-2's one in four.** One option is a
random seat in every match, acting at random at half its decisions. Gen-2's second pilot found that
a random seat does not lengthen a game: its games ended on a median turn of 18, against 19 for games
between two Forge AIs. Its records hold the outcomes Forge's AI never chooses, such as Bolt aimed at
an 8/8 or a block the AI would not make. Every `can_play = false` in the corpus comes from it,
because the AI records only the candidates that already passed its own legality check.

**Five things count against a random seat in every match.**

- **Random choices vary the outcome, not the rule.** Most rules the corpus lacks need a card or a
  board that self-play rarely deals: two legends, a Saga, a battle, an anthem beside a −1/−1
  effect. A random seat changes what happens to the cards on the table and brings no new ones.
  Coverage decks and directed forks are what bring them.
- **Random-seat boards are thinner.** In the second pilot they held 4.40 creatures per record,
  against 5.62 in games between two Forge AIs, because random attacks and blocks lose creatures.
  The rules that need crowded boards, such as layers, the legend rule and combats with several
  blockers, would grow rarer.
- **The Forge AI plays differently against a weak opponent.** It races and wins early, and enters
  fewer complex combats. In gen-2 that applied to a quarter of the corpus. In every match it applies
  to all of it.
- **The validation strata would have no games between two Forge AIs.** Nothing would then measure
  the model on the boards Forge's own play produces, the comparison gen-2's random-seat slice is
  scored against.
- **Wasted records and the looping play grow with the share.** Random targets on a wide board
  mostly produce a resolution that changes nothing, which gen-2 accepted as a limit. The free
  activation repeated about 2,190 times in one turn, in the section above, has to be fixed first.

**The per-rule detectors decide the share.** Gen-2's records carry the random-seat flag, so the
detectors run over gen-2's corpus can count each rule per game in random-seat games and in games
between two Forge AIs. A rule found mostly in random-seat games argues for a larger share, and one
found only in rich boards argues against. Whatever share is chosen, the validation strata keep
games between two Forge AIs.

**A softer random seat is a second option.** Instead of drawing uniformly at half its decisions,
the seat would choose among the Forge AI's better options with some temperature. Its play would
stay plausible but differ from the AI's, without throwing creatures away. It depends on whether
Forge's AI exposes a ranking of its options, which has not been checked.

### The held-out coverage round over-represents some texts, and what-if legality sampling has nothing left to sample

**The held-out coverage round covers more than the held-out texts.** `collect-coverage
--only-cards` targets every text carried only by held-out cards. That set is larger than the
held-out texts themselves.

**What-if legality queries no longer reach the corpus.** A what-if query is the AI asking whether
an attack or block would be legal without committing to it. In gen-2, `--legality-rate` drew each
query before identical snapshots were merged, so a question the AI asked K times survived with
probability 1 − (1 − r)^K, and blocker what-ifs outnumbered real block decisions about five to one.
Gen-3's legality hook is asked by Forge's declare steps rather than by the AI, so every legality
record is a real declaration and there is nothing left to sample
([the Forge hooks](#the-forge-hooks-and-collectors-recorded-planned-events-as-real-ones-and-several-outcomes-reached-the-wrong-record)).

### The observed-mana probe counts the mana each event names

**The knowledge probes' observed-mana target sums each event's `mana_by_color`.** In gen-2 it read
`params.amount`, a field the collector never writes, so every event counted as one mana whatever it
produced ([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#the-observed-mana-probe-label-counted-mana-events-rather-than-mana)).

**The collector's map is the mana actually produced, not the colours a land could make.** The
collector counts the colour letters of the mana string Forge reports once the mana is in the pool,
after replacement effects (`PatchedCollectors.manaByColor`). Every multi-colour map in a sample of
gen-2 events came from a source that makes two or more mana at once, such as Orzhov Basilica's
`{W: 1, B: 1}`.

**An event whose map is empty counts as unreadable.** It adds nothing to the target, and a record
whose mana events are all empty gets no label. The collector writes no map at all when it cannot
read the mana, so the empty maps seen in gen-2's events are not yet explained, and gen-3's first
pilot should count them.

### Training could draw more of the records a weak output field needs, from the whole corpus

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

### A record drawn for a rule trains only that rule's outputs

**A record drawn extra times for one rule also repeats every other label it carries.** A trample
combat drawn three extra times to balance rule 702.19 also teaches damage taken, life change and
the gate three extra times. None of those outputs was short of records. Rebalancing by rule would
therefore unbalance the outputs it did not target. Multi-label learning calls this label
co-occurrence: oversampling an example for a rare label oversamples the common labels it carries.

**Gen-3 starts with a hard mask: a record drawn for a rule trains that rule's outputs and nothing
else.** The heads map names each rule's outputs, existing and new, so the mask needs no labelling
by hand. The rebalancer knows which rule it drew a record for once the detectors tag each record
with the rules it shows. A record drawn at its natural share, for no rule, keeps every output.
Gen-2 already has the mechanism: the loss skips a field the record says nothing about, and the
verdict head takes a mask per column. Fine-tuning a language model uses the same technique when it
leaves the prompt out of the loss.

**The gate stays on for the rule's own entities.** When a rule's effect does not happen, the model
usually learns it as an entity left unaffected. A creature that survives Shock has no event, and
its only label is the gate at zero. Masking the gate would remove the negative half of every
minimal pair.

**The mask removes the bias in the labels, not in the boards.** The trunk still reads the boards a
rule's copies bring, such as many infect boards when infect is rebalanced. That skews what the
model reads, not what it is told the answer is, and training on varied boards limits it. The
rule's own outputs still need the importance weight of the calibration constraint above.

**A soft mask replaces the hard one if the hard mask is shown to cost the other outputs.** Under
the soft mask, a record drawn k times keeps its other outputs at weight 1/k on each copy. Its
copies then add up to one natural example for those outputs, and no correct label is thrown away.
The hard mask is the soft mask with that weight at zero.

**The cost to look for is in records that exist only for one rule.** A record from a directed fork
is collected for its rule alone, so under the hard mask its other labels never train anything. The
sign is an output the mask switches off on many records staying weak. The test is two arms on the
same corpus, one per mask. If the soft-mask arm scores the switched-off outputs better on the
card-disjoint validation while the rule tests score the same, the hard mask is costing them data,
and the soft mask replaces it.

**A thresholded loss on the extra copies is a third stage, tried only if the soft mask still pulls
the other outputs.** A copy's other outputs then train only where the model is clearly wrong. Below
a threshold of error they get no gradient, and above it a small one. A duplicate then stops pushing
its labels once the model has learned them, which is the pressure oversampling adds. It still
corrects the model where the board shows something the model gets badly wrong. The established
forms are the dead-zone losses: the hinge loss, and the ε-insensitive loss of support vector
regression. Focal loss is the smooth form. It scales the loss by (1 − p)^γ, where p is the
probability the model gives the correct answer.

**The threshold applies to the extra copies only.** The natural copy of each record keeps the full
loss, so the other outputs' calibration is still set by records at their natural share. A dead
zone on a probability would otherwise teach the model to stop at the threshold, and focal loss is
known to shift calibration. A record collected for one rule alone has no natural copy. Whether its
other outputs get the threshold or one pass at full loss is left to this stage.

**The thresholded loss has settings the soft mask does not.** Each output type needs its own
meaning of clearly wrong: a probability for binary and categorical outputs, a tolerance for counts
and deltas. A base weight comes on top. Two arms with different thresholds differ in what each
output effectively trained on, which makes them harder to compare.

### A rare wording of a common effect needs examples of its own, and field weighting cannot see it

**Gen-2's arms doubt that an activated ping's target is affected, because training holds few such
pings.** Prodigal Sorcerer's "{T}: deal 1 damage to any target" and its relatives resolve about 70
times in gen-2's whole training corpus, about 45 of them at a creature. "Any target" damage spells
resolve about seven times as often. On arm `e64-lethality`, run B64's last epoch and arm
`e256-lethality`, the gate at the pinged creature sits near an even chance; the gate is the model's
predicted chance that a card is affected at all. Reading the same records with "target creature" in
place of "any target" lifts it above nine in ten. The model therefore knows the effect and has not
tied this wording to its target. The last epochs of runs B128 and B256 learn it, so whether a
checkpoint does is the luck of its fit ([the gen-2
record](2026-09-18-effect-model-gen2-improvements-design.md#where-the-model-doubts-that-a-hit-creature-is-affected-the-text-leaves-the-victim-open-or-is-rare-in-training)).

**Field weighting does not reach it.** The gate removes about three quarters of its constant
prediction's loss over the whole corpus and stays level after epoch 20, so a controller watching
each field's loss has no reason to raise it. The weak spot is one wording in
one situation, an activated "any target" ability whose target is a creature, and the field's loss
averages it away. Three parts of gen-3 could supply it:

- **Per-rule rebalancing, with cells that cross a rule with the kind of ability exercising it.** Rule
  115, targets, crossed with activated abilities would give these pings a quota of their own.
- **Retargeting copies.** Gen-2's lethality copies already copy most qualifying pings, but each copy
  keeps its record's board and target, so the gate sees the same 45 boards over and over. A copy that
  hands the ping to another creature on the board, and labels the new target's damage and death by
  the rule, would show the gate new boards from the same records.
- **Collection aimed at the wording.** A coverage round over cards with an activated "any target"
  damage ability would add creature targets beyond what the AI's own choices supply.

**The measure is gen-2's toughness sweep.** Its activated damage records should give the same gate
read with "any target" and with "target creature". Probes that pick training records should also drop
repeated rows: the curated corpus writes a rare text's records several times, and gen-2's boundary
chart counted three Suffocating Blast resolutions as twelve of its 40 rows.

### The Comprehensive Rules could index what the corpus covers, rule by rule

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

**Illustrating a rule draws on three sources, in order of preference.** A situation found in a
real game comes with real decks and real decisions around it. A situation built on purpose comes
with whatever the builder chose.

1. **The collected corpus.** The detector that counts a rule also finds the records that
   illustrate it, with the rule's effect happening and with a close case in which it does not.
2. **Directed forks of live games.** Coverage rounds target cards, and a rare interaction between
   rules may never occur in a sealed game by itself. Gen-2's forks already change a live game: an
   interventional resolution forces an ability onto the stack, and a damage-step probe strips one
   keyword from one creature. Directed by rule, a fork would wait for a board that meets the rule's
   precondition and change the one fact that flips its outcome. The real record and the forked one
   are then a minimal pair.
3. **Puzzle-style starting states, as a last resort.** Forge ships 372 puzzles under
   `forge-gui/res/puzzle/`, each a starting game state: life totals, the active player and phase,
   and every zone's cards with their links. Written for a rule, such a state is a unit test the
   collector could play. It serves only a rule that neither the corpus nor a directed fork reaches.

**The rule tests that judge the model must stay out of its training.** If the same situations both
balance the corpus and measure the model, they stop measuring generalization. A frozen set of
rule tests, drawn from games kept out of training, would join the knowledge probes as an
evaluation. Collection targeted at a rule would feed training separately.

**Rules could also be the pools of the adaptive sampling above.** A per-rule detector gives each
record the rules it exercises, which is the same index that sampling by field needs. The two items
would share the decision between pools, an index and a database.

### A rule is learnable only when the model reads what decides it and an output is trained on its outcome

**A record that shows a rule does not make the model learn it.** Two more things have to hold. The
model's input has to carry every fact that decides the outcome. One of its outputs has to be
trained on the outcome. Gen-2 misses one or the other in places the rules care about.

**Some outcomes are recorded and never predicted.** A spell's resolution outcome (resolved,
fizzled, partially fizzled) is written on the record's cost half. The model receives it as an
input on the acting ability's slot, and a fizzled spell writes no effect half at all. The model is
therefore told that Red Elemental Blast fizzled and never asked to predict it. Several event types
reach no output either: damage prevented, a regeneration, a player losing the game, an extra turn,
day becoming night. Each still marks the entity as affected, so the model learns that something
happened to it, not what.

**Some deciding facts are recorded and never read.** The input leaves out subtypes, so "Goblins
you control get +1/+1" cannot tell a Goblin from an Elf except through the ability text. It
leaves out the static, triggered and replacement lines of every permanent, as the snapshot section
below describes. It reads that a permanent is attached to something but not to what, and that a
creature blocks but not which attacker.

**State that gen-2 does not record needs an input slot and an output as well as a record field.**
Recording the monarch, day and night or a dungeon's room lets a detector count the rules that use
them. The model also needs a slot to read each one, and an output to predict when it changes.

**Gen-3 maps every rule to its input and output before changing either.** For each mechanical and
legality rule the map says whether an existing output is trained on the outcome, whether a record
carries the outcome with no output trained on it, or whether nothing carries it. It also lists the
deciding facts the model does not read. The new outputs and input slots then follow from how many
rules each would serve, and they change together with the record schema.

### The snapshot should list every line of a permanent, not only its spell and activated abilities

**The model never reads a permanent's statics, triggers or replacements.** `SnapshotBuilder.printedKeys` fills a permanent's card-slot lines from Forge's `getSpellAbilities()`, which holds the cast spell and the activated abilities. Attachment-granted lines come from the same call on each attachment. About a quarter of the permanents on the battlefield carry a line this leaves out, and the knowledge-probes record gives the counts. Keyword lines are unaffected, because the keyword flags carry them.

**What a static has already done reaches the model, and what it decides at the moment of the record does not.** An anthem's +1/+1 is in the toughness and a granted keyword is in the flags. A "can't block" static, a replacement that exiles instead of destroying, a damage doubler, or a lord whose removal shrinks its creatures are text the model needs and never sees. Pacifism shows it an enchantment attached to a creature and nothing more.

**The fix is in the collector, so gen-2 keeps its corpus and measures the cost instead.** `printedKeys` and `grantedAttached` would add the card's live triggers, static abilities and replacement effects, through `getTriggers()`, `getStaticAbilities()` and `getReplacementEffects()`. Reading Forge's live lists respects a creature that has lost its abilities, which reading the card's sidecar at training time would not. Abilities a permanent gains from another permanent's static, such as a lord granting a mana ability, still appear nowhere, and are worth listing in the same change. A board-heavy record then encodes more texts per step, so training slows somewhat.

**Gen-2's board-abilities ablation is the baseline.** It scores each gen-2 arm on legality restricted by a non-keyword static and on resolutions with a replacement on another permanent, beside controls of the same kinds. A gen-3 model that reads the new lines should close those subsets' loss toward their controls, and noise in its card slots should start to cost something there.

### Early stopping could read a smoothed loss in which every output field counts

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

### The summed validation loss disagrees with the probes over training, and gen-3 judges checkpoints by probes and rule scenarios

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

### The training loss puts every output field on one scale, with amount distributions that fit

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

### The model has to learn every rule that depends on whose turn or which step it is, and collection records which check decided each one

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

### A combat record's snapshot should be taken before the damage step deals any damage

**A combat record's snapshot already shows the damage of the first creature or player hit.** `BusBracketCollector.openCombatBracket` takes the snapshot when the step's first damage event arrives, from `onCardDamaged` or `onPlayerDamaged`. By then Forge has marked that first damage, so the creature it hit shows it as marked damage, and a player hit first shows the lower life total. In 26 of the 40 one-on-one blocks gen-2's combat grid reads from the validation strata, the blocker's marked damage equals the attacker's power. The attacker never shows damage in those records.

**The model can read part of the answer off the board.** Gen-2's input gives each card a thermometer of toughness minus marked damage, so a blocker the attacker already damaged shows whether that damage is lethal. A combat record then asks about a board on which half the step has happened, and how much of the outcome leaks depends on which damage Forge dealt first.

**The fix is to take the snapshot at the start of the damage step.** The damage-step probe forks already copy the game at the phase event, before any damage, so the observed record can take its snapshot at the same moment. The combat shape is read at the first damage event, as now, because no creature has left combat by then. The assignment can be read at any point of the step, because Forge holds each step's assignment as it was made.

### The board input should say which attacker each blocker blocks

**A card slot says that a creature is blocking, not which attacker it blocks.** The snapshot records the pairing in each creature's combat status, `blocking` and `blocked_by`, but the card features keep only a blocking bit and a count of blockers. With two blocked attackers on the board, the model cannot tell which blocker deals damage to which attacker, so a combat's deaths depend on a pairing it has to guess. Gen-2's combat grid reads only combats with one blocked attacker for this reason.

**The gap reaches abilities as well as combat damage.** Skewer Slinger deals 1 damage to the creature it blocks or is blocked by. Both of gen-2's probe records for it have several creatures in combat, and run B256's last epoch gives the damaged creature a gate of 0.13 and 0.02 ([the gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#where-the-model-doubts-that-a-hit-creature-is-affected-the-text-leaves-the-victim-open-or-is-rare-in-training)).

**The pairing belongs in the trunk's input.** Two forms would carry it: each card slot gains the board position of the attacker it blocks, or the trunk's attention gains a bias between a blocker and its attacker, as it could between an aura and the permanent it enchants. Either needs no new collection, since the snapshot already holds the pairing.

### The keyword flags should say what a creature has protection from

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

### Combat could get lethality copies of its own, as damage spells have in gen-2

**Gen-2's lethality copies cover damage spells only, and combat is now the weaker of the two.** A copy redraws a qualifying record's damage amount and the hit creature's toughness, and labels the death by the rule rather than by what Forge did. The gen-2 record's combat grids show where combat stands without them. Before the copies, the model separated a lethal block from a non-lethal one better than a lethal spell from a non-lethal one. With the copies and a tenth of the learning rate, the spell grid's error falls below every combat grid's. The combat boundary stays soft on every checkpoint, the attacker's death sits further from the rule than the blocker's, and a first striker is predicted to die as often as a creature without the keyword.

**A combat copy would redraw the numbers of a qualifying block and label every death by the rule.** It would set one creature's power and the other's toughness, label both creatures' deaths and their damage taken by rules 510 and 702, and could grant a damage-step keyword with the step marker set to match. The first-strike step makes 6% of combat records, and blocked fights are a part of those. A keyword copy could show a first striker surviving its blocker as often as training needs.

**Whether gen-3 adopts them depends on what the rest of gen-3 already fixes.** Four items above change the same outcomes: the snapshot taken before the damage step, the block pairing in the input, the whole first-strike step collected, and training that draws more of the records a weak field needs. Combat copies are worth their cost only for what remains after those. Three questions to settle with them:

- Which blocks qualify: single blocks only, or several blockers once the pairing is in the input, with damage assignment order and trample.
- Whether rule labels or Forge's outcome is the target. Gen-2's spell copies chose the rule. Combat copies would make the same choice, unless a rule-by-rule coverage plan settles it differently.
- How many copies, and how they share an epoch with the field-weighted sampling.

The combat grid measures the result on blocks from the validation strata, which copies never touch.

### The withheld keyword reaches training as its definition, so gen-3's zero-shot slice is the first that tests reading one

**Every gen-2 arm deleted cascade from its input rather than expanding it.** `--withhold-keyword` is meant to keep the keyword's token out of training and replace each occurrence with its reminder text. The batcher removed the token before keyword expansion ran, and the flag that should have forced the expansion was never read. A cascade keyword line therefore reached the encoder as its `[CLS]` token alone. Evaluation builds its batches the same way, so the model never read cascade in any form.

**Gen-2's zero-shot slice measured a keyword with no text, not a keyword read through its definition.** The slice compares the effect head's per-field results on records where cascade acts or sits on the board with the same fields for keywords that trained. With cascade's line empty, those records differ from the others by a missing ability rather than by an unfamiliar one. The slice says nothing about how a new set's keyword would be read, and it is not a baseline for gen-3's.

**Gen-3 runs force-expand the withheld keyword wherever it appears.** One tokenizer method expands the withheld keyword as it would a keyword unknown to the vocabulary. It then drops the keyword's token everywhere, including inside the reminder text of another keyword that names it, such as reach's mention of flying. Training, evaluation and `encode-abilities` all tokenize through it. The cache row of the withheld keyword is therefore its definition read by the trained encoder. Gen-2's cache encoded cascade's line from a token training never updated ([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#cascades-line-in-every-gen-2-cache-was-encoded-from-a-token-training-never-updated)). A run without `--withhold-keyword` takes exactly the draws it took before.

### The corpus build, the targets and the probes were reviewed, and gen-3 trains without the defects found

A review of `src/effects` and the probe scripts after the gen-2 runs found fifteen defects that shaped gen-2's results ([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#sixteen-defects-in-the-corpus-build-the-training-labels-the-evaluator-and-the-probes-shaped-gen-2s-results-and-gen-3-is-built-without-them)) and thirteen more on paths gen-2 never ran. All of them are fixed on master before gen-3's first collection. Several fixes change what a field means or how the loss weighs it, so no gen-3 number is comparable with gen-2's on those fields.

#### What the model trains on

**An output field is scored only where its record says something about it.** Each field is supervised where the affected gate fires and the field is present. A field is present when the record writes it, or when the field has a silent value and the record's class supervises it. The silent values are the ones whose absence means nothing happened: zone stayed, no damage, no counters, no P/T change, no type, colour or keyword gained or lost, no control or attachment change, and no change to a player's life, cards, library, poison or mana. `tap_state`, both durations, the legality bits and `min_blockers` have no silent value and train only where an event or a decision payload sets them. Player slots are never scored on permanent fields, nor permanents on player fields. Continuous records now also supervise P/T, keyword, type and colour changes. The constant-prediction floor reads the same presence, so explained deviance measures the same thing the loss trains.

**`target_legal` now has nothing to learn from.** A decision payload lists a spell's legal targets and never its illegal ones, so with presence the field sees only positives and its floor is undefined. Target legality is learned through the affected gate, which a decision record raises exactly on its legal targets, and the probe ladder already reads it there. Gen-3 either drops the field or has the collector record the entities a target restriction excluded.

**A trigger record supervises only whether the trigger fired.** It produces no per-entity targets. The triggering event stays in its input as the pending-event overlay.

**The value head reads static boosts and tells placing counters from removing them.** `AddPower$` and `AddToughness$` are read, and a value that is not a literal number is masked. Counter removal has a target of its own, `counters_removed`, because one chain can both put and remove counters. Moving or adding-or-removing counters masks both targets. Where a script omits an amount, the value is Forge's own default: a draw draws one card, and putting or removing counters moves one.

**A masked number is masked in the numeric input too.** The masked-language-model term zeroes the numeric value and its flag wherever it masks a token.

**Count targets are clamped at zero.** Healing subtracts from `damage_taken`, and a negative count made the Poisson loss unbounded below.

**Every head's loss is reduced the same way.** Each row's loss is summed over its targets and then averaged over rows: a record for the effect, created-objects and verdict heads, a text for the value, API and masked-language-model heads. Before, the API-key term was averaged over its 1,245 key columns, which made it about a thousandth of the API-type term under the same weight. With the sum, the API-key term over 1,245 key columns made up close to a third of the initial loss at gen-2's API weight. The default API weight is therefore 0.003, which puts the weighted key term between the value and masked-language-model terms. At that weight the API-type term carries almost nothing. Separate weights for types and keys belong to the loss redesign ([one scale for every field](#the-training-loss-puts-every-output-field-on-one-scale-with-amount-distributions-that-fit)).

| term, one 32-record batch at initialisation | weighted, API weight 0.05 | weighted, API weight 0.003 |
|---|---:|---:|
| per-entity fields | 67.0 | 67.0 |
| created objects | 30.7 | 30.7 |
| API keys | 44.5 | 2.7 |
| masked-language model | 6.6 | 6.6 |
| value head | 1.0 | 1.0 |
| verdict head | 0.8 | 0.8 |
| API types | 0.3 | 0.02 |

#### The corpus build

**The training plan counts only records the build will train on.** The survey routes each game once, at the end of its shard: held out, else drawn for game-disjoint, else training. Only training games' records size the class, family and text quotas, and the write pass applies the same routing. Rarity keeps counting the whole corpus. The manifest records each class's target beside its delivered count and warns when a class falls short. A seven-shard smoke build delivered 99.8–102.7% of every class's target.

**A held-out text's game cap counts only resolutions.** A held-out game is admitted to the card-disjoint stratum only for the held-out texts it resolves, so every admitted game gives gate 1 a record.

**The holdout covers token and variant scripts.** A token script carrying a held-out text counts as a carrier and is held out by its script file. `collect-variants` drops any variant that prints a held-out text or template. `build-corpus` drops a training record acting through a held-out text, and counts the drops as their own stratum.

**Training shards are dealt by game.** Repacking assigns each game to a shard by a seeded hash of its `game_id` and shuffles each shard, so a shard mixes games from every run and source.

**Three build faults that gen-2 never hit are fixed.** An unjoinable key on a decision's candidate is refused rather than crashing the plan. A curated shard keeps the generation of the raw records it came from. A build that fails partway leaves no manifest whose digest still matches the checkpoints.

#### Encoding and the baselines

**The `no-state` baseline keeps the acting ability.** It zeroes every ability vector on the board except `[ACT]`'s, all card and player features, and every `[GLOBAL]` input except the record-kind, moment and subkind flags.

**The identity baseline gives each text its own row.** Its table is indexed exactly by text, built from the sidecars, with one shared row for texts it never saw. The 2^17-row hash table it replaces put more than a quarter of the texts in a row with another.

**The prose surface works.** Value and API targets are read from the line's script rather than its prose. Lethality copies edit the prose amount and shift the role spans with it. Role spans reach the encoder on the prose surface.

**Training records its seed on the checkpoint, and gradient accumulation counts its steps within the epoch.**

#### Evaluation and probes

**Gate 2 reads every field through the affected gate.** A count or amount is read as the gate's probability times the field's value, and a category as the gate's probability times the class's. Gate 2, the sweeps and the zone probes share this reading. The probe ladder's `damage_taken` target still reads the amount alone, because every one of its items took damage.

**Gate 2 counts records from the curated corpus and leaves out the forks.** It reads the curated game-disjoint stratum and skips probe forks and interventional records. Its 200-record threshold counts distinct records. Agreement is averaged over each record's carriers, because each carrier is its own perturbation.

**Gate 1 takes its seen texts from the curated training stratum.** It streams the card-disjoint stratum and keeps only resolution records of unseen texts. A checkpoint without a curated corpus skips gate 1 and says why.

**The checkpoint names the corpus, sidecars and variant tree that every probe and grid reads.** A probe set frozen against another corpus is refused.

**A probe line is held out only when every card carrying it is held out.** A text printed only on held-out cards never trained, even when the text itself is not held out.

**The embedding probes count each record once.** The profiles de-duplicate the training stratum's copies by `record_id`.

**Three unbuilt checks are reported as not implemented.** The evaluator no longer says the corpus lacks the forks they need.

**`scorer-smoke-test` refuses a scratch directory that already holds vectors.** A rerun in the same directory no longer splices `e` in a second time.

## Part 2: the steps taken

### Step 1, the rules catalogue: every rule is triaged, and every mechanical rule has a scenario not yet checked against Forge

**The rules catalogue is the part of the rules index built so far.** It covers the first and third
items of the plan in part 1: whether a record can show each rule, and which game situations
illustrate it. It went from the rules text to scenarios in three stages: split the rules into
units, triage every rule, then write a scenario for every rule with an outcome. Decisions are out
of scope throughout, because the model predicts what the engine does once they are made. The
catalogue lives in `rules/`. `rules/README.md` gives its formats, `rules/STATUS.md` the per-unit
checklist, and `rules/FINDINGS.md` the detail behind this step.

#### The rules split into 410 units, one per section or per keyword

**The 2026-09-25 edition of the Comprehensive Rules, from magic.wizards.com, holds 3,166 numbered
rules.** The split writes one line per rule, with its text, examples and cross-references, and one
readable file per unit. A unit is one three-digit section, such as 704, state-based actions. Sections
701 and 702, keyword actions and keyword abilities, hold a third of all rules, so there each keyword
is its own unit: 702.2 is deathtouch. A unit is small enough to read whole.

**Chapters 8 and 9 are out of scope without further reading.** They hold the multiplayer rules and
the casual variants, and collection plays two-player games.

#### Triage: half of the rules have an observable outcome, and one in eight of those needs a new record field

**Triage asked of every rule whether the engine decides an observable outcome once the decisions are
made.** Each rule got one of five categories:

- **mechanical:** the engine decides an outcome, such as a zone change, damage, a resolution or
  fizzle, or a trigger firing. Legality rechecked at resolution, as in 608.2b, counts here.
- **legality:** the engine decides whether an action is allowed when it is taken, such as casting
  timing, legal targets or legal blocks.
- **decision:** a choice a player makes, such as ordering triggers or assigning combat damage.
- **definitional:** a term or characteristic with no outcome of its own.
- **out of scope:** cannot occur in a Forge two-player game, or is not game mechanics.

A mechanical or legality rule also got its observability, the gen-2 record kinds that could show it,
and its outcome in one sentence.

**Mechanical and legality rules are about half the document.** Chapter 7, the keyword sections,
holds more than half of the mechanical rules.

| category | rules |
|---|---:|
| mechanical | 1,422 |
| legality | 226 |
| definitional | 775 |
| out of scope | 608 |
| decision | 135 |

Out of scope covers the 302 rules of chapters 8 and 9 and 306 others: game start and mulligans,
deck construction, stickers, the casual-variant card types of chapter 3, ante, and physical card
handling.

**Most mechanical and legality rules are visible in gen-2's record fields, and one in eight needs a
new field.** A direct rule shows in a field such as a resolution's outcome or a playability verdict.
An indirect rule needs a detector over a record's events, such as a state-based death after a
resolution. A rule that needs new fields has an outcome a game shows but no record captures.

| observability, mechanical and legality rules | rules |
|---|---:|
| direct | 708 |
| indirect | 729 |
| needs new fields | 210 |
| none | 1 |

**The rules that need new fields fall into a few families, and game-level designations are the
largest.** Those designations are day and night, the monarch, the initiative, dungeon progress,
speed and the city's blessing. The other families are designations on a permanent, such as renowned,
a soulbond pair or a Room's unlocked halves; characteristics the snapshot does not carry, such as
mana value, the X chosen and an Adventure's face on the stack; which layer applied which effect;
what a linked ability remembers; and what happened earlier in the turn. `rules/FINDINGS.md` maps
each family to its rules.

#### Scenarios: every mechanical or legality rule but 95 has a pair of boards that flips its outcome

**A scenario is a minimal pair, written so that changing one fact flips the rule's outcome.** It
sets a board, lists every decision already taken, and gives at least two variants. The base variant
is the board as set. Each other variant changes exactly one fact, and is labelled by whether the
rule's effect happens, does not happen, or takes a third branch. Grizzly Bears dealt 2 damage by
Shock dies; a Centaur Courser in its place survives. The scenarios follow four rules:

- **Real cards, as plain as possible.** Every card is named as Forge names it, and a vanilla card
  is preferred so that no other ability interferes.
- **Distinct mechanisms get distinct scenarios.** Red Elemental Blast whose target stops being blue
  in response fizzles under 608.2b. Pyroblast aimed at a non-blue spell resolves and does nothing.
  Both look like "nothing happened" and are different rules.
- **Expected outcomes are observable only:** zone moves, life, damage, counters, power and
  toughness, whether a stack object resolved, fizzled or was countered, whether a trigger fired, or
  whether an action was legal.
- **A rule no situation can illustrate is skipped, with the reason.**

Each scenario was then checked against the rules text and against the Forge script of every card
it uses.

**Nearly every scenario pairs a variant in which the rule's effect happens with a close one in which
it does not.** Resolution and playability records carry more than half of the scenarios.

| scenarios by record kind | count |
|---|---:|
| resolution | 636 |
| playability | 414 |
| trigger | 312 |
| continuous | 216 |
| rewrite | 93 |
| combat | 92 |
| all scenarios | 1,763 |

| variants by polarity | count |
|---|---:|
| fires | 1,984 |
| does not fire | 1,960 |
| other, a third branch | 307 |

**Most of the 95 skipped rules describe a situation no Forge card produces.** Two instances of
rampage on one creature and mana produced as a Phyrexian symbol are examples. The rest need a
command zone or a sideboard, or have no observable difference between their cases.

#### No scenario has been checked against Forge yet, and about 75 point at a suspected Forge departure from the rules

**Every expected outcome is a reading of the rules text and of the cards' Forge scripts.** Until
they are compared with what Forge does in the same situation, some of them are wrong at a rate not
yet measured.

**Many scenarios depend on how Forge implements the situation.** 762 scenarios carry a note about
it. Most of these notes flag engine behaviour that reading the scripts could not settle, or
describe how a card is scripted. About 90 need setup state that a starting-state file may not
express, such as a designation or a turn's history. About 75 name a suspected Forge departure from
the rules, such as connive 0 firing the "connives" trigger. `rules/FINDINGS.md` lists ten to test
first.

### Step 2, the heads map: the model can learn under a third of the rules today, and reading every line of a permanent is the input change that adds the most

**The heads map asks of each mechanical and legality rule whether the model reads what decides it
and whether an output is trained on its outcome.** It checks the 1,647 rules against gen-2's input
layout and output heads, as the code builds them. A rule's scenarios make the check concrete. The
difference between a variant in which the rule's effect happens and one in which it does not is
the outcome an output has to be trained on. Each variant's change is a fact the model has to read.
`rules/MODEL-FIELDS.md` lists what the model reads and predicts, and `rules/heads/` holds one line
per rule.

Each rule gets one of four output verdicts:

- **trained:** an existing output is trained on the outcome, as gen-2 collects and curates records.
- **recorded:** a record carries the outcome, but no output is trained on it.
- **missing:** no record carries the outcome.
- **planned:** the gen-3 turn-rule section already plans the output, as a timing or failed-check
  column of the legality verdict.

#### Three quarters of the rules reach an existing output, and under a third are learnable today

**A rule is learnable when an output is trained on its outcome and the model reads every fact that
decides it.** Most rules that reach an output still depend on a fact the model does not read.

| output | inputs complete | inputs incomplete | all |
|---|---:|---:|---:|
| trained | 505 | 753 | 1,258 |
| recorded | 58 | 144 | 202 |
| missing | 26 | 101 | 127 |
| planned | 33 | 27 | 60 |
| all | 622 | 1,025 | 1,647 |

**The zone outcome, the trigger bit and the castability verdict carry the most rules.** The zone
outcome says whether a permanent died, was exiled or returned to hand. The trigger bit says whether
a triggered ability fired. The castability verdict says whether a spell or ability could be played.

| existing output, on trained rules | rules |
|---|---:|
| zone outcome | 378 |
| trigger fired | 319 |
| can play | 245 |
| life change | 146 |
| damage taken | 120 |
| affordable | 116 |
| +1/+1 counters | 89 |
| power change | 88 |
| toughness change | 78 |
| gate: the entity was affected at all | 77 |

#### Reading every line of a permanent is the input that matters most, and gen-3 already plans it

**The model reads only a permanent's spell and activated abilities.** A rule that turns on a
static, triggered or replacement ability of a permanent other than the acting one cannot be learned:
an anthem on another creature, an enchantment that says creatures can't block, a cost reducer. More
than a quarter of the rules need those lines. For 187 of them they are the only gap.

| fact the model does not read | rules needing it | rules for which it is the only gap |
|---|---:|---:|
| every line of a permanent (planned) | 432 | 187 |
| values chosen earlier: a named colour or creature type, X paid, an optional cost paid | 175 | 75 |
| designations and game-level state: day and night, monarch, dungeon room, Class level | 139 | 47 |
| what happened earlier in the turn or the game | 97 | 35 |
| subtypes | 90 | 17 |
| which face is up, and what a card is a copy of | 88 | 35 |
| the objects exiled with a card, and the contents of hidden zones | 68 | 29 |
| card name, mana cost, token script | 59 | 17 |
| pending shields and delayed triggers: a regeneration shield, "the next time" effects | 57 | 36 |
| what a permanent is attached to, what an attacker attacks, what a stack object targets | 54 | 18 |
| timestamps and order | 28 | 6 |
| owner, active player, priority holder (planned) | 24 | 4 |

**241 rules miss a fact the snapshot already records, which needs only a change to the input
layout.** Subtypes, which face is up, what a card is a copy of, its name, what it is attached to,
what an attacker attacks and what a stack object targets are all in the snapshot and left out of
the model's input. "Goblins you control get +1/+1" is the plainest case: the snapshot knows which
creatures are Goblins, and the model does not.

**Values chosen earlier are the largest gap that needs new collection.** The snapshot records the
choices of the acting ability only: its targets, modes and X. It does not keep a choice made
earlier for a permanent on the board, such as the colour Gods Willing named or the creature type
Cavern of Souls named.

#### 389 rules reach no existing output, and timing and game-level state are the largest families

**A fizzle is the plainest case of a recorded outcome no output is trained on.** The record says
that Red Elemental Blast fizzled, the model reads that as an input, and no output predicts it. The
same holds for a player losing the game, a permanent being copied, or day becoming night.

| outcome with no existing output | rules | rules for which it is the only gap |
|---|---:|---:|
| timing and failed checks of a legality verdict (planned) | 76 | 39 |
| designations and game-level state changing | 76 | 8 |
| zone destinations the zone outcome merges | 51 | 4 |
| copies of spells and permanents | 40 | 4 |
| subtypes, abilities and text gained or lost | 38 | 8 |
| a face turning up, transforming or flipping | 33 | 10 |
| costs after reductions and increases | 30 | 3 |
| phasing in, regeneration, prevention, a replacement's result | 28 | 1 |
| permanent spell resolutions | 27 | 1 |
| a player winning or losing | 26 | 20 |
| targets changed, delayed triggers created, libraries shuffled | 26 | 3 |
| removed from combat, became blocked | 20 | 7 |
| a spell fizzling or partly fizzling | 18 | 10 |
| extra and skipped turns and phases | 16 | 6 |
| legal targets and attack targets | 16 | 4 |
| turn-based actions | 15 | 1 |
| subgames and restarts | 15 | 1 |

**Three of these families are gaps in what gen-2 collects and trains on, not in its heads.**

- **A permanent spell's resolution never reaches training.** A creature or other permanent spell
  resolves through Forge's implicit cast ability, which has no line of text. `build-corpus` refuses
  every resolution record whose acting ability maps to no line. The record of a permanent entering
  the battlefield is therefore dropped, and with it the legend rule after a second Isamaru,
  Hound of Konda resolves.
- **Turn-based actions are dropped.** The draw in the draw step, the cleanup discard, untapping and
  a Saga's lore counter happen while no record is open. The collector discards events that arrive
  then, as the first section of part 1 describes.
- **The zone outcome merges destinations.** Any move to a library reads as the top of the library,
  and entering the battlefield from any zone reads as a blink. A card shuffled into its library, put
  on the bottom, or returned from the graveyard to the battlefield is labelled as something else.

### Step 3, the gen-3 fields: 33 new inputs and 22 new outputs make all but 77 rules learnable

**Every name the heads map proposed is merged into one list of fields, each with a slot, a shape
and a source.** The heads map named what each rule lacks one rule at a time, so it used many
names for one thing. A choice remembered by a permanent appeared as chosen values, a linked
choice and stored die results, for example. Each name now maps to one field, to one of the
extensions or fixes, or to a reason it is left out. Each rule's line in `rules/heads/` carries the
fields it needs and whether it is learnable once they are in. `rules/gen3-fields.json` holds the
list and the merge, and `rules/GEN3-FIELDS.md` gives every field's shape and the rules it serves.

| change | count |
|---|---:|
| new inputs | 33 |
| new outputs | 22 |
| existing inputs or outputs extended | 5 |
| already planned in this record | 7 |
| fixes to collection and curation | 3 |

**With every change in, the model can learn three times as many rules as today.** The 77 left
out need something the design leaves out on purpose.

| rules | count |
|---|---:|
| learnable today | 505 |
| learnable with every change above | 1,570 |
| left out | 77 |

#### What rules text describes is read as lines, links are edges, and only state stays numeric

**A field that rules text describes is read through the e vectors of lines, never through a
vocabulary of effect kinds.** Forge itself builds most lasting effects as lines. A prevention
shield, "creatures can't block this turn" or "you may cast that card this turn" is an Effect card
in the command zone, holding the triggers, statics and replacements of the spell that made it. A
floating effect is therefore a card slot in the command zone, read like a permanent, with the e
vectors of the source lines that created it. A dungeon is a card slot too, its rooms its lines.
Only what text cannot say stays numeric: the objects the effect touches, its duration, and the
amount left on a shield that counts down.

**Which of a card's lines are in force is a flag on each line.** A line is active when its face is
up, its Class level is reached or its Room door is unlocked. A spell's kicker or alternative cost
is a flag on the line it was cast with, and a modal spell's choice a flag on the chosen mode.
Temporary grants become lines as well, with a keyword read through its reminder text. Their own
flag keeps them apart from printed lines, which the gate-2 keyword test needs, since it strips a
keyword from the channel that carries it.

**A link between two objects is an edge, not a flag on one end.** What an Aura is attached to,
what an attacker attacks, what a spell targets and what a permanent exiled are each an attention
bias between two slots, as the block-pairing section proposes for blockers. Names are edges too:
the rules only compare names, as the legend rule does, so two slots whose names match are linked,
and no name embedding is learned.

**Four outputs take a shape other than the one the heads map named.** A player losing is one binary
per player, because in a two-player game a win is the other player's loss and a draw is both
losing. A cost reduction and a base power and toughness change are signed deltas, against the
printed cost and the previous base, like the existing power change. The legality of an attack
target needs an attackers record anchored to one attacker, as a blockers record already is,
which is the third fix.

#### The card slot takes the most new inputs, and seven only expose what the snapshot records

**The 33 inputs fall into seven places in the model's input.**

| where | inputs |
|---|---|
| card slot (13) | subtypes, face state, is a copy, is a token, pending-event details, mana cost, printed loyalty and defense, designations, X paid, chosen values, costs paid, history, order rank |
| ability-token flags (1) | active, granted temporarily, cast with, chosen |
| edges (9) | attached to, attacks, targets, source of a stack object, soulbond pair, a battle's protector, exiled with, merged with, same name |
| player slot (5) | speed, city's blessing, rad counters, monarch, initiative |
| global slot (1) | day or night |
| command-zone and library card slots (3) | floating effects, dungeons, library cards an effect touches |
| acting-ability slot (1) | die rolls and coin flips, taken as given |

**Seven inputs need no new collection.** Subtypes, whether a card is a copy, whether it is a token,
the same-name edges, and what a card is attached to, attacks or targets are already in the
snapshot. Face state and pending-event details are partly there.

**Chosen values, line flags and history are the inputs the most rules need.** A rule counts for an
input it needs through another one: every rule that needs floating effects needs every line of a
permanent as well.

| input | rules needing it | rules for which it, with what it requires, is the whole change |
|---|---:|---:|
| chosen values | 119 | 35 |
| line flags | 115 | 24 |
| history | 91 | 29 |
| subtypes | 90 | 14 |
| order rank | 66 | 6 |
| designations | 61 | 15 |
| face state | 57 | 31 |
| floating effects | 57 | 38 |

Every line of a permanent, already planned, still leads all of them: 573 rules need it, and it is
the whole change for 166.

#### The new outputs are each small, and most train on a field the records already carry

**Copies, designation changes and the face a card ends on are the outputs the most rules need.**
No new output serves more than 40 rules. A player losing and a spell fizzling serve fewer, but
for more of their rules they are the whole change.

| output | rules needing it | rules for which it, with what it requires, is the whole change |
|---|---:|---:|
| copies of spells and permanents | 40 | 4 |
| designations changed | 39 | 4 |
| face after a face change | 33 | 8 |
| cost adjustment | 30 | 4 |
| player state changed (monarch, initiative, dungeon, speed) | 27 | 2 |
| game lost | 26 | 20 |
| combat status changed | 20 | 7 |
| resolution outcome | 18 | 10 |

The other 14 outputs serve between 2 and 19 rules each. Two of them, day becoming night and
extra or skipped turns, belong to no card or player and need a global head beside the
created-objects head. The five extensions change existing outputs and inputs. The largest is the
zone outcome, which 64 rules need to tell apart the destinations it now merges. Seventeen of the
22 outputs need no new collection, because the records already carry the event or field they
would be trained on, and three more need it only in part.

#### The 77 rules left out are subgames, banding, casual variants and engine procedure

**No left-out rule needs a field the design meant to keep.** Subgames and the restart of the game
(Shahrazad, Karn Liberated) and controlling another player (Mindslaver) leave out 20 rules.
Attractions, sideboards and other casual mechanics leave out 17, procedure in the middle of an
action, such as the order of pending extra turns, 15, and banding 8. The rest are characteristics
the input already shows with no change to predict, such as a colour derived from a mana cost, and
a few names of one or two rules. A rule can fall in more than one group. `rules/GEN3-FIELDS.md`
lists each with its reason.

### Steps still to take, as the catalogue, the heads map and the gen-3 fields set them

#### Each scenario specifies its rule's detector, and the detectors run over gen-2's corpus first

**A scenario already states what its rule's detector looks for.** It gives the record kind and the
observable outcome. Its contrasting variant gives the near miss the detector must reject. The 729
indirect rules each need a detector, and the 708 direct rules need little more than a field filter.

**Running the detectors over gen-2's corpus comes before any gen-3 collection.** The counts show
which rules self-play already illustrates, with a record in which the rule's effect happens and one
in which it does not. They also show which rules self-play reaches only on one side, and which it
never reaches. Directed forks have to fill the last two groups. The order in which to fill them can
follow how often each rule's mechanic appears in the sets collection draws from.

**Rules could also be the pools of the adaptive sampling in part 1.** A per-rule detector gives
each record the rules it exercises, which is the same index that sampling by field needs. The two
items would share the decision between pools, an index and a database.

#### The scenarios become search patterns and fork instructions, and puzzles are kept for what neither reaches

**Each scenario says what to look for in a game and which single fact to change.** The base variant
is the situation a detector searches for, in the corpus or on a live board during collection. Each
other variant's change is the perturbation a directed fork applies to that board. The real record
and the forked one then form a minimal pair taken from a real game, with real decks around it. The
decisions are the ones the game made, so they need no conversion.

**The variants' changes have to become fork operations.** Each change is an English sentence. A
pass would classify every change into a small set of operations: change a creature's power or
toughness, add or remove a keyword, change a colour or type, add or remove counters, tap or untap a
permanent, change a life total, or replace one card with another. The counts per operation show
which few operations reach the most rules, and so which to build first. Gen-2's damage-step probe
is already one of them: it strips one keyword from one creature through the layer system.

**A directed fork needs a trigger on the live board as well as an operation.** The collector runs
the rule's detector on the board at each phase boundary, the moment interventional forks already
use. When the board meets the rule's precondition, it forks, applies the operation and plays the
fork on. The budgets gen-2 puts on forks still apply: a score check against the live game before
any change, and a cap on forks per real resolution.

**Puzzle-style starting states are the last resort.** They serve only the rules whose preconditions
never arise in a sealed game, even under directed forks. Forge ships 372 puzzles under
`forge-gui/res/puzzle/`, and whether a collecting worker can load one is still unchecked. A puzzle
needs the scenario's decisions written as an action list as well: cast, activate, target, pay,
attack, block, assign damage, pass. Converting scenarios to puzzles waits until the detectors and
directed forks show how many rules are left.

**A record whose outcome differs from its scenario's expected outcome is either a wrong scenario or
a Forge departure from the rules.** A confirmed Forge departure is recorded on the scenario. The
rule's records then stay out of the rule tests until Forge is patched on the hooks branch or the
departure is accepted. Where Forge plays a situation as expected, Forge's outcome and the rule's
outcome are the same. The question the combat-copies section leaves open, rule or Forge outcome as
the label, therefore matters only for the rules that disagree.

**The rule tests that judge the model come from games outside training.** If the same situations
both balance the corpus and measure the model, they stop measuring generalization. A rule's test
records are taken from games placed in the validation strata, as the knowledge probes are, and
split into a selection half and a reporting half as the stopping section describes. Records found
in real games carry the cards those games used. The catalogue's scenarios lean on Grizzly Bears in
about half of them, so a test built from them would test the rule for Grizzly Bears only.

#### A global designation block and a few per-card fields would record more than half of the rules gen-2 cannot see

**More than half of the 210 rules that need new fields would be recorded by one global block and a
handful of per-card fields.** Several of the designations come from sets that sealed pools can draw
from: day and night from Innistrad: Midnight Hunt, dungeons from Adventures in the Forgotten Realms,
the city's blessing from Rivals of Ixalan.

- **The global block** holds day or night, the monarch, the initiative holder, each player's dungeon
  and room, speed, the city's blessing and enduring stories. It also holds per-turn counts the
  "this turn" conditions read: spells cast, permanents put into a graveyard, and abilities each
  permanent activated.
- **Each permanent** gains its designations as a set (renowned, solved, prepared, and so
  on), the creature it is paired with, a battle's protector, a Class's level, a Room's unlocked
  halves, loyalty, and the objects exiled with it.
- **Each card on the stack and battlefield** gains its mana value, the X chosen, the face or
  alternative characteristics in force, and base power and toughness beside current.

**Layer attribution, subgames, restarts and control of another player are not worth collecting.**
The `continuous` record already shows the result of every layer, and the 613 scenarios test results
rather than order. A model that predicts those results has learned what the order decides, so rule
613 needs no new field and no detector. Subgames, the Karn Liberated restart and control of another
player are rare enough to leave out.

#### The model's inputs and outputs change in the order the heads map sets, cheapest first

**The list runs from the change that needs the least work to the one that needs the most.**

1. **Every line of a permanent.** It is already planned, and it alone makes the most rules learnable.
2. **The facts the snapshot already records.** Subtypes, faces, copy sources, names, attachments,
   attack targets and stack targets need input slots and no new collection.
3. **The collection and curation gaps.** Permanent-spell resolutions have to survive
   `build-corpus`, and turn-based actions need a record of their own. Both then feed outputs that
   already exist.
4. **Outputs for outcomes the records already carry.** The resolution outcome, a player winning
   or losing, copies, phasing, regeneration, prevention, and extra and skipped turns are each a new
   output over a record field or event that exists. The zone outcome gains the destinations it now
   merges.
5. **New collection with its own inputs and outputs.** The global designation block and the
   per-card fields above, values chosen earlier, the objects exiled with a card, pending shields and
   what happened earlier in the turn each need a record field, an input slot, and an output wherever
   they can change.

**Running the map again over each new layout measures what the change bought.** The map is a pure
reading of the layout and the scenarios, so a rerun counts the rules each change made learnable.

## Part 3: outcome

To be filled in once gen-3 is collected, trained and evaluated: whether it closed the gaps part 1
lists, what still needs changing for it to work, and the improvements its results suggest.
