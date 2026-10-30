# Ability effect model — gen-3 (draft)

## TL;DR

Gen-2's model can reach a good loss without learning many rules of the game: its corpus is balanced
by card text and rule family rather than by rule, its collectors lost or mislabelled some events,
and its input and outputs leave out what decides many rules. Gen-3 plans to collect a new corpus
without gen-2's defects, rebalance training rule by rule, read every line of a permanent tied to its
card, and judge checkpoints by probes and rule scenarios. A catalogue of the Comprehensive Rules finds 1,647 rules
with an outcome the engine decides, and gives all but 95 of them a minimal pair of boards. Checked
against the model, only 505 of those rules are learnable today. 33 new inputs, 22 new outputs and
three collection fixes, with the items already planned, would make 1,570 learnable. No gen-3 corpus is collected yet, and no
scenario has been checked against Forge.

## Background

Gen-3 is the third generation of the ability effect model. Gen-2 is described in
[`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md),
called the gen-2 record below. Gen-3 collects its corpus from scratch and keeps no gen-1 or gen-2
record. Its record schema is therefore a new version: a field can change meaning or go away, and no
gen-3 reader needs to read a gen-2 shard.

A few terms recur. A record is one observed game event: an ability resolving, a combat, a trigger, a
legality decision. A resolution record comes in two halves joined by a link, the cost half and the
effect half. A record names the ability that acted by its provenance key, the script file, face,
trait kind and index of the Forge trait. A sidecar is the file beside each converted card that maps
those keys to the card's rendered ability lines. An output field is one quantity the effect head
predicts for a card or a player, such as damage taken or life change, and the gate is the head's
predicted chance that a card is affected at all. The random seat is a player that makes a random
legal choice at a share of its decisions instead of the Forge AI's. A probe fork is a copy of the
game, taken at the combat damage step, in which one keyword is stripped from one creature to see
what changes.

The subsections below give gen-2's gaps, the most important first. Gen-2 counts coverage by card
text, not by the rules of the game, and its model cannot learn many rules even where a record shows
them. Training, early stopping and the loss each have a gap of their own. The rules that depend on
whose turn it is, and several combat and keyword rules, are recorded or read wrongly. The last
subsections hold the collector and build defects, which gen-3 fixes before its first game.

### Gen-2 balances its corpus by card text and rule family, so whether a rule of the game is covered is never counted

**Coverage is measured by text and by rule family, not by the rules of the game.** A rule family is
the script-level class `build-corpus` balances on, such as an ability's API type or a trigger's
mode. Whether the corpus shows the legend rule, a creature dying at toughness 0, or two continuous
effects applied in layer order is never counted. Coverage rounds target cards, and a rare
interaction between rules may never occur in a sealed game at all.

**The rules run to thousands of numbered subrules, and many can never appear in a record.** Deck
construction, mulligans, multiplayer variants and most priority bookkeeping have no counterpart in
an effect record. How much of the document a record can show, and how much of that the corpus does
show, is unknown.

### A record that shows a rule does not make the model learn it

**Some outcomes are recorded and never predicted.** A spell's resolution outcome (resolved,
fizzled, partially fizzled) is written on the record's cost half. The model receives it as an
input on the acting ability's slot, and a fizzled spell writes no effect half at all. The model is
therefore told that Red Elemental Blast fizzled and never asked to predict it. Several event types
reach no output either: damage prevented, a regeneration, a player losing the game, an extra turn,
day becoming night. Each still marks the entity as affected, so the model learns that something
happened to it, not what.

**Some deciding facts are recorded and never read.** The input leaves out subtypes, so "Goblins
you control get +1/+1" cannot tell a Goblin from an Elf except through the ability text. It reads
that a permanent is attached to something but not to what, and that a creature blocks but not
which attacker.

**State that gen-2 does not record needs an input slot and an output as well as a record field.**
The monarch, day and night or a dungeon's room appear in no record. Recording them would let a
detector count the rules that use them. The model would also need a slot to read each one, and an
output to predict when it changes.

### The model never reads a permanent's statics, triggers or replacements

**`SnapshotBuilder.printedKeys` fills a permanent's card-slot lines from Forge's
`getSpellAbilities()`, which holds the cast spell and the activated abilities.** Attachment-granted
lines come from the same call on each attachment. About a quarter of the permanents on the
battlefield carry a line this leaves out, and the knowledge-probes record gives the counts. A
keyword's own statics and triggers are left out with them
([keywords](#keywords-reach-the-model-only-as-flags-so-it-infers-what-each-one-does-from-outcomes-alone)).

**What a static has already done reaches the model, and what it decides at the moment of the
record does not.** An anthem's +1/+1 is in the toughness and a granted keyword is in the flags. A
"can't block" static, a replacement that exiles instead of destroying, a damage doubler, or a lord
whose removal shrinks its creatures are text the model needs and never sees. Pacifism shows it an
enchantment attached to a creature and nothing more.

### The model cannot tell which card an ability line sits on, nor what a link between two cards points to

**An ability slot carries no trace of its card.** The trunk reads the board as one sequence: a
`[CARD]` slot for each entity, followed by an ability slot for each of its lines. An ability slot is
the sum of its `e`, an embedding of its slot kind, a position embedding and the charm-mode flag.
Position ids restart at 0 at each `[CARD]`, so the first line of every card has position 1.
Attention knows no order beyond those ids. The trunk therefore reads the lines on the board as one
unordered set, and cannot tell whether lifelink sits on the 2/2 or on the 5/5. The slot's entity id
is kept to key the targets and never reaches the trunk. The specification asks for the reset and
nothing else, and the code does what it asks.

**The gen-2 probes found the board's lines unused, and this explains it.** Replacing the `e` in the card blocks' ability
slots with noise costs nothing measurable on any gen-1 or gen-2 checkpoint ([probes
record](2026-09-19-effect-knowledge-probes-design.md#the-trunk-relies-on-the-acting-abilitys-e-and-replacing-the-e-of-the-abilities-on-the-board-costs-nothing-as-expected)).
A model that cannot place a line on its card has little use for the line. Gen-3's plan to list every
line of a permanent rests on fixing this.

**Several links are recorded with one end only.** A card slot says that the card is attached,
blocking or blocked, and a stack object says how many targets it has, but neither says which slot
is at the other end. Only the links to the acting ability have both ends, as a target bit and a
source bit on each card, because a record has one `[ACT]` slot. With two Auras and two creatures on
the board, the model cannot tell which Aura enchants which creature.

**Which permanents a static affects is a training label and never an input.** A `continuous` record
lists every permanent its static changes and how, and the model is trained to predict that list. On
every other record a creature's boosts are one sum, and nothing says which static supplies them. To
predict that destroying an anthem kills a damaged token, the model has to combine the anthem's
text, the token's boost and what the `continuous` records taught it about that anthem.

### Keywords reach the model only as flags, so it infers what each one does from outcomes alone

**Gen-2's snapshot gives a keyword to the model only as a flag.** The keyword factory attaches a
keyword's script to the card as a static or a trigger, and the snapshot lists neither, so no
keyword's text reaches a card slot. The model has to infer what each flag means from the outcomes it
sees.

**The knowledge probes' evasion grid shows what that costs.** A horsemanship blocker raises the
predicted chance of a block against every evasive attacker: against a flying attacker more than a
flying blocker does, and against a fear attacker more than the artifact and black blockers the rule
allows. A reach blocker barely moves the prediction against a flyer, though the rule treats reach
exactly like flying there. Flying's ranking of allowed over forbidden blocks sits near chance ([the
evasion
grid](2026-09-19-effect-knowledge-probes-design.md#evasion-the-model-knows-an-evasive-attacker-is-hard-to-block-and-which-blocker-answers-it-only-for-horsemanship-and-fear)).

**Forge already writes most of what a keyword does as a script fragment.** The keyword factory turns
75 of Forge's 202 keywords into statics and triggers. `extract-keyword-definitions` records each
fragment as `generated_script`, and flying's fragment names its own answer:

```
Flying        Mode$ CantBlockBy | ValidAttacker$ Creature.Self | ValidBlocker$ Creature.withoutFlying+withoutReach
Horsemanship  Mode$ CantBlockBy | ValidAttacker$ Creature.Self | ValidBlocker$ Creature.withoutHorsemanship
Hexproof      Mode$ CantTarget  | Activator$ Opponent | ValidTarget$ Card.Self
```

Shroud, ward, protection, flanking, prowess, exalted, vigilance, defender and cascade have a
fragment too. Reach, shadow, first strike, deathtouch, trample, lifelink, menace and indestructible
have none: the engine implements them in code. The capture records each keyword's template without
an instance's value, and several keywords that take a value, such as annihilator and bushido, show
no fragment there, so the count of 75 is a floor.

### Some output fields stay weak while the others improve

**A few fields with a real amount to learn barely moved between the first two epochs that scored
them.** The model predicts 46 output fields per entity. Each is scored by its explained deviance:
the share of the loss of the best constant prediction that the model removes, where 0% is no better
than always guessing the field's average. The table reads the first gen-2 arm.

| field | loss of the constant prediction | epoch 3 | epoch 4 |
|---|---:|---:|---:|
| `keywords_gained` | 0.565 | 21% | 21% |
| `poison_delta` | 0.138 | −11% | −14% |
| `life_delta`, for comparison | 2.402 | 80% | 81% |

**Training cannot respond to a weak field.** Gen-2 fixes each record's share of training when the
corpus is built, and the trainer only shuffles within the one shard it holds.

### A rare wording of a common effect gets few examples, and field weighting cannot see it

**Gen-2's arms doubt that an activated ping's target is affected, because training holds few such
pings.** Prodigal Sorcerer's "{T}: deal 1 damage to any target" and its relatives resolve about 70
times in gen-2's whole training corpus, about 45 of them at a creature. "Any target" damage spells
resolve about seven times as often. On arm `e64-lethality`, run B64's last epoch and arm
`e256-lethality`, the gate at the pinged creature sits near an even chance. Reading the same
records with "target creature" in place of "any target" lifts it above nine in ten. The model
therefore knows the effect and has not tied this wording to its target. The last epochs of runs
B128 and B256 learn it, so whether a checkpoint does is the luck of its fit ([the gen-2
record](2026-09-18-effect-model-gen2-improvements-design.md#where-the-model-doubts-that-a-hit-creature-is-affected-the-text-leaves-the-victim-open-or-is-rare-in-training)).

**The field's loss averages the weak spot away.** The gate removes about three quarters of its
constant prediction's loss over the whole corpus and stays level after epoch 20, so a controller
watching each field's loss would have no reason to raise it. The weak spot is one wording in one
situation, an activated "any target" ability whose target is a creature.

**Gen-2's lethality copies do not show the gate new boards.** They already copy most qualifying
pings, but each copy keeps its record's board and target, so the gate sees the same 45 boards over
and over.

### The validation loss is too noisy to stop on, and it disagrees with the probes about which epoch is best

**The card-disjoint loss moves between epochs by about as much as it improves.** Gen-2 training
keeps the epoch with the lowest card-disjoint validation loss and stops after five epochs without a
new lowest, a rule called patience. Late in gen-2's run with 64-dimensional ability embeddings, the
loss went up and down by about 0.1 from one epoch to the next while its trend fell by less than
that per epoch.

| epoch | card-disjoint loss |
|---:|---:|
| 20 | 3.9391 |
| 21 | 4.0106 |
| 22 | 3.9189 |
| 23 | 4.0816 |
| 24 | 3.7801 |
| 25 | 3.7344 |

**An unusually low epoch ends training early and is the checkpoint kept.** It resets the patience
counter and sets a mark that ordinary epochs then fail to beat. Five ordinary epochs in a row then
end training while the trend is still falling. The kept checkpoint is also the luckiest epoch, so
its loss overstates the model.

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

**The loss rises because the model grows more confident, not because it ranks worse.** The rise
sits mostly in the gate and in poison, with power and toughness changes next. In three runs of
four, the probe asking whether a card is affected scores higher on the same records. Log-loss
charges heavily for a confident miss, and a text the model never saw produces a few of those. The
summed loss therefore mixes how well the model ranks outcomes with how well its confidence matches
its accuracy, and the second can move it the wrong way.

### A few large fields carry most of the loss, and a Poisson likelihood fits amounts badly

**The loss is a sum over the 46 output fields, and the gate, damage taken and life carry most of
it.** A field such as `keywords_gained` can get worse without moving the total.

**Life, damage and card counts are scored by a Poisson likelihood, which assumes the variance
equals the mean.** Life and damage swings vary more than that, so a rare large swing is penalized
as if it were nearly impossible.

### The model has to learn every rule that depends on whose turn or which step it is, and gen-2 records several of them wrongly

**The `can_play` label leaves out the casting-timing rule, so the corpus labels a sorcery playable
on the opponent's turn.** A `decision` record's three verdict bits are `can_play`, `affordable` and
`has_legal_target`, and `can_play` is Forge's `SpellAbility.canPlay()`. That call covers split
second and a script's own restrictions: the phases it names, whose turn it requires, activation
limits and zones. The default timing rule lives in a separate call, `canCastTiming`, which no hook
records. Under that rule a spell is cast at sorcery speed unless it is an instant or has flash, and
an activated ability with `SorcerySpeed$` or a loyalty cost needs sorcery timing too. The hook in
`AiController.canPlayAndPayForFace` fires after the AI's own timing check whatever that check
answered, so a sorcery the AI weighs during the opponent's turn is written as playable.

**The AI writes only candidates that already passed `canPlay`.** `AiController.canPlayAndPayFor`
returns before the hook for a candidate that fails it, so every `can_play = false` in the corpus
comes from the random seat.

**On the AI seat, gen-1 and gen-2 `has_legal_target` and `affordable` record the AI's choices
rather than the rules.** `has_legal_target` reads whether the AI had set targets on the candidate.
The AI sets targets only when its evaluation gets that far and finds a target it wants, so a
candidate the AI put off is written with no legal target. `affordable` comes from
`ComputerUtilCost.canPayCost`, which applies the AI's cost preferences on top of the rules. It
refuses a planeswalker's last loyalty and a Ward cost the AI dislikes, and it draws from the shared
random stream to decide. The random seat uses the same cost check, and walks a charm's leftover
chain of chosen modes instead of the modes the charm offers. Verdict-head results and probes on
gen-1 and gen-2 records inherit all of these errors.

**Split second on the stack reaches the model only as the stack card's ability text.** The card
slots' keyword flags have no `split_second`, and the knowledge probes found the trunk reads little
of other cards' text.

**Casting speed is one case of a wider family.** Many texts depend on whose turn or which step it
is:

- **activation restrictions**: "activate only during your turn", "only during combat", "only once
  each turn", "only if you control an artifact";
- **resolution branches**: a spell or ability that does more, or something else, when it resolves
  on its controller's turn or an opponent's;
- **conditional statics**: a permanent whose static abilities differ between its controller's turn
  and an opponent's, so that its keywords or its restrictions change when the turn passes.

A single `can_play` bit or a resolution's outcome often does not say which condition decided it.

**The input states whose turn it is and who holds priority only relative to the acting player.**
The active player is the player whose turn it is (rule 102.1), and changes only when the turn does.
The player with priority is the one allowed to act now (rule 117), which passes back and forth
within a step. The acting player is the record's own notion: the controller of the ability being
recorded, or the player making the decision. Gen-2's global slot says whether the acting player is
the active player and whether they hold priority, a player slot says whether it is the acting
player, and a card slot says whether the acting player controls it. To learn that an opponent's
creature belongs to the active player, the model has to combine two of those flags. The snapshot
records no owner at all, which matters wherever control has changed: a stolen creature, "return it
to its owner's hand", a card going to its owner's graveyard.

**Each condition is seen mostly one way.** Forge's AI casts sorcery-speed spells on its own turn and
holds most instants for the opponent's. Training on that teaches the base rate rather than the
condition, as Forge's lethal removal did for deaths in gen-2.

### Combat records and keyword flags leave out what decides several fights

**A combat record's snapshot already shows the damage of the first creature or player hit.**
`BusBracketCollector.openCombatBracket` takes the snapshot when the step's first damage event
arrives, from `onCardDamaged` or `onPlayerDamaged`. By then Forge has marked that first damage, so
the creature it hit shows it as marked damage, and a player hit first shows the lower life total.
In 26 of the 40 one-on-one blocks gen-2's combat grid reads from the validation strata, the
blocker's marked damage equals the attacker's power. Gen-2's input gives each card a thermometer of
toughness minus marked damage, so the model can read part of the answer off the board.

**A card slot says that a creature is blocking, not which attacker it blocks.** The snapshot records
the pairing in each creature's combat status, `blocking` and `blocked_by`, but the card features
keep only a blocking bit and a count of blockers. With two blocked attackers on the board, the model
cannot tell which blocker deals damage to which attacker. Gen-2's combat grid reads only combats
with one blocked attacker for this reason. The gap reaches abilities too: Skewer Slinger deals 1
damage to the creature it blocks or is blocked by, and run B256's last epoch gives the damaged
creature a gate of 0.13 and 0.02 on gen-2's two probe records for it ([the gen-2
record](2026-09-18-effect-model-gen2-improvements-design.md#where-the-model-doubts-that-a-hit-creature-is-affected-the-text-leaves-the-victim-open-or-is-rare-in-training)).

**Gen-2's keyword flags collapse every protection into one flag.** The overlay maps
`protection_from_red`, `protection_from_blue` and every other protection onto a single `protection`
slot. Protection's text never reaches a card slot, printed or granted, because its fragment is a
static the snapshot leaves out. Gods Willing's "gains protection from the color of your choice"
therefore reaches the model only as the flag, and the flag cannot say which colour. Gen-2's evasion
grid shows the cost: its two protection rows, protection from the blocker's colour and from another
colour, give the model identical inputs and so identical predictions, where the rule forbids the
first block and allows the second.

**Protection is not always from a colour.** Forge's card scripts print protection on 196 cards.
Colours are most of it, and the rest is a long tail, several of its qualities on a single card.
Self-play shows the rare ones in a handful of games at best.

| protection from | printed protection lines |
|---|---:|
| a colour | 153 |
| a creature type, such as Goblins, Dragons or Vampires | 23 |
| a card type, such as artifacts, creatures or instants | 21 |
| multicolored, monocolored, or each colour | 9 |
| a condition, such as mana value 3 or less or a counter | 4 |
| a creature subset, such as legendary creatures or Gods | 3 |
| everything | 1 |

**Combat has no lethality copies, and its boundary stays soft.** A gen-2 lethality copy redraws a
qualifying damage spell's amount and the hit creature's toughness, and labels the death by the rule
rather than by what Forge did. Before the copies, the model separated a lethal block from a
non-lethal one better than a lethal spell from a non-lethal one. With the copies and a tenth of the
learning rate, the spell grid's error falls below every combat grid's. The combat boundary stays
soft on every checkpoint, the attacker's death sits further from the rule than the blocker's, and a
first striker is predicted to die as often as a creature without the keyword.

### Collection can lose or misplace events without leaving a trace

**Events dropped by the collectors are never counted.** An engine event that arrives while no
record is open is discarded, and nothing records that it happened. A state-based death was lost
this way for a whole gen-1 corpus before anyone noticed.

**A death in a later pass of a state-based check can be written into a trigger's record instead of
the resolution's.** Gen-2 holds a resolution's effect half open until the next action begins, so
that the deaths of the state-based check after it join the half. A trigger reaching the stack counts
as such an action. Forge's check runs in passes, and it puts each dies-trigger on the stack at the
end of the pass whose death set it off. That trigger closes the resolution's effect half, and a
death in the next pass lands in the trigger's bracket instead. For example, a Lightning Bolt kills a
lord while a Blood Artist is on the battlefield. Blood Artist's trigger reaches the stack after the
first pass. The damaged Goblins that only the lord's +1/+1 kept alive die in the second pass. Their
deaths are written into the trigger's effect half when it resolves, although they came before it.

**The first-strike damage step lost everything but its damage to players.** The collector decided
that an event was combat damage by checking that the phase name contains `COMBAT_DAMAGE`. Forge
names the first-strike step `COMBAT_FIRST_STRIKE_DAMAGE`, which does not contain it. In that step
only damage to a player opened a combat record, because that path reads the event's own combat
flag. Damage to creatures, the life changes and the deaths went to the ordinary bracket, which no
ability holds open during a damage step, so they reached no record. Gen-1 and gen-2 were both
collected this way.

**The loss left two marks on gen-2's raw shards.** A first-strike-step record exists only when a
player took damage, and it holds that damage and nothing it caused: the defending player's life
never moves. A first-strike fight between creatures alone has no record at all, so no record shows
a first striker killing its blocker before the blocker strikes back. The probe forks are collected
on another path and record the step correctly. On a sample of 40 raw shards the first-strike step
occurs in one game in ten and makes 6% of combat records.

| first-strike steps, 40 raw shards | count |
|---|---:|
| seen in any record | 361 |
| with a combat record, a player having taken damage | 282 |
| of those, with a blocked fight | 12 |
| with no combat record | 79 |

**Gen-2 repairs its built corpus and leaves the raw shards alone.**
`scripts/repair_first_strike_steps.py` adds each record's missing player outcomes from the record's
own damage events: the life loss, the lifelinker's gain, and poison for infect damage. It drops a
record holding a blocked fight in which either side strikes first, because that fight's creature
damage and deaths are gone. Gen-2's real first-strike knowledge therefore covers unblocked
attackers, and its creature-side knowledge of first strike comes from the probe forks alone.

**Damage events on a probe fork lose their source.** On a probe fork, every `damage_dealt` event
lacks `params.cause`, and damage to a creature reports `combat: false`. Observed combats carry both
correctly. The source can be recovered from the payload's `assignment_choices`. Because the source
is missing, several attackers' identical hits produce byte-identical events, which inflates the
validator's "repeats an event" check. Gen-1 and gen-2 shards both carry the gap.

### Gen-2's hooks and collectors recorded planned events as real ones, and several outcomes reached the wrong record

**Nine collector defects shaped gen-2's labels.** The gen-2 record describes each
([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#nine-collector-defects-shaped-gen-2s-labels-and-gen-3-collects-without-them)).
The largest is legality: a block or attack candidate was asked about while the AI was still
planning, so a creature planned for one block read as forbidden for another. Others put a combat
trigger's draw into the combat record, wrote a trigger record for a tap the AI was only planning,
and gave a whole effect one note where it made several kinds of object.

**What-if legality queries were sampled before they were de-duplicated.** A what-if query is the AI
asking whether an attack or block would be legal without committing to it. In gen-2,
`--legality-rate` drew each query before identical snapshots were merged, so a question the AI
asked K times survived with probability 1 − (1 − r)^K, and blocker what-ifs outnumbered real block
decisions about five to one.

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
likewise recorded as `granted_trait`.

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
creature, are both decisions the model should learn the outcome of.

**The snapshot may hide a creature's dependence on a boost.** After the base became 0/0 under the
+2/+2 boost, the snapshot still reported `base [2,2], boosts [0,0]`. It appears to report the
printed base and fold set-base effects into the boosts, so a creature that survives only because of
a boost looks like an ordinary 2/2. This needs checking in general, not only on this card.

### The held-out coverage round covers more than the held-out texts

**`collect-coverage --only-cards` targets every text carried only by held-out cards.** That set is
larger than the held-out texts themselves.

### Gen-2 deleted its withheld keyword instead of expanding it, and defects in the build, targets and probes shaped its results

**Every gen-2 arm deleted cascade from its input rather than expanding it.** `--withhold-keyword`
is meant to keep the keyword's token out of training and replace each occurrence with its reminder
text. The batcher removed the token before keyword expansion ran, and the flag that should have
forced the expansion was never read. A cascade keyword line therefore reached the encoder as its
`[CLS]` token alone, in training and evaluation alike. Gen-2's zero-shot slice therefore measured a
keyword with no text, not a keyword read through its definition, and it is not a baseline for
gen-3's.

**A review of `src/effects` and the probe scripts after the gen-2 runs found fifteen defects that
shaped gen-2's results, and thirteen more on paths gen-2 never ran.** The gen-2 record lists the
fifteen
([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#sixteen-defects-in-the-corpus-build-the-training-labels-the-evaluator-and-the-probes-shaped-gen-2s-results-and-gen-3-is-built-without-them)).
Several change what a field means or how the loss weighs it.

**The observed-mana probe counted mana events rather than mana.** Its target read
`params.amount`, a field the collector never writes, so every event counted as one mana whatever it
produced
([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#the-observed-mana-probe-label-counted-mana-events-rather-than-mana)).

## Description of the experiment

**Gen-3 sets one goal: it rebalances its training examples rule by rule, so that no keyword or
ability counts as rare in what the model trains on.** Five moves serve it, in order of how much of
the plan rests on them:

1. **An index of the corpus by the rules of the game.** A catalogue of the Comprehensive Rules says
   which rules a record can show, detectors count how many records show each, and scenarios say
   which situations illustrate each.
2. **Inputs and outputs that let the model learn each rule.** Every rule is mapped to what decides
   it and to the output trained on its outcome, and the missing ones are added together with the
   record schema. Each line is tied to its card, and each link between two cards is read as a
   relation between their slots.
3. **Training that draws records by rule.** Adaptive sampling draws more of what a weak field or a
   short rule needs, and a record drawn for a rule trains only that rule's outputs.
4. **Checkpoints judged by probes and rule scenarios**, with every output field on one scale in the
   loss.
5. **Collection that records what decides each rule**: every line of a permanent, which rules check
   a verdict failed, whose turn it is, combat as it starts, each action's state-based deaths in the
   action's own record, and gen-2's collector defects closed.

The subsections below take these in order, then two options still open. Each points to its outcome
in [Result](#result) where one exists. The catalogue, the mapping of rules to inputs and outputs,
and the fixes to gen-2's defects have run so far, and no gen-3 corpus has been collected.

### Gen-3 indexes its corpus by the rules of the game

**For each rule, gen-3 records three things.** Each answers part of the coverage gap in
[Background](#gen-2-balances-its-corpus-by-card-text-and-rule-family-so-whether-a-rule-of-the-game-is-covered-is-never-counted):

1. whether the records the collectors write can show it at all;
2. how many records in the corpus do show it;
3. which game situations would illustrate it, as a unit test illustrates one behaviour of code.

The counts feed corpus balancing and collection, and the illustrations become a test set. The
catalogue's triage and scenarios are in [Result](#half-of-the-comprehensive-rules-have-an-observable-outcome-and-all-but-95-of-those-have-a-scenario-not-yet-checked-against-forge),
and no detector has been written yet.

**A triage pass comes first, because many rules can never appear in a record.** Sorting every rule
into observable in a record, observable only indirectly, or out of scope decides how large the
rest of the work is. A language model makes the first pass over the document, and a second checks
the first. Decisions are out of scope throughout, because the model predicts what the engine does
once they are made.

**Each mechanical rule gets a scenario: a minimal pair of boards on which changing one fact flips
its outcome.** A scenario sets a board, lists every decision already taken, and gives at least two
variants. Grizzly Bears dealt 2 damage by Shock dies; a Centaur Courser in its place survives.
Distinct mechanisms get distinct scenarios. Red Elemental Blast whose target stops being blue in
response fizzles under rule 608.2b, while Pyroblast aimed at a non-blue spell resolves and does
nothing. Both look like "nothing happened" and are different rules.

**Counting a rule needs a detector that recognizes it in a record, and each scenario says what its
detector looks for.** The scenario gives the record kind and the observable outcome, and its
contrasting variant gives the near miss the detector must reject. A detector is also a coverage
counter, so the same code puts per-rule counts in the corpus manifest. Detectors range from trivial
to impossible:

| rule | how a record shows it | detector |
|---|---|---|
| 702.2, deathtouch | a combat with a deathtouch creature, and the probe forks that strip it | exists: the combat family and the probe keyword |
| 704.5f, a creature with toughness 0 dies | a zone change during the state-based check after a resolution | derivable from the effect half's events |
| 613, the order of continuous-effect layers | no field records which layer applied what; the `continuous` record shows the result | none: a model that predicts each layer's result has learned what the order decides |

**The detectors run over gen-2's corpus before any gen-3 collection.** The counts show which rules
self-play already illustrates, with a record in which the rule's effect happens and one in which it
does not. They also show which rules self-play reaches on one side only, and which it never
reaches. The order in which to fill those can follow how often each rule's mechanic appears in the
sets collection draws from.

**Illustrating a rule draws on three sources, in order of preference.** A situation found in a
real game comes with real decks and real decisions around it. A situation built on purpose comes
with whatever the builder chose.

1. **The collected corpus.** The detector that counts a rule also finds the records that
   illustrate it.
2. **Directed forks of live games.** Gen-2's forks already change a live game: an interventional
   resolution forces an ability onto the stack, and a damage-step probe strips one keyword from one
   creature. Directed by rule, a fork waits for a board that meets the rule's precondition and
   changes the one fact that flips its outcome. The real record and the forked one are then a
   minimal pair from a real game.
3. **Puzzle-style starting states, as a last resort.** Forge ships 372 puzzles under
   `forge-gui/res/puzzle/`, each a starting game state: life totals, the active player and phase,
   and every zone's cards with their links. Written for a rule, such a state is a unit test the
   collector could play. It serves only a rule that neither the corpus nor a directed fork reaches.

**The rule tests that judge the model come from games outside training.** If the same situations
both balance the corpus and measure the model, they stop measuring generalization. A rule's test
records are taken from games placed in the validation strata, as the knowledge probes are, and
split into a selection half and a reporting half as [the stopping
plan](#the-summed-validation-loss-disagrees-with-the-probes-over-training-and-gen-3-judges-checkpoints-by-probes-and-rule-scenarios)
describes. Collection aimed at a rule feeds training separately.

**A record whose outcome differs from its scenario's expected outcome sends the scenario back for
review, because the scenario is the likelier of the two to be wrong.** Each expected outcome is an
automated reading of the rules text and the cards' scripts, never checked against a game. Forge's
engine is mature and widely played. The review rereads the rule, the scripts and
the record, and corrects the scenario where the reading was wrong. Only a disagreement that survives
the review is a suspected Forge departure from the rules. A confirmed departure is recorded on the
scenario, and the rule's records stay out of the rule tests until Forge is patched on the hooks
branch or the departure is accepted. Where Forge plays a situation as expected, Forge's outcome and the rule's
outcome are the same, so the choice between a rule label and Forge's outcome matters only for the
rules that disagree.

**Rules could also be the pools of the adaptive sampling below.** A per-rule detector gives each
record the rules it exercises, which is the same index that sampling by field needs. The two items
would share the decision between pools, an index and a database.

#### A directed fork needs an operation and a trigger on the live board

**Each scenario says what to look for in a game and which single fact to change.** The base variant
is the situation a detector searches for, in the corpus or on a live board during collection. Each
other variant's change is the perturbation a directed fork applies to that board. The decisions are
the ones the game made, so they need no conversion.

**The variants' changes have to become fork operations.** Each change is an English sentence. A
pass classifies every change into a small set of operations: change a creature's power or
toughness, add or remove a keyword, change a colour or type, add or remove counters, tap or untap a
permanent, change a life total, or replace one card with another. The counts per operation show
which few operations reach the most rules, and so which to build first. Gen-2's damage-step probe is
already one of them.

**The collector runs the rule's detector on the board at each phase boundary, the moment
interventional forks already use.** When the board meets the rule's precondition, it forks, applies
the operation and plays the fork on. The budgets gen-2 puts on forks still apply: a score check
against the live game before any change, and a cap on forks per real resolution.

**A puzzle needs the scenario's decisions written as an action list as well**: cast, activate,
target, pay, attack, block, assign damage, pass. Whether a collecting worker can load a puzzle has
not been checked. Converting scenarios to puzzles waits until the detectors and directed forks show
how many rules are left.

### A rule is learnable only when the model reads what decides it and an output is trained on its outcome

**Gen-3 maps every rule to its input and output before changing either.** For each mechanical and
legality rule the map says whether an existing output is trained on the outcome, whether a record
carries the outcome with no output trained on it, or whether nothing carries it. It also lists the
deciding facts the model does not read. The scenarios make the check concrete: the difference
between a variant in which the rule's effect happens and one in which it does not is the outcome
an output has to be trained on, and each variant's change is a fact the model has to read. The new
outputs and input slots then follow from how many rules each would serve, and they change together
with the record schema. [Result](#gen-2s-model-can-learn-under-a-third-of-the-rules-and-reading-every-line-of-a-permanent-is-the-change-that-adds-the-most)
gives the map and the fields it produced.

**What rules text describes is read as lines, never as a vocabulary of effect kinds.** Forge itself
builds most lasting effects as lines. A prevention shield, "creatures can't block this turn" or
"you may cast that card this turn" is an Effect card in the command zone, holding the triggers,
statics and replacements of the spell that made it. A floating effect is therefore a card slot in
the command zone, read like a permanent, with the e vectors of the source lines that created it. A
dungeon is a card slot too, its rooms its lines. Only what text cannot say stays numeric: the
objects the effect touches, its duration, and the amount left on a shield that counts down.

**Which of a card's lines are in force is a flag on each line.** A line is active when its face is
up, its Class level is reached or its Room door is unlocked. A spell's kicker or alternative cost
is a flag on the line it was cast with, and a modal spell's choice a flag on the chosen mode.
Temporary grants become lines as well, with a keyword read through its reminder text. Their own
flag keeps them apart from printed lines, which the gate-2 keyword test needs, since it strips a
keyword from the channel that carries it.

**A link between two objects is a typed relation between their slots, not a flag on one end.**
What an Aura is attached to, what an attacker attacks, what a spell targets and what a permanent
exiled are each a relation the trunk's attention reads, as [the relation
plan](#every-link-between-two-slots-is-a-typed-relation-in-attention-a-lines-card-included)
describes. Names are relations too: the rules only compare names, as the legend rule does, so two
slots whose names match are related, and no name embedding is learned.

**Outputs use the existing field types and supervision.** A new output is a binary, a count, a
signed delta, a categorical or a multi-binary, supervised only by the records that carry it. An
amount is a scalar as raw plus log1p, with a thermometer where the model compares it with damage or
toughness. Outputs that belong to no card or player, such as day becoming night, need a global head
beside the created-objects head.

**A global designation block and a few per-card fields would record more than half of the 210 rules
that need new fields.**
Several of the designations come from sets that sealed pools can draw from: day and night from
Innistrad: Midnight Hunt, dungeons from Adventures in the Forgotten Realms, the city's blessing from
Rivals of Ixalan.

- **The global and player slots** hold day or night, the monarch, the initiative holder, speed, the
  city's blessing and rad counters, and per-turn counts the "this turn" conditions read: cards
  drawn, life gained and lost, permanents put into a graveyard.
- **Each permanent** gains its designations as a set (renowned, suspected, goaded, and so on), the
  creature it is paired with, a battle's protector, the X it was cast with, the values it chose,
  and the objects exiled with it.
- **Each card** gains its subtypes, its face, whether it is a copy, its mana cost, and its printed
  loyalty and defense.

**Layer attribution, subgames, restarts and control of another player are not worth collecting.**
The `continuous` record already shows the result of every layer, and a model that predicts those
results has learned what the order decides. Subgames, the Karn Liberated restart and control of
another player are rare enough to leave out.

**The changes go in from the one that needs the least work to the one that needs the most.**

1. **Every line of a permanent**, already planned below, with the `line of` relation that ties each
   line to its card.
2. **The facts the snapshot already records.** Subtypes, faces, copy sources, names, attachments,
   attack targets and stack targets need input slots and no new collection.
3. **The collection and curation gaps.** Permanent-spell resolutions have to survive
   `build-corpus`, turn-based actions need a record of their own, and attackers records are anchored
   to one attacker, as blockers records already are. All three then feed outputs that exist.
4. **Outputs for outcomes the records already carry.** The resolution outcome, a player losing,
   copies, phasing, regeneration, prevention, and extra and skipped turns are each a new output over
   a record field or event that exists. The zone outcome gains the destinations it now merges.
5. **New collection with its own inputs and outputs**: the designation block, values chosen
   earlier, the objects exiled with a card, floating effects and what happened earlier in the turn.

**Running the map again over each new layout measures what the change bought.** The map is a pure
reading of the layout and the scenarios, so a rerun counts the rules each change made learnable.

### Training could draw more of the records a weak output field needs, from the whole corpus

**When a field stays weak, training shows more of the records that supervise it.** The idea is
established as adaptive data mixing and adaptive loss weighting. Large language-model pretraining
stores its data as separate sources and draws from each by a mixing weight. Methods such as DoReMi
raise a source's weight while its loss stays high against a reference model. Multi-task training,
of which the 46 output fields are a case, balances the tasks' loss weights instead, with methods
such as GradNorm and uncertainty weighting.

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
else.** The map of rules to outputs names each rule's outputs, existing and new, so the mask needs
no labelling by hand. The rebalancer knows which rule it drew a record for once the detectors tag
each record with the rules it shows. A record drawn at its natural share, for no rule, keeps every
output. Gen-2 already has the mechanism: the loss skips a field the record says nothing about, and
the verdict head takes a mask per column. Fine-tuning a language model uses the same technique when
it leaves the prompt out of the loss.

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

**Three parts of gen-3 could supply the activated pings [Background](#a-rare-wording-of-a-common-effect-gets-few-examples-and-field-weighting-cannot-see-it) finds short.**

- **Per-rule rebalancing, with cells that cross a rule with the kind of ability exercising it.**
  Rule 115, targets, crossed with activated abilities would give these pings a quota of their own.
- **Retargeting copies.** A copy that hands the ping to another creature on the board, and labels
  the new target's damage and death by the rule, would show the gate new boards from the same
  records.
- **Collection aimed at the wording.** A coverage round over cards with an activated "any target"
  damage ability would add creature targets beyond what the AI's own choices supply.

**The measure is gen-2's toughness sweep.** Its activated damage records should give the same gate
read with "any target" and with "target creature". Probes that pick training records should also
drop repeated rows: the curated corpus writes a rare text's records several times, and gen-2's
boundary chart counted three Suffocating Blast resolutions as twelve of its 40 rows.

### The summed validation loss disagrees with the probes over training, and gen-3 judges checkpoints by probes and rule scenarios

**Gen-3's control score is built from probes and rule scenarios.** The control score is what
stopping reads and what picks the checkpoint the run keeps. Gen-3 builds it from the model-level
knowledge probes, the damage-by-toughness and combat grids, and the frozen rule tests of the rules
index. Every input is a forward pass, so nothing is trained to compute it. Built per rule, the score
follows the balance training aims for rather than how often a situation happens in self-play. It
answers the disagreement [Background](#the-validation-loss-is-too-noisy-to-stop-on-and-it-disagrees-with-the-probes-about-which-epoch-is-best)
shows.

**Stopping reads a smoothed score.** Two forms are common:

- **A moving mean of the last three epochs' scores.** Patience counts epochs without a new best
  mean. The kept checkpoint is the middle epoch of the best window.
- **An exponential moving average of the weights.** Training keeps a running average of the
  parameters, each step moving it a small fraction toward the current weights, and validation scores
  the average. The average is itself less noisy than any single epoch's weights and is usually a
  slightly better model, so it is both what stopping reads and what the run keeps. `resume.pt` would
  have to carry it.

**Every output field counts equally where a loss enters the score.** Each field is scored by its
explained deviance on the card-disjoint stratum, and the score reads the mean over the fields. Two
details need deciding:

- **Fields with almost nothing to learn.** A field whose constant prediction already has a tiny
  loss, such as `poison_delta`, has an explained deviance that swings widely on a few records. Such
  fields need a floor on their constant-prediction loss to enter the mean, or a weight that grows
  with it.
- **What a point is worth.** An equal mean rewards a weak field going from 20% to 30% as much as
  life going from 80% to 90%. That is the intent, but it lets a weak field's noise steer stopping,
  which makes the smoothing more necessary, not less.

The same per-field scores are what adaptive sampling reads on the training side, so the two share
their bookkeeping.

**The probe sets grow, and a selection half picks while a reporting half measures.** Choosing the
best of many epochs on a score favours the epoch whose noise happened to help, so the chosen
epoch's score overstates it by an amount that grows with the noise. More probe records shrink that
noise. Selecting and reporting on the same records would still flatter the chosen epoch, so each
probe set and the rule tests are split in two. One half picks the checkpoint, and the other half
reports it. With the sets enlarged, each half can be as large as gen-2's whole set.

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

**Each field's loss is divided by the loss of its constant prediction.** After the division, a
field counts by the share of its constant prediction's loss that remains, 1 when it has learned
nothing. This is the training-side twin of the explained deviance the control score reads. A field
with almost nothing to learn, such as poison, needs a floor on its constant-prediction loss, or its
noise is amplified. Per-rule rebalancing does not make the division unnecessary: with every rule
equally represented, a yes-or-no field on every card and a life-change amount still sit on
different scales.

**Amounts get a distribution wider than Poisson.** A negative binomial, or a distribution over bins
of the amount, fits life and damage swings better. The constant prediction the division uses is
then computed with the same distribution, so the two changes combine.

**Rule-labelled copies fill the gaps the real data leaves.** Gen-2's lethality copies redraw the
amount and the toughness of a qualifying damage spell and label the death by the rule. Where the
rules index finds a rule with too few qualifying or varied real records, copies of the same kind
supply them. The combat copies among the open options are one case.

**Two designs are not taken:**

- **Learned field weights**, such as uncertainty weighting. Each field's weight is trained along
  with the model, and a field the model finds hard ends up weighted down, which works against the
  per-rule rebalancing.
- **Label smoothing on the gate.** It would cap the gate's confidence, the main cause of the loss
  rise. Once checkpoints are judged by probes, that confidence no longer misleads the choice.

### The snapshot should list every line of a permanent, not only its spell and activated abilities

**`printedKeys` and `grantedAttached` add the card's live triggers, static abilities and
replacement effects**, through `getTriggers()`, `getStaticAbilities()` and
`getReplacementEffects()`. Reading Forge's live lists respects a creature that has lost its
abilities, which reading the card's sidecar at training time would not. Abilities a permanent gains
from another permanent's static, such as a lord granting a mana ability, still appear nowhere, and
are listed in the same change. A board-heavy record then encodes more texts per step, so training
slows somewhat.

**Gen-2's board-abilities ablation is the baseline.** It scores each gen-2 arm on legality
restricted by a non-keyword static and on resolutions with a replacement on another permanent,
beside controls of the same kinds. A gen-3 model that reads the new lines should close those
subsets' loss toward their controls, and noise in its card slots should start to cost something
there.

### Every link between two slots is a typed relation in attention, a line's card included

**Each link is a directed relation type, and each type adds a learned vector to a key and to a
value.** When slot i attends to slot j, the key of j gains the vector of the relation from i to j
before the score is taken. The value of j gains a second vector of that relation before it is passed
to i. The key vector lets whether i attends to j depend on the relation and on what both slots hold.
The value vector makes the message say what kind of partner sent it, so a creature receives its
blocker's information as a blocker's. Several relations on one pair add as vectors. The mechanism
is relation-aware self-attention, introduced for relative word positions by Shaw, Uszkoreit and
Vaswani, [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
(NAACL 2018), and applied to typed relations over a graph by Wang et al., [RAT-SQL: Relation-Aware
Schema Encoding and Linking for Text-to-SQL Parsers](https://arxiv.org/abs/1911.04942) (ACL 2020).
Gen-3's first model has this trunk, so no gen-3 checkpoint is trained without it.

**A card's lines are tied to it by the `line of` relation.** It links a card slot with each of its
ability slots, in both directions, and links the lines of one card with each other. A charm's mode
slots are tied to their root by `mode of`. The trunk then knows exactly which card each line sits
on, and reads the card's own lines as its own.

| relation, one type per direction | source |
|---|---|
| line of, and line of the same card | the entity each ability slot is built from |
| mode of | the charm root each mode line follows |
| attached to | the snapshot's `attached_to` |
| blocks, attacks | the combat status's `blocking` and `blocked_by`, and the attack target |
| targets, source of | the acting ability's targets and source, and each stack object's targets |
| affects, as a static | a new snapshot field listing, for each static line on the board, the entities it changes |
| controls, owns | the controller, and the planned `owner` field |
| has the same name | the names in the snapshot |
| exiled with | the planned field of objects exiled with a card |

**The bits and counts that already describe a link stay as features.** Attention takes weighted
averages, so it cannot count a creature's blockers reliably, and menace turns on that count. The
target and source bits stay as the plain answer for the acting ability, and the controller bit as
the plain answer to whether a card is the acting player's.

**How the collector lists what each static affects on every snapshot has to be established.** The
`continuous` collector already computes it, one static at a time, by recomputing the board without
that static. Running that for every static on every snapshot would be too slow, so the field needs a
cheaper reading of Forge's layer state.

**The probes measure what the relations add.** Noise in the card blocks' ability slots should start
to cost something. Gen-2's combat grid reads only combats with one blocked attacker, and gen-3's can
read combats with several. A probe that destroys an anthem should predict the deaths of exactly the
damaged creatures the anthem kept alive.

**Five alternatives are not taken:**

- **A learned scalar per relation type and head, added to the attention score**, as Graphormer
  does for graph edges (Ying et al., [Do Transformers Really Perform Bad for Graph
  Representation?](https://arxiv.org/abs/2106.05234), NeurIPS 2021). It changes how much a slot
  attends to its partner, not what it receives, so the kind of relation reaches the receiver only
  through the partner's content. The trunk's four heads cannot each specialise in one of a dozen
  directed types. Summed scalars are ambiguous, since 0.8 can be 0.3 + 0.5 or eight times 0.1.
- **A random group id per card, drawn for each record and added to the card's slot and its lines.**
  It would tie lines to cards without a custom layer. It adds to every slot a random vector the
  model has to learn to ignore, and the model has to learn to match equal ids, where `line of`
  states the pairing exactly.
- **Group ids for every link.** A slot carries one group id, and the links are many to many: a
  static changes many permanents, and a permanent can be changed by several statics. Several ids
  summed on one slot bring back the ambiguity of the scalars.
- **An embedding of each card's place on the board.** It gives the order of the cards a meaning
  that the game does not.
- **A card slot holding the board position of its partner**, such as the attacker a blocker blocks.
  It carries one partner per feature, and the model has to learn to match the position against the
  other slot's.

**The relation terms cost one product per relation type, not per pair.** The key term is computed
once for each slot and each relation type and then looked up for each pair. The value term sums each
slot's attention weights by relation type and multiplies them by the value vectors. Both scale with
batch × slots × relation types. The value term needs the attention weights explicitly, so the trunk
replaces `nn.TransformerEncoderLayer` with a layer of its own. The trunk is small beside the ability
encoder, so explicit attention should cost a small share of a step. Step time and peak memory on one
fixed shard, before and after, measure it.

**The custom layer is written test first, because its mistakes train without an error.** A
transposed relation matrix or one built in a different slot order still lowers the loss. Five tests
guard it:

- With both relation tables at zero and the weights copied from `nn.TransformerEncoderLayer`, the
  layer reproduces the stock layer's outputs and gradients in fp32.
- Permuting the slots and the relation matrix together permutes the outputs the same way.
- A relation set from i to j changes the output of i and not of j.
- On a toy task that only the relation can solve, such as reading a feature of the partner among
  identical-looking slots, a small model learns with the relation and stays at chance with the
  tables at zero.
- The anthem probe above passes on real records.

### Gen-3 is 256 wide throughout, `e` included

**The ability encoder, `e` and the trunk are each 256 wide.** Gen-2 kept the encoder and the trunk
at 256 and swept the width of `e` over 64, 128 and 256. Its 256-wide continuation checkpoints come
closest to the rule on the scorecard on average
([gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#the-64-wide-and-256-wide-runs-split-the-measures-and-the-128-wide-runs-trail-on-the-probes-at-every-stage)).
Gen-3 adds inputs, output fields and relation vectors, and each needs room in the same vectors. The
sweep trained one seed per width and could not tell a width effect from a seed effect, so the
choice rests on the room gen-3's additions need more than on a measured gain.

### A keyword with a Forge script gets its own `e`, and keeps its flag only where other text names it

**Every keyword whose fragment Forge can produce gets a line, and so an `e`, in its card slot.** The
fragment need not be printed on the card: the factory's copy, attached at runtime, is the line. The
snapshot fix in the section above reads it with the rest of the card's statics and triggers, and
with each instance's own value, such as annihilator 2 or ward {2}. Two collector rules make that
happen. A trait Forge marks `Secondary$ True` is kept rather than skipped. Its provenance key names
the keyword line the card prints, so the sidecar joins it
([keywords](#keywords-reach-the-model-only-as-flags-so-it-infers-what-each-one-does-from-outcomes-alone)).

**A keyword keeps its flag where other text tests for it by name, or where the engine still checks
it.** A keyword plays two parts. One is what it does, which the fragment carries. The other is a
property other text tests. Flying's own fragment asks whether the blocker is `withoutFlying`, and
"destroy each creature with flying" and "creature without flying" ask the same. A flag answers that
in one bit, where an `e` would make the trunk recognise a flying line from its vector. Flying,
horsemanship and every other keyword a selector names therefore keep their flag beside the new line.

**A hybrid keyword keeps its flag too.** Where the engine checks a keyword by name as well as
running its fragment, the fragment does not carry the whole keyword. Protection is the case to
check: its fragment prevents damage and forbids blocking, targeting and attaching, and the engine
may still test it elsewhere. Only a scripted keyword that no selector names and the engine never
checks loses its flag. The list follows from the selectors in Forge's card scripts and the keyword
checks in its engine code.

**A keyword with neither a flag nor a fragment keeps the reminder-text fallback.** For a known
keyword the engine implements, the flag is exact, and a text would hold nothing the flag does not. A
new set's keyword that the engine implements has no flag slot, and its reminder text, which an
unknown keyword already expands to, is the only signal the model gets.

**Gen-2's evasion grid is the measure.** With flying's fragment in the attacker's `e`, a flying or a
reach blocker should lift the predicted block against a flyer, and a horsemanship blocker should
lift it only against a horsemanship attacker. The flying row's ranking should rise from near chance,
and the horsemanship blocker's lift against flying and fear attackers should fall.

### The model has to learn every rule that depends on whose turn or which step it is, and collection records which check decided each one

**Gen-3 computes all three verdict bits in the collector, from the rules alone, for both seats.** The
AI's hook passes only the candidate, before the AI evaluates it and ahead of its `canPlay` filter.
The collector's `PlayabilityVerdict` checks mana and additional costs without the AI's cost
preferences. It counts each targeting clause's candidates against the clause's minimum. It judges a
charm by how many of its offered modes have reachable targets. The random seat filters its own
plays on the same verdict.

**Each candidate carries a fourth verdict bit, `timing`, and Forge's own answer is the label.** The
bit holds `canCastTiming`, computed by `PlayabilityVerdict` for the AI seat and the random seat
alike. Recomputing the rule in Python from the snapshot would miss what Forge already applies:

- flash granted by a static, such as Leyline of Anticipation or Vedalken Orrery;
- a static that holds opponents to sorcery timing, such as Teferi, Time Raveler;
- a spell cast through another effect, which ignores its own timing.

Where Forge can name the static that decided the timing, its key goes in `responsible_static`, as
for the other verdicts. The keyword flags gain `split_second`. `can_play` becomes Forge's whole
answer, `canPlay` and `canCastTiming` together, and the verdict head gains a column for `timing`
beside it.

**Each candidate records which rules checks it fails, not only that it fails.** Beside `timing`, a
candidate carries `failed_checks`, a list naming each check Forge found unmet: split second, casting
speed, whose turn, phase or step, combat, activation limit, a script condition, the zone, a static
prohibition, and the cost. Each named check carries its `responsible_static` where a static decided
it. Forge groups most of these inside `SpellAbilityRestriction.canPlay`, so the hook calls the
parts separately where Forge exposes them, and the implementation has to confirm how finely it can
split them. The verdict head gains a column per check.

**A resolution records which branches of its chain ran.** Forge evaluates each chain segment's
`Condition$` as it resolves. The resolution payload gains `conditions_met`, one entry per segment
the chain conditions, saying whether the condition held, keyed by the segment index `attributed_to`
already uses. An outcome alone can be ambiguous, because a branch that ran may change nothing
visible on this board.

**The snapshot records whether each static line is in force.** Once the snapshot lists a permanent's
statics, each listed static carries `active`, Forge's answer to whether its condition holds now. A
permanent with different statics on its controller's turn then shows the text of both, which one is
in force, and, through the `continuous` records, what the one in force does.

**The board input says who controls, who owns, whose turn it is and who holds priority, for every
player and every card.**

| gen-3 input | slot | source |
|---|---|---|
| is the active player | each player | computed from the existing snapshot |
| holds priority | each player | computed from the existing snapshot |
| controller is the active player | each card | computed from the existing snapshot |
| controller holds priority | each card | computed from the existing snapshot |
| owner is the acting player | each card | new `owner` field per entity, at collection |
| owner differs from controller | each card | new `owner` field per entity, at collection |

The acting-player bits stay. Every row but the owner rows is an input-layout change computed from
what the snapshot already holds.

**Each condition is balanced between met and unmet in the curated corpus.** The fix has three parts:

- **Curation splits each condition's budget by its value.** Gen-2's curation already splits each
  class over rule families and then over outcome signatures, with one water-filling allocator, a
  reuse cap on short cells and the shortfall reported in the manifest. For a record whose acting
  line, candidate or listed static carries a turn or step condition, the signature gains whether the
  condition held, taken from `failed_checks`, `conditions_met` or the static's `active`. Met and
  unmet then draw equal budgets, a short side repeats up to the reuse cap, and the manifest names
  every condition that stays one-sided.
- **Collection evaluates both players' candidates at every priority stop.** A decision record is a
  rules-level question and takes no action, so the collector can ask it for the player without
  priority too. This needs the hook to evaluate candidates without priority correctly, because
  Forge's `canPlay` assumes the asking player could act; the implementation has to establish what
  Forge answers there.
- **Interventional forks fill the side that stays short.** A text whose resolution branch is seen
  on one turn only is a candidate for a fork that resolves it at a phase boundary of the other
  player's turn, under the existing fork rules and budgets.

**A knowledge probe measures each kind.** It takes records whose acting line, candidate or board
carries a turn or step condition and edits whose turn it is and the phase, then reads the prediction
the condition governs: the matching verdict column for a restriction, the predicted outcome or
branch for a resolution, and the affected creature's keywords or legality for a conditional static.
Further subsets cover casting speed under the same edits and a split-second spell on the stack.
Gen-2 has no such columns, so its baseline is `can_play` and its outcome predictions under the same
edits.

### A combat record's snapshot should be taken before the damage step deals any damage

**The snapshot is taken at the start of the damage step.** The damage-step probe forks already copy
the game at the phase event, before any damage, so the observed record can take its snapshot at the
same moment. The combat shape is read at the first damage event, as now, because no creature has
left combat by then. The assignment can be read at any point of the step, because Forge holds each
step's assignment as it was made.

### The keyword flags should say what a creature has protection from

**Gen-3 gives each protection its quality through its line.** [The keyword
decision](#a-keyword-with-a-forge-script-gets-its-own-e-and-keeps-its-flag-only-where-other-text-names-it)
gives protection a line from its fragment, and an instance's fragment names what it protects from.
Any quality then reaches the trunk as text, including ones no flag anticipates. The keyword file's
capture shows the template with the quality left blank, so the collector reads the instance's
fragment rather than the template. Protection also keeps its flag, since the engine may test it by
name. Fixed flags per colour, per card type and for everything remain the fallback if an instance's
fragment turns out not to carry the quality.

**The rare qualities need built examples, or the flags carry nothing the model learned.** Rule
702.16 says what protection does: damage from a source with the quality is prevented, and the
creature cannot be enchanted or equipped, blocked or targeted by such a source. Three sources could
supply the examples, and per-rule rebalancing decides how many each quality gets:

- **Rule-labelled copies.** A copy of a real record grants a creature a protection quality and
  relabels the four outcomes by the rule, as the lethality copies relabel a death: the damage
  prevented, the block or target made illegal, the Aura put into the graveyard.
- **Variant scripts.** `collect-variants` already swaps selectors from a whitelist. A protection
  keyword's quality is such a selector, so variants of real cards with another quality are played
  for real.
- **Interventional forks.** A fork grants protection mid-game and plays the turn out, so the
  outcomes come from Forge rather than from a rule written beside it.

Gen-2's evasion grid becomes the measure: its protection rows extend to the other qualities, from a
creature type to everything.

### A record holds an action and the whole state-based check that follows it

**State-based actions have no record of their own.** What a state-based check does is written into
the record of the action that left the board in that state. Shock deals 2 damage to a Grizzly
Bears, and the Bears' death is part of Shock's effect half. The same holds for a creature shrunk to
0 toughness, a token ceasing to exist, the loser of the legend rule and an Aura falling off. The
model then learns rule 704.5g as what Shock does to a creature with 2 toughness, which is the
question the effect model answers. Gen-2 already files most of these deaths this way. A combat
damage step's record holds its own deaths too, because it stays open until the step ends.

**The held effect half takes every death until Forge's state-based check is over, whatever reaches
the stack between its passes.** Gen-2 took the next action to mean that the check was over. A
dies-trigger put on the stack between two passes is such an action, and it comes before the check's
last pass ([Background](#collection-can-lose-or-misplace-events-without-leaving-a-trace)). Forge freezes the
game's view tracker for the whole check, every pass included. The collector already reads that
freeze to tell a state-based death from a cost paid for the next spell. In gen-3, a trigger that
reaches the stack while the tracker is frozen opens its own bracket without closing the held half.
Every death in the check's later passes joins the held half, and the half is written at the first
action after the check ends.

**The trigger keeps its own record.** Its bracket holds what it does when it resolves, and none of
the deaths of the check that put it on the stack. A death filed this way stays marked as no
clause's doing, as in gen-2, so a creature a destroy clause killed and a creature that died of
lethal damage stay distinct in the record.

**A state-based death with no held resolution and no open damage step still reaches no record.**
Such a death follows a change made outside any resolution, combat or land played. The count of
discarded events in the next subsection shows how often it happens.

### Collection counts what it drops, and gen-2's collector defects are closed before the first game

**Every discarded event is counted.** Gen-3 counts each event that arrives while no record is open,
by event type and engine phase, behind a debug flag, and runs a short pilot with the count on. The
count would have caught the first-strike loss on the first pilot.

**The defects that change what a record says are fixed before gen-3's first collection.** The nine
hook and collector defects the gen-2 record describes, the first-strike step, the withheld keyword
deleted instead of expanded, and the review's defects in the corpus build, targets and probes are
each fixed on master. [Result](#gen-2s-defects-are-fixed-on-master-before-gen-3s-first-collection)
lists the fixes.

**Three smaller gaps are closed in the same pass:**

- **A probe fork's damage events carry their source**, recovered from the payload's
  `assignment_choices`.
- **Abilities reached through Flashback or a clone's copy are keyed.** The plays they record are
  ones the model should learn from. The variant keys under the card tree are looked into.
- **The random seat stops looping on a free, repeatable ability**, and the snapshot's report of a
  set base under a boost is checked on more cards than the Gigantoplasm.

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
  mostly produce a resolution that changes nothing, which gen-2 accepted as a limit. The looping
  free activation has to be fixed first.

**The per-rule detectors decide the share.** Gen-2's records carry the random-seat flag, so the
detectors run over gen-2's corpus can count each rule per game in random-seat games and in games
between two Forge AIs. A rule found mostly in random-seat games argues for a larger share, and one
found only in rich boards argues against. Whatever share is chosen, the validation strata keep
games between two Forge AIs.

**A softer random seat is a second option.** Instead of drawing uniformly at half its decisions,
the seat would choose among the Forge AI's better options with some temperature. Its play would
stay plausible but differ from the AI's, without throwing creatures away. It depends on whether
Forge's AI exposes a ranking of its options, which has not been checked.

### Combat could get lethality copies of its own, as damage spells have in gen-2

**A combat copy would redraw the numbers of a qualifying block and label every death by the
rule.** It would set one creature's power and the other's toughness, label both creatures' deaths
and their damage taken by rules 510 and 702, and could grant a damage-step keyword with the step
marker set to match. The first-strike step makes 6% of combat records, and blocked fights are a
part of those. A keyword copy could show a first striker surviving its blocker as often as training
needs.

**Whether gen-3 adopts them depends on what the rest of gen-3 already fixes.** Four items above
change the same outcomes: the snapshot taken before the damage step, the block pairing in the input,
the whole first-strike step collected, and training that draws more of the records a weak field
needs. Combat copies are worth their cost only for what remains after those. Three questions to
settle with them:

- Which blocks qualify: single blocks only, or several blockers once the pairing is in the input,
  with damage assignment order and trample.
- Whether rule labels or Forge's outcome is the target. Gen-2's spell copies chose the rule, and
  the rules index's comparison of scenarios with Forge settles it rule by rule.
- How many copies, and how they share an epoch with the field-weighted sampling.

The combat grid measures the result on blocks from the validation strata, which copies never touch.

## Result

**Gen-2's model can learn under a third of the rules that have an outcome, and the planned inputs and
outputs would make all but 77 of them learnable.** No gen-3 corpus has been collected and no gen-3
model trained. What has run is the rules catalogue, the map of every rule to the model's inputs and
outputs, the merge of that map into a list of gen-3 fields, and the fixes to gen-2's defects on
master. A rule is learnable when an output is trained on its outcome and the model reads every fact
that decides it.

| mechanical and legality rules | count |
|---|---:|
| learnable today | 505 |
| learnable with every planned input, output and fix | 1,570 |
| left out on purpose | 77 |
| all | 1,647 |

### Gen-2's model can learn under a third of the rules, and reading every line of a permanent is the change that adds the most

**The map checks each of the 1,647 mechanical and legality rules against gen-2's input layout and
output heads, as the code builds them.** Each rule gets one of four output verdicts:

- **trained:** an existing output is trained on the outcome, as gen-2 collects and curates records.
- **recorded:** a record carries the outcome, but no output is trained on it.
- **missing:** no record carries the outcome.
- **planned:** the turn-rule plan already adds the output, as a timing or failed-check column of the
  legality verdict.

`rules/MODEL-FIELDS.md` lists what the model reads and predicts, and `rules/heads/` holds one line
per rule.

#### Three quarters of the rules reach an existing output, and under a third are learnable today

**Most rules that reach an output still depend on a fact the model does not read.**

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
| gate | 77 |

#### Reading every line of a permanent is the input that matters most, and gen-3 already plans it

**More than a quarter of the rules turn on a line the model does not read.** A rule that depends on a
static, triggered or replacement ability of a permanent other than the acting one cannot be learned:
an anthem on another creature, an enchantment that says creatures can't block, a cost reducer. For
187 rules those lines are the only gap.

**The map counts a line as read whichever card it sits on.** Gen-2's input does not say which card
that is ([Background](#the-model-cannot-tell-which-card-an-ability-line-sits-on-nor-what-a-link-between-two-cards-points-to)),
so for a rule that turns on which permanent carries a line, the count of rules learnable today is
an upper bound. The counts with every planned change assume the `line of` relation.

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
choices of the acting ability only: its targets, modes and X. It does not keep a choice made earlier
for a permanent on the board, such as the colour Gods Willing named or the creature type Cavern of
Souls named.

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
  the battlefield is therefore dropped, and with it the legend rule after a second Isamaru, Hound of
  Konda resolves.
- **Turn-based actions are dropped.** The draw in the draw step, the cleanup discard, untapping and
  a Saga's lore counter happen while no record is open, and the collector discards events that
  arrive then.
- **The zone outcome merges destinations.** Any move to a library reads as the top of the library,
  and entering the battlefield from any zone reads as a blink. A card shuffled into its library, put
  on the bottom, or returned from the graveyard to the battlefield is labelled as something else.

### 33 new inputs and 22 new outputs would make all but 77 rules learnable

**Every name the map proposed is merged into one list of fields, each with a slot, a shape and a
source.** The map named what each rule lacks one rule at a time, so it used many names for one
thing: a choice remembered by a permanent appeared as chosen values, a linked choice and stored die
results. Each name now maps to one field, to an extension or a fix, or to a reason it is left out.
Each rule's line in `rules/heads/` carries the fields it needs and whether it is learnable once they
are in. `rules/gen3-fields.json` holds the list and the merge, and `rules/GEN3-FIELDS.md` gives every
field's shape and the rules it serves.

| change | count |
|---|---:|
| new inputs | 33 |
| new outputs | 22 |
| existing inputs or outputs extended | 5 |
| already planned in this record | 7 |
| fixes to collection and curation | 3 |

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

**Copies, designation changes and the face a card ends on are the outputs the most rules need.** No
new output serves more than 40 rules. A player losing and a spell fizzling serve fewer, but for more
of their rules they are the whole change.

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

The other 14 outputs serve between 2 and 19 rules each. The five extensions change existing outputs
and inputs. The largest is the zone outcome, which 64 rules need to tell apart the destinations it
now merges. Seventeen of the 22 outputs need no new collection, because the records already carry
the event or field they would be trained on, and three more need it only in part.

**Four outputs take a shape other than the one the map named.** A player losing is one binary per
player, because in a two-player game a win is the other player's loss and a draw is both losing. A
cost reduction and a base power and toughness change are signed deltas, against the printed cost
and the previous base, like the existing power change. The legality of an attack target needs an
attackers record anchored to one attacker, as a blockers record already is, which is the third fix.

#### The 77 rules left out are subgames, casual mechanics, engine procedure and banding

**No left-out rule needs a field the design meant to keep.** Subgames and the restart of the game
(Shahrazad, Karn Liberated) and controlling another player (Mindslaver) leave out 20 rules.
Attractions, sideboards and other casual mechanics leave out 17, procedure in the middle of an
action, such as the order of pending extra turns, 15, and banding 8. The rest are characteristics
the input already shows with no change to predict, such as a colour derived from a mana cost, and a
few names of one or two rules. A rule can fall in more than one group. `rules/GEN3-FIELDS.md` lists
each with its reason.

#### The map read gen-2's targets as they were before the review changed them

**The map's verdicts predate the target changes master made after the gen-2 review.** Each output
field is now supervised where the gate fires and the field is present, a trigger record trains only
whether it fired, and `target_legal` sees only positives. A rule the map counts as trained through
the triggering event's targets, or through `target_legal`, may no longer be. The 58 rules trained
on `target_legal` are the largest group affected. `rules/MODEL-FIELDS.md` describes the earlier
targets too.

**Neighbouring rules can read one mechanism differently.** The map was judged rule by rule. A
planeswalker's and a battle's entering counters were first read as trained, while the rule beside
them filed the same counters under the refused permanent-spell resolution; the refusal decides it,
and both now read as recorded. Others may remain, so a rule's line carries a note where the
judgement was close.

### Half of the Comprehensive Rules have an observable outcome, and all but 95 of those have a scenario not yet checked against Forge

**The catalogue lives in `rules/`.** `rules/README.md` gives its formats, `rules/STATUS.md` the
per-unit checklist, and `rules/FINDINGS.md` the detail behind this section. It took the
2026-09-25 edition of the Comprehensive Rules, from magic.wizards.com, split it into units, triaged
every rule, and wrote a scenario for every rule with an outcome.

#### The rules split into 410 units, one per section or per keyword

**The edition holds 3,166 numbered rules.** The split writes one line per rule, with its text,
examples and cross-references, and one readable file per unit. A unit is one three-digit section,
such as 704, state-based actions. Sections 701 and 702, keyword actions and keyword abilities, hold
a third of all rules, so there each keyword is its own unit: 702.2 is deathtouch. Chapters 8 and 9,
the multiplayer rules and the casual variants, are out of scope without further reading, because
collection plays two-player games.

#### Triage finds half of the rules have an observable outcome, and one in eight of those needs a new record field

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
largest.** Those designations are day and night, the monarch, the initiative, dungeon progress, speed
and the city's blessing. The other families are designations on a permanent, such as renowned, a
soulbond pair or a Room's unlocked halves; characteristics the snapshot does not carry, such as the
X chosen and an Adventure's face on the stack; which layer applied which effect; what a linked
ability remembers; and what happened earlier in the turn. `rules/FINDINGS.md` maps each family to
its rules.

#### Every mechanical or legality rule but 95 has a pair of boards that flips its outcome

**Nearly every scenario pairs a variant in which the rule's effect happens with a close one in which
it does not.** Every card is named as Forge names it, and a vanilla card is preferred so that no
other ability interferes. Expected outcomes are observable only: zone moves, life, damage, counters,
power and toughness, whether a stack object resolved, fizzled or was countered, whether a trigger
fired, or whether an action was legal. Each scenario was checked against the rules text and against
the Forge script of every card it uses. Resolution and playability records carry more than half of
the scenarios.

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

**The scenarios lean on a few vanilla cards, with Grizzly Bears in about half of them.** A rule test
built from them would test the rule for Grizzly Bears only, which is why the rule tests come from
real games.

#### No scenario has been checked against Forge yet, and about 75 point at a suspected Forge departure from the rules

**Every expected outcome is a reading of the rules text and of the cards' Forge scripts.** Until
they are compared with what Forge does in the same situation, some of them are wrong at a rate not
yet measured.

**Many scenarios depend on how Forge implements the situation.** 762 scenarios carry a note about
it. Most of these notes flag engine behaviour that reading the scripts could not settle, or describe
how a card is scripted. About 90 need setup state that a starting-state file may not express, such
as a designation or a turn's history. About 75 name a suspected Forge departure from the rules, such
as connive 0 firing the "connives" trigger. `rules/FINDINGS.md` lists ten to test first.

### Gen-2's defects are fixed on master before gen-3's first collection

**Every defect that changes what a record says, how a field is trained or what a probe reads is
fixed before gen-3's first game.** The subsections below list the fixes, from the collector to the
probes.

#### The Forge hooks and collectors no longer record planned events as real ones

**Gen-3's hooks and collectors fix all nine defects the gen-2 record describes.** The fixes are
commits on the `effect-record-hooks` branch of `../forge` and in the connector.

**Block and attack legality are asked by Forge's declare steps, before anyone declares.** The hook
lists every creature the player controls as a candidate, and the combat holds no blocks yet when it
is asked. A creature planned for one block can no longer read as forbidden for another. A tapped or
"can't block" creature is always listed as forbidden. The answer is the same whatever controls the
seat, so the random seat no longer writes legality records of its own, and the AI's internal
legality queries are no longer hooked. Every legality record is therefore a real declaration, one
per defender or attacker per combat. `--legality-rate`, which sampled the AI's what-if queries, now
has nothing to sample; removing the flag is left to the gen-3 collection work.

**A reveal from a library during a resolution names the players whose libraries it came from.** A
scry already names the player who scried in the same way. A reveal from a hand, a graveyard or any
other zone names nobody. So does a reveal made while paying a cost, which happens before the spell
it pays for resolves. A clash names both clashing players.

**Each kind of object an effect makes gets its own note, and a token's note carries its keywords.**
Bestial Menace now records a Snake, a Wolf and an Elephant. The collector writes a token's script id
and keywords with the code that writes every token on the board, so the created-objects field's
keyword flags have something to learn from.

**Prevented damage names the creature or player it would have been dealt to.**

**A note that names no subject marks nothing affected.** Coin flips, token creation, copies, votes
and ventures no longer mark the acting card as affected.

**A combat record carries its own damage step's assignment, as the players made it.** Forge copies
the assignment the moment it is complete, before prevention or redirection rewrites the damage, and
holds it until the next step's assignment replaces it. The collector writes that copy whole. A
double striker therefore shows its assignment in both steps, a prevented hit still shows the damage
assigned, and the observed record and its probe fork read the same thing.

**A combat record stops where the stack starts.** The first ability or spell to go on the stack in
a damage step writes the step's combat record, and nothing after it joins one. A combat-damage
trigger's draw, token or discard is written in the trigger's own resolution record, and damage a
spell deals in the window is not combat damage.

**Trigger records describe only events that happen.** The engine's query of which triggers an event
would fire no longer notifies the hook, so the AI's questions about taps it is planning write
nothing. A static trigger is also evaluated once per event rather than once when its event is
collected and again when it runs.

**A prevention that prevented nothing writes no note.** A shield whose amount is used up no longer
records a prevention of 0.

**The first-strike step is recorded whole.** The collector now tests both damage-step names, and a
collector test fixes the case.

**The verdict bits are computed from the rules.** `PlayabilityVerdict` computes `can_play`,
`affordable` and `has_legal_target` for both seats without the AI's cost preferences. The timing
bit and the failed checks are not built yet.

**Two bits of the `library_events` field are still never set.** The target mapping writes the
`scry` bit for both scry and surveil, and nothing writes the `surveil` or `tutor` bit. The mapping
is in `src/effects/domain/effect_targets.py` and is left to the gen-3 target work.

#### The withheld keyword reaches training as its definition, so gen-3's zero-shot slice is the first that tests reading one

**Gen-3 runs force-expand the withheld keyword wherever it appears.** One tokenizer method expands
the withheld keyword as it would a keyword unknown to the vocabulary. It then drops the keyword's
token everywhere, including inside the reminder text of another keyword that names it, such as
reach's mention of flying. Training, evaluation and `encode-abilities` all tokenize through it. The
cache row of the withheld keyword is therefore its definition read by the trained encoder. Gen-2's
cache encoded cascade's line from a token training never updated ([gen-2
record](2026-09-18-effect-model-gen2-improvements-design.md#cascades-line-in-every-gen-2-cache-was-encoded-from-a-token-training-never-updated)).
A run without `--withhold-keyword` takes exactly the draws it took before.

#### The corpus build, the targets and the probes were reviewed, and gen-3 trains without the defects found

**All of the review's defects are fixed on master before gen-3's first collection.** Several fixes
change what a field means or how the loss weighs it, so no gen-3 number is comparable with gen-2's
on those fields.

##### What the model trains on

**An output field is scored only where its record says something about it.** Each field is
supervised where the affected gate fires and the field is present. A field is present when the
record writes it, or when the field has a silent value and the record's class supervises it. The
silent values are the ones whose absence means nothing happened: zone stayed, no damage, no
counters, no P/T change, no type, colour or keyword gained or lost, no control or attachment change,
and no change to a player's life, cards, library, poison or mana. `tap_state`, both durations, the
legality bits and `min_blockers` have no silent value and train only where an event or a decision
payload sets them. Player slots are never scored on permanent fields, nor permanents on player
fields. Continuous records now also supervise P/T, keyword, type and colour changes. The
constant-prediction floor reads the same presence, so explained deviance measures the same thing the
loss trains.

**`target_legal` now has nothing to learn from.** A decision payload lists a spell's legal targets
and never its illegal ones, so with presence the field sees only positives and its floor is
undefined. Target legality is learned through the affected gate, which a decision record raises
exactly on its legal targets, and the probe ladder already reads it there. Gen-3 either drops the
field or has the collector record the entities a target restriction excluded.

**A trigger record supervises only whether the trigger fired.** It produces no per-entity targets.
The triggering event stays in its input as the pending-event overlay.

**The value head reads static boosts and tells placing counters from removing them.** `AddPower$`
and `AddToughness$` are read, and a value that is not a literal number is masked. Counter removal
has a target of its own, `counters_removed`, because one chain can both put and remove counters.
Moving or adding-or-removing counters masks both targets. Where a script omits an amount, the value
is Forge's own default: a draw draws one card, and putting or removing counters moves one.

**A masked number is masked in the numeric input too.** The masked-language-model term zeroes the
numeric value and its flag wherever it masks a token.

**Count targets are clamped at zero.** Healing subtracts from `damage_taken`, and a negative count
made the Poisson loss unbounded below.

**Every head's loss is reduced the same way.** Each row's loss is summed over its targets and then
averaged over rows: a record for the effect, created-objects and verdict heads, a text for the
value, API and masked-language-model heads. Before, the API-key term was averaged over its 1,245
key columns, which made it about a thousandth of the API-type term under the same weight. With the
sum, the API-key term made up close to a third of the initial loss at gen-2's API weight. The default
API weight is therefore 0.003, which puts the weighted key term between the value and
masked-language-model terms. At that weight the API-type term carries almost nothing. Separate
weights for types and keys belong to the loss redesign ([one scale for every
field](#the-training-loss-puts-every-output-field-on-one-scale-with-amount-distributions-that-fit)).

| term, one 32-record batch at initialisation | weighted, API weight 0.05 | weighted, API weight 0.003 |
|---|---:|---:|
| per-entity fields | 67.0 | 67.0 |
| created objects | 30.7 | 30.7 |
| API keys | 44.5 | 2.7 |
| masked-language model | 6.6 | 6.6 |
| value head | 1.0 | 1.0 |
| verdict head | 0.8 | 0.8 |
| API types | 0.3 | 0.02 |

##### The corpus build

**The training plan counts only records the build will train on.** The survey routes each game once,
at the end of its shard: held out, else drawn for game-disjoint, else training. Only training games'
records size the class, family and text quotas, and the write pass applies the same routing. Rarity
keeps counting the whole corpus. The manifest records each class's target beside its delivered count
and warns when a class falls short. A seven-shard smoke build delivered 99.8–102.7% of every class's
target.

**A held-out text's game cap counts only resolutions.** A held-out game is admitted to the
card-disjoint stratum only for the held-out texts it resolves, so every admitted game gives gate 1 a
record.

**The holdout covers token and variant scripts.** A token script carrying a held-out text counts as a
carrier and is held out by its script file. `collect-variants` drops any variant that prints a
held-out text or template. `build-corpus` drops a training record acting through a held-out text,
and counts the drops as their own stratum.

**Training shards are dealt by game.** Repacking assigns each game to a shard by a seeded hash of its
`game_id` and shuffles each shard, so a shard mixes games from every run and source.

**Three build faults that gen-2 never hit are fixed.** An unjoinable key on a decision's candidate is
refused rather than crashing the plan. A curated shard keeps the generation of the raw records it
came from. A build that fails partway leaves no manifest whose digest still matches the checkpoints.

##### Encoding and the baselines

**The `no-state` baseline keeps the acting ability.** It zeroes every ability vector on the board
except `[ACT]`'s, all card and player features, and every `[GLOBAL]` input except the record-kind,
moment and subkind flags.

**The identity baseline gives each text its own row.** Its table is indexed exactly by text, built
from the sidecars, with one shared row for texts it never saw. The 2^17-row hash table it replaces
put more than a quarter of the texts in a row with another.

**The prose surface works.** Value and API targets are read from the line's script rather than its
prose. Lethality copies edit the prose amount and shift the role spans with it. Role spans reach the
encoder on the prose surface.

**Training records its seed on the checkpoint, and gradient accumulation counts its steps within the
epoch.**

##### Evaluation and probes

**Gate 2 reads every field through the affected gate.** A count or amount is read as the gate's
probability times the field's value, and a category as the gate's probability times the class's.
Gate 2, the sweeps and the zone probes share this reading. The probe ladder's `damage_taken` target
still reads the amount alone, because every one of its items took damage.

**Gate 2 counts records from the curated corpus and leaves out the forks.** It reads the curated
game-disjoint stratum and skips probe forks and interventional records. Its 200-record threshold
counts distinct records. Agreement is averaged over each record's carriers, because each carrier is
its own perturbation.

**Gate 1 takes its seen texts from the curated training stratum.** It streams the card-disjoint
stratum and keeps only resolution records of unseen texts. A checkpoint without a curated corpus
skips gate 1 and says why.

**The checkpoint names the corpus, sidecars and variant tree that every probe and grid reads.** A
probe set frozen against another corpus is refused.

**A probe line is held out only when every card carrying it is held out.** A text printed only on
held-out cards never trained, even when the text itself is not held out.

**The embedding probes count each record once.** The profiles de-duplicate the training stratum's
copies by `record_id`.

**Three unbuilt checks are reported as not implemented.** The evaluator no longer says the corpus
lacks the forks they need.

**`scorer-smoke-test` refuses a scratch directory that already holds vectors.** A rerun in the same
directory no longer splices `e` in a second time.

#### The observed-mana probe counts the mana each event names

**The knowledge probes' observed-mana target sums each event's `mana_by_color`.** The collector's
map is the mana actually produced, not the colours a land could make. It counts the colour letters
of the mana string Forge reports once the mana is in the pool, after replacement effects
(`PatchedCollectors.manaByColor`). Every multi-colour map in a sample of gen-2 events came from a
source that makes two or more mana at once, such as Orzhov Basilica's `{W: 1, B: 1}`.

**An event whose map is empty counts as unreadable.** It adds nothing to the target, and a record
whose mana events are all empty gets no label. The collector writes no map at all when it cannot
read the mana, so the empty maps seen in gen-2's events are not yet explained, and gen-3's first
pilot should count them.

## Future plans

**The next steps turn the catalogue and the map into collection and training.**

- **Rerun the map against master's targets**, then make the input and output changes in the order
  the Description sets, rerunning the map after each to count what it made learnable.
- **Run the per-rule detectors over gen-2's corpus.** The counts say which rules self-play covers,
  which need directed forks, and how large a share of matches the random seat should get.
- **Classify the scenarios' variant changes into fork operations**, build the few that reach the
  most rules, and leave puzzle-style setups for the rules that remain.
- **Compare the scenarios with Forge**, starting with the ten suspected Forge departures in
  `rules/FINDINGS.md`.
- **Build the collection items not yet in**: every line of a permanent, the timing bit and failed
  checks, the owner field, the combat snapshot at the start of the damage step, the entities each
  static affects, permanent-spell resolutions, turn-based action records, anchored attackers
  records and the dropped-event count.
- **Build the relation-aware trunk at 256 wide before gen-3's first training run**, its tests
  written before it, and measure its step time and peak memory on a fixed shard.
- **Train with the hard mask on rule-drawn records first**, and move to the soft mask only if a
  comparison shows the hard mask costs the other outputs.
