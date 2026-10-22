# Ability effect model — gen-2

This record covers gen-2, the second generation of a model that learns what Magic: The Gathering abilities do by watching Forge play. Forge is the open-source Magic rules engine, and its AI plays both sides of the sealed games the model learns from. Every ability on a Forge card is a line of script, such as `SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Creature`. An encoder reads that script and outputs one vector `e`, the ability's embedding. An effect head reads `e` beside a snapshot of the board and predicts, for every card and player, what the ability changes: which creatures die, how much damage each takes, whose life changes, which keywords are gained. The training data are effect records, one per observed event such as an ability resolving or a combat damage step, each holding the board before the event and what the event changed.

Gen-2 changes what the encoder reads, collects a new corpus with a seat that sometimes plays at random, and trains a sweep: arms, runs from scratch that differ in the width of `e`, and continuation runs that carry them further. This record judges those changes and the rules they targeted: the death rule and its boundary, combat, gate 2, how a checkpoint is chosen, the losses and the width. Its sibling, [the knowledge-probes record](2026-09-19-effect-knowledge-probes-design.md), reports where in the model each kind of knowledge lives, and what the model knows of evasion, targeting, colour and zones. The results come first, organized by question and read through the measures that [the section after the summary](#seventeen-checkpoints-from-twelve-runs-read-on-two-validation-strata-kept-out-of-training) defines. The design of each change follows them, and the runs, the fixes they forced and the defects found afterwards come last, with the data in [`2026-09-18-effect-model-gen2/`](2026-09-18-effect-model-gen2/README.md).

## What gen-2 found

**The model learns that damage kills a creature only when training shows it the rule directly.** Forge's AI aims its removal at creatures it can kill, so nearly every damage spell in the corpus is lethal, and the first arms predicted deaths from a creature's toughness alone. Lethality copies, training copies of qualifying records with the damage and the toughness redrawn and the death labelled by the rule, gave the model the damage-versus-toughness comparison. The copies make up 2% of the records trained and cost nothing on the validation loss. Where a checkpoint still misses a lethal hit, as on activated 1-damage pings, it mostly doubts that the hit creature is affected at all, and that doubt follows the ability's wording and how rarely training shows it ([the death rule](#the-model-compares-damage-with-toughness-only-after-training-shows-it-the-rule-directly)).

![Sixteen-panel grid of predicted death by damage and toughness for a damage spell's target, one row per width of e and one column per stage of training](2026-09-18-effect-model-gen2/charts/damage-grids.png)

*Each cell is the predicted death of a damage spell's target, averaged over 20 real records with the spell's damage and the target's toughness edited. Rows are the width of `e`; columns run from the arm without copies through the arm with copies to each continuation run's kept and last epoch. Orange cells predict a death more likely than not and blue cells less likely, brighter meaning more likely on both sides. The rule says the target dies on and below the black staircase, where the damage is at least the toughness. Each panel's error is the mean absolute difference from the rule's 0 or 1 over its 64 cells; run B256's last epoch has the lowest, in bold. Data: `damage_grids.csv`.*

**Where the lethal boundary sits wanders from epoch to epoch at the full learning rate, and settles near the rule at a tenth of it.** Twenty more epochs at the same rate sharpen the boundary but move its position by up to a point of damage between consecutive epochs. A tenth of the rate holds it in a narrow band near the rule, from whichever checkpoint the run starts ([the boundary](#where-the-boundary-sits-wanders-between-epochs-at-the-full-learning-rate-and-settles-near-the-rule-at-a-tenth-of-it)).

**The validation loss and the probes disagree about which checkpoint is best.** In every continuation run, the probes rate the last epoch as good as or better than the epoch with the lowest card-disjoint loss. The loss rises late in training because the model grows more confident on texts it never trained on, mostly about whether a card is affected at all and about poison, and not because it ranks outcomes worse. Gen-3 judges checkpoints by probes instead ([validation against probes](#the-probes-keep-improving-after-the-card-disjoint-loss-has-bottomed-out)).

**In combat, the model applies power against toughness softly and learns indestructible and most of deathtouch, and the grid that finds no first strike may understate it.** The combat grid kept each base record's damage assignment when it edited the board, so its first-strike and power readings ask about boards that contradict their own input. The corpus also lost every real blocked fight of the first-strike damage step to a collector bug, so only the probe forks showed a first striker surviving the creature it kills first, too few to teach the step ([combat](#in-combat-the-model-applies-power-against-toughness-softly-and-learns-indestructible-and-most-of-deathtouch-and-the-grid-that-finds-no-first-strike-may-understate-it)).

**Three combat keywords hold on every checkpoint, and double strike swings far beyond its sampling noise.** Lifelink, indestructible and trample agree with real outcomes well above gate 2's bar on every checkpoint. Double strike fails and passes between checkpoints of one run, by several times what its records allow by chance ([gate 2](#three-combat-keywords-hold-on-every-checkpoint-and-double-strike-swings-far-beyond-its-sampling-noise)).

**Validation losses swing between epochs by about as much as a run improves, and a tenth of the learning rate cuts the swing several times over.** A difference of a tenth of a loss point between two checkpoints at the full rate is within that swing. The output fields with many records plateau by epoch 20, and the small ones, such as poison, swing more than they improve ([training](#validation-losses-swing-between-epochs-by-about-as-much-as-a-run-improves)).

**No width of `e` leads on every measure, and the three widths do not line up in order.** The 64-wide line has the lowest loss on unseen texts at every stage of training. The 256-wide line has the lowest loss on known texts in new games and, at a tenth of the learning rate, the best probes by a small margin; its continuation checkpoints also rank best on average over thirteen measures. The 128-wide line has the weakest probes at every stage, which points to the seeds rather than to the width ([width](#the-64-wide-and-256-wide-lines-split-the-measures-and-the-128-wide-line-trails-on-the-probes-at-every-stage)).

**Gen-2's `e` spreads over two to three times as many directions as gen-1's, and a linear probe reads a line's kind and amounts from it far better** ([embedding probes](#gen-2s-e-spreads-over-two-to-three-times-gen-1s-directions-and-reads-a-lines-kind-and-amounts-far-better)).

**Where knowledge lives, the probes record finds that `e` carries what a line states and the trunk works out what the board decides.** The board and `e` alone, without the trunk, give nearly all of what the model knows about whether an ability is affordable and how long a pump lasts. On deaths, trigger firing, targets and damage taken, a probe on the trunk reads as much as the model's own answer, and the probes cannot say whether `e` holds what those questions need. On deaths, `e` adds only a few hundredths of AUC over a random vector of its width, and nothing at width 256. `e` holds at most a sixth of how much a text's outcome varies across boards on trained lines and none on unseen ones, less than gen-1's ([the probes record's summary](2026-09-19-effect-knowledge-probes-design.md#what-the-probes-found)).

**Gen-2's first arm trailed gen-1 on the part of the loss the two generations share, in the fields whose targets gen-2 made larger or new** ([gen-2 against gen-1](#the-first-arm-trailed-gen-1-on-the-loss-the-two-generations-share-in-the-fields-whose-targets-gen-2-made-larger-or-new)).

**Collector defects shaped several of gen-2's labels, and qualify some of the results above and in the probes record.** The largest: more than half of the corpus's "may not block" labels name a creature the AI had already planned to block with, though no rule kept it from blocking ([the collector defects](#collector-defects-taught-the-ais-block-plan-as-a-rule-and-bent-the-affected-gate-and-the-combat-input)).

**The zero-shot keyword check measured nothing.** Every arm deleted cascade, the keyword withheld from training, instead of expanding it to its definition, so the check compared records missing an ability with records whose abilities trained ([the withheld keyword](#every-arm-deleted-the-withheld-keyword-instead-of-expanding-it-so-the-zero-shot-check-measured-nothing)).

## Seventeen checkpoints from twelve runs, read on two validation strata kept out of training

Every result below reads one or more of seventeen gen-2 checkpoints, a checkpoint being the saved weights of a model at one epoch of training. An epoch is one pass of the training schedule: 5,000 steps of 32 records each. The subsections define the model's parts, the data each measure reads, the measures, the runs, and the size of a difference that counts.

### The trunk reads each ability's `e` in a slot of its own beside the board

The effect head is a transformer: a network that reads a set of input vectors, its slots, and lets each output depend on all of them. Its shared body is the trunk, and small output heads read the trunk to produce each predicted field. The trunk reads a slot for the game, one per player, an `[ACT]` slot holding the acting ability's `e`, and for each object in the snapshot a `[CARD]` slot followed by the `e` of that object's spell and activated abilities. Statics, triggers and replacements on the board have no slot. `e` is the part meant for reuse: deckbuilding and drafting models read one fixed `e` per ability without any board, and a future game-playing agent would read it as its ability representation.

![Diagram of the effect model. A Forge script passes through a tokenizer and the ability encoder to a vector e of width 64, 128 or 256. In training only, noise is added to e, an MLM head reads the encoder's tokens, and value, API-type and parameter-key heads read e alone. e feeds the [ACT] slot and the ability slots of each object's block in the trunk's input, beside [GLOBAL], two [PLAYER] slots and [CARD] slots. Statics, triggers and replacements are never in a block. A six-layer trunk feeds the created-objects, verdict and per-entity heads. Numbered badges mark where read-out ladder rungs 0, 1, 1w, 1o, 2 and 3 read, and a panel shows the three ablation scopes.](2026-09-18-effect-model-gen2/charts/diagram-model-and-probes.png)

*Dashed parts exist only in training. The MLM head, a masked-language-model head, trains the encoder to restore hidden words of the script. The rose badges mark the input each rung of the probes record's read-out ladder probes, and the slot rows at the bottom right show which slots each ablation scope replaces; [the probes record](2026-09-19-effect-knowledge-probes-design.md) defines both.*

### Two validation strata keep whole games out of training, and one of them holds texts the model never trained on

The card-disjoint stratum holds games containing a held-out card, whose ability texts the model never trained on. The game-disjoint stratum holds whole games whose ability texts also appear in training. A validation loss is the training loss measured on a fixed sample of 2,048 records from one stratum, the same records every epoch. Each run keeps the epoch with the lowest card-disjoint loss, called its kept epoch. The knowledge probes read a frozen probe set drawn from both strata, so every checkpoint is probed on the same items.

![Diagram of how games are split. A game with any record naming a held-out card goes to the card-disjoint stratum; otherwise a hash of its game id sends a small fixed share to the game-disjoint stratum and the rest to training. Both strata feed a fixed validation sample and the frozen probe set.](2026-09-18-effect-model-gen2/charts/diagram-validation-strata.png)

*Whole games move together, never single records, because the records of one game share a board. Held-out texts are picked by a hash of each text's masked template, and training games are played from pools without any held-out card. The card-disjoint loss chooses the epoch a run keeps.*

### Three gates and a probe mean score every checkpoint

The three gates come from gen-1's design. Gate 1 compares the encoder with a baseline that memorizes each text, and gen-2 does not run it, because it trains no such baseline. Gate 2 checks that combat keywords move predictions the way they move real outcomes. Gate 3 checks that `e` does not collapse onto a few directions.

The knowledge probes are recorded in [the probes record](2026-09-19-effect-knowledge-probes-design.md). A probe asks a model a targeted question, such as whether a creature dealt 3 damage at toughness 4 dies, by editing a real record and reading the prediction, or by fitting a small model to read a known property from the model's vectors. A yes/no probe is scored by AUC, the chance that a true case is ranked above a false one, so 0.5 is chance. An amount is scored by R², the share of its variance the probe explains.

The probe mean summarizes a checkpoint in one number. It is the mean AUC of the model's own answer, read from its trained output heads with no probe fitted, on eight questions over the game-disjoint probe records: whether a card is affected, whether it dies, whether a trigger fires, whether an ability is affordable, whether a target is legal, whether a target carrying protection is legal, whether a creature may block, and whether a pump lasts until end of turn. Four of the eight read labels that gen-2's defects distort. May block, whether a card is affected and whether an ability is affordable carry collector defects ([the collector defects](#collector-defects-taught-the-ais-block-plan-as-a-rule-and-bent-the-affected-gate-and-the-combat-input)). May block and pump duration also read fields whose unset targets were trained as 0 ([the zero-filled targets](#every-affected-entity-was-scored-on-every-output-field-and-a-field-with-no-event-was-taught-as-0)).

### A checkpoint is named after its run: an arm, then an A run at the full learning rate, then a B run at a tenth of it

Gen-2 trains in three stages. An arm is a run from scratch, named after the width of `e`: `e64`, `e128` and the others. An arm whose name ends in `-lethality` trained with lethality copies. Arm `e64 (earlier inputs)` is `e64`'s first run, on the model inputs as they were before three corrections that every later run uses. An A run resumes its arm from the arm's last epoch for twenty more epochs at the same learning rate, the step size of each weight update. A B run starts from its A run's kept epoch and trains twenty epochs at a tenth of that rate.

Each continuation is probed at its kept epoch, as `A64 best`, and at its last, as `A64 last`. Run A256 reached its lowest card-disjoint loss at its last epoch, so it has a single checkpoint. Run B256 never went below the checkpoint it started from, so it keeps run A256's epoch 53; its own lowest epoch, 55, is probed as `B256 best`. Gen-1 is the checkpoint of the previous generation. The full run table is in [Details](#the-runs-and-the-checkpoints).

![Gantt chart of the gen-2 runs by epoch, one bar per run grouped by width of e, with arrows from each continuation to the weights it starts from and markers at each probed checkpoint](2026-09-18-effect-model-gen2/charts/run-lineage.png)

*Each bar is one run. An A run resumes its arm after the arm's last epoch; a B run starts from its A run's kept epoch at a tenth of the learning rate. Triangles mark arms, squares A runs and circles B runs; a filled marker is a run's kept epoch and a hollow one its last, and `B256 best` is half filled. Grey bars trained without lethality copies, and the hatched bar is `e64 (earlier inputs)`. Data: `runs.csv`, `checkpoints.csv`.*

### Each run is one seed, so a difference between runs counts only where it exceeds the swing between epochs

Every run is a single training run with one seed, so two runs can differ by chance alone. A seed fixes a run's initial weights, the order of its data, its dropout and its noise. Dropout switches off a random share of the network's units at each training step. Three sources of variation make two equally good runs score differently, and one run measures only two of them.

| source | what varies | measured from one run |
|---|---|---|
| validation sample | the card-disjoint loss of one model on another sample of held-out texts | yes, by resampling the held-out texts |
| movement within a run | the loss from one epoch to the next as the weights keep changing | yes, from the epochs on the plateau |
| seed | the level a run settles at | no; it needs a second run |

The epoch-to-epoch swing sets a floor, not the whole margin. On arm `e64 (earlier inputs)` the plateau's swing was about a tenth of a loss point, and [the training results](#validation-losses-swing-between-epochs-by-about-as-much-as-a-run-improves) measure it on every run. Seed variance is usually larger, because a run with another seed tends to sit at another level for the whole run rather than only to wobble differently. Consecutive epochs start from each other's weights, so their spread also understates the noise in their mean.

| arm `e64 (earlier inputs)`, epochs 14 to 20 | mean | standard deviation |
|---|---:|---:|
| card-disjoint loss | 4.06 | 0.11 |
| of it, the output fields | 3.58 | 0.12 |

The runs are therefore compared by agreement across measures. A width that wins on both losses, on the probes and on the death-rule and combat probes is more convincing than one that wins on a single number, although all of them come from the same run. The scorecard in [the width section](#the-64-wide-and-256-wide-lines-split-the-measures-and-the-128-wide-line-trails-on-the-probes-at-every-stage) is that comparison. Two stronger checks were designed and not run: a comparison of two checkpoints paired on the same held-out texts, with a confidence interval bootstrapped over texts, and a second seed of one run, which would measure the spread between seeds.

## Results

### The model compares damage with toughness only after training shows it the rule directly

Without help, the model predicts a creature's death from its toughness alone. The toughness sweep edits one hit creature's toughness from 1 to 8 on 20 real damage records and reads the predicted death. In arm `e64`, a 1-damage spell and a 3-damage spell give the same curve, falling between toughness 4 and 6. Each curve should instead fall just after its own ability's damage, at the dotted line of its colour.

![Toughness sweep by damage group, one row per width of e and one column per stage of training: the range of records as a band, the median as a line](2026-09-18-effect-model-gen2/charts/toughness-sweep-records.png)

*One row per width of `e`; the columns run from the arm without copies through the arm with copies to each continuation run's kept and last epoch. Each band spans the lowest to the highest record of one damage group at each toughness, and the solid line is the group's median; the count of records is in the legend. The rule says a group stops dying at its dotted line, and the black line marks an even chance. The 1-damage spell group and the activated-ping group share the line at 1.5, drawn in alternating blue and purple dashes. An activated ping is a permanent's paid ability that deals 1 damage, such as Prodigal Sorcerer's "{T}: deal 1 damage to any target"; the 1-damage spells and triggered abilities form their own group. Data: `sweep_records.csv`, `toughness_sweep_records.csv`.*

The corpus rarely shows a damage spell falling short of its target's toughness. Forge's AI aims its removal at creatures it can kill, so a model fitting the corpus learns that "a small creature hit by removal dies" and has no reason to compare the amount with the toughness. Combat shows many more short hits, but there both numbers are on the board; a spell's amount reaches the trunk only through `e`.

| damage in 40 training shards | lethal | not lethal |
|---|---:|---:|
| a spell or ability's damage to a targeted creature | 170 | 36 |
| combat damage to a creature | 3,252 | 1,314 |

The fix is training copies labelled by the rule. A record qualifies when its acting ability deals one stated amount of damage, or -X/-X, and nothing in its chain can change whether the creature dies. Each copy redraws the amount and the hit creature's toughness from 1 to 8 and labels the death by rule 704.5f-g, so about half the copies are lethal. About one training record in four hundred qualifies, and `--lethality-copies 8` makes the copies about 2% of the records an epoch trains on. The target is the rule as written, not Forge's AI, so a label taken from the rule is what the model should learn.

With the copies, most lines fall at or within one point of their own damage, and the damage-by-toughness grid at the top of this record gains its diagonal. The card-disjoint loss did not get worse, so the copies cost nothing on the objective.

Three faults remain, and the grids show all of them:

- **The boundary is soft.** Where the damage exactly equals the toughness, the prediction sits mostly between two fifths and four fifths rather than near one.
- **The lethal side tops out near 0.85 on the arms and near 0.9 on the continuation runs.** That ceiling comes from the gate, the model's predicted chance that a card is affected at all, rather than from the death. On clearly lethal hits the predicted death given that the creature is affected, the quantity [the boundary chart](#where-the-boundary-sits-wanders-between-epochs-at-the-full-learning-rate-and-settles-near-the-rule-at-a-tenth-of-it) plots, is 0.99. On a quarter to a third of the training records the arms doubt that the hit creature is affected at all ([the gate](#where-the-model-doubts-that-a-hit-creature-is-affected-the-text-leaves-the-victim-open-or-is-rare-in-training)).
- **Activated pings at a toughness-1 creature are predicted to spare it on most 64-wide checkpoints and on arm `e256-lethality`.** Arm `e64 (earlier inputs)` predicts the death, and the 128-wide checkpoints and run A256 do too. `B256 best` and `B256 last` pass only narrowly. The 256-wide line loses the fault between its arm and its continuation, so it is a property of each checkpoint rather than of the width. Much of the fault is in the gate, because training holds few activated "any target" pings.

| activated 1-damage pings at toughness 1, median predicted death | checkpoints |
|---|---|
| 0.90 | `e64 (earlier inputs)` |
| 0.06 to 0.24 | `e64`, `e64-lethality`, A64 best and last, B64 best and last |
| 0.10 | `e256-lethality` |
| 0.55 and 0.62 | B256 last, B256 best |
| 0.68 to 0.93 | every 128-wide checkpoint, A256 |

*Data: `sweep_records.csv`, `toughness_sweep_records.csv`.*

The damage sweep reads the same rule from the other side: it fixes a toughness-4 target and re-encodes one spell's script with `NumDmg$` from 1 to 8. The rule puts the step between 3 and 4. Arm `e64` barely rises with the damage. With copies, every checkpoint rises steeply, and where it crosses one half moves from checkpoint to checkpoint, which [the boundary section](#where-the-boundary-sits-wanders-between-epochs-at-the-full-learning-rate-and-settles-near-the-rule-at-a-tenth-of-it) explains.

![Damage sweep curves in four panels, the checkpoints without lethality copies and then one panel per width, over grey quadrants marking where a prediction falls on the wrong side of the rule](2026-09-18-effect-model-gen2/charts/damage-sweep.png)

*The damage sweep for every checkpoint. The first panel holds the checkpoints trained without lethality copies, gen-1 among them; the other three hold one width each. Grey quadrants are where a prediction falls on the wrong side of an even chance: a death predicted below 4 damage, or a survival from 4 up. Dotted triangles are the arm, dashed squares the A run and solid circles the B run; a filled marker is a run's kept epoch and a hollow one its last. `B256 best` is half filled, and gen-1 is a dark grey cross. A checkpoint with every point on the rule's side is ticked in the legend and drawn bold: arm `e64-lethality`, run A256 and all six B-run checkpoints. Data: `sweeps.csv` (sweep `damage_spell`).*

#### Where the model doubts that a hit creature is affected, the text leaves the victim open or is rare in training

On the arms, a quarter to a third of the boundary chart's 40 training rows put the gate below an even chance at the hit creature, though the creature was hit in every one. The B runs keep three or four such rows, and in runs B64 and B256 every one is a triggered ability. The doubt falls on three kinds of text.

| acting line | rows | `e64-lethality` | `e256-lethality` | run B64, last | run B128, last | run B256, last |
|---|---:|---:|---:|---:|---:|---:|
| spell | 23 | 8 | 4 | 0 | 1 | 0 |
| activated ability | 8 | 1 | 1 | 0 | 1 | 0 |
| mode of a modal spell | 4 | 0 | 0 | 0 | 0 | 0 |
| triggered ability | 5 | 5 | 5 | 4 | 1 | 3 |
| all rows | 40 | 14 | 10 | 4 | 3 | 3 |

*Rows whose gate at the hit creature is below one half, per checkpoint. The 40 rows hold 31 distinct records. The curated corpus writes a rare text's records several times, and the probe takes the rows as it finds them, so three Suffocating Blast resolutions fill twelve rows. Data: `boundary_records_gate.csv`.*

- **A victim left to a choice.** Purging Scythe deals 2 damage to the creature with the least toughness, and its controller picks among ties. When several creatures share the least toughness, doubting each of them is the right answer.
- **A spell with two targets.** Suffocating Blast counters a spell and deals 3 damage to a creature. Arm `e64-lethality` puts the creature's gate between a third and two fifths, and run B256's last epoch near certainty.
- **Activated "any target" damage abilities**, such as Prodigal Sorcerer's ping. The checkpoints that miss the pings in the toughness sweep doubt their target.

The pings' doubt comes from their wording, not from the board. The probe set's toughness sweep holds five activated damage records, four of them 1-damage pings. Each was read again after one edit to the record or to the acting text. Only replacing "any target" with "target creature" lifts the gate above nine in ten on every checkpoint read. Moving the record to the main phase or untapping the source moves it by a tenth at most. Reading the line as a spell lifts it on run B64's last epoch and lowers it on arm `e256-lethality`. A healer's "prevent the next 1 damage" text in the ping's place puts the gate at zero, as it should, since a prevention shield changes nothing on the board.

![Dot plot of the gate at the targeted creature under six edits of five activated damage records, one marker per checkpoint for four checkpoints, with the rule's answer as a black tick](2026-09-18-effect-model-gen2/charts/ping-gate-edits.png)

*The gate at the targeted creature, at the record's own toughness, averaged over the five records. Four checkpoints were read: arms `e64-lethality` and `e256-lethality` (triangles) and the last epochs of runs B64 and B256 (circles). The shaded row is the only edit that puts every checkpoint above 0.9. The bottom row is a control whose rule answer is 0. Data: `ping_gate_edits.csv`.*

The input and the labels are not the cause. Every one of these records lists the hit creature as the ability's target, and training marks the creature targets of resolved activated damage abilities affected eight to nine times in ten. What training lacks is volume. The first 150 of the 444 training shards hold 24 resolved "any target" damage abilities with a tap in their cost, so the whole corpus holds about 70, about 45 of them at a creature. That count already includes the curated corpus's repeats of rare texts. "Any target" damage spells resolve about seven times as often, and their creature targets are always marked affected. With so few examples, whether a checkpoint ties the targeted creature to the gate for this wording comes down to its fit, which is why the fault comes and goes between checkpoints of one width.

| acting line, 150 training shards | resolutions | targets that are players | targets that are creatures | creature targets | marked affected |
|---|---:|---:|---:|---:|---:|
| spell, any target | 182 | 21% | 79% | 115 | 100% |
| spell, target creature | 178 | 1% | 99% | 168 | 98% |
| activated, any target, tap in its cost | 24 | 35% | 65% | 15 | 87% |
| activated, any target, other cost | 16 | 43% | 50% | 7 | 86% |
| activated, target creature, tap in its cost | 10 | 0% | 100% | 10 | 80% |
| activated, target creature, other cost | 13 | 0% | 100% | 13 | 92% |

*Data: `damage_targets_in_training.csv`.*

The probe set's triggered damage records add a fourth kind, where the input cannot identify the victim. Skewer Slinger deals 1 damage to the creature it blocks or is blocked by. Both of its records have several creatures in combat, and the card features say that a creature blocks and how many creatures block it, not which attacker each blocker blocks. Run B256's last epoch gives the two victims a gate of 0.13 and 0.02. The gen-3 draft proposes the pairing as an input ([gen-3 draft](2026-XX-XX-effect-model-gen3-design.md#the-board-input-should-say-which-attacker-each-blocker-blocks)), and examples of their own for rare wordings like the pings ([gen-3 draft](2026-XX-XX-effect-model-gen3-design.md#a-rare-wording-of-a-common-effect-needs-examples-of-its-own-and-field-weighting-cannot-see-it)).

### Where the boundary sits wanders between epochs at the full learning rate, and settles near the rule at a tenth of it

At the full learning rate the boundary sharpens over twenty more epochs, but its position moves from one epoch to the next by more than that sharpening. The chart reads the same 40 qualifying training records at every epoch. It plots the predicted death given that the hit creature is affected, the death head alone without the gate, for a hit lethal by one point, exactly lethal, and short by one point. A sharp boundary at the right place puts "lethal by one" near 1 and "short by one" near 0, on either side of an even chance. Because the 40 records are the same every epoch, the movement is the model's, not a sample's.

![Predicted death given that the hit creature is affected, for hits lethal by one, exactly lethal and short by one, epoch by epoch, one row per width of e](2026-09-18-effect-model-gen2/charts/boundary-by-epoch.png)

*One row per width of `e`. The left panel holds the arm, dotted, and the A run, dashed, both at the full learning rate; the right panel holds the B run, solid, at a tenth of it. Dark purple downward triangles are hits lethal by one point, mid-purple circles exactly lethal hits, and light purple upward triangles hits short by one; the grey region is above an even chance, and a tinted span marks the arm's epochs. Dashed verticals mark each run's lowest card-disjoint epoch. Arm `e256-lethality`'s epochs were not read. In the B runs, after run B128's first few epochs, a hit short by one point stays at or below an even chance and an exactly lethal hit above it. In the A runs both cross it. Data: `boundary_by_epoch.csv`.*

In run A128 a hit short by one point is predicted to kill anywhere from one time in six to three times in four, depending on the epoch. Run A256's range is as wide, and run A64's narrower but of the same kind. The epoch a run keeps can land anywhere in that range: run A128's kept epoch has the boundary moved furthest toward a death of any checkpoint, while run A64's happened to land near the rule. Arm `e256-lethality`'s kept epoch also leans toward a death: its damage sweep crosses one half before 3 damage, where arm `e128-lethality`'s crossed after 5 and arm `e64-lethality`'s near the rule.

At a tenth of the learning rate the position settles. Run B128 started from run A128's misplaced epoch 30, pulled the boundary back within a few epochs, and then held it. The gap between "lethal by one" and "short by one" holds near six tenths in runs B64 and B256 and near a half in run B128, and in all three B runs the error against the rule falls slowly. On the probe set's grids, run B256's last epoch has the lowest error against the rule of any checkpoint. Run B128's last epoch, run A256's checkpoint and `B256 best` follow, level with each other. The training-records chart and the probe-set grids agree.

Dropping to a tenth of the rate is therefore what places the boundary. More epochs at the full rate do not. On the damage sweep, all six checkpoints of the B runs, kept and last, keep every point on the rule's side of an even chance. Of the full-rate checkpoints trained with lethality copies, only arm `e64-lethality`, by a hundredth at 4 damage, and run A256's single checkpoint do. Run A256's boundary moved between epochs as much as run A128's, and its last epoch is the one where it happened to sit near the rule.

### The probes keep improving after the card-disjoint loss has bottomed out

In every continuation run whose last epoch was not also its lowest, the last epoch has a higher card-disjoint loss than the kept epoch, yet the probe mean rates it as good or better. The chart plots each checkpoint's [probe mean](#three-gates-and-a-probe-mean-score-every-checkpoint) against each validation loss, with an arrow from each run's lowest card-disjoint epoch to its last. Every arrow points right on the card-disjoint panel, toward a worse loss, and level or up, toward better probes.

![Two scatter panels of probe mean against card-disjoint and game-disjoint loss for every gen-2 checkpoint, with an arrow per continuation run from its lowest card-disjoint epoch to its last](2026-09-18-effect-model-gen2/charts/probes-vs-loss.png)

*Colour gives the width of `e`: blue 64, orange 128, green 256. Triangles are arms, squares A runs and circles B runs; a filled marker is the lowest card-disjoint epoch and a hollow one the last. Grey triangles trained without lethality copies, the lightest being `e64 (earlier inputs)`. Run A256 reached its lowest card-disjoint loss at its last epoch, so it has a single point and no arrow. Run B256's arrow starts at its own lowest epoch, 55, drawn half filled, not at the checkpoint it keeps. Data: `checkpoints.csv`, `probe_ladder.csv` (rung 3, the model's own answer, game-disjoint records).*

The card-disjoint loss rises because the model grows more confident, not because it ranks outcomes worse. More than half of every run's rise is in the gate field, whether a card was affected at all, and poison adds the next-largest part in four runs of five. On the same card-disjoint records, the probe asking whether a card is affected scores as high or higher at the last epoch in four runs of five. Log-loss, the training loss, charges heavily for a confident miss, and a text the model never saw produces a few.

![Stacked horizontal bars per continuation run of the change in each output field's card-disjoint loss from the run's lowest card-disjoint epoch to its last, with the net change marked](2026-09-18-effect-model-gen2/charts/loss-rise-by-field.png)

*Rises extend right and falls left; the black tick is the net change. The gate, poison, power and toughness change and zone outcome have their own segments, and the other 42 output fields share one. Gen-2's 47 output-field terms do not sum to its card-disjoint loss: the rest, 0.40 to 0.57 on the continuation runs, is its own segment. Gen-1's field terms sum to its loss within half a hundredth. Run A256's lowest epoch is its last, so it has no rise, and run B256's bar runs from its own lowest epoch, 55. Data: `epochs.csv`, `fields_by_epoch.csv`.*

Across all seventeen checkpoints, both losses track the probes: lower-loss checkpoints have better probes, on either loss. Among the fourteen checkpoints trained with lethality copies the correlation is at the edge of what fourteen points can tell from none, which is a rank correlation of about 0.55.

| rank correlation with the probe mean, sign set so that positive means agreement | card-disjoint loss | game-disjoint loss |
|---|---:|---:|
| all 17 checkpoints | +0.71 | +0.73 |
| the 14 checkpoints with lethality copies | +0.55 | +0.53 |

*Run A256's single checkpoint counts once. Data: `checkpoints.csv`, `probe_ladder.csv`.*

Gen-3 therefore judges its checkpoints by a score built from the probes and from rule scenarios, edits of real records that each test one rule as the damage grid does, rather than by the summed loss ([gen-3 draft](2026-XX-XX-effect-model-gen3-design.md#the-summed-validation-loss-disagrees-with-the-probes-over-training-and-gen-3-judges-checkpoints-by-probes-and-rule-scenarios)).

#### At a tenth of the learning rate, two runs' losses rise with their confidence and the third repairs a misplaced boundary

Two of the three runs at a tenth of the learning rate show the confidence rise within a single run. The card-disjoint losses of runs B64 and B256 creep up across the run, while run B128's falls for a few epochs and then holds level. All three training losses keep falling. The rises are several times their epoch-to-epoch swing.

- **The rise in runs B64 and B256 is confidence on unseen texts.** In run B64 the gate field carries about two thirds of the rise. In run B256 the gate and poison together rise by more than the net, while the other fields fall. The zone outcome and the damage stay level or improve in both. Both runs' last epochs score as well as their lowest-loss epochs on the probes, and their error against the death rule keeps falling.
- **Run B128 had a misplaced boundary to repair.** It started from the checkpoint whose boundary leaned furthest toward a death, and its first epochs pull that boundary back, which is where its early gain comes from. Its gate loss falls slightly instead of rising.

| at a tenth of the learning rate | run B64 | run B128 | run B256 |
|---|---:|---:|---:|
| card-disjoint trend, per ten epochs | +0.04 | 0.00 | +0.05 |
| game-disjoint trend, per ten epochs | +0.03 | −0.02 | +0.01 |
| training-loss trend, per ten epochs | −0.08 | −0.12 | −0.08 |
| affected gate's card-disjoint loss, first epoch → last | 1.06 → 1.15 | 1.16 → 1.13 | 1.04 → 1.11 |

*Each trend is a straight-line fit over the run's epochs after its first three. Data: `epochs.csv`, `fields_by_epoch.csv`.*

### In combat the model applies power against toughness softly and learns indestructible and most of deathtouch, and the grid that finds no first strike may understate it

Every checkpoint gets the direction of the plain combat rule right, with a boundary as soft as the damage grid's. Indestructible is learned outright and deathtouch mostly. First strike barely moves the prediction, on a grid whose edited boards contradict their own damage input.

The combat grid asks the same question of a block that the damage grid asks of a spell. It takes 40 real one-on-one blocks from the validation strata, edits the attacker's power and the blocker's toughness, or the blocker's power and the attacker's toughness, from 1 to 8, and reads the predicted death. A keyword variant grants one creature a keyword and compares the prediction with the rule. The script is `scripts/effect_knowledge_probes/combat_grid.py`.

The grid's edited boards contradict their own damage-assignment input, so its first-strike and power-axis readings may understate what the model knows. Each card slot of a combat record carries the damage the creature assigns, beside its power. The grid edited power and the damage step and kept the base record's assignment. A creature without first strike therefore still showed its damage assigned in the first-strike step, and an attacker edited to power 8 still showed the base record's assignment. The checkpoints below were not re-read with the grid's assignment set from the edited board ([the grid's assignment](#the-combat-grids-edits-have-to-move-the-damage-assignment-with-them-2026-10-21)).

A combat record covers one damage step, and the model predicts that step alone. The grid reads a first striker in the first-strike step, where it deals its damage and the creature it fights deals none, and a double striker's second hit in the regular step, with the first hit already marked on the blocker.

![Diagram of one combat's two damage steps as Forge runs them, the span each gen-2 combat record captured, the parts that reached no record, five numbered collector defects and where gen-3 moves the record's boundaries](2026-09-18-effect-model-gen2/charts/diagram-combat-step.png)

*One combat as Forge runs it, and what a gen-2 combat record captured. Hatched parts reached no record. The crimson numbers are collector defects. Defect 1 is in [the fixes](#the-first-arm-exposed-three-faults-in-the-models-inputs-and-the-arm-trained-again-on-corrected-ones-2026-10-14), defects 2 to 4 in [the collector defects](#nine-collector-defects-shaped-gen-2s-labels-and-gen-3-collects-without-them), and defect 5, the snapshot timing, below. The violet marks are where gen-3 moves the record's boundaries; the snapshot at the step's start is a proposal of the gen-3 draft, and the collector already makes the other changes.*

![Combat grids of run B64's last epoch: plain blocker and attacker deaths, and the change in predicted death with deathtouch and with first strike](2026-09-18-effect-model-gen2/charts/combat-grids.png)

*Run B64's last epoch. Top: the predicted death of the blocker and of the attacker with no keyword, coloured as the damage grids, orange above an even chance and blue below; the rule kills on and below the staircase. Bottom: the change in predicted death when the attacker has deathtouch, where the rule adds a death above the staircase, and when the attacker has first strike in the first-strike step, where the rule removes the attacker's death below it. The bottom row reads the stale damage assignment described above. Data: `combat_grids.csv`.*

The plain rule's boundary is blurred over two to three points of toughness. The attacker's death sits further from the rule than the blocker's on every checkpoint. The likely reason is in the data: a combat record's snapshot is taken at the step's first damage event, so the creature damaged first already shows that damage. In 26 of the 40 blocks the blocker is marked with exactly the attacker's power, and the attacker is never marked. When the lethal damage is already on the board, every checkpoint predicts the death nine times in ten or more; when the model must add the opposing power itself, it is far less sure. The grid clears both creatures' marked damage before editing. The gen-3 draft proposes taking the snapshot at the start of the damage step ([gen-3 draft](2026-XX-XX-effect-model-gen3-design.md#a-combat-records-snapshot-should-be-taken-before-the-damage-step-deals-any-damage)).

![Strip plot of the share of the rule's change each combat keyword produces in the predicted death, one row per keyword and one marker per checkpoint](2026-09-18-effect-model-gen2/charts/combat-keywords.png)

*Each marker is one checkpoint's (with − without) / (rule − without): the change the keyword makes in the mean predicted death, over the cells where the keyword changes the rule's answer, as a share of the change the rule makes. 0 means the keyword is ignored and 1 means the rule's full change. Colour gives the width of `e` and the shape the stage, as in the other charts; grey triangles trained without copies. Data: `combat_grids.csv`.*

The keywords sort into three groups:

- **Learned outright: indestructible.** No checkpoint predicts an indestructible blocker's death more than one time in fourteen, on average over the cells where the damage is lethal.
- **Learned mostly: deathtouch and damage already marked.** Deathtouch moves the non-lethal cells between half and nearly all of the way to a death, depending on the checkpoint, and fades at the blocker's highest toughness on every checkpoint. A double striker's second hit is read from the damage the first one marked. Run A64's last epoch reads it best, ahead of run B64's last epoch and run B256's last, close behind each other.
- **Not learned, as far as the grid can tell: first strike.** In the first-strike step a first striker cannot die, yet every checkpoint on the corrected inputs predicts its death within a tenth of its death without the keyword, for an attacker and a blocker alike. Only arm `e64 (earlier inputs)` moves it about half of the way.

The corpus explains the last group. A collector bug lost the creature damage and deaths of every first-strike damage step, and the repair dropped the blocked first-strike fights it could not recover. The probe forks are the only blocked first-strike fights the model sees, and they do not teach the step. The collector now records the first-strike step whole, and the gen-3 draft adds combat copies labelled by the rule as an option.

### Three combat keywords hold on every checkpoint, and double strike swings far beyond its sampling noise

Lifelink, indestructible and trample pass gate 2 on every checkpoint with room to spare. First strike, deathtouch, infect and double strike vary between checkpoints far more than sampling allows. Double strike fails the gate on seven of the sixteen checkpoints scored on it, and infect fails it on arm `e64` alone.

Gate 2 asks, on the game-disjoint stratum, whether the model's predictions move the way a damage-step keyword moves the real outcome. A probe fork copies a real combat and strips the keyword from one creature. The gate counts how often the model's predicted change and the real change point the same way, and passes a keyword at 70% over at least 200 qualifying records.

![Strip plot of gate 2's direction agreement per keyword, one marker per checkpoint, with a grey band of the sampling noise around each row's median](2026-09-18-effect-model-gen2/charts/gate2-keywords.png)

*Each row gives its keyword's median number of qualifying records. The grey band is the row's median plus or minus the 95% half-width that sampling alone would give at that median, so a spread wider than the band is the model's. The black line is the gate's 70%. Arm `e64 (earlier inputs)` scored first and double strike on the wrong damage step and is left out of those rows. Data: `gate2.csv`.*

Lifelink and indestructible agree 97% of the time or more on every checkpoint, and trample 87% or more. Lifelink and indestructible vary by about as much as sampling allows, or a little more. Trample ranges over ten points, eight times its sampling half-width, so its variation is the model's, but no checkpoint comes near the gate's bar.

The other four also move far more than sampling allows:

- First strike ranges over about twenty points on 2,503 records, whose sampling half-width is under one and a half. Run A256's checkpoint scores lowest, just above the gate's 70%.
- Deathtouch and infect each range over twelve to fourteen points.
- Double strike ranges from 44% to 85% on 294 records, eight times its sampling half-width. Within run B64 it fails at the kept epoch and passes at the last.

Nearly every qualifying first-strike and double-strike case is an unblocked attacker, read through the defending player's life, because the corpus lost the blocked first-strike fights. Neither the lethality copies nor the width moves any row consistently. A keyword verdict from one checkpoint is therefore a draw from that range, and gate 2's pass or fail on double strike says little about the run. Gate 2 also read the effect fields without the gate and counted some records twice ([the gate-2 defect](#gate-2-read-the-effect-fields-without-the-affected-gate-over-a-population-up-to-a-third-larger-than-its-distinct-records)).

### Validation losses swing between epochs by about as much as a run improves

Both validation losses fall quickly for about ten epochs and slowly after that. From then on, single epochs land a tenth of a point or more above or below the trend, about as far as the trend falls in five to ten epochs. The validation samples are fixed, so the swing is the model changing between epochs and not a different sample. Each run keeps its lowest card-disjoint epoch, so the kept epoch is partly the luckiest one.

![Card-disjoint, game-disjoint and training loss by epoch for every run, one column per width of e](2026-09-18-effect-model-gen2/charts/training-losses.png)

*One column per width of `e`. Line style gives the stage: dotted the arm with copies, dashed the A run at the full rate, solid the B run at a tenth of it; grey is the arm without copies. The large marker with a black edge is each run's lowest card-disjoint epoch, shaped by stage: triangle for the arm, square for the A run, circle for the B run. The bottom row is the training loss, averaged over each epoch's 5,000 steps; it runs about a tenth higher with lethality copies, which add records the corpus rarely shows. Data: `epochs.csv`.*

The card-disjoint loss swings more than the game-disjoint loss in almost every run. The card-disjoint stratum holds 500 texts the model never saw, and how well the model generalizes to each of them changes from epoch to epoch. The training loss moves by two to three hundredths in every run.

The training loss also shows why the validation losses swing. After a short warmup, in which the learning rate climbs from zero, it stays at its peak for the whole run, so the weights never settle: each epoch ends at a different point near the bottom of the loss surface. The training loss is averaged over the 5,000 steps of an epoch and smooths that movement out, while a validation loss scores the single set of weights the epoch ended on. When runs B64 and B128 drop to a tenth of the rate, their training loss falls by about a tenth in the first epoch and by about two tenths within two, against run A64 and run A128 on the same shards. The weights move closer to the bottom once their steps are smaller.

At a tenth of the learning rate the swing shrinks between two and six times on card-disjoint loss and about three times on game-disjoint loss, while the training loss keeps the same small movement. The three B runs are the only runs whose single epochs can be compared with each other at face value.

![Epoch-to-epoch swing of each loss, from the arm through the A run to the B run, one line per width of e](2026-09-18-effect-model-gen2/charts/loss-noise.png)

*The standard deviation of each loss around its own five-epoch moving mean, from epoch 10 on, on a log scale. Each line runs from a width's arm through its A run to its B run, and the ÷ figure is how many times smaller the B run's swing is than the A run's. Grey triangles trained without lethality copies. Data: `epochs.csv`.*

The early stopper, which ends a run after five epochs without a new lowest card-disjoint loss, ended every arm while its trend was still falling, and every continuation run lowered its arm's loss further. A difference of a tenth of a point between two runs' kept epochs at the full rate is inside the swing and says little.

#### The large output fields plateau by epoch 20, and the small ones swing more than they improve

The fields with many records, such as whether a card was affected, its zone outcome and the keywords it gained, rise smoothly to a plateau where they remove between three quarters and seven eighths of a constant prediction's loss. The small fields swing by tens of points from one epoch to the next. Explained deviance is the share of a constant prediction's loss the model removes: 0% is no better than always predicting the field's average.

![Card-disjoint explained deviance by epoch for six output fields, along the three lethality-copy lines](2026-09-18-effect-model-gen2/charts/fields-explained.png)

*Card-disjoint explained deviance per epoch along the lethality-copy lines. Colour gives the width of `e`, and the line style the stage: dotted the arm, dashed the A run, solid the B run. Below 0 is worse than a constant prediction. Data: `fields_by_epoch.csv`.*

Poison is the clearest case. After the first twenty epochs it moves between a fifth and seven tenths of its loss on the 64-wide line from epoch to epoch, and the 256-wide line stays in the same band. On the 128-wide line it rarely rises above two fifths and often falls below zero, worse than a constant prediction. Since the widest and the narrowest `e` agree, the 128-wide line's poison is more likely its seed's level than an effect of width. Types gained and blocking legality swing by about thirty points. A comparison of two checkpoints on any of these fields is mostly a comparison of two draws. Gen-3's field-normalized loss and its adaptive sampling address these fields directly.

### The 64-wide and 256-wide lines split the measures, and the 128-wide line trails on the probes at every stage

The 64-wide line has the lowest card-disjoint loss at every stage. The 256-wide line has the lowest game-disjoint loss, level with run A128's last epoch at the full-rate stage. On the probes, the 64-wide arm leads the arms, runs A64 and A256 tie, and run B256 leads the B runs. Run B256's last epoch has the best probe mean of any checkpoint. The 128-wide line has the highest card-disjoint loss and the weakest probes at every stage. Each point in the chart is a checkpoint trained with lethality copies, at three stages: the arm, the A run at the full learning rate and the B run at a tenth of it.

![Four slope panels of card-disjoint loss, game-disjoint loss, probe mean and participation ratio across the arm, A run and B run, one line per width of e](2026-09-18-effect-model-gen2/charts/width.png)

*Each panel's title says whether lower or higher is better. The solid line and filled markers join each width's kept epochs, the dashed line and hollow markers its last epochs; run B256's point is its own lowest epoch, 55, drawn half filled, not the checkpoint it keeps. The grey bar on each loss panel reaches one standard deviation of the epoch-to-epoch swing above the stage's lowest kept loss. The participation ratio is defined in [the next section](#gen-2s-e-spreads-over-two-to-three-times-gen-1s-directions-and-reads-a-lines-kind-and-amounts-far-better). Data: `checkpoints.csv`, `probe_ladder.csv`, `gate3.csv`.*

The gaps between the lines are at the edge of the epoch-to-epoch swing at the full rate. The card-disjoint gap between the 64-wide and 128-wide lines is a tenth to two tenths of a point, and the probe means of the three lines lie within a hundredth and a half of each other at each stage. The 64-wide line keeps the lowest card-disjoint loss and the 128-wide line the weakest probes through both continuation runs, so those two gaps are more than one lucky epoch. The probe lead between the 64-wide and 256-wide lines changes hands between stages.

The order of the lines points to the seeds rather than to the width. A width effect that grows or shrinks with the width would put the 128-wide line between the other two. On the card-disjoint loss and the probes it trails both. Each line is one seed, so [a second seed per width](#each-run-is-one-seed-so-a-difference-between-runs-counts-only-where-it-exceeds-the-swing-between-epochs) is what would separate the two explanations.

The scorecard sets every gen-2 checkpoint against thirteen measures at once, and no checkpoint leads on most of them. Seven different checkpoints top a row. The 256-wide continuation checkpoints have the best mean rank, and the arms rank near the bottom. The B runs cluster at the top of the damage-grid and damage-sweep rows, and the activated-ping row cuts across the widths. The colours give rank, not size, so on rows such as the sweeper's slope a dark cell can lead by a hundredth.

![Heatmap of thirteen measures by seventeen checkpoints grouped by width of e, each row coloured by rank, with the best in bold and a mean-rank row at the bottom](2026-09-18-effect-model-gen2/charts/scorecard.png)

*Each row is coloured by rank among the checkpoints, darker better; ▲ marks a measure where higher is better and ▼ one where lower is better, and the best cell is bold. The last row is each checkpoint's mean rank over the rows above. The rows from blocking onward come from the probes record. Data: `checkpoints.csv`, `probe_ladder.csv`, `damage_grids.csv`, `sweeps.csv`, `sweep_records.csv`, `gate2.csv`, `evasion_ranking.csv`, `colour_pips_ranking.csv`, `zone_amounts.csv`, `zone_who.csv`, `target_flips.csv`.*

A wider `e` spreads its variance over few more effective directions. Every checkpoint trained with lethality copies sits between 9 and 12 directions; arm `e64`, without copies, has 7.5. The 256-wide arm has about 9.4, and runs A256 and B256 between 10.5 and 11.9, the highest of them less than one direction above the best narrower checkpoint, so the vector's extra width goes almost unused. What a nonlinear probe reads from `e` against a random vector of the same width is in [the probes record](2026-09-19-effect-knowledge-probes-design.md#e-adds-most-over-a-random-vector-of-its-width-on-what-a-line-states-and-next-to-nothing-on-deaths-trigger-firing-and-block-legality). On the death rule, combat, gate 2 and the board-abilities subsets, the 256-wide arm lands inside or at the edge of the range of the other checkpoints.

### Gen-2's `e` spreads over two to three times gen-1's directions, and reads a line's kind and amounts far better

The participation ratio counts the equally weighted directions that would give `e`'s spread of variance: 3.9 for gen-1 and between 7.5 and 11.9 for every gen-2 checkpoint. A linear probe, a straight-line fit from `e` to a property of the line, reads a line's API type, its mana cost and each of its stated amounts far better from gen-2's `e`. The API type is the first word of the script, the kind of effect, such as `DealDamage`. The wider `e` reads the amounts better still, though a linear probe with more inputs fits more of any label, so the table bounds what width adds rather than measuring it. Gen-2 trains two heads that read `e` alone, one predicting the API type and one the stated amounts, so part of the gain is by construction; the probes record tests whether the trunk uses it.

| linear probe on `e`, all texts | gen-1 | `e64`, epoch 25 | run B64, epoch 62 | run B128, epoch 50 | `e256-lethality`, epoch 28 |
|---|---:|---:|---:|---:|---:|
| API type, accuracy | 0.64 | 0.89 | 0.91 | 0.91 | 0.91 |
| mana cost, R² | 0.06 | 0.53 | 0.58 | 0.63 | 0.61 |
| damage amount, R² | 0.34 | 0.48 | 0.50 | 0.60 | 0.70 |
| counters amount, R² | 0.18 | 0.54 | 0.54 | 0.65 | 0.64 |
| life amount, R² | 0.09 | 0.31 | 0.33 | 0.40 | 0.45 |

*Data: `embedding_probes.csv`, `gate3.csv`.*

Gate 3 passes on every gen-2 checkpoint but one. Arm `e64` put 30.03% of its variance on its top component, just past the gate's 30% limit. That component sorts lines by what they do, from damage and exile aimed at a creature to spell-copy and cast triggers, so it is a stretched direction rather than a collapse.

The probes record reads `e` with a small nonlinear probe and against the trunk. It finds that `e` carries a line's costs, timing and most of its amounts, that it beats a random vector of its width strongly on what a line states and barely on what the board decides, and that the trunk works out deaths and trigger firing ([the probes record's summary](2026-09-19-effect-knowledge-probes-design.md#what-the-probes-found)).

### The first arm trailed gen-1 on the loss the two generations share, in the fields whose targets gen-2 made larger or new

The 47 output-field terms are the comparable part of the two generations' losses. Arm `e64 (earlier inputs)` matched gen-1 on them at epoch 9 and trailed it by a tenth to a fifth of a point from epoch 12 to 18. Arm `e64`, on the corrected inputs, trailed by a tenth to a quarter of a point from epoch 9 to 25. Gen-2's corpus puts more of its constant-prediction loss in fields the ability text decides, and less in the gate. `pt_duration`, how long a power and toughness change lasts, is a target gen-1 never had in practice, because gen-1 recorded every pump as instant. On keywords gained, the template holdout leaves a held-out keyword grant no near-twin in training.

| field, epoch 18, card-disjoint | floor, gen-1 | floor, gen-2 | loss, gen-1 | loss, arm `e64 (earlier inputs)` |
|---|---:|---:|---:|---:|
| gate | 5.05 | 4.63 | 1.164 | 1.169 |
| keywords gained | 0.43 | 0.57 | 0.065 | 0.164 |
| +1/+1 counters | 0.12 | 0.43 | 0.026 | 0.079 |
| `pt_duration` | 0.00 | 0.23 | 0.000 | 0.043 |
| mana deltas, six fields | 0.10 | 0.41 | 0.024 | 0.049 |

*The floor is a field's constant-predictor loss: the loss of always predicting the field's average. Data: `gen1_vs_e64_field_terms.csv`, `fields_by_epoch.csv`.*

### Collector defects taught the AI's block plan as a rule, and bent the affected gate and the combat input

Nine places where the corpus records something other than what happened in the game reach fields the model trains on. They sit in the Forge hooks and the connector's collectors and were found after the gen-2 runs. The gen-2 corpus is not collected again. Two of the playability verdict bits also carry the AI's answer rather than the rules'. The table names the field each one bends and the result it qualifies; the evidence for each is in [Details](#nine-collector-defects-shaped-gen-2s-labels-and-gen-3-collects-without-them).

| defect | field it bends | result it qualifies |
|---|---|---|
| most "may not block" labels name a creature in the AI's unfinished block plan | `blocker_legal` | the probe mean's may-block question; the probes record's may-block rung and the evasion grid's level |
| a reveal from a library reaches no card and no player | the `reveal` bit of `library_events` | none in either record |
| created tokens carry no keywords, and several kinds of token are recorded as the first | the created-objects field | none in either record |
| prevented damage names nobody it was prevented for | the affected gate | none in either record |
| an ability's own card is marked affected by the tokens, coins and copies it produces | the affected gate | the card-affected question of the probe mean and of the probes record's ladder |
| a double striker's second hit is recorded with no damage assigned | the combat input's damage assignment | the combat grid's double-strike second hit |
| events of abilities that resolve after combat damage are written into the combat record | combat records' draws, tokens, discards and mills; those triggers' own resolutions | none; the combat grid reads deaths only |
| "tapped for mana" trigger records are partly the AI's questions about planned taps | the trigger's fired bit, on a few mana cards | none measured |
| a prevention shield with nothing left to prevent records a prevention of 0 | the `damage_prevented` amount | none in either record |
| `affordable` and `has_legal_target` come partly from the AI's cost check and targeting | two playability verdict bits | the probe mean's affordable question; the probes record's affordability and colour probes |

The block plan is the largest. More than half of the "may not block" labels in real block decisions name a creature the AI had already planned to block with, and two thirds name a creature no static restricts. Only one in fourteen names a tapped creature, a refusal the rules make.

![Bar chart of the shares of gen-2's "may not block" labels, real block-legality records and anchoring attackers that reflect the AI's block plan rather than a rule](2026-09-18-effect-model-gen2/charts/planned-block-labels.png)

*Real decisions in the game-disjoint stratum, as the collector counted them. The sub-counts within a group may overlap, so they are not stacked. Data: `legality_planned_blocks.csv`.*

The corpus build, the training targets, the evaluator and the probe scripts carry fifteen further defects, which change what every arm learned on a few fields and bias some reported figures ([Details](#fifteen-defects-in-the-corpus-build-the-training-targets-the-evaluator-and-the-probes-shaped-gen-2s-results-and-gen-3-is-built-without-them)). Two of them reach the probe mean: its may-block and pump-duration questions read fields whose unset targets were trained as 0.

## What gen-3 takes from these results

The gen-3 draft, [`2026-XX-XX-effect-model-gen3-design.md`](2026-XX-XX-effect-model-gen3-design.md), carries each of these forward:

- **Rebalancing by rule**, so that no keyword or rule is rare in training, with rule-labelled copies where real records are too few. The lethality copies are the first case, and combat copies are proposed.
- **Checkpoints judged by probes and rule scenarios**, from the disagreement between the loss and the probes, with each field's loss divided by its constant prediction's.
- **The first-strike step collected whole**, which the collector already does, and two draft proposals from the combat results: **the combat snapshot taken before damage and the block pairing in the input**.
- **Every line of a permanent on the board**, from the board-abilities results in the probes record.
- **Block legality recorded before anyone declares, outcome notes that name only what they happened to, each damage step's own assignment, and combat records that stop where the stack starts**, from [the collector defects](#nine-collector-defects-shaped-gen-2s-labels-and-gen-3-collects-without-them) found after the runs.
- **The withheld keyword read through its definition**, so the zero-shot check measures what it was meant to ([gen-3 draft](2026-XX-XX-effect-model-gen3-design.md#the-withheld-keyword-reaches-training-as-its-definition-so-gen-3s-zero-shot-slice-is-the-first-that-tests-reading-one)).

## Design: what gen-2 changes from gen-1, and why

Gen-1 is the model trained on the corpus of 2026-09-17 (run `09.17f`, filed under `models/effects/runs/2026-09-17-full-textless-corpus/`), together with four baselines. Its design rationale is [`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md), called the design doc below. Each baseline trains on the identical pipeline and differs from gen-1 in one input, so each answers one question. `identity` gives every distinct ability text a free vector in place of the encoder's output, and asks whether the encoder reads text at all. `taxonomy` replaces that vector with a hash of the script's API type and parameter keys, and measures what reading a script's parameters adds over knowing its effect category. `state-only` zeroes every ability vector and gives each record kind its floor. `no-state` also removes the board, so the model can learn only each text's average effect.

The encoder has two surfaces. The script surface is the Forge script text, and the prose surface, the rendered rules text, is the fallback for a line with no script. A head is a network trained on top of a shared representation to predict one kind of output.

Each of the first ten sections below records one gap found while gen-1 trained, the evidence for it, and the change gen-2 makes. The three after them set how gen-2 is collected, evaluated and trained, and the run plan orders the work. Gen-2's gate figures are not compared with gen-1's figure by figure, because the corpus, the split and the encoding all change.

Three changes came during the runs, and the design sections do not describe them. Lethality copies, training copies whose death is labelled by the damage rule, joined after the first arms ([the death rule](#the-model-compares-damage-with-toughness-only-after-training-shows-it-the-rule-directly)). Thermometer codes for the board's amounts and "at least k" targets for the value head replaced plain numbers after the first arm ([the input fixes](#the-first-arm-exposed-three-faults-in-the-models-inputs-and-the-arm-trained-again-on-corrected-ones-2026-10-14)). And every arm deleted the withheld keyword where the design expands it ([the withheld keyword](#every-arm-deleted-the-withheld-keyword-instead-of-expanding-it-so-the-zero-shot-check-measured-nothing)).

### The encoder reads only the first script line of an ability, so a trigger's effect never reaches it

On the script surface the encoder is fed one Forge script line per ability: the line the ability's
runtime trait was parsed from. An ability that Forge writes across several lines loses every line
but the first. For a triggered ability the first line is the trigger condition, and the effect is a
separate `SVar` line the trigger names through `Execute$`. Ajani's Pridemate encodes as the trigger
line alone:

```
Execute$ TrigPutCounter | Mode$ LifeGained | TriggerDescription$ Whenever you gain life, put a +1/+1 counter on CARDNAME. | TriggerZones$ Battlefield | ValidPlayer$ You
```

The `DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1` line that does the work is not in the text.
The model knows that this is a life-gain trigger which does something the description calls "put a
+1/+1 counter"; it does not see the counter type or the count as parameters.

The omission is near-universal for triggers and common for the rest.

| converted card corpus, excluding keyword lines | count |
|---|---:|
| lines with a script surface | 41,914 |
| triggered lines | 16,135 |
| triggered lines whose `Execute$` effect is absent from the encoded text | 16,098 |
| lines with a sub-ability chain below the first effect | 8,820 (21%) |
| chain length 1 / 2 / 3 / 4 or more sub-abilities | 5,595 / 2,247 / 698 / 280 |
| longest chain | 13 sub-abilities |
| median ratio of full-chain text to root text, chained lines only | 1.4× |
| 90th percentile of that ratio | 2.1× |

Modal abilities lose every mode. A charm's root line names its modes only by label, as in `Choices$
DBDmgC,DBDmgP`, and where it has a description, the description ends in the placeholder `ABILITY`
that Forge fills from the modes at runtime. The encoder sees that the ability offers a choice and
not what any choice does.

| charm lines | count |
|---|---:|
| charm lines in the converted card corpus | 783, on 781 cards |
| charm lines whose encoded text names no mode | 742, on 740 cards |

The mechanism is in the converter. `TraitScript.of` renders `script_text` from the trait's own
parameter map, and a trait's map holds only the line the trait was parsed from. For a trigger that
is the `T:` line. The effect the trigger executes is a separate `SpellAbility` reachable through
`getOverridingAbility()`, and for any ability the sub-abilities are reachable through
`getSubAbility()`. The converter already walks both chains: the sub-abilities become the
`sub_ability_links` index paths the record attribution uses, and the prose renderer walks the same
chain to find the description. Only the script text stops at the root.

The design doc's argument for the script surface assumed the whole mechanism was present. It gives
`Cost$`, `ValidTgts$`, `Mode$` and `NumDmg$` as the predicate structure every record supervises, and
it counts on paraphrases of one mechanism collapsing to one script. Neither holds for the effect
half of a trigger, because that half is never encoded. Gen-1 still sees the effect through the
description parameter: Forge attaches the rules text to the root line, so the prose is present for
all but about one line in eighty. The encoder is reading prose plus the trigger condition or the
first effect's parameters, and the rest of the mechanism reaches it only as English.

The description is not a full substitute. The design doc's third argument for the script surface is
synthetic script variants, which perturb a parameter and play the variant for engine ground truth of
a text that never existed. A variant that changes `CounterNum$ 1` to `CounterNum$ 2` on a trigger
produces a record whose encoded text is identical to the original, because the changed line is not
in the text. The variant records then teach the model that one text has two outcomes.

#### Gen-2 encodes the whole chain in resolution order

`script_text` becomes the concatenation of every script line the ability owns, in this order:

- the root line;
- for a trigger, the ability it executes through `Execute$`;
- each sub-ability in chain order, including a `RepeatSubAbility` where one exists.

Replacement effects get the same treatment through `ReplaceWith$`. A dedicated token separates the
segments. Each segment after the root opens with the label its parent referenced it by, so
`SubAbility$ SV2` in one segment and `SV2:` opening a later one give the model the link. The labels
are renumbered by position within the line, so the link survives and the author's name for the SVar
does not ([SVar labels](#gen-2-renumbers-svar-labels-by-position-because-the-names-authors-give-them-carry-no-mechanism)). A charm's modes are not inlined into its root line. Each mode is
encoded on a line of its own ([charms](#gen-2-encodes-and-records-each-mode-of-a-charm-as-its-own-ability)).

Four things follow from the change, and the third is the one that decides the order of work.

1. **The converter rewrites the sidecars but not the converted text.** `script_text` is a field of
   the sidecar, the `.provenance.json` file `convert` writes beside each converted card to map each
   rendered line to the Forge script it came from. The rendered prose does not change, so the
   byte-identity guarantee the sidecar contract makes holds.
2. **The script vocabulary is rebuilt.** `build-vocab --surface script` scans the sidecars' script
   lines; after the change those lines carry the chained segments and the separator token. Most
   parameter keys already appear on root lines, so the vocabulary grows by the numbered labels,
   one per position in the longest chain, and the rarer sub-ability parameters.
3. **The holdout is chosen on the chained text.** A text is held out by the hash of its masked
   template, the normalised script text with its numbers, `CARDNAME` and descriptions masked
   ([gate 1](#gate-1-scores-fewer-than-half-of-the-held-out-texts-and-a-third-of-the-recorded-ones-have-a-numeric-twin-in-training)), and the corpus's rarity table is keyed on the script text. Changing
   `script_text` changes every key. The holdout is therefore computed on the chained text before the
   new corpus is collected, which is the only point at which changing it costs no training games
   ([collecting from scratch](#gen-2-collects-its-corpus-from-scratch-because-the-holdout-and-the-random-seat-both-act-at-collection-time)).
4. **Sequence length grows, and most chains stay well under the cap.** The encoder truncates at 512
   tokens, and the table above gives the growth on chained lines. The distribution of chain lengths
   in tokens is measured after the rebuild. A chain that exceeds the cap is logged rather than
   silently cut from the tail, because the tail of a chain is its last effect.

Prose stays as it is: the fallback for a line with no script, which is a keyword-derived line or a
synthetic land mana line, and nothing else.

Gen-2 does not build the design doc's paired-prose loss. That loss would pull a line's prose
encoding toward its script encoding, so that a card with prose but no Forge script would still get a
usable `e`. Gen-2 serves cards implemented in Forge, and every one of them has a script.

Under gen-2's encoding, Ajani's Pridemate's counter reaches the encoder as parameters, a chain of
sub-abilities reaches it segment by segment, and each mode of a charm gets its own `e`.

![Diagram of what the encoder reads for Ajani's Pridemate in gen-1 and gen-2, with insets for Inspiring Overseer's three-segment chain and Barbed Lightning's charm](2026-09-18-effect-model-gen2/charts/diagram-encoder-reads.png)

*Top: Ajani's Pridemate in Forge's script, a trigger line and the SVar its `Execute$` names. Gen-1
encodes the root line only, so the counter type and count reach the model only as the English of
the description. Gen-2 encodes the root, a `[SEG]` separator and the SVar line, with the label
`TrigPutCounter` renamed `SV1`. Bottom left: Inspiring Overseer's trigger, its `Execute$` effect and
that effect's `SubAbility$`, three segments. Bottom right: Barbed Lightning's charm root keeps only
its choice, and each mode's option line, opened by `SV1` because numbering restarts on every line,
gets its own `e`.*

#### Gen-2 renumbers SVar labels by position, because the names authors give them carry no mechanism

Forge's card scripts name their SVars freely, and most names are used once. The table counts the
labels that `Execute$`, `SubAbility$` and `ReplaceWith$` reference across Forge's card scripts.

| SVar labels referenced in Forge's card scripts | count |
|---|---:|
| distinct labels | 3,488 |
| references | 36,501 |
| labels referenced once | 2,355 |
| labels referenced fewer than five times | 3,009 |

Labels kept as written damage the vocabulary. A label is one whole token, so a name used once
either takes a vocabulary slot of its own or reads as `[UNK]`. A label read as `[UNK]` no longer
links a reference to the segment it names, which is the only reason the label is in the text.

Labels kept as written also damage the holdout. Two texts that do the same thing under different
names, `SubAbility$ DBDraw` and `SubAbility$ DBDrawCard`, are different texts. The rarity table
counts them apart, and the holdout hashes them apart, so one twin can train while the other sits in
the card-disjoint stratum as an unseen text.

Gen-2 renames every chain label in a line's `script_text` to `SV1`, `SV2`, … in order of first
appearance. A label keeps its number everywhere in the line, so a reference and the segment it names
still match. Numbering restarts on every line. The rename covers the values of `Execute$`,
`SubAbility$`, `RepeatSubAbility$` and `ReplaceWith$`, each item of `Choices$`, and the label that
opens each segment. Amount SVars, such as the `X` of `NumDmg$ X`, are not chain links and keep their
names. What a label loses is redundant: `DBDraw` says "draw", and the segment it opens says
`DB$ Draw`.

#### Gen-2 encodes and records each mode of a charm as its own ability

A charm's outcome depends on which modes its caster chose, and no single text can show that choice.
The converter already emits one `option` line per mode beside the charm's root line. Gen-1's
sidecars give those lines no provenance key and no script text, and the provenance join resolves
every chosen mode to the root line. A charm resolution therefore acts through one text whose
outcome is damage on one record and a drawn card on another, depending on a choice the model never
sees.

Gen-2 makes each mode a separate ability, in four steps.

1. **Each `option` line carries its mode's chain.** Its sidecar entry holds the mode's provenance
   key and a `script_text` holding the mode's script line, opened by its label, followed by its
   sub-abilities. The root line keeps only its own script line: the choices, how many modes to
   choose, and whether a mode may repeat.
2. **Each mode gets its own `e`.** The effect model adds a learned option-kind embedding to every
   ability row that comes from an `option` line. Option rows follow their root row within the
   card's block, so the flag and the order link a mode to its charm.
3. **A patched worker records each chosen mode as its own effect.** A modal resolution becomes one
   cost record acting through the root line and one effect half per chosen mode, acting through
   that mode's `option` line. Each effect half carries only its mode's events and a snapshot taken
   just before that mode resolved. All of them share one `link_id`, which therefore joins a cost
   record with its effect halves rather than only a pair. A mode chosen twice, as a Pawprint charm
   allows, produces two effect halves.
4. **A degraded worker keeps the root line.** Without the engine hooks the collector cannot tell
   where one mode's events end and the next begin. A degraded worker writes one effect half
   through the root line carrying every chosen mode's events.

### Every outcome in the corpus is one Forge chose, so the model can learn abilities without learning targets

Every record comes from games in which both seats follow Forge's AI. The corpus is therefore
observational under one decision rule, one policy in reinforcement-learning terms. A resolution
record exists because Forge decided to take that action, and Forge takes an action only when it
already predicts the outcome it wants. In the records, "Lightning Bolt targets a creature" and "that
creature dies" are therefore almost the same event, and "Murder targets a creature" is never paired
with an indestructible target. A model reaches a good loss on that corpus by learning that Bolt
means death. It never has to learn that the target's toughness decides it, because no record shows
Bolt on a creature that survived. Combat records do show damage that failed to kill, and the effect
head carries some of what it learns there across record kinds. The part of its mapping that depends
on the ability is still learned only on the boards and targets Forge chose.

A corpus like this understates any measure of whether the encoder reads text. A lookup table learns
"Bolt means death" as well as the encoder does, so the encoder's margin over gen-1's identity
baseline is smaller than it would be on a corpus where the outcome depended on the target.

Off-policy play, in which one seat takes actions Forge's AI would not have chosen, is the cheap
general fix. Forge's rules engine computes every record's outcome, so an action Forge's AI would
never take still produces a correct record when it resolves. The fork probes already in the corpus,
games copied at the damage step and replayed with one keyword stripped from one creature, fix only
the damage-step keywords, and each further case would need its own fork machinery. The records
off-policy play adds are the ones the corpus lacks:

- Bolt on an 8/8, Murder on an indestructible creature, a player exiling their own creature. The
  ability resolves and the entity stays, or the wrong side loses something. These teach the effect
  head's affected-or-unaffected output and its zone-outcome field to read the target's state.
- Targeting through a ward cost the caster cannot pay. The cost record's outcome field already
  carries `countered`; the class is empty in gen-1's corpus.
- A 1/1 attacking into a 4/4, and blocks Forge would never assign, including the combats deathtouch
  decides, which Forge's combat AI declines ([rare-keyword combats](#two-build-settings-leave-gate-2-and-the-trainer-short-of-rare-keyword-combats)). Combat records are
  the observational complement, and random declarations reach the region Forge's attack logic never
  enters.
- "Tap target creature" on a creature that is already tapped. A resolution that changes nothing,
  which the corpus has never recorded.

#### Gen-2 collects one match in four with a seat that takes a random legal action at half its decisions

The random seat is Forge's AI with its choices overridden at the decision points that matter: the
spell or ability to play, its targets, the attackers to declare, and the blocks to assign. At each
of those points it acts at random with one probability, the same at every point, and then draws
from the legal options. Otherwise it takes Forge's ranked choice. Land drops, mana payment
and mulligans stay with Forge, because a random land drop produces no record the corpus lacks and
only makes the board less like one the model will be asked about. The other seat stays the standard
Forge AI, so the game keeps a shape Forge would produce and ends. Two random seats never play each
other.

The override lives in the connector, not in Forge. The match worker already builds both seats as
`LobbyPlayerAi` in `GamePlayer`, and Forge routes spell choice, target choice and combat
declarations through `PlayerControllerAi`, so the random seat is a controller subclass the worker
installs on one seat. Nothing on the `effect-record-hooks` branch changes.

A random draw picks one element at a time, because the legal attacks, blocks and target sets are
combinations too numerous to list and draw from.

| Decision | Random draw |
|---|---|
| spell or ability | uniform over the playable spells and abilities other than mana abilities |
| attackers | each creature that can attack does so with probability ½ |
| blockers | each creature that can block picks uniformly from no block and the attackers it may legally block |
| targets | a count uniform between the ability's minimum and maximum, then that many distinct legal targets |

A declaration Forge rejects as illegal is redrawn. After a fixed number of failed draws the seat
takes Forge's choice for that decision. The play draw never passes while something is playable,
because a pass produces no resolution record, and resolutions Forge would not choose are what the
random seat is for. Mana abilities are left out of the play draw because mana payment stays with
Forge. A mana ability drawn at random would tap a land for mana that nothing spends.

Four details are fixed in this design rather than left to the collection run.

1. **Records from the random seat carry a flag.** The flag lets the evaluation score the model
   separately on off-policy records, which is the number that says whether off-policy collection
   worked.
2. **Playability decision records are unaffected.** The `decision` subkind records the rules'
   verdict on every candidate, not the action the seat took, so a random seat produces the same
   decision records a Forge seat does. Nothing in that class needs the flag.
3. **One match in four has a random seat, and it acts at random at half its decision points.** The
   random seat produces rare outcomes for common texts, and the attacks Forge's AI would not make,
   which are the only source of block decisions against such attacks once what-if queries are not
   collected ([what-if queries](#most-blockers-records-are-the-ais-what-if-queries-and-nothing-in-a-record-says-which-kind-it-is)). On-policy games supply everything else, so the two share
   one corpus. The two settings multiply to one eighth, the expected share of one seat's decisions
   taken at random per match played. A seat that acted at random at every point would lose most
   games early, and its boards would say little about the boards the model is asked about.
4. **Random-seat games are played in both collections.** They appear in the training collection,
   played from pools depleted of every card that carries a held-out text, and in the full-strength
   collection, played from complete pools, that supplies the card-disjoint stratum. The off-policy
   records of training therefore have held-out counterparts to be scored on.

Two limits are accepted. Random targets on a wide board mostly produce uninformative no-ops, so the
share of useful records per game is lower than under Forge's play. And random attacks and blocks
cost the random seat life and creatures, so its games run shorter and its boards are thinner than
Forge's. The probability of acting at random sets how far both go. The first pilot collection, at
one match in eight and a quarter of the decisions, yielded too few block decisions against random
attacks to replace the what-if queries, which is what sets the values above
([the first pilot](#the-first-pilot-collection-showed-every-gen-2-mechanism-working-and-two-problems-2026-10-09)).

### The rarity weight is flat for 97% of texts and nothing balances rule families, so rare texts and rare rules train least

The rarity weight the trainer applies within a class does nothing for the tail of the curated
corpus. Curation is the `build-corpus` step that selects which collected records enter the training
corpus. It fills each class, a record category such as the effect halves of resolution records or
the legality records, to a fixed share of the corpus. A record is weighted by the inverse square
root of the games its text was seen in, with a ceiling of twenty times the weight of the
most-observed text. The most-observed texts are the five basic-land mana abilities, each seen in
about ninety thousand games. Measured against that reference, the ceiling binds for every text seen
in fewer than about 228 games, which is 97% of the rarity table. A text recorded in one game and a
text recorded in two hundred receive the same weight. Only the mana abilities, the enters-tapped
replacement and the eight hundred or so texts above that line weigh less. The figures below are from
the manifest of the corpus gen-1 trained on.

| gen-1 corpus manifest | |
|---|---:|
| texts in the rarity table | 31,561 |
| most-observed text (`Add {B}`), games | 91,133 |
| 99th / 95th / 50th percentile, games | 447 / 153 / 11 |
| texts seen in one game | 2,639 (8%) |
| texts seen in four games or fewer | 8,681 (28%) |
| texts at the weight ceiling | 30,740 (97%) |

Under the flat weight, a rare text reaches the encoder about a hundred times less often than a
common one. A curated shard holds 2,000 records. An epoch, one pass of the training schedule, visits
256 shards in 5,000 training steps of 32 records each, so it consumes about a third of each shard it
visits. The weighted shuffle decides which third, heaviest first. With a flat table that draw is
uniform, so a text's share of an epoch is its share of the records that survived the per-text cap.
The cap is 200 records, and a text seen in one game contributes one to three.

The per-text cap is the only setting that changes the mix of texts in the curated corpus, and it
acts only on the most common texts. The cap drops about three of every five resolution-effect
records read and about half or more in every other ability-keyed class. A cap is a ceiling and never
a floor, so curation cannot add tail records. What it can do is stop the common texts from crowding
them out, and in gen-1 it does that only down to two hundred records per text.

| class | records read | kept | dropped by the cap |
|---|---|---|---|
| resolution-effect | 2,226,674 | 233,317 | 1,311,253 |
| resolution-cost | 1,832,641 | 61,719 | 943,357 |
| trigger | 1,864,406 | 62,447 | 936,613 |
| continuous | 1,354,894 | 92,007 | 875,225 |
| rewrite | 187,434 | 54,130 | 101,220 |

#### Gen-2 sets the weight ceiling against the 99th-percentile text and logs what the tail receives

The ceiling is expressed against the text at the 99th percentile of games rather than the single
most-observed one. With the reference at about 450 games, a one-game text weighs about twenty times
the reference and the ceiling barely binds, which is what the ceiling was written to do: stop a
single text from dominating a batch without flattening everything below it. The mana abilities and
the few hundred texts above the reference weigh less than it, which is where they belong. The rarity
table in the manifest does not change; only the trainer's reading of it does.

Nothing in gen-1 reports whether training reaches the tail. The manifest's unique texts per output
show only whether curation kept it. The trainer's epoch line therefore gains the share of records trained
by rarity bucket of their text: one game, two to four, five to nineteen, twenty or more.

#### A per-text weight shows a mechanism once per variant, so a keyword with few variants stays rare

Equalising texts does not equalise mechanisms. Every "deal N damage to target X" script is its own
text, so the damage mechanism reaches the encoder once for each of its variants, and a keyword
written on a handful of cards reaches it a handful of times however heavily each of its texts is
weighted. The weight fix above removes the bias between texts and leaves this one untouched. The
table groups the rarity table's texts into rule families: by Forge API type, trigger mode or
keyword.

| rule family | distinct texts | games |
|---|---|---|
| ChangesZone triggers | 5,393 | 312,514 |
| Pump | 1,989 | 73,949 |
| DealDamage | 1,224 | 66,343 |
| Flying | 608 | 20,153 |
| Suspend | 52 | 3,095 |
| Ninjutsu | 22 | 738 |
| Cascade | 11 | 1,052 |
| Dredge | 7 | 602 |
| Cypher | 0 | 0 |

Under equal per-text weights the damage family gets well over a hundred times the attention of
dredge.

#### Gen-2 balances the curated corpus across rule families, as close to uniform as the records allow

The embedding should carry a comparable understanding of every rule of the game, not a picture of
how often each rule comes up in Forge's games. The ideal is the same number of examples for every
rule. No mapping from records to the numbered rules of the comprehensive rules exists, but Forge
implements roughly one class per rule mechanism. Each record kind therefore takes its family from
the mechanism vocabulary Forge already uses, and a keyword line's family is the keyword whatever the
record kind.

| Record kind | Family |
|---|---|
| resolution, effect and cost halves | the acting ability's API type |
| trigger | the trigger mode |
| continuous | the static mode |
| rewrite | the replacement type |
| combat | the damage-step keywords present in the fight, or none |
| playability | the mode of the static that forbids it, or else the verdict that failed |

A playability verdict that fails on mana has no static behind it, and neither does one that fails
for want of a target. Where no static is responsible, the first verdict that failed names the
family: cannot be played, unaffordable, or no legal target. Without that rule, every unaffordable
and every untargetable candidate would share one family with the candidates that pass.

`build-corpus` gives every family within a class the same target number of records. A family with
more records than the target is cut down to it. A family with fewer is written whole, and each of
its records is repeated up to a reuse cap. Each family therefore receives the target or its records
times the reuse cap, whichever is smaller.

The reuse cap is what stops uniformity from turning into memorization, in which the model predicts
from the identity of a text rather than from what the text says. Dredge has seven texts in about six
hundred games. A budget equal to that of the ChangesZone triggers would replay those few cards and
boards hundreds of times, and the model would learn the cards rather than the rule.

The per-text cap counts copies as well as distinct records. A record repeated four times counts four
against its text's cap of two hundred, so no text is replayed past the cap. A family made of a few
texts therefore stays short rather than filling its target with them, and its shortfall goes on the
list below.

A family short of its target is a collection problem rather than a sampling one. Cypher has no
record at all, and no weight reaches it. The manifest reports each family's available records, its
target, the records written, the repeats and the shortfall. The shortfall list is the target list
for `collect-coverage` and `collect-variants`.

Inside a family, the text weight and the per-text cap of the previous subsections balance the texts.
The epoch line reports the share of records trained per family beside the share per rarity bucket.
The validation results are reported per family and averaged over families, so the evaluation weighs
the rules the way training does.

#### Within a family, curation balances the outcomes as well as the texts

A family's records are balanced over a coarse outcome signature: which pairs of zone outcome and
changed-or-not occur among the affected entities. Lightning Bolt killing its target and Lightning
Bolt leaving it standing then appear in comparable numbers. On-policy play records almost only the
first.

The signature ignores how many entities share a pair, so that a mass effect is not split by board
size. A wrath that kills three creatures and one that kills seven produce the same outcome on boards
of different sizes. If counts entered the signature, each board size would become a cell with its
own share, and the rare large boards would be repeated to fill it.

Balancing the outcomes changes which boards the model sees and not the right answer on any of them,
because most outcomes follow from the board under the rules: Bolt on a creature with three toughness
always kills it. It shifts the head only where the outcome depends on a player's choice, such as
which creature an opponent sacrifices, and that choice is Forge's policy rather than a rule.

The minority outcomes come mostly from the random seat, so this balance is what turns the random
seat's games into training signal. The random-seat flag is not itself a curation key: the signature
selects the minority outcomes directly, from whichever seat produced them. The manifest records the
on-policy and off-policy record counts per class beside the existing per-class counts.

### The training noise on the ability vector is under 1% of its length, so it does not stop the effect head memorizing texts

Gen-1's noise on `e` is too small to stop the effect head memorizing texts, because nothing keeps
`e` small and the noise does not scale with it. The design doc's first memorization lever keeps `e`
small and noisy in training, so that the effect head cannot use `e` as a precise per-text lookup key
("Coverage and memorization are separate tail problems, and both get levers" in the design doc).
Gen-1 implements only the noise half. The encoder adds Gaussian noise with a fixed standard
deviation of 0.05 to each of the 64 coordinates (`e_noise` in `ability_encoder.py`). `e` is a linear
projection of the encoder's output, restrained only by the optimizer's weight decay, a pull of every
weight toward zero by a factor of 0.01 of the learning rate per step. A larger `e` makes the same
absolute noise relatively smaller, and the effect head reads `e` through a linear layer, which
absorbs any scale.

In the trained gen-1 encoder the vectors are large enough that the noise no longer matters. The
figures are over every row of the shipping cache under `output/effects/abilities/cardsfolder/`, with
the taxonomy baseline's files left out. The noise moves a vector by less than a hundredth of its
length. Along the ninth principal direction, the weakest the encoder uses, the spread of the vectors
is almost ninety times the noise.

| shipping ability cache | |
|---|---|
| vectors | 66,142 |
| length, mean | 53.4 |
| length, 10th / 50th / 90th percentile | 40.5 / 54.3 / 64.4 |
| length, shortest | 23.6 |
| standard deviation of one coordinate, over all vectors | 6.8 |
| standard deviation along the ninth principal direction (design doc Outcome) | 4.4 |
| noise, standard deviation per coordinate | 0.05 |
| noise, expected length in 64 dimensions | 0.40 |

The effect head can therefore tell texts apart at a precision far finer than the typical distance
between two of them, which is the lookup the lever was meant to make expensive. In gen-1 only the
design doc's other four levers acted against memorization.

#### Gen-2 draws the noise with the covariance of the `e` vectors, estimated as a running average

Five fixes are open, and they differ in whether the encoder can still escape the noise.

- **Normalizing each `e` to a fixed length before the noise.** The fixed noise then has a scale it
  cannot escape. It also discards length, which ranges over a factor of nearly three in the shipping
  cache and may carry information. It changes the vectors the cache stores, so every consumer reads
  a different `e`.
- **A penalty on the length of `e` in the loss.** It adds a second weight, and the balance between
  that weight and the noise is what sets the noise's effective size.
- **One noise level for every direction, proportional to the average spread of `e`.** Rescaling `e`
  as a whole then changes nothing. The encoder can still widen a few directions, though. The noise
  is sized to the average, so along those directions it is small against the spread, and they can
  carry a precise per-text key. The same reward for piling variance into a few directions is what
  gate 3's concentration check flags.
- **Noise in each coordinate proportional to that coordinate's spread.** Widening one coordinate no
  longer helps. The coordinates are arbitrary axes, however. A signal laid along a diagonal across
  many coordinates meets noise smaller relative to its spread by up to the square root of the width
  of `e`.
- **Noise with the covariance of the `e` vectors.** The noise is wide along the directions in which
  the vectors spread widely and narrow along the others, diagonals included. The ratio of noise to
  spread, the noise ratio, is then the same along every direction, and no reshaping of `e` changes
  it.

Gen-2 takes the last. Five details settle it.

1. **The covariance is measured around the mean, with the gradient stopped.** A common offset added
   to every vector separates no two texts, so it must not raise the noise. With the gradient
   stopped, the covariance is treated as a constant when the loss is differentiated, so the noise
   cannot become a path through which the loss changes `e`.
2. **The covariance is a running average over recent batches.** One batch spans at most one
   direction fewer than it has distinct texts. A batch with fewer distinct texts than `e` has
   dimensions would leave some directions without noise, and those directions could carry a key. One
   batch's estimate also varies with the batch size, which differs between sweep arms that fit the 8
   GB card differently. The average starts from the first batch's covariance.
3. **The noise ratio ramps up over the first epoch.** It rises linearly from zero to its value, the
   same way in every arm. Early in training `e` grows quickly and the running average lags behind
   it, which makes the noise smaller than intended rather than larger. The ramp makes the start of
   training independent of that lag.
4. **The noise moves from the encoder to the batcher.** It is applied to the vectors the batch
   carries, which is where the running average is kept.
5. **The noise ratio comes from a pilot on the gen-1 corpus.** Gen-1's effective ratio per
   coordinate was under a hundredth, from an absolute noise of 0.05 against a spread of 6.8. The
   pilot trains at ratios of 0.05, 0.1 and 0.2 with the auxiliary head of the next section switched
   on. It reads each run's results on unseen texts and the concentration of its `e`. Every sweep arm
   then uses the chosen noise ratio.

The cache and its consumers do not change. The noise is applied in training only, and the encoded
`e` keeps its form and its unconstrained scale.

Gate 3 is where the change should show. Gate 3 caps the share of the embedding's variance on its top
principal component. Under noise with the covariance of the `e` vectors, the noise ratio is the same
along every direction, so widening one direction separates no more texts. The encoder separates the
most texts by spreading them over as many directions as it can. The change is therefore expected to
lower the top component's share, and gate 3 reports whether it does.

### The ability vector keeps less of an ability's amounts and costs than a hash of its parameter keys

Gen-1's `e` holds less about amounts and costs than a hash that records only which parameters a
script sets. The figures are linear probes from the design doc's Outcome. The taxonomy hash holds
every parameter key's presence and no value.

| Property | Metric | gen-1 `e` | taxonomy hash |
|---|---|---:|---:|
| damage amount | R² | 0.34 | 0.55 |
| mana in the activation cost | R² | 0.06 | 0.33 |

Lightning Bolt's nearest neighbours show that the amount is lost. They are the same sentence with 1,
2, 4, 5, 6, 7, 10 and X damage, all at a cosine similarity of at least 0.997.

The effect head's loss rarely depends on the number. Forge casts Bolt mostly at creatures it kills,
so on the records the corpus holds, three damage and four have the same outcome. Costs are
supervised only through the cost halves of resolution records and the playability verdicts, and few
of the effect head's output fields score either. The outcome balancing above makes the amount matter
more often, but nothing in the loss asks `e` for a cost at all.

#### Gen-2 adds an auxiliary head, used in training only, that reads from `e` the values the script states

The auxiliary head reads `e` alone and predicts numbers parsed from the script:

- damage dealt, power and toughness change, counters placed and cards drawn;
- the mana cost by colour, plus its generic part;
- whether the cost taps or sacrifices.

The labels come from the sidecar's script parameters, for every segment of the chain, at no cost. A
label is masked where the script states no fixed number, as with X. The head follows the pattern of
the script-API head, gen-1's training-only head that predicts each script's API type from `e`: a
small loss weight, used in training only, and filtered out at save time.

Costs come from the line's `Cost$` alone. A spell's mana cost is printed on the card rather than on
any line, and it pays for the whole card, so no line is labelled with it. A spell line's mana-cost
labels are masked even when an additional cost puts mana in its `Cost$`, as in Bone Splinters'
`Cost$ B Sac<1/Creature>`. The effect head reads the card's mana value and colours from the board
snapshot and relates them to the card's abilities itself.

The weight is the one setting to watch. Gen-1's `e` predicted observed outcomes better than the
parsed script features did, because it groups abilities by what they do rather than by how their
scripts are written. A large weight would pull `e` back toward the parser's grouping.

The game agent reads `e` through the cache, and the amounts and costs are what it needs to weigh one
card against another. The knowledge probes that read amounts and costs from `e` become a check that
the head did its job. The probes that ask whether the model uses an amount against a toughness stay
independent of it.

### Gate 1 scores fewer than half of the held-out texts, and a third of the recorded ones have a numeric twin in training

Held-out texts are chosen by hash over the converted texts, not over the recorded ones, and nearly
half of them were never recorded. Sealed self-play never cast the cards that carry them, and
`collect-coverage` keeps held-out cards out of its decks by design, so nothing fills the gap. Of the
texts that were recorded, a third were seen in four games or fewer. The eligibility rule, which
holds out a text only if at most eight cards carry it, excludes almost nothing, because nine texts
in ten are on exactly one card.

| held-out texts, gen-1 | count |
|---|---:|
| held-out texts | 735 |
| with any record | 393 |
| with a gate-one resolution record | 342 |
| recorded, seen in one game | 43 |
| recorded, seen in two to four games | 82 |
| recorded, seen in five to nineteen games | 142 |
| recorded, seen in twenty games or more | 126 |
| converted texts carried by one card | 33,456 of 36,981 |

The validation sample that selects checkpoints is dominated by the well-recorded held-out texts. The
sample is drawn by smallest record hash, which is uniform over records, and the gate-one slice holds
about eighteen thousand records over its 342 texts. The resolution slots of the 2,048-record sample
therefore go to the texts with the most records, and a text seen in one game holds a few at most.
Gate 1 itself averages over entities and records, not over texts. The same well-recorded texts
therefore decide all three of its margins over the identity baseline: affected-entity F1,
zone-outcome accuracy and count deviance. Gate 1 therefore measures how the model reads the
well-recorded held-out texts. How it reads a held-out text recorded in a single game is measured
nowhere.

A third of the recorded held-out texts have a twin in training. A twin is a training text that is
identical once numbers, `CARDNAME` and the description parameters are masked, such as `NumDmg$ 2`
held out beside `NumDmg$ 3` in training. This masked form is the text's template. Corpus-wide, two
recorded texts in five share a template with another. The identity baseline cannot exploit a twin,
because it keys on the exact text. The encoder can, and reading `NumDmg$` is what it should do. A
gate averaged over held-out texts of which a third are one number away from a training text measures
interpolation between twins more than reading. The design doc's four-way stratified report does not
isolate these twins: its numeric-extrapolation and novel-combination strata are different cases.

| numeric twins | count |
|---|---:|
| recorded held-out texts | 393 |
| with a training text equal up to numbers, names and descriptions | 137 |
| recorded texts corpus-wide | 31,561 |
| sharing such a template with another recorded text | 13,320 |

#### Gen-2 holds out whole templates and reports unseen texts per text and per rule family

Four changes. The first two change how the card-disjoint sample is drawn and how its results are
reported. The third changes the holdout key, which acts at collection time. The fourth is a coverage
round that gives the held-out texts enough records to score.

1. **The card-disjoint sample is drawn per held-out text.** Its resolution slots fill round-robin
   over the held-out texts, a fixed number of records per text by smallest hash within the text,
   until the class quota is met. Every recorded held-out text is then in the sample, and the loss
   that selects checkpoints weighs a one-game text comparably to a hundred-game one. The
   card-disjoint stratum already caps games per held-out text for the same reason; this is the same
   rule one level down.
2. **Results on unseen texts are reported per text and by bucket.** A per-text mean sits beside the
   per-record numbers. The buckets are the rarity bucket of the held-out text's games and the rule
   family of the text, and the family results are also averaged over families. The family view is
   what shows a keyword the model reads badly, which a mean over the damage spells would hide. The
   memorization gap is the gap between a family's results on the game-disjoint stratum, whose texts
   trained, and on the card-disjoint stratum, whose texts did not. Gate 1 itself is not computed,
   because gen-2 trains no identity baseline ([baselines](#gen-2-trains-no-baselines-because-the-knowledge-probes-measure-what-the-baselines-measured)).
3. **The holdout key is the masked template.** A text is held out when the hash of its template
   falls below the holdout fraction, a number of templates per thousand (`--holdout-permille`), so a
   text's numeric twins go with it and every held-out text is unseen in training, numbers included.
   The template is taken from the chained script text, and the new corpus is depleted against it
   ([collecting from scratch](#gen-2-collects-its-corpus-from-scratch-because-the-holdout-and-the-random-seat-both-act-at-collection-time)). Chain labels need no mask, because the encoding text
   has already renumbered them ([SVar labels](#gen-2-renumbers-svar-labels-by-position-because-the-names-authors-give-them-carry-no-mechanism)). The eligibility rule counts the cards that
   carry any text of the template, not the cards of one text: a template such as `Pump +N/+N until
   end of turn` spans thousands of cards, and holding it out would deplete them all. The holdout
   fraction is set against the share of cards the holdout depletes, which `holdout-cards` reports
   before collection starts. The build lists the held-out texts with no gate-one record and those under a floor of
   five games, which is the target list for the next point.
4. **A coverage round over the held-out cards alone.** `collect-coverage` gains the inverse of
   `--exclude-cards`: a run restricted to the held-out cards, writing full-strength records. Every
   game it produces names a held-out card, so the split rule routes it to the card-disjoint stratum
   on its own. The round runs until every held-out text with a castable carrier reaches the floor.
   The held-out texts it reports as still under the floor are the ones no game can cover.

### Keyword expansion builds the cache from definitions gates 1 and 2 never score, and misses every keyword whose name is not a single token

Keyword expansion replaces a keyword token with its definition, the reminder text Forge prints for
the keyword ("New keywords are handled by expansion dropout over their definitions" in the design
doc). It has four defects. The cache and the gates read different encodings of every keyword line. A
keyword whose name does not become a single token never finds its definition: every hyphenated
keyword, and every multi-word keyword the vocabulary does not already know. The definition inserted
is a damaged template. And `build-vocab` scans Forge's generated keyword scripts into the script
vocabulary, though no expansion reads them. A fifth gap is in the evaluation: gen-1 withheld no
keyword, so the check that a keyword is read through its definition measured nothing. The evidence
comes from `scripts/effect_embedding_probes/keyword_expansion.py`, with its outputs in
`output/effects/reports/keyword-expansion-20260919/`. What the analysis found about the encoder
itself is in the design doc's Outcome, in "A keyword token encodes only loosely like its own
definition" and "An unknown keyword expands only when its name is one word".

The shipping cache encodes every known keyword as its definition, while the effect head trained on
the keyword token three times in four and gates 1 and 2 score the token every time. The expansion
probability is the chance that a known keyword is replaced by its definition, and each caller sets
its own. Training uses 0.25. Validation and gates 1 and 2 go through `SurfaceBatcher` with 0.
`encode-abilities`, which builds the cache, uses 1.0, so that two runs of the command write the same
file. An unknown keyword is expanded at every probability. Forty keyword rows of the shipping cache
match the full expansion at a mean cosine of 1.0000 and the unexpanded encoding at 0.77. Gate 3, the
ward canary (the distance between ward and the card lines that spell out its effect), the embedding
probes and the deployed head all read the cache. For a keyword line, each of them therefore reads an
input the head saw only a quarter of the time in training, and never the input gates 1 and 2 scored.

Expansion also fires on a keyword word inside other lines, such as a grant's description, so the
cache moves more than the keyword lines. Keyword lines move furthest. Cosine similarity below is
between a text's unexpanded and fully expanded encodings, 1 for the same direction.

| unique script texts in the corpus | 37,106 |
|---|---:|
| changed by full expansion | 7,658 (20.6%) |
| of which keyword lines | 1,925 |
| of which other lines naming a keyword | 5,733 |
| cosine over changed texts, median / 10th percentile / minimum | 0.968 / 0.748 / −0.662 |
| cosine over changed keyword lines, median / 10th percentile | 0.893 / 0.371 |
| cosine over changed other lines, median / 10th percentile | 0.979 / 0.868 |
| share of a text's 10 nearest neighbours that the other encoding keeps, 500 changed texts | 0.33 |

Expansion finds a definition through the token the tokenizer produced, so a keyword whose name is
not a single word expands only if the vocabulary already knows it. A multi-word name becomes one
token, `first_strike`, only by matching a merged vocabulary entry, which exists only for known
keywords. A hyphenated name never expands: the tokenizer cuts `Jump-start` at the hyphen and the
definition is filed under `jump-start`. Jump-start, web-slinging and beam me up never expand, on 24
lines. Starting intensity, on 20 more, has no reminder text to expand to. The same mismatch between
a display name and its token disables `HOST_BODIED_KEYWORDS`, the list of keywords whose body is
printed on the host card and which are never to expand. It spells `level up` and `read ahead` with a
space, the tokens are `level_up` and `read_ahead`, and both expand, on 26 and 10 lines.

The definition inserted is damaged in four ways.

- `_instantiate` strips only `%s`. The 38 templates written with `%d` or `%1$s` keep the specifier
  as tokens: toxic's reads "also get % d poison counters".
- No caller passes the line's own values. `expand_keywords` accepts `instance_values` and nothing
  supplies them, so a template loses its value and the line's parameter is left dangling after the
  sentence.
- 16 definitions contain `[UNK]`, the placeholder token the encoder reads for any word missing from
  the vocabulary. `build-vocab` scans the definitions but keeps only the 5,000 most frequent tokens,
  and 21 words of reminder text fall below that cut: `encoded`, `promise`, `specified` and
  `teammate` among them.
- Five templates (enlist, increment, read ahead, station, web-slinging) use the typographic
  apostrophe `’`, which becomes a token of its own rather than the `'` every other text uses.

| keyword line | what the encoder reads under expansion |
|---|---|
| `Toxic:1` | players dealt combat damage by this creature also get % d poison counters . : 1 |
| `Ward:2` | whenever this permanent becomes the target of a spell or ability an opponent controls , counter it unless that player . : 2 |
| `Enchant:Creature` | target a % 1 $ s as you cast this . this card enters attached to that % 1 $ s . : creature |
| `Start your engines` | if you have no speed , it [UNK] at 1 . it [UNK] once on each of your turns when an opponent loses life . max speed is 4 . |

No expansion reads Forge's generated keyword scripts, yet `build-vocab` scans them into the script
vocabulary. Both surfaces expand to the reminder template: `_definition_text` returns it, and no
code substitutes the generated script.

#### Gen-2 builds the cache with known keywords left as tokens

`encode-abilities` expands at probability 0, the setting validation and gate 2 use. The choice keeps
what 1.0 was chosen for: an unknown keyword still expands at every probability, so the cache stays
deterministic and a new set still reads through its definitions. It also makes the cache the input
the head saw three times in four in training and the one validation and gate 2 score. Expanding at
1.0 in training as well would make the two agree the other way, but it removes the keyword token the
design keeps, and the design doc's Outcome shows the encoder has learned that token as a symbol of
its own rather than as its definition.

1. **One setting, read by both callers.** The probability for scoring and for the cache is one
   constant that `SurfaceBatcher`'s scoring path and `AbilityEncoderRunner` both read, just as
   `SurfaceBatcher.text_of` is the one definition of the encoding text. Each caller setting its own
   value is how the cache and the gates came to read different inputs.
2. **The geometry is measured again.** Gate 3, the ward canary and the embedding probes are re-run
   on the new cache. Their keyword lines then describe the keyword token, which is what the head
   reads.

#### Gen-2 recognises a keyword line by its display name, not by a vocabulary token

The definition is looked up by the display name, the part of a keyword line's script text before its
first colon, matched case-insensitively against the definitions table. The whole name is replaced,
however many tokens it split into, and the text after the colon becomes the line's instance values.
A keyword word inside another line keeps the token path it has in gen-1.

1. **The match is checked against the sidecar.** The batcher sees only the encoding text, so the
   match is made on the text. The corpus build asserts that every text it matches belongs to sidecar
   lines whose `script_api_type` is `Keyword`. The sidecar's `line_kind` is not the key: it is the
   converter's category, `alternate cost` for jump-start.
2. **The host-bodied list is checked on the same path.** It compares display names, so level up and
   read ahead are excluded as the list intends.
3. **The keywords that never expand in gen-1 are the test.** Jump-start, web-slinging and beam me up
   must each expand both with and without a merged vocabulary token for their name.

#### Gen-2 fills each template with its line's values, strips every format specifier and reserves the definition words

1. **Instance values are filled.** The values after the colon fill the placeholders in order, and
   `%1$s` repeats its value at each use. Forge formats some values before filling them: ward's `%s`
   stands for a clause such as "pays {2}", not the bare `2`. Those formatted strings come from
   Forge, recorded per keyword by `extract-keyword-definitions`, rather than being rebuilt in
   Python.
2. **Every specifier is stripped where no value fills it.** The pattern covers `%s`, `%d` and the
   positional forms, and no expansion leaves a `%` token.
3. **The apostrophe is normalised.** `’` becomes `'` when the definitions are loaded.
4. **The definition words are reserved in the vocabulary.** `build-vocab` seeds every word of every
   template after stripping, as it already seeds `[PAD]` and `[CLS]`, so `--target-size` cannot cut
   them. The build asserts that no definition expands to `[UNK]`. The reservation lands in the
   script-vocabulary rebuild that the chain encoding and the camel-case split
   ([tokenizer](#the-script-tokenizer-is-never-called-and-the-prose-grammar-that-runs-in-its-place-splits-selectors-better)) already require, so it costs no rebuild of its own.

#### Gen-2 keeps the reminder template on both surfaces and stops scanning generated scripts

The generated script is not worth expanding to. It is the root line of the keyword's trait
alone. Prowess's is its `SpellCast` trigger condition without the +1/+1 it grants, and no trigger
among the 75 scripts carries the effect it executes: the same first-line gap the chain section
describes. Its description parameter holds the reminder text anyway. And only 75 of the 202 keywords
have one, so expanding to it would read three keywords in five in one language and the rest in
another.

1. **The template stays the definition on both surfaces.** The generated script is revisited once
   `extract-keyword-definitions` renders the whole chain, as the converter will for card lines.
2. **`build-vocab` stops scanning generated scripts**, so the capped vocabulary counts only text the
   encoder reads.

#### Gen-2 withholds one keyword, so the zero-shot check has something to measure

The zero-shot keyword check is the only test that a keyword the model never trained on is read
through its definition, which is how a new set's keyword reaches the model. Gen-1 withheld no
keyword, so the check reported nothing. The design doc's Outcome adds a reason to run it: an unknown
keyword's expansion lands far from the vector of the trained keyword it defines.

1. **Every sweep arm withholds the same keyword.** `train-effect-model --withhold-keyword` keeps its
   token out of training and expands every occurrence to its definition. The keyword is chosen
   before the sweep, among the script-generated keywords new sets introduce, with enough records in
   the corpus for its fields to be scored.
2. **The check measures.** Gen-1's evaluator names the withheld keyword and stops. Gen-2 adds the
   measurement the design doc specifies: the effect head's per-field results on records where the
   withheld keyword acts or sits on the board, beside the same fields for keywords that trained.

Every gen-2 arm deleted cascade, the withheld keyword, instead of expanding it, so the check measured
nothing ([the withheld keyword](#every-arm-deleted-the-withheld-keyword-instead-of-expanding-it-so-the-zero-shot-check-measured-nothing)).

### The script tokenizer is never called, and the prose grammar that runs in its place splits selectors better

`AbilityTokenizer.tokenize_script` is the tokenizer written for the script surface, and no code
calls it. Encoding for the cache (`ability_encoder_runner.py`) and encoding during training
(`surface_batching.py`) both pass the script text to `tokenize`, the prose grammar. The vocabulary
is built with the prose grammar too. `build-vocab --surface script` hands the script lines to the
shared price-predictor vocabulary builder, which splits them with `MtgTokenizer`. The design doc
counts a script tokenizer that splits compound selectors such as `Creature.nonDragon+OppCtrl` among
the costs of the script surface, in the design doc's section "The script is the primary surface;
prose is the paired secondary".

The prose grammar already makes that split. It ends a word at every character that is not a letter
or an underscore, so `.` and `+` end a word and are kept as tokens of their own. `tokenize_script`
makes the same cut at `.` and `+` but drops the separators, and before cutting it keeps letters,
digits and `-` together in one run. A sign then either vanishes, so `NumAtt$ +1` reads as `1`, or
fuses with its number into `-1`. A threshold fuses with its digit, so `powerGE4` becomes one token
and the 4 never reaches the numeric embedding, the encoder's separate input path that reads a
number's value. The vocabulary was built with the prose grammar, so every fused token is unknown to
it.

| script text | `tokenize` (runs) | `tokenize_script` (never called) |
|---|---|---|
| `ValidTgts$ Creature.nonDragon+OppCtrl` | validtgts $ creature . nondragon + oppctrl | validtgts $ creature nondragon oppctrl |
| `NumAtt$ +1` | numatt $ + 1 | numatt $ 1 |
| `NumDef$ -1` | numdef $ - 1 | numdef $ -1→[UNK] |
| `CounterType$ P1P1` | countertype $ p 1 p 1 | countertype $ p1p1→[UNK] |
| `ValidTgts$ Creature.powerGE4` | validtgts $ creature . powerge 4 | validtgts $ creature powerge4→[UNK] |

Measured against the shipping vocabulary `models/effects/vocab-script.txt`, the unused tokenizer
produces more than twice the unknown tokens of the one that runs. The sample is 5,000 script lines
from randomly drawn cards. "Parameters only" is the same lines with the text of every
`*Description$` parameter removed, which leaves the part of a line the script tokenizer was written
for.

| on 5,000 script lines | `tokenize` | `tokenize_script` |
|---|---|---|
| unknown-token rate, whole line | 0.34% | 0.85% |
| unknown-token rate, parameters only | 0.47% | 1.12% |
| lines with at least one unknown token | 413 (8%) | 824 (16%) |
| tokens per line, mean (median), whole line | 32.5 (34) | 30.4 (31) |
| tokens per line, mean (median), parameters only | 18.4 (20) | 17.6 (19) |

Neither grammar splits compound words, and compound words are where the unknown tokens come from.
Forge writes restrictions and SVar labels in camel case: `nonDragon`, `YouCtrl`,
`TrigDestroyYourLand`. Both grammars cut only at characters that are not letters, so each compound
is one token. `YouCtrl` and `OppCtrl` share nothing, and `nonDragon` shares nothing with the
`dragon` token the prose surface knows. In the parameters-only sample, camel-case compounds are
about one token in six and three quarters of the unknown tokens. For nearly two thirds of those
unknown compounds, every part is already in the vocabulary.

| parameters only, `tokenize`, 5,000 script lines | |
|---|---|
| tokens | 92,191 |
| tokens from a camel-case compound | 15,612 |
| distinct compound tokens | 1,174 |
| unknown tokens | 433 |
| unknown tokens that are compounds | 327 |
| unknown compounds whose parts are all in the vocabulary | 205 |

#### Gen-2 deletes the script tokenizer and splits camel-case compounds on the script surface

`tokenize_script` is removed rather than wired in, because the grammar that runs keeps the signs,
separators and digits it loses. The prose grammar gains one rule on the script surface: a word is
also cut where a lowercase letter is followed by an uppercase one, so `nonDragon` reads as `non
dragon` and `YouCtrl` as `you ctrl`.

1. **The vocabulary scan and the tokenizer apply the same rule.** `build-vocab --surface script`
   already stages the script lines as a text file for the shared builder, so it applies the split to
   that staged text. The price predictor's `MtgTokenizer` does not change. `tokenize` applies the
   rule when the loaded vocabulary is a script vocabulary, which `surface_of` already determines
   from the vocabulary path.
2. **The rule lands with the chain encoding's vocabulary rebuild.** The chain change rebuilds the
   script vocabulary anyway. The holdout hash reads the normalised script text rather than its
   tokens, so this rule does not move the held-out set.
3. **The measurement above is repeated on the rebuilt vocabulary.** The build log reports the
   unknown-token rate on the parameters and the number of distinct compound parts, so the rebuild
   shows whether the unknown compounds went away.
4. **Chain labels stay whole.** The chain encoding links a segment to its parent by its numbered
   label ([SVar labels](#gen-2-renumbers-svar-labels-by-position-because-the-names-authors-give-them-carry-no-mechanism)). Read by the prose grammar, `SV2` becomes `sv 2`, and the digit
   reaches the numeric embedding as if it were an amount. The tokenizer therefore keeps whole the
   values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `Choices$` and `ReplaceWith$`, and the
   label that opens each segment. `build-vocab` reserves `sv1` up to the longest chain's label
   count, so the vocabulary size setting cannot drop one.
5. **A counter type is one token.** The prose grammar reads `CounterType$ P1P1` as `p 1 p 1`, so the
   digits of a counter type reach the numeric embedding as if they were amounts. On the script
   surface a counter type's value stays whole, `p1p1` or `m1m1`, in the tokenizer and in the
   vocabulary scan alike.

### Two build settings leave gate 2 and the trainer short of rare-keyword combats

Gate 2 cannot score six of its eight keywords for want of records, and two `build-corpus` settings
decide how many records it and the trainer get. The game-disjoint stratum that gate 2 reads is 1,000
games drawn uniformly. Curation keeps combat records for training at one rate whatever the fight.
The design doc's section [on why deathtouch and indestructible combats are
scarce](2026-09-04-ability-effect-model-design.md#deathtouch-and-indestructible-combats-are-scarce-because-few-creatures-carry-the-keywords-and-forges-combat-ai-avoids-the-combats-deathtouch-decides)
gives two causes: few creatures carry the keywords, and Forge avoids the combats deathtouch decides.
Both settings act on the first cause only. The random seat acts on the second.

#### Gen-2 draws the game-disjoint stratum toward games with rare-keyword combats

A uniform draw of 1,000 games holds too few of the rare keywords' combats for gate 2. The counts
below are gate 2's qualifying records on gen-1's stratum, against a floor of 200.

| Keyword | Qualifying records, 1,000 uniform games |
|---|---:|
| first strike | 47 |
| deathtouch | 69 |
| trample | 179 |
| indestructible | 19 |
| wither | 11 |
| infect | 73 |

Drawing the stratum toward the games that hold these combats gives gate 2 enough records without
enlarging the stratum. A game enters the stratum when a hash of its `game_id` falls under a
threshold, and the threshold is higher for a game that holds a qualifying combat for a keyword short
of the floor. The cost falls on training. Wither qualifies about once per hundred games in the
stratum above and about twice per hundred over both of gen-1's validation strata. A corpus of
gen-1's size therefore holds roughly 1,300 to 2,600 wither combats, and reaching 200 takes between a
thirteenth and a seventh of them out of training. The other keywords lose less.

The rule depends only on the game, so a rebuild over a grown corpus keeps every game it already
placed in the stratum and only adds new ones. The seeded sample gen-1's `build-corpus` draws redraws
the whole stratum whenever the corpus grows. That would move the games the knowledge probes read
into training, and the probes would score later models on boards they trained on.

The stratum over-represents rare-keyword games, so its results are reported per rule family and
averaged over families, the balance training is built to. The manifest records which games entered
for a keyword, so a figure over the stratum can also be read without them.

#### Rule-family balancing gives the rare-keyword combats their own budget

Gen-1's curation cannot favour a deathtouch combat over an ordinary one. It keeps 158,348 of the
1,066,917 combat records it reads, 15%, and chooses them by a hash of the record id, so the choice
is uniform over fights. The per-text cap and the rarity weight never see a combat record, because a
combat record carries no acting ability text.

On the 1,000 game-disjoint games, a damage-step keyword appears in about a quarter of combat
records, and the five rarest in about one in twelve.

| Combat records, 1,000 game-disjoint games | Records | Share |
|---|---:|---:|
| all | 8,762 | 100% |
| with any of the eight damage-step keywords | 2,051 | 23.4% |
| with deathtouch, double strike, indestructible, infect or wither | 740 | 8.4% |

Gen-2 balances combat records by the family table of the rarity section. A combat's family is the
set of damage-step keywords present in the fight, so a wither fight and a fight between two vanilla
creatures receive the same target. The manifest's per-family counts show how far each keyword's
combats fall short of it.

### Most `blockers` records are the AI's what-if queries, and nothing in a record says which kind it is

The collector writes a legality record every time Forge's AI computes a legal attacker or blocker
set, not only when a player declares. Forge computes those sets while planning in its main phases,
at upkeep, and while the attacking AI simulates how the defender would block each candidate
attacker. Only one `blockers` record in twenty-two is taken at a real blocking decision. Two in
three `attackers` records are.

A record counts as a real decision here by the same test as the design doc's avoidance analysis. An
`attackers` record is real when its snapshot phase is `combat_declare_attackers` and its actor is
the active player. A `blockers` record is real when its phase is `combat_declare_blockers` and its
anchor attacker is attacking. The counts are over both validation strata, 6,848 whole games.

| Snapshot phase | `attackers` records | `blockers` records |
|---|---:|---:|
| upkeep | 267 | 9,017 |
| draw | 26 | 1,819 |
| main 1 | 987 | 15,231 |
| beginning of combat | 1,250 | 31,040 |
| declare attackers | 4,813 | 55,317 |
| declare blockers | 0 | 6,050 |
| combat damage, first-strike damage, end of combat | 0 | 810 |
| main 2 | 0 | 7,977 |
| end of turn, untap | 0 | 761 |
| total | 7,343 | 128,022 |
| real decisions | 4,813 (65.5%) | 5,895 (4.6%) |

The legal list in a legality record is a rules fact about the board in its snapshot, and the
forbidden list of a `blockers` record is not. No `attackers` record in either stratum lists a tapped
or summoning-sick creature as a legal attacker, and no `blockers` record lists a tapped creature as
a legal blocker. The legality head learns from a class dominated by planning queries about combats
Forge was only considering, and a record gives no way to tell the two kinds apart. The forbidden side
of a `blockers` record also carries the AI's unfinished block plan
([the block-plan defect](#most-block-legality-labels-that-say-a-creature-may-not-block-describe-the-ais-unfinished-block-plan)).

Two further collector rules shrink the real decisions the corpus holds.

- De-duplication keys on the subkind and the payload, the list of legal ids, and ignores the board.
  The collector's comment says only byte-identical records collapse, but a record from a later turn
  with the same legal creatures on a different board is dropped as a duplicate. A what-if query
  earlier in the same turn can consume the key, so the real decision that follows it is dropped too.
- After de-duplication, `--legality-rate` keeps one record in ten. The two validation strata hold
  fewer than one real declare-attackers decision a game.

#### Gen-2 marks each legality record as a real decision or a what-if, as collection metadata

The collector applies the test above as it writes each record. It knows the phase, the active player
and whether the anchor attacker is attacking in the game's combat, and it writes an envelope flag
naming a what-if query.

#### Gen-2 includes the board in the legality de-duplication key

The key becomes the subkind, the payload and the snapshot, the same three parts
`allowDistinctRecord` already hashes for the other record kinds. The repeats the de-duplication was
written for, the AI re-asking during one combat evaluation, share a board and still collapse. A
later turn with the same creatures on a different board is kept.

The pilot collection tests, before the full run, the assumption that the re-asks share a board. If the snapshot carries the attack declarations the AI is simulating, each re-ask has a board
of its own. The legality class then grows back toward the four fifths of the corpus it held before
the collector de-duplicated legality records.

#### Gen-2 records every real decision and collects no what-if queries

The real decisions are exempt from `--legality-rate`, and the rate applies to what-if queries only.
Real decisions happen only at the two declare steps of each turn, so exempting them adds little
volume. They are also the only legality records that join to the combat that follows, which is what
a measurement of Forge's attack and block choices needs.

The rate defaults to 0, so no what-if query is collected. A what-if's verdict is true, but which
queries reach the corpus is not representative of play. The AI asks about one combat many times
while it plans an attack. The rate is drawn for each query before de-duplication, so a query asked
`K` times survives with probability `1 − (1 − rate)^K`. The kept queries are therefore weighted by
how long the AI deliberated over a board, which favours the wide boards it plans longest. The one
thing they hold that real decisions lack is blocks against attacks the AI considered and rejected.
Those blocks reach the corpus instead as real block decisions against the random seat's attacks, on boards the
game actually reached.

Curation splits the legality class equally over the halves the corpus holds, so a corpus without
what-ifs fills the class from real decisions. The manifest records the real and what-if counts per
subkind, and the evaluation reports the legality fields on real decisions separately.

### Gen-2 collects its corpus from scratch, because the holdout and the random seat both act at collection time

A holdout cannot change without new games, because depletion freezes it into the games when they are
played. The training games are drawn from pools depleted of every card that carries a held-out text,
which is what lets a text-keyed holdout cost no training games. Gen-2 changes both inputs to the
holdout. The encoded text becomes the whole script chain, and the holdout unit becomes the masked
template. A new holdout applied to games depleted against the old one would move most of them out of
training, by the arithmetic in the design doc's section [on depleting the
pools](2026-09-04-ability-effect-model-design.md#the-held-out-set-is-built-by-depleting-the-pools-not-by-discarding-games).
The off-policy random seat also needs games played with it.

Everything collection fixes is therefore settled before the first game:

- the holdout: masked templates of the chained script text with its labels renumbered, with the
  eligibility rule counted per template ([gate 1](#gate-1-scores-fewer-than-half-of-the-held-out-texts-and-a-third-of-the-recorded-ones-have-a-numeric-twin-in-training));
- per-mode modal records: each chosen mode of a charm recorded through its own `option` line
  ([charms](#gen-2-encodes-and-records-each-mode-of-a-charm-as-its-own-ability));
- the two new envelope fields, the random-seat flag and the what-if legality flag. Each is
  collection metadata, like `mode`, `fork` and `synthetic`, and never reaches the model. Each is
  additive, so the frozen-schema contract test gains one line per field, and both are in the schema
  before the first record;
- the random seat's share of matches ([the random seat](#gen-2-collects-one-match-in-four-with-a-seat-that-takes-a-random-legal-action-at-half-its-decisions)), and its probability of acting
  at random as the pilot leaves it;
- the rule that places a game in the game-disjoint stratum, which depends only on the game itself
  ([rare-keyword combats](#two-build-settings-leave-gate-2-and-the-trainer-short-of-rare-keyword-combats)).

A pilot of a few hundred games runs before the full collection, and the run plan says what it
checks.

### Gen-2 trains no baselines, because the knowledge probes measure what the baselines measured

The four baselines cost about two days of GPU time against the gen-1 corpus, and they answer
questions about the design rather than about one model. Gen-1 settled three of them. Reading the
text lowers the card-disjoint loss by about a third from the board-alone floor. The encoder's
margins over memorization clear every gate-1 threshold, the narrowest by nearly a factor of two.
Knowing an ability's API type and parameter keys recovers about two fifths of the encoder's gain.
None of gen-2's changes is expected to reverse these findings.

Each baseline's measurement has a stand-in that costs no training run and is computed on every sweep
arm.

| Baseline | What it measured | Gen-2 stand-in |
|---|---|---|
| `identity` | whether the encoder reads text rather than recalling it | the memorization gap, per rule family |
| `taxonomy` | what reading a script's parameters adds over its effect category | probes from `e` to the API type and the parsed parameters, beside the same probes from the parsed features themselves |
| `state-only` | what the board alone predicts | the probes record's board-only probe, which reads the board features with every `e` zeroed |
| `no-state` | whether conditioning on the board yields a better `e` | the state-dependence probe, which asks whether `e` predicts how much a text's outcome varies across boards |

The `no-state` question was never answered. Gen-1's `no-state` run zeroed the acting ability's
vector along with the board, so no gradient reached its encoder. The state-dependence probe is the
first measurement of the claim the design rests on: that training `e` against the board yields a
better `e`.

Without an identity checkpoint, `evaluate-effect-model` reports gate 1 as skipped and runs gates 2
and 3 and the checks it reports without a threshold, such as the ward canary. It already behaves that way when no `--variant-checkpoint identity`
is given.

### Gen-2 is a sweep over the width of `e`, compared by hand

Gen-2's training time goes to a sweep rather than to baselines. The axis is the width of `e`. The
encoder's depth and width were planned as a second axis where time allowed, and were not run: every
run uses an encoder of four layers 256 wide. Arms are compared by hand after
the runs, and no selection rule is fixed in advance.

Three things keep the arms comparable.

- **One corpus and one split.** Every arm trains on the same `build-corpus` output, so a difference
  between arms is a difference between models.
- **The same settings outside the swept axis.** The noise ratio and its ramp, the auxiliary head's
  weight, the withheld keyword and the curriculum are the same in every arm. The curriculum is the
  schedule on which the sparse output fields join the loss. The running-average covariance keeps the
  noise level independent of the batch size an arm needs to fit 8 GB.
- **Measures that do not grow with the width of `e`.** A probe that reads a wider vector has more
  inputs and scores higher for that alone, so every probe on `e` is reported against a control of
  the same width, a fixed random vector per text
  ([the probes record](2026-09-19-effect-knowledge-probes-design.md#a-random-vector-of-es-width-separates-what-e-holds-from-what-its-width-buys)). Concentration is compared by
  participation ratio, the number of equally weighted directions that would give the same spread.
  The top component's share is not comparable across widths.

A wider `e` is also a cheaper per-text key for the head to memorize, so each arm's memorization gap
is read beside every gain it shows. The comparison draws on:

- each arm's knowledge-probe scorecard;
- the card-disjoint results per rule family, and the memorization gap;
- gates 2 and 3, the ward canary and the zero-shot keyword check.

Strategic value, such as a card's win rate, is not compared: it is the game agent's to learn.

### The run plan pilots the collection before the full run, and does the gen-1 work while collection runs

The plan has six stages. Stage 0 reads only gen-1's artifacts and can overlap stages 1 to 3. Every
later stage needs the one before it.

#### Stage 0 runs the probes' first pass and the noise pilot on gen-1's corpus

Both runs read the gen-1 corpus, whose manifest keys texts on the root line alone. They therefore
run against gen-1's sidecars: before `convert` is re-run in stage 1, or against a copy of
`output/cardsfolder/` and `output/tokenscripts/` kept aside for them.

1. **The knowledge probes on the gen-1 checkpoint**
   (`models/effects/runs/2026-09-17-full-textless-corpus/latest.pt`), following the probes record's
   run plan. This is the harness's first run, and it finishes before any gen-2 arm needs it.
2. **The noise pilot.** Three short runs at ratios 0.05, 0.1 and 0.2, with the ramp and the
   auxiliary head on. The auxiliary head reads root-line labels only, since gen-1's sidecars carry
   no chain. The chosen ratio is fixed for every sweep arm.

#### Stage 1 fixes the encoded text and the holdout before any game is played

1. `python -m effects extract-keyword-definitions`, recording Forge's formatted value strings per
   keyword.
2. `python -m price_predictor convert`, writing sidecars whose `script_text` is the whole chain
   with its labels renumbered, and whose `option` lines carry their modes.
3. `python -m effects build-vocab --surface script`. The build log's unknown-token rate on the
   parameters is the check on the camel-case split, and the build asserts that no definition expands
   to `[UNK]`.
4. `python -m effects holdout-cards` with the template key. The share of cards it depletes sets the
   holdout fraction before anything is collected.
5. `python -m sealed generate-pools --exclude-cards` with that list.

#### Stage 2, a pilot collection, checks the new fields and the minority outcomes

1. A few hundred games of `python -m sealed match-outcomes --effect-records`, with both new envelope
   fields and the pilot's random seat: one match in eight, acting at random at a quarter of its
   decisions. The pilot raised both to the values the off-policy section gives.
2. `python -m effects validate-corpus` on the first minutes, and `python -m effects field-coverage`
   on the pilot, which shows whether both new fields vary.
3. `python -m effects build-corpus` over the pilot shards into a scratch directory. Its manifest
   answers three questions:
   - Does each family hold enough records of the minority outcomes to balance on?
   - Does the legality class stay near its gen-1 size under the board-keyed de-duplication, or grow
     back toward four fifths of the corpus?
   - Do the on-policy and off-policy counts and the real and what-if counts look as intended?
4. The random seat's probability is raised only if the minority outcomes are too few.

#### Stage 3 collects for training and for the card-disjoint stratum

1. Depleted sealed self-play with the random seat, for training.
2. Full-strength collection with the random seat, for the card-disjoint stratum.
3. `python -m effects collect-coverage --training-corpus` over the cards self-play never reaches.
4. The coverage round restricted to the held-out cards.
5. `python -m effects collect-variants`.

Each run gets `validate-corpus` over its first minutes.

#### Stage 4 builds the one corpus every arm reads

1. `python -m effects build-corpus` with rule-family balancing, outcome balancing and the
   game-disjoint stratum drawn toward rare-keyword combats.
2. The manifest's shortfall list goes back to `collect-coverage` and `collect-variants`, and its
   list of held-out texts under the floor goes back to the coverage round. The corpus is then
   rebuilt. The game-disjoint stratum keeps its games across the rebuild.
3. The final build is the one every sweep arm reads.

#### Stage 5 trains each arm and runs the gates and both probe suites on it

For each arm, in order:

1. `python -m effects train-effect-model --corpus <build> --withhold-keyword <keyword>`, at the
   arm's width of `e`. The epoch line's shares per family and per rarity bucket
   show whether the tail is reached.
2. `python -m effects encode-abilities`, with known keywords left as tokens.
3. `python -m effects evaluate-effect-model`: gates 2 and 3, the ward canary, the nearest
   neighbours and the zero-shot keyword check. The role-polarity check and the matched
   real-versus-fork check run if the corpus holds mana and fork records. Gate 1 reports as skipped.
4. The gen-1 embedding probes in `scripts/effect_embedding_probes/`, against the arm's cache:
   `build_texts.py` first, then `effect_profiles.py`, `pca_directions.py`, `lexical_probes.py`,
   `linear_probes.py`, `type_merging.py`, `neighbours.py` and `keyword_expansion.py`. Their
   `taxonomy` columns have no gen-2 counterpart, and the parsed-script features stand in for them.
   `pca_directions.py` adds the participation ratio.
5. The knowledge probes, following the probes record's run plan.

The arms are then compared by hand, and the Results section records the comparison. The
participation ratio, the embedding probes per checkpoint and the zero-shot measurement were built
for this stage, and the embedding probes ran on every gen-2 checkpoint.

#### Stage 6 compares the arms against the noise between runs

Each arm is a single run with one seed, so two arms can differ by chance alone. The plan read a
difference between arms against the swing between epochs, then by agreement across measures, and
only then by a paired comparison or a second seed. [Reading the results](#each-run-is-one-seed-so-a-difference-between-runs-counts-only-where-it-exceeds-the-swing-between-epochs)
records how the comparison was made and which of these checks ran.

## Details

### The runs and the checkpoints

Every gen-2 run trains on the stage-4 corpus (digest `b3adb69a2950`) with an encoder of four layers 256 wide, a noise ratio of 0.1, cascade withheld, seed 1255389894, 32 records per batch and 5,000 batches per epoch. An arm stops after five epochs without a new lowest card-disjoint loss and keeps its lowest epoch. A continuation run trains a fixed twenty epochs, keeps every epoch's checkpoint, and is probed at both its kept epoch and its last. An A run resumes its arm from the arm's last epoch, not its kept one, so run A64 starts at epoch 34 after `e64-lethality`'s last epoch, 33, though the arm keeps epoch 28. A B run starts from its A run's kept epoch. [The run-lineage chart](#a-checkpoint-is-named-after-its-run-an-arm-then-an-a-run-at-the-full-learning-rate-then-a-b-run-at-a-tenth-of-it) draws the same table. Every directory below is under `models/effects/runs/`, and each holds its logs, each opening with its command line and commit, and its probe reports; a continuation's last-epoch probes are in the same directory name with `-last` appended.

| run | what it is | width of `e` | lethality copies | learning rate | epochs | kept epoch | directory |
|---|---|---:|---:|---:|---|---:|---|
| `e64` (earlier inputs) | the first arm, on the inputs before the corrections below | 64 | — | full | 1–20 | 15 | `2026-10-gen2-e64-earlier-inputs/` |
| `e64` | the first arm again, on the corrected inputs | 64 | — | full | 1–30 | 25 | `2026-10-gen2-e64/` |
| `e128` | `e64` at twice the width | 128 | — | full | 1–27 | 22 | `2026-10-gen2-e128/` |
| `e64-lethality` | `e64` with lethality copies | 64 | 8 | full | 1–33 | 28 | `2026-10-gen2-e64-lethality/` |
| `e128-lethality` | `e128` with lethality copies | 128 | 8 | full | 1–27 | 22 | `2026-10-gen2-e128-lethality/` |
| run A64 | `e64-lethality` continued | 64 | 8 | full | 34–53 | 42 | `2026-10-gen2-e64-lethality-ext20/` |
| run B64 | run A64's epoch 42 continued | 64 | 8 | a tenth | 43–62 | 43 | `2026-10-gen2-e64-lethality-lr0.1/` |
| run A128 | `e128-lethality` continued | 128 | 8 | full | 28–47 | 30 | `2026-10-gen2-e128-lethality-ext20/` |
| run B128 | run A128's epoch 30 continued | 128 | 8 | a tenth | 31–50 | 37 | `2026-10-gen2-e128-lethality-lr0.1/` |
| `e256-lethality` | `e128-lethality` at width 256 | 256 | 8 | full | 1–33 | 28 | `2026-10-gen2-e256-lethality/` |
| run A256 | `e256-lethality` continued | 256 | 8 | full | 34–53 | 53 | `2026-10-gen2-e256-lethality-ext20/` |
| run B256 | run A256's epoch 53 continued | 256 | 8 | a tenth | 54–73 | 53, run A256's | `2026-10-gen2-e256-lethality-lr0.1/` |

Each run reads the same shards in the same order as the run it continues would have, since every draw is seeded by the step's place in the run. Run B256's epoch 55, `B256 best`, is probed in the same directory name with `-best` appended. Data: `runs.csv`, `checkpoints.csv`.

### Fixes the runs forced, in the order they surfaced

#### The noise pilots set the noise ratio at 0.1, from the caches rather than the losses (2026-10-08)

Three three-epoch pilots on gen-1's corpus, at noise ratios of 0.05, 0.1 and 0.2 and with the same seed, could not be told apart on validation loss: the three ratios differ by about 2%. Their caches could. At 0.1 the top principal component takes the smallest share of the variance and the participation ratio is the highest. At 0.2 a single direction carries more of the variance than gate 3 allows, though the mean pairwise cosine is lowest. One reading is that under heavy noise the encoder moves its signal onto one high-variance axis the noise cannot swamp; the pilots do not test it. The value head, which reads `e` alone, gets worse as the noise grows. Data: `noise_pilot.csv`, `gate3.csv`.

#### The first pilot collection showed every gen-2 mechanism working, and two problems (2026-10-09)

The pilot played 833 games and wrote 321,792 records, and `validate-corpus` held all twenty of its judged invariants. Gen-2 holds a resolution's record through the state-based check that follows it, so the record carries the deaths its own damage causes. Gen-1 wrote none of these: none of 565 lethal-looking resolution damage events in a sample of gen-1's shards carried its death.

- **A fork split a creature spell by its own enters trigger's modes.** A fork that forces a creature spell also resolves the creature's modal enters trigger, and it counted the trigger's modes as the spell's. A fork now counts a mode only when its charm resolves to the forced ability's own provenance key.
- **Block decisions against random attacks were too few to replace the AI's what-if queries.** The random seat, a player that takes a random legal action at a share of its decisions, attacked at random at only a quarter of its decisions. Collection therefore stops recording what-if queries, and the random seat plays one match in four and acts at random at half its decisions. In the second pilot, real AI blocks against random attacks grew about twentyfold, and games with a random seat kept the length of games between two Forge AIs.

Data: `pilot_collections.csv`.

#### The second pilot ran every gen-2 command on gen-2 records, and five defects surfaced (2026-10-09)

The second pilot wrote 2,698 games and 812,800 records in 30 minutes on 12 workers, and then went through curation, training, encoding, evaluation and both probe suites. Its 300-step training run tested that the commands run, not what the model learns.

- **Pumps were recorded as instant, in gen-1 and gen-2 alike.** The collector wrote `instant` for any event whose script declared no `Duration$`, though Forge ends a pump at end of turn unless the script says otherwise. An undeclared duration is now the one Forge applies.
- **A charm line carrying several roots keyed its modes under the first one alone.** Two triggers sharing one charm, such as Appa's enters and attacks triggers, render as one line carrying both keys. Each mode line now carries the mode under every key of its charm line.
- **One runaway game produced magnitudes in the millions.** A Bristly Bill game doubled every creature's +1/+1 counters until one resolution added 2,098,760 power, and its shard trained at a loss ten thousand times its neighbours'. Every count and signed amount is now scored at no more than 40, in the losses, the constant-predictor floor and the evaluation, and the records are left unchanged.
- **Legality families were combinations of rules rather than rules.** A board's restricting static modes were joined into one family name, so every stacked combination founded a family with an equal share of the budget. A legality record now belongs to the rarest single mode on its board, and the pilot's 45 families became 28 single rules.
- **Two evaluation tools read the wrong input.** The decodability battery matched no card because of letter case, and `scorer-smoke-test` spliced in the shipping cache whatever checkpoint it was given. Both are fixed.

#### The encoder pads each group of 32 texts to its own longest, so every arm trains at gen-1's batch size (2026-10-13)

The first arm ran out of the 8 GB of GPU memory early in its first epoch. A training step encodes every ability text on its records' boards, a hundred to three hundred texts, and gen-2 encodes an ability's whole chain, which can reach the encoder's limit of 512 tokens. Padded to the step's longest text, every short line cost as many positions as the long one. The encoder now sorts a step's texts by length and encodes them in groups of 32, each padded to its own longest. Padding is masked out of attention, so each text encodes exactly as before.

| | padded to the step's longest | grouped by length |
|---|---:|---:|
| records per batch | 16 | 32 |
| share of the encoder's input that is padding | 79% | 22% |
| highest GPU memory in one step | 6.4 GB, out of memory | 2.9 GB |

#### The first arm exposed three faults in the model's inputs, and the arm trained again on corrected ones (2026-10-14)

The corrections are computed from fields the records already hold, so the corpus was neither collected nor built again. The first run's directory keeps the record of the uncorrected inputs.

- **Three input vocabularies missed the names the collector writes.** The model one-hot encodes a record's phase, a permanent's counters and its keywords against fixed lists, with a final slot for anything else. Five of the thirteen phases, the +1/+1 and -1/-1 counters and two keywords were spelled differently in the records, so 35% of records fell into the "other" phase and 58% of counters into the "other" counter type. The inputs now map each collector name onto its slot.
- **The damage an ability deals reached `e`, but not in a form the trunk can compare with a toughness.** Two changes give both sides of the comparison one form. The board's amounts gain thermometer codes, which write the value n as n ones followed by zeros, so a single dot product with a one-hot of t tests "at least t". The value head learns each stated amount as forty "at least k" targets, which pushes `e` to carry the amount the same way.
- **The first-strike damage step's outcomes were never collected.** The collector recognized combat damage by the phase name containing `COMBAT_DAMAGE`, which Forge's `COMBAT_FIRST_STRIKE_DAMAGE` does not. In that step only damage to players was recorded. `scripts/repair_first_strike_steps.py` repairs the built corpus from each record's own damage events. It drops every blocked first-strike fight, whose creature damage and deaths are gone. The collector now tests both names. Data: `first_strike_repair.csv`.

Three evaluation checks were corrected at the same time:

- Gate 2 scores first strike and double strike on the first-strike step, where the keyword makes its difference.
- Probe amount labels are capped at 40, so Jumbo Cactuar's +9,999 power no longer decides a probe's R².
- The affordability sweep sets the acting player's lands to match the mana it gives them.

A power cut stopped the retrained arm at epoch 9. Training now rewrites a resume file at the end of every epoch, and `--resume` continues from it bit for bit.

#### The combat grid has to read one damage step at a time (2026-10-19)

The combat grid's first version granted first strike and double strike on regular-step boards and scored them against a rule spanning both steps. A record covers one step, and the model predicts that step alone, so the grid now sets the step marker to match the keyword and reads each step against its own rule. Only regular-step records qualify as the grid's base, which left the same 40 blocks qualifying. The grid also clears the damage a combat snapshot already shows, which the combat results describe.

#### The combat grid's edits have to move the damage assignment with them (2026-10-21)

The combat grid now sets both creatures' damage assignment from the edited board. A creature that strikes in the step assigns its power to the creature it fights, and one that does not strike assigns nothing. Before, the grid kept the base record's `assignment_choices`, which the input reads as each creature's number of targets and total damage assigned. Every cell off the base record's own power, and every first-strike-step cell, therefore asked about a board whose assignment disagreed with its power or its step.

#### Every arm deleted the withheld keyword instead of expanding it, so the zero-shot check measured nothing

Every gen-2 arm read cascade, the keyword withheld from training, as an empty line. `--withhold-keyword` is meant to keep the keyword's token out of training and replace each occurrence with its reminder text. The batcher removed the token before keyword expansion ran, and the expansion it should have forced never happened. A cascade line therefore reached the encoder as its `[CLS]` token alone, in training and in evaluation. The zero-shot slice compares the effect head's per-field results on records where cascade acts or sits on the board with the same fields for keywords that trained. With cascade's line empty, it compared records missing an ability with records whose abilities trained, which says nothing about reading a keyword through its definition. Gen-3 force-expands the withheld keyword wherever it appears ([gen-3 draft](2026-XX-XX-effect-model-gen3-design.md#the-withheld-keyword-reaches-training-as-its-definition-so-gen-3s-zero-shot-slice-is-the-first-that-tests-reading-one)).

### Nine collector defects shaped gen-2's labels, and gen-3 collects without them

In nine places, found in the Forge hooks and the connector's collectors after the gen-2 runs, the corpus records something other than what happened in the game. Each one reaches a field the model trains on, and the first reaches two probes. A last entry, two playability verdict bits that carry the AI's answer, was known from the design and shapes labels the same way. The gen-2 corpus is not collected again. Gen-3 collects from scratch with all nine fixed ([gen-3 design](2026-XX-XX-effect-model-gen3-design.md#the-forge-hooks-and-collectors-recorded-planned-events-as-real-ones-and-several-outcomes-reached-the-wrong-record)).

#### Most block-legality labels that say a creature may not block describe the AI's unfinished block plan

More than half of the gen-2 corpus's "may not block" labels name a creature the AI had already planned to block with. The `blockers` hook fires inside Forge's AI blocking code, once for every attacker the AI considers while it assigns blocks. Forge tests whether a creature may block against the combat the AI is filling in, and that combat already holds the blocks the AI has planned so far. A creature planned to block one attacker therefore reads as forbidden for every attacker considered after it, and an attacker's own planned blockers read as forbidden when the AI reinforces that block. The collector counts each of these queries as a real decision, because it is asked during the declare-blockers step about an attacking creature. Most attackers therefore anchor more than one real record in a turn, each on a board showing a different stage of the plan. The AI drops creatures that may not block at all, such as tapped ones, before it assigns blocks, so true refusals are a small part of the forbidden list. The `attackers` records are not affected: none was taken after attackers were declared.

| game-disjoint stratum, real decisions | count |
|---|---:|
| `blockers` records | 80,306 |
| … whose snapshot shows blocks the AI had already planned | 47,053 |
| forbidden blockers listed | 147,584 |
| … naming no restricting static | 98,946 |
| … naming a creature the snapshot shows blocking | 83,141 |
| … naming a tapped creature | 10,952 |
| attackers anchoring a record, per game and turn | 35,093 |
| … anchoring more than one real record | 18,793 |

The `blocker_legal` field therefore learns that a creature already assigned to a block may not block, which no rule says. Two probes read these labels. The evasion grid draws its base records from them, and its predictions sit low even where every block is legal. The probe ladder's `may_block` target is read from them directly. The menace case in the hook, a "can't block alone" creature dropped from a double block, is a smaller part of the same defect. Data: `legality_planned_blocks.csv`.

#### A reveal from a library reaches no card and no player

No reveal from a library sets the `reveal` bit of the `library_events` field in the gen-2 corpus. Forge reports a reveal at one engine choke point that knows no acting ability, and the note named no subject. The model's targets fold an event only into the entities it names, so the event reached no target. Clash reveals its cards through a path of its own, and its note named the clash spell's own card rather than a player. The `reveal` bit is therefore trained on clash spells alone, and on the wrong kind of entity. The embedding probes count `card_revealed` events by presence, which this does not change. Their counts of which side an event happened to will differ between gen-2 and gen-3, because gen-3's reveals name a player.

#### Created tokens carry no keywords, and an effect making several kinds of token records them as the first kind

Every keyword flag of the created-objects field is 0 in the gen-2 corpus, so a 1/1 flying Spirit is taught as having no keywords. A token's note described the token by power, toughness and types, and gave no keywords. The note also let the first token stand for the whole batch. Bestial Menace, which makes a Snake, a Wolf and an Elephant, is recorded as three Snakes, and a token copy of several different creatures as copies of the first one. Their counts, power, toughness and types in the created-objects field are wrong. The note's token id was the token's printed name where the board snapshot uses its script name. This changes only the order of a record's created-object slots.

#### Prevented damage names nobody it was prevented for

The affected gate never learns that a prevention shield touches what it protects. A prevention shield's note named the damage's source and not the creature or player it protected, so the protected entity is never marked affected by the prevention.

#### An ability's own card is marked affected by the coins, tokens and copies it produces

The affected gate learns that an ability changes its own card whenever it makes something. When a note named no subject, the collector named the acting ability's card instead. Token creation, made cards, copies, coin flips, clashes, votes, ventures and spell copies all name no subject. Each one therefore marks the card whose ability acted as affected, with no change recorded on it. For a creature whose triggered or activated ability makes tokens, this marking is the only reason the record shows the creature affected.

#### A double striker's second hit is recorded with no damage assigned

Nearly every double striker that hits in both damage steps is recorded at the regular step with no damage assigned, while the record's events show it dealing damage. Each card slot of a combat record's input carries how many targets the creature assigns combat damage to and how much it assigns in total, read from the record's `assignment_choices`. The collector believed Forge kept one assignment table for a whole combat, so at the regular step it left out every assignment the first-strike step had already written. Forge in fact clears the table after each damage step. What the collector left out was therefore a double striker's second assignment whenever it repeated the first, which for an unblocked double striker is every time.

| full-strength raw shards | count |
|---|---:|
| regular-step damage sources that also dealt first-strike damage | 1,188 |
| … with no row in the record's `assignment_choices` | 1,167 |
| regular-step damage sources with double strike in the snapshot | 1,552 |
| … with no row in the record's `assignment_choices` | 1,181 |

The observed record also read the table at the step's first damage or prevention event. By then Forge's damage replacements had begun rewriting the table in place, so a prevented hit could show as less damage assigned, or none, and the input carried part of the outcome. The probe forks read the table before any replacement and left nothing out. The training corpus therefore teaches two conventions at once. A fork shows a double striker assigning damage in both steps, and an observed record shows it assigning nothing in the second. The combat grid's double-strike second-hit variant reads a board with the assignment intact, a combination real double strikers never showed in training. Gate 2 builds both of its predictions from observed records, so the forks' different reading does not reach it. Data: `hook_defects_in_corpus.csv`.

#### Events from abilities that resolve after combat damage are written into the combat record

Combat records carry the draws, tokens, discards and mills of abilities that resolved after the damage. A damage step has a priority window after its damage is dealt, and combat-damage triggers resolve in it: "whenever this creature deals combat damage to a player, draw a card", the monarch's transfer, a saboteur's discard. The collector filed every event of a damage step into the step's combat record. Those triggers' own resolution records therefore miss their effects, and a creature a spell damaged in the window is recorded as taking combat damage. Combat damage draws no card and makes no token, yet the combat records hold nearly three thousand drawn cards and nearly two thousand made tokens.

| full-strength raw shards | count |
|---|---:|
| observed combat records | 192,895 |
| … holding an event attributed to a resolving ability | 2,601 |
| `card_drawn` events in observed combat records | 2,718 |
| `token_created` events in observed combat records | 1,904 |
| `card_discarded` events in observed combat records | 1,029 |
| `card_milled` events in observed combat records | 1,003 |
| `monarch_changed` events in observed combat records | 562 |

The model therefore learns from a combat record that combat itself sometimes draws cards or makes tokens, and from the trigger's resolution record that the trigger does less than its text says. The combat grid reads only deaths, and a block qualifies only when its deaths obey the damage rule, so a trigger's extra death keeps a record out of the grid rather than changing what it reads.

#### "Tapped for mana" trigger records are partly the AI's questions about taps it was planning

Some trigger records for triggers that fire when a land is tapped for mana describe a tap that never happened. Forge's AI, while it plans how to pay a cost, asks the engine which of these triggers a tap of each land would fire, so that it can count the extra mana. The trigger hook sat in the engine check that answers the question, so every answer became a trigger record. The fired label is true of the planned tap, and the board is the real one, but the record presents the tap as an event. A fifth of these records repeat another's game, turn, step, land and verdict, which a real tap does only when the land untaps and taps again in the same step. The planned and the real taps cannot be told apart from the records. The cards affected are a small set, such as Wild Growth, Utopia Sprawl, Overgrowth and Mana Flare, and the planning queries also spend those lines' per-game budget of trigger records.

| full-strength raw shards | count |
|---|---:|
| `TapsForMana` trigger records | 4,911 |
| … fired | 2,700 |
| … repeating another's game, turn, step, land and verdict | 996 |

#### A prevention shield with nothing left to prevent records a prevention of 0

One `damage_prevented` event in fifteen prevented nothing. A shield that prevents a fixed amount, once that amount is used up, still answers Forge that it applied, with the damage unchanged. The hook wrote the difference, 0, as a prevention. The `damage_prevented` amount is therefore trained on zeros that stand for no event, and a probe counting prevention events by presence counts these too. In gen-2 the note named no recipient, so the affected gate is not touched by it.

| full-strength raw shards | count |
|---|---:|
| `damage_prevented` events | 4,030 |
| … of amount 0 | 262 |

#### Two of the three playability verdict bits are partly the AI's answer rather than the rules'

In the gen-2 corpus, `has_legal_target` and `affordable` record in part what Forge's AI chose rather than what the rules allow. On a Forge seat's candidate, `has_legal_target` records whether the AI had set targets on it when the hook fired. The AI sets targets only when its evaluation gets that far and finds a target it wants. A candidate it put off until later therefore reads as having no legal target, and one evaluated earlier can keep that evaluation's targets. On both seats, `affordable` comes from Forge's AI cost check. That check refuses some costs the rules allow: a planeswalker's last loyalty, a Ward cost the AI dislikes and a Casualty sacrifice. It also adds the mana that taxes such as Nether Void's would demand. The random seat walks a charm's leftover chain of chosen modes rather than the modes the charm offers, so a charm with no choosable mode reads as having a legal target. The verdict head's `has_legal_target` column therefore learns in part when the AI chooses to target, and its `affordable` column learns some of the AI's cost preferences. The random seat filters its own plays through the same cost check, so it is kept from the costs that check refuses. Gen-3 computes all three bits from the rules alone ([gen-3 design](2026-XX-XX-effect-model-gen3-design.md#the-model-has-to-learn-every-rule-that-depends-on-whose-turn-or-which-step-it-is-and-collection-records-which-check-decided-each-one)).

### Fifteen defects in the corpus build, the training targets, the evaluator and the probes shaped gen-2's results, and gen-3 is built without them

In fifteen places, found in `src/effects` and the probe scripts after the gen-2 runs, the corpus, the model's targets or a reported figure differs from what the design says. Each defect was confirmed against the code and, where it leaves a trace, counted on the gen-2 corpus and the arms' logs. Four of them change what every gen-2 arm learned: zero-filled targets, triggers supervising their cause, the value head's misread amounts, and the corpus build's short training set. The rest change a probe, a gate's population or a single cache row. None of them flips a gate verdict. Gen-3 trains and is scored without any of them ([gen-3 design](2026-XX-XX-effect-model-gen3-design.md#the-corpus-build-the-targets-and-the-probes-were-reviewed-and-gen-3-trains-without-the-defects-found)).

#### Every affected entity was scored on every output field, and a field with no event was taught as 0

Gen-2 trained the model to predict 0 for any output field an affected entity had no event for. The effect head's fields are conditional: each is meant to train only on entities its record actually says something about. The loss instead scored every field on every entity whose affected gate fires, and filled the fields the record left unset with 0. The constant-prediction floor, against which every field's explained deviance is reported, was computed on the same filled targets, so the deviance numbers never showed it.

The zeros mean something false for five groups of fields:

- `tap_state` wrote 0 for an untap event and 0 for no tap event at all, so an untap effect could not be told from nothing happening.
- `pt_duration` and `type_color_duration` read an unset duration as class 0, `instant`. Every anthem's continuous boost and every continuous record trained as instant.
- `target_legal` is set only on a decision's legal targets, always to 1. Its only negatives were the zeros filled in on entities of other record kinds, so it learned which records are decisions, not which targets are legal.
- `attacker_legal`, `blocker_legal` and `min_blockers` were filled with 0 on every resolution, combat and continuous entity.
- Player slots trained zone and tap fields at 0, and permanents trained life and mana fields at 0.

Fields whose 0 does mean "nothing changed", such as damage, counters and keywords gained, are unaffected. Two probes read the distorted fields: the probe ladder's `pt_until_end_of_turn` target reads the `pt_duration` head, and its `may_block` target reads `blocker_legal`. The ladder reads target legality through the gate rather than the `target_legal` field, so that target is unaffected.

#### Trigger records taught the model to predict the event that fired the trigger

Every gen-2 arm learned, on trigger records, to predict the triggering event as the trigger's own effect. A trigger record's input already carries the event that fired it, and the design supervises only the verdict head's fired bit there. The target derivation instead read the triggering event as the record's outcome. A "whenever a creature dies, draw a card" trigger was therefore trained to predict that a creature died. A trigger that did not fire carries no event, so on trigger records the affected gate learned that an entity is affected exactly when the trigger fired. Gate F1 and explained deviance on trigger records include these targets.

#### The value head never saw a static power boost and counted removed counters as placed

The value head, the training-only head that reads from `e` the amounts a script states, was taught that every anthem, aura and equipment grants no power or toughness. It looked for the parameters `PowerBonus` and `ToughnessBonus`, which no Forge script uses. Static boosts are written `AddPower$` and `AddToughness$`. A line naming neither is read as stating a change of 0, which is trained rather than masked. Separately, the counters target summed `CounterNum` for every API, so a script removing or moving counters was taught as placing them.

| converted lines | count |
|---|---:|
| lines with `AddPower$`, trained as stating a power change of 0 | 1,928 |
| lines with `AddToughness$`, trained as stating a toughness change of 0 | 1,521 |
| `RemoveCounter` lines trained as placing counters | 171 |

The value head's weight is 0.05, so the push on `e` is small, but it falls systematically on one large family of cards. The probe ladder's value-vector rung reads the same targets.

#### The corpus build planned the training set over records it then sent to validation, and delivered 82% of it

The gen-2 training set holds a fifth fewer records than its plan, and its class mix is not the one asked for. `build-corpus` counts the records available for each class, family and text before it decides which games go to validation. It then sizes every quota on those counts. The write pass sends held-out, card-disjoint and game-disjoint games away, and every quota slot their records held goes unwritten. No step reports the gap, because the manifest's `shortfall` measures only the `--training-records` ceiling, which gen-2 left unset.

| class | share of planned records written |
|---|---:|
| all classes, 892,205 of 1,089,699 | 82% |
| `playability-decision`, 67,527 of 109,028 | 62% |
| `combat` | 78% |
| `continuous` | 84% |
| other classes | 85–87% |

Decisions lost most because the held-out coverage games, which all go to validation, are rich in decisions. They make up 7.6% of the written mix against 10% asked, and combat 19.0% against 20%. Each text's copies fall about 15% below their quota, and more for texts concentrated in the full-strength runs, where a third of the games were held out. The per-family `available` and `shortfall` columns of the manifest count the same records and are inflated by them.

#### Each training shard came from a single collection run, so every batch did

Every gen-2 batch was drawn from the few hundred games of one collection run. `build-corpus` repacked the training stratum into 2,000-record shards by concatenating its inputs in order. Each shard therefore holds three to six consecutive worker lifetimes of one run and one source, with a record's copies side by side. The trainer draws whole shards at random but shuffles only within the shard it holds, so a shard's correlation reaches every batch drawn from it, and two copies of one record can share a batch.

#### The masked-language-model term could read a masked number off its own input

The masked-language-model term, which trains the encoder to restore hidden tokens, taught nothing about reading an amount from its context. Masking replaced a digit's token, but the encoder's separate numeric input still carried the hidden value. Restoring a masked `3` in `NumDmg$ 3` therefore meant copying it. The term's weight is 0.1, so the loss of signal is small.

#### Cascade's line in every gen-2 cache was encoded from a token training never updated

The cached `e` of the withheld keyword is close to an arbitrary vector in every gen-2 arm. Training and evaluation drop cascade's token, so the token's embedding never received a gradient. `encode-abilities`, which writes the cache, kept the token and encoded the Cascade line from it. Every reader of the cache sees this row: the nearest-neighbour and UMAP checks, the decodability battery and the embedding probes. The evaluator's zero-shot slice builds its inputs as training does, so this row does not reach it; [the deletion](#every-arm-deleted-the-withheld-keyword-instead-of-expanding-it-so-the-zero-shot-check-measured-nothing) is what emptied that slice.

#### Gate 2 read the effect fields without the affected gate, over a population up to a third larger than its distinct records

Gate 2's per-field agreement for first strike and double strike is biased low. The field heads train only on affected entities, so each reads "if affected, how much". Gate 2 compared those conditional values between the real board and the board with the keyword removed. When first strike is removed in the first-strike step, the defending player is no longer hit in that step at all. What moves is the affected gate, which gate 2 never read. On `B256 best` the reported agreement on `opponent.damage_taken` was 48% for first strike and 16% for double strike. Double strike passed at 71.4%, just above the 70% bar, so a gated reading would most likely have raised it.

Gate 2's counts are also inflated, in three ways. It kept the probe forks, which copy a real combat with one other keyword stripped, so their remaining carriers are counted a second time. It counted every carrier in a combat as one qualifying record. And it read the raw shards rather than the curated stratum, so it also scored the 216 first-strike records the first-strike repair dropped as unrepairable. No count crosses the 200-record threshold either way.

| keyword | observations | from probe forks | distinct records, forks excluded |
|---|---:|---:|---:|
| first strike | 2,079 | 326 | 1,645 |
| double strike | 250 | 33 | 211 |
| indestructible | 334 | 62 | 248 |
| wither | 139 | 17 | 104 |

The observations are counted on the curated game-disjoint stratum, so the first-strike figure leaves out the repair's 216 dropped records.

#### The knowledge probes filed trained texts as held-out lines

The line ladder's held-out stratum is about one-eighth trained texts, so every gen-2 scorecard overstates the held-out line scores and understates the gap to trained lines. The probe filed a text as held-out when any card carrying it is held out. A held-out card also prints ordinary lines such as Flying and Trample, which train on every other card that carries them. Of 1,363 labelled texts filed held-out, 175 are also printed on cards that trained.

#### The embedding probes counted each curated copy as a separate record

The embedding probes' per-text record counts include `build-corpus`'s copies, and so overweight rare texts. The effect profiles are read from the training stratum, where a record appears up to four times as separate lines with one `record_id`. The counts feed the `n/(n+5)` weights of the profile probes, so a text with a single record copied four times weighs 0.44 instead of 0.17. Rare texts are also the noisiest, which likely lowers the weighted R² of the profile tables.

#### The knowledge probes scored variant cards without their text

The knowledge probes and grids built their model without the variant-script tree that training read, so a variant card's abilities resolved to no text in them. Nine of the 300 game-disjoint probe games come from the variants run, and 463 of 16,698 game-disjoint record items, 2.8%, were scored on an input training never showed. The evaluator was given the tree on every arm and is unaffected.

#### The observed-mana probe target counted mana events rather than mana

The knowledge probes' observed-mana target counts every mana event as one mana. It read a field the collector never writes. A multi-mana producer is therefore labelled as making one mana, and the target loses the variance it exists to measure. On the first arm the target scored no rung, so no gen-2 conclusion rests on it.

#### The card-disjoint cap admitted held-out games that resolve no held-out text

One held-out game in eight admitted to the card-disjoint stratum gives gate 1 no record. The cap allows 50 games per held-out text, and it counted a game whenever any record acted through the text, including a decision that only considered casting it. Gate 1 scores resolutions alone. Of 4,814 admitted games, 616 held no resolution of a held-out text, and each took a slot a scorable game could have used. Gen-2 trains no identity baseline and never ran gate 1, but these games are part of the card-disjoint loss that chooses each arm's checkpoint.

#### Two held-out templates reached training through a token and a variant script

The holdout keeps every game naming a held-out card out of training, and a held-out card is found in `cardsfolder` alone. A token script and a variant script carrying a held-out template therefore trained like any other text. Sixteen training records of 889,773 act through them: eleven through the Illusion token `u_0_2_illusion_mesmerize`, and five through the variant `act_of_treason_variant_12052`, whose target swap reads as a held-out template.

#### The evaluator gave a false reason for three checks it never ran

The evaluator's reports on every gen-2 arm say that three checks need fork records the corpus does not have yet. The checks, the matched real-versus-fork comparison, the probe-diff re-check and the role-polarity probe, were never implemented. The game-disjoint stratum holds 6,818 combat forks. No number depends on them.

## Data and charts

Every file below is in [`2026-09-18-effect-model-gen2/`](2026-09-18-effect-model-gen2/README.md), whose README gives each file's columns and source in full. `scripts/effect_gen2_report/build_data.py` writes the data from the run directories. `scripts/effect_gen2_report/make_charts.py` draws the data charts from the data alone, and `make_diagrams.py` draws the four diagrams, which hold no data.

| file | contents |
|---|---|
| `data/runs.csv`, `data/checkpoints.csv` | every run, and every probed checkpoint with its validation losses |
| `data/epochs.csv` | every epoch's training and validation losses, gate F1, zone accuracy and deviance |
| `data/fields_by_epoch.csv` | every epoch's card-disjoint loss and explained deviance per output field |
| `data/damage_grids.csv` | the damage-by-toughness grids |
| `data/boundary_by_epoch.csv` | the lethal-boundary metric on 40 training records, per epoch |
| `data/combat_grids.csv` | the combat grids, per keyword variant |
| `data/gate2.csv`, `data/gate3.csv` | gate 2 per keyword, gate 3 and the participation ratio |
| `data/probe_ladder.csv`, `data/sweeps.csv`, `data/sweep_records.csv` | the knowledge probes' scores and sweeps, which the probes record discusses |
| `data/embedding_probes.csv` | the embedding probes on each cache |
| the other `data/*.csv` | the smaller tables of the fixes above |
| `charts/*.png` | every chart and diagram in this record and the probes record |
