# Ability effect model — knowledge probes

This record covers a suite of probes that asks the ability effect model what it knows about Magic, and where in the model that knowledge lives. The model learns from effect records: Forge, the open-source Magic rules engine, plays both sides of sealed games, and each record holds the board before an event, such as an ability resolving, and what the event changed. The model is described in the gen-2 record, [`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md), and its original design in [`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md).

The model has two parts. An encoder turns each Forge ability script into one vector `e`. An effect head reads the `e` vectors together with the board and predicts what the ability does to every card and player. The effect head's shared body, the trunk, is a transformer: a stack of layers in which every input slot can draw on every other. It reads one slot for the game (`[GLOBAL]`), one per player (`[PLAYER]`), an `[ACT]` slot holding the acting ability's `e`, and a block for each object in the board snapshot: a `[CARD]` slot followed by the `e` of each spell or activated line the object carries. A permanent's statics, triggers and replacements are never in a block. Small output heads read the trunk and predict each output field, such as whether an entity dies or how much damage it takes.

![Diagram of the effect model. A Forge script passes through a tokenizer and the ability encoder to a vector e of width 64, 128 or 256. In training only, noise is added to e, an MLM head reads the encoder's tokens, and value, API-type and parameter-key heads read e alone. e feeds the [ACT] slot and the ability slots of each object's block in the trunk's input, beside [GLOBAL], two [PLAYER] slots and [CARD] slots. Statics, triggers and replacements are never in a block. A six-layer trunk feeds the created-objects, verdict and per-entity heads. Numbered badges mark where read-out ladder rungs 0, 1, 1w, 1o, 2 and 3 read, and a panel shows the three ablation scopes.](2026-09-18-effect-model-gen2/charts/diagram-model-and-probes.png)

*Dashed parts exist only in training. The MLM head, a masked-language-model head, trains the encoder to restore hidden words of the script. The rose badges mark the input each rung of the read-out ladder probes, and the slot rows of the ablation panel show which slots each ablation scope replaces.*

Where knowledge lives matters because `e` is the part that gets reused. A future game-playing agent would read the precomputed `e` of each ability and may retrain or replace the trunk. Whatever the trunk works out from the board each time it predicts is lost to such an agent, which would have to learn it again from a far weaker training signal.

This record reports where the model's knowledge lives and the probes aimed at single rule families: evasion, targeting, mana colour, card movement and the sweeps. The gen-2 record judges gen-2's training changes and the rules they targeted: the death rule, the lethal boundary, the combat grid, gate 2, checkpoint selection, the losses and the width of `e`.

The suite has three kinds of measurement:

- **The read-out ladder** fits small probes to inputs taken from progressively deeper in the model, from the board alone to the trunk's vectors, and sets them beside the model's own answer. It measures how much of the model's knowledge `e` alone carries, and how much `e` adds over a random vector of its width.
- **Ablations** replace `e` in the trained model and measure how much each output field's loss rises. The loss is the prediction error the model was trained to minimise.
- **Board sweeps and edit grids** edit one input of a real record, such as a target's toughness, the lands a player has or the keyword a creature carries, and read how the model's prediction moves.

The design also planned a third method, a shallow trunk trained on frozen `e`. It never ran.

Each corpus withholds two validation strata from training, whole games at a time ([diagram](2026-09-18-effect-model-gen2/charts/diagram-validation-strata.png), [the gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#two-validation-strata-keep-whole-games-out-of-training-and-one-of-them-holds-texts-the-model-never-trained-on)). The card-disjoint stratum holds games that contain a held-out card, whose ability texts the model never trained on. The game-disjoint stratum holds games the model never saw whose ability texts it did train on. The probe items are drawn from both strata and frozen once per corpus, so every model trained on that corpus is probed on the same items.

The suite probed gen-1 and 17 gen-2 checkpoints from twelve training runs. A checkpoint is the model's saved weights at one epoch of training, a fixed block of 5,000 training batches. The first gen-2 runs are the arms: `e64` and `e128`, named after the width of `e`, and `e64-lethality`, `e128-lethality` and `e256-lethality`, which add lethality copies. A lethality copy is a training copy of a damage record with the damage and the hit creature's toughness redrawn, and the death labelled by the rule. `e64 (earlier inputs)` is the first run of arm `e64`, on inputs later corrected.

Continuation runs carry the arms with copies further. Runs A64, A128 and A256 resume their arm after its last epoch at the full learning rate, the step size of each weight update, and runs B64, B128 and B256 start from their A run's kept epoch at a tenth of it. Each run keeps the epoch with the lowest loss on card-disjoint records, and a continuation run is probed at that kept epoch (`A64 best`) and at its last (`A64 last`). Run A256's lowest epoch is its last, so it counts once. Run B256 never went below the checkpoint it started from and keeps run A256's epoch 53; `B256 best` is its own lowest epoch, 55. "The 14 checkpoints with copies" are the 17 less the three arms without copies, and "the median over 14 checkpoints" is taken over them. The [run lineage chart](2026-09-18-effect-model-gen2/charts/run-lineage.png) draws every run and checkpoint, the gen-2 record [explains the naming](2026-09-18-effect-model-gen2-improvements-design.md#a-checkpoint-is-named-after-its-run-an-arm-then-an-a-run-at-the-full-learning-rate-then-a-b-run-at-a-tenth-of-it), and its [runs table](2026-09-18-effect-model-gen2-improvements-design.md#the-runs-and-the-checkpoints) lists their directories.

A probe is scored by AUC for a yes/no question, the chance that a true case scores above a false one, so 0.5 is chance. An amount is scored by R², the share of its variance the prediction explains, where 0 is no better than predicting the amount's mean. The results come first below, in the order of the summary, each with a chart. The design of the suite follows them, and the run details come last. All the data is in [`2026-09-18-effect-model-gen2/`](2026-09-18-effect-model-gen2/README.md), shared with the gen-2 record, and every chart is drawn from it by `scripts/effect_gen2_report/make_charts.py`.

The charts share one key. Colour gives the width of `e`: blue 64, orange 128, green 256. A point's marker gives the stage, a triangle for an arm, a square for an A run and a circle for a B run; a line's style gives it too, dotted for an arm, dashed for an A run and solid for a B run. A filled marker is a run's kept epoch and a hollow one its last; `B256 best` is half filled. The arms without copies are grey triangles, `e64 (earlier inputs)` a lighter grey, and gen-1 dark grey.

## What the probes found

**`e` carries the answers a line states, and the trunk answers what the board decides.** From the board and `e` alone, a probe recovers nearly all of what the model knows about whether an ability is affordable and how long a pump lasts. A random vector of `e`'s width closes nearly half of affordability's gap by itself. On deaths, trigger firing, targets, which cards are affected and damage taken, a probe on the trunk's vector reads about as much as the model's own answer, or more on damage taken, and the board plus `e` reads far less. Even the true numbers the script states take a probe only slightly past the board alone on these questions, so the ladder cannot say whether `e` holds what they need. Whether a creature may block is known from the board alone, partly because gen-2's block labels come from the AI's block plan ([the ladder](#e-carries-the-answers-a-line-states-and-the-trunk-answers-what-the-board-decides)).

**`e` adds far more than a random vector of its width on what a line states, and almost nothing on what the board decides.** Costs, amounts, timing and the side an effect lands on gain the most. On deaths, `e` adds at most four hundredths of AUC at widths 64 and 128 and less than nothing at width 256, and on trigger firing a few hundredths at every width, so four times the width does not move either into `e` ([e over a random vector](#e-adds-far-more-than-a-random-vector-of-its-width-on-what-a-line-states-and-almost-nothing-on-what-the-board-decides)).

**Gen-2's `e` holds whether a line taps or sacrifices almost perfectly, and its timing and most of its amounts well, where gen-1's held little of these.** Costs and timing transfer to lines the model never trained on. Power and toughness changes do not ([line properties](#gen-2s-e-holds-whether-a-line-taps-or-sacrifices-almost-perfectly-and-its-timing-and-most-of-its-amounts-well)).

**`e` barely holds how much a text's outcome varies from board to board, which training against the board was meant to put there.** A probe reads at most a sixth of that spread from `e` on trained lines and nothing on held-out lines, where gen-1's `e` read a quarter. The probes find no support for the claim the model's design rests on, that training the encoder against the board gives a better `e` than training it on text alone ([state dependence](#e-barely-holds-how-much-a-texts-outcome-varies-from-board-to-board)).

**The trunk reads the acting ability's `e` and ignores the `e` of the abilities already on the board.** Replacing `e` with noise at the `[ACT]` slot alone costs as much as replacing it everywhere, and replacing it in the card blocks costs nothing measurable, on every checkpoint ([ablation](#the-trunk-reads-the-acting-abilitys-e-and-ignores-the-e-of-the-abilities-on-the-board)).

**A restriction the board snapshot does not list costs the model the most on attack legality, on every checkpoint.** The snapshot gives each permanent only its spell and activated abilities, so a Pacifism reaches the model as an enchantment with no text. Where such a static keeps a creature from attacking, the attack-legality loss is between one and a half and four and a half times its loss with no restriction ([board abilities](#a-restriction-the-board-snapshot-does-not-list-costs-the-model-the-most-on-attack-legality)).

**The model knows an evasive attacker is hard to block, but pairs it with the blocker that answers it only for horsemanship and fear.** Intimidate is read as fear, reach barely answers flying, and shadow and "can't be blocked" change nothing. The predicted chance of any block is low, because the block labels the model trained on mark many legal blocks as forbidden ([evasion](#evasion-the-model-knows-an-evasive-attacker-is-hard-to-block-and-which-blocker-answers-it-only-for-horsemanship-and-fear)).

**The model follows who controls a target, and barely checks the card type a restriction names.** A creature that loses its creature type stays a legal target of "target creature" ([targeting](#targeting-the-model-follows-who-controls-a-target-and-barely-checks-the-card-type-a-restriction-names)).

**The model counts the mana a player has and barely checks its colour.** One land fewer lowers the predicted affordable verdict four to ten times as much as one land of the wrong colour, and two extra lands of the wrong colour make an uncastable spell look more affordable than a castable one. The affordable label the model trained on carries some of Forge's AI cost policy as well as the rules ([colour](#colour-the-model-counts-the-mana-a-player-has-and-barely-checks-its-colour)).

**The model knows which player draws, discards or mills and where a moved card goes, and barely uses the number of cards the script states.** Six cards drawn are predicted as about two. Retargeting a draw, discard or mill moves only part of the predicted effect to the new player ([zones](#zones-the-model-knows-which-player-draws-discards-or-mills-and-where-a-moved-card-goes-and-barely-uses-the-number-of-cards)).

**The affordable verdict never clearly passes one half as mana is added, and a sweeper is predicted to kill about one creature in five, on every gen-2 checkpoint.** The board-size sweep adds opposing creatures one at a time, and the predicted deaths rise by about a fifth of a death per creature, where a sweeper kills one per creature ([sweeps](#the-affordable-verdict-never-clearly-passes-one-half-and-a-sweepers-predicted-deaths-rise-by-a-fifth-per-creature-added)).

## Results

### `e` carries the answers a line states, and the trunk answers what the board decides

On two questions `e` carries the answer: whether an ability is affordable, and how long a pump lasts. On five other board questions a probe on the trunk's vector reads about as much as the model's own answer or more, and the board plus `e` reads far less. On those questions the ladder cannot tell whether `e` lacks the answer or the probe cannot use it.

The ladder asks one question, such as "does this creature die?", of six inputs taken from progressively deeper in the model, and each input is a rung. A question is about one card on the board, the target, or about the acting ability itself.

- **Rung 0, the board alone.** The input features the trunk receives, with every `e` zeroed, concatenated into one vector:
  - the game: turn, phase and step, whose turn it is and who holds priority, the kind of record;
  - the acting ability's slot without its `e`: the announced X, the number of modes, targets and choices;
  - the acting player and one opponent: life, cards in hand, library and graveyard, poison, mana available by colour;
  - each player's objects in the snapshot, as the average of their card features;
  - for a question about one card, that card's own features: card types, supertypes and colours, mana value, power and toughness with their boosts and counters, marked damage, counters, zone, tapped and summoning sick, combat status, keyword flags, whose it is, whether it is targeted.
- **Rung 1, the board plus `e`.** Rung 0's vector with three `e` vectors appended. The first is the acting ability's. The second is the average `e` of the target card's own spell and activated lines; its keywords, such as indestructible, are already among rung 0's keyword flags. The third is a partner's, where the question has one: for a trigger, the line whose event fired it, where [the interaction join](#interaction-labels-come-from-joining-triggers-to-resolutions-because-no-trigger-record-names-the-ability-behind-its-event) finds one, and for a blocker, the attacker's average `e`. The vectors are only concatenated; no transformer combines them with the board.
- **Rung 1w, the width control.** Rung 1 with each `e` replaced by a fixed random vector of the same width, one per text.
- **Rung 1o, the oracle.** Rung 0 plus the numbers the acting line's script states, and its partner's: damage, power and toughness change, counters placed and removed, cards drawn, mana cost by colour, and whether the cost taps or sacrifices. It carries nothing else of the line, such as what kind of effect it is or how long it lasts, and none of the target card's own lines.
- **Rung 2, the trunk's vector.** The trained trunk's output at the target card's `[CARD]` slot, or at `[ACT]` for a question about the ability: one vector the transformer computed after attending over the whole board, every card's ability vectors included.
- **Rung 3, the model's own answer.** The trained model's output head applied to that same trunk vector, with no probe. For damage taken it is the predicted amount alone, because every item took damage.

Rungs 0 to 2 are not read by the model's output heads. Each is read by a probe, a small network fitted afresh for one question on one input. The probe is a two-layer MLP (multi-layer perceptron: layers of weighted sums with a non-linear function between them, which can read a relationship a linear model misses), with the same size and settings on every checkpoint. A linear probe is fitted beside it, and its scores are in the data. Each probe is fitted on four fifths of the items and scored on the fifth it never saw, five times over. Items about one card are split by that card, and items about the acting ability by its text, so a probe is never scored on a card or text it was fitted on. A probe's score therefore says how much of the answer its input holds in a form a small network can read. On a question about one card the acting ability's text can still fall on both sides, which [the next section](#e-adds-far-more-than-a-random-vector-of-its-width-on-what-a-line-states-and-almost-nothing-on-what-the-board-decides) returns to.

The share `e` carries is (rung 1 − rung 0) / (rung 3 − rung 0): the fraction of the distance from the board alone to the model's own answer that `e` closes without the trunk. A low share has two possible causes. Either `e` lacks the answer, or a probe cannot combine `e` with the board, as when it must compare a damage amount with a toughness. The oracle rung separates the two: it hands the probe the true numbers in place of `e`. A share below nine tenths is interpreted only where the oracle rung closes at least half the gap. Below that, even the true numbers leave the probe far from the model's answer, and the probe is the limit.

| question, card-disjoint records | item | share carried by `e`, median over 14 checkpoints | range | share a random vector closes, median | share the oracle closes, median | share interpreted |
|---|---|---:|---|---:|---:|---|
| how long a pump lasts | a permanent whose power or toughness a resolution changed: does the change end at end of turn | 1.01 | 0.90–1.06 | 0.10 | 0.24 | yes, `e` closes the gap |
| affordable | a candidate in a playability decision: can its controller pay for it | 0.93 | 0.85–0.97 | 0.45 | 0.85 | yes |
| damage taken | a creature a resolution damaged: how much damage, scored by R² | 0.56 | 0.46–0.75 | 0.13 | 0.15 | no |
| legal target | a battlefield permanent in a targeting decision: is it a legal target | 0.51 | 0.19–0.58 | −0.63 | 0.03 | no |
| trigger fires | an evaluated trigger: does it fire | 0.30 | 0.22–0.36 | 0.10 | 0.15 | no |
| legal target, protected | the same, for a permanent with hexproof, shroud, protection or ward | 0.28 | 0.17–0.41 | −0.22 | −0.15 | no |
| card affected | a battlefield permanent in a resolution: does the resolution change it at all | 0.23 | 0.16–0.37 | −0.12 | 0.05 | no |
| dies | a creature a resolution damaged: does it die | 0.20 | 0.04–0.30 | 0.16 | 0.04 | no |
| may block | a creature in a block-legality record: may it block this attacker | — | — | — | — | no share: the model scores no better than the board alone |

*MLP probes. A negative share means the rung scores below the board alone. Data: `probe_ladder.csv`.*

![The read-out ladder for nine board questions: each rung's median and range, and each checkpoint's share with the random-vector and oracle shares marked, rows where the share is not interpreted shaded](2026-09-18-effect-model-gen2/charts/probe-ladder.png)

*(a) Each rung's median over the 14 checkpoints with copies, card-disjoint records; the line spans the checkpoints. Yes/no questions are scored by AUC, damage taken by R² on its own axis. (b) Each checkpoint's share carried by `e`, in the shared key; the grey tick marks the median share a random vector of `e`'s width closes, and the hollow purple diamond the median share the oracle rung closes. Shaded rows are the questions whose share is not interpreted. May block has no share, because the model's answer scores no better than the board alone. Data: `probe_ladder.csv`.*

- **What the line states, `e` carries.** How long a pump lasts turns on the script, and whether an ability is affordable turns on its cost. Rung 1 closes nine tenths or more of the gap on both at the median, with no trunk. On affordability, a random vector of `e`'s width closes nearly half the gap by itself, so `e`'s content accounts for about half of that share.
- **What the board decides, the trunk answers, and the ladder cannot say whether `e` could.** Deaths, trigger firing, which cards are affected, target legality and damage taken depend on the board as much as on the line. On each of them a probe on the trunk's vector reads about as much as the model's own answer, or more on damage taken, and the board plus `e` reads far less. The oracle closes a twentieth of the gap or less on deaths, affected cards and targets, and about a seventh on triggers and damage taken. A one-step probe on the board and the true numbers therefore stays far from the answer, and a low share may be the probe's limit rather than `e`'s. The oracle carries only the stated numbers, and it scores below rung 1 on every question but may block, where the two are level.
- **On deaths, `e` adds no more than a random vector of its width.** Rung 1 and rung 1w score the same on deaths at the median, and on the four 256-wide checkpoints rung 1 scores below rung 1w ([e over a random vector](#e-adds-far-more-than-a-random-vector-of-its-width-on-what-a-line-states-and-almost-nothing-on-what-the-board-decides)).
- **Whether a creature may block is known from the board alone, partly because the labels come from the AI's block plan.** A probe on the board alone scores about as well as one on the board plus `e`, and a few hundredths below one on the trunk's vector. On card-disjoint records the model's own answer scores no better than the board alone, and below it on most checkpoints, so no share is defined. More than half of the gen-2 corpus's "may not block" labels name a creature the AI had already planned to block another attacker with, and the snapshot shows that planned block ([the gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#collector-defects-taught-the-ais-block-plan-as-a-rule-and-bent-the-affected-gate-and-the-combat-input), [chart](2026-09-18-effect-model-gen2/charts/planned-block-labels.png)). A probe on the board can read the plan and predict the label without knowing any rule, so this question measures partly how well a vector shows the plan. Gen-3 records block legality before anyone declares, from the rules alone.

Rungs 2 and 3 read the same trunk vector in two ways, and they agree on every question but damage taken and may block. The probe at rung 2 is fitted to one question alone. The output head at rung 3 was trained with every other output field at once, and is what the model actually answers. A gap between them can be knowledge the trunk holds and the model's own answer does not use. It can also come from where the probe is fitted: on the stratum it is scored on, so on card-disjoint records it learns from labelled examples of texts the head never trained on. On may block, the model's answer trails the rung-2 probe on card-disjoint records and matches it on game-disjoint ones, which points to the probe learning the held-out cards' restrictions from its own fitting data. On damage taken, a probe reads about two thirds of the variance from the trunk's vector, and the model's own damage field about half at the median and up to two thirds.

On game-disjoint records the oracle still closes under a fifth of the gap on the board questions at the median, so the same rule leaves their shares uninterpreted there too. Pump duration and affordability keep median shares above nine tenths there.

Gen-1 and gen-2 were probed on different corpora, so they compare by direction only. Gen-1's `e` carried affordability as gen-2's does, and gen-1 could not be asked about deaths at all, because its corpus never recorded a death caused by a resolution.

### `e` adds far more than a random vector of its width on what a line states, and almost nothing on what the board decides

Rung 1 minus rung 1w measures what `e`'s content adds to a probe beyond the extra inputs its width brings. On the properties a line states it adds a large amount on nearly every checkpoint, with mana production the exception. On the board questions the line does not settle, deaths, trigger firing, which cards are affected, block legality and target legality, it adds at most 0.13 of AUC on any checkpoint.

![Rung 1 minus rung 1w per probe target and checkpoint, grouped by family and by width of e](2026-09-18-effect-model-gen2/charts/e-over-random.png)

*Each cell is (rung 1 − rung 1w) × 100 for one checkpoint, from the MLP probe. Board questions are read on card-disjoint records, line properties on held-out lines, except damage per mana (trained lines only) and the interaction pair, whether one line's event fires another's trigger (all pairs). Purple cells are gains and red cells losses; an AUC difference is coloured as twice its value, so that it shares one scale with an R² difference. A random vector's R² falls below zero, so an amount's gain can pass 100. The outlined row is deaths. Gen-1 is probed on its own corpus. Data: `probe_ladder.csv`.*

- **Deaths do not move into `e` with width.** On deaths `e` adds between a hundredth and four hundredths of AUC at widths 64 and 128, and less than nothing at width 256, where a random vector of the same width scores higher. Trigger firing gains a few hundredths at every width. Four times the width of `e` therefore does not move what decides deaths or triggers into the vector.
- **The board questions whose answer the line states gain more.** Damage taken, pump duration and affordability all gain clearly.
- **Costs and amounts gain the most, timing and side clearly, and mana production least.** The total cost, the generic cost and most coloured pips gain the most. The side an effect lands on, its controller's, an opponent's or both, gains about as much as the timing and the costs that tap or sacrifice. The amount of mana a line produces swings from a large gain to a large loss between checkpoints of the 64-wide line.
- **Gen-1's `e` gains far more on three properties.** On the amount of mana a line produces, whether it produces any colour, and how much a text's outcome varies across boards, gen-1's `e` beats a random vector by far more than any gen-2 checkpoint. It also beats every gen-2 checkpoint by a smaller margin on six more, among them colourless mana production and whether a change lasts until end of turn or comes as a counter. The two generations were probed on different corpora.
- **The 128-wide and 256-wide `e` score below a random vector on colourless pips** on most checkpoints, while the 64-wide `e` gains on them.

On a board question about one card, the probe's items are split by that card, so the acting ability's text can fall on both sides. Rung 1w then gives each acting text a fixed code, and the probe can learn that text's usual answer from it. On those questions the comparison is between `e` and a code that names the acting text. On the line properties each item is one text, and rung 1w measures width alone.

Rung 1 minus rung 1w is the comparison of the sweep's arms the design planned: what each arm's `e` holds. The gen-2 record compares the arms' own answers to the probe questions ([width](2026-09-18-effect-model-gen2-improvements-design.md#the-64-wide-and-256-wide-lines-split-the-measures-and-the-128-wide-line-trails-on-the-probes-at-every-stage)).

### Gen-2's `e` holds whether a line taps or sacrifices almost perfectly, and its timing and most of its amounts well

A probe reading `e` alone, with no board and no trunk, recovers whether a line taps or sacrifices nearly perfectly on every gen-2 checkpoint, and whether it is instant speed or repeatable nine times in ten on trained lines. Gen-1's `e` read whether a line sacrifices or is instant speed close to chance on lines it never trained on. The width control, the same probe fitted on a fixed random vector of the same width per text, sits near chance on every yes/no property and below zero on every amount, so the gains are content and not width.

![Line properties read from e alone, trained and held-out lines, gen-1 against gen-2 checkpoints](2026-09-18-effect-model-gen2/charts/line-properties.png)

*Seven checkpoints in the shared key: gen-1, the three arms with copies, and the last epochs of runs B64, B128 and B256. Each line runs from the score on trained lines (small dot), texts the model trained on, to the score on held-out lines (marker), texts it never saw. The grey tick is the same probe on a random vector of `e`'s width, on held-out lines. Data: `probe_ladder.csv` (rungs 1 and 1w, MLP, strata `trained` and `held-out`).*

Amounts transfer less well than costs, and a wider `e` does not change that: the 256-wide arm reads damage, cost and counters on unseen lines slightly worse than the 64-wide and 128-wide arms, and power change slightly better. The damage amount and the total mana cost read nearly as well on held-out lines as on trained ones. Power and toughness changes read well on trained lines and poorly on held-out ones, with toughness change at a quarter of its variance or less. Gen-2 changed two things at once. Its encoder reads an ability's whole chain of script lines, and two heads that read `e` alone train it. The probes cannot separate the two.

Every gen-2 held-out score is somewhat high. About one held-out text in eight also trains on a card that is not held out, such as Flying, so the held-out lines mix in trained texts ([the gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#the-knowledge-probes-filed-trained-texts-as-held-out-lines)).

### `e` barely holds how much a text's outcome varies from board to board

Training the encoder against the board was meant to leave a trace in `e` of how much a text's outcome depends on the board, and the probes barely find one. On trained lines a probe reads between a twentieth and a sixth of that spread from `e`, the least on the 256-wide continuation runs. On held-out lines it reads nothing: its R² falls below zero on every gen-2 checkpoint but `e64 (earlier inputs)`, which sits at zero. Gen-1's `e` read a quarter of the spread on its held-out lines. The two generations were probed on different corpora, so they compare by direction only. The label is the spread of a text's observed outcomes across the boards it resolved on: "draw a card" scores near zero, and a sweeper, whose deaths follow the size of the opposing board, scores high.

| outcome spread across boards, R² of an MLP probe | trained lines | held-out lines |
|---|---:|---:|
| gen-1's `e` | 0.18 | 0.24 |
| gen-2's `e`, median over 17 checkpoints | 0.16 | −0.16 |
| gen-2's `e`, range | 0.06 to 0.18 | −0.24 to 0.03 |
| a random vector of `e`'s width, gen-2 | −0.42 to −0.19 | −0.38 to −0.18 |
| a random vector of `e`'s width, gen-1 | −0.36 | −0.54 |

*Gen-2 has 2,044 trained and 307 held-out lines with the label; gen-1 has 2,028 and 155. The bottom row of the [e over a random vector](#e-adds-far-more-than-a-random-vector-of-its-width-on-what-a-line-states-and-almost-nothing-on-what-the-board-decides) chart shows the same probe as a gain over the random vector. Data: `probe_ladder.csv` (family `state-dependence`).*

The probe finds no support on gen-2 for the claim the model design rests on: that training the encoder against the board yields a better `e` than training it on text alone. It is the probe the design built to test that claim.

### The trunk reads the acting ability's `e` and ignores the `e` of the abilities on the board

Replacing `e` with noise matched to the vectors' mean and covariance raises the loss most on the affected gate, keywords gained and power change, and it raises it just as much when only the `[ACT]` slot is replaced. The affected gate is the model's predicted chance that a card is changed at all. Replacing only the `e` in the card blocks, the ability slots that follow each `[CARD]` slot, costs nothing measurable, on every gen-2 checkpoint as on gen-1.

![Loss increase per field when e is replaced, for three replacements and three scopes, one marker per checkpoint](2026-09-18-effect-model-gen2/charts/ablation.png)

*One panel per replacement: noise matched to the vectors' mean and covariance, the mean `e` of the line's API type, and the `e` of the nearest other text. Per field, one marker per checkpoint with copies for each scope: every ability slot, the `[ACT]` slot alone, and the ability slots of the card blocks alone. Medians under 0.1 are printed. Data: `ablation.csv`.*

Two other replacements show what the trunk takes from `e`. The `e` of the nearest other text costs little on any field, so fine differences between similar texts matter little to the trunk. The mean `e` of the line's API type, the first word of its script, costs nearly as much as noise on keywords gained. Every keyword grant shares the `Keyword` API type, so the type's mean says that a keyword is granted and not which.

The ablation shows which `e` the trunk uses, not what the trunk works out from it. The trunk needs the acting `e` to know which ability it is looking at, and the ladder above shows a probe on the trunk's vector answering the board questions that the board plus `e` does not.

### A restriction the board snapshot does not list costs the model the most on attack legality

Where a static the snapshot does not list keeps a creature from attacking, every checkpoint's attack-legality loss rises well above its loss with no restriction. The snapshot gives each permanent its spell and activated abilities as lines in its card block, and nothing else. A permanent's statics, triggers and replacements reach the model only through what they have already done to the board: an anthem's +1/+1 is in the toughness, a granted keyword is in the flags. A Pacifism shows the model an enchantment attached to a creature and no text, and a Rest in Peace an enchantment that does nothing. The board-abilities ablation pairs records where an unlisted line decides the answer with a control of the same kind where none does, 600 records per subset ([the subsets](#the-card-blocks-hold-no-static-trigger-or-replacement-so-a-second-ablation-measures-what-their-absence-costs)).

![Each checkpoint's loss on restricted records divided by its loss on the unrestricted control, per restriction, log scale](2026-09-18-effect-model-gen2/charts/board-abilities.png)

*Each marker is one checkpoint's loss on the field named, on records where the restriction or line is present, divided by its loss on the unrestricted control of the same kind. A marker right of ×1 means the unlisted line raises the model's loss. The range at the right spans the checkpoints. Data: `board_abilities.csv`.*

- **Attack legality is the clearest case.** Where a static that is not a keyword keeps a creature from attacking, as Pacifism does, the loss is between one and a half and four and a half times the unrestricted loss. Every checkpoint's restricted loss sits above every checkpoint's unrestricted one. A keyword restriction, which the card's keyword flags carry, is the easiest case of the three.
- **Block legality follows the same order with smaller gaps.** On the arms a static restriction costs about as much as no restriction, and on the continuation runs up to about twice as much.
- **On resolutions, a replacement or a static on another permanent about doubles the zone-outcome loss at the median.** That is the case where a destroy turns into an exile, or does nothing.

On the restricted subsets, noise in the card blocks moves the charted fields' loss by six hundredths at most, in either direction, because the lines that would matter are not in the card blocks to begin with. The subsets differ in more than the unlisted line, since a board with Pacifism is a different game state from one without, so the gaps bound the cost of the missing text rather than measure it exactly. Gen-3 lists every line of a permanent in its snapshot, and these losses are the baseline it is measured against.

### Evasion: the model knows an evasive attacker is hard to block, and which blocker answers it only for horsemanship and fear

Every gen-2 checkpoint lowers the predicted chance of a block when the attacker gains flying, horsemanship or fear, and all but one when it gains intimidate. Only horsemanship and fear are also tied to the blockers that answer them. The evasion grid takes 40 real block-legality records from the validation games, one per game, in which the blocker was legal and neither creature has an evasion keyword. It grants the attacker one keyword and gives the blocker one trait, then reads the model's predicted chance that this blocker may block this attacker. The rule's answer is computed for every edited record. The script is `scripts/effect_knowledge_probes/evasion_grid.py`.

The absolute level of the prediction is not a probability to read at face value. Even with no keyword and no trait, where every block is legal, arm `e64-lethality` predicts about one in ten, run B64's last epoch about one in three and run B256's last epoch about one in two. The model learned that level from labels that mark many legal blocks as forbidden, the block-plan labels the ladder section describes ([the gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#collector-defects-taught-the-ais-block-plan-as-a-rule-and-bent-the-affected-gate-and-the-combat-input)). The comparisons below are therefore within each checkpoint: how much a blocker trait lifts the prediction against an attacker with the keyword, against how much it lifts it against an attacker without one. These rankings within a row compare edits of one record and are less exposed to the labels than the absolute level.

![Heatmap of the lift each blocker trait gives the predicted chance of a block, per attacker keyword, median over 17 checkpoints, cells the rule allows marked](2026-09-18-effect-model-gen2/charts/evasion-grid.png)

*Each cell is the median over the 17 gen-2 checkpoints of the lift: the predicted chance of a block with the blocker's trait, divided by the chance with no trait, against the same attacker. The top row is the reference, the lift against an attacker with no keyword. Under each lift, n/17 counts the checkpoints where the trait lifts the prediction more against that keyword than against a plain attacker. A black corner marks a block the rule allows, and a hollow corner a block it allows only when the attacker is black too. Data: `evasion_grid.csv`.*

- **Horsemanship and fear are paired with their answers.** A horsemanship blocker lifts the prediction several times more against a horsemanship attacker than against a plain one, and an artifact or black blocker does the same against fear, on every checkpoint.
- **Intimidate is read as fear.** An artifact blocker answers it, as the rule says, but so does a black one, which the rule allows only when the attacker is black too, and sharing a colour, the answer the rule names, barely registers.
- **Flying is only half paired.** A flying blocker answers a flying attacker somewhat more than a plain one. Reach, which the rule treats exactly like flying here, lifts the prediction hardly more than no trait at all.
- **Some traits read as "a good blocker" whatever the attacker.** A horsemanship or flying blocker lifts the prediction against every attacker, a plain one included, though neither trait matters against a plain attacker.
- **Shadow and "can't be blocked" change nothing.** Granting either leaves the row nearly identical to the plain one. A shadow blocker is predicted to block a little less against every attacker, and not more against a shadow one, which is why the shadow row ranks below chance.
- **The two protection rows are identical by construction.** The keyword flags carry protection without its colour, so protection from the blocker's colour and protection from another colour give the model the same input. The [gen-3 draft](2026-XX-XX-effect-model-gen3-design.md) gives the flags the quality.

The ranking within each attacker row, allowed blocks against forbidden ones, puts the same picture on one scale. Horsemanship ranks near perfectly on every checkpoint, fear well above chance, flying and intimidate near chance, and shadow below it. No width and no stage of training moves the pooled ranking off about 0.6, so what the model knows about evasion was settled early and did not change with the continuation runs.

![Ranking AUC per attacker keyword, one marker per checkpoint](2026-09-18-effect-model-gen2/charts/evasion-ranking.png)

*AUC of the predictions on the blocks a row allows against the ones it forbids, over the grid's cells and records, in the shared key. "Can't be blocked" and protection from the blocker's colour allow no block and have no ranking. Data: `evasion_ranking.csv`.*

### Targeting: the model follows who controls a target, and barely checks the card type a restriction names

On real targeting decisions the model separates legal targets from illegal ones almost perfectly, whatever the restriction, and the flips show that much of this rests on what a target looks like rather than on the restriction's own condition. The probe takes 1,500 real targeting decisions from the validation games, at most three per game. Each battlefield permanent on the board is an item, labelled legal or not by Forge's own list of legal targets. The model's answer is the affected gate at that permanent's slot. Items are grouped by the acting ability's `ValidTgts$` restriction and by the kind of condition it states. The script is `scripts/effect_knowledge_probes/target_restrictions.py`.

| kind of restriction | items | legal | AUC, median over 17 checkpoints | range |
|---|---:|---:|---:|---|
| a card type only, such as `Creature` or `Any` | 10,859 | 2,410 | 0.999 | 0.997–0.999 |
| who controls it, such as `Creature.YouCtrl` | 1,414 | 242 | 0.998 | 0.984–0.999 |
| other, such as `Permanent.nonLand` | 1,225 | 224 | 0.982 | 0.959–0.989 |
| attacking, blocking or tapped | 451 | 36 | 0.992 | 0.977–0.996 |
| a colour, such as `Creature.nonBlack` | 420 | 77 | 0.967 | 0.958–0.989 |
| a number, such as `Creature.powerLE2` | 267 | 57 | 0.974 | 0.966–0.985 |
| a keyword, such as `Creature.withFlying` | 60 | 11 | 0.980 | 0.935–0.989 |

*Data: `target_families.csv`, and per restriction `target_restrictions.csv`. A supertype family holds 79 items with 2 legal ones, too few to weigh.*

The illegal targets on a real board are mostly easy: a land when the restriction asks for a creature, or the acting player's own creature when it asks for an opponent's. The flips ask the harder question. Each takes a legal target of a restriction with a single condition and breaks exactly that condition, which by the rule makes it illegal.

![Predicted legality of each flip's targets before and after the flip, one marker per checkpoint](2026-09-18-effect-model-gen2/charts/target-flips.png)

*For each flip, each checkpoint's mean predicted legality of the targets before the flip (grey ticks) and after it (markers, in the shared key); the arrow joins the medians. By the rule every target is legal before the flip and illegal after it. Rows with fewer than 20 targets are greyed as too few to weigh. Data: `target_flips.csv`.*

- **Control is learned.** A target of "target creature you control" handed to the opponent falls from near certain to one in four or less on every checkpoint but `e64 (earlier inputs)`, which stops at about two in five.
- **A creature that loses its creature type stays a legal target of "target creature".** The prediction barely moves on 1,380 targets. The edit leaves the permanent its power, toughness and combat status, so the model may read creature-ness from those rather than from the type line. Either way, the model does not check the condition the restriction states. Lands that lose the land type barely move either, except on `e64`, where they fall by a third.
- **The other flips are too few to weigh.** Artifacts that lose the artifact type fall by about a third, on 8 targets. Enchantments that lose the enchantment type rest on 7 targets, and the opponent's creatures handed to the acting player on 14, which the model already doubted before the flip.

No restriction in the sample states `nonLegendary` alone, so the legendary flip did not run.

### Colour: the model counts the mana a player has and barely checks its colour

On every gen-2 checkpoint, taking one land away lowers the predicted affordable verdict four to ten times as much as turning one land to a colour the cost cannot use. Two extra lands of the wrong colour raise the verdict above the exact board that casts the spell, even when a pip is left unpaid. The probe takes 300 real playability decisions from the validation games, at most two per game, whose candidate has a mana cost basic lands pay one pip each. It replaces the acting player's lands with untapped basics in six configurations. The off colour is the first of white, blue, black, red and green the cost does not name. The rule's answer is computed for every configuration by matching pips to lands. The script is `scripts/effect_knowledge_probes/colour_pips.py`.

| configuration | the acting player's untapped basics | the rule |
|---|---|---|
| exact | one land of each coloured pip's colour, one off-colour land per point of generic mana | affordable |
| surplus | exact, plus two off-colour lands | affordable |
| one short | exact, less one land | not affordable |
| pip missing | exact, with one coloured pip's land turned off-colour | not affordable |
| pip missing, +2 | pip missing, plus two off-colour lands | not affordable |
| all off colour | as many lands as the cost, all off-colour | not affordable |

The verdict's absolute level is low on every checkpoint: the exact board, which always casts the spell, is predicted affordable at about one in five. The comparisons are therefore made within each decision, as the change from the exact board.

![Change in the affordable verdict from the exact lands for each configuration, and the ranking AUCs, one marker per checkpoint](2026-09-18-effect-model-gen2/charts/colour-pips.png)

*Left: each checkpoint's mean change in the predicted affordable verdict from the exact lands, over 300 decisions, in the shared key; the exact lands are the zero line, and each row label gives the rule's answer. Right: the AUC of affordable boards ranked above unaffordable ones, over every configuration, for the exact and surplus boards against the three off-colour ones, and for the exact board against one land short. Data: `colour_pips.csv` and `colour_pips_ranking.csv`.*

| configuration | change from exact, median over 17 checkpoints | range | decisions below exact, median |
|---|---:|---|---:|
| surplus | +0.106 | +0.064 to +0.134 | 0% |
| one short | −0.108 | −0.069 to −0.136 | 100% |
| pip missing | −0.018 | −0.009 to −0.027 | 75% |
| pip missing, +2 | +0.087 | +0.056 to +0.119 | 1% |
| all off colour | −0.021 | −0.011 to −0.033 | 79% |

*Data: `colour_pips.csv`; per kind of cost, `colour_pips_by_cost.csv`.*

- **The amount of mana is read.** One land fewer lowers the verdict on every decision and every checkpoint, and two lands more raise it.
- **The colour is barely read.** A missing pip lowers the verdict on three decisions in four, by a sixth as much as a missing land at the median and between a quarter and a tenth across checkpoints. Lands that are all off-colour lower it barely more than one wrong land does.
- **More mana of the wrong colour outweighs the missing colour.** A board one pip short with two extra off-colour lands is predicted more affordable than the exact board on nearly every decision, on every checkpoint.
- **The pips of one colour are not counted.** On the 52 costs with a repeated pip, such as `{1}{U}{U}` given a single Island, the missing pip leaves the verdict where it was.
- **Ranking by colour is close to chance.** Ranked against the unaffordable boards that only differ in colour, the affordable ones score an AUC of 0.52 to 0.56. Ranked against one land short, the exact board scores 0.63 to 0.73.

The training corpus holds the lesson. In a sample of 21 training shards, 204 of the 894 land-payable candidates Forge judged unaffordable had enough mana in the wrong colours. The ladder above finds that the board plus `e` carries nearly all of what the model knows about affordability. That fits a model that judges affordability from the cost and the amount of mana, and never matches the cost's pips against the colours the lands make.

**The `affordable` label gen-2 trained on carries some of Forge's AI cost policy, and the colour lesson is in it as stated.** The label comes from Forge's AI cost check. That check matches pips to mana sources by the rules. It also refuses some costs the rules allow, such as a planeswalker's last loyalty, a Ward cost the AI dislikes and a Casualty sacrifice. A few candidates with enough mana of the right colours are therefore labelled unaffordable, in training and in the probes' own labels. Gen-3 labels affordability from the mana and additional-cost checks alone ([gen-3 design](2026-XX-XX-effect-model-gen3-design.md#the-model-has-to-learn-every-rule-that-depends-on-whose-turn-or-which-step-it-is-and-collection-records-which-check-decided-each-one)).

### Zones: the model knows which player draws, discards or mills and where a moved card goes, and barely uses the number of cards

Draws, discards, mills and moves between hand, library, graveyard, exile and the battlefield are predicted well on every gen-2 checkpoint. The weak points are two. The number of cards a script states moves the prediction far less than it should. Handing a player target to the other player moves only part of the predicted effect with it. The probe takes 2,000 real resolutions from the validation games, at most three per game, whose acting chain names an API that moves cards: `Draw`, `Discard`, `Mill`, `ChangeZone`, `ChangeZoneAll`, `Dig`, and the destroy and sacrifice APIs so that exile has something to be told apart from. The script is `scripts/effect_knowledge_probes/zone_effects.py`.

The model knows whether a player draws, discards or mills, and which player, and predicts fewer cards than the player gets. Card counts are read at the player slots. The model's answer for a player is the affected gate times the field's predicted count.

| cards a player | player slots | affected | AUC, median over 17 checkpoints | range | more predicted for the affected player, median | actual cards, mean | predicted cards, median |
|---|---:|---:|---:|---|---:|---:|---:|
| draws | 4,000 | 616 | 0.989 | 0.981–0.992 | 97% | 1.35 | 1.09 |
| discards | 4,000 | 148 | 0.997 | 0.995–0.998 | 99% | 1.21 | 0.90 |
| mills | 4,000 | 120 | 0.995 | 0.981–0.997 | 98% | 2.76 | 2.17 |

*The "more predicted for the affected player" column covers the records where exactly one player was affected: 419 for draws, 124 for discards, 84 for mills. Data: `zone_counts.csv`.*

- **Whether a player is affected, and which one, is known.** The model separates affected players from the rest at an AUC near 0.99, and predicts more cards for the affected player than for the other on about nine records in ten or more, on every checkpoint.
- **The count is predicted below the real one.** Every field's median prediction is short of the mean actual count, by a fifth for draws, a quarter for discards and a fifth for mills.

Where a moved card goes is right on most moves, and two pairs of destinations get confused. Card moves are read at the moved card's slot, grouped by the zone the card was in. Whether a card moves at all is ranked at an AUC of 0.991 to 0.995 for permanents, 0.992 to 0.998 for cards in a graveyard and 0.977 to 0.990 for cards in hand. For the cards that did move, the chart shows how often the model's most probable destination other than staying put is the right one. The head names an outcome by its event, so a card put into a graveyard from anywhere is `died` and a card put onto the battlefield is `blinked`; the chart names the destination zone.

![Share of moved cards whose most probable destination is right, by starting zone and destination, one marker per checkpoint, with each row's most common wrong destination](2026-09-18-effect-model-gen2/charts/zone-outcomes.png)

*One marker per checkpoint, in the shared key. Rows with fewer than 20 cards are greyed, and rows with fewer than nine are left out. The right margin names each row's most common wrong destination and on how many of the 17 checkpoints it is the most common. Data: `zone_outcomes.csv`, which holds every row, with the outcome predicted most often in place of the right one.*

- **From the battlefield, all four common destinations are right on more than nine moves in ten** at the median: graveyard, exile, hand and library. The usual mistake is between graveyard and exile, a destroy read as an exile or the reverse.
- **From the graveyard, exile and library are near perfect.** Returning a card to its owner's hand and returning it to the battlefield are confused with each other, and some checkpoints get only three in five of either right.
- **Discards are mostly right.** A card discarded from hand goes to the graveyard in the model's answer on six moves in seven at the median, and to exile on most of the rest.
- **Blinks are the weakest move, on too few cards to weigh.** A permanent exiled and returned to the battlefield is right on seven of its 10 moves at the median and on three on the worst checkpoint.

The predicted count rises with the number of cards the script states, but covers a fraction of the range. The amount sweep sets the literal `NumCards$` of 40 draw, 40 discard and 40 mill scripts to each value from 1 to 6, re-encodes the text with the checkpoint's own encoder, and reads the predicted count at the affected player, as the gen-2 record's damage sweep does for damage.

![Predicted cards at the affected player as the script's number of cards rises, one line per checkpoint, against the script's number](2026-09-18-effect-model-gen2/charts/zone-amounts.png)

*One line per checkpoint, in the shared key; grey lines are the arms without copies. The black line is the script's number. Data: `zone_amounts.csv`.*

- **Draws and mills rise far less than the rule.** From one card to six, the predicted draw rises from about 1.2 to 1.9 at the median, where the rule rises from 1 to 6. Mills about double, from 1.8 to 3.9.
- **Discard barely moves**, from 0.8 to 1.1 at the median, and the earliest checkpoint does not rise at all.
- **The continuation runs follow the number more closely than the arms do.** At six cards milled the arms predict 1.9 to 3.5, and the continuation runs 3.4 to 4.9. Every continuation checkpoint but run A256's predicts more than any arm.

The amount sweep reads the head only, so it cannot say whether the trunk holds the number and the head fails to use it.

The predicted effect follows a player target only partly. The target flip takes the draw, discard and mill records with exactly one player target, where the targeted player is the one affected, and hands the target to the other player. The rule moves the whole effect with it.

![Share of the predicted cards still at the originally targeted player after the target is handed over, one marker per checkpoint](2026-09-18-effect-model-gen2/charts/zone-who.png)

*After the target is handed to the other player: the share of the predicted cards still at the originally targeted player, in the shared key. The rule puts it at 0. Before the flip that share is 95% to 98% at the median per effect. Data: `zone_who.csv`, which also holds each player's predicted count before and after.*

- **Most of the predicted cards move, and a quarter to a third stay.** Before the flip the targeted player holds nineteen twentieths of the predicted cards at the median. After it, the originally targeted player still holds about a third of a draw and a quarter of a discard or a mill.
- **How much moves varies widely between checkpoints.** The 64-wide continuation runs move nearly all of a discard to the new target, and the 256-wide continuation runs nearly all of a mill. Arm `e128-lethality` leaves two thirds of a discard and a mill where the target was. No order by width or by stage of training holds across the three effects.

### The affordable verdict never clearly passes one half, and a sweeper's predicted deaths rise by a fifth per creature added

The affordability and board-size sweeps look the same on every gen-2 checkpoint. The toughness and damage sweeps measure the damage-versus-toughness rule, which only appeared once training added lethality copies, so the gen-2 record discusses them with its damage grids: [the death rule](2026-09-18-effect-model-gen2-improvements-design.md#the-model-compares-damage-with-toughness-only-after-training-shows-it-the-rule-directly) and [where the boundary sits](2026-09-18-effect-model-gen2-improvements-design.md#where-the-boundary-sits-wanders-between-epochs-at-the-full-learning-rate-and-settles-near-the-rule-at-a-tenth-of-it).

![Affordability sweep and board-size sweep, one line per checkpoint](2026-09-18-effect-model-gen2/charts/other-sweeps.png)

*Left: the predicted affordable verdict as the acting player's untapped mana rises, one line per gen-2 checkpoint in the shared key; the numbers along the bottom are the items behind each point, and the bold line marks an even chance. The flat grey line near 0.29 is `e64 (earlier inputs)`, probed before the sweep set the lands ([the fix](#the-affordability-sweep-sets-the-lands-as-well-as-the-mana-2026-10-15)). Right: predicted deaths summed over the board as opposing creatures are added, one line per checkpoint, gen-1 dark; the dashed line is the rule's one death per creature. Data: `sweeps.csv`.*

- **The affordable verdict rises with the mana given and never clearly passes one half.** Only arm `e256-lethality` reaches it, by under a hundredth, at 7 and 8 mana, where the points rest on 6 and 4 items. The rise is gradual because each point averages items with different costs. The model separates affordable candidates from unaffordable ones at an AUC of 0.98, so it reads affordability partly from something the sweep does not set.
- **A sweeper is predicted to kill about a fifth of the creatures added.** Every gen-2 checkpoint rises by about a fifth of a death per creature, with or without copies and at every width, where a sweeper kills one per creature. Gen-1 rose by a fiftieth. The sweep cannot tell whether the records' sweepers carry conditions that spare most creatures or the model spreads its confidence thinly over the board.

## Design of the suite

The design has three methods and a set of board sweeps. Method A, the read-out ladder, fits probes to inputs taken from progressively deeper in the model. Method B, an ablation, replaces `e` in the frozen model and measures how much the loss rises. A frozen model's weights stay as trained and are not updated. Method C would train a shallow trunk on frozen `e`, and never ran. Board sweeps edit one input of a real record and read the full model's prediction. The existing embedding probes in `scripts/effect_embedding_probes/` read `e` alone. They label each text from its script or from its effect profile, the shares of its resolutions in which affected entities died, changed zone, took damage and so on. The probes below also read the trunk and compare the two.

### The probes measure what game knowledge each model holds, and how much of it lives in `e` rather than in the trunk

The suite has three goals. The first is to find game knowledge the model encodes beyond what the existing embedding probes show. The second is to measure how much of that knowledge lives in `e` and how much in the trunk. Gen-3 aims to hold as much of the knowledge as possible in `e`, so this measurement shapes gen-3's architecture. The third is to compare the arms of gen-2's sweep.

Knowledge that lives in `e` survives into the game agent, and knowledge that lives only in the trunk may not. The game agent ([`2026-09-12-game-playing-agent-design.md`](2026-09-12-game-playing-agent-design.md)) reuses the ability encoder through its cache, which stores one precomputed `e` per ability line. It may retrain the trunk or replace it. Whatever the trunk re-derives from context every time it computes a prediction is then lost to the agent, which has to learn it again from its own much weaker training signal.

One vector per ability is also easier to interpret. It can be inspected, compared with another ability's vector and reused by any consumer without running a board through the trunk.

### The probes cover what an ability does to the board, in ten families, and leave strategic value to the game agent

The encoder's job is to capture an ability's immediate effect on the board. Strategic value, such as a card's win rate or its price, is out of scope for two reasons. The training corpus does not contain it, so the model has no signal to learn it from. Learning it is the game agent's job.

The ten families below are what the probes target. Each label comes from a source that exists without the model: the parsed script, or an outcome observed in the effect-record corpus.

| Family | Targets | Label sources |
|---|---|---|
| Magnitudes and thresholds | damage, power and toughness change, counter and life amounts; whether the effect kills a creature of toughness *t*; each amount per mana of cost | parsed script parameters; observed outcomes in resolution records |
| Mana production | colours produced, amount, any colour, spending restrictions, conditional production | the script's mana parameters; the `mana_produced` events of mana resolution records |
| Mana usage | the ability's own mana cost; cost reducers and taxes on other spells; whether it is affordable on a given board | the line's `Cost$`, which labels no spell line, because a spell's mana cost belongs to its card; playability decision records |
| Timing and non-mana costs | instant speed, a tap cost, a sacrifice or life cost | the script |
| Interactions | whether ability A's event fires ability B's trigger, over pairs of lines | trigger records joined to resolution records |
| Side | whose permanents and resources the effect changes: its controller's, the opponents', or both | the side of each affected entity in resolution records; the script's target and player restrictions |
| Evasion and blocking | which defending creatures may legally block a given attacker | legality records of the `blockers` subkind |
| Target legality | whether an ability may target a given permanent that has hexproof, shroud, protection or ward | the legal-target sets of playability decision records |
| Duration and repeatability | whether a change lasts until end of turn, stays, or comes as a counter; whether the ability can be used again | the events' duration field; the line kind and whether the cost taps or sacrifices |
| State dependence | how much a text's outcome varies across the boards it resolved on | the spread of each text's observed outcomes across its resolution records |

The embedding probes already show that the encoder keeps which kind of outcome an ability produces and loses its amounts and costs. The first five families extend that to thresholds, to mana and to pairs of abilities, and they ask the same questions of the trunk.

The last five families each target a property the first five leave out.

- **Side** is the property that separates removal from a pump spell and a sweeper from an anthem. The fifth principal component of gen-1's `e` vectors already follows the opponent's share of affected entities, and no existing probe targets it.
- **Evasion and blocking** is where gen-1 learned least. Gen-1's two blocker fields score exactly at the level of the state-only baseline, the gen-1 model trained with every ability vector zeroed so that it predicts from the board alone. The text taught the model nothing about who may block whom.
- **Target legality** decides whether a removal spell can be aimed at all.
- **Duration and repeatability** separate abilities whose immediate effect is identical: a +1/+1 counter and a pump until end of turn, or a repeatable activation and a one-shot spell.
- **State dependence** tests the claim the model design rests on: that training the encoder against the board yields a better `e` than training it on text alone. The model design record's `no-state` baseline was meant to test that claim by reading the acting ability's text without the board. It zeroed the board and every ability vector, the acting one included. Its encoder therefore never trained, and the claim had no test before this probe.

### The read-out ladder measures the vector's share of what the model knows, and is the headline for gen-3 (method A)

The ladder reads one target from inputs of increasing depth, the rungs defined at the top of [the ladder results](#e-carries-the-answers-a-line-states-and-the-trunk-answers-what-the-board-decides). Rungs 0 to 2 fit the same two probes to their input, a linear probe and a small MLP, both fixed as the probe-architecture subsection below describes. Rung 3 is the model's own prediction. Rungs 1w and 1o are controls on rung 1.

Every rung is scored so that higher is better. Rung 0 is what the board alone tells a probe. Rung 3 is what the whole model knows.

The headline for gen-3 is the vector's share, defined with [the ladder results](#e-carries-the-answers-a-line-states-and-the-trunk-answers-what-the-board-decides) and computed for each of the two probes. A share near 1 means `e` carries the knowledge, and a share near 0 means the trunk supplies it.

A property of a line that does not depend on the board, such as the damage it deals or its mana cost by colour, gets a short ladder: `e` alone, its width control, and the trunk's output at `[ACT]`. A line property a probe reads much better from the trunk's output than from `e` is one the trunk re-derives from context, and one `e` could have held.

The per-layer view would probe the output of each trunk layer in turn. It shows at what depth a fact appears, which feeds the sizing of gen-3's trunk.

#### Every probe's folds are split by the card or text an item is about, so no probe can look up that card's or text's answer

Every probe is fitted on all folds but one and scored on the fold left out, in turn. `e` is a fixed function of the text. A probe fitted on some records of a text and scored on other records of the same text could read the label off which text it is, and that lookup would inflate rung 1. A line property's folds are therefore split by text, and a question about the acting ability is split by the acting text. A question about one card pools that card's lines, so its folds are split by card, as the embedding probes already group them. The acting text can then fall on both sides of a fold ([what that does to rung 1w](#e-adds-far-more-than-a-random-vector-of-its-width-on-what-a-line-states-and-almost-nothing-on-what-the-board-decides)). Rungs 0, 2 and 3 use the same folds, so the rungs of one target are comparable.

#### The share is reported only where the model knows more than the board

The share is computed only where rung 3 exceeds rung 0 by a minimum gap, fixed before the first run at 0.05. Below that gap the denominator is near zero and the ratio means nothing. Gen-1 already has such fields. The state-only baseline is the trained-model counterpart of rung 0. Reading the text gains nothing over it on the two blocker fields, and one percentage point on damage taken. The raw rungs are printed for every target either way.

A share above 1 is possible. The probe is fitted to this one target alone, while the model's heads are trained on every output field together. The probe can therefore extract the target from `e` better than the heads do. A share above 1 says that `e` holds the target at least as readably as the model applies it.

#### The oracle rung says whether a low share is the vector's fault or the probe's

Rung 1o separates the two causes of a low share. [The ladder results](#e-carries-the-answers-a-line-states-and-the-trunk-answers-what-the-board-decides) name both causes and state the rule for which shares are interpreted. A probe that cannot combine `e` with the board was expected on the thresholds, where the probe must compare an amount with a toughness.

#### The width control keeps arms of different widths comparable

A probe that reads a wider vector has more inputs, and it scores higher on that account alone. Rung 1w gives the probe a vector of the same width with no content, a fixed random vector per text. Where folds are split by text such a vector carries nothing from one text to another, so whatever it adds over rung 0 is what width alone buys. Where they are split by card it also names the acting text.

The arms of gen-2's sweep are compared here on rung 1 minus rung 1w, which says what each arm's `e` holds. The share's denominator is each arm's own rung 3, so an arm with a weaker head shows a larger share. The share answers where knowledge lives, for gen-3's design. The gen-2 record compares the arms' own answers.

#### Both probes are fixed across every checkpoint, and the MLP's share is the headline

Rungs 0 to 2 use a fixed pair of probes: a linear probe and a two-layer MLP of fixed width. Its hyperparameters, the settings chosen before fitting such as its width, learning rate and regularisation strength, are fixed once for every checkpoint. A difference in a rung between two checkpoints then reflects the models and not the probe.

Every rung reports both probes' scores, and the share and the arm comparison are computed once per probe type. One computation never mixes the two types across rungs, because a ratio of a linear score to an MLP score would compare two different instruments. The MLP's share is the headline: the game agent reads `e` through non-linear layers, so the MLP's score is the closer measure of what it can use. The linear share shows whether a fact sits in `e` in a form a linear read-out finds.

### Replacing `e` in the frozen trunk measures how much the trunk relies on it, and supports the ladder (method B)

Method B runs each frozen model over the validation samples and reports each output field's loss increase when `e` is replaced by one of three substitutes.

| Replacement for `e` | What the loss increase measures |
|---|---|
| noise matched to the vectors' mean and covariance | whether the trunk relies on `e` at all |
| the mean `e` of the line's API type, the first word of its Forge script, which names the effect the line runs | whether it relies on `e` beyond the API type |
| the `e` of the other text whose vector lies closest | how sensitive it is to fine differences between texts |

The same replacements are also applied to the `[ACT]` slot alone and to the ability slots of the card blocks alone. The comparison shows whether the trunk leans on the acting ability's vector or on the vectors of the abilities around it.

Ablation measures reliance, not location. A trunk can lean on `e` and still do the work itself: it may need `e` to know which ability it is looking at and then compute everything else from the board. That is why the ladder is the headline and the ablation supports it.

#### The card blocks hold no static, trigger or replacement, so a second ablation measures what their absence costs

The full ablation cannot show what an unlisted line costs. Its card-block replacement costs almost nothing because the lines that would matter were never in the card blocks, and the records where they matter are a small part of its sample. Who may block a creature, and what a destroy turns into, are the two places this matters most. About a quarter of the permanents on the battlefield in card-disjoint records carry such a line on their current face, triggers most often.

| battlefield permanents in 1,500 records of the first card-disjoint shard, counted once per record | count |
|---|---:|
| sampled | 24,363 |
| with a trigger, a non-keyword static or a replacement | 6,394 |
| with a trigger | 3,882 |
| with a non-keyword static | 3,010 |
| with a replacement | 1,274 |

The board-abilities ablation runs method B on six subsets of the validation strata instead. Each pairs records where an unlisted line decides the answer with a control of the same kind where none does.

| subset | records |
|---|---|
| legality-static | attack or block legality, a creature kept out by a static that is not a keyword line, such as Pacifism |
| legality-keyword | the same, kept out by keyword lines only, such as protection, which the keyword flags carry |
| legality-none | the same, nobody kept out |
| resolution-replacement | an effect half, with a replacement line on a permanent other than the source |
| resolution-static | an effect half, with a non-keyword static on another permanent and no replacement |
| resolution-none | an effect half, with no static, trigger or replacement line on any other permanent |

Gen-2 does not list the missing lines; the gen-3 draft records the collector change that would. The subsets' losses on gen-2's checkpoints are the baseline that change is measured against. A model that reads the lines should lower the loss on legality-static and resolution-replacement toward their controls, and its card-block ablation there should rise above zero.

### The share and the ablation together say which families gen-3 must move into `e`

Each pattern of share, reliance and oracle rung points to one next step: a change to gen-3's architecture, more corpus records, or a stronger probe.

- A family with a low share and high reliance is one where the trunk does work `e` could carry. Gen-3 then moves that work into `e` in one of two ways. It can add auxiliary heads, extra output layers that predict the family's labels from `e` alone, so that only `e` can satisfy their training signal. Or it can use a thinner trunk that cannot re-derive the knowledge itself.
- A family that no rung predicts well is knowledge the model lacks altogether. That is a corpus problem, not an architecture problem, and the fix is records that show the effect.
- A family whose oracle rung stays below rung 3 needs a stronger probe before its share says anything.

The design fixes no cut-off for a low or a high share, so shares are compared family against family rather than against a fixed threshold.

### Board sweeps edit one input of a real record and read where the full model's prediction steps

A threshold that depends on the board is probed by a board sweep. The board sweep takes a real record, edits one input over a range of values, and reads the full model's prediction at each value. A model that knows the threshold shows a step at the right value.

- **Toughness sweep.** A damage ability sits in the acting slot. One target creature's toughness runs from 1 to 8, and the sweep reads the predicted death probability for that creature. A model that knows Lightning Bolt shows a step between toughness 3 and toughness 4.
- **Affordability sweep.** An ability sits in the acting slot. The acting player's untapped production runs from none to more than enough, and the sweep reads the predicted affordable verdict. The verdict should step at exactly the cost: `{2}{R}` needs three mana, one of them red.
- **Board-size sweep.** A sweeper sits in the acting slot. The number of opposing creatures runs from 0 to 8, by copying a creature entity of the record's board, and the sweep reads the predicted deaths summed over the board. They should rise one for one. This is the scaling calibration the model design record asks for.
- **Damage sweep.** The board stays fixed with a target of toughness 4. The acting text is one damage spell's script with `NumDmg$` set from 1 to 8, and the checkpoint's own encoder encodes each version. The predicted death should step between 3 and 4. The toughness sweep moves the board, and the damage sweep moves the text. The damage sweep therefore tests whether `e` carries the amount in a form the trunk can compare with toughness. Run on a triggered damage ability as well, it also tests gen-2's whole-chain encoding, which encodes every script line an ability owns as one text. On a triggered ability the damage amount sits in a sub-ability's script line, and only whole-chain encoding brings that line to the encoder.

### A shallow trunk trained on frozen `e` would imitate the game agent's reuse of the encoder, and stays off by default (method C)

Method C trains a deliberately shallow trunk on frozen `e`, with one transformer layer or none, and compares it with the full model. It is the closest imitation of the game agent reusing the encoder with a trunk of its own.

Method C costs several GPU hours per checkpoint, against under twenty minutes for the rest of the suite. Two training runs of the same shallow trunk also differ by chance, so its result compares less well between checkpoints than the ladder's probes, whose architecture and settings are fixed across checkpoints. It therefore runs only behind a flag, and no run set it.

### Four families need labels assembled from several corpus records or fields

#### Interaction labels come from joining triggers to resolutions, because no trigger record names the ability behind its event

No trigger record names the ability that caused its event. About half the records name a `cause` entity in their event's params, but `cause` is the object that caused the event, not the ability. No record sets `attributed_to`. The counts are over the 1,000 games of gen-1's game-disjoint stratum.

| Trigger records | Count |
|---|---:|
| all | 15,335 |
| fired | 8,377 |
| naming a `cause` entity | 7,752 |
| naming a `cause` entity and fired | 4,380 |
| setting `attributed_to` | 0 |

The interaction label is therefore built by a join. A fired trigger record carries its pending event. The resolution record in the same game whose events contain that event names the ability that produced it. That ability and the trigger's line form a positive pair. The join rate is the share of fired trigger records whose pending event the join finds in a resolution record.

Pairs the join cannot reach are mined from the scripts. A death trigger paired with a sacrifice outlet, or a lifegain trigger paired with a lifegain ability, is a positive pair by construction.

Trigger records that were evaluated and did not fire are the negatives. They occur naturally in the corpus, because the collector records non-fired evaluations of the same event type.

#### Playability decision records carry everything affordability and target legality need

A decision record is enough to recompute whether a candidate ability was affordable. Each candidate carries its `affordable` flag and its `cost_after_adjustment`, and each player carries `untapped_production` and `floating_mana` by colour. The mana-usage family reads both its labels and the affordability sweep from these fields. The same records carry each candidate's legal-target set, which labels the target-legality family.

#### State dependence is labelled from the spread of each text's own outcomes

A text's state-dependence label is how much its observed outcome varies across its resolution records. Each record's outcome is summarised by the effect-profile statistics the embedding probes use. The label is the spread of those statistics across the text's records. In the probe fit, each text is weighted by n/(n+5), where n is its number of resolutions, so a text seen once counts for little.

"Draw a card" scores near zero. A sweeper, whose deaths follow the size of the opposing board, scores high. A probe that reads this label from `e` shows whether training against the board left a trace in the vector.

### Probe items are keyed by provenance, so one frozen probe set serves every checkpoint trained on a corpus

Every probe item is keyed by its provenance, never by its text. Gen-2's whole-chain encoding changes every line's text, so a probe item keyed by text in gen-1 would match nothing in gen-2. Its provenance key still names the same trait. The provenance key `(script_file, face, trait_kind, index_within_kind)` names the Forge script, the card face, the kind of trait and the trait's position among its kind that a line was rendered from. A provenance sidecar is the file `convert` writes beside each converted card, mapping each provenance key to its rendered line.

Four parts of the suite let it compare checkpoints.

- **A model adapter.** One loader takes any checkpoint and resolves its vocabulary, its cache and its width of `e`, so no probe assumes gen-1's paths or its 64 dimensions.
- **A frozen probe set with a digest.** The probe items, their labels and the games they come from are enumerated once per corpus and hashed. Every checkpoint trained on that corpus is probed on the same set, and the scorecard records the digest.
- **Probe games no later corpus build trains on.** The card-disjoint games name a held-out card, so no build ever trains on them. Gen-2's `build-corpus` places a game in the game-disjoint stratum by a rule that depends only on the game, so a rebuild over a grown corpus keeps every game-disjoint probe game out of training as well.
- **A scorecard comparison.** One command reads several scorecards and sets them side by side, per family and per rung.

Gen-1 and gen-2 are probed on different corpora, because gen-2's corpus is collected anew. Their scorecards compare by direction rather than figure by figure.

### The model takes `e` as a separate input, so methods A and B need no model change

`EffectModel.forward` in `src/effects/domain/effect_model.py` takes `e_vectors` as an argument of its own and returns the trunk's output at every slot. A replacement for `e` is a different tensor passed in that argument. Rungs 2 and 3 read the returned outputs and the heads applied to them. The per-layer view would read each layer's output through a forward hook, a callback PyTorch runs on a layer's output each time the model computes a prediction, which also leaves the model untouched.

### The build is a probe script directory beside the embedding probes

The suite is analysis tooling like the other probe scripts. Its normative description is the probe-tooling section of [`../specs/2026-10-06-ability-effect-model-gen2.md`](../specs/2026-10-06-ability-effect-model-gen2.md).

- **Code.** A new `scripts/effect_knowledge_probes/` directory sits beside `scripts/effect_embedding_probes/` and reuses its loaders. Its modules are `labels`, `ladder`, `sweeps`, `ablation`, `compare` and `run`. `run` probes one checkpoint, with method C behind a flag. `compare` reads several scorecards.
- **Probe records.** They come from both validation strata of the corpus the checkpoint trained on. Every board-dependent result is reported for the two strata separately. Line-level probes read every line of the checkpoint's cache, and report the held-out and the trained lines separately too.
- **Output.** Tables and one JSON scorecard per checkpoint go to `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`.
- **Tests.** Unit tests cover the share calculation and its minimum-gap rule, the label extractors, the board-sweep editor and the fold assignment, which must never put one group in two folds. Fixtures are built from real records.
- **Cost.** A run takes under twenty minutes per checkpoint on the same 8 GB GPU the training runs use, and runs are queued between trainings rather than beside them.

### Run plan: freeze the probe set, run the cheap probes first, then compare

The suite runs once per checkpoint, after the checkpoint's cache exists. Gen-1 is probed first, as soon as the suite is built, and each gen-2 arm after its training finishes.

#### Before the first run on a corpus

1. **Freeze the probe set.** Enumerate the probe items by provenance key, their labels, the probe games of both strata and the board-sweep records, and write the digest. Gen-1's set is built over gen-1's corpus, against gen-1's provenance sidecars, as the gen-2 record's stage 0 describes. Gen-2's set is built once over gen-2's final corpus and serves every arm.
2. **Measure the interaction join rate.** It is a property of the corpus, not of a checkpoint, so it is measured once per probe set.

#### For each checkpoint, in order

The order puts the probes that read only the cache first. A broken cache or a wrong adapter then shows before any GPU time is spent on the trunk.

1. **Probes that read `e` alone.** Every line property from the cache, with its width control. No GPU is needed.
2. **The state-dependence probe**, also from the cache.
3. **The probes that read validation records.** The trunk's output at `[ACT]` for every line property, then rungs 0, 1, 1w, 1o, 2 and 3 for every board-dependent target, on both strata.
4. **The four board sweeps**: toughness, affordability, board size and damage.
5. **Method B**, the ablation, over the validation samples.
6. **The per-layer view**, on the families with a low share, where gen-3's trunk sizing needs the depth at which the fact appears.

Method C stays off unless a family's results call for it.

#### After the last arm

1. **`compare` over every scorecard of the sweep**, ranking arms on rung 1 minus rung 1w per family. Beside it sit the memorization gap, how much better an arm does on texts it trained on than on unseen ones, and the other results the gen-2 record lists.
2. **The Results section records the comparison**, and reads gen-1 against gen-2 by direction.

### Four grids added after the first runs probe single rule families on edited records

Four probes were added after the first runs, each aimed at one rule family. Each edits real validation records and reads the full model, as the board sweeps do. The evasion grid grants keywords and blocker traits. The targeting probe scores real targeting decisions and breaks a restriction's condition. The colour probe sets a player's lands to configurations whose affordability the rule decides. The zone probe scores card counts and destinations, sweeps a script's number of cards and hands a player target over. Each is described with its results above. The gen-2 record's damage and combat grids follow the same pattern for the death rule and combat.

### Method C, the per-layer view and most of the interaction family have no results here

Method C never ran, and neither did the per-layer view. Interactions are reported only through the interaction join rate in the details and the one pair probe in the [e over a random vector](#e-adds-far-more-than-a-random-vector-of-its-width-on-what-a-line-states-and-almost-nothing-on-what-the-board-decides) chart. The linear probes' scores are in the data and not discussed. The observed-mana target counted mana events rather than mana and scored no rung, so no conclusion rests on it ([the gen-2 record](2026-09-18-effect-model-gen2-improvements-design.md#the-observed-mana-probe-target-counted-mana-events-rather-than-mana)).

## Details

### Each probe set is frozen once per corpus, and a probing run takes under twenty minutes

Every gen-2 checkpoint is probed on one probe set, frozen over gen-2's corpus with every amount label capped at 40 (digest `ab6452679131`). The first run of arm `e64` used the set before the cap (digest `927c88b0fae4`), and gen-1 a set frozen over gen-1's corpus (digest `361fc75ce1cc`). Each scorecard is `reports/knowledge-probes-*/scorecard.json` in the checkpoint's run directory under `models/effects/runs/`, and gen-1's is `output/effects/reports/knowledge-probes-2026-09-17-full-textless-corpus-20261009/scorecard.json`. The gen-2 record's [runs table](2026-09-18-effect-model-gen2-improvements-design.md#the-runs-and-the-checkpoints) lists every run and checkpoint.

A probing run takes 14 to 19 minutes and under 0.6 GiB of GPU memory, against a budget of two hours and 8 GB, and freezing a probe set a few minutes more. The board-abilities subsets of arm `e64` ran separately, in `models/effects/runs/2026-10-gen2-e64/reports/board-run/`.

| probe set | line items | record items | interaction pairs | interaction join rate |
|---|---:|---:|---:|---:|
| gen-1's corpus | 34,802 | 33,329 | 1,792 | 6.8% |
| gen-2's corpus | 36,178 | 35,699 | 2,010 | 7.0% |

The interaction join pairs a fired trigger with the resolution record whose events contain its pending event. Most fired triggers are enters-the-battlefield or cast triggers, whose cause is a permanent spell's own cast, which has no ability line and no resolution record, so the join rate stays low. The script-mined pairs supply the interactions the join cannot reach.

### The first run, on gen-1, found two gaps that gen-2 closed (2026-10-08)

**Gen-1's verdict head was never trained.** Gen-1's trainer never called the verdict loss, so the two questions that head serves, whether a candidate is affordable and whether a trigger fires, had no model read-out. A head with random weights can still score far from chance, because the trunk's output at `[ACT]` is dominated by one direction that on its own separates affordable candidates. Gen-2 trains the verdict head, and its read-outs match the trunk's.

| read-out of gen-1's trunk at `[ACT]`, affordability | AUC |
|---|---:|
| top principal component (47.5% of the variance) | 0.877 |
| gen-1's untrained verdict head | 0.899 |
| 200 fresh verdict heads, median distance from 0.5 | 0.187 |
| 200 fresh verdict heads, share at least 0.4 from 0.5 | 1.0% |

**Gen-1's corpus never recorded a death caused by a resolution.** Every `dies` label was zero, and the toughness and damage sweeps read 0.000 at every value. Gen-1's collectors wrote a resolution's record before Forge ran its state-based actions and discarded the deaths those actions produced. Gen-2's collectors hold a resolution's record through the state-based check that follows it, so gen-2's corpus carries these deaths, and gen-2 is the first generation the death sweeps can measure.

### The affordability sweep sets the lands as well as the mana (2026-10-15)

The affordability sweep first edited only the acting player's untapped production, one number per colour, and left every land on the board as it was. The verdict then stayed flat near 0.29 from no mana to more than enough, because the model reads affordability from the lands themselves. The sweep now sets the acting player's lands to match the mana it gives, and the verdict rises with it. Run `e64 (earlier inputs)` was probed before the change, on 2026-10-13, and its affordability sweep is the flat one.
