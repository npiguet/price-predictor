# Ability effect model — knowledge probes

Design record for a suite of probes on the ability effect model described in the model design record, [`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md). An ability encoder turns each ability line into one vector `e`. `e` depends only on the line's text, so lines that read the same, on whatever card, share one `e`. An effect head then predicts what an ability does to the board. The effect head has two parts. The first part, the trunk, is a stack of transformer layers. Each layer recomputes every slot's vector from the vectors of all the slots in a sequence. The sequence holds one slot for the game, one per player, an `[ACT]` slot holding the acting ability's `e`, and, for each object on the board, a `[CARD]` slot followed by that object's ability vectors. The second part, the output heads, is a set of small layers. They read the trunk's output at those slots and predict each output field, such as whether an entity dies or how much damage it takes.

Gen-1 is the checkpoint of run `09.17f`, the model weights that run saved. The model design record's Outcome section reports its results. Gen-2 is a sweep: a set of training runs, called arms, that differ in the width of `e` and the size of the encoder, described in the gen-2 record, [`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md). The suite runs on gen-1 and on every arm.

Each corpus withholds two validation strata from training. The game-disjoint stratum holds whole games whose ability texts the model trained on. The card-disjoint stratum holds games that contain a held-out card, whose ability texts the model never saw.

A probe is a small model fitted to predict a known label from a model's internal vector. How well it predicts measures how readably the vector holds that label. The existing embedding probes in `scripts/effect_embedding_probes/` read `e` alone. They label each text from its script or from its effect profile: the shares of its resolutions in which affected entities died, changed zone, took damage and so on. The probes designed here also read the trunk, and they compare the two.

The suite has three methods and a set of board sweeps. Method A, the read-out ladder, fits probes to inputs taken from progressively deeper in the model. Method B, an ablation, replaces `e` in the frozen model and measures how much the loss rises. A frozen model's weights stay as trained and are not updated. The loss is the prediction error the model was trained to minimise, measured per output field. Method C trains a shallow trunk on frozen `e`. Board sweeps edit one input of a real record and read the full model's prediction.

## The probes measure what game knowledge each model holds, and how much of it lives in `e` rather than in the trunk

The suite has three goals. The first is to find game knowledge the model encodes beyond what the existing embedding probes show. The second is to measure how much of that knowledge lives in `e` and how much in the trunk. Gen-3 aims to hold as much of the knowledge as possible in `e`, so this measurement shapes gen-3's architecture. The third is to compare the arms of gen-2's sweep.

Knowledge that lives in `e` survives into the game agent, and knowledge that lives only in the trunk may not. The game agent ([`2026-09-12-game-playing-agent-design.md`](2026-09-12-game-playing-agent-design.md)) reuses the ability encoder through its cache, which stores one precomputed `e` per ability line. It may retrain the trunk or replace it. Whatever the trunk re-derives from context every time it computes a prediction is then lost to the agent, which has to learn it again from its own much weaker training signal.

One vector per ability is also easier to interpret. It can be inspected, compared with another ability's vector and reused by any consumer without running a board through the trunk.

## The probes cover what an ability does to the board, in ten families, and leave strategic value to the game agent

The encoder's job is to capture an ability's immediate effect on the board. Strategic value, such as a card's win rate or its price, is out of scope for two reasons. The training corpus does not contain it, so the model has no signal to learn it from. Learning it is the game agent's job. The gen-2 record's run plan checks strategic value separately, with the decodability battery. The battery is a set of probes that read the sealed pipeline's per-card win rates from each card's pooled `e`, which is the mean and the maximum of the card's ability vectors placed side by side.

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
- **State dependence** tests the claim the model design rests on: that training the encoder against the board yields a better `e` than training it on text alone. The model design record's `no-state` baseline was meant to test that claim by reading the acting ability's text without the board. It zeroed the board and every ability vector, the acting one included. Its encoder therefore never trained, and the claim remains untested.

## The read-out ladder measures the vector's share of what the model knows, and is the headline for gen-3 (method A)

The ladder reads one target from inputs of increasing depth, and each input is a rung. Rungs 0 to 2 fit the same two probes to their input: a linear probe, and a small MLP (multi-layer perceptron), a network of linear layers with a non-linear function between them, which can read a relationship a linear probe misses. Both are fixed as the probe-architecture subsection below describes. Rung 3 is the model's own prediction. Two further rungs are controls on rung 1. For a target that depends on the board, the rungs are:

| Rung | Input to the probe |
|---|---|
| 0 | the raw features the trunk receives, with every `e` zeroed |
| 1 | rung 0 plus the acting ability's `e` and, for a target about one entity, the pooled `e` of that entity's own abilities, with no trunk |
| 1w, width control | rung 1 with each `e` replaced by a fixed random vector per text, of the same width |
| 1o, oracle | rung 0 plus the true values the target depends on, parsed from the script, such as the damage amount or the cost by colour |
| 2 | the trunk's output at the relevant slot: `[ACT]` for a target about the ability, the target's `[CARD]` slot for a target about one entity |
| 3 | the model's own head prediction, with no probe |

Rung 1 includes the target entity's own ability vectors because an outcome such as "this creature dies" can turn on that creature's abilities, indestructible for one.

Every rung is scored so that higher is better. A yes/no target is scored by AUC, the probability that a random positive item scores above a random negative one. An amount is scored by R². Rung 0 is what the board alone tells a probe. Rung 3 is what the whole model knows.

The headline for gen-3 is the vector's share, (rung 1 − rung 0) / (rung 3 − rung 0), reported with a bootstrap confidence interval for each of the two probes. It is the fraction of the gap between the board alone and the whole model that the ability vectors close without any trunk. A share near 1 means `e` carries the knowledge, and a share near 0 means the trunk supplies it.

A property of a line that does not depend on the board, such as the damage it deals or its mana cost by colour, gets a short ladder: `e` alone, its width control, and the trunk's output at `[ACT]`. A line property a probe reads much better from the trunk's output than from `e` is one the trunk re-derives from context, and one `e` could have held.

An optional per-layer view probes the output of each trunk layer in turn. It shows at what depth a fact appears, which feeds the sizing of gen-3's trunk.

### Every probe is fitted on folds split by text, so no probe can look up a text's label

The cross-validation folds never put one text on both sides. Every probe is fitted on all folds but one and scored on the fold left out, in turn. `e` is a fixed function of the text. A probe fitted on some records of a text and scored on other records of the same text could read the label off which text it is, and that lookup would inflate rung 1. Where a probe item pools the lines of one card, the folds are grouped by card, as the embedding probes already group them. Rungs 0, 2 and 3 use the same folds, so the rungs of one target are comparable.

### The share is reported only where the model knows more than the board

The share is computed only where rung 3 exceeds rung 0 by a minimum gap, fixed before the first run. Below that gap the denominator is near zero and the ratio means nothing. Gen-1 already has such fields. The state-only baseline is the trained-model counterpart of rung 0. Reading the text gains nothing over it on the two blocker fields, and one percentage point on damage taken. The raw rungs are printed for every target either way.

A share above 1 is possible. The probe is fitted to this one target alone, while the model's heads are trained on every output field together. The probe can therefore extract the target from `e` better than the heads do. A share above 1 says that `e` holds the target at least as readably as the model applies it.

### The oracle rung says whether a low share is the vector's fault or the probe's

A low rung 1 has two causes. Either `e` lacks the knowledge, or the probe cannot combine `e` with the board. The second is likely for the thresholds, where the probe must compare an amount with a toughness. Rung 1o gives the probe the true amount instead of `e`. If rung 1o comes close to rung 3, the probe can do the comparison, and a low rung 1 means `e` lacks the amount. If rung 1o stays well below rung 3, the probe is the limit, and the share on that target is not interpreted.

### The width control keeps arms of different widths comparable

A probe that reads a wider vector has more inputs, and it scores higher on that account alone. Rung 1w gives the probe a vector of the same width with no content, a fixed random vector per text. Under folds split by text such a vector carries nothing from one text to another, so whatever it adds over rung 0 is what width alone buys.

The arms of gen-2's sweep are compared on rung 1 minus rung 1w, not on the share. The share's denominator is each arm's own rung 3, so an arm with a weaker head shows a larger share. The share answers where knowledge lives, for gen-3's design.

### Both probes are fixed across every checkpoint, and the MLP's share is the headline

Rungs 0 to 2 use a fixed pair of probes: a linear probe and a two-layer MLP of fixed width. Its hyperparameters, the settings chosen before fitting such as its width, learning rate and regularisation strength, are fixed once for every checkpoint. A difference in a rung between two checkpoints then reflects the models and not the probe.

Every rung reports both probes' scores, and the share and the arm comparison are computed once per probe type. One computation never mixes the two types across rungs, because a ratio of a linear score to an MLP score would compare two different instruments. The MLP's share is the headline: the game agent reads `e` through non-linear layers, so the MLP's score is the closer measure of what it can use. The linear share shows whether a fact sits in `e` in a form a linear read-out finds.

## Replacing `e` in the frozen trunk measures how much the trunk relies on it, and supports the ladder (method B)

Method B runs each frozen model over the validation samples and reports each output field's loss increase when `e` is replaced by one of three substitutes.

| Replacement for `e` | What the loss increase measures |
|---|---|
| noise matched to the vectors' mean and covariance | whether the trunk relies on `e` at all |
| the mean `e` of the line's API type, the first word of its Forge script, which names the effect the line runs | whether it relies on `e` beyond the API type |
| the `e` of the other text whose vector lies closest | how sensitive it is to fine differences between texts |

The same replacements are also applied to the `[ACT]` slot alone and to the card slots alone. The comparison shows whether the trunk leans on the acting ability's vector or on the vectors of the abilities around it.

Ablation measures reliance, not location. A trunk can lean on `e` and still do the work itself: it may need `e` to know which ability it is looking at and then compute everything else from the board. That is why the ladder is the headline and the ablation supports it.

## The share and the ablation together say which families gen-3 must move into `e`

Each pattern of share, reliance and oracle rung points to one next step: a change to gen-3's architecture, more corpus records, or a stronger probe.

- A family with a low share and high reliance is one where the trunk does work `e` could carry. Gen-3 then moves that work into `e` in one of two ways. It can add auxiliary heads, extra output layers that predict the family's labels from `e` alone, so that only `e` can satisfy their training signal. Or it can use a thinner trunk that cannot re-derive the knowledge itself.
- A family that no rung predicts well is knowledge the model lacks altogether. That is a corpus problem, not an architecture problem, and the fix is records that show the effect.
- A family whose oracle rung stays below rung 3 needs a stronger probe before its share says anything.

No cut-off for a low or a high share exists yet, so the first runs' shares are compared family against family rather than against a fixed threshold.

## Board sweeps edit one input of a real record and read where the full model's prediction steps

A threshold that depends on the board is probed by a board sweep. The board sweep takes a real record, edits one input over a range of values, and reads the full model's prediction at each value. A model that knows the threshold shows a step at the right value.

- **Toughness sweep.** A damage ability sits in the acting slot. One target creature's toughness runs from 1 to 8, and the sweep reads the predicted death probability for that creature. A model that knows Lightning Bolt shows a step between toughness 3 and toughness 4.
- **Affordability sweep.** An ability sits in the acting slot. The acting player's untapped production runs from none to more than enough, and the sweep reads the predicted affordable verdict. The verdict should step at exactly the cost: `{2}{R}` needs three mana, one of them red.
- **Board-size sweep.** A sweeper sits in the acting slot. The number of opposing creatures runs from 0 to 8, by copying a creature entity of the record's board, and the sweep reads the predicted deaths summed over the board. They should rise one for one. This is the model design record's scaling calibration, which no run has yet measured.
- **Damage sweep.** The board stays fixed with a target of toughness 4. The acting text is one damage spell's script with `NumDmg$` set from 1 to 8, and the checkpoint's own encoder encodes each version. The predicted death should step between 3 and 4. The toughness sweep moves the board, and the damage sweep moves the text. The damage sweep therefore tests whether `e` carries the amount in a form the trunk can compare with toughness. Run on a triggered damage ability as well, it also tests gen-2's whole-chain encoding, which encodes every script line an ability owns as one text. On a triggered ability the damage amount sits in a sub-ability's script line, and only whole-chain encoding brings that line to the encoder.

## A shallow trunk trained on frozen `e` imitates the game agent's reuse of the encoder, and stays off by default (method C)

Method C trains a deliberately shallow trunk on frozen `e`, with one transformer layer or none, and compares it with the full model. It is the closest imitation of the game agent reusing the encoder with a trunk of its own.

It costs several GPU hours per checkpoint, against one to two for the rest of the suite. Two training runs of the same shallow trunk also differ by chance, so its result compares less well between checkpoints than the ladder's probes, whose architecture and settings are fixed across checkpoints. It therefore runs only behind a flag, when a family's results call for it.

## Four families need labels assembled from several corpus records or fields

### Interaction labels come from joining triggers to resolutions, because no trigger record names the ability behind its event

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

### Playability decision records carry everything affordability and target legality need

A decision record is enough to recompute whether a candidate ability was affordable. Each candidate carries its `affordable` flag and its `cost_after_adjustment`, and each player carries `untapped_production` and `floating_mana` by colour. The mana-usage family reads both its labels and the affordability sweep from these fields. The same records carry each candidate's legal-target set, which labels the target-legality family.

### State dependence is labelled from the spread of each text's own outcomes

A text's state-dependence label is how much its observed outcome varies across its resolution records. Each record's outcome is summarised by the effect-profile statistics the embedding probes use. The label is the spread of those statistics across the text's records. In the probe fit, each text is weighted by n/(n+5), where n is its number of resolutions, so a text seen once counts for little.

"Draw a card" scores near zero. A sweeper, whose deaths follow the size of the opposing board, scores high. A probe that reads this label from `e` shows whether training against the board left a trace in the vector.

## Probe items are keyed by provenance, so one frozen probe set serves every checkpoint trained on a corpus

Every probe item is keyed by its provenance, never by its text. Gen-2's whole-chain encoding changes every line's text, so a probe item keyed by text in gen-1 would match nothing in gen-2. Its provenance key still names the same trait. The provenance key `(script_file, face, trait_kind, index_within_kind)` names the Forge script, the card face, the kind of trait and the trait's position among its kind that a line was rendered from. A provenance sidecar is the file `convert` writes beside each converted card, mapping each provenance key to its rendered line.

Four parts of the suite let it compare checkpoints.

- **A model adapter.** One loader takes any checkpoint and resolves its vocabulary, its cache and its width of `e`, so no probe assumes gen-1's paths or its 64 dimensions.
- **A frozen probe set with a digest.** The probe items, their labels and the games they come from are enumerated once per corpus and hashed. Every checkpoint trained on that corpus is probed on the same set, and the scorecard records the digest.
- **Probe games no later corpus build trains on.** The card-disjoint games name a held-out card, so no build ever trains on them. Gen-2's `build-corpus` places a game in the game-disjoint stratum by a rule that depends only on the game, so a rebuild over a grown corpus keeps every game-disjoint probe game out of training as well.
- **A scorecard comparison.** One command reads several scorecards and sets them side by side, per family and per rung.

Gen-1 and gen-2 are probed on different corpora, because gen-2's corpus is collected anew. Their scorecards compare by direction rather than figure by figure.

## The model takes `e` as a separate input, so methods A and B need no model change

`EffectModel.forward` in `src/effects/domain/effect_model.py` takes `e_vectors` as an argument of its own and returns the trunk's output at every slot. A replacement for `e` is a different tensor passed in that argument. Rungs 2 and 3 read the returned outputs and the heads applied to them. The per-layer view reads each layer's output through a forward hook, a callback PyTorch runs on a layer's output each time the model computes a prediction, which also leaves the model untouched.

## The build is a probe script directory beside the embedding probes

The suite is analysis tooling like the other probe scripts. Its normative description is the probe-tooling section of [`../specs/2026-10-06-ability-effect-model-gen2.md`](../specs/2026-10-06-ability-effect-model-gen2.md).

- **Code.** A new `scripts/effect_knowledge_probes/` directory sits beside `scripts/effect_embedding_probes/` and reuses its loaders. Its modules are `labels`, `ladder`, `sweeps`, `ablation`, `compare` and `run`. `run` probes one checkpoint, with method C behind a flag. `compare` reads several scorecards.
- **Probe records.** They come from both validation strata of the corpus the checkpoint trained on. Every board-dependent result is reported for the two strata separately. Line-level probes read every line of the checkpoint's cache, and report the held-out and the trained lines separately too.
- **Output.** Tables and one JSON scorecard per checkpoint go to `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`.
- **Tests.** Unit tests cover the share calculation and its minimum-gap rule, the label extractors, the board-sweep editor and the fold assignment, which must never put one text in two folds. Fixtures are built from real records.
- **Cost.** A run takes one to two GPU hours per checkpoint, on the same 8 GB GPU the training runs use, so runs are queued between trainings rather than beside them.

## Run plan: freeze the probe set, run the cheap probes first, then compare

The suite runs once per checkpoint, after the checkpoint's cache exists. Gen-1 is probed first, as soon as the suite is built, and each gen-2 arm after its training finishes.

### Before the first run on a corpus

1. **Freeze the probe set.** Enumerate the probe items by provenance key, their labels, the probe games of both strata and the board-sweep records, and write the digest. Gen-1's set is built over gen-1's corpus, against gen-1's provenance sidecars, as the gen-2 record's stage 0 describes. Gen-2's set is built once over gen-2's final corpus and serves every arm.
2. **Measure the interaction join rate.** It is a property of the corpus, not of a checkpoint, so it is measured once per probe set.

### For each checkpoint, in order

The order puts the probes that read only the cache first. A broken cache or a wrong adapter then shows before any GPU time is spent on the trunk.

1. **Probes that read `e` alone.** Every line property from the cache, with its width control. No GPU is needed.
2. **The state-dependence probe**, also from the cache.
3. **The probes that read validation records.** The trunk's output at `[ACT]` for every line property, then rungs 0, 1, 1w, 1o, 2 and 3 for every board-dependent target, on both strata.
4. **The four board sweeps**: toughness, affordability, board size and damage.
5. **Method B**, the ablation, over the validation samples.
6. **The per-layer view**, on the families with a low share, where gen-3's trunk sizing needs the depth at which the fact appears.

Method C stays off unless a family's results call for it.

### After the last arm

1. **`compare` over every scorecard of the sweep**, ranking arms on rung 1 minus rung 1w per family. Beside it sit the memorization gap, how much better an arm does on texts it trained on than on unseen ones, and the other results the gen-2 record lists.
2. **The Outcome section records the comparison**, and reads gen-1 against gen-2 by direction.

## Outcome / Result

### Gen-1 (2026-10-08)

The first run probed the gen-1 checkpoint (`models/effects/runs/2026-09-17-full-textless-corpus/latest.pt`) against a probe set frozen over gen-1's curated corpus and read through a kept-aside copy of gen-1's sidecars. The scorecard is `output/effects/reports/knowledge-probes-2026-09-17-full-textless-corpus-20261009/scorecard.json`, probe-set digest `361fc75ce1cc`. Unless a table says otherwise, scores are the MLP probe's: AUC for yes/no targets, R² for amounts.

#### The suite fits its budget with room to spare

A full probing run took 19 minutes of wall time and peaked at 0.58 GiB of GPU memory, against a budget of two hours and 8 GB. Freezing the probe set took a few minutes more and needs no GPU.

| item | value |
|---|---|
| line items | 34,802 |
| record items | 33,329 |
| interaction pairs | 1,792 |
| interaction join rate | 6.8% |
| wall time, probing run | 1,160 s |
| peak GPU memory | 0.58 GiB |

The join rate is low because the join can pair a trigger only with a resolution record that names the ability behind its event. Most fired triggers are entering-the-battlefield or cast triggers. Their cause is a permanent spell's own cast, which has no ability line, and a spell-cast event has no resolution record at all. The script-mined pairs supply the interactions the join cannot reach.

#### `e` carries what kind of ability a line is, and little of its amounts

A probe reading `e` alone recognises most line properties far above the width control, the same probe fitted on a fixed random vector per text. Amounts are the exception: `e` explains a third of the variance in damage, little of the mana cost and none of the toughness change. This is the gap the value head is meant to close.

| line property | `e` | width control | trunk at `[ACT]` |
|---|---:|---:|---:|
| lasts as a counter (`as_counter`) | 0.998 | 0.512 | 0.994 |
| lasts until end of turn | 0.990 | 0.510 | 0.987 |
| produces colourless mana | 0.996 | 0.524 | 0.992 |
| repeatable | 0.863 | 0.501 | 0.964 |
| affects each player or permanent | 0.894 | 0.497 | 0.929 |
| damage amount (R²) | 0.348 | −0.102 | 0.682 |
| power change (R²) | 0.523 | −0.326 | 0.638 |
| toughness change (R²) | −0.011 | −0.192 | 0.507 |
| total mana cost (R²) | 0.142 | −0.517 | 0.048 |

These are trained lines. On held-out lines, damage and counters placed read lower, and power change and toughness change fall to near zero.

#### Affordability lives in `e`; blocking legality does not

On the board-dependent targets, rung 0 reads the board with every `e` zeroed and rung 1 adds the acting line's `e`. Affordability rises from 0.74 to 0.95 when `e` is added, far above the width control, so the vector carries the cost the verdict turns on. Whether a creature may block barely moves between the two rungs, so `e` adds nothing the board does not already say.

| target, card-disjoint | rung 0 | rung 1 | 1w | 1o | rung 2 | rung 3 | share |
|---|---:|---:|---:|---:|---:|---:|---:|
| affordable | 0.736 | 0.953 | 0.798 | 0.965 | 0.947 | — | — |
| may block | 0.671 | 0.674 | 0.670 | 0.677 | 0.711 | 0.748 | 0.03 |
| entity affected | 0.893 | 0.919 | 0.867 | 0.899 | 0.995 | 0.997 | 0.25 |
| trigger fires | 0.781 | 0.831 | 0.799 | 0.818 | 0.851 | — | — |
| legal target | 0.929 | 0.948 | 0.907 | 0.936 | 0.980 | 0.955 | — |

Rung 1w replaces each `e` with its width control, rung 1o adds the parsed script values instead of `e`, rung 2 is the trunk's output and rung 3 the model's own prediction. Affordability and blocking show the same pattern on the game-disjoint stratum. Target legality has no share because the board alone already reads it: rung 3 exceeds rung 0 by less than the minimum gap. Its rung 3 is read from the gate, which a decision record raises on exactly its legal targets. The `target_legal` field is trained only on legal targets, always to 1, so its output carries nothing about an illegal entity.

#### Gen-1's verdict head was never trained, so affordability and trigger firing have no model read-out

Gen-1's trainer never called the verdict loss, and its verdict head sits at its initial weights. The two targets that head serves, whether a candidate is affordable and whether a trigger fires, therefore have no rung 3 and no share. The trunk does carry both: rung 2 reaches 0.95 on affordability and 0.85 on firing.

A head with random weights can still score far from 0.5, which is why rung 3 is withheld rather than read. The trunk's output at `[ACT]` is dominated by one direction, which holds nearly half its variance and on its own separates affordable candidates from unaffordable ones. A freshly initialized head lands 0.4 or more from chance about one time in a hundred, and gen-1's own head, at 0.899, sits at that edge.

| read-out of the trunk at `[ACT]`, affordability | AUC |
|---|---:|
| top principal component (47.5% of the variance) | 0.877 |
| gen-1's untrained verdict head | 0.899 |
| 200 fresh verdict heads, median distance from 0.5 | 0.187 |
| 200 fresh verdict heads, share at least 0.4 from 0.5 | 1.0% |

#### No death is ever predicted from damage, because gen-1's corpus never records one

Every `dies` label is zero, and the toughness and damage sweeps read 0.000 at every value. Gen-1's collectors wrote a resolution's record before Forge ran its state-based actions, and discarded the deaths those actions then produced. A creature killed by a burn spell therefore appears in the record with its damage and without its death, and the model has learned that damage kills nothing. On a sample of twelve real shards, none of 565 lethal-looking resolution damage events carries the death, while combat records, written after the check, carry most of theirs.

The board-size sweep shows the model does predict deaths when the record itself carries them: predicted deaths summed over the board rise steadily from 0.003 with no opposing creature to 0.154 with eight. The collectors now hold a resolution's record through the state-based check and add the deaths of the creatures its own events touched, so gen-2's corpus carries them; a twelve-match pilot keeps fourteen of fourteen.

| sweep | 1 or 0 | 4 | 8 |
|---|---:|---:|---:|
| toughness of the target, its predicted death | 0.000 | 0.000 | 0.000 |
| `NumDmg$` on a spell, toughness-4 target's death | 0.000 | 0.000 | 0.000 |
| opposing creatures, deaths summed over the board | 0.003 | 0.089 | 0.154 |

The affordability sweep reads the verdict head and is not run on gen-1.

#### The trunk reads `e` almost only at `[ACT]`, and any nearby text's `e` does as well

Replacing `e` with matched noise raises the loss most on keywords gained, the gate and damage taken. Applied to `[ACT]` alone the increase is almost the same as applied everywhere, and applied to the card slots alone it is near zero, so the trunk barely uses the vectors of the abilities on the board. Replacing `e` with the nearest other text's vector costs almost nothing on any field.

| field | base loss | noise, every slot | noise, `[ACT]` | noise, card slots | API-type mean | nearest text |
|---|---:|---:|---:|---:|---:|---:|
| gate | 1.010 | +1.190 | +1.082 | +0.044 | +0.318 | +0.017 |
| keywords gained | 0.066 | +1.122 | +1.080 | +0.020 | +0.694 | +0.015 |
| damage taken | 0.543 | +0.280 | +0.250 | +0.002 | +0.062 | +0.000 |
| life change | 0.363 | +0.295 | +0.258 | +0.003 | +0.087 | +0.006 |
| colours gained | 0.025 | +0.182 | +0.188 | −0.001 | +0.114 | +0.019 |

### Gen-2 arms

To be filled in after the gen-2 sweep, with each arm read against gen-1 by direction.
