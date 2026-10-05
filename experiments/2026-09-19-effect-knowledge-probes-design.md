# Ability effect model — knowledge probes

Design record for a suite of probes on the ability effect model. The model is the one described in [`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md): an ability encoder that turns each ability line into one vector `e`, and an effect head whose transformer layers, the trunk, read those vectors together with the board and predict what an ability does to it. Gen-1 is the checkpoint of run `09.17f` whose results that record's Outcome section reports. Gen-2 is a sweep over the width of `e` and the size of the encoder, described in [`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md). The suite runs on gen-1 and on every arm of that sweep.

A probe is a small model fitted to predict a known label from a model's internal vector. How well it predicts measures how readably the vector holds that label. The existing embedding probes in `scripts/effect_embedding_probes/` read `e` alone and label each text from its script or its observed effect profile. The probes designed here also read the trunk, and they compare the two.

## The probes measure what game knowledge each model holds, and how much of it lives in `e` rather than in the trunk

The suite has three goals. The first is to find game knowledge the model encodes beyond what the current probes show. The second is to measure how much of that knowledge lives in the ability vectors `e` and how much in the trunk, which informs gen-3's architecture: gen-3's objective is that as much of the knowledge as possible lives in `e`. The third is to compare the arms of gen-2's sweep, which differ in how wide `e` is and how large the encoder is.

Knowledge that lives in `e` survives into the game agent, and knowledge that lives only in the trunk may not. The game agent ([`2026-09-12-game-playing-agent-design.md`](2026-09-12-game-playing-agent-design.md)) swaps the effect encoder in through the one-vector-per-ability-line cache interface. It may retrain the trunk or replace it. Whatever the effect trunk re-derives from context on every forward pass is then lost to the agent, which has to learn it again from its own much weaker training signal.

One vector per ability is also easier to interpret. It can be inspected, compared with another ability's vector and reused by any consumer without running a board through the trunk.

## The probes cover what an ability does to the board, in ten families, and leave strategic value to the game agent

The encoder's job is an ability's immediate effect on the board. Strategic value, such as a card's win rate or its price, is out of scope for two reasons. The training corpus does not contain it, so the model has no signal to learn it from. And learning it is the game agent's job. The decodability battery, which reads the sealed pipeline's per-card win rates from pooled `e`, runs beside this suite as a downstream check in gen-2's run plan.

The ten families below are what the probes target. Each label comes from a source that exists without the model: the parsed script, or an outcome observed in the effect-record corpus.

| Family | Targets | Label sources |
|---|---|---|
| Magnitudes and thresholds | damage, power and toughness change, counter and life amounts; whether the effect kills a creature of toughness *t*; each amount per mana of cost | parsed script parameters; observed outcomes in resolution records |
| Mana production | colours produced, amount, any colour, spending restrictions, conditional production | the script's mana parameters; the `mana_produced` events of mana resolution records |
| Mana usage | the ability's own mana cost; cost reducers and taxes on other spells; whether it is affordable on a given board | the script's cost; playability decision records |
| Timing and non-mana costs | instant speed, a tap cost, a sacrifice or life cost | the script |
| Interactions | whether ability A's event fires ability B's trigger, over pairs of lines | trigger records joined to resolution records |
| Side | whose permanents and resources the effect changes: its controller's, the opponents', or both | the side of each affected entity in resolution records; the script's target and player restrictions |
| Evasion and blocking | which defending creatures may legally block a given attacker | legality records of the `blockers` subkind |
| Target legality | whether an ability may target a given permanent that has hexproof, shroud, protection or ward | the legal-target sets of playability decision records |
| Duration and repeatability | whether a change lasts until end of turn, stays, or comes as a counter; whether the ability can be used again | the events' duration field; the line kind and whether the cost taps or sacrifices |
| State dependence | how much a text's outcome varies across the boards it resolved on | the spread of each text's observed outcomes across its resolution records |

The embedding probes already show that the encoder keeps outcome parameters and loses amounts and costs. The first five families extend that to thresholds, to mana and to pairs of abilities, and they ask the same questions of the trunk.

The last five families each target a property the first five leave out.

- **Side** is the property that separates removal from a pump spell and a sweeper from an anthem. Gen-1's fifth principal direction already follows the opponent's share of affected entities, and no probe targets it.
- **Evasion and blocking** is where gen-1 learned least. Its two blocker fields sit exactly at the floor of the baseline that reads no ability text, so the text taught the model nothing about who may block whom.
- **Target legality** decides whether a removal spell can be aimed at all.
- **Duration and repeatability** separate abilities whose immediate effect is identical: a +1/+1 counter and a pump until end of turn, or a repeatable activation and a one-shot spell.
- **State dependence** measures the claim the design rests on. The design record's `no-state` baseline was meant to test it and never trained its encoder.

### Interaction labels come from joining triggers to resolutions, because no trigger record names the ability behind its event

No trigger record names the ability that caused its event. The 1,000 game-disjoint games of gen-1's corpus hold 15,335 trigger records, and 8,377 of them fired. 7,752 name a `cause` entity in their event's params, but `cause` is the object that caused the event, not the ability. None sets `attributed_to`.

The interaction label is therefore built by a join. A fired trigger record carries its pending event. The resolution record in the same game whose events contain that event names the ability that produced it. That ability and the trigger's line form a positive pair. The join rate is measured on the first build.

Pairs the join cannot reach are mined from the scripts. A death trigger paired with a sacrifice outlet, or a lifegain trigger paired with a lifegain ability, is a positive pair by construction. Trigger records that were evaluated and did not fire are the negatives, and they occur naturally in the corpus because the collector records non-fired evaluations of the same event type.

### Playability decision records carry everything affordability and target legality need

A decision record is enough to recompute whether a candidate ability was affordable. Each candidate carries its `affordable` flag and its `cost_after_adjustment`, and each player carries `untapped_production` and `floating_mana` by colour. The mana-usage family reads both its labels and its board sweep from these fields. The same records carry each candidate's legal-target set, which labels the target-legality family.

### State dependence is labelled from the spread of each text's own outcomes

A text's state-dependence label is how much its observed outcome varies across its resolution records. The outcome of each record is summarised as the same profile statistics the embedding probes use: the share of affected entities that died, changed zone, took damage and so on. The label is their spread across the text's records, weighted by n/(n+5) where n is the text's number of resolutions, so that a text seen once counts for little. "Draw a card" scores near zero. A sweeper, whose deaths follow the size of the opposing board, scores high. A probe that reads this label from `e` shows whether training against the board left a trace in the vector.

## Board sweeps edit one input of a real record and read where the full model's prediction steps

A threshold that depends on the board is probed by a sweep. The sweep takes a real record, edits one input over a range of values, and reads the full model's prediction at each value. A model that knows the threshold shows a step at the right value.

- **Toughness sweep.** A damage ability sits in the acting slot. One target creature's toughness runs from 1 to 8, and the probe reads the predicted death probability for that creature. A model that knows Lightning Bolt shows a step between toughness 3 and toughness 4.
- **Affordability sweep.** An ability sits in the acting slot. The acting player's untapped production runs from none to more than enough, and the probe reads the predicted affordable verdict. The verdict should step at exactly the cost: `{2}{R}` needs three mana, one of them red.
- **Board-size sweep.** A sweeper sits in the acting slot. The number of opposing creatures runs from 0 to 8, by copying a creature entity of the snapshot, and the probe reads the predicted deaths summed over the board. They should rise one for one. This is the design record's scaling calibration, which no run has yet measured.
- **Damage sweep.** The board stays fixed with a target of toughness 4, and the acting text is one damage spell's script with `NumDmg$` set from 1 to 8, each encoded by the checkpoint's own encoder. The predicted death should step between 3 and 4. The toughness sweep moves the board; this sweep moves the text, so it tests whether `e` carries the amount in a form the trunk compares against toughness. On a trigger, the amount reaches the encoder only through gen-2's whole-chain encoding, so the sweep also shows whether that change works.

The vector-only rungs of the ladder below probe the same thresholds as numbers read from `e`: the damage amount a line deals, the mana it costs by colour.

## The read-out ladder measures the vector's share of what the model knows, and is the headline for gen-3 (method A)

The ladder reads one target from inputs of increasing depth, and each input is a rung. Rungs 0 to 2 fit the same small probe to their input, and rung 3 is the model's own prediction. Two further rungs are controls on rung 1. For a target that depends on the board, the rungs are:

| Rung | Input to the probe |
|---|---|
| 0 | the raw features the trunk receives, with every `e` zeroed |
| 1 | rung 0 plus the acting ability's `e` and, for a target about one entity, the pooled `e` of that entity's own abilities, with no trunk |
| 1w, width control | rung 1 with each `e` replaced by a fixed random vector per text, of the same width |
| 1o, oracle | rung 0 plus the true values the target depends on, parsed from the script, such as the damage amount or the cost by colour |
| 2 | the trunk's output at the relevant slot: `[ACT]` for a target about the ability, the target's `[CARD]` slot for a target about one entity |
| 3 | the model's own head prediction, with no probe |

Every rung is scored on one scale where higher is better, such as AUC for a yes/no target or R² for an amount. Rung 0 is what the board alone tells a probe. Rung 3 is what the whole model knows. A property of a line that does not depend on the board gets a short ladder: `e` alone, its width control, and the trunk's state at `[ACT]`. A line property the trunk decodes much better than `e` is one the trunk re-derives from context, and one `e` could have held.

### The share is reported only where the model knows more than the board

The headline for gen-3 is the vector's share, (rung 1 − rung 0) / (rung 3 − rung 0), reported with a bootstrap confidence interval. It is the fraction of the gap between the board alone and the whole model that the ability vectors close without any trunk. For a target about one entity, those are the acting ability's vector and the entity's own, because an outcome such as "this creature dies" can turn on the target's abilities, indestructible for one. A share near 1 means `e` carries the knowledge, and a share near 0 means the trunk supplies it.

The share is computed only where rung 3 exceeds rung 0 by a minimum gap, fixed before the first run. Gen-1 has fields where the text adds at most a point over the board, the two blocker fields and damage taken among them. On those the denominator is near zero and the ratio means nothing. The raw rungs are printed for every target either way.

A share above 1 is possible and has a meaning. A probe fitted to one target can read it from `e` better than the multi-task head uses it, and the share then says `e` holds the target at least as readably as the model applies it.

### The oracle rung says whether a low share is the vector's fault or the probe's

A low rung 1 has two causes. Either `e` lacks the knowledge, or the probe cannot combine `e` with the board. The second is likely for the thresholds, where the probe must compare an amount with a toughness. Rung 1o gives the probe the true amount instead of `e`. If rung 1o comes close to rung 3, the probe can do the comparison, and a low rung 1 means `e` lacks the amount. If rung 1o stays well below rung 3, the probe is the limit, and the share on that target is not interpreted.

### The width control keeps arms of different widths comparable

A probe that reads a wider vector has more inputs, and it scores higher on that account alone. Rung 1w gives the probe a vector of the same width with no content, a fixed random vector per text. Under text-disjoint folds such a vector carries nothing from one text to another, so whatever it adds over rung 0 is what width alone buys. The arms of gen-2's sweep are compared on rung 1 minus rung 1w.

The share is not used to compare arms. Its denominator is each arm's own rung 3, so an arm with a weaker head shows a larger share. The share answers where knowledge lives, for gen-3's design.

### Every probe that reads `e` is fitted on folds split by text

`e` is a fixed function of the text. A probe fitted on some records of a text and scored on other records of the same text can read the label off which text it is, and rung 1 is inflated by exactly that lookup. The folds therefore never put one text on both sides. Where a probe item pools the lines of one card, the folds are grouped by card, as the embedding probes already are. Rungs 0, 2 and 3 use the same folds, so the rungs of one target compare.

### The probe architecture is fixed across every checkpoint

Rungs 0 to 2 use a fixed pair of probes: a linear probe and a two-layer MLP of fixed width. An MLP, a multi-layer perceptron, is a small network of linear layers with a non-linear function between them, so it can read a relationship a linear probe misses. Its hyperparameters are frozen. A difference in a rung between two checkpoints then reflects the models and not the probe.

An optional per-layer view probes the output of each trunk layer in turn. It shows at what depth a fact appears, which feeds the sizing of gen-3's trunk.

## Replacing `e` in the frozen trunk measures how much the trunk relies on it, and supports the ladder (method B)

An ablation replaces part of a model's input with something uninformative and measures how much the loss rises. Method B runs each frozen model over the validation samples and reports each output field's loss increase when `e` is replaced by one of three substitutes.

| Replacement for `e` | What the loss increase measures |
|---|---|
| noise matched to the vectors' mean and covariance | whether the trunk relies on `e` at all |
| the mean `e` of the line's API type | whether it relies on `e` beyond the API type |
| the `e` of the nearest other text | how sensitive it is to fine differences between texts |

The same replacements are also applied to the `[ACT]` slot alone and to the card slots alone. The comparison shows whether the trunk leans on the acting ability's vector or on the vectors of the abilities around it.

Ablation measures reliance, not location. A trunk can lean on `e` and still do the work itself: it may need `e` to know which ability it is looking at and then compute everything else from the board. That is why the ladder is the headline and the ablation supports it.

Vectors from another model cannot serve as a fourth replacement. They live in a different space from the one the trunk's input projection learned, so the trunk would read them as noise with a different spread.

## A shallow trunk trained on frozen `e` imitates the game agent's reuse of the encoder, and stays off by default (method C)

Method C trains a deliberately weak head on frozen `e`, with one attention layer or none, and compares it with the full trunk. It is the most faithful imitation of the game agent reusing the encoder with a trunk of its own.

It is optional and off by default. It costs several GPU hours per checkpoint, against one to two for the rest of the suite. And two training runs of the same head differ by chance, so its result is less comparable between checkpoints than the ladder's frozen probes.

## The model's forward pass takes `e` as a separate input, so methods A and B need no model change

`EffectModel.forward` in `src/effects/domain/effect_model.py` takes `e_vectors` as an argument of its own and returns the trunk's hidden state at every slot. A replacement for `e` is a different tensor passed in that argument. Rungs 2 and 3 read the returned states and the heads applied to them. The per-layer view reads each layer's output through a forward hook on the trunk's layers, which also leaves the model untouched.

## The suite runs on any checkpoint from its first build, because gen-2 is a sweep

The durable part of the design is its concepts:

- the ten families;
- the ladder, its controls and its headline share;
- the ablation;
- the fixed probe architecture;
- keying every probe item by provenance, `(script_file, face, trait_kind, index_within_kind)`, never by text.

The last rule exists because gen-2 encodes the whole script chain, which changes every line's text. A probe item keyed by text in gen-1 would match nothing in gen-2, while its provenance key still names the same trait.

Comparing the arms of a sweep is comparing checkpoints, so the suite is built for it from the start. Four pieces make that possible.

- **A model adapter.** One loader takes any checkpoint and resolves its vocabulary, its cache and its width of `e`, so no probe assumes gen-1's paths or its 64 dimensions.
- **A frozen probe set with a digest.** The probe items, their labels and the games they come from are enumerated once per corpus and hashed. Every checkpoint trained on that corpus is probed on the same set, and the scorecard records the digest.
- **Probe games that later builds keep.** The probe records come from the corpus's game-disjoint stratum. Gen-2's `build-corpus` places a game in that stratum by a rule that depends only on the game, so a rebuild over a grown corpus keeps every probe game out of training.
- **A scorecard comparison.** One command reads several scorecards and sets them side by side, per family and per rung.

Gen-1 and gen-2 are probed on different corpora, because gen-2's corpus is collected anew. Their scorecards compare by direction rather than figure by figure.

## The build is a probe script directory beside the embedding probes

The suite is analysis tooling like the other probe scripts, so it has no root spec and no spec-kit directory.

- **Code.** A new `scripts/effect_knowledge_probes/` directory sits beside `scripts/effect_embedding_probes/` and reuses its loaders. Its modules are `labels`, `ladder`, `sweeps`, `ablation`, `compare` and `run`. `run` probes one checkpoint, with method C behind a flag. `compare` reads several scorecards.
- **Probe records.** They come from both validation strata of the corpus the checkpoint trained on. The game-disjoint stratum holds texts the model trained on, and the card-disjoint stratum holds texts it never saw. Every board-dependent result is reported for the two separately. Line-level probes read every line of the checkpoint's cache, and report the held-out and the trained lines separately too.
- **Output.** Tables and one JSON scorecard per checkpoint go to `output/effects/reports/knowledge-probes-<checkpoint>-<date>/`. The scorecard records the probe set's digest.
- **Tests.** Unit tests cover the share calculation and its minimum-gap rule, the label extractors, the sweep editor and the fold assignment, which must never put one text in two folds. Fixtures are built from real records.
- **Cost.** A run takes one to two GPU hours per checkpoint, on the same 8 GB GPU the training runs use, so runs are queued between trainings rather than beside them.

## The share and the ablation together say where gen-3 puts its pressure

No thresholds exist yet, so the first runs' figures are read by direction rather than against a cut-off.

- A family with a low share and high reliance is one where the trunk does work `e` could carry. Gen-3 then puts pressure on `e`, for example with auxiliary heads that read straight from `e`, or with a thinner trunk that cannot re-derive the knowledge itself.
- A family that no rung decodes is knowledge the model lacks altogether. That is a corpus problem, not an architecture problem, and the fix is records that show the effect.
- A family whose oracle rung stays below rung 3 needs a stronger probe before its share says anything.

## Run plan: freeze the probe set, run the cheap probes first, then compare

The suite runs once per checkpoint, after the checkpoint's cache exists. Gen-1 is probed first, as soon as the suite is built, and each gen-2 arm after its training finishes.

### Before the first run on a corpus

1. **Freeze the probe set.** Enumerate the probe items by provenance key, their labels, the probe games of both strata and the sweep records, and write the digest. Gen-1's set is built over gen-1's corpus, against gen-1's sidecars, as the gen-2 record's stage 0 describes. Gen-2's set is built once over gen-2's final corpus and serves every arm.
2. **Measure the interaction join rate.** It is a property of the corpus, not of a checkpoint, so it is measured once per probe set.

### For each checkpoint, in order

The order puts the probes that read only the cache first. A broken cache or a wrong adapter then shows before any GPU time is spent on the trunk.

1. **Probes that read `e` alone.** Every line property from the cache, with its width control. No GPU is needed.
2. **The state-dependence probe**, also from the cache.
3. **The probes that run the trunk.** The trunk's `[ACT]` state for every line property, then rungs 0, 1, 1w, 1o, 2 and 3 for every board-dependent target, on both strata.
4. **The four sweeps**: toughness, affordability, board size and damage.
5. **Method B**, the ablation, over the validation samples.
6. **The per-layer view**, where a family's share is low enough for gen-3 to need it.

Method C stays off unless a family's results call for it.

### After the last arm

1. **`compare` over every scorecard of the sweep**, ranking arms on rung 1 minus rung 1w per family, beside the memorization gap and the other results the gen-2 record lists.
2. **The Outcome section records the comparison**, and reads gen-1 against gen-2 by direction.

## Outcome / Result

To be filled in after the gen-1 run and after the gen-2 arms, with each family's ladder, controls and headline share, the ablation's loss increases per field, the four sweeps, the state-dependence probe, and the join rate of the interaction labels.
