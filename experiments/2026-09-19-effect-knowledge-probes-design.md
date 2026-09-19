# Ability effect model — knowledge probes

Design record for a suite of probes on the ability effect model. The model is the one described in [`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md): an ability encoder that turns each ability line into one vector `e`, and an effect head whose transformer layers, the trunk, read those vectors together with the board and predict what an ability does to it. Gen-1 is the checkpoint of run `09.17f` whose results that record's Outcome section reports. Gen-2's changes are in [`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md).

A probe is a small model fitted to predict a known label from a model's internal vector. How well it predicts measures how readably the vector holds that label. The existing embedding probes in `scripts/effect_embedding_probes/` read `e` alone and label each text from its script or its observed effect profile. The probes designed here also read the trunk, and they compare the two.

## The probes measure what game knowledge the model holds, and how much of it lives in `e` rather than in the trunk

The suite has two goals. The first is to find game knowledge the model encodes beyond what the current probes show. The second is to measure how much of that knowledge lives in the ability vectors `e` and how much in the trunk. The second measurement informs gen-3's architecture, whose objective is that as much of the knowledge as possible lives in `e`.

Knowledge that lives in `e` survives into the game agent, and knowledge that lives only in the trunk may not. The game agent ([`2026-09-12-game-playing-agent-design.md`](2026-09-12-game-playing-agent-design.md)) swaps the effect encoder in through the one-vector-per-ability-line cache interface. It may retrain the trunk or replace it. Whatever the effect trunk re-derives from context on every forward pass is then lost to the agent, which has to learn it again from its own much weaker training signal.

One vector per ability is also easier to interpret. It can be inspected, compared with another ability's vector and reused by any consumer without running a board through the trunk.

## The probes cover what an ability does to the board, in five families, and leave strategic value to the game agent

The encoder's job is an ability's immediate effect on the board. Strategic value, such as a card's win rate or its price, is out of scope for two reasons. The training corpus does not contain it, so the model has no signal to learn it from. And learning it is the game agent's job.

The five families below are what the probes target. Each label comes from a source that exists without the model: the parsed script, or an outcome observed in the effect-record corpus.

| Family | Targets | Label sources |
|---|---|---|
| Magnitudes and thresholds | damage, power and toughness change, counter and life amounts; whether the effect kills a creature of toughness *t* | parsed script parameters; observed outcomes in resolution records |
| Mana production | colours produced, amount, any colour, spending restrictions, conditional production | the script's mana parameters; the `mana_produced` events of mana resolution records |
| Mana usage | the ability's own mana cost; cost reducers and taxes on other spells; whether it is affordable on a given board | the script's cost; playability decision records |
| Timing and non-mana costs | instant speed, a tap cost, a sacrifice or life cost | the script |
| Interactions | whether ability A's event fires ability B's trigger, over pairs of lines | trigger records joined to resolution records |

The embedding probes already show that the encoder keeps outcome parameters and loses amounts and costs. The families here extend that to thresholds, to mana and to pairs of abilities, and they ask the same questions of the trunk.

### Interaction labels come from joining triggers to resolutions, because no trigger record names the ability behind its event

No trigger record names the ability that caused its event. The 1,000 game-disjoint games hold 15,335 trigger records, and 8,377 of them fired. 7,752 name a `cause` entity in their event's params, but `cause` is the object that caused the event, not the ability. None sets `attributed_to`.

The interaction label is therefore built by a join. A fired trigger record carries its pending event. The resolution record in the same game whose events contain that event names the ability that produced it. That ability and the trigger's line form a positive pair. The join rate is measured on the first build.

Pairs the join cannot reach are mined from the scripts. A death trigger paired with a sacrifice outlet, or a lifegain trigger paired with a lifegain ability, is a positive pair by construction. Trigger records that were evaluated and did not fire are the negatives, and they occur naturally in the corpus because the collector records non-fired evaluations of the same event type.

### Playability decision records carry everything affordability needs

A decision record is enough to recompute whether a candidate ability was affordable. Each candidate carries its `affordable` flag and its `cost_after_adjustment`, and each player carries `untapped_production` and `floating_mana` by colour. The mana-usage family reads both its labels and its board sweep from these fields.

## Board sweeps edit one field of a real snapshot and read where the full model's prediction steps

A threshold that depends on the board is probed by a sweep. The sweep takes a real snapshot, edits one field over a range of values, and reads the full model's prediction at each value. A model that knows the threshold shows a step at the right value.

- **Toughness sweep.** A damage ability sits in the acting slot. One target creature's toughness runs from 1 to 8, and the probe reads the predicted death probability for that creature. A model that knows Lightning Bolt shows a step between toughness 3 and toughness 4.
- **Affordability sweep.** An ability sits in the acting slot. The acting player's untapped production runs from none to more than enough, and the probe reads the predicted affordable verdict. The verdict should step at exactly the cost: `{2}{R}` needs three mana, one of them red.

The vector-only rungs of the ladder below probe the same thresholds as numbers read from `e`: the damage amount a line deals, the mana it costs by colour.

## The read-out ladder measures the vector's share of what the model knows, and is the headline (method A)

The ladder reads one target from four inputs of increasing depth, and each input is a rung. Rungs 0 to 2 fit the same small probe to their input, and rung 3 is the model's own prediction. For a target that depends on the board, the rungs are:

| Rung | Input to the probe |
|---|---|
| 0 | the raw features the trunk receives, with every `e` zeroed |
| 1 | rung 0 plus the acting ability's `e` and, for a target about one entity, the pooled `e` of that entity's own abilities, with no trunk |
| 2 | the trunk's output at the relevant slot: `[ACT]` for a target about the ability, the target's `[CARD]` slot for a target about one entity |
| 3 | the model's own head prediction, with no probe |

The headline is the vector's share, (rung 1 − rung 0) / (rung 3 − rung 0), reported with a bootstrap confidence interval. Every rung is scored on one scale where higher is better, such as AUC for a yes/no target or R² for an amount. Rung 0 is what the board alone tells a probe. Rung 3 is what the whole model knows. The share is the fraction of the gap between them that the ability vectors close without any trunk. For a target about one entity, those are the acting ability's vector and the entity's own, because an outcome such as "this creature dies" can turn on the target's abilities, indestructible for one. A share near 1 means `e` carries the knowledge, and a share near 0 means the trunk supplies it.

Rungs 0 to 2 use a fixed pair of probes: a linear probe and a two-layer MLP of fixed width. An MLP, a multi-layer perceptron, is a small network of linear layers with a non-linear function between them, so it can read a relationship a linear probe misses. Its hyperparameters are frozen. A change in a rung between generations then reflects the model and not the probe.

A property of a line that does not depend on the board gets a short ladder: `e` alone against the trunk's state at `[ACT]`. A line property the trunk decodes much better than `e` is one the trunk re-derives from context, and one `e` could have held.

An optional per-layer view probes the output of each trunk layer in turn. It shows at what depth a fact appears, which feeds the sizing of gen-3's trunk.

## Replacing `e` in the frozen trunk measures how much the trunk relies on it, and supports the ladder (method B)

An ablation replaces part of a model's input with something uninformative and measures how much the loss rises. Method B runs the frozen gen-1 model over the validation samples and reports each output field's loss increase when `e` is replaced by one of three substitutes.

| Replacement for `e` | What the loss increase measures |
|---|---|
| noise matched to the vectors' mean and covariance | whether the trunk relies on `e` at all |
| the mean `e` of the line's API type | whether it relies on `e` beyond the API type |
| the `e` of the nearest other text | how sensitive it is to fine differences between texts |

The same replacements are also applied to the `[ACT]` slot alone and to the card slots alone. The comparison shows whether the trunk leans on the acting ability's vector or on the vectors of the abilities around it.

Ablation measures reliance, not location. A trunk can lean on `e` and still do the work itself: it may need `e` to know which ability it is looking at and then compute everything else from the board. That is why the ladder is the headline and the ablation supports it.

The taxonomy baseline's vectors cannot be used as a fourth replacement. They live in a different space from the one the trunk's input projection learned, so the trunk would read them as noise with a different spread.

## A shallow trunk trained on frozen `e` imitates the game agent's reuse of the encoder, and stays off by default (method C)

Method C trains a deliberately weak head on frozen `e`, with one attention layer or none, and compares it with the full trunk. It is the most faithful imitation of the game agent reusing the encoder with a trunk of its own.

It is optional and off by default. It costs several GPU hours per run, against one to two for the rest of the suite. And two training runs of the same head differ by chance, so its result is less comparable between generations than the ladder's frozen probes.

## The model's forward pass takes `e` as a separate input, so methods A and B need no model change

`EffectModel.forward` in `src/effects/domain/effect_model.py` takes `e_vectors` as an argument of its own and returns the trunk's hidden state at every slot. A replacement for `e` is a different tensor passed in that argument. Rungs 2 and 3 read the returned states and the heads applied to them. The per-layer view reads each layer's output through a forward hook on the trunk's layers, which also leaves the model untouched.

## The concepts carry across generations, and the implementation targets gen-1 alone

The durable part of the design is its concepts:

- the five families;
- the ladder and its headline share;
- the ablation;
- the fixed probe architecture;
- keying every probe item by provenance, `(script_file, face, trait_kind, index_within_kind)`, never by text.

The last rule exists because gen-2 encodes the whole script chain, which changes every line's text. A probe item keyed by text in gen-1 would match nothing in gen-2, while its provenance key still names the same trait.

The implementation targets gen-1 alone and is ported when a later generation needs it. Comparing generations has known requirements that this build leaves to that port. A later corpus rebuild could put the probe games into training, and each generation holds out different texts. Comparability therefore needs a model adapter layer, a permanent reserve of probe games excluded from every future corpus build, a frozen and versioned probe set with a digest, and a tool that compares scorecards across generations. None of these is built now.

## The gen-1 build is a probe script directory beside the embedding probes

The suite is analysis tooling like the other probe scripts, so it has no root spec and no spec-kit directory.

- **Code.** A new `scripts/effect_knowledge_probes/` directory sits beside `scripts/effect_embedding_probes/` and reuses its loaders. Its modules are `labels`, `ladder`, `sweeps`, `ablation` and `run`. `run` is the single entry point, with method C behind a flag.
- **Probe records.** They come from the 1,000 game-disjoint games, which gen-1 never trained on. Line-level probes use every line, and report separately the lines whose text the checkpoint held out and the lines whose text it trained on.
- **Output.** Tables and one JSON scorecard go to `output/effects/reports/knowledge-probes-<date>/`. The scorecard is what a later generation is compared against by hand.
- **Tests.** Unit tests cover the share calculation, the label extractors and the sweep editor, with fixtures built from real records.
- **Cost.** A run takes one to two GPU hours. The first run waits until the `state-only` and `no-state` baselines finish training, because they hold the same 8 GB GPU.

## The share and the ablation together say where gen-3 puts its pressure

No thresholds exist yet, so the first run's figures are read by direction rather than against a cut-off.

- A family with a low share and high reliance is one where the trunk does work `e` could carry. Gen-3 then puts pressure on `e`, for example with auxiliary heads that read straight from `e`, or with a thinner trunk that cannot re-derive the knowledge itself.
- A family that no rung decodes is knowledge the model lacks altogether. That is a corpus problem, not an architecture problem, and the fix is records that show the effect.

## Outcome / Result

To be filled in after the first run, with each family's ladder and headline share, the ablation's loss increases per field, the two sweeps, and the join rate of the interaction labels.
