# Contract: catalogue extensions

**Feature**: `025-ability-effect-model-gen3`
Authority: root spec § 4–§ 7; file formats in [`../../../rules/README.md`](../../../rules/README.md),
which gains these sections.

## Scenario `detector`

```json
"detector": {
  "family": "sba_zone_move",
  "record_kind": "resolution",
  "outcome": "A 0-toughness creature moves to its owner's graveyard in the state-based loop after the resolution.",
  "near_miss": "The creature stays on the battlefield at toughness 1 or more.",
  "params": {"check": "toughness_zero"},
  "needs": []
}
```

or

```json
"detector": {"none": "The layer order shows only in the continuous record's result, which no field separates by layer."}
```

or, for a scenario stating another's observable fact (FR-002b),

```json
"detector": {"same_as": "702.9b-1", "reason": "Both state that a flier can't be blocked except by flying or reach."}
"detector": {"same_as": ["301.1-1", "302.1-1"], "reason": "The union of each card type's own rule."}
```

- `family` names a module of `effects.domain.rule_detectors`, which declares its `params`.
- No two scenarios of different rules share one declaration; a `same_as` names scenarios that
  declare a detector, of rules other than its own, and its rules are counted under theirs. Two
  scenarios of one rule stating one fact each declare the detector.
- `needs` lists gen-3 field names; on a gen-2 record the rule is not countable.

## Variant `operation`

```json
{"label": "tougher", "polarity": "does_not_fire", "delta": "…", "expected": "…", "operation": "pt"}
{"label": "no-thalia", "polarity": "does_not_fire", "delta": "Thalia is not on the battlefield.", "expected": "…",
 "operation": "remove_permanent"}
{"label": "other-target", "polarity": "does_not_fire", "delta": "A targets Hill Giant instead.", "expected": "…",
 "operation": {"unclassifiable": "a different target; no fork operation forces a choice"}}
```

`operation` ∈ `pt`, `keyword`, `colour_type`, `counters`, `tap`, `life`, `replace_card`,
`remove_permanent`, `add_permanent`, `move_card`. The base variant carries none. A variant that
differs by a choice, a target or a timing, changes the phase or the turn, or changes more than one
fact is unclassifiable.

## Scenario `departure`

```json
"departure": {"status": "confirmed", "resolution": "accepted",
              "forge_behaviour": "Forge keeps the creature on the battlefield.", "note": "…"}
```

Accepting rewrites every variant's `expected` to Forge's behaviour in the same edit.

## Heads map snapshot

`build-corpus` writes into `manifest.json`:

```json
"rule_map": {"704.5f": {"verdict": "trained", "outputs": ["zone_outcome"]}, "…": {}},
"rule_map_digest": "…"
```

`outputs` keeps only names in the head-output list `rule_catalogue.head_outputs()` derives from
`HeadField`, the verdict and global heads' layouts, `gate` and `created_objects`. A rule with
no heads line is `missing`. The snapshot holds every tagged rule and every rule a copy can be drawn
for.

## Directed-rule registry (Java)

`FE/directed/DirectedRules.java` lists the registered rules (`DirectedRules.REGISTERED`), one class
per rule in `FE/directed/rules/`. A `DirectedRule` names its rule id, the scenario whose base variant
its precondition waits for (two scenarios of a rule may share a variant label), a `BoardPrecondition`
(`bind(Game)`, returning the `Binding` of live objects by role, one that throws read as not met) and
`Map<String variantLabel, Application>`, each `Application` naming its `ForkOperation` and building the
`BoardEdit` from the fork's binding (an edit of another operation is refused). A JUnit contract test
reads `rules/scenarios/*.json` and fails when a registered scenario does not exist or does not list the
rule, a registered variant is missing, is the base variant, or its operation differs, when a card a
rule class names (a `static final String` the card database knows) is not one of its scenario's
`cards`, and when a rule `rules/directed-fork-rules.txt` lists has no registration.
`DirectedRulesMain` prints the registered rule ids, one per line.

A precondition binds only a board where its variant's edit makes the variant true for the turn the
fork plays: a `remove_permanent` variant binds the one permanent doing what it removes (no second
one would do it in its place); a "one land short" variant binds only when the active player holds no
card it could play as a land or has used its land drops; a spell the variant waits for is castable
at this boundary (a sorcery only in the active player's hand); a `replace_card` variant binds a
permanent whose replacing card survives the fork's first state-based check carrying the permanent's
marked damage, deathtouch damage, counters, boosts and resolved effects
(`ReplaceCardOperation.replacementDies`, the reading the edit's own refusal shares), and whose
keyword the variant takes away is of its own text (none granted by an attachment, a static or a
resolved effect); a `counters` variant adding a +1/+1 counter binds no permanent with a -1/-1
counter, and one taking +1/+1 counters away binds a creature that survives without them. A
`replace_card` edit refuses (`replace_card replacement dies at the first state-based check`) a
replacing creature that what it carried leaves at toughness 0 or less, with lethal damage, or with
deathtouch damage. 702.63c's `keyword` edit refuses (`keyword vanishing not left at one instance`)
a permanent the edit's closing static check leaves with other than one instance of vanishing.

An `Application` may also name the records its fork keeps (`KeptRecords`, FR-010c), built from the
fork's binding before the edit: the decision candidates of the cards it names
(`Application.keepingCandidatesOf(role)`) and the not-fired evaluations of the triggers those cards
host (`keepingNegativesOf(role)`). A card is named by id, which it keeps through the fork's zone
changes. In the fork's recording, `PriorityCandidates` writes a kept candidate at every priority stop
without drawing either sampling stage, and `TriggerChannel` offers a kept not-fired evaluation
without asking `TriggerNegativeSampler.offer` and counts it toward its line's cap of 24 not-fired
records only, never toward its mode's balance; deduplication applies to both. Kept records are fork
records like any other: the fork's lines count toward `--max-records-per-game` when they are written
to the shard, they enter neither the trigger reservoir nor the mana reservoir, and the fork's
sampling stream is its own, so the live game's samples do not move. The registry fixes which variants
keep what: 405.6d, 601.2c, 702.84a, 702.126a, 702.127a, 702.187b, 702.188a and 718.3a keep their
card's candidates; 608.2p keeps its trigger host's negatives.

A registered variant makes the side the rule's counts lack. Where that side needs a rare fact, the
registered scenario's base is the common board and its variant adds the fact (104.2b, 104.3d,
111.10v, 119.7, 608.2p, 614.17d, 701.12e, 701.39a (through 701.39a-2), 701.42a, 702.65a,
702.103e (through 702.103e-2), 702.124j, 702.131a, 702.170c; 712.4a and 712.14c through
712.4a-2, 712.4a-1 inverted). Where the lacking side is a fact a contrasting variant of the same
polarity as the base already states, that variant is registered: 701.12g (`opponent_above`) and
702.105a (`tied`).

An `Application` may also name the one play its fork forces (`ForcedPlay`, FR-010b), built from
the fork's binding after the edit: `ForcedPlay.cast(actor, card)`, `activate(actor, permanent,
line)`, `answer(actor, source, yes)` or `attack(actor, attacker, defender)`, each argument a
binding role, refined by `.at(step)` (a `PhaseType` name, a later step of the turn),
`.targeting(roles...)` (one call per targeting clause, in chain order; a clause named by no call
is drawn), `.paying(roles...)` (what the cost sacrifices, discards or exiles), `.withX(value)` and
`.playing(line)`. A line is chosen by `ForcedPlay.Line.SPELL` (the card's own cast, no
alternative cost), `Line.api(name)`, `Line.keyword(name)` or a test of the rule's own; names are
strings, read only inside a fork. The fork attaches the play to its actor's seat in the copy only:
a random seat consults it first (`ForcedPlaySeat`), a plain AI seat is replaced by a
`ForcingSeatController`. At the actor's first priority in the step where the rules let the line
be played (`PlayabilityVerdict`) with its named targets legal (`canTarget`), the seat plays it
through `RandomSpellPlayer.play`, the engine's own paid cast path, its cost decided by the named
cards and its mana by the AI; the answer is given to the source's optional trigger or
confirmation and reported through the engine's confirm hook as an AI answer is; the attack is
added to the seat's own declaration and checked with `CombatUtil.validateAttackers`. A paid
forced play is written by the fork's recorders as a live play, with no member marking it. The
controller only flags what fails; `ForkPlayForward` throws the refusal between main-loop steps
and after the turn, before the fork's lines are written: `forced play has no legal line`
(`ForcedPlayRun.dryRun` on the live board before the fork is taken, spending no budget, else on
the edited copy or as the play is made), `forced play not made` (its step passed, or the turn
ended, unmade), `forced play cost not paid` (the AI's mana payment or a named cost card failed).
Its unnamed targets, X and modes are drawn from a stream seeded from the game's fork seed, drawn
only by a fork that forces.

The registry fixes which variants force what. Forced casts: 104.3d (the poison source), 607.2k
(the champion card), 614.17d (the reanimation spell, under the Cage), 701.24e (the searcher's
tutor), 701.39a (the bolster spell), 701.44b, 701.50f and 701.57b (the card that explores,
connives or discovers), 702.103e (bestowed, its target drawn: the battlefield's one creature),
702.124j (the partner-with card), 702.131a (the ascend spell, sorcery or instant), 702.170c
(Kellan Joins Up), 704.5f (the -N/-N spell at the bound creature). Forced activations: 111.10v
(the Mutagen token), 119.7 and 701.12g (the life-exchange tree, at the bound opponent for
701.12g), 701.12e, 702.65a and 702.65b (Aura swap), 702.53a (transmute), 702.71a (transfigure). A forced answer: 118.12b (Elfhame Sanctuary's search, yes). A forced
attack: 702.105a (the dethrone creature at the bound opponent).

Every fork refuses a game restart (FR-010d, `Forks.refuseARestart`): when its copy is made, before
each main-loop step a fork played on takes, and before a forced resolution drains its stack, a stack
item whose chain has an `ApiType.RestartGame` clause ends the fork as the refusal `game restart`.
