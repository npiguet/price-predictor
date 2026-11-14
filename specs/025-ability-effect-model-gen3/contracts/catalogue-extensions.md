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
  declare a detector, and its rules are counted under theirs.
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
