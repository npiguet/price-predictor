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

- `family` names a module of `effects.domain.rule_detectors`, which declares its `params`.
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

`outputs` keeps only names in the head-output list derived from `HeadField` and the verdict-head
columns. A rule with no heads line is `missing`.

## Directed-rule registry (Java)

`FE/directed/DirectedRules.java` maps rule id → `DirectedRule` with
`Optional<Binding> precondition(Game)` and `Map<String variantLabel, Application>`, each `Application`
naming its `ForkOperation`. A JUnit contract test reads `rules/scenarios/*.json` and fails when a
registered variant is missing or its operation differs. `DirectedRulesMain` prints the registered rule
ids, one per line.
