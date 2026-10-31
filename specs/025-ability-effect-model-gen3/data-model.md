# Data Model: Ability effect model — generation 3

**Feature**: `025-ability-effect-model-gen3` | **Date**: 2026-10-31
**Base**: [`../024-ability-effect-model-gen2/data-model.md`](../024-ability-effect-model-gen2/data-model.md),
[`../023-ability-effect-model/data-model.md`](../023-ability-effect-model/data-model.md)

Only what this feature adds or changes. Decisions cited as R1, C2, … are in [research.md](research.md).
Wire shapes are in [contracts/](contracts/).

## Rules catalogue

### Scenario (extended)

File `rules/scenarios/<unit>.json`, keys as `rules/README.md` specifies, plus:

| Field | Type | Rule |
|---|---|---|
| `detector` | `DetectorDeclaration` or `{"none": reason}` | required on every scenario (FR-002) |
| `departure` | `Departure`, optional | absent means no departure recorded |
| `variants[].operation` | `ForkOperation` or `{"unclassifiable": reason}` | required on every non-base variant (FR-003) |

### DetectorDeclaration (new)

| Field | Type | Rule |
|---|---|---|
| `family` | string | a registered detector family id (R1) |
| `record_kind` | string | equals the scenario's `record_kind` |
| `outcome` | string | the observable outcome the detector looks for, in one sentence |
| `near_miss` | string | the contrasting variant's outcome the detector rejects |
| `params` | object | the family's parameters (a trigger `Mode$`, a keyword, a verdict bit, an `Event$`, …) |
| `needs` | list of field names | gen-3 fields the detector reads; empty when it runs on gen-2 records |

Validation: `rules_catalogue.py validate` fails on an unknown family, on parameters the family does not
declare, and on a scenario with neither `detector` form.

### Departure (new)

| Field | Values |
|---|---|
| `status` | `suspected`, `confirmed` |
| `resolution` | `unpatched`, `patched`, `accepted` (only with `confirmed`) |
| `forge_behaviour` | prose: what Forge does |
| `note` | prose |

State transitions:

```
(none) ──review keeps the disagreement──► suspected ──confirmed──► confirmed/unpatched
confirmed/unpatched ──patch on effect-record-hooks──► confirmed/patched     (rule back in rule tests, expected unchanged)
confirmed/unpatched ──accept──► confirmed/accepted                         (expected rewritten to Forge's behaviour; rule back in rule tests)
```

A rule with any scenario at `confirmed/unpatched` has no rule-test records (FR-071).

### ForkOperation (enum, new)

`pt`, `keyword`, `colour_type`, `counters`, `tap`, `life`, `replace_card`. Mirrored by the Java
`ForkOperation` enum; a contract test pins both spellings.

### RuleMapEntry (snapshot, new)

Read from `rules/heads/<unit>.jsonl` by `build-corpus` only (R4).

| Field | Type | Rule |
|---|---|---|
| `rule` | rule id | |
| `verdict` | `trained` \| `recorded` \| `missing` \| `planned` | |
| `outputs` | list of head output names | filtered to outputs the model has |

The manifest stores `{rule → RuleMapEntry}` and `rule_map_digest`.

## Detection

### RuleTag (value object, new)

| Field | Type | Rule |
|---|---|---|
| `rule` | rule id | |
| `polarity` | `fires` \| `does_not_fire` \| `other` | from the variant situation the record matches |
| `observed` | bool | whether the scenario's outcome happened in the record |
| `entities` | list of entity refs | the entities the outcome names; a verdict-head rule names `[ACT]` |
| `scenario` | scenario id | |

- A record's tags hold one `RuleTag` per rule. When several scenarios of a rule tag it, polarity is
  `fires` if any says so, else `does_not_fire`, else `other`; entities are the union (R1).
- `observed ≠ (polarity == fires)` puts the record on the review list. Such a record is not a rule-test
  record.
- Tags are computed, never stored in collected shards. `build-corpus` stores them only through
  `curation` (below).

### RuleSource (enum, new)

`random_seat` (natural, game has a random seat), `forge_ai` (natural, two Forge AIs), `aimed` (record
carries `aimed_at`). A gen-2 record is natural.

### RuleCounts (new; `rule-counts.json` and the manifest)

```
rule → { polarity → { source → count } }  plus  not_countable: [rule], none: {rule → reason}, review: [{record_id, scenario}]
```

## Records

### EffectRecord envelope (extended)

| Field | Type | Kinds | Rule |
|---|---|---|---|
| `generation` | int, `3` | all | absent: gen-1 or gen-2 (feature 024's rule) |
| `aimed_at` | list of rule ids | all, aimed records only | directed fork: `[rule]`; `collect-coverage` / `collect-variants`: the run's `--directed-fork-rules`, possibly `[]`; absent on every other record |
| `directed` | `{variant, operation}` | all, directed-fork records only | `operation` a `ForkOperation` |
| `kind` | adds `turn_action` | | subkinds `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`, `lore_counter` |

`generation`, `aimed_at`, `directed` are collection metadata; `test_schema_compatibility.py` checks that
flipping them leaves `SurfaceBuilder.build` unchanged.

Flag signature added: directed fork = any kind, `interventional = false`, `fork = true`, `directed` set.

### Snapshot (extended)

Per entity: `owner`. Per card: `lines` (spell, activated, trigger, static, replacement and keyword
lines, each `{keys, kind, granted, active?, cast_with?, chosen?}`), `face_state`, `base_pt`
(set base) and `pt_boosts`, `designations`, `x_paid`, `chosen`, `costs_paid`, `history`, `order_rank`,
`real_face` (face-down only), `printed_loyalty`, `printed_defense`, `mana_cost`, `counters` (every
named type). Per player: `speed`, `city_blessing`, `rad`, `monarch`, `initiative`, `turn_counts`.
Global: `day_night` (`day` | `night` | null). Command zone: `floating_effects` and `delayed_triggers`
(`{source_lines, affected, remembered, duration, shield_left}`), `dungeons` (`{player, dungeon,
room}`). Extra card entities: `library_cards` (top card plus every one the events name, with
position), `exiled_with`, `mutate_components`, `last_known` (the pending event's subject). Relations:
`links: [{from, to, type}]` with the § 10.3 family names and target index. Shapes as
`rules/gen3-fields.json` gives them; wire form in [contracts/record-schema-delta.md](contracts/record-schema-delta.md).

### Payload changes

| Payload | Field | Rule |
|---|---|---|
| resolution | `conditions_met: [{segment, met}]` | one entry per chain segment with `Condition$` |
| playability `decision` candidate | `verdict.timing`, `failed_checks: [{check, responsible_static}]`, `legal_targets` (all clauses, no cap) | `timing = false` ⇒ a `casting_speed` entry |
| `attackers` | `anchor_attacker`, `legal_defenders`, `must_attack`, `tapped_by_attack?` | one record per attacker; `tapped_by_attack` only when the anchor was declared |
| `blockers` | `must_block` per listed creature | |
| `turn_action` | `events` | |
| `combat` | snapshot at the start of the damage step | shape at the first damage event |
| probe `combat` | damage events carry `source` | from `assignment_choices` |

## Curation

### CurationAnnotation (new, curated records only)

Member `curation` on every record `build-corpus` writes (T1):

| Field | Type | Rule |
|---|---|---|
| `natural_share` | int ≥ 0 | n: copies gen-2's budgets give the record |
| `copies` | int ≥ 1 | d: the record's copies written, natural and added |
| `drawn_rule` | rule id or null | set on an added copy only |
| `drawn_rules` | list of rule ids | every rule any added copy of this record was drawn for |
| `rule_entities` | list of entity refs | the drawn rule's tag entities, on an added copy |

Invariants: `copies ≤ reuse cap`; `drawn_rule ∈ drawn_rules`; a natural copy has `drawn_rule = null`.
A protection copy (FR-058) carries `synthetic = true` and its `drawn_rule`.

### RuleRow (new, manifest)

Per class, rule and polarity, each count split by `RuleSource`:

| Field | Rule |
|---|---|
| `available` | distinct tagged records |
| `share` | the half (or, for `other`, none) |
| `written` | natural plus added copies |
| `shortfall` | `max(0, share − written)` |

### CorpusManifest (extended)

Adds `rule_counts`, `rule_map`, `rule_map_digest`, `rules` (class → rule → polarity → `RuleRow`),
`one_sided_rules`, `one_sided_conditions`, `unbudgeted_rules` (rule → counts), `left_out_of_half`
(stratum → half → [rule]), `turn_action_left_out` (subkind → count), `subtypes` and `keywords` (the
closed lists), `halves` (stratum → half → [game_id]). Gen-3 manifests only; no gen-2 compatibility.

### Placement (new, `stratum_placement.py`)

Input: per-game inventories `game → {(rule, polarity): n}` and the game-disjoint draws. Output:
`GameStrata` gains `halves`. Deterministic for a given corpus; not stable across corpus growth (R3).

## Model

### Relation (new)

`(from_slot, to_slot, type)`, `type` an index into the frozen `RelationType` list (M4). Collated as a
`(N, 4)` long tensor `[batch, from, to, type]`. Built after context dropout and gate-2 stripping.

### RelationEncoderLayer (new)

Parameters: stock `TransformerEncoderLayer` names plus `rel_k`, `rel_v` of shape
`(heads, relation_types, head_width)`, zero-initialized; `rel_v` a buffer under
`--relation-values off`.

### Output layout (extended)

The 22 outputs of root § 11.2 and the § 11.3 extensions as `FieldSpec` rows; legality fields scored
on a gate-free path; a new `global` aux head; the verdict head gains `timing` and 11 failed-check
columns, a cost-adjustment delta per mana symbol kind, `resolution_outcome`, `effects_created` and
`replacement_result`. Each count field gains a learned log-dispersion.

### WeightAverage (new)

| Field | Rule |
|---|---|
| `state` | averaged tensors of encoder, model and identity table |
| `start_step` | optimizer step at warmup's end |
| `rate` | `grad_accum / (steps_per_epoch × ema_epochs)` |

### EffectCheckpoint (extended)

Adds the relation tables (inside the model state), the averaged weights (as the saved state),
`best_control_score`, `rule_mask`, `deviance_floor`, `ema_epochs`, `stratum_weight`,
`relation_values`, `relation_types`, `relation_target_indices`, `rule_map_digest`,
`loss_denominators`, `selection_set_digest`. Removes `best_val_loss`. `training_settings["input_layout"]
= "gen-3"`; any other value is refused (M1).

### resume.pt (extended)

Adds `weight_average` (`state`, `start_step`), `loss_denominators`, `best_control_score`.

## Selection and evaluation

### SelectionSet (new; replaces the frozen probe set)

File `output/effects/reports/selection-set-<corpus digest>.json`, written by the freeze:

| Field | Rule |
|---|---|
| `corpus_digest` | the curated corpus it was frozen against |
| `r` | 4 |
| `halves` | stratum → `selection` \| `reporting` → [game_id] |
| `probes` | stratum → half → probe type → [record ref] |
| `grids` | stratum → half → grid → {cells, records} |
| `rule_tests` | stratum → half → rule → {fires: [record ref + entities], does_not_fire: [...]} |
| `deviance_samples` | stratum → half → [record ref] |
| `standard_errors` | item → {hard, soft} or a scalar |
| `left_out` | stratum → half → component → [{item, reason}] |
| `digest` | sha256 over the canonical JSON |

Invariants checked at the freeze (exit 1 naming stratum, half, component): every component's
standard error ≤ 0.05; every polarity, probe type and record-drawn grid cell within r; every grid cell
non-empty.

### StratumScore, ControlScore (new)

```
component ∈ {probes, grids, rule_tests, deviance}; each rescaled, clipped below at −1
StratumScore = mean of the four components
ControlScore = W · card_disjoint + (1 − W) · game_disjoint
```

An empty component in an epoch is exit 1 (FR-068c).
