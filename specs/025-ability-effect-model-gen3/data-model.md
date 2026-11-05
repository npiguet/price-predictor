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

`pt`, `keyword`, `colour_type`, `counters`, `tap`, `life`, `replace_card`, `remove_permanent`,
`add_permanent`, `move_card`. Mirrored by the Java `ForkOperation` enum; a contract test pins both
spellings. Each is a silent edit of the fork's copy (FR-011a); none forces a choice, a target or a
timing.

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
records: n, rules: rule → { polarity → { source → count } }, not_countable: [rule], none: {rule → reason}, review: [{record_id, scenario}]
```

A rule no record tags is absent from `rules`; absent is not "not countable".

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

Flag signatures added: directed fork = any kind, `interventional = false`, `fork = true`, `directed`
set; protection fork (FR-058) = any kind, `interventional = true`, `fork = true`, its cost halves
carrying `link_id` as live halves do. A forced resolution's effect half is written without a
partner, and so is a protection fork's mana record (every event `mana_produced`), as a live one is;
no flag tells the two apart.

### Snapshot (extended)

Per entity: `owner`, `role` (null on tier entities). Per card: `lines` (spell, activated, trigger,
static, replacement and keyword lines, each `{keys, kind, granted, active?, cast_with?, chosen?,
fragment?, affected?}`; a keyword line's `fragment` is the instance as Forge spells it, `affected` a
static's cards then players), `face_state`, `base_pt` (set base) and `pt_boosts`, `designations`, `x_paid`, `chosen`, `costs_paid`, `history`, `order_rank`,
`real_face` (face-down only), `printed_loyalty`, `printed_defense`, `mana_cost`, `counters` (every
named type). Per player: `speed`, `city_blessing`, `rad`, `monarch`, `initiative`, `turn_counts`,
`max_land_plays` (int, null when unlimited; beside `this_turn.lands_played`).
Global: `day_night` (`day` | `night` | null). Command zone: `floating_effects` and `delayed_triggers`
(`{source_lines, affected, remembered, duration, shield_left}`), `dungeons` (`{player, dungeon,
room}`). Extra card entities are appended to `entities` after the tiers, each with its `role`:
`library_card` (top card plus every one the events name, with position), `exiled_with`,
`mutate_component`, `last_known` (the pending event's subject, under id `L<card id>`). `refs.random`
holds the half's dice and coin flips. Gen-2's `printed`, `granted_attached` and `pt` are gone.
Relations: `links: [{from, to, type, index?}]` with the § 10.3 family names, each link written once in
its forward direction, `index` on `targets` only; the four families the reader builds (`line_of`,
`line_of_the_same_card`, `mode_of`, `has_the_same_name`) are not collected; an `affects` link ends at
a card or a player (FR-026a), and a static line's or floating effect's `affected` lists cards, then
players. Shapes as
`rules/gen3-fields.json` gives them; wire form in [contracts/record-schema-delta.md](contracts/record-schema-delta.md).

### Payload changes

| Payload | Field | Rule |
|---|---|---|
| resolution | `conditions_met: [{segment, met}]` | on every effect half: one entry per conditional clause that resolved (any parameter `SpellAbilityCondition` reads), `segment` spelled as `attributed_to`; `[]` when none |
| playability `decision` candidate | `verdict.timing`, `failed_checks: [{check, responsible_static}]`, `legal_targets` (all clauses, no cap), `land_play` | `timing = false` ⇒ a `casting_speed` entry; a land-play candidate (`land_play = true`, FR-033a) fails only `whose_turn`, `casting_speed`, `zone`, `static_prohibition` or `land_play_limit` |
| `attackers` | `anchor_attacker`, `legal_defenders`, `must_attack`, `tapped_by_attack?` | one record per creature the attacking player controls, a tapped one included; `tapped_by_attack` only when the anchor was declared; gen-2's `legal_attackers` and `forbidden` kept; no `what_if` |
| `blockers` | `must_block` | the refs of the listed creatures a requirement makes block the anchor |
| `turn_action` | `events` | |
| `combat` | snapshot at the start of the damage step | shape at the first damage event |
| `combat`, observed and probe | damage events carry `source` | the bus's `cause`, else from `assignment_choices` |
| decision candidate | `cost_after_adjustment` | the cost after Forge's cost adjustment |
| events | `pt_change.set_pt`, zone changes' `library_position`, `card_looked_at.mode`, `token_created` controller and counters, `ability_change` subjects, new `state_flag_change` flags | wire shapes in [contracts/record-schema-delta.md](contracts/record-schema-delta.md) |

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
A protection copy (FR-058) is a synthetic record `<record_id>~pc<i>` with `natural_share = 0` and its
`drawn_rule`. A validation stratum holds a record once per copy, so a reader of distinct records
dedupes by `record_id` (FR-068a).

### RuleRow (new, manifest)

Per class, rule and polarity, each count split by `RuleSource`:

| Field | Rule |
|---|---|
| `available` | distinct tagged records |
| `share` | the half (or, for `other` and unbudgeted rules, none), per polarity |
| `written` | natural plus added copies |
| `added` | the copies rule budgets added |
| `shortfall` | `max(0, share − written)`, per polarity |

### CorpusManifest (extended)

Adds `rule_counts`, `rule_map`, `rule_map_digest`, `rules` (class → rule → polarity → `RuleRow`),
`one_sided_rules`, `one_sided_conditions`, `unbudgeted_rules` (rule → counts), `left_out_of_half`
(stratum → half → [rule]), `turn_action_left_out` (subkind → count), `subtypes` and `keywords` (the
closed lists), `halves` (stratum → half → [game_id]). Gen-3 manifests only; no gen-2 compatibility.

### Placement (new, `stratum_placement.py`)

Input: `GameCensus` (games, held-out games, random-seat games, keyword-combat games, held-out text
games, per-game inventories `game → {(rule, polarity): n}` of rule-test-eligible records) and a
`PlacementRule` (gen-2's draws, the pool factor, the half minimum, the item error). Only Forge-AI
games enter a validation stratum. Output: `GameStrata` gains `halves` and `left_out_of_half`.
Deterministic for a given corpus; not stable across corpus growth (R3).

## Model

### Relation (new)

`(from_slot, to_slot, type)`, `type` an index into the frozen `RELATION_TYPES` list (M4: 31 families,
each forward and reverse, `targets` per index 0, 1, 2 and 3+; 68 types). Collated as a `(N, 4)` long
tensor `[batch, from, to, type]`. Built after context dropout and gate-2 stripping, from the record's
`links`, the four reader-built families and `[ACT]`'s `targets` and `source_of`; a line end with no
slot reads its card's, and a link to an object with no slot is dropped.

### Card slot (extended)

A sequence of named parts (`card_segments`). Subtypes and chosen creature types are id lists into one
learned embedding; each named counter is a row of a counter embedding scaled by its count and its
log1p; the subtype and counter lists are Forge's, checked in (`forge_vocabularies.py`). Keyword flags
are a multi-hot over `KEPT_KEYWORD_FLAGS` (85). Floating effects and delayed triggers the snapshot
gives no entity are card slots in the command zone, with a duration one-hot and the shield left. A
dungeon is a card slot holding the one keyword line `convert` renders for all its rooms. Every line is
an ability slot with seven flags (active, granted, granted temporarily, cast-with flag and count raw
and log1p, chosen).

### RelationEncoderLayer (new)

Parameters: stock `TransformerEncoderLayer` names plus `rel_k`, `rel_v` of shape
`(heads, relation_types, head_width)`, zero-initialized; `rel_v` a buffer under
`--relation-values off`.

### Output layout (extended)

The 22 outputs of root § 11.2 and the § 11.3 extensions as `FieldSpec` rows (300 per-entity fields,
1,386 columns); legality fields scored on a gate-free path; a new shipped `global` head (turn-structure
changes, day/night change); the verdict head (170 columns) gains `timing`, one failed-check column per
`PlayabilityCheck`, a cost-adjustment delta per mana symbol kind (on casts only),
`resolution_outcome`, `effects_created` and `replacement_result`, each a `RecordField` with its own
mask column; created-objects slots gain counters, an other-controller bit, a kind, subtypes and
keywords. The subtype and keyword output lists are `output_vocabularies.py`, recorded in
`layout_settings`. Each count field gains a learned log-dispersion.

### WeightAverage (new)

| Field | Rule |
|---|---|
| `state` | averaged tensors of encoder, model and identity table |
| `start_step` | optimizer step at warmup's end |
| `rate` | `grad_accum / (steps_per_epoch × ema_epochs)` |

### EffectCheckpoint (extended)

Adds the relation tables (inside the model state), the averaged weights (as the saved state),
`best_control_score`, `rule_mask`, `deviance_floor`, `ema_epochs`, `stratum_weight`,
`relation_values`, `rule_map_digest`, `loss_denominators`, `selection_set_digest`, each under its
own key. Removes `best_val_loss`. `training_settings` holds `input_layout = "gen-3"` and, from
`layout_settings()`, the relation types, the relation target indices, the subtypes and named counters
and the closed output lists; any other layout or list is refused (M1).

### resume.pt (extended)

Adds `weight_average` (`state`, None while warmup lasts, and `start_step`), `loss_denominators`,
`best_control_score`; the stopper's state names its measure (`control_score`), and a resume file whose
stopper kept the loss is refused.

## Selection and evaluation

### SelectionSet (new; replaces the frozen probe set)

File `output/effects/reports/selection-set-<corpus digest>.json`, written by the freeze:

| Field | Rule |
|---|---|
| `corpus_digest` | the curated corpus it was frozen against |
| `r` | 4 |
| `halves` | stratum → `selection` \| `reporting` → [game_id] |
| `probes` | stratum → half → probe type → [{record_id, game_id, entity, label}] |
| `probe_kinds` | probe type → `binary` \| `amount` |
| `grids` | stratum → half → grid → {cells: [{cell, outcome}], records: [{record_id, game_id, entities}]} |
| `rule_tests` | stratum → half → rule → {fires: [{record_id, game_id, entities}], does_not_fire: [...]} |
| `deviance_samples` | stratum → half → [{record_id, game_id, copy}] |
| `standard_errors` | `stratum/half/component/item` → {hard, soft} for an output field, else a scalar; infinite written as null |
| `left_out` | stratum → half → component → [{item, reason}]; an untested rule's reason is `untested` |
| `settings` | the thresholds the freeze ran under, the grid target error and record cap included |
| `digest` | sha256 over the canonical JSON |

A grid keeps the fewest records, smallest keyed hash first, that bring its worst-case standard error
to 0.09, at most 512; the deviance sample holds every copy up to 16,384 a half. The deviance
component's items are the per-entity head's fields, the gate included.

Invariants checked at the freeze (exit 1 naming stratum, half, component): every component's
standard error ≤ 0.05; every polarity, probe type and record-drawn grid cell within r; every grid cell
non-empty.

### StratumScore, ControlScore (new)

```
component ∈ {probes, grids, rule_tests, deviance}; each rescaled, clipped below at −1
StratumScore = mean of the four components
ControlScore = W · card_disjoint + (1 − W) · game_disjoint
```

An empty component in an epoch is exit 1 (FR-068c). Scoring the reporting half of a saved
checkpoint (`evaluate-effect-model`, the probes) is lenient: an empty component is NaN and named.
