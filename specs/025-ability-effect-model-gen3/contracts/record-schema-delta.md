# Contract: record schema, gen-3 (delta)

**Feature**: `025-ability-effect-model-gen3`
Authority: [`../../2026-10-30-ability-effect-model-gen3.md`](../../2026-10-30-ability-effect-model-gen3.md) § 8.
Base: [`../../023-ability-effect-model/contracts/record-schema.md`](../../023-ability-effect-model/contracts/record-schema.md),
[`../../024-ability-effect-model-gen2/contracts/record-schema-delta.md`](../../024-ability-effect-model-gen2/contracts/record-schema-delta.md).

The text below becomes the base contract's "Gen-3" section (FR-019). The frozen-schema rule holds
within gen-3. Relative to gen-2 a field may change meaning or go away.

## Version

- Every gen-3 record carries `"generation": 3` as its first envelope member after `record_id`.
- Readers decide a record's generation by `generation`, else by `random_seat` (gen-2), else gen-1.
- `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the probes accept gen-3 only.
  `rule-counts` accepts gen-2 or gen-3, not mixed.

## Envelope

| Field | Shape | Present on |
|---|---|---|
| `generation` | `3` | every record |
| `aimed_at` | `["704.5f", …]` | aimed records only; `[]` allowed |
| `directed` | `{"variant": "tougher", "operation": "pt"}` | directed-fork records only |

Flag signatures, added row:

| record | `interventional` | `fork` | `directed` | `aimed_at` |
|---|---|---|---|---|
| directed fork | `false` | `true` | set | `[rule]` |

`mirror_of` on a directed-fork record names the live record open at the fork point, and is absent
when none was open.

## Kinds

`turn_action`, with `subkind` one of `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`,
`lore_counter`; payload `{"events": [...]}`. No `ability`. Never written to a curated stratum.

## Snapshot additions

Shapes follow `rules/gen3-fields.json`; the collector writes the field names below.

| Where | Field | Shape |
|---|---|---|
| entity | `owner` | player ref |
| card | `lines` | list of `{keys, kind, granted, active?, cast_with?, chosen?, fragment?}`; `kind` ∈ `spell`, `activated`, `trigger`, `static`, `replacement`, `keyword`; `active` on statics, and `false` on every line of a face not up |
| card | `face_state` | `front` \| `back` \| `flipped` \| `melded` \| `face_down` |
| card | `base_pt`, `pt_boosts` | `[p, t]`; boosts `{temporary: [p, t], counters: [p, t]}`; both `null` on a non-creature; base + temporary + counters = net P/T |
| card | `counters` | `{type: n}` over every named type |
| card | `designations`, `x_paid`, `chosen`, `costs_paid`, `history`, `order_rank`, `real_face`, `printed_loyalty`, `printed_defense`, `mana_cost` | as `rules/gen3-fields.json` |
| player | `speed`, `city_blessing`, `rad`, `monarch`, `initiative`, `turn_counts` | |
| global | `day_night` | `"day"` \| `"night"` \| `null` |
| command | `floating_effects`, `delayed_triggers` | `{id, source_lines, affected, remembered, duration, shield_left}`; `duration` ∈ `end_of_turn`, `until_leaves`, `permanent` |
| command | `dungeons` | `{player, dungeon, room}` |
| cards | `library_cards`, `exiled_with`, `mutate_components`, `last_known` | card entities with their role |
| snapshot | `links` | `[{from, to, type, index?}]`; `type` a § 10.3 family name; `index` only for `targets` |

A card's `lines` members:

| member | shape | present on |
|---|---|---|
| `keys` | list of 0 or 1 provenance key | every line; `[]` when the line resolves to no printed line |
| `kind` | one of the six kinds | every line |
| `granted` | bool: the line is not printed on this card (a static's, an attachment's, an effect's, an emblem's or a splice's) | every line |
| `active` | bool | statics of the face up (`checkConditions()`); `false` on every line of a face not up |
| `cast_with` | int ≥ 1: times the spell this card was cast as paid the line's cost (1 for an alternative cost or kicker, the count for multikicker) | `keyword` lines of a card on the stack or a permanent cast so |
| `chosen` | list of `Choices$` positions the ability on the stack chose, in chain order | the line of a modal ability on the stack |
| `fragment` | the keyword instance's own fragment as Forge writes it (`Ward:2`, `Annihilator:2`, `Protection:Goblin`) | `keyword` lines |

A `keyword` line's keys name the keyword line the card prints, else the static that granted the
keyword; a keyword a resolved effect granted has `keys: []` and is read by its `fragment`. The
triggers, statics and abilities a keyword instance builds are its line, not lines of their own; a
`Secondary$ True` trait is listed under its own key and joins its primary's line through the
sidecar. A land's play and a permanent's plain cast are not lines. A face not up (a face-down
card's real face, a double-faced card's other face) lists its printed lines with `active: false`;
a split card lists no face not up.

## Payload additions

| Payload | Field | Shape |
|---|---|---|
| resolution | `conditions_met` | `[{segment, met}]` |
| decision candidate | `verdict.timing` | bool |
| decision candidate | `failed_checks` | `[{check, responsible_static}]`; `check` ∈ `split_second`, `casting_speed`, `whose_turn`, `phase`, `combat`, `activation_limit`, `script_condition`, `zone`, `static_prohibition`, `cost`, `script_restriction` |
| decision candidate | `legal_targets` | union over every targeting clause, uncapped |
| `attackers` | `anchor_attacker`, `legal_defenders`, `must_attack`, `tapped_by_attack?` | one record per attacker |
| `blockers` | `must_block` per listed creature | bool |
| probe `combat` | damage events' `source` | entity ref |

## Behaviour changes readers rely on

- A combat record's snapshot precedes every damage event of its step.
- A resolution's effect half holds every death of the state-based loop after it. A trigger record
  never holds one.
- A prevention event with amount 0 is not written.
- `can_play = canPlay ∧ timing ∧ no static prohibition`.
- Decision candidates exist for both players at every priority stop.

## Removed

- `what_if` on `attackers` records (none is written).
- A card's `printed` and `granted_attached` (both now in `lines`) and `pt` (now `base_pt` and `pt_boosts`).
- The AI hook's decision records (replaced by the enumerator's).
