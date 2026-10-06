# Contract: record schema and sidecar delta

**Feature**: `024-ability-effect-model-gen2`
Authority: [`../../2026-10-06-ability-effect-model-gen2.md`](../../2026-10-06-ability-effect-model-gen2.md) §§ 4.1, 6.3, 6.5.
Base: [`../../023-ability-effect-model/contracts/record-schema.md`](../../023-ability-effect-model/contracts/record-schema.md),
[`../../023-ability-effect-model/contracts/provenance-sidecar.md`](../../023-ability-effect-model/contracts/provenance-sidecar.md).

This delta is applied to both base contracts and to `CLAUDE.md`'s shard and sidecar paragraphs in the
same change that implements it (FR-034).

## Envelope

| Field | Type | Present on | Rule | Compatibility |
|---|---|---|---|---|
| `random_seat` | bool | every gen-2 record | `actor_player` is the random seat | rule 1 (added); absent reads `false` |
| `what_if` | bool | gen-2 `playability` `attackers`/`blockers` | `false` real decision, `true` what-if | rule 1 (added); absent reads unknown |
| `actor_player` | player id | `playability` | the deciding player (table below) | **named exception to rule 1**, gen-2 only |
| `link_id` | string | `resolution` | one cost half + one or more effect halves | meaning widened, type and spelling unchanged |

`actor_player` on `playability` records:

| Subkind | Gen-2 | Gen-1 (feature 023) |
|---|---|---|
| `attackers` | controller of the candidate attackers | active player |
| `blockers` | controller of the candidate blockers | controller of the anchored attacker |
| `decision` | controller of the candidate ability | active player |

Both new fields join `COLLECTION_METADATA_FIELDS`; compatibility rule 4 applies.

Envelope key order in `EffectRecord.toJson`: `random_seat` after `synthetic`; `what_if` after
`random_seat`, written only on the two subkinds.

## Shard generation

| Shard | Marker | `actor_player` on `playability` |
|---|---|---|
| gen-2 | every record carries `random_seat` | the deciding player |
| gen-1 | no record carries `random_seat` | feature 023's meaning |

Readers of a records set refuse one holding both.

## Real decision (`what_if = false`)

| Subkind | Real when |
|---|---|
| `attackers` | snapshot phase `combat_declare_attackers` and `actor_player` is the active player, or written by the random seat for its own declaration |
| `blockers` | snapshot phase `combat_declare_blockers` and the anchored attacker is attacking, or written by the random seat for its own declaration |

De-duplication key: `(subkind, payload, snapshot)`. `--legality-rate` samples what-ifs only.

## Provenance key

```jsonc
{ "face": 0, "trait_kind": "spell", "index_within_kind": 0, "option": 2 }
```

`option` is written only for a charm mode; it is the mode's 0-based `Choices$` position. A key with
`option` never equals the key without it. Records' `ability` lists and sidecars' `provenance` lists
both carry it.

## Modal resolution

| Worker | Records |
|---|---|
| patched | one cost half (root key) + one effect half per chosen mode resolution (root key + `option`), shared `link_id`; each effect half's snapshot is taken at its mode's first clause; Pawprint and repeated modes give one half per resolution |
| degraded | one cost half + one effect half (root key) with every mode's events |
| fork (`ForkCollector`) | one effect half per chosen mode by the patched rule |

## Sidecar line

| Field | Gen-2 |
|---|---|
| `script_text` | `seg₀ [SEG] SV1: seg₁ [SEG] SV2: seg₂ …`, each segment its trait's parameters in key order, chain labels renamed `SV1…` by first appearance |
| charm-mode `option` line | `provenance: [root + option]`, `script_text` = the mode's chain opened `SV1:`, `script_api_type` = the mode's |
| charm root line | `Choices$ SV1,SV2,…`, no mode inlined |
| non-charm `option` line | `provenance: []`, `script_text: ""` (unchanged) |
| descriptionless charm | root key in `dropped_keys`; option lines keyed as above |

Example, a trigger `Execute$ TrigDraw` whose `TrigDraw` names `SubAbility$ DBChange`:

```
Execute$ SV1 | Mode$ ChangesZone | … [SEG] SV1: DB$ Draw | NumCards$ 1 | SubAbility$ SV2 [SEG] SV2: DB$ ChangeZone | …
```

## Keyword definitions file

Each entry adds `"formatter": "<Keyword.type simple name>"`. `reminder_template` has `’` replaced by
`'` at load, not in the file.
