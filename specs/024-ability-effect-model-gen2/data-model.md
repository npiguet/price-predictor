# Data Model: Ability effect model — generation 2

**Feature**: `024-ability-effect-model-gen2` | **Date**: 2026-10-06
**Base**: [`../023-ability-effect-model/data-model.md`](../023-ability-effect-model/data-model.md)

Only what this feature adds or changes. Every entity the base data model describes and this file
does not mention is unchanged.

## Ability identity

### ProvenanceKey (extended)

```
ProvenanceKey = (script_file, face, trait_kind, index_within_kind, option?)
```

| Field | Type | Rule |
|---|---|---|
| `option` | int ≥ 0, optional | a charm mode's 0-based position in its root's `Choices$`; absent names the root or a non-modal trait |

- JSON: an extra `"option"` member of the key object, written only when set. Additive under
  compatibility rule 1.
- Equality and hashing include `option`, so a mode key never matches its root key.
- Produced by: `RulesParser`/`ProvenanceRecorder` for sidecar `option` lines; `ProvenanceKey.resolve`
  for a cloned mode at resolution; `RandomChoices`/`ForkCollector` for fork modes.

### SidecarLine (extended)

| Field | Change |
|---|---|
| `script_text` | the encoding text: the trait's chain, segments joined by ` [SEG] `, each segment after the root opened by `SVn:`, chain labels renamed `SV1…` by first appearance (FR-001–003) |
| `provenance` | a charm-mode `option` line now carries `[root key + option]` |
| `script_api_type` | a charm-mode `option` line carries its mode's API type |
| `script_param_keys` | unchanged meaning: the root segment's keys |
| `line_kind` | unchanged; `option` lines keep `option` |

Validation:

- An `option` line with a key follows its root line, in `Choices$` order.
- A non-charm `option` line (die roll) has `provenance: []` and `script_text: ""`.
- A charm with no rendered root line lists the root key in `dropped_keys`.
- `.txt` files are byte-identical to the pre-feature conversion.

### EncodingText (value object)

The `script_text` of a line, or the prose for a line with no script (FR-005). It keys the rarity
table and, through `MaskedTemplate`, the holdout.

### MaskedTemplate (value object, new)

| Field | Rule |
|---|---|
| `text` | whitespace-normalized `script_text` with every digit run outside chain labels, `CARDNAME`, and every `*Description$` value replaced by `#` (FR-039) |

- Chain labels are the `SVn` tokens at the FR-002a positions; they keep their digits.
- Used only by the holdout; no encoding reads it.

## Records

### EffectRecord envelope (extended)

| Field | Type | Kinds | Rule |
|---|---|---|---|
| `random_seat` | bool | all | `actor_player` is the random seat. Absent (gen-1) reads `false` |
| `what_if` | bool | `playability`/`attackers`, `blockers` | real decision `false`, what-if `true`. Absent (gen-1) reads unknown |
| `actor_player` | player id | `playability` | **redefined** (FR-030a): the deciding player — candidates' controller (`attackers`), blockers' controller (`blockers`), candidate ability's controller (`decision`). Gen-1 shards keep the active player / attacking player meaning |
| `link_id` | string | `resolution` | **widened** (FR-029d): joins one cost record with every effect half of its resolution, one per chosen mode for a patched modal resolution |

Both new fields are collection metadata: listed in `COLLECTION_METADATA_FIELDS`, never in
`model_input_fields()`.

**Shard generation**: a shard whose records carry `random_seat` is gen-2; one without it is gen-1.
Readers that consume a records set refuse one that mixes the two (FR-033).

Validation (`validate-corpus`):

- A shard where any record carries `random_seat` carries it on every record.
- A shard where any legality record carries `what_if` carries it on every legality record.
- `what_if` appears on no other kind.
- A `link_id` joins exactly one cost half and at least one effect half. More than one effect half is
  valid only when every effect half acts through a key carrying `option` and they share a root.

### Modal resolution (record group)

| Record | Acting key | Snapshot | Events |
|---|---|---|---|
| cost half | the charm's root key | before the cost is paid | the cost's |
| effect half, one per chosen mode resolution | root key + `option` | just before that mode's first clause resolves | that mode's clauses' only |

A degraded worker writes one effect half acting through the root key with every mode's events
(FR-029e).

## Curation

### RuleFamily (value object, new)

A string naming a record's rules category, computed by `rule_family(record, sidecars)` per
research.md § Rule families. A keyword acting line overrides the kind's rule (FR-048).

### PlayabilityFamily (FR-047a)

| Subkind | Family |
|---|---|
| `decision` (per candidate) | responsible static's `Mode$`; else first false of `can_play` → `cannot-play`, `affordable` → `unaffordable`, `has_legal_target` → `no-legal-target`; else `none` |
| `attackers`, `blockers` | sorted, comma-joined distinct `Mode$` over `forbidden[].responsible_static`; `none` when empty |

### OutcomeSignature (value object, new)

`frozenset[(zone_outcome | "stayed", changed: bool)]` over the record's entities, from
`derive_targets`. Defined for `resolution-effect`, `rewrite`, `combat`, `continuous`. Counts ignored.

### SelectionCell

| Field | Meaning |
|---|---|
| `class` | one of the eight sampling classes |
| `half` | `real` / `what-if`, `playability-legality` only |
| `family` | `RuleFamily` |
| `signature` | `OutcomeSignature`, where defined |
| `texts` | `{text: [record_hash…]}` |

- A text's capacity in a cell is `min(reuse_cap × n, text_cap)`, `n` its distinct records there.
- `allocate(budget, capacities) → shares`: equal split, capacity-capped, redistributed until full or
  exhausted. Applied class → (half) → family → signature.
- Each record gets `copies ∈ {0, 1, …, reuse_cap}`; the write pass writes it that many times.

### Game-disjoint placement

`in_stratum = not names_held_out_card and crc32(game_id) % 10**6 / 10**6 < share`, with `share =
--game-disjoint-keyword-share` when the game holds a combat record gate 2 qualifies for a listed
keyword, else `--game-disjoint-share`. Depends on the game alone (FR-046).

### CorpusManifest (extended)

New fields, each omitted when equal to the gen-1 default so gen-1 digests are unchanged:

| Field | Shape |
|---|---|
| `holdout_unit` | `"template"` \| `"text"` (default `"text"` when absent) |
| `families` | per class, per family: `{available, share, written, repeats, shortfall}` |
| `signatures` | per class, per family: `{signature: written}` |
| `policy_counts` | per class: `{on_policy, off_policy}` |
| `legality_counts` | per subkind: `{real, what_if, unknown}` |
| `keyword_threshold_games` | game ids admitted under `--game-disjoint-keyword-share` |
| `held_out_texts_without_resolution` | texts with no gate-one resolution record |
| `held_out_texts_under_five_games` | `{text: games}` |
| `game_disjoint_share`, `game_disjoint_keyword_share`, `game_disjoint_keywords` | the flags as run |
| `text_cap`, `reuse_cap` | the flags as run |

`game_disjoint_target` is no longer written; a manifest that carries it is still read.

## Training

### ValueTargets (new)

Per encoding text, from `script_text` alone:

| Target | Loss | Source | Masked when |
|---|---|---|---|
| damage | count | `NumDmg$` summed over segments | any segment states a non-literal |
| power change, toughness change | count (signed via the existing direction/magnitude split) | `NumAtt$`, `NumDef$`, `PowerBonus$`, `ToughnessBonus$` | same |
| counters placed | count | `CounterNum$` | same |
| cards drawn | count | `NumCards$` on `Draw` segments | same |
| mana cost W, U, B, R, G, C, generic | count | root `Cost$` via `ManaCost.parse` | spell line; no `Cost$`; non-literal X |
| cost taps | binary | `T` shard in root `Cost$` | no `Cost$` |
| cost sacrifices | binary | `Sac<` shard in root `Cost$` | no `Cost$` |

A charm root line has no amounts, so its amount targets are all masked.

### NoiseState (new, training only)

| Field | Meaning |
|---|---|
| `sigma` | running `e_dim × e_dim` covariance, device tensor, no gradient |
| `step` | optimizer steps taken; `r = --e-noise × min(1, step / --steps-per-epoch)` |

### Checkpoint (extended)

| Field | Rule |
|---|---|
| `cards_folders` | the sidecar roots trained against (FR-063b) |
| `encoder_config` | already records `d_model`, `n_layers`, `n_heads`, `ff_dim`; `e_noise` is dropped on load if present; `ff_dim` follows `d_model × 4` |
| `split.holdout_unit` | `"template"` \| `"text"`, absent → `"text"` |
| `e_noise`, `value_weight`, `mlm_weight`, `mlm_mask_prob`, `api_weight` | the training flags as run |
| `n_api_types`, `n_param_keys`, `api_types`, `param_keys` | vocabularies of the script-API head |
| saved state dict | excludes `value_head`, `mlm_head`, `api_type_head`, `param_key_head` |

### EffectHeadInput (extended)

- `Slot` gains `option: bool`.
- After each `ABILITY` slot whose sidecar row is followed by keyed `option` rows, one `ABILITY` slot
  per such row, `option = True`, positions continuing the card's block.
- `collate_surfaces` emits an `option_kinds` tensor beside `slot_kinds`.
- `decision` records: `[ACT]` carries `e` of `candidates[0].ability`.

## Evaluation

### Breakdown keys

| Key | Values |
|---|---|
| `rarity_bucket` | `1`, `2-4`, `5-19`, `20+` games (shared with the epoch line) |
| `family` | `RuleFamily` |
| `policy` | `on` / `off` (`random_seat`) |
| `decision` | `real` / `what-if` / `unknown` (`what_if`) |

`measure` returns per-record losses and predictions; a pure `breakdowns.py` groups them by the keys
above and by text.

## Probe tooling

### ProbeSet (new)

| Field | Meaning |
|---|---|
| `corpus_digest` | the curated corpus manifest's digest |
| `items` | `[{key: ProvenanceKey, family, target, label}]` for line-level probes |
| `games` | `{card_disjoint: [game_id], game_disjoint: [game_id]}` |
| `record_items` | `[{record_id, stratum, family, target, label}]` for board-dependent probes |
| `sweeps` | per sweep: `[{record_id, entity?}]` |
| `join_rate` | the interaction join rate measured at freeze time |
| `digest` | sha256 over the canonical JSON of the fields above |

Written to `output/effects/reports/knowledge-probes-set-<corpus digest[:12]>.json`.

### Scorecard (new)

| Field | Meaning |
|---|---|
| `checkpoint`, `probe_set_digest`, `date` | identity |
| `families[f].targets[t].rungs[r]` | `{linear, mlp}` scores per stratum; rungs `0, 1, 1w, 1o, 2, 3` |
| `families[f].targets[t].share` | `{linear, mlp}` with 95% CI, or `null` under the minimum gap |
| `sweeps[s]` | `[{value, prediction}]` |
| `ablation[field][replacement][scope]` | loss increase |
| `runtime` | `{wall_seconds, peak_gpu_bytes}` for the run, checked against SC-011 |
| `per_layer` | optional, `--per-layer` |
| `method_c` | optional, `--method-c` |
