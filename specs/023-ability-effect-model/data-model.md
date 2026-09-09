# Data Model: Ability effect model

**Feature**: `023-ability-effect-model` | **Date**: 2026-09-06
Derived from [spec.md](spec.md); the authority for every field is
[`../2026-09-05-ability-effect-model.md`](../2026-09-05-ability-effect-model.md).

Two families of entity: the **corpus** entities that are collected and persisted, and the **model**
entities that exist only at training and inference time. The dividing line matters — the corpus is
append-only and expensive to rebuild, so it stores data rather than tensors, and every tensor layout
below is derived at training time from the corpus entities above it.

## Corpus entities

### EffectRecord

The unit of training. One observed game event or decision point.

| Field | Type | Rules |
|---|---|---|
| `record_id` | str | `{run_id}.{worker}-{lifetime}.{counter}`; unique across the run's shards |
| `run_id` | str (UUID) | the collecting invocation |
| `timestamp` | str | ISO 8601 UTC |
| `game_id` | str | `{run_id}.{worker}-{lifetime}.{game counter}`; the join key for a checkpoint's recorded split |
| `kind` | enum | `resolution` \| `rewrite` \| `continuous` \| `combat` \| `trigger` \| `playability` |
| `moment` | enum? | `resolution` kind only: `activation` \| `resolution` |
| `subkind` | enum? | `playability` kind only: `decision` \| `attackers` \| `blockers` |
| `link_id` | str? | joins the two halves of a resolution pair; absent where a half has no partner |
| `mirror_of` | str? | `fork = true` only: the `record_id` of the same-game real record mirrored |
| `variant_of` | str? | `synthetic = true` only: the card the perturbed script derives from |
| `mode` | enum | `patched` \| `degraded` |
| `interventional`, `fork`, `synthetic` | bool | collection metadata |
| `actor_player` | player ref | acting controller; active player for `combat`; the deciding player for `playability` |
| `ability` | ProvenanceKey[]? | the acting line; the chosen `option` line on modal resolutions; absent for `combat` and `playability` |
| `ability_unresolved` | str? | collection metadata: why `ability` came back empty on a kind that does name a line — `engine_effect` \| `no_card_state` \| `unknown_kind` \| `unindexable` |
| `state` | StateSnapshot | pre-event |
| `payload` | per-kind object | see below |

**Invariants**

- `mode`, `interventional`, `fork`, `synthetic` and `ability_unresolved` are metadata and never reach
  the model.
- A probe record is `kind = combat`, `fork = true`, `interventional = false`.
- An interventional resolution is `kind = resolution`, `interventional = true`, `fork = true`, no
  `link_id`, and has no activation partner.
- `record_id` and `game_id` carry the worker slot, because workers count independently, and the JVM
  lifetime, because the pool recycles workers on a timer and each new JVM counts from zero.
- Records are append-only. Later stages add kinds and snapshot tiers; they never redefine a field.

**State transitions**: none. A record is immutable once written.

### StateSnapshot

| Block | Contents |
|---|---|
| `global` | turn, phase, active player, priority player, stack size, combat substep, command-zone emblems |
| `players[]` | id, life, hand/library/graveyard sizes, poison, energy, this-turn counters, floating mana by colour, untapped production by colour |
| `entities[]` | identity (name, face, copy-source, token-script id, zone, controller); computed characteristics (type line, colours, mana value, P/T decomposed into base / boosts / counters); board state (tapped, sickness, damage, counters, combat status, attached-to, face-down); granted abilities, split into `granted_attached` (provenance keys) and `granted_temporary` (change-table grants: provenance keys where they resolve to a line, bare keyword strings where they do not); stack extras (own targets, announced per-target amounts, up-to-N counts) |
| `refs` | targets, source, chosen modes, X and announced values, resolution-time engine choices |
| `pending_event` | `rewrite` and `trigger` only: reference to the payload's incoming event |

**Rules**

- Characteristics are computed (post-layer), never printed. The one exception: a `continuous` record's
  snapshot has the acting static's own contributions removed from every layer channel it wrote.
- Inclusion tiers apply in order — (1) referenced objects, each in whatever zone it sits with that zone
  recorded; (2) core (global, battlefield, command-zone effect cards); (3) unreferenced stack;
  (4) unreferenced hand and graveyard. Tiers 1–2 from stage one, 3 from stage two, 4 from stage three.
  A reader treats an absent tier as uncollected, never as empty.
- Perspective is **not** stored. Controllers are absolute ids; mine/opponent tags derive at training
  time relative to `actor_player`.

### Event

`{type, subjects (refs), params, duration, attributed_to}`. The type vocabulary is the union of Forge
trigger types, bus events, and bracket diffs; the canonical member list and per-type field
normalization live in `effects/domain/event_schema.py`, guarded by a checked-in completeness test that
maps every Forge effect API class to a covered type or an explicit exclusion.

`attributed_to` is tri-state: a sub-ability chain index (`"2"`), the sentinel `"root"` where the acting
line's own clause acted, or the sentinel `"unresolved"` where a clause was sought and the pointer named
nothing on this chain. `null` is none of the three and means **unknown** — it is what a writer that
predates the sentinels left behind, and it covered `root` and `unresolved` at once, which made a dead
attribution channel indistinguishable from a working one.

### Per-kind payloads

| Kind / moment | Payload |
|---|---|
| `resolution` / `activation` | costs paid (mana by colour, permanents tapped, life, cards sacrificed/discarded/exiled); `outcome` ∈ {resolved, fizzled, partially_fizzled, declined, countered}. Only `resolved` and `partially_fizzled` have a linked effect half |
| `resolution` / `resolution` | attributed event list; attribution granularity is the sub-ability, root line as fallback |
| `rewrite` | incoming event, outgoing event (parameter maps deep-copied at the hook) |
| `continuous` | per-entity contributions (P/T boost, keywords, types, colours, name), coalesced per (game, static, board hash) |
| `combat` | declared attackers, block assignments, damage-assignment choices (inputs), damage-step event list (outcome); one record per damage step |
| `trigger` | the event, fired flag; non-fired negatives drawn from same-event-type evaluations at ~1:1 |
| `playability` / `decision` | per candidate: ability key, rules verdict, legal-target refs, cost after adjustment, responsible static |
| `playability` / `attackers` | legal-attacker refs; responsible static per forbidden attacker |
| `playability` / `blockers` | anchored attacker ref, per-entity legal-blocker bits, responsible static, `min_blockers` |

### ProvenanceKey and the sidecar

`ProvenanceKey = (script_file, face, trait_kind, index_within_kind)`. The script-file component
includes its tree (`cardsfolder`, `tokenscripts`, `variant-scripts`), since one filename occurs in more
than one. Converted-line ordinals are never a key.

The **sidecar** (`<name>.provenance.json`, beside the converted `.txt`) carries per rendered line: the
provenance key list (several where a line merged several traits; a deduplicated trait maps to no line),
the sub-ability links as index paths below the trait, the trait's script API type and parameter-key
list and script text, and prose role spans tagged `cost` | `effect` | `trigger-condition` |
`target-spec`.

**Relationship**: `EffectRecord.ability` → `ProvenanceKey` → sidecar line → converted `.txt` line →
ability cache row. That chain is the join the whole feature rests on.

### KeywordDefinition

`keyword → {reminder_template, generated_script?}`. The reminder template exists for every keyword; the
generated script exists only for the script-generated majority and only from stage four.

### VariantScript

A perturbed copy of one Forge card script — one parameter changed. Lives in
`output/effects/variant-scripts/` with a sidecar of its own, exists on the script surface only, and is
never converted to prose. A variant of a held-out card is held out with it.

## Model entities

### AbilityEmbedding `e`

The bottleneck: one fixed-width vector (`--e-dim`, default 64) per unique ability line, pooled through
a `[CLS]` token, with additive Gaussian noise (`--e-noise`) during training.

**Cache layout**: one file per source card under `output/effects/abilities/<tree>/…`, holding a
`float32` array of shape `(n_lines, e_dim)`, row-aligned with that source's sidecar — rendered lines for
a converted tree, script lines for the variant tree. Consumers read the cache-file + sidecar pair.

### EffectHeadInput

Derived at training time from a `StateSnapshot`. One token per slot, position ids resetting at each
`[CARD]`:

```
[GLOBAL] [ACT] [PLAYER] [PLAYER] [CARD] e e ... [CARD] e ...
```

| Slot | Carries |
|---|---|
| `[GLOBAL]` | turn/phase/priority, whose turn, stack size, combat substep, emblems, record kind/moment/subkind flag |
| `[ACT]` | acting line's `e`, announced values, resolution outcome flag; empty for `combat` and the `attackers`/`blockers` subkinds |
| `[PLAYER]` | the snapshot's player fields plus targeted flag |
| `[CARD]` | structured features plus overlay |
| ability tokens | the entity's printed and attachment-granted line `e` vectors; temporary grants ride the overlay |

Numeric overlay and player scalars enter as raw value plus a log1p copy; nothing is binned. Context
ability tokens drop at `--context-dropout`.

### Output heads

- **Per-entity head** — one shared head mapped over every `[CARD]` *and* `[PLAYER]` output: an
  affected/unaffected gate, then conditional field groups (permanent fields; player fields — life
  delta, cards drawn/discarded/milled, library events, mana delta; legality bits).
- **Created-objects head** — at `[GLOBAL]`, K = 4 slots of
  `{present, scripted flag, token-script id or characteristics, count}` in canonical order, plus an
  overflow flag.
- **Verdict head** — at `[ACT]`: playability verdict bits, predicted cost paid, trigger-fired bit.

### Checkpoint

The saved artifact keeps the encoder, the effect-head trunk, and the three output heads above; the
MLM, script-API, and paired-encoding heads are training-only and are **filtered out at save time**.

Beyond weights, a checkpoint records the artifacts needed to make a run reproducible and its gates
honest: the held-out card list, the `game_id` set across both validation strata, the `--vocab-path` and
`--keyword-definitions` paths, content hashes of both files, and the keyword withheld from training
(or null). `evaluate-effect-model` reads all of it from here and recomputes none of it, because the
corpus grows between runs and because the zero-shot check must measure the keyword the model was
actually trained without, not one an operator remembers choosing.

## Validation rules

| Rule | Enforced where |
|---|---|
| A trailing partial line in a shard is skipped, not an error | record reader |
| An event attributed to an unmapped sub-ability link falls back to the root line | collector |
| A trait deduplicated away maps to no line; its key is listed in the sidecar's `dropped_keys`, and a record naming it is kept with no line to join to | converter |
| A record naming a provenance key in neither `lines` nor `dropped_keys` fails loudly (the sidecar does not describe the card the record was collected against) | training-time join |
| The pooled-`e` scorer smoke test writes only into a scratch copy of the cards folder, never `output/cardsfolder/` | evaluator |
| A fork whose copy-score check disagrees before perturbation is discarded, still counting against budget | collector |
| Vocabulary or keyword-definition hash mismatch fails fast | every inference command |
| A `--variant-checkpoint` whose split or hashes differ from `--checkpoint` fails fast | evaluator |
| Every game holding a record that names a held-out card is excluded from training entirely | split derivation |
| Records outside the checkpoint's recorded `game_id` set are ignored, never re-stratified | evaluator |
| Coverage and variant matches never write `match-outcomes.txt` or `cards-played.txt` | collectors |
