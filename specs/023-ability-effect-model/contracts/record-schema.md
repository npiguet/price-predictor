# Contract: effect record schema

**Feature**: `023-ability-effect-model`
Authority: [`../../2026-09-05-ability-effect-model.md`](../../2026-09-05-ability-effect-model.md) § Corpus.

This schema is **fixed before stage-one collection begins**. Later stages widen the corpus — new
`kind` values become reachable, new snapshot tiers appear — but never redefine a field. That is what
lets a stage-one corpus stay trainable alongside stage-four records.

## File layout

```
output/effects/records/
  {run_id}.{worker}-{lifetime}.jsonl    one JSON record per line, append-only;
                                        one file per worker JVM lifetime
```

Readers load every `*.jsonl` in the directory and skip a trailing partial (non-newline-terminated)
final line — a JVM crash mid-write is expected, not exceptional.

## Envelope

```jsonc
{
  "record_id":      "3f2a…-uuid.4-mk3p9x2q.10237",  // {run_id}.{worker}-{lifetime}.{counter}
  "run_id":         "3f2a…-uuid",
  "timestamp":      "2026-09-06T15:31:12.152018Z",
  "game_id":        "3f2a…-uuid.4-mk3p9x2q.812",    // {run_id}.{worker}-{lifetime}.{game counter}
  "kind":           "resolution",           // resolution|rewrite|continuous|combat|trigger|playability
  "moment":         "resolution",           // resolution kind only: activation|resolution
  "subkind":        null,                   // playability kind only: decision|attackers|blockers
  "link_id":        "…",                    // joins a resolution pair; absent where no partner
  "mirror_of":      null,                   // fork records: the real record mirrored
  "variant_of":     null,                   // synthetic records: the source card name
  "mode":           "degraded",             // patched|degraded
  "interventional": false,
  "fork":           false,
  "synthetic":      false,
  "actor_player":   "P0",
  "ability":        [ /* ProvenanceKey[] */ ],
  "ability_unresolved": null,               // why `ability` is empty, where a line was sought
  "state":          { /* StateSnapshot */ },
  "payload":        { /* per-kind */ }
}
```

### Field rules

| Field | Rule |
|---|---|
| `record_id` | unique across the run's shards; carries the worker slot because workers count independently, and the JVM lifetime because each worker JVM counts from zero |
| `game_id` | same construction; **the join key for a checkpoint's recorded split** |
| `link_id` | absent on a half with no partner: a `fizzled`, `countered`, or `declined` cost record, or an interventional effect half |
| `mirror_of` | set only when `fork = true` and a same-game real counterpart exists |
| `variant_of` | set only when `synthetic = true` |
| `ability` | absent where no single line acts (`combat`, `playability`); the chosen `option` line on modal resolutions; several keys where the rendered line merged several traits |
| `ability_unresolved` | set only where `ability` is an empty array on a kind that does name a line; a closed vocabulary (`engine_effect` \| `no_card_state` \| `unknown_kind` \| `unindexable`) saying why no printed line was found. An empty `ability` alone cannot separate "the Monarch has no printed line in any tree" from "the resolver regressed", which is the ambiguity that hid a broken resolver for a whole collection run |
| `mode`, `interventional`, `fork`, `synthetic`, `ability_unresolved` | **collection metadata; never model inputs** |

### Flag signatures

| Record | `kind` | `interventional` | `fork` |
|---|---|---|---|
| ordinary observation | any | false | false |
| interventional resolution | `resolution` | true | true |
| damage-step probe | `combat` | false | true |

The probe's `interventional = false` is what distinguishes it from an intervention by flags alone.

## State snapshot

```jsonc
"state": {
  "global":  { "turn": 7, "phase": "…", "active": "P0", "priority": "P1",
               "stack_size": 1, "combat_substep": null, "emblems": [] },
  "players": [ { "id": "P0", "life": 14, "hand": 3, "library": 21, "graveyard": 9,
                 "poison": 0, "energy": 0,
                 "this_turn": { "creatures_died": 1, "spells_cast": 2, "lands_played": 1 },
                 "floating_mana": {…}, "untapped_production": {…} } ],
  "entities": [ { "id": "E12", "name": "…", "face": "…", "copy_source": null,
                  "token_script_id": null, "zone": "battlefield", "controller": "P0",
                  "types": {…}, "colors": [...], "mana_value": 3,
                  "pt": { "base": [2,2], "boosts": [1,1], "counters": [0,0] },
                  "tapped": false, "sick": false, "damage": 0, "counters": {…},
                  "combat": {…}, "attached_to": null, "face_down": false,
                  "granted_attached": [ /* ProvenanceKey[] */ ],
                  "granted_temporary": { "keywords": ["flying"],
                                         "abilities": [ /* ProvenanceKey[] */ ] },
                  "stack_extras": null } ],
  "refs":    { "targets": [...], "source": "E12", "modes": [...], "x": 3, "choices": {…} },
  "pending_event": null
}
```

**Rules**

- Characteristics are computed (post-layer), never printed. Exception: a `continuous` record's snapshot
  has the acting static's own contributions removed from every layer channel it wrote — the board is
  **recomputed with that static's layer entries dropped**, not stripped of the tokens it contributed.
  So a contribution can still appear in its own snapshot where a second static or the card's own
  printed text supplies it: an anthem granting trample to a creature printed with trample leaves
  trample in the board, which is the board that anthem acted on.
- Inclusion tiers, in order: (1) referenced objects — every entity-valued ref appears as an entity in
  whatever zone it sits, with that zone recorded; (2) core — global, battlefield, and command-zone
  effect cards; (3) unreferenced stack; (4) unreferenced hand and graveyard. Tiers 1–2 from stage one,
  3 from stage two, 4 from stage three. **An absent tier means uncollected, not empty.**
- Perspective is not stored. Controllers are absolute; mine/opponent derives at training time from
  `actor_player`.
- **The two grant sources are separate fields and must stay separate.** `granted_attached` holds
  attachment-granted abilities as provenance keys; `granted_temporary` holds what the timestamped
  change tables report, as provenance keys where a grant resolves to a printed line and as bare keyword
  strings where it does not (`gains flying until end of turn` names no line). They cannot be merged,
  because the model reads them differently: an entity's ability tokens are its printed and
  attachment-granted lines only, while temporary grants ride the overlay — and gate 2's perturbation
  removes a keyword from the ability token *or* from the temporarily-granted-keywords channel depending
  on which one carries it.

## Events

```jsonc
{ "type": "…", "subjects": ["E12"], "params": {…}, "duration": null, "attributed_to": "…" }
```

The type vocabulary is the union of Forge trigger types, bus events, and bracket diffs. The canonical
member list and per-type field normalization live in `src/effects/domain/event_schema.py`. A checked-in
completeness test maps every Forge effect API class to a covered type or an explicit exclusion.

### `attributed_to` is tri-state

| Value | Means |
|---|---|
| a chain index, e.g. `"2"` | the sub-ability at that position down the acting line's chain produced the event |
| `"root"` | the acting line's own top-level clause produced it |
| `"unresolved"` | a producing clause was sought and the pointer named nothing on this chain |
| `null` | **unknown** — what a writer that predates the sentinels left behind |

The first three are the tri-state; `null` is not one of them. One spelling covered "the root acted"
and "the pointer did not land" for a whole collection run, so 96.8% of resolution events said nothing
at all and a dead attribution channel was indistinguishable from a working one. The corpus is
append-only, so shards written before the sentinels keep their `null` and can never be disambiguated
— which is why the validator counts `absent` apart from the three rather than folding it into `root`.
Adding the two sentinels **widens the value set and does not redefine the field** (compatibility rule
1): it stays a nullable string, and a reader that predates them parses them as the strings they are.

### `zone_change` says where the card came from

`from_zone` is part of `zone_change`'s normalized params, not an optional extra. An event naming only
`to_zone` says a card arrived and not what left, and nothing then separates a graveyard recursion from
a token entering the battlefield. It is legitimately absent only where nothing left — a card *made*
rather than moved. `validate-corpus` measures the share that carries it, and the share whose `to_zone`
is `stack`, because a channel that is mostly casts is reporting the stack rather than the board.

## Payloads

| Kind / moment | Payload |
|---|---|
| `resolution` / `activation` | `{costs: {mana_by_color, tapped, life, sacrificed, discarded, exiled}, outcome}` where `outcome` ∈ `resolved` \| `fizzled` \| `partially_fizzled` \| `declined` \| `countered`. Only `resolved` and `partially_fizzled` have a linked effect half |
| `resolution` / `resolution` | `{events: Event[]}`; attribution granularity is the sub-ability, and the three states an event's `attributed_to` may report are above |
| `rewrite` | `{incoming: Event, outgoing: Event}` (parameter maps deep-copied at the hook) |
| `continuous` | `{contributions: [{entity, pt_boost, keywords, types, colors, name}], board_hash}` — one record per stable board |
| `combat` | `{attackers, blocks, assignment_choices, events: Event[], probed_keyword, probed_entity}` — one record per damage step; the two probe fields are set only on a damage-step probe fork and name what was stripped and from whom |
| `trigger` | `{event: Event, fired: bool}`; non-fired negatives drawn from same-event-type evaluations at ~1:1 |
| `playability` / `decision` | `{candidates: [{ability, verdict: {can_play, affordable, has_legal_target}, legal_targets, cost_after_adjustment, responsible_static}]}` |
| `playability` / `attackers` | `{legal_attackers, forbidden: [{entity, responsible_static}]}` |
| `playability` / `blockers` | `{anchor_attacker, legal_blockers, forbidden: [{entity, responsible_static}], min_blockers}` |

Verdicts are rules-level only. The AI's policy judgments ("another time", "life in danger") are never
recorded.

### A contribution's `types` and `colors`

Each is one flat list of tokens, so the operation rides on the token rather than on a field:

| Token | Means |
|---|---|
| `creature`, `W` | the static adds this type or colour |
| `-creature`, `-W` | it removes it |
| `=` | the tokens after it are a line the static **sets** rather than adds to |
| `=`, `C` | it sets the colour to none, which is what makes a permanent colourless |
| `all-creature-types` | it grants every creature type at once |
| `-all-creature-types` | it removes a whole class; likewise `-all-card-types`, `-all-super-types`, `-all-sub-types`, `-all-land-types`, `-all-artifact-types`, `-all-enchantment-types` |

Types are spelled as the snapshot spells an entity's own, and include subtypes and supertypes.
`types_gained` / `types_lost` cover the core types only, so the rest stay in the record without
reaching a head field.

A `=` reports what it sets and nothing lost. The displaced types are not in the record — the snapshot
beside it is the board after the static applied — so `types_lost` / `colors_lost` go unset, which reads
as "this record does not say" rather than as an empty set.

## Compatibility rules

### Why the id carries a lifetime

The supervisor recycles the longest-running worker every status interval and restarts crashed ones, so
an eight-hour run is hundreds of JVMs per worker slot and both counters behind an id restart at zero in
each. Stage one shipped `{run_id}.{worker}.{counter}`, which cannot satisfy this table's own uniqueness
rule under that condition: the first collected corpus reused 71–75% of its record ids and held 804
distinct `game_id` values for 31,662 games. The lifetime segment is what makes the documented rule true.

It is a change to how the id is *spelled*, not to what the field means or its type, and the number of
`.`-separated segments is unchanged — the token goes inside the middle segment, joined by `-`. So
`link_id`'s `{game_id}.link.{n}` suffix and every parse counting segments from the right are
unaffected, including the codebase's only structural parse, `EffectRecord.worker`. Corpora written
before the segment stay readable and report an empty lifetime.

### The rules

1. A field may be **added**; existing fields may not change meaning or type.
2. A new `kind` or `subkind` value may be introduced; existing values may not be repurposed.
3. Snapshot tiers are additive; absence is uncollected.
4. Nothing in this schema may become a model input that is listed above as collection metadata.
