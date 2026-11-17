# Effect-record format

What a reader or writer of effect records has to know that no single source file states, because the writer is Java and the readers are Python. The full contracts:

- `specs/023-ability-effect-model/contracts/record-schema.md` (base schema).
- `specs/024-ability-effect-model-gen2/contracts/record-schema-delta.md` and `specs/025-ability-effect-model-gen3/contracts/record-schema-delta.md` (each generation's changes).
- Python side: `domain/records.py`, `infrastructure/record_codec.py`, and `domain/shard_generation.py` for generations.

## Shards

- `{run_id}.{worker}-{lifetime}.jsonl.gz` at the top level of the collecting run's directory (its `--effect-records`, a `records` or `scratch` run of the layout), one JSON record per line, append-only.
- One shard per worker JVM lifetime, not per worker slot and not one shared file. Records are large enough that concurrent appends would interleave mid-record, and the supervisor replaces worker JVMs hundreds of times per run.
- Each file is a concatenation of complete gzip members, one per block of 256 records and one more at each game's end, after its game-end marker. A worker killed mid-write truncates the last member and the reader stops there, so a killed worker loses only the game in progress.
- A game that ended writes a `turn_action` / `game_end` marker as its last record. A gen-3 game whose shard holds no marker was cut; `build-corpus`, the collectors' counters and `rule-counts` take nothing from it (`domain/game_ends.py`).
- Damage is located, never skipped silently:
  - A last member that does not decompress ends the read with a warning naming the shard and byte.
  - A damaged member with another after it stops the read with `CorruptShardError`, because reading on would drop every later member unsaid.
  - A line that does not decode raises `RecordDecodeError`, naming the shard and the line.
  - `rule-counts`, `field-coverage`, `train-effect-model` and `evaluate-effect-model` exit 1 on either. `validate-corpus` reports it as the failed finding "every shard line decodes" and reads the other shards on. The collectors count a damaged shard up to the damage and list it in `run.json` as `corrupt_shards`, since one damaged shard must not stop a run.
- Every reader that decodes records checks each raw record against the wire's closed lists (`domain/wire_closed_lists.py`, each pinned to its source by a test). A value outside its list decodes as written, is counted in the process's `shard_io.DecodeTally`, and is logged at its first sighting.
- Plain `*.jsonl` still reads: older shards cannot be regenerated.
- Discovery recurses. A depleted run and a full-strength one can sit in subdirectories and read as one corpus, and anything else placed under the tree is read as corpus too.
- `record_id` is `{run_id}.{worker}-{lifetime}.{counter}` and `game_id` is `{run_id}.{worker}-{lifetime}.{game counter}`. Both carry the worker slot because workers count independently, and the lifetime because each JVM counts from zero. `game_id` is the join key for a checkpoint's recorded split, so a repeated one merges unrelated games. Corpora written before the lifetime segment existed stay readable and report an empty lifetime.

## Envelope

The envelope is frozen within a generation: a later capability makes new `kind` values reachable and adds members, and never redefines a field. `tests/unit/effects/domain/test_schema_compatibility.py` enforces the four compatibility rules; there is no version stamp.

| Member | Notes |
|---|---|
| `record_id`, `run_id`, `timestamp`, `game_id` | ids as above; timestamp ISO 8601 UTC |
| `generation` | `3` on gen-3, absent before; any other value is refused at decode (`UnknownGenerationError`) |
| `kind` | `resolution`, `rewrite`, `continuous`, `combat`, `trigger`, `playability`, gen-3 `turn_action` |
| `moment` | resolution kind only |
| `subkind` | playability kind; on `turn_action` one of `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`, `lore_counter`, `game_end` (the game-end marker, which carries `records_withheld`) |
| `link_id` | joins a cost half with its effect halves; see below |
| `mirror_of`, `variant_of` | the live record a fork mirrors; the variant source |
| `mode` | `patched` or `degraded` |
| `interventional`, `fork`, `synthetic` | flags; signatures below |
| `random_seat` | gen-2 on: `actor_player` is the random seat |
| `what_if` | gen-2 `attackers`/`blockers` only; gen-3 `attackers` records carry none |
| `aimed_at` | gen-3: the rule ids a record was collected for |
| `directed` | gen-3 directed forks only: `{variant, operation}` |
| `copy_dropped` | gen-3 fork records only: what the copier left out of the fork's copy, by kind; absent on a live record and on a lossless copy; it gates nothing |
| `actor_player` | on `playability` records of gen-2 on, the deciding player |
| `ability` | provenance keys, a charm mode's with an `option`; absent for `combat` and `playability` |
| `state`, `payload` | the board snapshot and the kind's payload |
| `curation` | curated shards only, written by `build-corpus` |

- `mode`, `interventional`, `fork`, `synthetic`, `random_seat`, `what_if`, `generation`, `aimed_at`, `directed`, `copy_dropped` and `curation` are collection and curation metadata and never reach the model.
- A `turn_action` record has no `ability`, its payload is its events, and no curated stratum holds it.

## Generations

- A reader takes a record's generation from `generation`, else from the presence of `random_seat` (gen-2), else gen-1. A gen-1 record reads `random_seat = false` and `what_if` unknown.
- Every reader refuses a records set mixing generations. `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the probes read gen-3 only; `rule-counts` reads gen-2 or gen-3.
- A curated shard keeps its source's generation, because the codec writes a record back as it read it.
- `actor_player` on a `playability` record is the one named exception to the frozen schema: gen-1 wrote the active or attacking player, later generations the deciding player.
- Relative to gen-2, a gen-3 field may change meaning or go away; the 025 contract's Gen-3 section lists each.

## Links and halves

- A resolution is a cost half and an effect half sharing a `link_id`.
- `link_id` is absent where a half has no partner: a mana record, the rescue record, an interventional half, and on gen-3 a triggered ability's or a copy's effect half, which nobody cast.
- A gen-3 cost half always carries one, and has no effect half when its item was countered.
- On a patched worker a modal resolution writes one cost half through the charm's root key and one effect half per chosen mode through root key + `option`, all sharing one `link_id`. A fizzled or declined charm writes one empty half per chosen mode. A degraded worker writes one root effect half.

## Flag signatures

| Record | `fork` | `interventional` | Other |
|---|---|---|---|
| Observed | false | false | `aimed_at` absent, or the run's rules on `collect-coverage` and `collect-variants` (`[]` when none) |
| Interventional resolution | true | true | effect half alone, no `link_id` |
| Damage-step probe | true | false | `mirror_of`; payload `probed_keyword`, `probed_entity` |
| Protection fork | true | true | any kind; cost halves linked as live ones |
| Directed fork | true | false | `directed` set, `aimed_at = [rule]`; `mirror_of` null, since a phase boundary leaves no live record open; its records may hold the one play its variant forced (FR-010b), unmarked |

A protection copy written by `build-corpus` is a synthetic record `<record_id>~pc<i>`.

## Gen-3 members worth knowing

- A snapshot lists every line of a card (`lines`), sets base P/T and boosts apart, and carries the facts of `rules/gen3-fields.json`, `links`, and extra card entities with a `role`.
- A line's `granted` means not printed on the card, and a granted line alone carries `grant`, why: `static`, `effect`, `counter` or `splice` (written by `LineGrant.java`, read as `state_snapshot.LineGrant`, a closed list). A copy prints what it copies, and a card prints a line it grants itself (a Class level's, listed beside its `Class:N` line under the same key), so neither is granted. The model's line flag is "granted by an effect", `grant == effect` (`effect_head_input.slot_lines`), never guessed from the key's trait kind. It says why, not how long: an `effect` grant lasts as long as its effect, and an emblem's own lines are `effect` and permanent. A granted line with no `grant`, from a record collected before the member existed, flags nothing beyond `granted`. The cases: `contracts/record-schema-delta.md` (025), the `lines` members.
- A referenced card is always the game's own object for it: a dies trigger's host is its graveyard card, and its last-known copy is only `L<id>`.
- `refs.x` is the paid X, else an `SVar:X` the acting chain names, read once before the resolution and never a `Count$Random`.
- `refs.targets` holds every target of the acting chain, root first; on an effect half, those left after Forge's illegal-target check. `refs.targets_at_cast` holds the same links' targets before that check, a target gone since as an `L<id>` entity of role `last_known`. `refs.targets_on_stack` names the refs that are a spell or ability on the stack, by its card.
- A provenance key's face is numbered over the converter's faces, never a face-down or empty-room state.
- A gen-3 resolution half in the superseded payload shape (a cost half without `events` or with an `outcome`, an effect half without `outcome`) is refused at decode (`record_codec.SupersededPayloadError`); the same shape on a gen-1 or gen-2 record reads. A hand-built gen-3 test record is in the current shape: `tests/unit/effects/builders.make_record` builds gen 3 by default.
- Decision candidates are enumerated by the collector itself at every priority stop and written as a two-stage sample (`--playability-rate`, then `--zone-candidate-rate` or `--plain-candidate-rate`). The sampling is what keeps a gen-3 game near 350 KB gzipped.
- Attackers records are one per creature.
- A rewrite record names one line in both `replaced_by` and `buffered` only when its replacement's own `ReplaceDamage`, `ReplaceSplitDamage` or `ReplaceEffect` ability ran in the event's place and Forge buffered that ability's sub-ability (Forcefield); any other buffered record has `replaced_by: []`. Two different lists, or a non-empty `replaced_by` beside `buffered: []`, is a contradiction `validate-corpus` fails. The shapes: `contracts/record-schema-delta.md` (025), the rewrite-record bullet.
- `validate-corpus` checks every gen-3 field is on every record that owes it, and `field-coverage` that each varies.

## Dropped-event counts

- `{run_id}.{worker}-{lifetime}.dropped-events.tsv` beside the shard, written only under `--count-dropped-events`: one `event_type<TAB>phase<TAB>count` line per pair per game, appended as each game ends, because a worker that is killed runs no shutdown hook.
- Lines repeat across games and are summed by the supervisor, which skips a line cut short by a kill.
- The suffix is not a shard suffix, so no shard reader takes the file for one.
