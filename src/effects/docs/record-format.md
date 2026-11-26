# Effect-record format

What a reader or writer of effect records has to know that no single source file states, because the writer is Java and the readers are Python. The full contracts:

- `specs/023-ability-effect-model/contracts/record-schema.md` (base schema).
- `specs/024-ability-effect-model-gen2/contracts/record-schema-delta.md` and `specs/025-ability-effect-model-gen3/contracts/record-schema-delta.md` (each generation's changes).
- Python side: `domain/records.py`, `infrastructure/record_codec.py` (the envelope and the record; each block's converters are in the `record_codec_*.py` modules beside it, and every reader goes through `record_codec`), and `domain/shard_generation.py` for the generation check.

## Shards

- `{run_id}.{worker}-{lifetime}.jsonl.gz` at the top level of the collecting run's directory (its `--effect-records`, a `records` or `scratch` run of the layout), one JSON record per line, append-only.
- One shard per worker JVM lifetime, not per worker slot and not one shared file. Records are large enough that concurrent appends would interleave mid-record, and the supervisor replaces worker JVMs hundreds of times per run.
- Each file is a concatenation of complete gzip members, one per block of 256 records and one more at each game's end, after its game-end marker. A worker killed mid-write truncates the last member and the reader stops there, so a killed worker loses only the game in progress.
- A game that ended with its outcome and its end-of-game records written writes a `turn_action` / `game_end` marker as its last record. A game whose shard holds no marker was cut; `build-corpus`, the collectors' counters and `rule-counts` take nothing from it (`domain/game_ends.py`).
- Damage is located, never skipped silently:
  - A last member that does not decompress ends the read with a warning naming the shard and byte.
  - A damaged member with another after it stops the read with `CorruptShardError`, because reading on would drop every later member unsaid.
  - A line that does not decode raises `RecordDecodeError`, naming the shard and the line.
  - `rule-counts`, `field-coverage`, `train-effect-model` and `evaluate-effect-model` exit 1 on either. `validate-corpus` reports it as the failed finding "every shard line decodes" and reads the other shards on. The collectors count a damaged shard up to the damage and list it in `run.json` as `corrupt_shards`, since one damaged shard must not stop a run.
- Every reader that decodes records checks each raw record against the wire's closed lists (`domain/wire_closed_lists.py`, each pinned to its source by a test). A value outside its list decodes as written, is counted in the `shard_io.DecodeTally` the reader is handed (`iter_shard_lines`, `read_shard` and `read_window` take `tally=`; every other reader tallies nothing, and its damaged last member is only logged), and is logged at its first sighting in the process. A caller that reports the tally owns one per read or per worker: `corpus_survey.WorkerContext.decode_tally` (`ReadGaps.take` hands it to the parent with each shard's result, which sums them), `rule_counts.count_shard`'s `RuleCounts.decode`, `ShardSource.decode_tally` for `validate-corpus`. A damaged last gzip member is walked and split apart: `shard_members.gzip_member_chunks` yields the decompressed chunks and `MemberDamage`; `shard_io` cuts lines and decides the damage's fate.
- Plain `*.jsonl` still reads: older shards cannot be regenerated.
- Discovery recurses. A depleted run and a full-strength one can sit in subdirectories and read as one corpus, and anything else placed under the tree is read as corpus too.
- `record_id` is `{run_id}.{worker}-{lifetime}.{counter}` and `game_id` is `{run_id}.{worker}-{lifetime}.{game counter}`. Both carry the worker slot because workers count independently, and the lifetime because each JVM counts from zero. `game_id` is the join key for a checkpoint's recorded split, so a repeated one merges unrelated games. Corpora written before the lifetime segment existed stay readable and report an empty lifetime.

## Envelope

The envelope is frozen within a generation: a later capability makes new `kind` values reachable and adds members, and never redefines a field. `tests/unit/effects/domain/test_schema_compatibility.py` enforces the four compatibility rules; there is no version stamp.

Which members the envelope, each snapshot block, each kind's payload, a contribution, a candidate and an event carry, and which of them the writer puts on every instance (`always`) rather than only `when` a condition holds, is one table, `domain/gen3_members.py`, read off the Java writer. The gen-3 decode reads every `always` member by index and defaults none (§ Generations). A member the writer adds or drops changes that table in the same change: `tests/unit/effects/application/test_gen3_members.py` fails on a fixture member the table lacks and on an `always` member a fixture instance lacks, and `tests/unit/effects/infrastructure/test_record_codec_required_members.py` on an `always` member the decode reads as optional.

| Member | Notes |
|---|---|
| `record_id`, `run_id`, `timestamp`, `game_id` | ids as above; timestamp ISO 8601 UTC |
| `generation` | `3` on every record; see § Generations |
| `kind` | `resolution`, `rewrite`, `continuous`, `combat`, `trigger`, `playability`, gen-3 `turn_action` |
| `moment` | resolution kind only |
| `subkind` | playability kind; on `turn_action` one of `draw`, `cleanup_discard`, `untap`, `phasing`, `day_night`, `lore_counter`, `fork_edit` (a protection or directed fork's premise, on fork records only: `forks.md` § Directed forks), `game_end` (the game-end marker, which carries `records_withheld`) |
| `link_id` | joins a cost half with its effect halves; see below |
| `mirror_of`, `variant_of` | the live record a fork mirrors; the variant source |
| `mode` | `patched` or `degraded` |
| `interventional`, `fork`, `synthetic` | flags; signatures below |
| `fork_kind` | fork records only, right after `fork`: `intervention`, `probe`, `protection` or `directed`, the kind of fork that wrote the record (`ForkKind.java`); absent on a live record |
| `random_seat` | every record: `actor_player` is the random seat |
| `what_if` | `blockers` records only; an `attackers` record carries none |
| `aimed_at` | gen-3: the rule ids a record was collected for |
| `directed` | gen-3 directed forks only: `{variant, operation}` |
| `copy_dropped` | gen-3 fork records only: what the copier left out of the fork's copy, by kind; absent on a live record and on a lossless copy; it gates nothing |
| `actor_player` | on `playability` records, the deciding player |
| `ability` | provenance keys, a charm mode's with an `option`; absent for `combat` and `playability` |
| `state`, `payload` | the board snapshot and the kind's payload |
| `curation` | curated shards only, written by `build-corpus` |

- `mode`, `interventional`, `fork`, `fork_kind`, `synthetic`, `random_seat`, `what_if`, `generation`, `aimed_at`, `directed`, `copy_dropped` and `curation` are collection and curation metadata and never reach the model.
- A `turn_action` record has no `ability`, its payload is its events, and no curated stratum holds it.

## Generations

- Every reader reads gen-3 records only. The decode (`record_codec.record_from_dict`, which every reader goes through) refuses a record with no `generation` member as `OlderGenerationError`, naming it gen-2 when it carries `random_seat` and gen-1 otherwise, and a `generation` other than the integer 3 as `UnknownGenerationError`; `shard_io` prefixes the shard and line. Both are `UnacceptedGenerationError`, which every command maps to exit 1. Gen-1 and gen-2 read several fields differently (`actor_player` on a `playability` record among them), with no per-field version to tell them apart.
- A gen-3 record whose envelope, snapshot block, payload, contribution, candidate, event or other site lacks a member `gen3_members.ALWAYS` owes that site is refused as `gen3_members.MissingMemberError`, naming each as `<site>.<member>` (`payload<combat>.blocks`); `shard_io` wraps it in `RecordDecodeError` with the shard and line, so a reader exits 1 on it as on any undecodable line. A default would read a writer that dropped `fork` as a live record, unsaid. The members of a member's own nested objects (an entity's `history`, `combat` or `granted_temporary`, a cost half's `costs`) are no site of the table and keep their defaults.
- `shard_io.require_generation` checks each shard's first record before a command reads the set, so `build-corpus`, `train-effect-model`, `evaluate-effect-model`, the probes, `rule-counts` and `scripts/rule_breadth.py` refuse an older set before any work; an older record further into a shard is refused where it is decoded. `validate-corpus` and `field-coverage` call it too, and `validate-corpus` refuses an older record further in on its raw line, naming shard and line (`shard_io.read_window`).
- The collectors refuse an older record met counting the shards already in their run directory or a `--count-from` one, with exit 1 before any game is played: it is no damage to count past, since every reader refuses the set it is in.
- A curated shard is gen-3, because the codec writes `generation` and `random_seat` on every record.
- What each field means in gen 3, where it differs from the earlier generations: the 025 contract's Gen-3 section.

## Links and halves

- A resolution is a cost half and an effect half sharing a `link_id`.
- `link_id` is absent where a half has no partner: a mana record, the rescue record, an interventional half, and a triggered ability's or a copy's effect half, which nobody cast.
- A gen-3 cost half always carries one, and has no effect half when its item was countered.
- On a patched worker a modal resolution writes one cost half through the charm's root key and one effect half per chosen mode through root key + `option`, all sharing one `link_id`. A fizzled or declined charm writes one empty half per chosen mode. A degraded worker writes one root effect half.

## Flag signatures

| Record | `fork` | `fork_kind` | `interventional` | Other |
|---|---|---|---|---|
| Observed | false | absent | false | `aimed_at` absent, or the run's rules on `collect-coverage` and `collect-variants` (`[]` when none) |
| Interventional resolution | true | `intervention` | true | effect half alone, no `link_id` |
| Damage-step probe | true | `probe` | false | `mirror_of`; payload `probed_keyword`, `probed_entity`; a probe of `first_strike` or `double_strike` at `combat_substep` `first_strike` spans two steps: its counterpart is the real first-strike record, which `mirror_of` names, and the regular-step record after it |
| Protection fork | true | `protection` | true | any kind; cost halves linked as live ones |
| Directed fork | true | `directed` | false | `directed` set, `aimed_at` its registration's rules (`forks.md` § Directed forks); `mirror_of` null, since a phase boundary leaves no live record open; its records may hold the one play its variant forced (FR-010b), unmarked |

A fork's record may hold `player_lost` and `player_won`, the outcome of the fork's own game, never the live game's: `forks.md` § Rules for every fork.

A protection copy written by `build-corpus` is a synthetic record `<record_id>~pc<i>`.

- `EffectRecord.origin` (`RecordOrigin`) reads what produced a record off `synthetic` and `fork_kind` alone: `SYNTHETIC` for a synthetic record, `LIVE` for an observed one, else its `fork_kind`'s (`FORCED`, `PROBE`, `PROTECTION_FORK`, `DIRECTED_FORK`). An intervention and a protection fork share their flags, so a reader that infers the kind from them confuses the two.
- `fork_kind` decodes into `records.ForkKind`, and a value outside it is refused. A synthetic record copies a live one and carries none, and a record whose `fork_kind` disagrees with its row above is refused (`EffectRecord._check_fork_kind`). A fork record without one predates the member: it is counted as `fork_kind <absent>` (`wire_closed_lists`), and its `origin` raises rather than guess.
- `EffectRecord.is_game_end_record` is the game-end record of the 025 contract's `player_lost` row: an effect half with `ability` empty and `ability_unresolved` null, no `link_id` and `outcome` null. It is not the `game_end` marker (`records.is_game_end`), and a mana record is not one, even unkeyed: its empty `ability` carries a reason. `validate-corpus` counts a game-end record apart and not as a silent empty ability.

## Gen-3 members worth knowing

- A snapshot lists every line of a card (`lines`), sets base P/T and boosts apart, and carries the facts of `rules/gen3-fields.json`, `links`, and extra card entities with a `role`.
- A line's `granted` means not printed on the card, and a granted line alone carries `grant`, why: `static`, `effect`, `counter` or `splice` (written by `LineGrant.java`, read as `state_snapshot.LineGrant`, a closed list). A copy prints what it copies, except a keyword a token or spell copy's copy effect added (`effect`, no key), a mutated permanent prints every merged card's lines, and a card prints a line its own static grants it (a Class level's, listed beside its `Class:N` line under the same key), so neither is granted. A line the card's own resolved effect grants (a manland's animated trigger or activated ability) is `effect`, though it keys to the animation's line. The model's line flag is "granted by an effect", `grant == effect` (`slot_lines.slot_lines`), never guessed from the key's trait kind. It says why, not how long: an `effect` grant lasts as long as its effect, and an emblem's own lines are `effect` and permanent. A granted line with no `grant`, from a record collected before the member existed, flags nothing beyond `granted`. The cases: `contracts/record-schema-delta.md` (025), the `lines` members.
- A referenced card is always the game's own object for it: a dies trigger's host is its graveyard card, and its last-known copy is only `L<id>`. One the game no longer holds, a token that ceased to exist (a Food sacrificed for its own ability), keeps its `E<id>`, the id `refs.source`, its lines and the events name it by, and is written as it last existed, role `last_known`, `zone` and `order_rank` null; links to it are written as to any entity.
- `combat.attacking` is the defender's ref, as the `attacks` link names it: `P<id>`, or `E<id>` for a planeswalker or battle; `E-1`, an entity no board holds, for the placeholder an attacker attacks once its planeswalker or battle has left (506.4c). Readers join it to `players` and `entities` through `damage_step_keywords.defender_ref`, never parse it.
- A player's `this_turn.creatures_died` counts the creatures that died under that player's control this turn (700.4, morbid's count), read off every graveyard's cards put there from the battlefield this turn as they last were (`PlayerJson`): a token that died counts, a stolen creature counts for its controller, and a bounced, blinked or exiled creature does not.
- One state is one byte string on every worker. Where Forge's order is a hash order that differs between JVMs, the writer sorts: an entity's `counters` and `granted_temporary.keywords` and a `token_created`'s `characteristics.keywords` by name, `must_block` links by the attacker's card id. A writer that took Forge's order would make one board two different inputs. Decode (`record_codec`) writes every counter type a record names in `state_snapshot.normalize_counter_type`'s spelling (`+1/+1` reads `P1P1`): an entity's `counters`, a cost half's `costs.counters`, a `counter_change`'s `counter_type` and a created token's `characteristics.counters`, keeping the record's order and adding two spellings of one type; a reader never normalizes a record's counter type.
- `refs.x` is the paid X, else an `SVar:X` the acting chain names, read once before the resolution and never a `Count$Random`.
- `refs.targets` holds every target of the acting chain, root first; on an effect half, those left after Forge's illegal-target check. `refs.targets_at_cast` holds the same links' targets before that check, a target gone since as an `L<id>` entity of role `last_known`. `refs.targets_on_stack` names the refs that are a spell or ability on the stack, by its card.
- An entity's `stack_extras` cover its item's whole chain, a targeted spell by its card. On an effect half the acting item's entity reads them from the targets as cast, as `refs.targets_at_cast` names them, so a fizzled half still shows its target and its count; read off the item after the check, they would leak the fizzle. Divided shares: a target the item still carries has its current share (a retarget moved the share to the new target), a target Forge's check removed its share as announced when the item went on the stack: `contracts/record-schema-delta.md` (025).
- A provenance key's face is numbered over the converter's faces, never a face-down or empty-room state.
- A resolution half in the superseded payload shape (a cost half without `events` or with an `outcome`, an effect half without `outcome`) is refused at decode (`record_codec.SupersededPayloadError`). A hand-built test record is in the current shape: `tests/unit/effects/builders.make_record` builds a gen-3 record and completes an effect half's `outcome` and `conditions_met`.
- Decision candidates are enumerated by the collector itself at every priority stop and written as a two-stage sample (`--playability-rate`, then `--zone-candidate-rate` or `--plain-candidate-rate`). The sampling is what keeps a gen-3 game near 350 KB gzipped.
- Attackers records are one per creature.
- A rewrite record names one line in both `replaced_by` and `buffered` only when its replacement's own `ReplaceDamage`, `ReplaceSplitDamage` or `ReplaceEffect` ability ran in the event's place and Forge buffered that ability's sub-ability (Forcefield); any other buffered record has `replaced_by: []`. Two different lists, or a non-empty `replaced_by` beside `buffered: []`, is a contradiction `validate-corpus` fails. The shapes: `contracts/record-schema-delta.md` (025), the rewrite-record bullet.
- `validate-corpus` checks that every `always` member of `domain/gen3_members.py` is on every instance of its site; it sees each raw line before the decode, so a line missing one fails both that finding, which names the member, and "every shard line decodes", which ends that shard's read at the line. `field-coverage` rows each member of the table, tagging `fixed` the ones the writer fixes (`generation`, `variant_of`, `synthetic`, `curation`, `state.tiers`) so their constancy never reads as a dead field.

## Dropped-event counts

- `{run_id}.{worker}-{lifetime}.dropped-events.tsv` beside the shard, written only under `--count-dropped-events`: one `event_type<TAB>phase<TAB>count` line per pair per game, appended as each game ends, because a worker that is killed runs no shutdown hook.
- `phase` is the engine phase in lower case (`main1`), or `none` before the game has one; a kept protection or directed fork's, or a written forced resolution's, count carries the fork's kind before it (`protection:main1`, `intervention:main1`).
- Lines repeat across games and are summed by the supervisor, which skips a line cut short by a kill.
- The suffix is not a shard suffix, so no shard reader takes the file for one.
