# Contract: CLI surface

**Feature**: `023-ability-effect-model`
Authority: [`../../2026-09-05-ability-effect-model.md`](../../2026-09-05-ability-effect-model.md) § CLI summary.

`python -m effects <subcommand>`, plus one flag added to an existing `sealed` subcommand. Defaults
below are contract, not suggestions.

## `python -m effects build-vocab`

| Flag | Default | Meaning |
|---|---|---|
| `--surface` | `prose` | `prose` \| `script`; `script` adds the sidecars' script lines to the scan and switches the `--vocab-path` default |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable |
| `--vocab-path` | `models/effects/vocab.txt` (`vocab-script.txt` under `--surface script`) | separate paths so a stage-four rebuild never overwrites the vocabulary stage-one-to-three checkpoints recorded |
| `--keyword-definitions` | `output/effects/keyword-definitions.json` | included in the scan |
| `--target-size` | 5000 | post-truncates the corpus-frequency vocabulary, preserving seeded specials and domain tokens |

Seeded specials: `[PAD]`, `[UNK]`, `cardname`, `[MASK]`, `[CLS]`.

## `python -m effects extract-keyword-definitions`

| Flag | Default |
|---|---|
| `--output` | `output/effects/keyword-definitions.json` |

Writes keyword → reminder-text template for every keyword; from stage four also the generated
implementation script, captured as text at the keyword factory, for the script-generated majority.

## `python -m sealed generate-pools` / `match-outcomes` (existing commands, one added flag each)

| Flag | Default |
|---|---|
| `--exclude-cards` | none — a newline-delimited Forge canonical name list, from `effects holdout-cards`. Listed cards are omitted from every booster, redrawing within the same rarity slot so pool size and rarity structure are unchanged |

**The collection path is `match-outcomes`.** A match worker opens its own pool per match and reads
no pools file, so `generate-pools --exclude-cards` does not deplete a collection run — it depletes a
pools file, for the consumers that read one. Both share one redraw implementation.

The flag takes a plain file so that `sealed` gains no import of `effects`. A depleted run and a
full-strength one are two `match-outcomes` invocations writing into the same shard directory, and
the trainer tells their games apart by whether a record names a held-out card.

## `python -m sealed match-outcomes` (existing command, one added flag)

| Flag | Default | Meaning |
|---|---|---|
| `--effect-records` | **none** | the instrumentation opt-in. Absent, no effect records are written anywhere and the command behaves exactly as today |
| cap/budget flags | see below | shared with the other collectors |

Adding the flag must not change `match-outcomes.txt` or `cards-played.txt` in format or content.

## `python -m effects collect-coverage` (stage two)

| Flag | Default | Meaning |
|---|---|---|
| `--effect-records` | `output/effects/records/` | destination shard directory |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | repeatable; deck candidates and the coverage unit come from the `output/cardsfolder/` entry alone |
| `--split-from` | none | a checkpoint whose held-out cards are excluded from every deck |
| `--target-records` | 50 | per-card satisfaction goal |
| `--decks-per-round` | 500 | decks played as matches per round |
| `--no-progress-rounds` | 3 | consecutive rounds without a new qualifying record before a card retires |
| cap/budget flags | see below | |

Behaviour the flags do not convey:

- The castability consult **only ranks** slots. It never drops a card from deck building, because being
  in a game is the precondition a stage-three intervention forks from.
- A card is satisfied once `--target-records` records across every shard have it as the acting line's
  host, an event subject, or a referenced ref. **Sitting on the battlefield in a snapshot does not
  count**, and the unit is not resolution records specifically — a vanilla creature has no acting line.
- The run ends when every card is satisfied or retired, and reports **two residues**: cards the consult
  judged uncastable, and castable cards that never reached `--target-records`. Each retired card is
  counted under its consult verdict. Both fall to stage-three interventions; SC-006 turns on this
  report existing.

## `python -m effects validate-corpus`

Measures the corpus invariants over a shard directory and exits non-zero on any breach. Meant to be
run against the **first few minutes** of a collection pass rather than against a finished corpus: every
defect found in the first collected run was already visible in its first minute of shards, and finding
them there costs minutes instead of the eight hours that run spent producing 14.7M unusable records.

| Flag | Default | Meaning |
|---|---|---|
| `--effect-records` | `output/effects/records/` | shard directory to read |
| `--limit` | 0 (read everything) | stop after this many records; every `record_id` is held in memory while it runs |
| `--min-keyed-rate` | 0.95 | share of records naming a line that must resolve it to a printed key |
| `--max-duplicate-rate` | 0.02 | share of a kind's records that may repeat an earlier record of the same kind in the same game |
| `--max-unpaired-link-rate` | 0.02 | share of link ids that may lack one activation and one resolution half |
| `--max-names-per-game` | 120 | distinct card names one `game_id` may show before it reads as more than one game |
| `--min-cost-evidence` | 200 | activation records needed before a cost channel reading zero means the collector rather than the pool |
| `--turn-jump-tolerance` | 1 | turns a snapshot may sit below the highest already seen in its game before that reads as a game boundary |
| `--max-duplicate-event-rate` | 0.02 | share of events that may repeat another event inside the **same record** |
| `--max-trigger-fired-share` | 0.65 | share of trigger records that may report `fired = true` |
| `--min-zone-change-from-zone-rate` | *(unset)* | floor on the share of `zone_change` events carrying `from_zone`; unset means measure and watch |
| `--min-attributed-rate` | *(unset)* | floor on the share of resolution events naming a producing clause; unset means measure and watch |
| `--min-fork-attributed-rate` | *(unset)* | the same floor over the events on **fork** records, measured apart because the fork collectors are different code; unset means measure and watch |
| `--min-cause-rate` | *(unset)* | floor on the share of the cause-declaring event types that populate `cause`; unset means measure and watch |
| `--min-rewrite-replaced-by-rate` | *(unset)* | floor on the share of `replaced` / `updated` rewrite records that name the ability that ran instead; unset means measure and watch. It has to stay watched until someone measures the healthy share — a replacement scripted with `ReplacementResult$` returns either outcome having run no ability, so the ceiling is below 1 and nobody knows by how much |
| `--max-mirror-turn-disagreement-rate` | 0 | share of fork records whose `turn` may differ from the record they mirror. Judged, not watched: a fork is taken *at* the moment it mirrors |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` | converted trees whose sidecars the corpus's provenance keys are joined against; repeatable |

### Judged and watched are separate

Every check reports **the number it measured** whether it passed or not — a threshold a run barely
clears is what an operator needs to see, and "duplicates: FAIL" says nothing about whether a fix worked
or merely moved. On top of that, a check with no threshold prints `[WATCH]` rather than
`[PASS]`/`[FAIL]` and **cannot fail the run**. The four flags above that default to unset are watched
until an operator gives them a floor; the probe count, the non-keyword provenance join, and the
keyword join on a machine with no converted tree, are watched always. A number is published as a
watched one *first* and promoted to a verdict by a threshold later — failing runs on a check nobody
has calibrated is how an operator learns to skip the whole report. The exit status and the closing
summary count judged invariants only.

| Invariant | Breached when |
|---|---|
| `record_id` is unique across the run's shards | any id repeats |
| each `game_id` names one game | a game id spans a backward turn jump beyond the tolerance, or shows more than `--max-names-per-game` distinct card names. **Deferred mana-reservoir flushes are exempt**, and the measurement says how many were exempted |
| every `link_id` joins one activation to one resolution | unpaired share above `--max-unpaired-link-rate` |
| trigger and rewrite records name an acting line | any lacks the `ability` field, or the keyed share is below `--min-keyed-rate` |
| resolution records name an acting line | same rule; the report also breaks down `ability_unresolved` by reason |
| continuous records name an acting line | same rule |
| every empty ability says why it is empty | a record on a line-naming kind carries `ability: []` and no `ability_unresolved` |
| keyword provenance keys join their sidecar | a `keyword` key appears in neither the named card's sidecar `lines` nor its `dropped_keys`; `[WATCH]` where no converted tree was readable |
| non-keyword provenance keys join their sidecar | never — always watched. Every other trait kind (`spell`, `static`, `trigger`, `replacement`) is joined and counted on its own line rather than in the keyword line's tail, because a mismatch there has a cause a keyword mismatch does not — a reconversion between collection and training, or a line the converter never emits at all — and nobody has yet measured its healthy value |
| every record was collected in patched mode | any record carries `mode = degraded` |
| `outcome` takes more than one value | every activation record reports the same outcome |
| cost fields are not all empty | no activation record paid anything at all, or — once the window holds `--min-cost-evidence` activations — `mana_by_color` or `tapped` was never populated. Only those two: sacrifice, discard, exile and life costs are genuinely rare in a sealed pool, and a check that cries wolf on them is one an operator learns to skip |
| snapshot tier depth is uniform across kinds | more than one `state.tiers` vector appears in the window |
| exact duplicates within a game stay rare, per kind | any kind's duplicate rate exceeds `--max-duplicate-rate` |
| no record repeats an event inside itself | duplicate-event share exceeds `--max-duplicate-event-rate`. Distinct from the row above, which compares whole records and reads 0.00% on a corpus duplicating one event in seven |
| trigger records draw negatives against positives | fired share exceeds `--max-trigger-fired-share`; the report breaks the ratio down per evaluated trigger mode as well as in aggregate |
| `zone_change` events say where the card came from | watched by default; breached only when `--min-zone-change-from-zone-rate` is given and the share falls below it. The `to_zone = stack` share rides along in the measurement |
| resolution events name what produced them | watched by default; breached only when `--min-attributed-rate` is given. The measurement breaks `attributed_to` down into sub-ability / root / unresolved / absent |
| fork records' events name what produced them | watched by default; breached only when `--min-fork-attributed-rate` is given. Measured apart from the row above because the fork collectors are different code: the observed records reached 84.8% naming a clause while every one of the 1,698 fork events in the same window said nothing, and the aggregate read as a channel merely warming up |
| rewrite records say which replacement result happened | any `rewrite` record carries no `result`. Judged at zero: the whole channel hangs off this field, and a record without it is either a pre-contract shard inside the window — which `validate-corpus` is not meant to be pointed at — or a listener reading the wrong argument slot. The measurement prints the result histogram, which says which |
| a rewrite's `outgoing` event is not a copy of its `incoming` one | any result-carrying `rewrite` serializes the two alike. There is no replacement that sets a parameter back to its own value, so an identity pair is a writer defect, not a rare game state; `null` is how "not rewritten in place" is spelled. Counted only over records that carry a `result`, because a pre-contract writer had no `null` to write |
| `outgoing` is null on the results that rewrote nothing | any `prevented` or `skipped` record carries an `outgoing` or a `replaced_by`. Both are bare returns above the `playSpellAbilityNoStack` call: no ability ran and nothing was written to the map. `not_replaced` is deliberately not judged here — its prevention branch writes `PreventedAmount` before returning — and this is the failure a reflective `InvocationHandler` makes silently, since nothing compiles against the signature |
| rewrites name the ability that ran instead | watched by default; breached only when `--min-rewrite-replaced-by-rate` is given. Measured over `replaced` and `updated` only, the two outcomes reachable after an ability ran |
| events that declare a cause name one | watched by default; breached only when `--min-cause-rate` is given. Measured over the types whose own params row declares `cause` (`zone_change`, `destroyed`, `sacrificed`) rather than the whole vocabulary, where a near-zero rate would be correct. It is also the identity channel the duplicate-event row depends on: two attackers dealing the same damage to the same player differ in nothing but their cause |
| a fork's turn agrees with the record it mirrors | any fork's `state.global.turn` differs from its `mirror_of` record's, above `--max-mirror-turn-disagreement-rate` (default **0**). A fork is taken *at* the moment it mirrors, so there is no healthy rate of disagreement; forks whose mirror fell outside the window are counted apart and judged not at all. It surfaced before as a backward turn jump in the game_id row, which named the game and blamed a merge that had not happened |
| probe forks were taken | never — always watched. Zero probes means `--probe-keywords` was empty, which is a launch choice rather than a defect |

A kind absent from the window is reported as holding, not as broken: a two-minute window need not
contain a `continuous` record, and a check that fails on silence teaches an operator to ignore it. The
mana-reservoir exemption is that principle applied to a check that was *already* crying wolf — mana
activations are reservoir-sampled and flushed at game end while their snapshot dates from when the
mana was made, so the un-exempted check failed 168 of 177 games on a healthy corpus.

Exit codes: 0 when every judged invariant holds, 1 when any is broken **or** when the directory holds
no shards.

## `python -m effects collect-variants` (stage four)

| Flag | Default |
|---|---|
| `--effect-records` | `output/effects/records/` |
| `--forge-cards-path` | `../forge/forge-gui/res/cardsfolder/` |
| `--variant-volume` | 0.2 (cap on variant records as a fraction of real records already present) |
| `--decks-per-round` | 500 |
| cap/budget flags | see below |

## Cap and budget flags (every collecting supervisor)

| Flag | Default | Meaning |
|---|---|---|
| `--mana-cap` | 1 | resolution records per unique (mana ability, mana produced) pair, per game |
| `--playability-rate` | 0.1 | fraction of `decision`-subkind logging points sampled; `attackers`/`blockers` are sampled at `--legality-rate` instead |
| `--legality-rate` | 0.1 | fraction of `legality`-subkind logging points kept, sampled after the dedup |
| `--interventions-per-game` | 2 | interventional resolutions per game (stage three) |
| `--probes-per-game` | 2 | damage-step probe forks per game — a budget, not a switch |
| `--probe-keywords` | none (**probes disabled**) | comma-separated canary-failing keywords; required for any probe at all |
| `--snapshot-tiers` | `1,2,3` | snapshot inclusion depth; a prefix of `1,2,3,4` holding at least `1,2` — 4 adds unreferenced hands and graveyards |

`continuous` records take no cap: coalescing per stable board is the cap.

**The flag set and the JVM's cap set are one set.** Each flag becomes an `effect.*` system property
(`--snapshot-tiers` → `effect.snapshot.tiers`, and so on) and the worker's `CollectionCaps` record
reads exactly those names; a test reads both sides and fails when either grows a knob the other does
not have. It grew two before that test existed: `effect.snapshot.tiers` and `effect.legality.rate`
were read by the JVM, documented in `MatchWorkerMain`, and settable by no command, so the tier-4
snapshot stage three needs took a code edit to request and the legality sampler ran at a rate a
javadoc called `--legality-rate` — a flag that did not exist.

**`--snapshot-tiers` is validated before the JVM sees it.** A vector that is not a prefix of
`1,2,3,4` holding at least `1,2` is a usage error, because both of the JVM's own failure modes are
worse: an unparseable vector falls back to the default *silently*, and a parseable non-prefix throws
inside `SnapshotBuilder` once a game is already running. Tiers are cumulative — 3 widens 2, 4 widens
3 — and the battlefield goes into every snapshot regardless, so a vector omitting 2 would tell a
reader the board was uncollected while the board sits in `entities`.

**`--probes-per-game` alone buys nothing.** The budget is spent only on keywords `--probe-keywords`
names, and it names none by default, so a run launched without it collects a corpus with zero probe
forks — which is what the 55,296-record smoke corpus was, with nothing saying so until gate 2 had no
engine-side branch to check against. Every collecting supervisor therefore prints one of these at
startup, before the first worker spawns:

```
Damage-step probes DISABLED: --probe-keywords is empty, so the --probes-per-game 2 budget buys no
fork at all. Pass --probe-keywords <keyword>[,<keyword>...] to take any.
Damage-step probes: up to 2 per game on trample, deathtouch
```

`validate-corpus` reports the probe count as a watched number for the same reason, so a run that
*meant* to probe is caught in its first minutes rather than at evaluation.

## `python -m effects holdout-cards`

Writes the depletion list that `sealed generate-pools --exclude-cards` reads: one Forge canonical
card name per line, every card carrying a held-out ability text.

| Flag | Default |
|---|---|
| `--out` | required — destination file |
| `--cards-folder` | `output/cardsfolder/` |
| `--holdout-permille` / `--holdout-max-carriers` | 20 / 8, matching `train-effect-model` |

Reports the held-out text count, the card count, and the share of the corpus those cards are. The
same flags must be used here and at training, or the depleted corpus and the split disagree.

## `python -m effects train-effect-model`

The full flag table is the root spec's § Training. Contract highlights:

| Flag | Default |
|---|---|
| `--records-dir` | `output/effects/records/` |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` |
| `--variant-scripts` | none (stage four: `output/effects/variant-scripts/`) |
| `--split-from` | none (compute the split); **required for variant runs**. Inherits the source checkpoint's split *and* its vocabulary and keyword-definition paths |
| `--vocab-path` | `models/effects/vocab.txt` |
| `--printings-path` | `resources/AllPrintings.json` — first-printing dates, used only to break gate-1 margins down by recency |
| `--keyword-definitions` | `output/effects/keyword-definitions.json` |
| `--model-output` | `models/effects/effect-model/` for `--variant full`, `models/effects/effect-model/{variant}/` otherwise |
| `--variant` | `full` (\| `identity` \| `state-only` \| `no-state` \| `taxonomy`) |
| `--e-dim` / `--e-noise` | 64 / 0.05 |
| `--keyword-expand-p` / `--context-dropout` | 0.25 / 0.15 |
| `--mlm-weight` / `--mlm-mask-prob` / `--api-weight` | 0.1 / 0.15 / 0.05 |
| `--curriculum-step` | 10000 |
| `--batch-size` / `--grad-accum` | 32 / 1 |
| `--kind-mix` | the eight-class mixture |
| `--context-cache` / `--cache-refresh` | off / 500 |
| `--steps-per-epoch` / `--epochs` / `--patience` | 5000 / 40 / 5 |
| `--shards-per-epoch` | 18 — record shards an epoch reads, one resident at a time; the walk advances each epoch and wraps |
| `--reserved-shards` | 4 — shards held back for the game-disjoint stratum. Every shard holding a held-out card is reserved on top of these, which is how a full-strength collection run's shards become the card-disjoint stratum |
| `--holdout-permille` / `--holdout-max-carriers` | 20 / 8 — an ability text is held out when at most `--holdout-max-carriers` cards carry it and `crc32` of its normalized script text modulo 1000 is below `--holdout-permille` |
| `--min-holdout-records` | 2000 — warn below this many unique-text resolution records in the card-disjoint stratum; empty is a hard failure |
| `--withhold-keyword` | none — withholds one implemented keyword's token from training so the zero-shot check has something to measure; its occurrences are always expanded |

Best checkpoint is selected by card-disjoint validation loss. The split holds out ability texts by a
stable hash of the text, and a card is held out when any of its lines carries one; it then **excludes from training every game
holding a record that names a held-out card**, in every shard the run reads; game-disjoint validation
is the reserved shards' remaining games.

The corpus is read one shard at a time and never held whole. Validation records are captured once from
the reserved shards and reused every epoch, so `--patience` compares like with like. Both strata are
enumerated from the games the run actually read, so a run that stops early records fewer games than a
full one.

Hardcoded, not flags: encoder d_model 256 / 4 layers / 4 heads; trunk d_model 256 / 6 layers / 4 heads;
`ff_dim` 4 × d_model; dropout 0.1; AdamW; lr 1e-4 constant after warmup; linear warmup over the first 5% of
`--epochs` × `--steps-per-epoch`; per-parameter-group gradient clip 1.0; seed 42.

## `python -m effects encode-abilities`

| Flag | Default |
|---|---|
| `--variant` | `full` — resolves both `--checkpoint` and the output suffix, so a variant is never encoded with another variant's weights |
| `--checkpoint` | the variant's `latest.pt`; an explicit value overrides |
| `--cards-folder` | `output/cardsfolder/`, `output/tokenscripts/` |
| `--variant-scripts` | stage four |
| `--vocab-path`, `--keyword-definitions` | the paths the checkpoint recorded from training |
| `--clean` | removes only files this command wrote |

Output: `output/effects/abilities/<tree>/…`, `<name>.npz` for `full` and `<name>.{variant}.npz`
otherwise, beside rather than replacing the shipping cache.

## `python -m effects evaluate-effect-model`

| Flag | Default |
|---|---|
| `--checkpoint` | `models/effects/effect-model/latest.pt` |
| `--variant-checkpoint NAME=PATH` | repeatable |
| `--records-dir`, `--cards-folder`, `--variant-scripts` | the trainer's defaults |
| `--vocab-path`, `--keyword-definitions` | the paths `--checkpoint` recorded |

The zero-shot keyword check needs no flag: the withheld keyword is read from `--checkpoint`, which
records it alongside the split.

**Splits are never a flag.** The held-out card list and `game_id` set come from `--checkpoint`. The
command fails fast when a `--variant-checkpoint` records a different split, or different vocabulary or
keyword-definition hashes, than `--checkpoint`.

## Exit-code and failure contract

| Condition | Behaviour |
|---|---|
| Vocabulary or keyword-definition hash mismatch | fail fast with the recorded vs actual hash |
| `--variant-checkpoint` split disagreement | fail fast naming both checkpoints |
| Variant training run without `--split-from` | fail fast |
| Missing connector JAR | the existing `forge_jvm` error naming the `mvn package` command |
| Stock (unpatched) Forge | **not** a failure — degrade, stamp `mode = degraded` |
| Trailing partial line in a shard | **not** a failure — skip it |
