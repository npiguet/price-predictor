# Contract: CLI surface (delta)

**Feature**: `025-ability-effect-model-gen3`
Authority: [`../../2026-10-30-ability-effect-model-gen3.md`](../../2026-10-30-ability-effect-model-gen3.md) § 15.
Base: [`../../024-ability-effect-model-gen2/contracts/cli.md`](../../024-ability-effect-model-gen2/contracts/cli.md).

Only flags and behaviour this feature adds, removes or changes. Defaults are contract. Every refusal
happens before any game, training step or file write, exits 1 and names what failed.

## `python scripts/rules_catalogue.py validate` (existing)

No new flag. Also fails on:

- a scenario with neither a `detector` declaration nor `{"none": reason}`;
- an unknown detector family, or parameters the family does not declare;
- a declaration whose `record_kind` differs from the scenario's, a family parameter naming a card
  (a `selector` such as `Card.namedX`), or a `failed_check` declaration whose `check` is not a
  `PlayabilityCheck` value;
- a non-base variant with neither `operation` (one of the ten) nor `{"unclassifiable": reason}`;
- an `operation` on the base variant (the first one), or a base variant whose `delta` is not empty;
- a `departure` outside its enums, or `resolution` without `status = confirmed`.

The enums are imported from `effects.domain.rule_catalogue`.

## `python scripts/rules_catalogue.py gen3-fields` (existing)

No new flag (a hidden `--rules-dir` for tests). First checks that every rule only `turn_action`
records show reads `recorded`, and exits 1 writing nothing otherwise. Then recomputes
`learnable_after_gen3` after a map rerun, prints the rules `NEWLY LEARNABLE` and `NO LONGER LEARNABLE`
against the `learnable` member of `rules/gen3-fields.json`, and rewrites that member; a first run
records the baseline.

## `python scripts/rules_catalogue.py directed-rules` (new)

| Flag | Default | Meaning |
|---|---|---|
| `--counts PATH` | required | gen-2's per-rule counts, as `rule-counts` writes them |
| `--cards-folder DIR` | `output/cardsfolder` | the converted tree whose card names the wording list uses |

Writes `rules/directed-fork-rules.txt` (FR-014b: the rules of `ALWAYS_DIRECTED`, 704.5f and 702.98a,
each printed with its reason, and every rule with fewer than 100 `fires`
records or no `does_not_fire` record that has a forkable variant; a rule the counts omit counts as
zero when one of its scenarios declares a detector that tags gen-2 records; never a rule of
`NOT_DIRECTED`, 104.2a, 500.11, 701.12d and 702.62d, or of `DEFERRED_DIRECTED`, rules that qualified after the list was fixed for the production collection, each printed with its reason) and
`rules/coverage-wording-cards.txt` (FR-038a: every card those rules' scenarios name, a back face or
meld result under its front card's converted name). Prints, per rule, its forkable variants, those of
the polarity its counts lack first and starred.

## `python -m effects rule-counts` (new)

| Flag | Default | Meaning |
|---|---|---|
| `--records-dir DIR` | required | shards to count; gen-2 or gen-3, not mixed |
| `--output PATH` | `output/effects/rule-counts.json` | counts file |
| `--cards-folder DIR` | `output/cardsfolder` and `output/tokenscripts` | sidecars the detectors read; repeatable, as `build-corpus` |
| `--variant-scripts DIR` | `output/effects/variant-scripts` | variant sidecars |
| `--rules-dir DIR` | `rules` | scenarios and their detector declarations |
| `--workers N` | 12, as `build-corpus` | parallel shard readers, 1 to 12 (exit 2, usage); never the CPU count |

Prints one line per rule, grouped by unit: `fires` and `does_not_fire` counts, each split
`random_seat / forge_ai / aimed`; then the rules declared `none` with their reasons, the rules not
countable on the records' generation, and the size of the review list; before them the rules
declared with zero records, the untaggable scenarios with their reasons and the `same_as` scenarios
with what they are counted under. The counts file lists a rule no record tags under
`declared_zero`, beside `untaggable` and `same_as`, and also holds `generation` (`gen-2` or `gen-3`, the records' generation)
and `carriers`: per rule, the script files of the cards whose lines are its tags' condition
entities, each with the line keys its tags act through and its record count, read by the steered
variant round (FR-058f), as `rule -> [{script_file, records, lines: [{key, records}]}]`, a key
spelled as a sidecar spells it (without `script_file`). Exit 1 when the directory holds no shard, holds a
gen-1 shard, or mixes generations.

## `python -m effects validate-corpus` (existing)

Flags this feature adds (each of `Thresholds`, `effects.application.corpus_validation.findings`):

```
--min-presence-records N        1000   gen-3 records a window needs before a kind every game
                                       writes that it lacks fails the run
--min-trigger-fired-share F     0.05   share of the trigger records of games that ended in the
                                       window that must report fired=true
--min-cost-evidence N           200    see below
--variant-scripts DIR           output/effects/variant-scripts/
                                       the perturbed-script tree whose keys a collect-variants
                                       corpus names
```

Behaviour:

- `--min-cost-evidence` sets two thresholds: the activation records needed before one cost channel
  never populated fails "cost fields are not all empty" (under it, watched), and the gen-3 cost
  halves needed before none of them meeting its effect half fails the cost-half pairing check.
- Kind presence (`KindPresence`) judges a window of at least `--min-presence-records` gen-3
  records: cost and effect halves, trigger, combat, the three playability subkinds and the draw
  step's turn action; the decision subkind only when the run's `run.json` records a
  `playability_rate` above 0 (the smallest any invocation recorded). Rarer kinds are watched.
- The positives floor reads the trigger records of games that ended in the window: a game has
  ended when one of its records holds `player_won` or `player_lost`. Under 50 such records it is
  watched.
- A keyword key in a variant tree the run was not given fails the keyword-join check as
  unchecked; a machine with no converted tree at all watches it.
- A copy keying its keyword lines into the copied card's file is watched.

Exit codes:

| Code | When |
|------|------|
| 0 | every invariant holds |
| 1 | an invariant breaks, including "every shard line decodes": a line that does not decode is that finding, naming shard and line (logged at ERROR); the shard's records before it are measured and the other shards read on |
| 1 | no shard under `--effect-records`, or a record whose `generation` the reader does not accept (`UnacceptedGenerationError`) |

Every invariant prints the number it measured, passed or not.

## `python -m sealed match-outcomes` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--playability-rate P` | 0.01 (was 0.1) | stage 1 of decision sampling (FR-033b): the share of enumerated decision candidates evaluated; a dropped one is not evaluated for the record |
| `--zone-candidate-rate R` | 0.0001 | share of zone-bucket candidates written, both stages together |
| `--plain-candidate-rate R` | 0.001 | share of plain-bucket candidates written, both stages together |
| `--count-dropped-events` | off | count events arriving with no open record, by event type and engine phase; print at the end |
| `--max-records-per-game N` | 5000 | records one game may write, its forks' included; past it the worker withholds the rest, counts them and prints one line naming the game, the turn and the acting card of the most frequent withheld record (FR-041a); below 1 is refused |
| `--directed-fork-rules LIST` | none | comma-separated rule ids |
| `--directed-forks-per-game N` | 2 | directed-fork budget per game, beside the existing fork budgets |
| `--directed-fork-probability P` | 0.25 | chance a matching boundary forks; must be in (0, 1] |

Startup, exit 1 naming the rule when a listed rule:

- has no detector declaration, or a declaration `none`;
- has no non-base variant classified into an operation;
- is not registered in `forge-connector` (checked by one run of `DirectedRulesMain`).

`--directed-fork-probability` outside (0, 1] exits 1. The flags ride the cap properties
`effect.directed.fork.rules`, `effect.directed.forks.per.game`, `effect.directed.fork.probability`,
`effect.count.dropped.events`, `effect.max.records.per.game`.

The three sampling rates ride `effect.playability.rate`, `effect.zone.candidate.rate` and
`effect.plain.candidate.rate`, and are shared with `collect-coverage` and `collect-variants` through
the common cap flags (`add_cap_flags`; Python `CollectionCaps.zone_candidate_rate`,
`plain_candidate_rate`). A rate outside [0, 1], or a candidate rate above `--playability-rate`, is
refused before any game, as the random-seat flags are (`candidate_rate_error`): exit 2 on
`match-outcomes`, 1 on the `effects` collectors. A special candidate (FR-033b) is written whenever
stage 1 keeps it. Every collecting run of a corpus, the held-out coverage round included, uses
the same three rates (FR-059a).

`--workers N` is 1 to 12 (exit 2, usage). With `--effect-records DIR`, `DIR` is the run directory:
the shards at its top level, every worker log under `DIR/logs/` (`<run id>.<slot>.log`, never
beside the shards). At every status line and at the end the supervisor scans what the logs gained
for failure lines (`price_predictor.domain.worker_failures`, as on the collectors) and prints them
by kind; at the end it prints the run's totals and records in its `run.json` entry `run_id`,
`matches`, `worker_failures`,
`worker_failure_examples` and `worker_survivors`. A run that completed matches and wrote no effect
record exits 1. `match-outcomes.txt` and `cards-played.txt` stay at `output/sealed/`.

## `python -m effects collect-coverage` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--wording-cards PATH` | none | one card name per line; a coverage round over them |
| `--count-from DIR` | none | repeatable; an earlier run's records directory whose coverage counts toward satisfaction, read once at the start and never written |
| `--count-dropped-events` | off | as `match-outcomes` |
| `--max-records-per-game N` | 5000 | as `match-outcomes` |
| `--directed-fork-rules LIST` | none | a label only: rule ids written to every record's `aimed_at`; no directed fork runs, and no startup check applies |

`--wording-cards` beside `--only-cards` exits 1. A run with none of `--exclude-cards`,
`--split-from` and `--training-corpus` exits 1 (FR-058g). A `--count-from` directory that does not
exist, or that holds or is held by `--effect-records` or another `--count-from`, exits 1.

The castability verdict only ranks cards and never drops one. A separate rule applies to a card
the consult leaves unanswered while answering others: Forge's card database does not hold it, no
deck can name it, and it leaves the coverage unit and the decks. No share of such cards stops the
run. A warning prints their count, their share of the consulted pool and the first twenty names,
and says that a large share means `../forge` is older than the converted cards; the run's
`run.json` entry records `unresolved_cards`, `unresolved_share` and `unresolved_examples`.

A run counts a round only once it plays one. A run that plays rounds and writes no record in its
own shards (`<run id>.*`) exits 1; a run that plays none exits 0 when every unit was already done
and 1 when units remain that no deck could hold.

## Both collectors (existing)

| Flag | Bound | Exit |
|---|---|---|
| `--workers N` | 1 to 12 | 2 (usage) |
| `--decks-per-round`, `--target-records`, `--no-progress-rounds`, `--min-text-games`, `--variant-games`, `--steer-floor`, `--steer-variants`, `--steer-iterations` | at least 1 | 2 (usage) |
| `--steer-min-gain`, and on every collecting command `--mana-cap`, `--interventions-per-game`, `--probes-per-game`, `--directed-forks-per-game` | at least 0 | 2 (usage) |

A missing `--exclude-cards`, `--only-cards`, `--split-from` or `--wording-cards` file exits 1
naming it, before any worker starts.

The run directory is `--effect-records` (or `--training-corpus`). It holds the shards at its top
level, `run.json`, every worker log and each round's progress file under `logs/`
(`logs/<run id>.<slot>.log`, `logs/<run id>.round-NNNN.progress.txt`), and the decks file the
workers read under `decks/` (`decks/coverage-decks.<run id>.txt`, `decks/variant-decks.<run id>.txt`).
After each round the collector scans the round's additions to the worker logs for failure lines
(`price_predictor.domain.worker_failures`: unresolved deck, match error, worker gave up, hook,
emitter, boundary step, withheld records, fork score mismatch, fork fidelity, staging, JVM error
and the rest, each found anywhere in a line, since a worker's stdout and stderr interleave
mid-line), prints them by kind, and names any worker the kill could not end and still running.
A log rotated twice between two scans counts as `log_unread`. The running command's entry in
`run.json`'s `invocations` records `run_id` (the supervisor's), `caps` (every `CollectionCaps`
field, resolved), `rounds`, `worker_failures` (kind to line count), `worker_failure_examples`
(kind to its first line) and `worker_survivors` (PIDs). The castability consult and the sidecar
pass are one-shot JVMs killed after `max(3600, 600 + 0.1 × scripts handed over)` seconds: the
consult then ranks every card castable, the sidecar pass exits 1.

The first JVM any command starts removes the variant scripts a `collect-variants` killed before
it could unstage them left in Forge's custom-cards directory (the files its staging list names),
and prints how many. A `collect-variants` records itself as the staging's owner before its sidecar
pass stages anything (`.staged-by-effects.owner` beside `.staged-by-effects.list`: its PID and its
process start time) and drops the record when it unstages. While the recorded process runs, PID
and start time both matching (a reused PID is no owner), every other command leaves the staging in
place and prints that it did. A staging list that cannot be read is reported and stops nothing; an
entry that resolves outside the custom-cards directory is never deleted.

## Match worker stop protocol (`MatchWorkerMain`, every collecting command)

A match worker stops at a game end when asked, so the game it is playing ends normally, writes its
game-end marker, and its shard is closed whole. The request is a file, not a console signal.

| `-D` property | Value | Meaning |
|---|---|---|
| `worker.stop.file` | a path | the worker stops once a file exists at this path; its content is ignored; the worker never creates or deletes it |
| `worker.recycle.seconds` | integer, at least 1 | the worker stops at the first game end after this many seconds counted from its `main` start (before Forge is initialised); absent: never; malformed or below 1: exit 2 before Forge is initialised |

- Both properties are read with the worker's other properties, before Forge is initialised.
- The worker asks before each match and at the end of each game, once the game's recording is
  closed and its marker written. A request is honoured within the remainder of the game in progress
  (or before the next match starts, between matches). No game starts after it.
- A stop file present at startup, or created while Forge initialises, stops the worker before its
  first match, once initialisation is over, with the stop line.
- Stopping mid-match leaves the match unfinished: no `match-outcomes.txt`, `cards-played.txt` or
  progress-file line for it; the games it finished keep their records and markers.
- On stopping the worker closes its shard (the last block written), prints one stdout line and
  exits 0:
  `Worker: stopped at a game end (<reason>); <n> matches generated in this JVM; output closed, exiting 0`,
  `<reason>` being `stop file <path>` or `recycling after <seconds> s`. The line matches no
  worker-failure pattern.
- With either property set the worker prints at startup
  `Graceful stop: at a game end, when <path> exists[ or ]the JVM has run <seconds> s`.
- What a supervisor cannot rely on: a game has no time bound of its own, and a request made while
  the worker initialises Forge waits for the end of initialisation. A worker that has not exited
  within the supervisor's timeout after the request is killed (`taskkill /F /T`) as before, and the
  game it was playing has no marker.

What the supervisor does:

- Give each worker its own stop-file path (for example `logs/<run id>.<slot>.stop`), deleted
  before the worker is spawned.
- To stop a worker (the run's end, recycling, a round's end): create the file, then wait for the
  process to exit; after the timeout, kill it.
- To recycle without killing: pass `worker.recycle.seconds` and restart a worker that exited 0
  after printing the stop line while the run continues; keep the hard kill only as the backstop for
  a worker past its recycle time by more than the timeout.
- Read exit 0 with the stop line as a normal end, never as a failure or a crash.
- Finish every graceful stop before the supervisor exits: the job object kills the workers when
  the supervisor's process ends.

A match that threw prints `Error generating match: <message>` with its stack trace, as before;
when the match had a random seat, `<message>` starts with `[random-seat match] `, whatever failed
after the seat was drawn: the match's set-up, a game, a game's recording closing. An `Error`
(out of memory, a stack overflow) ends the worker; in a random-seat match it first prints
`Error generating match: [random-seat match] <error>` on stderr. A failure before the seat is drawn
(building or loading the decks) names no random seat: the match had none yet.

A game whose marker could not be written prints, once per worker,
`Effect records: the game-end marker of game <game id> could not be written, so the game reads as cut (later failures are counted, not printed): <exception>`;
every such failure is counted under `game-end marker` in the worker's failure report
(`effect-record hook failures so far, listener calls that threw and lost their records: {…}; contained by Forge, per slot: {…}`).

## `python -m effects collect-variants` (existing)

Takes `--count-dropped-events`, `--max-records-per-game`, `--directed-fork-rules`,
`--directed-forks-per-game` and `--directed-fork-probability` as `match-outcomes` does, with the same startup checks and exits.
Every record carries `aimed_at` (the listed rules, or `[]`; in a steered game, the rules of its
two decks). A protection keyword line of a source script swaps its quality for another of
`PROTECTION_QUALITY_WHITELIST` (FR-058) before any other perturbation of that line is tried.

| Flag | Default | Meaning |
|---|---|---|
| `--round untargeted\|steered` | `untargeted` | which variant round runs (FR-058a, FR-058c) |
| `--variant-games N` | 2 | games each variant is played to |
| `--no-progress-rounds N` | 3 | rounds without a new game before a variant retires |
| `--rule-counts PATH` | — | steered only, required there: `rule-counts` over the corpus being collected |
| `--steer-floor N` | 100 | records below which a (rule, polarity) is short |
| `--steer-variants N` | 2000 | variants per iteration, split over the short pairs by shortfall |
| `--steer-min-gain N` | 1 | records of its short polarity an iteration must add to keep a rule |
| `--steer-iterations N` | 5 | iterations at most |
| `--continues RUN_ID` | none | repeatable; the supervisor run id of an interrupted run into the same `--effect-records`: its shards' games count before the first round |

`--variant-volume` is removed. Startup, exit 1: no holdout source (`--exclude-cards`,
`--split-from`, `--training-corpus`); `--round steered` without `--rule-counts`, or with a counts
file whose `generation` is not `gen-3`; `--continues` with `--round steered`, or naming a run id no
shard under `--effect-records` carries. A run that plays no round exits 0 when every variant
already has its games, and 1 when variants remain that no deck could hold; one that plays rounds and
writes no record exits 1. The manifest's
inputs list `--variant-scripts`, which the run reads and extends, and `rules/` in the steered round.
A steered round's deck file adds a fourth column, the
deck's rule (`LABEL;SET_CODE;Card1|...|Card40;RULE`); the worker stamps every record of a game
whose decks carry one with `aimed_at` listing its two decks' rules. The loadable list is `loadable.lst` in the variant tree, never a `.txt` file, and a round
removes its staged variants from Forge's custom-cards directory when it ends (FR-058b).

## `python -m effects build-corpus` (existing)

No new flag. Behaviour:

- Exit 1 unless every record carries `generation: 3`.
- Reads `rules/heads/` and `rules/scenarios/` once; writes the map snapshot and its digest. Exit 1
  before the survey, nothing under `--output` touched, when either holds nothing.
- Exit 1 naming shard and line on a corrupt gzip member before a shard's last, a line that does not
  decode or a `generation` naming none: before `--output` is cleared when the survey meets it, and
  saying the dataset was already cleared when the write pass meets it (a shard changed between the
  passes).
- Warns, naming every one, on raw shards whose size changed since the survey; the records past those
  the survey read in a shard go to no stratum (`dropped-unsurveyed` in `per_stratum`).
- Writes no `turn_action` record; counts them by subkind.
- Writes a `curation` member on every record it writes, and each protection copy drawn as a
  synthetic record `<record_id>~pc<i>`.
- Exit 1 when it can already tell that a rule-test component misses its precision target, naming
  stratum, half and component; the probes, grids and deviance wait for the freeze. The corpus and its
  manifest are written before this exit. The thresholds (300 games a half, 0.1, 0.05) and `rules_dir`
  are config fields with no flag.

## `scripts/effect_knowledge_probes/run.py --freeze-probe-set` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--corpus DIR` | required | the curated corpus to freeze against |
| `--checkpoint PATH` | — | refused with `--freeze-probe-set` |
| `--probe-set-dir DIR` | `--reports-dir` (`output/effects/reports/`) | where the set is written |

Writes `<probe-set-dir>/selection-set-<corpus digest>.json` ([selection-set.md](selection-set.md)).
A set already there is never replaced: the same set again leaves the file as it is, another exits 1
and leaves it alone.
Exit 1, writing nothing, when a half holds fewer than 300 games, a component misses its precision
target, a balance rule cannot be met or a grid cell is empty, naming stratum, half and component.
`scripts/effect_sweep_arm.py --freeze-probe-set` runs the freeze as a first step, before training.

## `python -m effects train-effect-model` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--rule-mask hard\|soft` | hard | mask on added copies |
| `--deviance-floor F` | 0.04 | constant-prediction loss floor, in nats |
| `--ema-epochs E` | 1 | span of the averaged weights |
| `--relation-values on\|off` | on | train the relation value tables |
| `--stratum-weight W` | 0.5 | card-disjoint share of the control score, in [0, 1] |
| `--patience N` | 5 | epochs without a new best control score |
| `--reports-dir DIR` | `output/effects/reports/` | where the selection set frozen for `--corpus` is found, by the corpus digest; may change on a resume |
| `--e-dim` | 256 | was 64 |

Startup, exit 1 when:

- a record lacks `generation: 3`;
- the manifest has no per-rule counts;
- no selection set exists for the corpus digest under `--reports-dir`, or the one there was frozen for another corpus or edited;
- resuming with a changed `--rule-mask`, `--deviance-floor`, `--ema-epochs`, `--stratum-weight` or
  `--relation-values`.

During training, exit 1 when an epoch meets a component with nothing to score, naming stratum, half
and component.

Epoch line adds, per stratum: `probes`, `grids`, `rule_tests`, `deviance`, `score`; then `control`
and `best`; then a line `deviance by head:` giving per stratum each head's mean explained deviance
over the items it scored and their count (`per_entity`, `verdict`, `created_objects`, `global`; a
head with none not shown); then a line of its own giving the control score's seconds against the
epoch's training seconds. Shard line adds `rel_k_norm` and `rel_v_norm` (pre-clip, last step of the shard).

## `python -m effects evaluate-effect-model` and the knowledge probes (existing)

`evaluate-effect-model`: no new flag. Exit 1 unless every record carries `generation: 3`, or when
the checkpoint's `input_layout` is not `gen-3`. It scores the reporting halves of the selection set
found by the corpus digest under `output/effects/reports/` (the config's `reports_dir`), under the
checkpoint's recorded `--rule-mask`, `--stratum-weight` and `--deviance-floor`, and prints per stratum
`reporting-half <stratum>` (components and score) with each component's items (the deviance items
also summed up by head), `rule-tests <stratum> reporting` per rule, and `reporting-half control` beside the checkpoint's `best_control_score`;
without a set it prints that the reporting half is skipped.

`scripts/effect_knowledge_probes/run.py --checkpoint PATH`:

| Flag | Default | Meaning |
|---|---|---|
| `--half both\|selection\|reporting` | both | the selection set's halves read and reported |
| `--probe-set-dir DIR` | `--reports-dir` | where the selection set is read |
| `--probe-records N` | 1000 | qualifying records each gen-3 probe reads per half |
| `--probe-per-game N` | 3 | of those, at most per game |

`--games-per-stratum` and `--items-per-target` are gone: the run reads the frozen set. `combat_grid.py`,
`evasion_grid.py`, `damage_grid.py` and `ladder_check.py` take `--half reporting|selection|both`
(default reporting) and `--probe-set-dir`; `combat_grid.py` takes `--blocks one|several` (default
one). Scorecards carry the selection set's digest and, per stratum and half, the probes, grids, rule
tests and § 14.5 probes.
