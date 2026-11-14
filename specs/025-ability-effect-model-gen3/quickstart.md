# Quickstart: Ability effect model — generation 3

**Feature**: `025-ability-effect-model-gen3`
Run plan and rationale: [`../../experiments/2026-10-30-effect-model-gen3-design.md`](../../experiments/2026-10-30-effect-model-gen3-design.md).
Base walkthroughs: [`../023-ability-effect-model/quickstart.md`](../023-ability-effect-model/quickstart.md),
[`../024-ability-effect-model-gen2/quickstart.md`](../024-ability-effect-model-gen2/quickstart.md).

Commands below are the ones this feature adds or changes. Paths are the defaults. Python runs with
`-u`, and long commands print periodic progress.

## Prerequisites

- `../forge` on `effect-record-hooks` with the gen-3 commits (three hooks, the `GameCopier` copy),
  built with `mvn install -DskipTests`; then `cd forge-connector && mvn install -DskipTests`.
- Gen-2's raw shards (for stage 1) and `rules/` on master.

## Stage 1 — detectors and per-rule counts on gen-2 (story 1)

```bash
python scripts/rules_catalogue.py validate
python -u -m effects rule-counts --records-dir output/effects/records-gen2 --workers 10
python scripts/rules_catalogue.py directed-rules --counts output/effects/rule-counts.json
```

Check: `validate` passes with every scenario declaring a detector or `none` and every non-base
variant classified. The printed table and `output/effects/rule-counts.json` agree. Review the listed
records against Forge, starting with `rules/FINDINGS.md`'s suspected departures, and record outcomes
on the scenarios. The counts decide which rules to enable for directed forks and coverage rounds:
`directed-rules` writes `rules/directed-fork-rules.txt`, `rules/coverage-wording-cards.txt` and
the part of the counts they are decided from, `rules/directed-fork-counts.json`, from them; record that choice in the gen-3 record.

## Stage 2 — the relation-aware trunk (story 4, no gen-3 data)

```bash
pytest tests/unit/effects/domain/test_relation_attention.py
python -u scripts/measure_trunk_step.py --shard <fixed gen-2 curated shard> --batch-size 32
```

Check: the four layer properties pass; the step fits 8 GB; record step time, peak memory and the
relation-triple count per record in the gen-3 record.

## Stage 3 — gen-3 collection pilot (story 2)

```bash
python -u -m sealed match-outcomes --effect-records --count-dropped-events \
    --directed-fork-rules 704.5f --directed-fork-probability 1 <usual pilot flags>
python -u -m effects validate-corpus --effect-records output/effects/records
python -u -m effects field-coverage --effect-records output/effects/records
python -u -m effects rule-counts --records-dir output/effects/records
```

Check: every record carries `generation: 3`; every § 8.13 field is non-constant in `field-coverage`
or listed as constant or absent with a reason the operator accepts; the dropped-event table is printed
with the count of `mana_produced` events whose `mana_by_color` is empty; no trigger record holds a
death of the state-based loop its trigger arrived in; directed-fork records carry `aimed_at`,
`directed`, `fork = true`. Record gzipped bytes per game and games per hour against a gen-2 run with
the same flags; if bytes per game more than double, raise it before stage 5.

`build-corpus` over a pilot judges the rule-test component alone, which a few dozen frequent rules
already pass, so it exits 0; the freeze exits 1 on a pilot, whose halves hold fewer than 300 games.
Read the manifest anyway; the freeze is not run on a pilot.

## Stage 4 — the map rerun after each layout change (story 5)

After each input or output change lands, update `rules/MODEL-FIELDS.md`, rerun the heads workflow
(`scripts/rules_heads_workflow.js`), then:

```bash
python scripts/rules_catalogue.py gen3-fields
```

Check: the rules newly learnable are printed and committed with the change.

## Stage 5 — full collection, coverage and variants

Long runs are launched from PowerShell. There the interpreter is `.venv\Scripts\python.exe` (a bare
`python` is the system interpreter, without the project's dependencies), and the directed-fork list
is built first, as its own step, then passed as `--directed-fork-rules $rules`:

```powershell
$rules = (Get-Content rules/directed-fork-rules.txt | Where-Object { $_.Trim() -and -not $_.StartsWith('#') }) -join ','
```

Before the first long run, the fork fidelity audit (T181): collect for 20 minutes into a scratch run
directory, stop the run, and count the `fork fidelity:` lines by field in its `logs/` worker logs
(the run's end prints their total as `fork_fidelity`). Then run `validate-corpus` on the scratch
run: its watched line "fork records whose copy left something out" gives the share of fork records
carrying each `copy_dropped` kind, and the share of forks (fork records grouped by game, game turn
and kind of fork; the wire names no fork, so the count is a lower bound). The workers' hook-failure line counts the forks whose copy, score
check, body or writer threw, as `fork <kind> copy`, `fork <kind> score check`, `fork <kind> body` and
`fork <kind> write`, and the forks that changed the live game, as `fork <kind> moved the live game`;
the refusal line (`Effect records: fork refusals so far, expected and no failure: {…}`) counts the
forks that declined, by kind and reason, and wrote nothing. None of these discards a record. The
audit reads a fork's copy as it was taken, before any perturbation: it cannot tell what a fork did
after that, and it never sees a fork that wrote nothing, which only the refusal line and the
failure counts account for.

```powershell
.venv\Scripts\python.exe -u -m sealed match-outcomes --workers 6 --effect-records <scratch run dir> --random-seat-share 0.25 --random-seat-probability 0.5 --interventions-per-game 4 --probes-per-game 4 --directed-fork-rules $rules --count-dropped-events
```

The collection itself:

```bash
python -u -m sealed match-outcomes --effect-records --directed-fork-rules "$(paste -sd, rules/directed-fork-rules.txt)" <run flags>
python -u -m effects collect-coverage --wording-cards <any-target cards> --exclude-cards <holdout list> <run flags>
python -u -m effects collect-coverage --wording-cards rules/coverage-wording-cards.txt --directed-fork-rules "$(paste -sd, rules/directed-fork-rules.txt)" --exclude-cards <holdout list> <run flags>
python -u -m effects collect-variants --round untargeted --exclude-cards <holdout list> <run flags>
python -u -m effects rule-counts --records-dir <the corpus being collected> --output <counts>
python -u -m effects collect-variants --round steered --rule-counts <counts> --exclude-cards <holdout list> <run flags>
```

The steered round runs last, on counts of everything collected before it (FR-058c). Every
collecting command takes the same three decision-sampling rates (FR-059a).

Each collecting command's run directory is its `--effect-records`: worker logs under `logs/`,
their failure lines printed as the run goes (after each round on a collector, at each status line
on `match-outcomes`) and kept in `run.json`. A `collect-coverage` run in a new run
directory passes each earlier run's directory as `--count-from`, so their coverage counts. An
interrupted `collect-variants` restarts into the same directory with the same `--seed` and
`--continues <run id>`, the run id its `run.json` records.

A collector's round end lets every worker finish the game it is playing (at most
`--stop-timeout`, 600 s, before a kill), and workers recycle themselves at a game end
(`--recycle-seconds`, 600 s, at least 300), so a run's games are cut only by a crash, a kill
past the timeout, a game outlasting its worker's recycle time by `--stop-timeout`, or a second
Ctrl-C. Each round's report and `run.json` say how the workers ended (`worker_stops`) and how
many cut games the counts left out (`cut_games`). Before launching a collection, know that:

- The connector must be rebuilt (`mvn install -DskipTests` in `forge-connector/`) before a gen-3
  collection: shards written by a connector built before the game-end marker hold no marker, so
  `build-corpus` refuses them ("every game cut") and `validate-corpus` fails them.
- To stop a run by hand, press Ctrl-C once: it reaches the supervisor alone, which asks every
  worker to finish its game and waits up to `--stop-timeout`. A second Ctrl-C kills the workers
  at once, cutting their games. Creating a worker's stop file by hand does not stop the run: the
  pool starts a fresh worker in its place.
- Every round starts a fresh pool, so the workers start Forge anew each round; a round ends up
  to `--stop-timeout` after its match budget is reached, and never waits for a worker still
  starting Forge (it is killed, with no game to cut).
- Under `match-outcomes`' best of N, a worker that recycles itself mid-match abandons the match
  in flight: it writes no outcome or cards-played row, and the games it finished keep their
  records and markers.
- A kill loses the game in progress only: every finished game is on disk whole, its marker
  included, before the next one starts.

## Stage 6 — curate and freeze (story 6, story 8)

```bash
python -u -m effects build-corpus --records-dir output/effects/records --output output/effects/corpus-gen3
python -u scripts/effect_knowledge_probes/run.py --freeze-probe-set --corpus output/effects/corpus-gen3
```

Check: the manifest gives, for every tagged rule in every class, share, written and shortfall per
polarity and source, and names one-sided rules and conditions, unbudgeted rules and rules left out of
a half. The freeze writes `selection-set-<digest>.json` or exits 1 naming the component that missed
its target.

Before stage 7, rerun the trunk measurement on a gen-3 curated shard and confirm it fits 8 GB:

```bash
python -u scripts/measure_trunk_step.py --shard <gen-3 curated shard> --batch-size 32
```

## Stage 7 — train (story 7)

```bash
python -u -m effects train-effect-model --corpus output/effects/corpus-gen3 --rule-mask hard
```

Check: each epoch line prints both strata's four components and the control score; the best epoch
saves encoder, trunk and heads together; `resume-best.pt` follows it. A soft-mask run is tried only if
a comparison shows the hard mask costs the other outputs. A `--relation-values off` run is scored on
the anthem probe and the combat grids beside the default.

## Stage 8 — report

```bash
python -u -m effects encode-abilities --checkpoint models/effects/effect-model/latest.pt
python -u -m effects evaluate-effect-model --checkpoint models/effects/effect-model/latest.pt
python -u scripts/effect_knowledge_probes/run.py --checkpoint models/effects/effect-model/latest.pt
```

Check: the scorecard reports every probe and the rule tests on the reporting half. Results go into the
gen-3 record's Result section only.
