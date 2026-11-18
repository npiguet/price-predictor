# Quickstart: Ability effect model — generation 3

**Feature**: `025-ability-effect-model-gen3`
Run plan and rationale: [`../../experiments/2026-10-30-effect-model-gen3-design.md`](../../experiments/2026-10-30-effect-model-gen3-design.md).
Base walkthroughs: [`../023-ability-effect-model/quickstart.md`](../023-ability-effect-model/quickstart.md),
[`../024-ability-effect-model-gen2/quickstart.md`](../024-ability-effect-model-gen2/quickstart.md).

The whole chain of commands for generation 3, from an empty `output/` and `models/` to the report.
Every path follows the artifact layout (`.claude/skills/artifact-layout/SKILL.md`): everything the
effects work reads or writes is under `output/effects/` and `models/effects/`, each run in its own
run directory, and no command relies on a default path. Python runs with `-u`, and long commands
print periodic progress.

## Paths and the launcher

Commands are PowerShell, run from the repository root. The interpreter is
`.venv\Scripts\python.exe` (a bare `python` is the system interpreter, without the project's
dependencies). Paste this block once per session; no command below needs editing.

- `$stage0` is the day the Stage 0 runs started, the date in their `inputs\` directory names, not
  today: `2026-11-18` for the runs that wrote `output\effects\gen3\inputs\2026-11-18-*`.
- Every later run's directory is `<parent>\<start>-<slug>`, its start date held in a variable set
  on the line before its directory's (`$pilotStart` before `$pilot`). A run's start date is the day
  it started, and a run that is resumed or restarted keeps its directory. `Run-Start` gives the
  date of the directory of that slug already under the parent, and today when there is none, so a
  block pasted again, on any day, resumes a run rather than moving it to a new directory. The
  variable is set only while unset, so a date set by hand is kept. To start a new run of a slug
  that already has a directory, set its variable to today before pasting
  (`$pilotStart = "2026-11-20"`); with two directories of one slug, `Run-Start` stops and asks for
  the date.
- What a command needs from you and no layout gives (the pilot's flags, a shard path) is a variable
  you set before pasting the block, named at each stage. `Set-StrictMode -Version 2.0` makes a
  command naming an unset variable fail before it starts ("The variable '$runFlags' cannot be
  retrieved because it has not been set"); `Set-StrictMode -Off` lifts it. No `<` or `>` is left
  in a command: inside `cmd /c` they are redirections.

```powershell
$env:PYTHONPATH = "src;."
$py     = ".venv\Scripts\python.exe"
$gen2   = "output\effects\gen2"
$gen3   = "output\effects\gen3"
$models = "models\effects\gen3"
Set-StrictMode -Version 2.0

# The day a run started, for its directory <parent>\<start>-<slug>: the date of the directory of that
# slug already under <parent> (a resumed or restarted run keeps it), else today.
function Run-Start($parent, $slug) {
    $named = '^\d{4}-\d{2}-\d{2}-' + [regex]::Escape($slug) + '$'
    $runs = @(Get-ChildItem -Directory -Path $parent -ErrorAction SilentlyContinue | Where-Object { $_.Name -match $named })
    if ($runs.Count -gt 1) {
        throw "Several '$slug' runs under ${parent}: $(($runs | ForEach-Object { $_.Name }) -join ', '). Set this run's start date variable to the day the one you mean started."
    }
    if ($runs.Count -eq 1) {
        $date = $runs[0].Name.Substring(0, 10)
        Write-Host "${slug}: the run started on $date"
        return $date
    }
    $date = Get-Date -Format yyyy-MM-dd
    Write-Host "${slug}: a new run, started on $date"
    return $date
}

# Inputs every later step reads. Stage 0 wrote the first three, on the day $stage0 names.
$stage0   = "2026-11-18"
$cards    = "--cards-folder $gen3\converted\cardsfolder --cards-folder $gen3\converted\tokenscripts"
$kw       = "$gen3\inputs\$stage0-keyword-definitions\keyword-definitions.json"
$vocab    = "$gen3\inputs\$stage0-vocab\vocab.txt"
$holdout  = "$gen3\inputs\$stage0-holdout-cards\holdout-cards.txt"
$tokens   = "--vocab-path $vocab --keyword-definitions $kw"
# The variant tree every variant run extends, and stage 5's rule counts the steered round reads.
if (-not (Test-Path variable:variantsStart)) { $variantsStart = Run-Start "$gen3\inputs" variant-scripts }
$variants = "$gen3\inputs\$variantsStart-variant-scripts"
if (-not (Test-Path variable:countsStart)) { $countsStart = Run-Start "$gen3\inputs" rule-counts }
$counts   = "$gen3\inputs\$countsStart-rule-counts\rule-counts.json"

# The directed-fork list, as the comma-separated value --directed-fork-rules takes.
$rules = (Get-Content rules\directed-fork-rules.txt | Where-Object { $_.Trim() -and -not $_.StartsWith('#') }) -join ','

# Runs a command with its console in its run directory's console.log, the command as its first line.
function Run-Logged($run, $cmd) {
    New-Item -ItemType Directory -Force $run | Out-Null
    $cmd | Out-File -Append -Encoding ascii "$run\console.log"
    cmd /c "$cmd >> `"$run\console.log`" 2>&1"
}
```

`Run-Logged` sends the console to the file alone, so that a Ctrl-C reaches the command and
nothing else. Follow a run from a second window with
`Get-Content "$pilot\console.log" -Wait -Tail 20` (any run's directory variable in place of
`$pilot`). Each command writes its own `run.json` in its run
directory; when a run ends, fill `purpose` there and set `status` to `keep`, `scratch` or
`superseded`.

Every command below writes only inside the run directory its flags name.
`scripts/rules_catalogue.py directed-rules` also rewrites three files under `rules/`, which are
committed.

## Prerequisites

- `../forge` on `effect-record-hooks` with the gen-3 commits (three hooks, the `GameCopier` copy),
  built with `mvn install -DskipTests`; then `cd forge-connector && mvn install -DskipTests`.
- Gen-2's raw shards and converted trees under `output\effects\gen2\` (for stage 1), and `rules/`
  on master.
- `resources\AllPrintings.json`.

## Stage 0 — the generation's converted trees and inputs

```powershell
Run-Logged "$gen3\converted" "$py -u -m price_predictor convert --cards-path ..\forge\forge-gui\res\cardsfolder --tokens-path ..\forge\forge-gui\res\tokenscripts --output-path $gen3\converted\cardsfolder --tokens-output-path $gen3\converted\tokenscripts"
Run-Logged (Split-Path $kw)      "$py -u -m effects extract-keyword-definitions --output $kw"
Run-Logged (Split-Path $vocab)   "$py -u -m effects build-vocab --surface prose $cards --keyword-definitions $kw --vocab-path $vocab"
Run-Logged (Split-Path $holdout) "$py -u -m effects holdout-cards $cards --printings-path resources\AllPrintings.json --out $holdout"
```

Check: `$gen3\converted\` holds `run.json`, `console.log`, `cardsfolder\` and `tokenscripts\`, and
every converted card has its `.provenance.json` sidecar. The generation is converted once, before
its first record is collected: records name their abilities by provenance keys that join against
these sidecars.

## Stage 1 — detectors and per-rule counts on gen-2 (story 1)

Set first: `$gen2Variants`, gen-2's variant-script tree (its run directory under `$gen2\inputs\`).

```powershell
& $py scripts\rules_catalogue.py validate
if (-not (Test-Path variable:gen2CountsStart)) { $gen2CountsStart = Run-Start "$gen2\inputs" rule-counts }
$gen2Counts = "$gen2\inputs\$gen2CountsStart-rule-counts"
Run-Logged $gen2Counts "$py -u -m effects rule-counts --records-dir $gen2\records --cards-folder $gen2\converted\cardsfolder --cards-folder $gen2\converted\tokenscripts --variant-scripts $gen2Variants --rules-dir rules --workers 10 --output $gen2Counts\rule-counts.json"
& $py scripts\rules_catalogue.py directed-rules --counts $gen2Counts\rule-counts.json --cards-folder $gen3\converted\cardsfolder
```

Check: `validate` passes with every scenario declaring a detector or `none` and every non-base
variant classified. The printed table and the written `rule-counts.json` agree. Review the listed
records against Forge, starting with `rules/FINDINGS.md`'s suspected departures, and record outcomes
on the scenarios. The counts decide which rules to enable for directed forks and coverage rounds:
`directed-rules` writes `rules/directed-fork-rules.txt`, `rules/coverage-wording-cards.txt` and
the part of the counts they are decided from, `rules/directed-fork-counts.json`, from them; record that choice in the gen-3 record.
On a machine without the gen-2 shards, `rules/directed-fork-counts.json` serves as `--counts`.

## Stage 2 — the relation-aware trunk (story 4, no gen-3 data)

Set first: `$gen2Shard`, one fixed gen-2 curated shard (a file under `$gen2\corpus\`).

```powershell
& $py -m pytest tests\unit\effects\domain\test_relation_attention.py
if (-not (Test-Path variable:gen2TrunkStart)) { $gen2TrunkStart = Run-Start "$gen2\reports" trunk-step }
$gen2Trunk = "$gen2\reports\$gen2TrunkStart-trunk-step"
Run-Logged $gen2Trunk "$py -u scripts\measure_trunk_step.py --shard $gen2Shard --batch-size 32 --output $gen2Trunk\trunk-step.json"
```

Check: the four layer properties pass; the step fits 8 GB; record step time, peak memory and the
relation-triple count per record in the gen-3 record.

## Stage 3 — gen-3 collection pilot (story 2)

A pilot is a scratch run: nothing outside `scratch\` reads it.

Set first: `$pilotFlags`, the pilot's other `match-outcomes` flags (its decks, match budget and
sampling rates), `""` for none.

```powershell
if (-not (Test-Path variable:pilotStart)) { $pilotStart = Run-Start "$gen3\scratch" pilot }
$pilot = "$gen3\scratch\$pilotStart-pilot"
if (-not (Test-Path variable:pilotChecksStart)) { $pilotChecksStart = Run-Start "$gen3\scratch" pilot-checks }
$pilotChecks = "$gen3\scratch\$pilotChecksStart-pilot-checks"
Run-Logged $pilot "$py -u -m sealed match-outcomes --workers 6 --effect-records $pilot --outcomes-dir $pilot --count-dropped-events --directed-fork-rules 704.5f --directed-fork-probability 1 $pilotFlags"
Run-Logged $pilotChecks "$py -u -m effects validate-corpus --effect-records $pilot $cards --variant-scripts $variants"
Run-Logged $pilotChecks "$py -u -m effects field-coverage --effect-records $pilot"
Run-Logged $pilotChecks "$py -u -m effects rule-counts --records-dir $pilot $cards --variant-scripts $variants --rules-dir rules --workers 10 --output $pilotChecks\rule-counts.json"
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

```powershell
& $py scripts\rules_catalogue.py gen3-fields
```

Check: the rules newly learnable are printed and committed with the change.

## Stage 5 — full collection, coverage and variants

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
if (-not (Test-Path variable:auditStart)) { $auditStart = Run-Start "$gen3\scratch" fork-fidelity }
$audit = "$gen3\scratch\$auditStart-fork-fidelity"
if (-not (Test-Path variable:auditChecksStart)) { $auditChecksStart = Run-Start "$gen3\scratch" fork-fidelity-checks }
$auditChecks = "$gen3\scratch\$auditChecksStart-fork-fidelity-checks"
Run-Logged $audit "$py -u -m sealed match-outcomes --workers 6 --effect-records $audit --outcomes-dir $audit --random-seat-share 0.25 --random-seat-probability 0.5 --interventions-per-game 4 --probes-per-game 4 --directed-fork-rules $rules --count-dropped-events"
Run-Logged $auditChecks "$py -u -m effects validate-corpus --effect-records $audit $cards --variant-scripts $variants"
```

The collection itself, one run directory per collecting command under `$gen3\records\`. Set
first:

- `$runFlags`, the flags all six collecting commands take (`--workers`, the three decision-sampling
  rates, the fork and random-seat flags);
- `$collectorFlags`, the flags only `collect-coverage` and `collect-variants` take (their budgets:
  `--decks-per-round`, `--target-records`, `--no-progress-rounds`), `""` for none;
- `$anyTargetCards`, the any-target coverage round's `--wording-cards` file.

Run the seven commands one at a time, each after the one before it has ended. `match-outcomes` has
no match budget: it plays until one Ctrl-C, then lets every worker finish its game, and has ended
when `console.log` shows `Done.` and the prompt is back. A collector ends itself when every card is
satisfied or retired.

```powershell
if (-not (Test-Path variable:fullStart)) { $fullStart = Run-Start "$gen3\records" full-strength }
$full      = "$gen3\records\$fullStart-full-strength"
if (-not (Test-Path variable:depletedStart)) { $depletedStart = Run-Start "$gen3\records" depleted }
$depleted  = "$gen3\records\$depletedStart-depleted"
if (-not (Test-Path variable:anytargetStart)) { $anytargetStart = Run-Start "$gen3\records" coverage-any-target }
$anytarget = "$gen3\records\$anytargetStart-coverage-any-target"
if (-not (Test-Path variable:wordingStart)) { $wordingStart = Run-Start "$gen3\records" coverage-wording }
$wording   = "$gen3\records\$wordingStart-coverage-wording"
if (-not (Test-Path variable:untargetStart)) { $untargetStart = Run-Start "$gen3\records" variants-untargeted }
$untarget  = "$gen3\records\$untargetStart-variants-untargeted"
if (-not (Test-Path variable:steeredStart)) { $steeredStart = Run-Start "$gen3\records" variants-steered }
$steered   = "$gen3\records\$steeredStart-variants-steered"

Run-Logged $full      "$py -u -m sealed match-outcomes --effect-records $full --outcomes-dir $full --directed-fork-rules $rules $runFlags"
Run-Logged $depleted  "$py -u -m sealed match-outcomes --effect-records $depleted --outcomes-dir $depleted --exclude-cards $holdout --directed-fork-rules $rules $runFlags"
Run-Logged $anytarget "$py -u -m effects collect-coverage --effect-records $anytarget $cards --wording-cards $anyTargetCards --exclude-cards $holdout --count-from $depleted $runFlags $collectorFlags"
Run-Logged $wording   "$py -u -m effects collect-coverage --effect-records $wording $cards --wording-cards rules\coverage-wording-cards.txt --directed-fork-rules $rules --exclude-cards $holdout --count-from $depleted --count-from $anytarget $runFlags $collectorFlags"
Run-Logged $untarget  "$py -u -m effects collect-variants --round untargeted --effect-records $untarget --forge-cards-path ..\forge\forge-gui\res\cardsfolder --variant-scripts $variants $cards --exclude-cards $holdout $runFlags $collectorFlags"
Run-Logged (Split-Path $counts) "$py -u -m effects rule-counts --records-dir $gen3\records $cards --variant-scripts $variants --rules-dir rules --workers 10 --output $counts"
Run-Logged $steered   "$py -u -m effects collect-variants --round steered --rule-counts $counts --effect-records $steered --forge-cards-path ..\forge\forge-gui\res\cardsfolder --variant-scripts $variants $cards --exclude-cards $holdout $runFlags $collectorFlags"
```

The `match-outcomes` run without `--exclude-cards` plays full-strength pools and feeds the
card-disjoint stratum; every other run keeps the held-out cards out. `$variants` is one tree shared
by every variant run (a variant is reused by its digest). Every reader given `$gen3\records` walks
every run under it.

The steered round runs last, on counts of everything collected before it (FR-058c). Every
collecting command takes the same three decision-sampling rates (FR-059a).

Each collecting command's run directory is its `--effect-records`: the `match-outcomes` runs keep
`match-outcomes.txt` and `cards-played.txt` there beside the shards (`--outcomes-dir`), worker logs under `logs/`,
their failure lines printed as the run goes (after each round on a collector, at each status line
on `match-outcomes`) and kept in `run.json`. A `collect-coverage` run in a new run
directory passes each earlier run's directory as `--count-from`, so their coverage counts. An
interrupted `collect-variants` restarts into the same directory with the same `--seed` and
`--continues <run id>`, the run id its `run.json` records.

A collector's round end lets every worker finish the game it is playing (at most
`--stop-timeout`, 600 s, before a kill), and workers recycle themselves at a game end
(`--recycle-seconds`, 600 s, at least 300). A run's games are cut by a crash, by the kill of the
longest-running worker at every status line (every 60 s; it passes over a worker still starting
Forge), which ends a game that crawls (about one in 400 lasts 8 to 20 minutes), by a kill past
the timeout, by a game outlasting its worker's recycle time by `--stop-timeout`, or by a second
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

```powershell
if (-not (Test-Path variable:corpusStart)) { $corpusStart = Run-Start "$gen3\corpus" curated }
$corpus = "$gen3\corpus\$corpusStart-curated"
if (-not (Test-Path variable:freezeStart)) { $freezeStart = Run-Start "$gen3\reports" selection-set }
$freeze = "$gen3\reports\$freezeStart-selection-set"
Run-Logged $corpus "$py -u -m effects build-corpus --records-dir $gen3\records --output $corpus $cards --variant-scripts $variants $tokens --workers 10"
& $py scripts\scan_output_vocabularies.py --manifest $corpus\manifest.json
& $py -m pytest tests\unit\effects\domain\test_layout_vocabularies.py
Run-Logged $freeze "$py -u scripts\effect_knowledge_probes\run.py --freeze-probe-set --corpus $corpus $cards $tokens --reports-dir $freeze"
```

`scan_output_vocabularies.py` rewrites `src/effects/domain/output_vocabularies.py`, the head's
closed subtype and keyword lists, with the lists the manifest recorded. The lists checked in come
from the 2026-11-15 pilot collections, and a checkpoint loads only against the lists it trained
on, so this step comes before any training. `test_layout_vocabularies.py` pins the head widths
those lists give (181 subtypes and 70 keywords, a per-entity head of 1,636 columns): when it
fails, set its pinned counts and widths to the new lists', then commit both files.

Check: the manifest gives, for every tagged rule in every class, share, written and shortfall per
polarity and source, and names one-sided rules and conditions, unbudgeted rules and rules left out of
a half. The freeze writes `selection-set-<digest>.json` or exits 1 naming the component that missed
its target.

Before stage 7, rerun the trunk measurement on a gen-3 curated shard and confirm it fits 8 GB.
Set first: `$gen3Shard`, one shard of `$corpus`.

```powershell
if (-not (Test-Path variable:gen3TrunkStart)) { $gen3TrunkStart = Run-Start "$gen3\reports" trunk-step }
$gen3Trunk = "$gen3\reports\$gen3TrunkStart-trunk-step"
Run-Logged $gen3Trunk "$py -u scripts\measure_trunk_step.py --shard $gen3Shard --batch-size 32 --output $gen3Trunk\trunk-step.json"
```

## Stage 7 — train (story 7)

```powershell
if (-not (Test-Path variable:trainStart)) { $trainStart = Run-Start "$models\effect-model" rule-mask-hard }
$train = "$models\effect-model\$trainStart-rule-mask-hard"
Run-Logged $train "$py -u -m effects train-effect-model --corpus $corpus $cards --variant-scripts $variants $tokens --reports-dir $freeze --model-output $train --rule-mask hard"
```

Check: each epoch line prints both strata's four components and the control score; the best epoch
saves encoder, trunk and heads together; `resume-best.pt` follows it. A soft-mask run is tried only if
a comparison shows the hard mask costs the other outputs. A `--relation-values off` run is scored on
the anthem probe and the combat grids beside the default. Each such run is a run of its own, in its
own directory under `$models\effect-model\`, its slug naming the one setting it varies.

## Stage 8 — report

The ability cache and the reports computed from a checkpoint take the name of the checkpoint's
run, its start date included, so they need no date of their own.

```powershell
$ckpt   = "$train\latest.pt"
$cache  = "$gen3\cache\$(Split-Path -Leaf $train)"
$report = "$gen3\reports\$(Split-Path -Leaf $train)"
Run-Logged $cache  "$py -u -m effects encode-abilities --checkpoint $ckpt $cards --variant-scripts $variants $tokens --output-root $cache"
Run-Logged "$report-evaluation" "$py -u -m effects evaluate-effect-model --checkpoint $ckpt $cards --variant-scripts $variants $tokens --corpus $corpus --abilities-root $cache --reports-dir $freeze --output $report-evaluation"
Run-Logged $report "$py -u scripts\effect_knowledge_probes\run.py --checkpoint $ckpt --corpus $corpus $cards $tokens --abilities-root $cache --probe-set-dir $freeze --output $report"
```

`evaluate-effect-model`'s decodability check reads a per-card win-rate table (`--win-rates`). No
stage here writes one under `output/effects/`, so the command above passes none and the check
reports itself skipped (`no win-rate table at …`). Its default names a file of the sealed
module's tree (`output/sealed/cards-win-rates.txt`): once such a file exists, the run reads it,
which the layout forbids an effects run, so name a table under `$gen3` with `--win-rates` before
that happens.

Check: the scorecard reports every probe and the rule tests on the reporting half. Results go into the
gen-3 record's Result section only.
