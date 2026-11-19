---
name: artifact-layout
description: Use when writing, proposing or handing the user ANY command line that writes files (convert, vocabulary, pool or deck generation, collection, build-corpus, training, evaluation, probe, audit, pilot, a script), including one the user will launch themselves; when choosing or naming a path under `output/` or `models/` for a flag, a brief to a subagent, a spec or a doc; when asked where a file goes or where to find one; when writing or moving a log; and before cleaning, archiving or deleting anything under `output/` or `models/`. Load it BEFORE typing the first path, even for a one-line command and even when the command has a default output path.
---

# Artifact layout: `output/` and `models/`

Every file the code or a run produces lives at a path that says, without opening anything, which
module made it, for which generation, what kind of thing it is, and which run made it. Nothing is
written anywhere else, and nothing is written at a level above its run directory.

## The tree

```
output/<module>/<gen>/converted/{cardsfolder,tokenscripts}/...
output/<module>/<gen>/<kind>/<run>/...
models/<module>/<gen>/<model>/<run>/...
```

| level | values | rule |
|---|---|---|
| `<module>` | `price-predictor`, `sealed`, `draft`, `effects` | the module whose work the run is part of, whichever package's command writes the files (see "One module, one tree") |
| `<gen>` | `gen1`, `gen2`, `gen3`, … | the module's own generation (its experiment record and spec name it); a module with no generations yet is `gen1` |
| `<kind>` | from the kind table below | fixed vocabulary; a new kind needs a line in this skill first |
| `<model>` | the model's name: `scorer`, `picker`, `encoder`, `agent`, `effect-model`, `transformer`, `sklearn` | one directory per model trained, baselines included (`effect-model-identity`) |
| `<run>` | `YYYY-MM-DD-<slug>` | see "Naming a run" |

Kinds under `output/<module>/<gen>/`:

| kind | holds | lifetime |
|---|---|---|
| `records` | raw collected corpora (shards, match outcomes, draft transcripts): append-only and expensive to rebuild | never deleted; archived to the NAS when superseded |
| `corpus` | curated datasets built from records (`build-corpus` output, generated pools, generated decks) | kept while a checkpoint references it |
| `inputs` | small generated files other steps read: holdout lists, keyword definitions, rule counts, variant scripts | kept while anything reads them |
| `cache` | derived and rebuildable from a checkpoint or a corpus: ability caches, encodings | deletable at any time |
| `reports` | evaluations, probes, audits worth keeping, analysis outputs, charts | kept; the experiment record links them |
| `scratch` | pilots, smoke runs, debugging collections, throwaway audits | deletable at any time; nothing outside `scratch` may read from it |

`converted/` is the generation's converted card and token trees, with their provenance sidecars,
shared by every run of that generation. It has no run level: `converted/` itself holds the
conversion's `run.json`, `console.log`, `cardsfolder/` and `tokenscripts/`. A generation is
converted once, before its first record is collected, because records name their abilities by
provenance keys that join against these sidecars: a reconversion after records exist starts a new
generation, or is archived together with every record that joined against the old one.

A run directory under `models/<module>/<gen>/<model>/` holds the checkpoints (`{timestamp}.pt`,
`latest.pt`, `resume.pt`, …) of one training run. The caches and the reports computed from its
checkpoints are not inside it: they go to `output/<module>/<gen>/cache/<run>/` and
`output/<module>/<gen>/reports/<run>/`, where `<run>` is the training run's own name, its date
included, however much later they are computed. Each of the two is one run, however many commands
write into it:

```
output/effects/gen3/reports/2026-11-20-e256-rule-mask/
  run.json              # one line in commands, one entry in invocations, per report command
  kept-epoch/           # the checkpoint the training run keeps
    knowledge-probes/   # one directory per report command, holding its console.log
    combat-grid/
  last-epoch/           # the run's final epoch, when it is not the kept one
    knowledge-probes/
```

- Each probed checkpoint gets a directory named for which epoch it is: `kept-epoch/`, `last-epoch/`,
  or `epoch-NN/` (`epoch-52/`) for any other, whatever made that epoch interesting. Each report
  command on it writes one directory inside that, named for the command (`knowledge-probes/`,
  `embedding-probes/`, `combat-grid/`), with its own `console.log`.
- A report command run later on the same training run appends to that reports run's `run.json`
  (`commands`, `invocations`). It does not start a new dated run.
- The cache run is a separate run with its own `run.json` and the same epoch directories; each epoch
  directory holds that checkpoint's cache directly, with the encoding command's `console.log` beside
  it.
- An evaluation or analysis that reads checkpoints of more than one training run is a run of its
  own under `reports/`, named for its own start date and purpose.

## One module, one tree

Work on one module is independent of work on every other module: everything that work produces
or reads lives under that module's own `output/<module>/` and `models/<module>/`.

- The module is the one being worked on, not the package whose command runs. `python -m
  price_predictor convert` run for the effects module writes `output/effects/<gen>/converted/`;
  `python -m sealed match-outcomes --effect-records` run for an effects collection writes under
  `output/effects/<gen>/records/`.
- A run reads only from its own module's tree. When effects work needs something another package
  makes (converted cards, a vocabulary, pools, decks, a checkpoint), that thing is produced again
  under `output/effects/` or `models/effects/`, by a run of its own with its own `run.json`. It is
  never read from `output/sealed/`, `output/price-predictor/` or another module's `models/` tree.
- So each module has its own `converted/` trees, and reconverting for one module changes nothing
  another module reads.
- Deleting, archiving or rebuilding one module's tree leaves every other module's runs reproducible.

## Naming a run

- `<run>` is `YYYY-MM-DD-<slug>`: the date the run started, then a lower-case kebab slug of at most
  four words saying what the run is (`2026-11-13-full-strength`, `2026-11-20-e256-rule-mask-soft`,
  `2026-11-21-knowledge-probes`).
- The slug names the run's purpose or its one varied setting, never its history. A continuation, a
  re-run or a fix is a new run with its own date and slug; its manifest names the run it continues (`continues` in `run.json`).
  Suffix chains (`-ext20-last`, `-lr0.1-best`) are not used.
- Two runs started the same day with the same slug take `-2`, `-3`.
- One run is one logical job: a collection across all its worker restarts, a training run across its
  resumes, every report on one training run's checkpoints (see "The tree"), one evaluation across
  several training runs. A resumed or restarted job writes into its existing run directory.

## What every run directory holds

| file | written | content |
|---|---|---|
| `run.json` | by the command itself: at start, completed at the end | the run manifest below |
| `console.log` | throughout | the command's stdout and stderr, unbuffered (`python -u … 2>&1 \| tee console.log`), its first line the command |
| `logs/` | throughout | every other log: Forge worker logs, per-process logs, supervisor logs, and the scratch counters a supervisor reads its workers' progress from; a collecting supervisor's per-slot stop files (`<run id>.<slot>.stop`) live here while its workers run and are deleted when its pool ends (kept only beside a worker that outlived the kill) |
| the artifacts | — | whatever the run produces, in the layout its command defines; inputs the run's own workers read while it runs (a collector's decks files) go in a subdirectory of their own, such as `decks/` |

`run.json` is one JSON object, read and edited by scripts as well as by hand:

```json
{
  "run": "output/effects/gen3/records/2026-11-13-full-strength",
  "purpose": "Full-strength collection for the card-disjoint stratum.",
  "status": "running",
  "superseded_by": null,
  "commands": ["python -u -m sealed match-outcomes --workers 6 ..."],
  "commits": {"repo": "7bb254d3", "forge": "d44cf28493c", "dirty": false},
  "inputs": ["output/effects/gen3/converted"],
  "continues": [],
  "started": "2026-11-13T09:12:00+01:00",
  "ended": null,
  "outcome": null
}
```

- The command writes `status` `running` at start and `finished` or `failed` at the end; the operator
  then sets `keep`, `scratch` or `superseded`, and fills `purpose`. `superseded_by` names the
  replacing run's path when `superseded`. A run that never reaches `keep` belongs under `scratch`.
- A resumed or restarted command appends its command line to `commands` and sets `status` back to
  `running`.
- `invocations` holds one entry per line of `commands`, in order: that command's own `commits`,
  `started`, `ended` and `outcome`, and what it records while it runs (a collector's supervisor
  `run_id`, resolved `caps`, worker-log failures and the `conversion` its keys join against;
  `convert`'s `conversion`, the Forge commit and sidecar digest every collection checks). The
  top-level `commits` and `started` are the run's first command's; `ended` and `outcome` its last's.
- The run directory is the command's output directory argument, or the directory holding its output
  file argument.
- `inputs` and `continues` hold run paths relative to the repository root.
- `outcome` is one line on how the run ended.
- `run.json` is the provenance record: anything a later reader needs to reproduce or trust the run
  is there or in `console.log`, never only in a chat or a commit message.

## Rules

1. Nothing is written directly under `output/`, `models/`, a module or a generation directory: no
   logs, no scripts, no loose JSON. A one-off shell script a run used is copied into that run's
   directory; a reusable one lives in `scripts/`.
2. Every input and output path is passed explicitly on the command line, wired to this layout.
   No run relies on a command's default path.
3. A run reads other runs of its own module by their full path and never copies their files. Its
   `run.json` lists every path it read. It reads nothing from another module's tree.
4. Anything a `keep` run reads is `keep` too. Nothing outside `scratch` reads from `scratch`.
5. A generation's directories are not edited once the next generation starts, except to mark runs
   `superseded`.
6. `records` are never deleted. Superseded records, corpora and checkpoints move to the NAS.
7. `cache` and `scratch` may be deleted at any time without asking.

## Archiving to the NAS

The NAS (`Y:\Nicolas\mtg\mtg-models-data\`) mirrors the same tree: a run moves from
`output/effects/gen2/records/2026-10-08-depleted/` to
`Y:\Nicolas\mtg\mtg-models-data\output\effects\gen2\records\2026-10-08-depleted\`, unchanged. A path
in a `run.json` stays valid by swapping the root. Moving a run never renames it.

## Checklist before starting a run

1. Module (the one being worked on), generation, kind (or model) and slug chosen; the run path is
   `output/<module>/<gen>/<kind>/<YYYY-MM-DD-slug>/` or `models/<module>/<gen>/<model>/<YYYY-MM-DD-slug>/`.
2. Every input and output path on the command line points into this layout, inside the same
   module's tree; the output points into the run directory.
3. The console goes to `console.log` in the run directory (`python -u … 2>&1 | tee …/console.log`),
   other logs under `logs/`.
4. At the end: in `run.json`, the operator fills `purpose` and sets `keep`, `scratch` or
   `superseded`.
