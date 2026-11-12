---
name: artifact-layout
description: Binding rules for where everything under `output/` and `models/` goes and how it is named — module, generation, kind, run directory, run manifest, console and worker logs, scratch, archiving to the NAS. Load BEFORE choosing an output or checkpoint path for any command, starting any run (collection, build, training, evaluation, probe, audit, pilot), writing or moving a log, or cleaning, archiving or deleting anything under `output/` or `models/`.
---

# Artifact layout: `output/` and `models/`

Every file the code or a run produces lives at a path that says, without opening anything, which
module made it, for which generation, what kind of thing it is, and which run made it. Nothing is
written anywhere else, and nothing is written at a level above its run directory.

## The tree

```
output/<module>/<gen>/<kind>/<run>/...
models/<module>/<gen>/<model>/<run>/...
```

| level | values | rule |
|---|---|---|
| `<module>` | `price-predictor`, `sealed`, `draft`, `effects` | the package that produced it, never the one that reads it |
| `<gen>` | `gen1`, `gen2`, `gen3`, … | the module's own generation (its experiment record and spec name it); a module with no generations yet is `gen1` |
| `<kind>` | from the kind table below | fixed vocabulary; a new kind needs a line in this skill first |
| `<model>` | the model's name: `scorer`, `picker`, `encoder`, `agent`, `effect-model`, `transformer`, `sklearn` | one directory per model trained, baselines included (`effect-model-identity`) |
| `<run>` | `YYYY-MM-DD-<slug>` | see "Naming a run" |

Kinds under `output/<module>/<gen>/`:

| kind | holds | lifetime |
|---|---|---|
| `records` | raw collected corpora (shards, match outcomes, draft transcripts): append-only and expensive to rebuild | never deleted; archived to the NAS when superseded |
| `corpus` | curated datasets built from records (`build-corpus` output, generated pools, generated decks) | kept while a checkpoint references it |
| `inputs` | small generated files other steps read: holdout lists, keyword definitions, rule counts, variant scripts, converted cards | kept while anything reads them |
| `cache` | derived and rebuildable from a checkpoint or a corpus: ability caches, encodings | deletable at any time |
| `reports` | evaluations, probes, audits worth keeping, analysis outputs, charts | kept; the experiment record links them |
| `scratch` | pilots, smoke runs, debugging collections, throwaway audits | deletable at any time; nothing outside `scratch` may read from it |

A run directory under `models/<module>/<gen>/<model>/` holds the checkpoints (`{timestamp}.pt`,
`latest.pt`, `resume.pt`, …) of one training run. The ability cache and the reports computed from a
checkpoint are not inside it: they go to `output/<module>/<gen>/cache/<run>/` and
`output/<module>/<gen>/reports/<run>/`, under a run name that names the checkpoint's run.

## Naming a run

- `<run>` is `YYYY-MM-DD-<slug>`: the date the run started, then a lower-case kebab slug of at most
  four words saying what the run is (`2026-11-13-full-strength`, `2026-11-20-e256-rule-mask-soft`,
  `2026-11-21-knowledge-probes`).
- The slug names the run's purpose or its one varied setting, never its history. A continuation, a
  re-run or a fix is a new run with its own date and slug; its manifest names the run it continues.
  Suffix chains (`-ext20-last`, `-lr0.1-best`) are not used.
- Two runs started the same day with the same slug take `-2`, `-3`.
- One run is one logical job: a collection across all its worker restarts, a training run across its
  resumes, one evaluation. A resumed or restarted job writes into its existing run directory.

## What every run directory holds

| file | written | content |
|---|---|---|
| `RUN.md` | at start, completed at the end | purpose (one sentence); status; the exact command line(s); repo commit and `../forge` commit (and "dirty" if either tree had changes); the input paths it read; the run(s) it continues; start and end time; one line on how it ended |
| `console.log` | throughout | the command's stdout and stderr, unbuffered (`python -u … 2>&1 \| tee console.log`), its first line the command |
| `logs/` | throughout | every other log: Forge worker logs, per-process logs, supervisor logs |
| the artifacts | — | whatever the run produces, in the layout its command defines |

`status` in `RUN.md` is one of `running`, `keep`, `scratch`, `failed`, `superseded by <run path>`.
A run that never reaches `keep` belongs under `scratch`.

`RUN.md` is the provenance record: anything a later reader needs to reproduce or trust the run is
there or in `console.log`, never only in a chat or a commit message.

## Rules

1. Nothing is written directly under `output/`, `models/`, a module or a generation directory: no
   logs, no scripts, no loose JSON. A one-off shell script a run used is copied into that run's
   directory; a reusable one lives in `scripts/`.
2. A path is passed explicitly on the command line whenever the command's default would put the
   file elsewhere than this layout. Defaults that disagree with this skill are a bug to fix.
3. A run reads other runs by their full path and never copies their files. Its `RUN.md` lists every
   path it read.
4. Anything a `keep` run reads is `keep` too. Nothing outside `scratch` reads from `scratch`.
5. A generation's directories are not edited once the next generation starts, except to mark runs
   `superseded`.
6. `records` are never deleted. Superseded records, corpora and checkpoints move to the NAS.
7. `cache` and `scratch` may be deleted at any time without asking.

## Archiving to the NAS

The NAS (`Y:\Nicolas\mtg\mtg-models-data\`) mirrors the same tree: a run moves from
`output/effects/gen2/records/2026-10-08-depleted/` to
`Y:\Nicolas\mtg\mtg-models-data\output\effects\gen2\records\2026-10-08-depleted\`, unchanged. A path
in a `RUN.md` stays valid by swapping the root. Moving a run never renames it.

## Checklist before starting a run

1. Module, generation, kind (or model) and slug chosen; the run path is
   `output/<module>/<gen>/<kind>/<YYYY-MM-DD-slug>/` or `models/<module>/<gen>/<model>/<YYYY-MM-DD-slug>/`.
2. `RUN.md` written with purpose, status `running` (or `scratch`), command, commits and inputs.
3. The command writes its artifacts inside the run directory, its console to `console.log`, its other
   logs under `logs/`.
4. At the end: `RUN.md` gets its end time, how it ended, and its final status.
