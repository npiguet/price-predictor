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
python -u -m effects rule-counts --records-dir output/effects/records-gen2
```

Check: `validate` passes with every scenario declaring a detector or `none` and every non-base
variant classified. The printed table and `output/effects/rule-counts.json` agree. Review the listed
records against Forge, starting with `rules/FINDINGS.md`'s suspected departures, and record outcomes
on the scenarios. The counts decide which rules to enable for directed forks and coverage rounds;
record that choice in the gen-3 record.

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

A pilot corpus built from these records cannot meet the precision target: `build-corpus` writes the
corpus and manifest, then exits 1. Read the manifest anyway; the freeze is not run on a pilot.

## Stage 4 — the map rerun after each layout change (story 5)

After each input or output change lands, update `rules/MODEL-FIELDS.md`, rerun the heads workflow
(`scripts/rules_heads_workflow.js`), then:

```bash
python scripts/rules_catalogue.py gen3-fields
```

Check: the rules newly learnable are printed and committed with the change.

## Stage 5 — full collection, coverage and variants

```bash
python -u -m sealed match-outcomes --effect-records --directed-fork-rules "$(paste -sd, rules/directed-fork-rules.txt)" <run flags>
python -u -m effects collect-coverage --wording-cards <any-target cards> <run flags>
python -u -m effects collect-coverage --wording-cards rules/coverage-wording-cards.txt --directed-fork-rules "$(paste -sd, rules/directed-fork-rules.txt)" <run flags>
python -u -m effects collect-variants --directed-fork-rules 702.16 <run flags>
```

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
python -u -m effects encode-abilities --checkpoint models/effects/latest.pt
python -u -m effects evaluate-effect-model --checkpoint models/effects/latest.pt
python -u scripts/effect_knowledge_probes/run.py --checkpoint models/effects/latest.pt
```

Check: the scorecard reports every probe and the rule tests on the reporting half. Results go into the
gen-3 record's Result section only.
