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
- a non-base variant with neither `operation` (one of the seven) nor `{"unclassifiable": reason}`;
- a `departure` outside its enums, or `resolution` without `status = confirmed`.

The enums are imported from `effects.domain.rule_catalogue`.

## `python scripts/rules_catalogue.py gen3-fields` (existing)

No new flag. Recomputes `learnable_after_gen3` after a map rerun, and prints the rules newly learnable
against the previous `rules/gen3-fields.json`.

## `python -m effects rule-counts` (new)

| Flag | Default | Meaning |
|---|---|---|
| `--records-dir DIR` | required | shards to count; gen-2 or gen-3, not mixed |
| `--output PATH` | `output/effects/rule-counts.json` | counts file |
| `--cards-folder DIR` | `output/cardsfolder` and `output/tokenscripts` | sidecars the detectors read; repeatable, as `build-corpus` |
| `--variant-scripts DIR` | `output/effects/variant-scripts` | variant sidecars |
| `--rules-dir DIR` | `rules` | scenarios and their detector declarations |
| `--workers N` | as `build-corpus` | parallel shard readers |

Prints one line per rule, grouped by unit: `fires` and `does_not_fire` counts, each split
`random_seat / forge_ai / aimed`; then the rules declared `none` with their reasons, the rules not
countable on the records' generation, and the size of the review list. Exit 1 when the directory
holds no shard or mixes generations.

## `python -m sealed match-outcomes` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--count-dropped-events` | off | count events arriving with no open record, by event type and engine phase; print at the end |
| `--directed-fork-rules LIST` | none | comma-separated rule ids |
| `--directed-forks-per-game N` | 2 | directed-fork budget per game, beside the existing fork budgets |
| `--directed-fork-probability P` | 0.25 | chance a matching boundary forks; must be in (0, 1] |

Startup, exit 1 naming the rule when a listed rule:

- has no detector declaration, or a declaration `none`;
- has no non-base variant classified into an operation;
- is not registered in `forge-connector` (checked by one run of `DirectedRulesMain`).

`--directed-fork-probability` outside (0, 1] exits 1. The flags ride the cap properties
`effect.directed.fork.rules`, `effect.directed.forks.per.game`, `effect.directed.fork.probability`,
`effect.count.dropped.events`.

## `python -m effects collect-coverage` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--wording-cards PATH` | none | one card name per line; a coverage round over them |
| `--count-dropped-events` | off | as `match-outcomes` |
| `--directed-fork-rules LIST` | none | a label only: rule ids written to every record's `aimed_at`; no directed fork runs, and no startup check applies |

## `python -m effects collect-variants` (existing)

Takes `--count-dropped-events`, `--directed-fork-rules`, `--directed-forks-per-game` and
`--directed-fork-probability` as `match-outcomes` does, with the same startup checks and exits.
Every record carries `aimed_at` (the listed rules, or `[]`).

## `python -m effects build-corpus` (existing)

No new flag. Behaviour:

- Exit 1 unless every record carries `generation: 3`.
- Reads `rules/heads/` and `rules/scenarios/` once; writes the map snapshot and its digest.
- Writes no `turn_action` record; counts them by subkind.
- Exit 1 when it can already tell that a rule-test component misses its precision target, naming
  stratum, half and component.

## `scripts/effect_knowledge_probes/run.py --freeze-probe-set` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--corpus DIR` | required | the curated corpus to freeze against |
| `--checkpoint PATH` | — | no longer accepted with `--freeze-probe-set` |

Writes `output/effects/reports/selection-set-<corpus digest>.json`
([selection-set.md](selection-set.md)). Exit 1 when a component misses its precision target or a
balance rule cannot be met, naming stratum, half and component.

## `python -m effects train-effect-model` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--rule-mask hard\|soft` | hard | mask on added copies |
| `--deviance-floor F` | 0.04 | constant-prediction loss floor, in nats |
| `--ema-epochs E` | 1 | span of the averaged weights |
| `--relation-values on\|off` | on | train the relation value tables |
| `--stratum-weight W` | 0.5 | card-disjoint share of the control score, in [0, 1] |
| `--patience N` | 5 | epochs without a new best control score |
| `--e-dim` | 256 | was 64 |

Startup, exit 1 when:

- a record lacks `generation: 3`;
- the manifest has no per-rule counts;
- no selection set exists for the corpus digest;
- resuming with a changed `--rule-mask`, `--deviance-floor`, `--ema-epochs`, `--stratum-weight` or
  `--relation-values`.

During training, exit 1 when an epoch meets a component with nothing to score, naming stratum, half
and component.

Epoch line adds, per stratum: `probes`, `grids`, `rule_tests`, `deviance`, `score`; then `control`
and `best`. Shard line adds `rel_k_norm` and `rel_v_norm` (pre-clip, last step of the shard).

## `python -m effects evaluate-effect-model` and the knowledge probes (existing)

No new flag. Exit 1 unless every record carries `generation: 3`, or when the checkpoint's
`input_layout` is not `gen-3`. Scorecards add the § 14.5 probes and the rule tests, per half.
