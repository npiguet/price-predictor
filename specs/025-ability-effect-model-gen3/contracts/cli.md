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
`NOT_DIRECTED`, 104.2a, 500.11, 701.12d and 702.62d, each printed with its reason) and
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
| `--workers N` | the CPU count, as `build-corpus` | parallel shard readers |

Prints one line per rule, grouped by unit: `fires` and `does_not_fire` counts, each split
`random_seat / forge_ai / aimed`; then the rules declared `none` with their reasons, the rules not
countable on the records' generation, and the size of the review list. The counts file omits a rule
no record tags. Exit 1 when the directory holds no shard, holds a gen-1 shard, or mixes generations.

## `python -m sealed match-outcomes` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--playability-rate P` | 0.01 (was 0.1) | stage 1 of decision sampling (FR-033b): the share of enumerated decision candidates evaluated; a dropped one is not evaluated for the record |
| `--zone-candidate-rate R` | 0.0001 | share of zone-bucket candidates written, both stages together |
| `--plain-candidate-rate R` | 0.001 | share of plain-bucket candidates written, both stages together |
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

The three sampling rates ride `effect.playability.rate`, `effect.zone.candidate.rate` and
`effect.plain.candidate.rate`, and are shared with `collect-coverage` and `collect-variants` through
the common cap flags (`add_cap_flags`; Python `CollectionCaps.zone_candidate_rate`,
`plain_candidate_rate`). A rate outside [0, 1], or a candidate rate above `--playability-rate`, is
refused before any game, as the random-seat flags are (`candidate_rate_error`): exit 2 on
`match-outcomes`, 1 on the `effects` collectors. A special candidate (FR-033b) is written whenever
stage 1 keeps it. The held-out coverage round sets `--playability-rate` to 1 and keeps the two
candidate rates.

## `python -m effects collect-coverage` (existing)

| Flag | Default | Meaning |
|---|---|---|
| `--wording-cards PATH` | none | one card name per line; a coverage round over them |
| `--count-dropped-events` | off | as `match-outcomes` |
| `--directed-fork-rules LIST` | none | a label only: rule ids written to every record's `aimed_at`; no directed fork runs, and no startup check applies |

`--wording-cards` beside `--only-cards` exits 1.

## `python -m effects collect-variants` (existing)

Takes `--count-dropped-events`, `--directed-fork-rules`, `--directed-forks-per-game` and
`--directed-fork-probability` as `match-outcomes` does, with the same startup checks and exits.
Every record carries `aimed_at` (the listed rules, or `[]`). A protection keyword line of a source
script swaps its quality for another of `PROTECTION_QUALITY_WHITELIST` (FR-058) before any other
perturbation of that line is tried.

## `python -m effects build-corpus` (existing)

No new flag. Behaviour:

- Exit 1 unless every record carries `generation: 3`.
- Reads `rules/heads/` and `rules/scenarios/` once; writes the map snapshot and its digest.
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
and `best`, then a line of its own giving the control score's seconds against the epoch's training
seconds. Shard line adds `rel_k_norm` and `rel_v_norm` (pre-clip, last step of the shard).

## `python -m effects evaluate-effect-model` and the knowledge probes (existing)

`evaluate-effect-model`: no new flag. Exit 1 unless every record carries `generation: 3`, or when
the checkpoint's `input_layout` is not `gen-3`. It scores the reporting halves of the selection set
found by the corpus digest under `output/effects/reports/` (the config's `reports_dir`), under the
checkpoint's recorded `--rule-mask`, `--stratum-weight` and `--deviance-floor`, and prints per stratum
`reporting-half <stratum>` (components and score) with each component's items, `rule-tests <stratum>
reporting` per rule, and `reporting-half control` beside the checkpoint's `best_control_score`;
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
