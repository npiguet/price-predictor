# Contract: knowledge-probe suite

**Feature**: `024-ability-effect-model-gen2`
Authority: [`../../2026-10-06-ability-effect-model-gen2.md`](../../2026-10-06-ability-effect-model-gen2.md) § 11.2;
labels from [`../../../experiments/2026-09-19-effect-knowledge-probes-design.md`](../../../experiments/2026-09-19-effect-knowledge-probes-design.md).

## Layout

```text
scripts/effect_knowledge_probes/
├── __init__.py       # makes the suite a package; its modules import one another by name
├── common.py         # checkpoint/vocab/cache/sidecar loading; probe_context(checkpoint) (FR-075)
├── labels.py         # the ten families' label extractors (FR-079)
├── ladder.py         # rungs, linear and MLP probes, folds, share and its CI (FR-080–083)
├── sweeps.py         # the four board sweeps (FR-084)
├── ablation.py       # method B replacements and scopes (FR-085)
├── board.py          # the board-abilities ablation subsets (FR-085b)
├── board_edits.py    # BoardEdit: one-field edits to a record's board, chained, never in place
├── record_walk.py    # shard-order record selection by predicate, capped per game
├── probe_set.py      # the frozen probe set, its digest, load_probe_set (FR-076)
├── scorecard.py      # the typed scorecard run.py writes and compare.py reads (FR-088)
├── compare.py        # side-by-side scorecards (FR-087)
└── run.py            # entry point; --freeze-probe-set (FR-076)
```

The one-off probes (`ladder_check.py`, `damage_grid.py`, `combat_grid.py`, `evasion_grid.py`,
`colour_pips.py`, `target_restrictions.py`, `zone_effects.py`) sit beside them. `sweeps` and every
one-off probe but `ladder_check.py` select or edit records through `record_walk` and `board_edits`.

No module does work at import time. The suite is an importable package; the entry scripts are run by
path and put `scripts/` on `sys.path` before importing it.

## Inputs

`common.probe_context(checkpoint)` resolves one checkpoint's probing inputs, an explicit argument
overriding each: the curated corpus the checkpoint recorded, its manifest and digest, the sidecar
trees the checkpoint recorded (else `output/cardsfolder/`, `output/tokenscripts/`), and the variant
tree the manifest recorded. It refuses a checkpoint naming no curated corpus, and a corpus rebuilt
since the checkpoint trained on it.

## Probe set

- Enumerated once per curated corpus by `run.py --freeze-probe-set`.
- Keyed by provenance; holds labels, probe games of both validation strata, sweep records, and the
  interaction join rate.
- File: `output/effects/reports/knowledge-probes-set-<corpus digest[:12]>.json`, with `digest`
  = sha256 of its canonical JSON (sorted keys, no whitespace) excluding `digest`.
- Every probing run reads it through `load_probe_set`, which refuses a set frozen against another
  corpus's digest or one that no longer matches its own `digest`, and records `digest` in the
  scorecard.

## Rungs

| Rung | Features | Probes |
|---|---|---|
| 0 | trunk raw input features, every `e` zeroed | linear, MLP |
| 1 | rung 0 + acting `e` (+ target entity's pooled `e`) | linear, MLP |
| 1w | rung 1 with each `e` → fixed random vector per text, same width, seed 42 | linear, MLP |
| 1o | rung 0 + the parsed script values the target depends on | linear, MLP |
| 2 | trunk output at `[ACT]` or the target's `[CARD]` slot | linear, MLP |
| 3 | the model's own prediction, read through `HeadReading` gated by the affected probability; `damage_taken` alone reads the amount ungated (conditional), since every one of its items took damage | none |

Board-independent line properties use `e`, its width control, and trunk output at `[ACT]`.
Scores: AUC for yes/no targets, R² for amounts. Board-dependent results are reported per stratum.
Line-level results are reported for held-out and trained lines separately. A line is held out only
when every card carrying it is a held-out card.

The observed-mana label sums the counts of each `mana_produced` event's `mana_by_color`. An empty
map is mana the collector could not read and counts for nothing; a record whose every such event is
empty gets no label.

## Share

`share = (rung1 − rung0) / (rung3 − rung0)` per probe type, 95% bootstrap CI over texts (1,000
resamples). Not reported when `rung3 − rung0 < 0.05`. Headline: MLP share.

## Sweeps

| Sweep | Edit | Range | Read |
|---|---|---|---|
| toughness | one target creature's toughness | 1–8 | that creature's predicted death |
| affordability | acting player's untapped production | 0 to cost + 2 | predicted `affordable` verdict |
| board size | opposing creature count, by copying a creature entity | 0–8 | predicted deaths summed over the board |
| damage | acting text's `NumDmg$`, re-encoded by the checkpoint's encoder | 1–8 | toughness-4 target's predicted death; once on a spell, once on a triggered ability |

## Ablation

Per output field, loss increase when `e` is replaced by (a) Gaussian noise matched to the cache's
mean and covariance, (b) the mean `e` of the line's API type, (c) the `e` of the nearest other text;
each applied to every slot, to `[ACT]` only, and to card slots only.

## Scorecard

`output/effects/reports/knowledge-probes-<checkpoint stem>-<date>/scorecard.json`, shape in
[data-model.md § Scorecard](../data-model.md#scorecard). Tables beside it, one markdown file per family.

## Budget

Fits 8 GB of GPU memory; at most two GPU hours per checkpoint (SC-011). Each scorecard records the
run's wall time and peak GPU memory so the budget is checked, not assumed. Features for rungs 0–2 are
extracted in one batched forward pass per stratum and held on the host; probes then fit on the
cached features.
