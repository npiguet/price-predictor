# Contract: selection set and control score

**Feature**: `025-ability-effect-model-gen3`
Authority: root spec § 14. Replaces the gen-2 frozen probe set
([`../../024-ability-effect-model-gen2/contracts/knowledge-probes.md`](../../024-ability-effect-model-gen2/contracts/knowledge-probes.md)'s
probe-set file).

## File

`output/effects/reports/selection-set-<corpus digest>.json`, canonical JSON, `digest` a sha256 over
everything but itself. Fields as [data-model.md](../data-model.md#selectionset-new-replaces-the-frozen-probe-set).

## Freeze

1. Read the manifest's halves (placement's, research Decision R3), map snapshot and rule budgets. The
   freeze assigns no game to a half. Exit 1 naming stratum and half when a half holds fewer than 300
   games.
2. Collect probe records, grid records, rule-test records (detector tags; not aimed, not probe fork, not
   interventional, not polarity `other`, not on the review list, rule `trained`, no unpatched confirmed
   departure) and the deviance sample.
3. Balance with r = 4 by seeded draw: rule-test polarities, probe types, record-drawn grid cells.
4. Compute each item's worst-case standard error (research Decision E4). Leave out items above 0.1 and
   name them. Name every rule whose detector finds no test record in a half as untested.
5. Check every component's `√(Σ SEᵢ²) / m ≤ 0.05`; otherwise exit 1 naming stratum, half and component.
6. Write the file.

The thresholds (300 games, 0.1, 0.05, r = 4) are keyword arguments of the freeze with these defaults;
no operator flag overrides them (research Decision E5).

## Score

| component | item score | rescaled |
|---|---|---|
| probes | rung-3 AUC or R² per probe type | AUC → 2·AUC − 1; R² as is |
| grids | per grid, MAE of cell-mean predicted probability against the rule's cell outcome | 1 − MAE / MAE(grid mean) |
| rule tests | per rule, mean over its outputs of the AUC ranking `fires` above `does_not_fire` | 2·AUC − 1 |
| deviance | per field, 1 − loss / constant loss, weighted under the run's `--rule-mask` | as is |

Every item is clipped below at −1. A component is the mean of its items, and a stratum score the mean
of its four components. `control = W · card_disjoint + (1 − W) · game_disjoint`. The deviance
component leaves out fields whose constant loss is at or below `--deviance-floor`.

Selection reads the selection halves each epoch, on the averaged weights, under bf16 autocast.
Reporting reads the reporting halves once, on the selected checkpoint, through the same scorer.
