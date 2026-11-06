# Contract: selection set and control score

**Feature**: `025-ability-effect-model-gen3`
Authority: root spec § 14. Replaces the gen-2 frozen probe set
([`../../024-ability-effect-model-gen2/contracts/knowledge-probes.md`](../../024-ability-effect-model-gen2/contracts/knowledge-probes.md)'s
probe-set file).

## File

`output/effects/reports/selection-set-<corpus digest>.json`, canonical JSON, `digest` a sha256 over
everything but itself. Fields as [data-model.md](../data-model.md#selectionset-new-replaces-the-frozen-probe-set).
A reader refuses a missing file, one frozen for another corpus, or one edited since (`SelectionSetError`).

## Freeze

1. Read the manifest's halves (placement's, research Decision R3), map snapshot and rule budgets. The
   freeze assigns no game to a half. Exit 1 naming stratum and half when a half holds fewer than 300
   games.
2. Collect, over each half's distinct records (a record's first occurrence), probe records, grid
   records, rule-test records (detector tags; not aimed, not a fork, not interventional, not
   synthetic, not polarity `other`, not on the review list, rule `trained`, no unpatched confirmed
   departure) and the deviance sample (every copy, the smallest keyed hashes up to 16,384 a half). A
   grid keeps one record a game, then the fewest records by smallest keyed hash that bring its
   worst-case standard error to 0.09, at most 512, both recorded in `settings`. The probe items come
   from the probe scripts through ports (`scripts/effect_knowledge_probes/selection.py`).
3. Balance with r = 4 by seeded draw: rule-test polarities, probe types (against the smallest type
   that meets the item target), record-drawn grid cells.
4. Compute each item's worst-case standard error (research Decision E4; a deviance item's under both
   rule masks, for every shipped head's output terms). Leave out items above 0.1 and name them, a rule with one polarity only as such. Name
   every rule whose detector finds no test record in a half as `untested`.
5. Check every component's `√(Σ SEᵢ²) / m ≤ 0.05` and every grid cell non-empty; otherwise exit 1
   naming stratum, half and component.
6. Write the file.

The thresholds (300 games, 0.1, 0.05, r = 4) are keyword arguments of the freeze with these defaults;
no operator flag overrides them (research Decision E5).

## Score

| component | item score | rescaled |
|---|---|---|
| probes | rung-3 AUC or R² per probe type | AUC → 2·AUC − 1; R² as is |
| grids | per grid, MAE of cell-mean predicted probability against the rule's cell outcome | 1 − MAE / MAE(grid mean) |
| rule tests | per rule, mean over its outputs of the AUC ranking `fires` above `does_not_fire` | 2·AUC − 1 |
| deviance | per output term of every shipped head, 1 − loss / constant loss, weighted under the run's `--rule-mask` | as is |

A rule test reads each output at the tag's entities (averaged) or at the record. The selection set
records no scenario category, so an output's category, or member, is the one whose observed share in
the frozen records' own targets differs most between `fires` and `does_not_fire`, read as its
complement where the difference is negative; a count or delta takes the sign of the mean difference;
an output no head carries, or one no `fires` record observes, is not read, and a rule left with none
is left out. Every item is clipped below at −1. A component is the mean of its items, and a stratum score the mean
of its four components. `control = W · card_disjoint + (1 − W) · game_disjoint`. The deviance
component's items are the terms the loss denominators divide by: each per-entity field and the gate
(item named by the field, `damage_taken`), each verdict field at `[ACT]`, each global field at
`[GLOBAL]`, and each created-objects slot column summed over the slots plus the overflow flag (items
named `<head>.<term>`: `verdict.can_play`, `global.turn_structure_change`, `created_objects.count`).
A term's loss is the deviance sample's under the run's mask, the verdict and global fields weighed by
their own output's weight and the created-objects terms by the head's; its constant loss is the
best constant's over the same copies. The component leaves out items whose constant loss is at or
below `--deviance-floor`.

Selection reads the selection halves each epoch, on the averaged weights, under bf16 autocast: its
records are prepared once at startup and kept packed on the host, and each epoch runs the forward
alone. An empty component exits 1. Reporting reads the reporting halves once, on the selected
checkpoint, through the same scorer (`control_score.score_half`) under the checkpoint's recorded mask,
stratum weight and deviance floor, leniently: an empty component is NaN and named.
