# Contract: selection set and control score

**Feature**: `025-ability-effect-model-gen3`
Authority: root spec § 14. Replaces the gen-2 frozen probe set
([`../../024-ability-effect-model-gen2/contracts/knowledge-probes.md`](../../024-ability-effect-model-gen2/contracts/knowledge-probes.md)'s
probe-set file).

## File

`<probe-set-dir>/selection-set-<corpus digest>.json` (the directory the freeze is given, and every
reader's `--reports-dir` or `--probe-set-dir`), canonical JSON, `digest` a sha256 over
everything but itself. Fields as [data-model.md](../data-model.md#selectionset-new-replaces-the-frozen-probe-set).
A reader refuses a missing file, one frozen for another corpus, one edited since, or one whose grid
cells hold no `answers`, frozen before grids kept each record's answer (`SelectionSetError`).
A writer never replaces a set already at the path by another (`SelectionSetError`; checkpoints were
selected against the one on disk); writing the same set again leaves the file as it is.

## Freeze

1. Read the manifest's halves (placement's, research Decision R3), map snapshot and rule budgets. The
   freeze assigns no game to a half. Exit 1 naming stratum and half when a half holds fewer than 300
   games.
2. Collect, over each half's distinct records (a record's first occurrence), probe records, grid
   records, rule-test records (detector tags; not aimed, not a fork, not interventional, not
   synthetic, not polarity `other`, not on the review list, rule `trained`, no unpatched confirmed
   departure) and the deviance sample (every copy, the smallest keyed hashes up to 16,384 a half,
   each copy weighed by its `curation.class_weight`; FR-059a). A
   grid keeps one record a game, then the fewest records by smallest keyed hash that bring its
   worst-case standard error to 0.09, at most 512, both recorded in `settings`. Each cell keeps
   the rule's outcome for each kept record (`answers`, in the grid's `records` order). The probe items come
   from the probe scripts through ports (`scripts/effect_knowledge_probes/selection.py`). Every grid
   cell is a board a record could show (`domain/grid_edits.py`):
   - a combat cell is a snapshot of its damage step, phase and substep both;
   - a pair carrying any protection (`protection_from_<quality>`) qualifies no combat grid;
   - a double striker's second hit holds, read at the blocker, the 28 cells whose attacker power is
     below the blocker's toughness, and qualifies, read at the attacker, only a block whose attacker
     power is below its blocker's toughness: a blocker the first hit destroys deals no second-step
     damage;
   - the damage grid qualifies a record only where the amount edit takes on both surfaces' texts,
     the line's script and its prose.
3. Balance with r = 4 by seeded draw: rule-test polarities, probe types (against the smallest type
   that meets the item target), record-drawn grid cells.
4. Compute each item's worst-case standard error (research Decision E4, a grid's `B` over its
   records' answers: `B = (1/C) Σ_c (1/N) Σ_r |ȳ − y_rc|`, `ȳ` the mean answer; a rule test's or a binary probe's clusters by game: `n₊` and `n₋` are the distinct games among its positive and among its negative rows; a deviance item's under both
   rule masks, for every shipped head's output terms, `√(Σ_g S_g²) / Σ_g S_g` with `S_g` summing over
   game g's copies the constant's loss times the copy's mask weight times its class weight, the
   constant fitted at the class weights). Leave out items above 0.1 and name them, a rule with one polarity only as such. A
   rule whose effective outputs are all per-entity counts only the rows whose record carries one of those outputs
   at a named outcome entity toward n and its standard error, and one with a polarity holding none is left out, `reason` `no outcome
   entity: <named> of <rows> fires, <named> of <rows> does_not_fire name one`; a rule with a verdict,
   global or created-objects output keeps every row. Name
   every rule whose detector finds no test record in a half as `untested`, and in every half every
   rule whose manifest `effective_outputs` is empty as untested for having no output to read.
5. Check every component's `√(Σ SEᵢ²) / m ≤ 0.05` and every grid cell non-empty; otherwise exit 1
   naming stratum, half and component.
6. Write the file.

The thresholds (300 games, 0.1, 0.05, r = 4) are keyword arguments of the freeze with these defaults;
no operator flag overrides them (research Decision E5).

## Score

| component | item score | rescaled |
|---|---|---|
| probes | rung-3 AUC or R² per probe type | AUC → 2·AUC − 1; R² as is |
| grids | per grid, MAE of each record's predicted probability in each cell against the rule's outcome for that record there, averaged over the cell's records, then over the cells | 1 − MAE / MAE(grid mean), the grid mean's MAE taken against the same answers |
| rule tests | per rule, mean over its outputs of the AUC ranking `fires` above `does_not_fire` | 2·AUC − 1 |
| deviance | per output term of every shipped head, 1 − loss / constant loss, weighted under the run's `--rule-mask` | as is |

A rule test reads each of the rule's effective outputs (the manifest's `effective_outputs`) at the
tag's outcome entities (averaged; research R1), the entities a rule-test record row holds, or at the
record. An output is read only where the record's targets define it, and a read at an entity the
record gives no slot is counted and logged. The selection set
records no scenario category, so an output's category, or member, is the one whose observed share in
the frozen records' own targets differs most between `fires` and `does_not_fire`, read as its
complement where the difference is negative; a count or delta takes the sign of the mean difference;
an output no head carries, or one no `fires` record observes, is not read, and a rule left with none
is left out. Every item is clipped below at −1. A component is the mean of its items, and a stratum score the mean
of its four components; the grid component is the mean over the damage, combat and evasion grids of
each grid's mean over its scored variants (`combat/…` and `combat-several/…` both being the combat
grid's; `grid_edits.grid_of`), a grid with none scored left out. `control = W · card_disjoint + (1 − W) · game_disjoint`. The deviance
component's items are the terms the loss denominators divide by: each per-entity field and the gate
(item named by the field, `damage_taken`), each verdict field at `[ACT]`, each global field at
`[GLOBAL]`, and each created-objects slot column summed over the slots plus the overflow flag (items
named `<head>.<term>`: `verdict.can_play`, `global.turn_structure_change`, `created_objects.count`).
A term's loss is the deviance sample's under the run's mask, the verdict and global fields weighed by
their own output's weight and the created-objects terms by the head's, every copy also weighed by its
`class_weight`; its constant loss is the best constant's over the same copies at the same weights,
the constant fitted at them. A copy of class weight k scores as k repeated copies. Probes, grids and
rule tests read no class weight. The component leaves out items whose constant loss is at or
below `--deviance-floor`.

Selection reads the selection halves each epoch, on the averaged weights, under bf16 autocast: its
records are prepared once at startup and kept packed on the host, and each epoch runs the forward
alone. An empty component exits 1. Reporting reads the reporting halves once, on the selected
checkpoint, through the same scorer (`control_score.score_half`) under the checkpoint's recorded mask,
stratum weight and deviance floor, leniently: an empty component is NaN and named.
