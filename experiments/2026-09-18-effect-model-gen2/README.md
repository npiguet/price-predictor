# Gen-2 effect model: data and charts

The data and charts behind two records:

- [`../2026-09-18-effect-model-gen2-improvements-design.md`](../2026-09-18-effect-model-gen2-improvements-design.md), the gen-2 record: training, the death rule, combat, gates and width.
- [`../2026-09-19-effect-knowledge-probes-design.md`](../2026-09-19-effect-knowledge-probes-design.md), the probes record: where the model's knowledge lives, the board sweeps, the ablations.

To regenerate, from the repository root:

```
python scripts/effect_gen2_report/build_data.py   # data/ from models/effects/runs/
python scripts/effect_gen2_report/make_charts.py  # charts/ from data/ alone
```

`build_data.py` reads the run directories under `models/effects/runs/`, which are not in the repository. A run that does not exist yet is skipped, so re-running it as the sweep continues adds the new runs. Figures that exist only in the records' text are kept in `scripts/effect_gen2_report/transcribed.py`, each block naming where it was measured.

## Names

A **run** is one training run: an arm (`e64`, `e128`, `e64-lethality`, ...) or a continuation (A64, B64, A128, B128, ...). A **checkpoint** is the run's kept epoch, the one with the lowest card-disjoint loss, named after the run (`e64-lethality`, `A64 best`). A continuation run's last epoch is also probed and is named `A64 last`. `e64 (earlier inputs)` is arm `e64`'s first run, before the input corrections. `gen-1` is the gen-1 checkpoint.

## Data

| file | one row per | source |
|---|---|---|
| `runs.csv` | run: width of `e`, lethality copies, learning-rate scale, the run it continues, first and last epoch | `train.log` |
| `epochs.csv` | run and epoch: training, card-disjoint and game-disjoint loss, gate F1, zone accuracy, deviance | the epoch summary lines of `train.log` |
| `fields_by_epoch.csv` | run, epoch and output field: card-disjoint loss and explained deviance (% of the constant prediction's loss removed) | the `fields:` line after each epoch summary |
| `checkpoints.csv` | probed checkpoint: its run, role (best or last), epoch and validation losses | `train.log`, and the `-last` directories |
| `gate2.csv` | checkpoint and keyword: qualifying records, direction agreement, verdict | `evaluate.log` |
| `gate3.csv` | checkpoint: mean pairwise cosine, top component share, participation ratio | `evaluate.log`, the embedding probes' `pca_full_variance.md`; gen-1 and the noise pilots transcribed |
| `probe_ladder.csv` | checkpoint, target, stratum, rung and probe: score (AUC, or R² for amounts) | the knowledge probes' `scorecard.json`; for gen-2, rung 3 of `damage_taken` from `ladder-check-amount-alone.txt` beside each checkpoint, which reads the predicted amount without the affected gate |
| `sweeps.csv` | checkpoint, sweep and edited value: mean prediction over the sweep's records | `scorecard.json` |
| `sweep_records.csv` | checkpoint, sweep, record and edited value: that record's prediction | `scorecard.json` |
| `toughness_sweep_records.csv` | toughness-sweep record: the ability's damage and kind (spell, activated, triggered) | the frozen probe set, transcribed |
| `ablation.csv` | checkpoint, field, replacement and scope: loss increase when `e` is replaced | `scorecard.json` |
| `board_abilities.csv` | checkpoint, board-abilities subset and field: loss, and its increase under card-slot noise | `scorecard.json`, or `reports/board-run/` for arm `e64` |
| `embedding_probes.csv` | checkpoint and line property: the linear embedding probe's score | the embedding probes' `probes_script.csv`; gen-1 transcribed |
| `damage_grids.csv` | checkpoint, record set, damage and toughness: predicted death | `grid-*.txt` beside each run; the arms' grids transcribed |
| `boundary_by_epoch.csv` | run and epoch: error against the rule, predicted death lethal by 1, exactly lethal and short by 1, and the gap | `boundary-by-epoch.txt`; the arms' epochs transcribed |
| `combat_grids.csv` | checkpoint, keyword variant, creature read, striker's power and struck creature's toughness: predicted death | `combat-grid.txt` and `combat-grid-steps.txt` |
| `evasion_grid.csv` | checkpoint, keyword granted to the attacker and trait given to the blocker: predicted chance of a legal block, and the rule's answer | `evasion-grid.txt` beside each checkpoint |
| `evasion_ranking.csv` | checkpoint and attacker keyword: mean prediction on allowed and forbidden blocks, and their ranking AUC | `evasion-grid.txt` |
| `target_families.csv`, `target_restrictions.csv` | checkpoint and kind of target restriction, or each `ValidTgts$` restriction: items, legal items, AUC | `target-restrictions.txt` beside each checkpoint |
| `target_flips.csv` | checkpoint and flip: targets, mean predicted legality before and after | `target-restrictions.txt` |
| `colour_pips.csv` | checkpoint and land configuration: decisions, the rule's answer, mean predicted affordable verdict, its mean change from the exact board and the share of decisions below it | `colour-pips.txt` beside each checkpoint |
| `colour_pips_ranking.csv` | checkpoint and set of configurations: ranking AUC of affordable over unaffordable boards | `colour-pips.txt` |
| `colour_pips_by_cost.csv` | checkpoint and kind of cost: mean drop below the exact board for each unaffordable configuration | `colour-pips.txt` |
| `zone_counts.csv` | checkpoint and card-count field: player items, affected players, AUC, share predicting more for the affected player, actual and predicted counts | `zone-effects.txt` beside each checkpoint |
| `zone_moves.csv`, `zone_outcomes.csv` | checkpoint and starting zone, or starting zone and outcome: cards, moved cards, AUC; mean probability of the right outcome, share whose top destination is right, the outcome most often predicted instead | `zone-effects.txt` |
| `zone_amounts.csv` | checkpoint and API: predicted count at the affected player for `NumCards$` 1 to 6 | `zone-effects.txt` |
| `zone_who.csv` | checkpoint and API: predicted count at the targeted and the other player, before and after the target is handed over | `zone-effects.txt` |
| `noise_pilot.csv`, `pilot_collections.csv`, `first_strike_repair.csv`, `lethal_hits_in_corpus.csv`, `gen1_vs_e64_field_terms.csv`, `legality_planned_blocks.csv`, `hook_defects_in_corpus.csv` | the smaller tables of the gen-2 record's details | transcribed |

Rung names in `probe_ladder.csv` follow the probes record: `0` the board alone, `1` the board plus `e`, `1w` a random vector of the same width, `1o` the true script values, `2` the trunk's output, `3` the model's own prediction (probe `model`).

## Charts

| chart | shows | data |
|---|---|---|
| `training-losses.png` | validation losses by epoch, per width | `epochs.csv` |
| `loss-noise.png` | each loss's spread around its five-epoch moving mean | `epochs.csv` |
| `probes-vs-loss.png` | probe mean against each validation loss, best-to-last arrows | `checkpoints.csv`, `probe_ladder.csv` |
| `fields-explained.png` | explained deviance by epoch for six fields | `fields_by_epoch.csv` |
| `damage-grids.png` | damage-by-toughness grids | `damage_grids.csv` |
| `toughness-sweep-records.png` | toughness sweep by damage group: range of records as a band, median as a line | `sweep_records.csv`, `toughness_sweep_records.csv` |
| `boundary-by-epoch.png` | the lethal boundary per epoch along each width's line | `boundary_by_epoch.csv` |
| `damage-sweep.png` | the damage sweep over quadrants marking the rule's side of an even chance | `sweeps.csv` |
| `combat-grids.png` | combat grids of the latest run B64 checkpoint | `combat_grids.csv` |
| `combat-keywords.png` | each combat keyword's effect against the rule | `combat_grids.csv` |
| `gate2-keywords.png` | gate 2 agreement by keyword and checkpoint | `gate2.csv` |
| `probe-ladder.png` | the read-out ladder per board target | `probe_ladder.csv` |
| `line-properties.png` | line properties read from `e` alone | `probe_ladder.csv` |
| `board-abilities.png` | losses where an unlisted static or replacement decides the answer | `board_abilities.csv` |
| `ablation.png` | loss increase when `e` is replaced with noise, by scope | `ablation.csv` |
| `evasion-grid.png` | predicted block legality by attacker keyword and blocker trait, rule cells outlined | `evasion_grid.csv` |
| `evasion-ranking.png` | ranking AUC per attacker keyword across checkpoints | `evasion_ranking.csv` |
| `target-flips.png` | predicted target legality before and after breaking a restriction's condition | `target_flips.csv` |
| `colour-pips.png` | change in the affordable verdict from the exact board per land configuration, and the ranking AUCs | `colour_pips.csv`, `colour_pips_ranking.csv` |
| `zone-outcomes.png` | share of moved cards whose most probable destination is right, by starting zone and destination | `zone_outcomes.csv` |
| `zone-amounts.png` | predicted cards at the affected player as `NumCards$` runs from 1 to 6 | `zone_amounts.csv` |
| `zone-who.png` | share of the predicted cards at the targeted player before and after retargeting | `zone_who.csv` |
| `width.png` | losses, probes and participation ratio by width | `checkpoints.csv`, `probe_ladder.csv`, `gate3.csv` |
