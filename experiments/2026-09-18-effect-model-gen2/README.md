# Gen-2 effect model: data and charts

The data and charts behind two records:

- [`../2026-09-18-effect-model-gen2-improvements-design.md`](../2026-09-18-effect-model-gen2-improvements-design.md), the gen-2 record: training, the death rule, combat, gates and width.
- [`../2026-09-19-effect-knowledge-probes-design.md`](../2026-09-19-effect-knowledge-probes-design.md), the probes record: where the model's knowledge lives, the board sweeps, the ablations.

To regenerate, from the repository root:

```
python scripts/effect_gen2_report/build_data.py     # data/ from models/effects/runs/
python scripts/effect_gen2_report/make_charts.py    # the data charts, from data/ alone
python scripts/effect_gen2_report/make_diagrams.py  # the six diagrams, which read no data
```

Both drawing scripts also take function names on the command line and then redraw only those charts or diagrams, as in `make_charts.py ping_deaths`.

`build_data.py` reads the run directories under `models/effects/runs/`, which are not in the repository. A run that does not exist yet is skipped, so re-running it as the sweep continues adds the new runs. Figures that exist only in the records' text are entered by hand in the CSVs themselves, and the Data table below marks them transcribed. When `build_data.py` rewrites a CSV that mixes computed and hand-entered rows, it keeps every hand-entered row the run directories do not produce.

## Names

A **run** is one training run: an arm (`e64`, `e128`, `e64-lethality`, ...) or a continuation. An arm trains from scratch, and one whose name ends in `-lethality` trained with lethality copies. An A run (A64, A128, A256) resumes its arm from the arm's last epoch for twenty epochs at the full learning rate; a B run (B64, B128, B256) starts from its A run's kept epoch and trains twenty epochs at a tenth of it. A **checkpoint** is the run's kept epoch, the one with the lowest card-disjoint loss, named after the run (`e64-lethality`, `A64 best`). A continuation run's last epoch is also probed and is named `A64 last`. Run A256's lowest epoch is its last, so `A256 best` and `A256 last` are one checkpoint. Run B256 never went below the checkpoint it started from and keeps run A256's epoch 53; `B256 best` is its own lowest epoch, 55, probed in the run directory's `-best` copy. `e64 (earlier inputs)` is arm `e64`'s first run, before the input corrections. `gen-1` is the gen-1 checkpoint.

Every data chart uses one visual language. Colour is the width of `e`: blue 64, orange 128, green 256. The stage of training is the marker shape for points (triangle for an arm, square for an A run, circle for a B run) and the line style for lines (dotted, dashed, solid). A filled marker is a run's kept epoch and a hollow one its last; `B256 best` is half filled. Arms without lethality copies are grey triangles, `e64 (earlier inputs)` the lightest, and gen-1 a dark grey cross.

## Data

| file | one row per | source |
|---|---|---|
| `runs.csv` | run: width of `e`, lethality copies, learning-rate scale, the run it continues, first and last epoch | `train.log` |
| `epochs.csv` | run and epoch: training, card-disjoint and game-disjoint loss, gate F1, zone accuracy, deviance | the epoch summary lines of `train.log` |
| `fields_by_epoch.csv` | run, epoch and output field: card-disjoint loss and explained deviance (% of the constant prediction's loss removed) | the `fields:` line after each epoch summary |
| `checkpoints.csv` | probed checkpoint: its run, role (best or last), epoch and validation losses | `train.log`, and the `-last` and `-best` directories |
| `gate2.csv` | checkpoint and keyword: qualifying records, direction agreement, verdict | `evaluate.log` |
| `gate3.csv` | checkpoint: mean pairwise cosine, top component share, participation ratio | `evaluate.log`, the embedding probes' `pca_full_variance.md`; gen-1 and the noise pilots transcribed |
| `probe_ladder.csv` | checkpoint, target, stratum, rung and probe: score (AUC, or R² for amounts) | the knowledge probes' `scorecard.json`; for gen-2, rung 3 of `damage_taken` from `ladder-check-amount-alone.txt` beside each checkpoint, which reads the predicted amount without the affected gate |
| `sweeps.csv` | checkpoint, sweep and edited value: mean prediction over the sweep's records | `scorecard.json` |
| `sweep_records.csv` | checkpoint, sweep, record and edited value: that record's prediction | `scorecard.json` |
| `toughness_sweep_records.csv` | toughness-sweep record: the ability's damage and kind (spell, activated, triggered) | the frozen probe set, transcribed |
| `ping_gate_edits.csv` | edit to the toughness sweep's activated damage records: mean P(affected) at the targeted creature, per checkpoint | scratch reads of four checkpoints, transcribed |
| `damage_targets_in_training.csv` | kind of damage line: its training resolutions, who they target, and how often a creature target is marked affected | scratch scan of 150 training shards, transcribed |
| `boundary_records_gate.csv` | kind of acting line among the boundary chart's 40 training rows: rows whose P(affected) at the hit creature is below one half, per checkpoint | scratch reads of five checkpoints, transcribed |
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
| `noise_pilot.csv`, `pilot_collections.csv`, `first_strike_repair.csv`, `lethal_hits_in_corpus.csv`, `gen1_vs_e64_field_terms.csv`, `legality_planned_blocks.csv`, `hook_defects_in_corpus.csv` | the smaller tables of the gen-2 record's details and its gen-1 comparison | transcribed |
| `rule_families.csv` | rule family in the rarity table of gen-1's corpus: distinct texts and games | transcribed from the gen-2 record's design section |

Rung names in `probe_ladder.csv` follow the probes record: `0` the board alone, `1` the board plus `e`, `1w` a random vector of the same width, `1o` the true script values, `2` the trunk's output, `3` the model's own prediction (probe `model`).

## Charts

`make_charts.py` draws every data chart below, and `make_diagrams.py` the six diagrams.

| chart | shows | data |
|---|---|---|
| `run-lineage.png` | every gen-2 run by epoch, what each continuation starts from, and the probed checkpoints | `runs.csv`, `checkpoints.csv` |
| `training-losses.png` | training and validation losses by epoch, per width, each run's lowest card-disjoint epoch marked | `epochs.csv` |
| `loss-noise.png` | each loss's spread around its five-epoch moving mean, from the arm through the A run to the B run | `epochs.csv` |
| `probes-vs-loss.png` | probe mean against each validation loss, arrows from each run's lowest card-disjoint epoch to its last | `checkpoints.csv`, `probe_ladder.csv` |
| `loss-rise-by-field.png` | each continuation run's late card-disjoint rise, split by output field | `epochs.csv`, `fields_by_epoch.csv` |
| `fields-explained.png` | explained deviance by epoch for six fields | `fields_by_epoch.csv` |
| `damage-grids.png` | damage-by-toughness grids, with each panel's error against the rule | `damage_grids.csv` |
| `toughness-sweep-records.png` | toughness sweep by damage group: range of records as a band, median as a line | `sweep_records.csv`, `toughness_sweep_records.csv` |
| `lethal-hits.png` | lethal and not-lethal hits by kind of damage in 40 training shards | `lethal_hits_in_corpus.csv` |
| `ping-deaths.png` | each checkpoint's predicted death of a toughness-1 creature under four activated pings | `sweep_records.csv`, `toughness_sweep_records.csv` |
| `boundary-by-epoch.png` | predicted death given affected, for hits lethal by one, exactly lethal and short by one, per epoch | `boundary_by_epoch.csv` |
| `damage-sweep.png` | the damage sweep for every checkpoint, with the wrong side of the rule shaded | `sweeps.csv` |
| `damage-sweep-triggered.png` | the damage sweep on a triggered ability, whose amount sits in the line it executes | `sweeps.csv` |
| `ping-gate-edits.png` | the affected gate at a ping's targeted creature under six edits, each from the record as collected, four checkpoints | `ping_gate_edits.csv` |
| `combat-grids.png` | combat grids of B64 last: plain deaths, and the change with deathtouch and with first strike | `combat_grids.csv` |
| `combat-keywords.png` | the share of the rule's change each combat keyword produces, per checkpoint | `combat_grids.csv` |
| `gate2-keywords.png` | gate 2 agreement by keyword and checkpoint, against the sampling noise at each row's median | `gate2.csv` |
| `width.png` | losses, probe mean and participation ratio across the three stages, per width | `checkpoints.csv`, `probe_ladder.csv`, `gate3.csv` |
| `scorecard.png` | thirteen measures by seventeen checkpoints, coloured by how far each goes toward the rule, with each checkpoint's mean distance toward it | `checkpoints.csv`, `probe_ladder.csv`, `damage_grids.csv`, `sweeps.csv`, `sweep_records.csv`, `gate2.csv`, `evasion_ranking.csv`, `colour_pips_ranking.csv`, `zone_amounts.csv`, `zone_who.csv`, `target_flips.csv` |
| `planned-block-labels.png` | shares of the "may not block" labels and block records that reflect the AI's block plan | `legality_planned_blocks.csv` |
| `embedding-probes.png` | linear probes on `e` for gen-1 and every gen-2 checkpoint, and each checkpoint's participation ratio | `embedding_probes.csv`, `gate3.csv` |
| `gen1-vs-first-arm.png` | the shared field terms by epoch for gen-1 and the two `e64` runs, and the gap at epoch 18 by field | `fields_by_epoch.csv` |
| `noise-pilot.png` | the stage-0 noise pilots: validation measures and the shape of `e` at three noise ratios | `noise_pilot.csv`, `gate3.csv` |
| `rule-families.png` | distinct texts and games per rule family in gen-1's corpus | `rule_families.csv` |
| `probe-ladder.png` | the read-out ladder per board target, and the share `e` closes | `probe_ladder.csv` |
| `e-over-random.png` | rung 1 minus rung 1w: what `e` adds over a random vector of its width, per target and checkpoint | `probe_ladder.csv` |
| `line-properties.png` | line properties and the outcome spread across boards read from `e` alone, trained and held-out lines | `probe_ladder.csv` |
| `board-abilities.png` | loss on restricted records over loss on unrestricted controls, per checkpoint | `board_abilities.csv` |
| `ablation.png` | loss increase when `e` is replaced, by replacement and scope | `ablation.csv` |
| `share-vs-reliance.png` | the share `e` carries against the trunk's reliance on the acting ability's `e`, per board question | `probe_ladder.csv`, `ablation.csv` |
| `other-sweeps.png` | affordability sweep and board-size sweep | `sweeps.csv` |
| `evasion-grid.png` | median lift in predicted block legality by attacker keyword and blocker trait, rule-allowed cells marked | `evasion_grid.csv` |
| `evasion-ranking.png` | ranking AUC per attacker keyword across checkpoints | `evasion_ranking.csv` |
| `target-flips.png` | the change in predicted target legality when a restriction's condition is broken | `target_flips.csv` |
| `colour-pips.png` | change in the affordable verdict from the exact board per land configuration, and the ranking AUCs | `colour_pips.csv`, `colour_pips_ranking.csv` |
| `zone-outcomes.png` | share of moved cards whose most probable destination is right, by starting zone and destination | `zone_outcomes.csv` |
| `zone-amounts.png` | predicted cards at the affected player as `NumCards$` runs from 1 to 6 | `zone_amounts.csv` |
| `zone-who.png` | share of the predicted cards left at the originally targeted player after retargeting | `zone_who.csv` |

| diagram | shows |
|---|---|
| `diagram-model-and-probes.png` | the model's parts, the trunk's slots, where each read-out ladder rung reads, and the ablation scopes |
| `diagram-validation-strata.png` | how whole games are split into training and the two validation strata |
| `diagram-combat-step.png` | one combat's two damage steps, what a gen-2 combat record captured, and the collector defects |
| `diagram-encoder-reads.png` | what the encoder reads for a trigger, a chain and a charm, gen-1 against gen-2 |
| `diagram-design-changes.png` | gen-2's pipeline from the Forge scripts to the evaluation, each design change under the stage where it acts |
| `diagram-noise-shapes.png` | the training noise on one `e` under each noise option the design weighed, on a schematic cloud of `e` vectors |
