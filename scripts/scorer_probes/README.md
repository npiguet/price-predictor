# Scorer-behavior probes

The probe battery behind
[`experiments/2026-08-27-scorer-preferences.md`](../../experiments/2026-08-27-scorer-preferences.md)
— inference-only analyses of the gen-4 production sealed scorer. Every script
loads the production checkpoint via `probe_lib.py`, scores decks in batched
forward passes, and writes its outputs to `output/scorer-probes/` (gitignored).
The session that produced the document left its exact outputs there
(`t*_report.md`, `t*_results.json`, the CSVs); rerunning a script regenerates
them.

Run from this directory with the repo venv:
`..\..\.venv\Scripts\python.exe <script>`. Most scripts accept `--smoke`
(tiny CPU run). External inputs: the Y: corpus files (`probe_lib.YDATA`),
`output/cardsfolder-512/`, `models/sealed/scorer/512-…_mwlog.pt`, and for
`forge_hints.py` / `human_rank_probe.py` / `cardtool.py` a sibling Forge
checkout at `../forge`.

## Script → document section

| script | document section(s) it feeds | output |
|---|---|---|
| `probe_lib.py` | shared harness (checkpoint load, batched scoring, corpus readers, card features, win-rate labels) | — |
| `t0_landscape.py` | "Three quarters of the score range…"; builder means; inputs to T7-A | `t0_decks.csv` |
| `t5_ablation.py` | "The card-text embedding carries the signal…" (ablation table, bag-of-text) | `t5_results.json` |
| `t5b_det_groups.py` | per-group ablation of the 32 deterministic features (paired stats, both post-cutoff corpora) — "Inside the deterministic features…" | `t5b_results.json` |
| `make_text_pca.py` | PC1/top-k variance shares; input for t6 P1 | `text_pca_512.npz` |
| `t6_mechanism.py` | "The pooling layer is a plain average…", "Two numbers per card…", OOD envelope | `t6_report.md`, `t6_results.json` |
| `t1_meansum.py` | add-a-card penalty; land-class deltas ("…wants 22 spells and 18 lands") | `t1_add_deltas.csv`, `t1_report.md` |
| `t3_ladders.py` | all of "Deck-shape preferences" except spell counts (color, creature, curve, spread, splash, fixing ladders) | `t3_ladders.csv`, `t3_report.md` |
| `t2_marginal_values.py` + `t2_analyze.py` | all of "Card preferences" (v_swap, category/MV/flying/rarity tables, regressions) | `t2_card_values.csv`, `t2_report.md` |
| `t4_synergy.py` | all of "Synergy and interactions" (dose-response ΔΔ, duplicates, removal ladder) | `t4_results.json`, `t4_report.md` |
| `t7_artifacts.py` | "The ruler" calibration; builder fingerprint; 22/23/24-spell preference; sibling-checkpoint agreement; draft-rank and blacklist joins | `t7_results.json`, `t7_report.md` |
| `forge_hints.py` | Forge `AI:RemoveDeck` + human draft-rank extraction (input to t7-D2) | `forge_hints.csv` |
| `post_hoc_slices.py` | prose numbers with no t-script: within-set correlations, set offsets, dynamic range, nonbasic-land counts, gen4-512 color distribution, card-class slices (tricks/counterspells/planeswalkers/vehicles/X/hybrid), text-length null, on/off-color add decomposition, multi-face slice, forge-best add robustness | stdout |
| `make_figures.py` | renders the document's seven figures (calibration, ablation, PC-truncation, builder scores, shape ladders, card values, synergy dose) from the staged outputs | `experiments/images/2026-08-27-scorer-*.png/.svg` |
| `label_probe.py` → `rarity_probe.py` / `human_rank_probe.py` | label-level numbers ("0.57 sd above human pick order", power-vs-toughness, rarity gradient) — run in that order (`labels_joined.csv` feeds the other two) | `labels_joined.csv`, stdout |
| `synergy_pairs.json` + `verify.py` (+ `cardtool.py`, `dump.py`) | the curated 60-triple synergy dataset used by t4, with its validator and browsing helpers | — |

## Order for a from-scratch rerun

1. `t0_landscape.py` (t7-A reads its CSV)
2. `make_text_pca.py` (t6 P1 reads the npz)
3. `forge_hints.py` (t7-D2 reads its CSV)
4. `t1_meansum.py`, `t2_marginal_values.py` → `t2_analyze.py`, `t3_ladders.py`,
   `t4_synergy.py`, `t5_ablation.py`, `t5b_det_groups.py`, `t6_mechanism.py`, `t7_artifacts.py` — any order
5. `post_hoc_slices.py`; label-level: `label_probe.py` → `rarity_probe.py`,
   `human_rank_probe.py`

Scores are checkpoint-specific: rerunning against a different scorer checkpoint
(pass `Probe(checkpoint=…)` or edit `probe_lib.SCORER_CKPT`) is the intended
way to repeat the study on a future generation.
