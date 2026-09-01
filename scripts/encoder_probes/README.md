# Encoder-behavior probes

The probe battery behind
[`experiments/2026-08-28-encoder-preferences.md`](../../experiments/2026-08-28-encoder-preferences.md)
— label-side and inference-only analyses of the gen-4 production sealed card
encoder. Scripts load the production checkpoint and cached embeddings via
`probe_lib.py` and write their outputs to `output/encoder-probes/`
(gitignored). The session that produced the document left its exact outputs
there (`p0_report.md`, `l_report.md`, `r1r2_report.md`, `s_report.md`,
`c_report.md`, `q_report.md`, plus the CSV/JSON/pickle artifacts each script
names); rerunning a script regenerates them.

Run from the repo root with the repo venv:
`.venv\Scripts\python.exe scripts\encoder_probes\<script>`. External inputs:
the Y: corpus (`Y:\Nicolas\mtg\mtg-models-data\sealed\training-data\matches-bo1\`,
read-only), `output/cardsfolder-512/`,
`models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt`,
`scripts/scorer_probes/forge_hints.csv`, and `resources/AllPrintings.json`.

## Script → document section

| script(s) | document section it feeds | output |
|---|---|---|
| `probe_lib.py` | shared harness: checkpoint load + bit-exact re-encode, label↔text↔embedding join, seed-42 split reconstruction, fidelity/honest ridge probes, placebo edits, manifold gate, equivalence classes | — |
| `p0_build.py` | builds/caches everything the harness serves; validation tables | `p0_report.md`, `join_table.pkl`, `probes_*.pkl`, `equivalence_classes.json` |
| `p0b_card_table.py` | the per-card nameable-feature table (mana-cost breakdown, type/keyword/phrase flags, text-shape counts) joined to the labels; the feature source for the shared loaders and the 135-feature comparison in `s_r18.py` | `card_table.pkl` |
| `l_mediation.py`, `l_extraflags.py`, `l_analyze.py` | "The labels grade the builder…" — the w/m/d channel decomposition, forge-best mirror, game-length strata, set offsets, play/draw noise floor, cast_lift anatomy, blacklist channel | `l_mediation_table.pkl`, `l_report.md` |
| `r1_*.py` | "What transfers is mostly a bag of words…" — shuffle battery, line-attribution index, scope/negation flips | `r1r2_report.md` |
| `r2_*.py` | "Half of the encoder's knowledge is memorized…" — half-probe design, equivalence-class bound, placebo brittleness | `r1r2_report.md` |
| `s_r3.py`, `s_r4.py`, `s_r5.py` | "Nine heads supervise about three real axes" | `s_report.md` |
| `s_r8*.py` | played_rate as the agency axis; the five orthogonal collapse-class directions | `s_report.md` |
| `s_r17*.py` | "The encoder disagrees with its labels in the response…" — residual table, blacklist over-prediction, neighborhood smoothing | `s_report.md` |
| `s_r18*.py` | the nameability curve; encoder vs 135-feature table | `s_report.md` |
| `c1_keywords.py`, `c1b_interactions.py` | the keyword ladder, flying×size, deathtouch+trample | `c_report.md` |
| `c8_kw_vanilla.py` | the label side of the keyword figure: keyword values vs vanilla creatures on one joint baseline (supersedes `c7_labelside.py`'s keyword family, whose keyword-count control benchmarked each keyword against the other keywords' flying-heavy mix), plus the per-keyword channel split | `c8_kw_vanilla.csv` |
| `c9_kw_pairs.py` | the 120-pair keyword interaction matrix (layout-matched 2×2s with a fear control, both slot orders, double-centered); overturns c1b's deathtouch+trample bonus | `c9_kw_pairs.csv` |
| `c10_cost_sweep.py` | the mana-cost section: generic-cost sweeps on both heads (card costs and activated-ability costs), the {T} contrasts, and the same-text-different-cost label families | `c10_*.csv` |
| `c11_pip_ladder.py` | the pip-intensity ladder: real single-pip {k}{M} cards (k 2–6, {M} = one fixed color) re-encoded at the four color intensities of their own MV, both heads, per card class | `c11_pip_ladder.csv` |
| `c12_on_curve.py` | the encoder's mana curve figure: vanilla statlines P 0–5 × T 1–6 swept over MV 1–8, best-read cost per statline | `c12_on_curve.csv`, `c12_best_mv.csv` |
| `c2_statlines.py` | P−T gradient, N/N and cost sweeps, {X}, the color fee | `c_report.md` |
| `c3_removal.py` | the spell-effect ladder and aura variants | `c_report.md` |
| `c4_body_vs_spell.py` | body premium, dork-vs-rock | `c_report.md` |
| `c5_types.py` | tribal-noun scale, type swaps, taplands | `c_report.md` |
| `c6_spot.py`, `c7_labelside.py` | riders/timing spot checks; the counterfactual-vs-correlational split | `c_report.md` |
| `q1_blocks.py`, `q2_attention.py` | pool-query specialization (none) | `q_report.md` |
| `q3_decode.py` | "The embedding is a card description first…" — decodability table, {X} as one pip, integer collapse | `q_report.md` |
| `q4_token_geometry.py` | the integer collapse is unordered tokens, not learned diminishing returns — number-line projections of the integer tokens plus the label utility curve by printed power | `q4_token_geometry.csv`, `q4_power_labels.csv` |
| `make_figures.py` | renders the document's figures (channels, memorization, shuffle, keywords, spells, decode, integers) from the staged outputs | `experiments/images/2026-08-28-encoder-*.png/.svg` |

## Order for a from-scratch rerun

1. `p0_build.py` (everything imports its caches), then `p0b_card_table.py` (the feature table the shared loaders merge in)
2. `l_mediation.py` → `l_analyze.py` (CPU; label side)
3. `r1_*` / `r2_*` (GPU), `s_*` (CPU) — independent of each other
4. `c1`–`c12` (GPU), `q1`–`q3`

Probes and predictions are checkpoint-specific: repointing
`probe_lib`'s checkpoint path and rerunning `p0_build.py --force` is the
intended way to repeat the study on a future encoder generation. The
label-side `l_*` scripts depend only on the corpus and rerun unchanged.
