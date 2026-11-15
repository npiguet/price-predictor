# NAS reorganisation map (2026-11-15)

This maps every folder and loose file on the NAS (`Y:\Nicolas\mtg\mtg-models-data\`) to its place in the
**artifact-layout** tree. Nothing has been moved yet.

All paths are relative to the NAS root. Current paths use `\`, as they appear on the NAS. Targets use
`/` and start with `output/` or `models/`, so the NAS root mirrors the repository root.

Snapshot: 818 directories, 1,023,784 files, 88.2 GB. File modification times run from 2026-03-17 to
2026-10-21. 945k of the files sit in the effects gen-2 ability caches (see note C1).

## How dates and names were chosen

- **Run date.** Where a log states the run's start, that date is used: effects gen-2 `train.log` /
  `arm.log` headers, timestamped draft and sealed training logs, and the effects gen-1 `train.log` first
  time cross-checked against the first checkpoint. Where no log has it, the date is the earliest file
  modification time in the run. Those runs are marked `≈` and may be a day late.
- **Slugs.** A name that already conforms (`YYYY-MM-DD-slug`, slug of 4 words or fewer) is kept.
  Other names get the date plus the varied setting. The `genN` prefix is dropped because the path
  already carries the generation. Suffix chains (`-ext20`, `-lr0.1`, `-last`) become slugs
  (`-continued`, `-tenth-lr`) or sub-directories (`last-epoch/`).
- **Architecture tokens.** Sealed sweeps write layers, heads and pool queries as one token (`6l4h4q`)
  to stay within four words.
- **`experiments/` sub-folders disappear.** Runs sit flat under their model. Whether a run was the
  production pick or a sweep point goes in its `run.json` `status` (`keep` / `superseded`).
- **Checkpoint files keep their names** inside the run directory. A `*_training.txt` /
  `*-training.log` / `train.log` becomes the run's `console.log`. Worker logs and `*.progress.txt`
  files go to `logs/`.
- **No run on the NAS has a `run.json`.** Each run directory needs one backfilled after the move. The
  command line and commit can be recovered from the log's first line for most runs.

---

## Root

| current | target | note |
|---|---|---|
| `mtg-models-data.iml` | delete | IDE project file, not an artifact |
| `.claude\settings.local.json` | delete | Claude Code settings, not an artifact |
| `draft\training-data\` | delete | empty |

---

## price-predictor (gen1)

| current | target |
|---|---|
| `price-predictor\sklearn\20260317-141905.joblib` | `models/price-predictor/gen1/sklearn/2026-03-17-baseline/` |
| `price-predictor\transformer\latest.pt` | `models/price-predictor/gen1/transformer/2026-04-05-production/` ≈ (only evidence is the mtime, 04-05 01:03) |
| `price-predictor\transformer\vocab.txt` | `output/price-predictor/gen1/inputs/2026-03-26-vocabulary/vocab.txt` |

---

## sealed

### Generation boundaries used

gen1 = scorer on Forge-built-deck matches (before 04-26). gen2 = the 04-26 / 04-30 scorer records.
gen3 = sealed encoder and scorer (05-11 to 05-14). gen4 = the 512-wide encoder and margin-weighted
scorer (05-15 to 05-21). gen5 = picker (05-21 onward).

Data files are assigned by the `genN` tag in their names. A match collection belongs to the generation
whose decks it plays. Gen-0 (Forge's own builder) has no generation in the layout, so it goes under
gen1, whose scorer was trained on it (Q4).

### Models

**gen1 scorer.** Five runs, one per depth.

| current files (in `sealed\trained-models\gen1\scorer\`) | model run | reports run (`output/sealed/gen1/reports/…`) |
|---|---|---|
| `2layers_lr1e-05.pt.gz`, `2layers_training.txt` | `models/sealed/gen1/scorer/2026-04-23-2-layers/` ≈ | `2026-04-23-2-layers/` ← `2layers_eval.txt`, `_decks-forge.txt`, `_decks-scorer.txt`, `_pools.txt` |
| `3layers_*` | `…/2026-04-23-3-layers/` ≈ | same pattern |
| `4layers_*` | `…/2026-04-23-4-layers/` ≈ | same pattern |
| `5layers_*` | `…/2026-04-24-5-layers/` ≈ | same pattern |
| `6layers_*` | `…/2026-04-24-6-layers/` ≈ | same pattern |

**gen2 scorer** (`sealed\trained-models\gen2\scorer\`). Start times are from the logs.

| current prefix | model run `models/sealed/gen2/scorer/…` | reports run `output/sealed/gen2/reports/…` |
|---|---|---|
| `best_l6_full_training` | `2026-04-29-l6-full/` | `2026-04-29-l6-full/` ← `_decks.txt` |
| `best_phaseB_full_training` | `2026-04-29-phase-b-full/` | `2026-04-29-phase-b-full/` ← `_decks.txt` |
| `best_l6_1epoch` | `2026-04-29-l6-one-epoch/` | `2026-04-29-l6-one-epoch/` ← `_decks.txt` |
| `best_phaseB_1epoch_emblr1e-07` | `2026-04-29-phase-b-emb-lr1e-7/` | `…` ← `_decks.txt` |
| `best_phaseB_1epoch_emblr1e-06` | `2026-04-30-phase-b-emb-lr1e-6/` | (no decks file) |
| `forge_decks.txt`, `gen1_decks.txt`, `gen1_pools.txt` | **delete**: byte-identical copies of gen1 `6layers_decks-forge.txt`, `_decks-scorer.txt`, `_pools.txt` (MD5 checked) | |

**gen3 encoder** (`sealed\trained-models\gen3\encoder\`). All runs read the cleaned bo1 `cards-played`.

| current | target |
|---|---|
| `vocab.txt` | `output/sealed/gen3/inputs/2026-05-11-vocabulary/vocab.txt` |
| `full-20260511-223352-attn-6l-2h-2q-0.1mlm-128d.*` | `models/sealed/gen3/encoder/2026-05-11-full-attn-6l2h2q-128d/` |
| `full-20260511-232345-attn-6l-2h-2q-0.1mlm-256d.*` | `models/sealed/gen3/encoder/2026-05-12-full-attn-6l2h2q-256d/` |
| `experiments\20260511-HHMMSS-<cfg>.{pt,-training.log}` (16 runs) | `models/sealed/gen3/encoder/2026-05-11-<cfg>/` |

The 16 sweep `<cfg>` slugs, in current-name order: `dual-6l4h4q`, `dual-4l4h4q`, `dual-2l4h4q`,
`dual-1l4h4q`, `dual-6l8h8q`, `dual-6l16h16q`, `dual-6l4h4q-mlm0.02`, `attn-6l4h4q`, `attn-4l4h4q`,
`attn-2l4h4q`, `attn-1l4h4q`, `attn-6l4h4q-128d`, `attn-6l4h4q-64d`, `attn-6l2h2q-64d`,
`attn-6l2h2q-128d`, `attn-6l2h2q`. Names without `attn` used the default `--pool-mode dual`, so they
are tagged `dual`. `0.1mlm` is the default and is dropped.

**gen3 scorer** (`sealed\trained-models\gen3\scorer\`)

| current prefix | model run | reports run `output/sealed/gen3/reports/…` |
|---|---|---|
| `256-best_l6_h4_s4_ff1088_mlp256_lr1e-05` | `models/sealed/gen3/scorer/2026-05-13-enc256/` | `2026-05-13-enc256/` ← `-decks.txt`, `-decks-forge.txt`, `-pools.txt`, `-eval.txt`; the `…2.txt` second evaluation goes in `eval-2/` |
| `128-best_l6_h4_s4_ff1088_mlp256_lr1e-05` | `models/sealed/gen3/scorer/2026-05-13-enc128/` | `2026-05-13-enc128/` ← same four files |

**gen4 encoder** (`sealed\trained-models\gen4\encoder\`)

| current | target |
|---|---|
| `vocab.txt` | `output/sealed/gen4/inputs/2026-05-11-vocabulary/vocab.txt`. It is identical to gen3's copy (MD5); both are kept (Q5). |
| `full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.*` (production) | `models/sealed/gen4/encoder/2026-05-17-attn-6l8h8q-512d/` |
| `experiments\full-20260515-210145-attn-6l-4h-4q-…-512d.*` | `…/2026-05-15-attn-6l4h4q-512d/` |
| `experiments\full-20260515-223639-attn-6l-2h-2q-…` | `…/2026-05-15-attn-6l2h2q-512d/` |
| `experiments\full-20260516-091957-attn-6l-8h-2q-…` | `…/2026-05-16-attn-6l8h2q-512d/` |
| `experiments\full-20260516-200340-attn-6l-2h-8q-…` | `…/2026-05-16-attn-6l2h8q-512d/` |
| `experiments\full-20260516-212351-attn-6l-4h-8q-…` | `…/2026-05-16-attn-6l4h8q-512d/` |
| `experiments\full-20260516-232534-attn-6l-8h-4q-…` | `…/2026-05-16-attn-6l8h4q-512d/` |
| `experiments\full-20260517-105613-attn-6l-16h-16q-…` | `…/2026-05-17-attn-6l16h16q-512d/` |
| `experiments\full-20260517-155247-attn-8l-8h-8q-…` | `…/2026-05-17-attn-8l8h8q-512d/` |

**gen4 scorer** (`sealed\trained-models\gen4\scorer\`). All target `models/sealed/gen4/scorer/…`.

| current prefix | target run |
|---|---|
| `512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog` (production) | `2026-05-18-enc512-ff2176-mwlog/` |
| `256-best_l6_h4_s4_ff1088_mlp256_lr1e-05_mwlin` | `2026-05-18-enc256-mwlin/` |
| `experiments\256-best_…_lr1e-05` | `2026-05-17-enc256/` |
| `experiments\512-best_l6_h4_s4_ff1088_mlp256_lr1e-05` | `2026-05-17-enc512-ff1088/` |
| `experiments\512-best_l6_h4_s4_ff2176_mlp512_lr1e-05` | `2026-05-17-enc512-ff2176/` |
| `experiments\512-…ff2176…_mwlin` | `2026-05-18-enc512-ff2176-mwlin/` |
| `experiments\256-…_mwlog` | `2026-05-18-enc256-mwlog/` |

**gen5 picker** (`sealed\trained-models\gen5\picker\`). All target `models/sealed/gen5/picker/…`.

| current | target run |
|---|---|
| `best_20260524_203230-4l-8h-4top256-1e-5lr.*` (production) | `2026-05-24-4l8h-4top256-lr1e-5/` |
| `experiments\best_20260521_071908-4l-8h-no_normalization` | `2026-05-21-4l8h-no-normalization/` |
| `experiments\best_20260521_212220-4l-8h-normalization` | `2026-05-21-4l8h-normalization/` |
| `experiments\best_20260521_225348-6l-4h-0.1dropout` | `2026-05-21-6l4h-dropout0.1/` |
| `experiments\best_20260522_041629-6l-4h` | `2026-05-22-6l4h/` |
| `experiments\best_20260522_171840-6l-8h` | `2026-05-22-6l8h/` |
| `experiments\best_20260523_101152-4l-8h-16top64s` | `2026-05-23-4l8h-16top64s/` |
| `experiments\best_20260523_132702-4l-8h-16top64s-1e-4lr` | `2026-05-23-4l8h-16top64s-lr1e-4/` |
| `experiments\best_20260523_232104-4l-8h-16top128s-1e-4lr` | `2026-05-23-4l8h-16top128s-lr1e-4/` |

### Data (`sealed\training-data\`)

**Match collections → `records`.** A run date marked ≈ is the file's last write. The collection
started earlier.

| current | target |
|---|---|
| `matches-b07\match-outcomes-gen0.txt.gz` | `output/sealed/gen1/records/2026-04-23-forge-decks/` ≈ |
| `matches-b07\match-outcomes-gen1-vs-0.txt.gz` | `output/sealed/gen1/records/2026-04-26-vs-gen0/` ≈ |
| `matches-b07\match-outcomes-gen2-vs-0-1.txt.gz` | `output/sealed/gen2/records/2026-05-04-vs-gen0-gen1/` ≈ |
| `matches-b07\match-outcomes-gen3-vs-0-2.txt` + `cards-played-gen3-vs-0-2.txt` | `output/sealed/gen3/records/2026-05-14-vs-gen0-gen2/` ≈ |
| `matches-b07\match-outcomes-gen3-vs-forge-best.txt` + `cards-played-…` | `output/sealed/gen3/records/2026-05-15-vs-forge-best/` ≈ |
| `matches-b07\match-outcomes-gen4-vs-forge-best-gen3.txt` + `cards-played-…` | `output/sealed/gen4/records/2026-05-21-vs-forge-best-gen3/` ≈ |
| `matches-b07\match-outcomes-gen5-vs-gen4-forge.txt` + `cards-played-…` | `output/sealed/gen5/records/2026-05-26-vs-gen4-forge/` ≈ |
| `matches-bo1\match-outcomes-bo1-embedding.txt` + `cards-played-bo1-embedding.txt` | `output/sealed/gen3/records/2026-05-10-bo1-embedding/` ≈ (the corpus the gen-3 sweep trained on) |
| `matches-bo1\cards-played-bo1-embedding.cleaned.txt` | `output/sealed/gen3/corpus/2026-05-11-bo1-cleaned/cards-played.txt` |
| `matches-bo1\cards-win-rates.txt` | `output/sealed/gen3/reports/2026-05-17-card-win-rates/`. It was made the day the gen-3 record gained its win-rate bias analysis (commit d081cc0e). |
| `matches-bo1\*.7z` (3 files, 2026-09-15) | Compressed copies of the three `.txt` files above. Each `.7z` goes beside its `.txt` (both kept, Q6). |

**Generated pools and decks → `corpus`**

| current | target |
|---|---|
| `decks\generated-decks-gen0.txt.gz` | `output/sealed/gen1/corpus/2026-05-02-decks-forge/` ≈ |
| `decks\generated-decks-gen1.txt.gz` | `output/sealed/gen1/corpus/2026-04-30-decks/` ≈ |
| `decks\generated-decks-gen2.txt.gz` | `output/sealed/gen2/corpus/2026-05-02-decks/` ≈ |
| `pools\pools-gen2a.txt.gz`, `pools-gen2b1…`, `pools-gen2ba…` | `output/sealed/gen2/corpus/2026-05-01-pools-2a/`, `…-pools-2b1/`, `…-pools-2ba/` |
| `decks\generated-decks-gen2a.txt.gz`, `…2ba…`, `…2b1…` | `output/sealed/gen2/corpus/2026-05-01-decks-2a/`, `2026-05-01-decks-2ba/`, `2026-05-02-decks-2b1/` ≈ |
| `pools\pools-gen3-128.txt`, `pools-gen3-256.txt` | `output/sealed/gen3/corpus/2026-05-13-pools-enc128/`, `…-pools-enc256/` |
| `decks\generated-decks-gen3-128.txt`, `…-256.txt` | `output/sealed/gen3/corpus/2026-05-14-decks-enc128/`, `…-decks-enc256/` ≈ |
| `pools\pools-gen4-256.txt`, `…-512.txt` | `output/sealed/gen4/corpus/2026-05-18-pools-enc256/`, `…-pools-enc512/` |
| `decks\generated-decks-gen4-256.txt`, `…-512.txt` | `output/sealed/gen4/corpus/2026-05-19-decks-enc256/`, `…-decks-enc512/` ≈ |
| `pools\pools-gen5.txt` | `output/sealed/gen5/corpus/2026-05-25-pools/` ≈ |
| `decks\generated-decks-gen5.txt` | `output/sealed/gen5/corpus/2026-05-26-decks/` ≈ |

---

## draft

Model name `agent` for every generation. Each run's `-training.log` becomes its `console.log`. Its
yardstick, draft-games and per-checkpoint scoring outputs go to `output/draft/<gen>/reports/<same run
name>/`.

**gen1** (`draft\trained-models\gen1\`). Imitation learning on `drafts-greedy.jsonl`.

| current | target |
|---|---|
| `l6_lr3e-4_decay_20260604_080249.*` (production) | `models/draft/gen1/agent/2026-06-04-l6-lr-decay/` |
| `experiments\l4_lr3e-4_20260603_121316.*` | `models/draft/gen1/agent/2026-06-03-l4-lr3e-4/` |
| `experiments\l4_lr3e-5_20260603_211958.*` | `…/2026-06-03-l4-lr3e-5/` (resumes the previous run: `continues`) |
| `experiments\l4_lr3e-6_20260603_234318.*` | `…/2026-06-03-l4-lr3e-6/` (`continues` the lr3e-5 run) |
| `experiments\l8_lr3e-4_decay_20260604_232956.*` | `…/2026-06-04-l8-lr-decay/` |
| `drafts-gen1-argmax.jsonl` | `output/draft/gen1/records/2026-06-10-argmax-live-play/drafts.jsonl` ≈ |
| `drafts-gen1-argmax-deck-analysis.log` | `output/draft/gen1/reports/2026-06-10-argmax-deck-analysis/console.log` |
| `drafts-gen1-sample-t1.0.jsonl` | `output/draft/gen2/records/2026-06-14-gen1-sample-t1/drafts.jsonl` ≈. All three gen-2 RL runs train on this file (their logs say so), so it is gen-2's corpus even though gen-1 drafted it. |
| `drafts-gen1-sample-t1.0-deck-analysis.log` | `output/draft/gen2/reports/2026-06-14-gen1-sample-t1-analysis/console.log` |

**gen2** (`draft\trained-models\gen2\experiments\`). Offline RL.

| current | target |
|---|---|
All three share `--lr 3e-5 --kl-coef 0.1 --gae-lambda 0.95 --rollout-temperature 1.0` on the gen-1
sample corpus. The slugs name what changed.

| current | flags that differ | target |
|---|---|---|
| `lr3e-5_kl0.1_ent0.1_val1.0_gae0.95_20260614_234040.*` | `--entropy-coef 0.1` | `models/draft/gen2/agent/2026-06-14-entropy0.1/` |
| `20260615_074224.{pt,-training.log}` | `--entropy-coef 0.01 --value-weight 0.1` | `models/draft/gen2/agent/2026-06-15-entropy0.01-value0.1/` |
| `20260615_190126.*` | as above, plus `--entropy-decay-after 1000000` | `models/draft/gen2/agent/2026-06-15-entropy0.01-value0.1-decay/` |
| `*-yardstick.log` | | `output/draft/gen2/reports/<run>/yardstick.log` |

**gen3** (`draft\trained-models\gen3\`). Online GRPO.

| current | target |
|---|---|
| `temperature-on-all-agents\lr1e-4_t2_20260805_183818.*` | `models/draft/gen3/agent/2026-08-05-t2-all-lr1e-4/` |
| `temperature-on-all-agents\lr1e-6_t2_20260805_201705.*` | `…/2026-08-05-t2-all-lr1e-6/` |
| `temperature-on-all-agents\lr1e-5_t2_20260805_221050.*` (promoted gen-3, the gen-4 anchor) | `…/2026-08-05-t2-all-lr1e-5/` |
| `temperature-on-all-agents\lr1e-5_t3_20260805_230506.*` | `…/2026-08-05-t3-all-lr1e-5/` |
| `lr1e-5_t2_20260806_112400.*` | `…/2026-08-06-t2-learner-lr1e-5/` |
| `lr1e-5_t3_20260806_144756.*` | `…/2026-08-06-t3-learner-lr1e-5/` |

The 08-05 runs are the gen-3 record's "field at T": one temperature for every ML agent, and their logs
print `T=2.0`. The 08-06 runs are "field at argmax": temperature on the learner only, and their logs
print `T=2.0 (learner samples; frozen play argmax)`. The `-learner` slug matches gen-4's
`t3learner` naming.
| each `-yardstick.log`, `-yardstick-drafts.jsonl` | `output/draft/gen3/reports/<run>/` |

**gen4** (`draft\trained-models\gen4\`). Start times are from the logs.

| current prefix | model run `models/draft/gen4/agent/…` |
|---|---|
| `lr1e-5_t2all_nodecay` | `2026-08-07-t2-all-no-decay/` |
| `lr1e-5_t2all_decay0.3` | `2026-08-08-t2-all-decay0.3/` |
| `lr1e-5_t3all_decay0.3` | `2026-08-08-t3-all-decay0.3/` |
| `lr1e-5_t3learner_t2field_decay0.3` | `2026-08-09-t3-learner-t2-field/` |

For each of the four runs:

- `<prefix>.pt` and `-training.log` go to the model run.
- `<prefix>-drafts.jsonl`, the online run's self-play drafts, goes to `output/draft/gen4/records/<run>/drafts.jsonl`.
- `-yardstick-v-forge*`, `-yardstick-v-gen3*` and `-draft-games.{txt,log}` go to `output/draft/gen4/reports/<run>/`.

The cross-run analyses become runs of their own:

| current | target |
|---|---|
| `picker-built\*.jsonl` (8 files) + top-level `lr1e-5_t2all_decay0.3-prefix-deck-scores.jsonl` | `output/draft/gen4/reports/2026-08-25-picker-prefix-scores/`. The top-level file is byte-identical to `picker-built\lr1e-5_t2all_decay0.3-prefix-deck-scores.jsonl` (MD5), so one copy is deleted. |
| `rebuild-greedy.log` + the 8 `*-greedy-prefix-deck-scores.jsonl` | `output/draft/gen4/reports/2026-08-25-greedy-prefix-scores/` (log → `console.log`) |

---

## effects

### gen1

**Records** (`effects\training-data\gen1\records-gen1\`). Per-supervisor `<uuid>.N.log` and
`<uuid>.progress.txt` files go to `logs/`. Shards, `match-outcomes.txt`, `cards-played.txt`,
`holdout-cards.txt` and `variant-decks.txt` stay at the run root.

| current | target |
|---|---|
| `full-strength\` | `output/effects/gen1/records/2026-09-11-full-strength/`. Its two `.bak` files were deleted 2026-11-15 (Q9). |
| `depleted\` | `output/effects/gen1/records/2026-09-13-depleted/` (4 supervisor restarts, one run) |
| `variants\` | `output/effects/gen1/records/2026-09-15-variants/` |

**Inputs, corpora, models**

| current | target |
|---|---|
| `training-data\gen1\variant-scripts-gen1\` | `output/effects/gen1/inputs/2026-09-15-variant-scripts/` |
| `trained-models\gen2\vocab.txt` (mtime **2026-09-09**, gen-1 era; differs from the price vocab) | `output/effects/gen1/inputs/2026-09-09-vocabulary/vocab.txt` |
| `training-data\gen1\gen1-keyword-definitions.json` (2026-10-09) | `output/effects/gen1/inputs/2026-10-09-keyword-definitions/` |
| `trained-models\gen1\gen1-vocab-script.txt` (2026-10-09) | `output/effects/gen1/inputs/2026-10-09-vocab-script/` |
| `trained-models\gen1\vocab-script-gen1-sidecars.txt` (2026-10-08) | `output/effects/gen1/inputs/2026-10-08-vocab-script-sidecars/` |

These three were built during gen-2 but are gen-1's only keyword definitions and vocab scripts, so
they move to gen1 rather than being deleted (Q10).
| `training-data\gen1\corpus-20260917-pre-textless\` | `output/effects/gen1/corpus/2026-09-17-pre-textless/` (superseded) |
| `training-data\gen1\corpus\` | `output/effects/gen1/corpus/2026-09-17-textless/`, plus `trained-models\gen1\effect-model\09.17c-build-corpus.log` → `console.log` and `…build-corpus.err.log` → `logs/` |
| `trained-models\gen1\2026-09-17-full-textless-corpus\` | `models/effects/gen1/effect-model/2026-09-17-full-textless-corpus/` (name kept; it conforms) |
| `trained-models\gen1\effect-model\09.17c.log`, `09.17d.log`, `09.17e.log` | the same run's `logs/`. These are three short attempts before the run that started 19:03. |
| `trained-models\gen1\effect-model\` (rest) | **delete**: its 28 checkpoints, `latest.pt` and `09.17f.log` duplicate the run above. `latest.pt` and the log were checked by MD5; the checkpoints match by name, size and mtime. |
| `trained-models\gen1\2026-09-17-identity\` | `models/effects/gen1/effect-model-identity/2026-09-18-baseline/` (started 09-18 09:49) |
| `trained-models\gen1\2026-09-17-taxonomy\` | `models/effects/gen1/effect-model-taxonomy/2026-09-18-baseline/` (09-18 15:20) |
| `trained-models\gen1\2026-09-17-state-only\` | `models/effects/gen1/effect-model-state-only/2026-09-19-baseline/` (09-19 00:07) |
| `trained-models\gen1\2026-09-17-no-state\` | `models/effects/gen1/effect-model-no-state/2026-09-19-baseline/` (09-19 09:50) |

The four baselines are named `2026-09-17-*` on the NAS after their corpus's date. Their logs put
their starts on 09-18 and 09-19.

### gen2

**Inputs and records** (`effects\training-data\gen2\`)

| current | target |
|---|---|
| `keyword-definitions.json` | `output/effects/gen2/inputs/2026-10-09-keyword-definitions/` |
| `holdout-cards.txt` | `output/effects/gen2/inputs/2026-10-09-holdout-cards/`. It is identical to the copies inside both `depleted\` dirs (MD5), which stay with their runs. |
| `trained-models\gen2\vocab-script.txt` (the `--vocab-path` every gen-2 run used) | `output/effects/gen2/inputs/2026-10-09-vocab-script/` |
| `variant-scripts\` | `output/effects/gen2/inputs/2026-10-12-variant-scripts/` |
| `records-gen2-tiers123\depleted\` | `output/effects/gen2/records/2026-10-09-depleted-tiers123/` (superseded: collected without snapshot tier 4) |
| `records-gen2-tiers123\full-strength\` | `output/effects/gen2/records/2026-10-09-full-strength-tiers123/` (superseded) |
| `records-gen2\depleted\` | `output/effects/gen2/records/2026-10-11-depleted/` (6 supervisor ids; the coverage passes appended here) |
| `records-gen2\full-strength\` | `output/effects/gen2/records/2026-10-12-full-strength/` |
| `records-gen2\held-out-coverage\` | `output/effects/gen2/records/2026-10-12-held-out-coverage/` |
| `records-gen2\variants\` | `output/effects/gen2/records/2026-10-12-variants/` |
| `corpus-gen2\` + `build-corpus-gen2.log` | `output/effects/gen2/corpus/2026-10-13-stage-4/` (log → `console.log`; `repairs.json` stays inside) |

**Training runs.** Starts are from `train.log` / `arm.log` headers. Names follow the gen-2 record's
run table.

| current (`effects\trained-models\gen2\…`) | new run name | record name | status |
|---|---|---|---|
| `2026-10-gen2-e64-earlier-inputs` | `2026-10-13-e64-earlier-inputs` | `e64` (earlier inputs) | superseded |
| `2026-10-gen2-e64-interrupted` | **deleted** 2026-11-15 (Q3) | — | — |
| `2026-10-gen2-e64` | `2026-10-14-e64` | `e64` | keep |
| `2026-10-gen2-e128` | `2026-10-15-e128` | `e128` | keep |
| `2026-10-gen2-e64-lethality` | `2026-10-16-e64-lethality` | `e64-lethality` | keep |
| `2026-10-gen2-e128-lethality` | `2026-10-16-e128-lethality` | `e128-lethality` | keep |
| `2026-10-gen2-e64-lethality-ext20` | `2026-10-17-e64-lethality-continued` | run A64 | keep |
| `2026-10-gen2-e64-lethality-lr0.1` | `2026-10-17-e64-lethality-tenth-lr` | run B64 | keep |
| `2026-10-gen2-e128-lethality-ext20` | `2026-10-18-e128-lethality-continued` | run A128 | keep |
| `2026-10-gen2-e128-lethality-lr0.1` | `2026-10-19-e128-lethality-tenth-lr` | run B128 | keep |
| `2026-10-gen2-e256-lethality` | `2026-10-19-e256-lethality` | `e256-lethality` | keep |
| `2026-10-gen2-e256-lethality-ext20` | `2026-10-20-e256-lethality-continued` | run A256 | keep |
| `2026-10-gen2-e256-lethality-lr0.1` | `2026-10-20-e256-lethality-tenth-lr` | run B256 | keep |

Each training run directory is split three ways, using `<run>` = the new name.

| contents | target |
|---|---|
| checkpoints (`*.pt`, `latest.pt`, `resume.pt`), `epochs\`, `train.log`, `arm.log`, `started-from.txt`; the matching `<x>-console.out` from the gen2 root → `console.log`, `<x>-console.err` → `logs/` | `models/effects/gen2/effect-model/<run>/` |
| `abilities\` | **deleted** (decided 2026-11-15, see C1) |
| `encode.log`; `reports\embedding-probes-*\cache\` | `output/effects/gen2/cache/<run>/kept-epoch/` |
| `evaluate.log`, `probes-*.log`, `knowledge-probes*.log`, `reports\embedding-probes-*\` (minus `cache\`), `reports\knowledge-probes-*\`, `reports\board-run\…` (e64 only → `knowledge-probes-board/`), and the later rule-check outputs (`combat-grid*`, `evasion-grid`, `colour-pips`, `target-restrictions`, `zone-effects`, `ladder-check-amount-alone`, `boundary-by-epoch`, `grid-*-epoch`, each `.txt` + `.err`) | `output/effects/gen2/reports/<run>/kept-epoch/` |

**Probe-only directories** each probed another epoch of a training run (their `source.txt` says
which).

| current | target |
|---|---|
| `…-e64-lethality-ext20-last` | `reports/2026-10-17-e64-lethality-continued/last-epoch/` |
| `…-e64-lethality-lr0.1-last` | `…/2026-10-17-e64-lethality-tenth-lr/last-epoch/` |
| `…-e128-lethality-ext20-last` | `…/2026-10-18-e128-lethality-continued/last-epoch/` |
| `…-e128-lethality-lr0.1-last` | `…/2026-10-19-e128-lethality-tenth-lr/last-epoch/` |
| `…-e256-lethality-ext20-last` | `…/2026-10-20-e256-lethality-continued/last-epoch/` |
| `…-e256-lethality-lr0.1-last` | `…/2026-10-20-e256-lethality-tenth-lr/last-epoch/` |
| `…-e256-lethality-lr0.1-best` (epoch 55) | `…/2026-10-20-e256-lethality-tenth-lr/epoch-55/` |

How the files in these directories split:

- `abilities\` is deleted (C1). The reports, logs, `arm.log`, `source.txt` and the matching root console files go to `reports/`; `encode.log` and the embedding-probes `cache\` go to the matching `cache/<run>/<epoch>/`.
- Their `latest.pt` copies **are deleted**. Each one duplicates an `epochs\` file of its training run; two were checked by MD5.
- B256's own `latest.pt` is a copy of A256's epoch 53, its kept checkpoint. It stays, and its `run.json` records that.

**Loose files at the gen2 root.** Besides the console files placed above:

| current | target |
|---|---|
| `e64-retrain-console.*` | `models/effects/gen2/effect-model/2026-10-14-e64/` (console) |
| `e64-probes-console.*` | `output/effects/gen2/reports/2026-10-14-e64/kept-epoch/logs/` |
| `e256-chain.log`, `b256-best-console.*`, `b256-best-probes.log` | `output/effects/gen2/reports/2026-10-20-e256-lethality-tenth-lr/logs/` and `epoch-55/` |

---

## Notes

- **C1. Ability caches: deleted, not moved (decided 2026-11-15).** The 19 `abilities\` trees held 945k
  of the NAS's 1.02M files (~1.1 GB) and were `cache`, rebuildable from each run's checkpoint with
  `encode-abilities`. Each tree was checked to hold only `.npz` files before deletion. The
  `embedding-probes-*\cache\` folders (~2 GB, 40 files) are the same kind of file but few enough to move;
  they are kept unless you decide otherwise.
- **C2. Missing from the NAS.**
  - **Effects gen-3 data and models.** My memory note says gen-3's first records, pilots and models were archived to the NAS on 2026-11-12. No file on the NAS is newer than 2026-10-21, and `Y:\Nicolas\mtg\` holds only this folder and a `.dlens` backup.
  - **Converted card trees.** There is no `converted/` for any module or generation.
  - **Draft gen-1's imitation corpus.** `drafts-greedy.jsonl`, which every gen-1 log trains on, is not on the NAS.
  - **Sealed scorer inputs.** The concatenated `match-outcomes-all.txt` the gen-2 to gen-4 scorers read, and the picker's `pools-picker.txt`, are not on the NAS.
- **C3. Paths cited elsewhere.** Experiment records and the gen-4 encoder logs cite current NAS
  paths, such as `gen3/temperature-on-all-agents/lr1e-5_t2_20260805_221050.pt` and
  `…\matches-bo1\cards-played-bo1-embedding.cleaned.txt`. Those citations go stale after the move and
  would need a follow-up edit.

## Open questions

- **Q1.** ~~The ability caches: delete or move?~~ Delete (decided 2026-11-15, C1).
- **Q2.** Decided 2026-11-15: reports are grouped one directory per training run, with `kept-epoch/`,
  `last-epoch/` and `epoch-NN/` inside it, not one run per report command.
- **Q3–Q10.** Decided 2026-11-15: `e64-interrupted` deleted; gen-0 stays under gen1; both vocab
  copies and both bo1 forms kept; draft gen-2 and gen-3 slugs read from the logs; `.bak` files
  deleted; gen-1 keyword definitions and vocab scripts move to gen1.
