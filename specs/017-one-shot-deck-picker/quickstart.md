# Quickstart: One-Shot Sealed Deck Picker

**Branch**: `017-one-shot-deck-picker` | **Spec**: [spec.md](spec.md)

End-to-end walkthrough: from a checked-out repository to a trained picker
participating in self-play matches.

## Prerequisites

1. Python venv set up per the project README:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126
   ```
2. A trained sealed encoder checkpoint at `models/sealed/encoder/latest.pt`
   (produced by `python -m sealed train-encoder`, spec 016).
3. The card-embedding `.npz` cache populated under `output/cardsfolder/`
   (produced by `python -m sealed encode-cards`).
4. A trained deck scorer at `models/sealed/scorer/latest.pt` (produced by
   `python -m sealed train-scorer`, specs 011 / 015). This is the picker's
   training reward source — it does **not** need to be the latest gen; any
   well-trained scorer works.
5. A pre-generated sealed pools file at, e.g.,
   `output/sealed/pools/pools.txt` (produced by `python -m sealed
   generate-pools --size 100000`). At least ~100k pools recommended per
   spec SC-002.
6. _(Optional)_ A second scorer checkpoint to serve as the auditor for the
   FR-030 cross-scorer audit, e.g., a previous-generation scorer at
   `models/sealed/scorer/gen3-best.pt`.

## Step 1 — Cold-start sanity check (manual, recommended)

Before launching a full training run, verify the picker's REINFORCE signal
is non-degenerate per spec § 3.6. This is **not** a CLI subcommand by design
(spec § 3.6 "Why this isn't a CLI feature"):

1. Compute `sigma_random_band` and `delta_forge_vs_random` from
   `output/sealed/match-outcomes.txt` using a one-off script: load the
   scorer, score every deck in the file, compute the std of scores within
   the `random` build-method tag and the mean difference between
   `forge-best` and `random` mean scores.
2. Instantiate a fresh `PickerModel(PickerConfig(embedding_dim=<cache
   width>, ...))`, sample 100 pools from `output/sealed/pools/pools.txt`,
   draw 1024 sampled decks per pool with the implemented sampler at
   `--temperature 1.0`, score them with the training scorer, compute the
   per-pool reward std across the 1024 samples.
3. Compare the typical per-pool std against `sigma_random_band`. If within
   ~3× (either direction), proceed. If much smaller (<1/10), the
   contingency plan is needed (Option A — separate spec).

A short ad-hoc Python script at the repo root is the expected form;
discard after the check.

## Step 2 — Train the picker

```powershell
python -m sealed train-picker `
  --pools-path output/sealed/pools/pools.txt `
  --scorer-checkpoint models/sealed/scorer/latest.pt `
  --auditor-scorer-checkpoint models/sealed/scorer/gen3-best.pt `
  --epochs 100 `
  --patience 10
```

Expected behavior:

- Loads the training scorer and (if configured) the auditor scorer. Both
  frozen, `.eval()`, on GPU.
- Loads the entire pools file; reserves the first 20% (`--val-fraction
  0.2`) as the validation slice.
- Builds a fresh picker (defaults: `d_model=embedding_dim`, `n_layers=4`,
  `n_heads=8`).
- Runs REINFORCE: 16 pools × 64 sampled decks per step (defaults).
- Logs one line per epoch with `policy_loss`, `entropy_loss`, `aux_loss`,
  `val_reward`, `audit_corr` (when auditor configured), and the
  distributional summaries from FR-032.
- Persists `latest.pt` every epoch (the resume point) and overwrites
  `best_{timestamp}.pt` whenever val reward sets a new best. The run-start
  timestamp keeps each run's best checkpoint distinct.
- Early-stops after 10 epochs without val-reward improvement.

Expected wall-clock per spec SC-002: a few hours on a single GPU for a
100k-pool corpus.

### Resuming an interrupted run

```powershell
python -m sealed train-picker `
  --pools-path output/sealed/pools/pools.txt `
  --resume models/sealed/picker/latest.pt
```

The resumed run restores picker weights, optimizer state, the epoch counter,
and `best_val_reward`. Architecture flags (`--n-layers`, `--n-heads`,
`--ff-dim`, `--dropout`, `--d-model`) are forbidden alongside `--resume`
— architecture is inherited from the checkpoint. Resumable hyperparameter
flags follow the resume-precedence rule: explicit CLI > checkpoint's
`train_config` > dataclass default.

### Bootstrapping from a prior picker

```powershell
python -m sealed train-picker `
  --pools-path output/sealed/pools/pools.txt `
  --picker-checkpoint models/sealed/picker/sa-warmstart.pt `
  --kl-coef 0.1
```

Loads picker weights only (no optimizer / epoch / val state). Architecture is
inherited from the checkpoint. The `--kl-coef 0.1` enables the KL penalty
against the bootstrap checkpoint's distribution — this is the contingency
plan's Option A configuration; not part of the primary plan, but the CLI
surface supports it.

## Step 3 — Generate decks with the trained picker

```powershell
python -m sealed pick-decks `
  --pools-path output/sealed/pools/eval-pools.txt `
  --picker-checkpoint models/sealed/picker/best_20260520-1430.pt `
  --label picker-gen1 `
  --output output/sealed/picker-decks.txt
```

(Substitute the actual `best_{timestamp}.pt` filename printed in the
training run's completion summary.)

Produces `output/sealed/picker-decks.txt`. One 40-card deck per input pool:
- 23 spells chosen by the picker via the deterministic walk (FR-006).
- The nonbasic lands the picker ranked above some of those spells.
- Basic lands filled by `compute_basic_lands` to total 40.

Each line: `picker-gen1;SET_CODE;Card1|...|Card40`. The `--label` value is
written verbatim and surfaces as the `method_A` / `method_B` tag in
downstream self-play matches.

### Resume a partial pick-decks run

```powershell
python -m sealed pick-decks `
  --pools-path output/sealed/pools/eval-pools.txt `
  --picker-checkpoint models/sealed/picker/best_20260520-1430.pt `
  --label picker-gen1 `
  --output output/sealed/picker-decks.txt `
  --resume
```

Counts the complete lines already in the output file, skips that many pools
from the front of `--pools-path`, appends the remaining decks. Matches
`build-decks --resume` semantics.

## Step 4 — Validate against Forge

```powershell
python -m sealed match-outcomes `
  --workers 12 `
  --side-a-decks output/sealed/picker-decks.txt `
  --best-of 7
```

Side A is sampled from the picker's deck file. Side B is rolled across the
four Forge build methods (4:3:2:1). Run for ≥ 200 matches per spec SC-004.
Inspect `output/sealed/match-outcomes.txt` after the run:

```powershell
# Approximate win-rate for the picker:
Select-String -Path output/sealed/match-outcomes.txt -Pattern ';picker-gen1;' |
  ForEach-Object { ($_.Line -split ';')[7] } |
  Group-Object { ($_ | Select-Object -First 1) }
```

A meaningfully-above-50% win rate against the forge-best opponent on at least
200 matches signals a successful training run (SC-004).

## Step 5 — Self-play continuation (optional)

The picker's deck file is a drop-in `--side-a-decks` / `--side-b-decks`
source. To grow the training corpus for the next scorer generation while
keeping the picker in the mix:

```powershell
python -m sealed match-outcomes `
  --workers 12 `
  --side-b-decks output/sealed/picker-decks.txt `
  --side-b-decks-weight 4 `
  --best-of 7
```

This rolls the picker against the four Forge methods on the right side. The
resulting `match-outcomes.txt` rows feed the next `train-scorer` run.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: ...expects {N}-wide card embeddings, but the .npz cache under <path> is {M}-wide` | Picker / scorer trained against a different encoder than the one that produced the current cache | Re-run `sealed encode-cards` with the right encoder, or load a picker that matches the cache. |
| `Error: --n-heads {H} does not divide --d-model {D}` | Bad CLI flag combination | Pick `n_heads` that divides `d_model`. |
| `Error: ...--kl-coef ... requires --picker-checkpoint` | Non-zero KL coef without a reference checkpoint | Either pass `--picker-checkpoint` or set `--kl-coef 0`. |
| `Error: --resume and --picker-checkpoint are mutually exclusive` | Both flags passed | Pick one — resume continues an existing run; picker-checkpoint bootstraps a fresh one. |
| Validation reward flat / NaN from epoch 0 | Degenerate reward landscape at random init | Re-run the cold-start sanity check (Step 1). If `sigma_random_band` swamps the per-pool std, the contingency plan (Option A — SA warmstart) is needed; out of scope for this spec. |
| `audit_corr` drifts far below the corpus baseline | Possible reward hacking | Restart from an earlier checkpoint with a higher `--entropy-coef`; if it continues, run the end-of-training Forge validation (Step 4) to confirm. |
