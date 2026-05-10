# Quickstart: End-to-end with the multi-head sealed encoder

This walkthrough produces a sealed scorer trained on the new
multi-head + MLM sealed encoder, starting from a fresh checkout. It
verifies SC-002 ("three commands end-to-end") for the v2 schema.

## Prerequisites

- Python 3.14 venv with project dependencies installed
  (`pip install -e ".[dev]" --extra-index-url
  https://download.pytorch.org/whl/cu126`).
- Java 17+ on PATH and `forge-connector` built
  (`cd forge-connector && mvn package -DskipTests`).
- Sibling `../forge` checkout built with `mvn install -DskipTests`.
- `resources/AllPrintings.json` and `resources/AllPricesToday.json` in
  place.
- Converted card corpus exists at `output/cardsfolder/` (run
  `python -m price_predictor convert` first if not).

The output directories `output/sealed/` and `models/sealed/encoder/` are
created automatically by their writers; no checked-in scaffolding is
required.

## Step 1 — Collect per-game card-play data

Run the supervisor for as long as you want to accumulate data. Each game
adds one line to `output/sealed/cards-played.txt` alongside the parent
match's line in `output/sealed/match-outcomes.txt`. Basic lands are
excluded at write time by the Java worker (FR-004a).

```powershell
python -m sealed match-outcomes
# Ctrl-C when satisfied with corpus size.
```

Sanity check after stopping:

```powershell
# match-outcomes line count vs cards-played line count
(Get-Content output/sealed/match-outcomes.txt).Count
(Get-Content output/sealed/cards-played.txt).Count
# The cards-played count should equal the sum of game counts in match-outcomes.
```

A Bo7 match with 4 games adds 1 line to `match-outcomes.txt` and 4 lines
to `cards-played.txt`.

## Step 2 — Build the sealed vocabulary (now seeds `[MASK]`)

```powershell
python -m sealed build-vocab
# Default: output/cardsfolder/ → models/sealed/encoder/vocab.txt, ~5000 tokens
```

Verify:

```powershell
Test-Path models/sealed/encoder/vocab.txt
(Get-Content models/sealed/encoder/vocab.txt | Select-Object -First 4) -join "`n"
# Expected first four lines: [PAD], [UNK], cardname, [MASK]
```

If the vocab predates this iteration and lacks `[MASK]`, `train-encoder`
will refuse to start with exit code 2 — re-run `build-vocab` to refresh.

## Step 3 — Train the sealed encoder (multi-head + MLM)

```powershell
python -m sealed train-encoder
# Defaults: 100 epochs, patience=20, shrinkage k=20, n-layers=6, n-heads=4,
#           mlm-weight=0.1, mlm-mask-prob=0.15
```

Watch the training log for per-epoch `train_loss` / `val_loss`. The new
log lines break out the regression and MLM components:

```
[2026-05-10 14:22:01] Epoch 1/100  train_loss=0.4721 (reg=0.4205, mlm=5.16)  val_loss=0.4519 (reg=0.4002, mlm=5.17)  best=*
```

Output artifacts:

- `models/sealed/encoder/{timestamp}.pt` — best-by-full-validation-loss
  checkpoint (regression heads and MLM head dropped at save time).
- `models/sealed/encoder/latest.pt` — copy of the same.
- `output/sealed/cards-win-rates.txt` — 23-column per-card label
  snapshot used by SC-005 verification.

Verify the encoder weights are present and contain only encoder modules
(no regression heads, no MLM head):

```powershell
python -c "import torch; ckpt = torch.load('models/sealed/encoder/latest.pt', map_location='cpu', weights_only=False); print(sorted({k.split('.')[0] for k in ckpt['model_state_dict']}))"
# Expected output: ['card_encoder', 'token_encoder']
# (Notably absent: 'regression_heads', 'mlm_head'.)
```

Verify the new `cards-win-rates.txt` schema:

```powershell
(Get-Content output/sealed/cards-win-rates.txt | Select-Object -First 1)
# Expected first line (header): card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;...
(Get-Content output/sealed/cards-win-rates.txt | Select-Object -Skip 1 -First 1).Split(';').Count
# Expected: 23
```

## Step 4 — Re-encode cards with the new encoder

The default `--encoder-checkpoint` is the sealed encoder, so this command
picks it up automatically:

```powershell
python -m sealed encode-cards --clean
# --clean forces a full re-encode so all .npz files come from the new encoder.
```

Verify a `.npz` shape:

```powershell
python -c "import numpy as np; arr = np.load('output/cardsfolder/A/Abandon_Hope.npz')['embedding']; print(arr.shape)"
# Expected: (2*d_model + FEATURE_COUNT,) — same shape as before; the new
# heads are training-only and bypassed by inference.
```

## Step 5 — Train a scorer on the sealed encoder (Phase A)

```powershell
python -m sealed train-scorer
# Default --encoder-checkpoint is models/sealed/encoder/latest.pt.
```

Verify the scorer checkpoint records the sealed encoder:

```powershell
python -c "import torch; ckpt = torch.load('models/sealed/scorer/latest.pt', map_location='cpu', weights_only=False); print(ckpt['train_config']['encoder_checkpoint'])"
# Expected: models/sealed/encoder/latest.pt
```

## Step 6 — Optional: Phase B fine-tune

Phase B fine-tunes the encoder jointly with the scorer:

```powershell
python -m sealed train-scorer `
    --scorer-checkpoint models/sealed/scorer/latest.pt `
    --embedding-lr 1e-5
# --encoder-checkpoint default is the sealed encoder.
```

## Step 7 — Evaluate

```powershell
python -m sealed evaluate-scorer --set BLB
# Win rate vs forge-best on Bloomburrow pools.
```

This is SC-004's deployment metric. Compare against an old run that used
the price-encoder by passing `--encoder-checkpoint
models/price-predictor/transformer/latest.pt` to `encode-cards` and a
fresh `train-scorer` run, then re-running `evaluate-scorer`.

## Verifying the success criteria

| SC | Verification command |
|---|---|
| SC-001 | `(Get-Content output/sealed/match-outcomes.txt \| ForEach-Object { $_.Split(';')[7].Length } \| Measure-Object -Sum).Sum` vs `(Get-Content output/sealed/cards-played.txt).Count` |
| SC-002 | Steps 2–4 above (three commands). |
| SC-003 | Steps 5–7 above run unmodified after Step 3. |
| SC-004 | Step 7 with and without `--encoder-checkpoint <price-encoder>` override. |
| SC-005 | Run `train-encoder --shrinkage-k 0` and `train-encoder --shrinkage-k 20`, diff `output/sealed/cards-win-rates.txt`. Low-observation rows should differ visibly across all 9 head columns; high-observation rows should differ by only a few thousandths. |
| SC-006 | `python -m sealed train-scorer --encoder-checkpoint models/price-predictor/transformer/latest.pt` runs without code changes. |

## Inspecting the per-card label map

`cards-win-rates.txt` is human-readable. Quick examples (PowerShell):

```powershell
# Top-10 highest-shrunk-score-play cards (skipping the header row)
Get-Content output/sealed/cards-win-rates.txt | Select-Object -Skip 1 -First 10 | ForEach-Object { $_.Split(';')[0,6] -join "`t" }

# Cards with no @play observations (empty raw_score_play column 6)
Get-Content output/sealed/cards-win-rates.txt | Select-Object -Skip 1 | Where-Object { $_.Split(';')[5] -eq "" } | Measure-Object | Select-Object -ExpandProperty Count
```

Or via Python:

```python
import csv
with open("output/sealed/cards-win-rates.txt", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    rows = list(reader)
print(rows[0]["card_name"], rows[0]["shrunk_score_play"], rows[0]["shrunk_color_lift_W"])
```

## Reverting to the price-encoder

One-step revert (per SC-006):

```powershell
python -m sealed encode-cards --clean `
    --encoder-checkpoint models/price-predictor/transformer/latest.pt `
    --vocab-path models/price-predictor/transformer/vocab.txt
python -m sealed train-scorer `
    --encoder-checkpoint models/price-predictor/transformer/latest.pt
```

No code changes required.
