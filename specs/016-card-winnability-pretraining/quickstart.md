# Quickstart: End-to-end with the sealed encoder

This walkthrough produces a sealed scorer trained on the new
encoder, starting from a fresh checkout. It is the user-facing
verification path for SC-002 ("three commands end-to-end").

## Prerequisites

- Python 3.14 venv with project dependencies installed
  (`pip install -e ".[dev]" --extra-index-url
  https://download.pytorch.org/whl/cu126`).
- Java 17+ on PATH and `forge-connector` built
  (`cd forge-connector && mvn package -DskipTests`).
- Sibling `../forge` checkout built with `mvn install -DskipTests`.
- `resources/AllPrintings.json` and `resources/AllPricesToday.json`
  in place.
- Converted card corpus exists at `output/cardsfolder/` (run
  `python -m price_predictor convert` first if not).

The output directories `output/sealed/` and `models/sealed/encoder/`
are created automatically by their writers (`CardsPlayedWriter`,
`SealedEncoderStore`, `tokenizer_store.save_vocabulary`); no
checked-in scaffolding is required.

## Step 1 — Collect per-game card-play data

Run the supervisor for as long as you want to accumulate data.
Each game adds one line to `output/sealed/cards-played.txt`
alongside the parent match's line in
`output/sealed/match-outcomes.txt`.

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

A Bo7 match with 4 games adds 1 line to `match-outcomes.txt` and 4
lines to `cards-played.txt`.

## Step 2 — Build the sealed vocabulary

```powershell
python -m sealed build-vocab
# Default: output/cardsfolder/ → models/sealed/encoder/vocab.txt, ~5000 tokens
```

Verify:

```powershell
Test-Path models/sealed/encoder/vocab.txt
(Get-Content models/sealed/encoder/vocab.txt | Select-Object -First 3) -join "`n"
# Expected first three lines: [PAD], [UNK], cardname
```

## Step 3 — Train the sealed encoder

```powershell
python -m sealed train-encoder
# Defaults: 100 epochs, patience=20, shrinkage k=20, n-layers=6, n-heads=4
```

Watch the training log for per-epoch `train_loss` / `val_loss` and
the early-stop announcement. Output artifacts:

- `models/sealed/encoder/{timestamp}.pt` — best-by-val-loss
  checkpoint.
- `models/sealed/encoder/latest.pt` — copy of the same.
- `output/sealed/cards-win-rates.txt` — per-card label snapshot
  used by SC-005 verification.

Verify the encoder weights are present and contain only
encoder modules (no regression head):

```powershell
python -c "import torch; ckpt = torch.load('models/sealed/encoder/latest.pt', map_location='cpu'); print(sorted({k.split('.')[0] for k in ckpt['model_state_dict']}))"
# Expected output: ['card_encoder', 'token_encoder']
# (Notably absent: 'regression_head' / 'output_head'.)
```

## Step 4 — Re-encode cards with the new encoder

The default `--encoder-checkpoint` is now the sealed encoder, so
this command picks it up automatically:

```powershell
python -m sealed encode-cards --clean
# --clean forces a full re-encode so all .npz files come from the new encoder.
```

Verify a `.npz` shape:

```powershell
python -c "import numpy as np; arr = np.load('output/cardsfolder/A/Abandon_Hope.npz')['embedding']; print(arr.shape)"
# Expected: (2*d_model + FEATURE_COUNT,) — same shape as the price-encoder
# version produced before this feature.
```

## Step 5 — Train a scorer on the sealed encoder (Phase A)

```powershell
python -m sealed train-scorer
# Default --encoder-checkpoint is now models/sealed/encoder/latest.pt.
# The saved scorer's train_config records the new encoder path.
```

Verify the scorer checkpoint records the sealed encoder:

```powershell
python -c "import torch; ckpt = torch.load('models/sealed/scorer/latest.pt', map_location='cpu'); print(ckpt['train_config']['encoder_checkpoint'])"
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

This is SC-004's deployment metric. Compare against an old run
that used the price-encoder by passing
`--encoder-checkpoint models/price-predictor/transformer/latest.pt`
to `encode-cards` and a fresh `train-scorer` run, then re-running
`evaluate-scorer`.

## Verifying the success criteria

| SC     | Verification command                                                                                       |
|--------|------------------------------------------------------------------------------------------------------------|
| SC-001 | `(Get-Content output/sealed/match-outcomes.txt \| ForEach-Object { $_.Split(';')[7].Length } \| Measure-Object -Sum).Sum` vs `(Get-Content output/sealed/cards-played.txt).Count` |
| SC-002 | Steps 2–4 above (three commands).                                                                          |
| SC-003 | Steps 5–7 above run unmodified after Step 3.                                                               |
| SC-004 | Step 7 with and without `--encoder-checkpoint <price-encoder>` override.                                  |
| SC-005 | Run `train-encoder --shrinkage-k 0` and `train-encoder --shrinkage-k 20`, diff `output/sealed/cards-win-rates.txt`. |
| SC-006 | `python -m sealed train-scorer --encoder-checkpoint models/price-predictor/transformer/latest.pt` runs without code changes. |

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
