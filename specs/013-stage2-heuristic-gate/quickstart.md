# Quickstart: Stage 2 Training — Heuristic Gate

**Feature**: 013-stage2-heuristic-gate | **Date**: 2026-03-31

## Prerequisites

1. **Stage 1 completed**: A Stage 1 checkpoint exists (default: `models/sealed/stage1/{set}/latest.pt`).
2. **Card embeddings**: `.npz` files present under `output/cardsfolder/` (from `encode-cards`).
3. **Sealed pools**: `pools.txt` present (from `generate-pools`).
4. **Card text files**: `.txt` files present alongside `.npz` files in `output/cardsfolder/`
   (needed for mana cost and mana ability parsing).

## Train Stage 2

```bash
# Using defaults (assumes Stage 1 completed for RVR)
python -m sealed train --stage 2

# Explicit paths
python -m sealed train \
    --stage 2 \
    --set RVR \
    --init-from models/sealed/stage1/RVR/latest.pt \
    --model-path models/sealed/stage2/RVR/latest.pt
```

Training runs until all 32 episodes in a batch complete 40 picks with mana score > 0.90.

## Resume Interrupted Training

Just re-run the same command. If the Stage 2 checkpoint exists at `--model-path`, training
resumes from it automatically:

```bash
python -m sealed train --stage 2
```

## Inspect Sample Picks

```bash
python -m sealed sample --stage 2 --n-samples 5
```

Output shows each deck's 40 picks, ideal vs actual mana source distributions per color,
and the heuristic score.

## Run Tests

```bash
cd src
pytest tests/unit/sealed/domain/test_mana_scorer.py -v
pytest tests/unit/sealed/ -v
```
