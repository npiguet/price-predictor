# Quickstart: Stage 1 Training — Legal Pick Gate

**Feature**: 012-sealed-stage1-training
**Date**: 2026-03-28

---

## Prerequisites

Stage 0 must be complete before running Stage 1:

```bash
# 1. Encode all card scripts to embeddings (skip if already done)
python -m sealed encode-cards \
    --encoder-path models/price-predictor/transformer/latest.pt \
    --vocab-path models/price-predictor/transformer/vocab.txt \
    --cards-path output/cardsfolder/

# 2. Generate sealed pools (skip if already done)
python -m sealed generate-pools \
    --set RVR \
    --size 10000 \
    --pools-path output/sealed/pools/RVR/
```

---

## Run Stage 1 Training

```bash
python -m sealed train --stage 1
```

Uses all defaults. Equivalent to:

```bash
python -m sealed train \
    --stage 1 \
    --set RVR \
    --pools-path output/sealed/pools/RVR/ \
    --cards-path output/cardsfolder/ \
    --model-path models/sealed/stage1/latest.pt \
    --batch-size 32
```

The model directory is created automatically on first run. Training resumes from the checkpoint if one exists.

---

## Monitor Progress

Console output prints one summary line per batch:

```
[ep 32] batch runs: 1,2,1,1,3,1,...  best_run=3  mean_reward=-0.667
```

- `batch runs`: `current_run` for each episode in the batch (how many legal picks before termination)
- `best_run`: global high-water mark
- `mean_reward`: average `(current_run / best_run) × 2 - 1` for the batch

Training halts automatically when `best_run = 40` is achieved in 100 consecutive episodes.

---

## Inspect Model Behaviour

At any point during or after training:

```bash
python -m sealed sample
```

Prints 10 pick sequences from random pools, showing card names in pick order and whether each run was legal:

```
Sample 1:
  Pick  1: Skyknight Legionnaire
  Pick  2: Mountain
  ...
  Pick 40: Swamp
  Result: SUCCESS (40/40 legal picks)

Sample 2:
  Pick  1: Counterspell
  Pick  2: Counterspell       ← illegal pick
  Result: ILLEGAL PICK at step 2 (1/40 legal picks)
```

---

## Resume After Interruption

Simply re-run the same train command. The checkpoint at `models/sealed/stage1/latest.pt` is loaded automatically and training continues from where it left off (`best_run`, `episode_count`, and replay buffer are all restored). The pool iteration restarts from pool 0.

---

## Roll Back to a Prior Checkpoint

Timestamped checkpoints are saved every 1000 episodes to `models/sealed/stage1/checkpoints/`.

```bash
# Use a specific checkpoint
python -m sealed train \
    --stage 1 \
    --model-path models/sealed/stage1/checkpoints/2026-03-28T14:32:11.pt
```

---

## Run Tests

```bash
# Fast unit tests only (milliseconds)
cd src && pytest tests/unit/sealed/ -v

# All tests including integration
cd src && pytest tests/ -v
```

---

## Custom Set or Paths

```bash
python -m sealed train \
    --stage 1 \
    --set MH3 \
    --pools-path output/sealed/pools/MH3/ \
    --cards-path output/cardsfolder/ \
    --model-path models/sealed/stage1-mh3/latest.pt \
    --batch-size 64
```
