# Quickstart: Recoverability-Based Per-Step Stage 2 Loss

## Prerequisites

- Stage 1 training complete (checkpoint at `models/sealed/stage1/{set}/latest.pt`)
- Pool files generated (`output/sealed/pools/{set}/pools.txt`)
- Card embeddings encoded (`output/cardsfolder/*.npz`)

## Running Stage 2 Training

```bash
cd src
python -m sealed train --stage 2 --set RVR
```

## Reading the Batch Log

Each batch prints a line like:

```
[ep 32] batch scores: 0.750,0.823,...  mean_score=0.801  shaping=0.12  imbalance=3.4  | collect=2.31s  update=0.45s  embed=0.12s
```

- **mean_score**: Average mana score across the batch (convergence target: all > 0.90)
- **shaping**: Batch-mean discrete shaping signal. Positive = picks are generally improving mana balance. Near zero = neutral. Negative = model is actively hurting its mana distribution. Values reflect discrete signals: +1/+0.5/0/-0.5/-1 per step.
- **imbalance**: Batch-mean final imbalance (L1 distance between ideal and actual at episode end). Lower is better. Target: < 3.0 for good mana bases.

## Sampling with Mana Cost Display

```bash
python -m sealed sample --stage 2 --set RVR --n-samples 5
```

Output now shows mana costs for non-land cards:

```
--- Sample 1 [SUCCESS] picks=40 score=0.850 ---
   1. {1}{U}{U} Counterspell
   2. {2}{R} Lightning Bolt
   3. Plains
   4. {3}{B}{B} Sengir Vampire
  ...

  Mana sources (ideal → actual):
    W:  4.0 →  4
    U:  6.5 →  7
    ...
```

## What to Watch For

1. **shaping stays near 0 for many batches**: The model isn't learning from the mana signal. Check that the pool contains a mix of spells and lands.
2. **shaping is consistently negative**: The model is actively worsening its mana distribution each pick. Check if the budget signal is dominating and the model is ignoring mana.
3. **imbalance not decreasing**: Mana balance isn't converging. The model may not be picking enough lands of the right colors.
4. **entropy collapse**: If the model's action entropy drops sharply, it may be over-specializing. This is a PPO tuning issue (entropy_coef), not a shaping issue.
