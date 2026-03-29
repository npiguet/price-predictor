# CLI Contract: train and sample subcommands

**Module**: `sealed` (`python -m sealed`)
**Feature**: 012-sealed-stage1-training
**Date**: 2026-03-28

---

## train

Launches a PPO training loop for the specified training stage.

```
python -m sealed train
    --stage      INT     Training stage to run (required). Currently only 1 is supported.
    --set        STR     MTG set code for the pool dataset. Default: RVR
    --pools-path PATH    Directory containing pools.txt. Default: output/sealed/pools/{set}/
    --cards-path PATH    Directory containing .npz embedding files. Default: output/cardsfolder/
    --model-path PATH    Path to latest.pt checkpoint (load + save target). Default: models/sealed/stage1/latest.pt
    --batch-size INT     Number of episodes collected per PPO update. Default: 32
```

### Startup validation (before training begins)

| Condition | Behavior |
|-----------|----------|
| `pools.txt` missing or empty | Exit code 2, error to stderr: `"Error: pools.txt not found or empty: {path}"` |
| Any card in pools.txt has no `.npz` file | Exit code 2, error to stderr: `"Error: missing embedding for card: {name}"` |
| `model-path` directory does not exist | Created automatically (including parents) |
| Checkpoint exists at `model-path` | Resume training; load model, optimizer, and training state |
| No checkpoint at `model-path` | Initialize model and training state from scratch |

### Console output during training

One line per batch to stdout:

```
[ep {episode_count}] batch runs: {r0},{r1},...  best_run={best_run}  mean_reward={mean:.3f}
```

Example:
```
[ep 64] batch runs: 1,3,2,1,5,2,1,4,2,1,2,3,1,2,4,2,1,3,2,1,1,4,2,3,1,2,1,3,2,1,2,4,1,2,3,1,2,1,3,2  best_run=5  mean_reward=-0.712
```

KL divergence warning (when any episode in the batch exceeds 1.5 nats):
```
[warn] KL divergence {kl:.2f} nats for episode at buffer index {i} — policy has drifted
```

### Completion

When `consecutive_successes >= 100`:
```
Stage 1 complete: 100 consecutive episodes with 40 legal picks. Model saved to {model_path}.
```
Exit code 0.

### Checkpointing

| Event | Action |
|-------|--------|
| End of each batch | Overwrite `{model_path}` (atomic: write temp then rename) |
| Every 1000 episodes | Write `{model_path.parent}/checkpoints/{timestamp}.pt` |

---

## sample

Generates human-readable pick sequences from a trained model checkpoint.

```
python -m sealed sample
    --set        STR     MTG set code for the pool dataset. Default: RVR
    --pools-path PATH    Directory containing pools.txt. Default: output/sealed/pools/{set}/
    --cards-path PATH    Directory containing .npz embedding files. Default: output/cardsfolder/
    --model-path PATH    Path to model checkpoint to load. Default: models/sealed/stage1/latest.pt
    --n-samples  INT     Number of pick sequences to generate. Default: 10
```

### Startup validation

| Condition | Behavior |
|-----------|----------|
| Checkpoint missing at `model-path` | Exit code 2, error to stderr |
| `pools.txt` missing or empty | Exit code 2, error to stderr |
| Any card has no `.npz` file | Exit code 2, error to stderr |

### Output format

One block per sample, separated by a blank line:

```
Sample {n}:
  Pick  1: Lightning Bolt
  Pick  2: Goblin Guide
  ...
  Pick 40: Plains
  Result: SUCCESS (40/40 legal picks)

Sample {n+1}:
  Pick  1: Counterspell
  Pick  2: Counterspell       ← illegal pick
  Result: ILLEGAL PICK at step 2 (1/40 legal picks)
```

Exit code 0 (sampling is always non-destructive).
