# CLI Contract: Stage 2 Training

**Feature**: 013-stage2-heuristic-gate | **Date**: 2026-03-31

## Modified Commands

### `python -m sealed train`

**New/changed arguments**:

| Argument | Type | Default | Change |
|----------|------|---------|--------|
| `--stage` | int | (required) | Now accepts `2` in addition to `1` |
| `--init-from` | path | `models/sealed/stage1/{set}/latest.pt` | **New**. Stage 1 checkpoint to initialise from. Only used when `--stage 2` and `--model-path` does not exist. |
| `--model-path` | path | stage-dependent | Default changes to `models/sealed/stage2/{set}/latest.pt` when `--stage 2` |

**Stage 2 training invocation**:
```bash
python -m sealed train \
    --stage 2 \
    --set RVR \
    --pools-path output/sealed/pools/RVR/ \
    --cards-path output/cardsfolder/ \
    --model-path models/sealed/stage2/RVR/latest.pt \
    --init-from models/sealed/stage1/RVR/latest.pt \
    --batch-size 32
```

**Checkpoint priority** (when `--stage 2`):
1. If `--model-path` exists → resume from it (full checkpoint: model + optimizer + state)
2. Else if `--init-from` exists → initialise model weights only, fresh optimizer, episode_count=0
3. Else → error: "Stage 1 checkpoint not found at {init-from}"

**Stdout format** (per batch):
```
[ep {count}] batch scores: {s1},{s2},...  mean_score={mean:.3f}  | dup={n}  | collect={t}s  update={t}s  embed={t}s
```

**Completion message**:
```
Stage 2 complete: full batch scored > 0.90. Model saved to {model-path}.
```

### `python -m sealed sample`

**New/changed arguments**:

| Argument | Type | Default | Change |
|----------|------|---------|--------|
| `--stage` | int | (new, optional) | Defaults to `1`. When `2`, uses Stage 2 output format. |
| `--model-path` | path | stage-dependent | Default changes to `models/sealed/stage2/{set}/latest.pt` when `--stage 2` |

**Stage 2 sample invocation**:
```bash
python -m sealed sample \
    --stage 2 \
    --set RVR \
    --pools-path output/sealed/pools/RVR/ \
    --cards-path output/cardsfolder/ \
    --model-path models/sealed/stage2/RVR/latest.pt \
    --n-samples 10
```

**Stage 2 output format** (per sample):
```
--- Sample {n} [SUCCESS] picks=40 score={score:.3f} ---
   1. Card Name 1
   2. Card Name 2
  ...
  40. Card Name 40

  Mana sources (ideal → actual):
    W:  6.3 →  6
    U:  4.2 →  4
    B:  0.0 →  0
    R:  0.0 →  0
    G:  6.5 →  7
    C:  0.0 →  0
  Lands: 17  Score: 0.952
```

## Unchanged Commands

- `python -m sealed encode-cards` — no changes
- `python -m sealed generate-pools` — no changes
