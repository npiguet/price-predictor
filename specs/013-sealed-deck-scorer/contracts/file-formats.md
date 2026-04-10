# File Format Contracts: Sealed Deck Scorer

## Validation Matches File

Written by the Python evaluation script, read by Java workers.

**Path**: `{work-dir}/validation-matches-{worker_id}.txt`

**Format**: One match per line.

```
deck_A_card1|card2|...|card40;pool_B_card1|card2|...|cardN
```

| Field | Separator | Description |
|-------|-----------|-------------|
| deck_A | `|` between cards, `;` after | Exactly 40 card names (scorer-built deck, including basic lands) |
| pool_B | `|` between cards | Full pool of cards (84-90 cards, no basic lands) |

**Example**:
```
Lightning Bolt|Mountain|Mountain|...;Llanowar Elves|Forest|Breeding Pool|...
```

## Validation Outcomes File

Written by Java workers, read by the Python evaluation script.

**Path**: `{input_file}-outcomes.txt` (e.g., `validation-matches-0.txt-outcomes.txt`)

**Format**: One outcome per line, corresponding to the same line number in the input file.

```
wins_A;wins_B
```

| Field | Type | Description |
|-------|------|-------------|
| wins_A | int (0-2) | Games won by deck A (scorer-built) |
| wins_B | int (0-2) | Games won by deck B (Forge-built) |

**Recovery**: Workers check outcome line count before starting. If the outcomes file already has N lines, the worker skips the first N matches in the input file (crash recovery).

## Model Checkpoint File

Saved by training, loaded by evaluation and resumed training.

**Path**: `{checkpoint-dir}/latest.pt` and `{checkpoint-dir}/best.pt`

**Format**: PyTorch `torch.save()` dict.

```python
{
    'model_state_dict': model.state_dict(),     # includes feat_mean, feat_std buffers
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': int,
    'best_val_loss': float,
    'config': {
        'd_model': 544,
        'n_layers': int,
        'n_heads': int,
        'n_seeds': int,
        'd_ff': int,
        'mlp_hidden': int,
    },
}
```
