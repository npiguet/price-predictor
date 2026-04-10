# CLI Contract: Sealed Deck Scorer

All commands are invoked via `python -m sealed <command>`.

## encode-cards (MODIFIED — extends existing command)

Encodes card scripts to 544-dimensional `.npz` embedding files.

```
python -m sealed encode-cards [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--encoder-path` | `models/price-predictor/transformer/latest.pt` | Pretrained transformer model |
| `--vocab-path` | `models/price-predictor/transformer/vocab.txt` | Tokenizer vocabulary |
| `--cards-path` | `output/cardsfolder/` | Directory with `.txt` card scripts |
| `--clean` | off | Delete all `.npz` files before encoding |

**Change from feature 011**: Output vectors are now 544-dimensional (was 512). Cards with existing `.npz` files of any dimension are skipped unless `--clean` is used. To upgrade from 512-dim to 544-dim, run with `--clean`.

**Exit codes**: 0 = success, 1 = completed with errors, 2 = fatal error (missing files)

## train-scorer (NEW)

Train the deck scorer model on match outcome data.

```
python -m sealed train-scorer [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--outcomes-path` | `output/sealed/match-outcomes.txt` | Match outcomes file |
| `--cards-path` | `output/cardsfolder/` | Directory with `.npz` card embeddings |
| `--checkpoint-dir` | `models/sealed/scorer/` | Directory for saving checkpoints |
| `--resume` | (none) | Path to checkpoint to resume training from |
| `--epochs` | `100` | Maximum training epochs |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `1e-3` | Learning rate for scorer parameters |
| `--unfreeze-embeddings` | off | Enable embedding fine-tuning (Phase B) |
| `--embedding-lr` | `1e-5` | Learning rate for embeddings (only when unfrozen) |
| `--n-layers` | `2` | Number of self-attention layers |
| `--n-heads` | `4` | Number of attention heads |
| `--n-seeds` | `4` | Number of PMA seed vectors |
| `--d-ff` | `1088` | Feedforward dimension |
| `--mlp-hidden` | `256` | Scoring MLP hidden dimension |
| `--val-interval` | `1` | Validate every N epochs |

**Output**:
- `{checkpoint-dir}/latest.pt` — overwritten after each validation
- `{checkpoint-dir}/best.pt` — overwritten only when validation loss improves
- Console: training loss, validation loss, prediction accuracy, embedding drift (if unfrozen)

**Exit codes**: 0 = success, 2 = fatal error

## evaluate-scorer (NEW)

Evaluate the trained scorer against Forge's deck builder.

```
python -m sealed evaluate-scorer [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `models/sealed/scorer/best.pt` | Model checkpoint to evaluate |
| `--cards-path` | `output/cardsfolder/` | Directory with `.npz` card embeddings |
| `--pools` | `20` | Number of evaluation pools |
| `--workers` | `4` | Number of Java worker processes |
| `--work-dir` | (temp dir) | Directory for validation match/outcome files |

**Output**: Console summary with pools evaluated, total games, and win rate.

**Exit codes**: 0 = success, 2 = fatal error
