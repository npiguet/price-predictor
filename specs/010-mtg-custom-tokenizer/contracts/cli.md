# CLI Contract: MTG Custom Tokenizer

## New Command: `vocabulary`

```
python -m price_predictor vocabulary [OPTIONS]
```

Builds the custom MTG tokenizer vocabulary from the converted card corpus and writes `vocab.txt`.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `models/transformer/` | Directory where `vocab.txt` is written |
| `--cards-path` | `./output` | Path to the converted card corpus (directory of `.txt` files) |
| `--freq-threshold` | `5` | Minimum corpus occurrences for a word to be included as a token |

### Output (JSON, stdout)

```json
{
  "vocab_path": "models/transformer/vocab.txt",
  "vocab_size": 5847,
  "domain_token_count": 312,
  "freq_threshold_token_count": 5535,
  "coverage_pct": 98.1,
  "unk_pct": 1.9
}
```

### Exit codes
- `0` — success
- `1` — cards path not found or no `.txt` files found
- `2` — output directory could not be created

---

## Modified Command: `train transformer`

Added argument:

| Option | Default | Description |
|--------|---------|-------------|
| `--vocab-path` | `models/transformer/vocab.txt` | Path to `vocab.txt` built by `vocabulary` command |

**Error if vocab.txt missing**: exits with code 1 and message:
```
Error: Vocabulary file not found at <path>. Run 'python -m price_predictor vocabulary' first.
```

---

## Modified Command: `evaluate transformer`

Added argument:

| Option | Default | Description |
|--------|---------|-------------|
| `--vocab-path` | `models/transformer/vocab.txt` | Path to `vocab.txt` |

Same error behavior as `train transformer`.

---

## Modified Command: `predict transformer`

Added argument:

| Option | Default | Description |
|--------|---------|-------------|
| `--vocab-path` | `models/transformer/vocab.txt` | Path to `vocab.txt` |

---

## Modified Command: `serve`

Added argument:

| Option | Default | Description |
|--------|---------|-------------|
| `--vocab-path` | `models/transformer/vocab.txt` | Path to `vocab.txt` |

---

## Workflow Sequence

```
# Step 1: Convert cards (existing command, unchanged)
python -m price_predictor convert

# Step 2: Build vocabulary (new — must run before train transformer)
python -m price_predictor vocabulary --output-dir models/transformer/

# Step 3: Train transformer (vocab.txt auto-discovered at default path)
python -m price_predictor train transformer

# Step 4: Evaluate
python -m price_predictor evaluate transformer

# Step 5: Predict
python -m price_predictor predict transformer --card "name: lightning bolt\n..."

# Step 6: Serve
python -m price_predictor serve
```
