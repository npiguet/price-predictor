# Implementation Plan: Sealed Dataset Preparation

**Branch**: `011-sealed-dataset` | **Date**: 2026-03-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/011-sealed-dataset/spec.md`

## Summary

Implement two independent CLI commands under the new `python -m sealed` module:

1. **`encode-cards`** — scans a folder of converted Forge card scripts, strips the `name:` line, passes each card's text through the pretrained price predictor transformer (producing a `cat([max_pool, mean_pool])` = 2×d_model embedding), and saves each result as a `.npz` file alongside the card script. Writes are atomic (temp → rename). Already-encoded cards are skipped.

2. **`generate-pools`** — adds a `PoolGenerator` class to the forge-connector Java module that uses Forge's internal booster-generation API to produce sealed pools (6 boosters per pool) for a configurable set code, filters out basic lands, and writes pool lines to a flat text file. The Python command invokes the connector JAR and streams progress to the terminal.

The `sealed` module is a new top-level Python package alongside `price_predictor`. It imports the encoder model and tokenizer from `price_predictor` (one-way dependency). A new `encode()` method is added to `CardPriceTransformerModel` to expose the pooled text representation without the output head or meta features.

## Technical Context

**Language/Version**: Python 3.14+ (sealed module), Java 17+ (forge-connector extension)
**Primary Dependencies**: PyTorch + existing MtgTokenizer (Python); forge-game 2.0.10-SNAPSHOT (Java — already in pom.xml)
**Storage**: `.npz` embedding files in cards-path folder; `pools.txt` flat text file in output/sealed/pools/{set}/
**Testing**: pytest (Python unit + integration); JUnit 5 (Java unit)
**Target Platform**: Local workstation (same as existing project)
**Project Type**: CLI tools (two independent commands)
**Performance Goals**: No hard targets — SC-001 requires completion without errors; skipping already-encoded cards must report zero processed (SC-002)
**Constraints**: No new Python pip dependencies; Java pool generator reuses existing forge-connector pom.xml dependencies
**Scale/Scope**: ~32,000 card scripts (encode-cards); 10,000 pools default (generate-pools)

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | ✅ Pass | Unit tests for CardEncoder (strip name, tokenize, pool), PoolFileWriter, SealedPoolService; integration test for encode-cards on a small fixture folder; Java unit test for PoolGenerator |
| II. Simplicity First | ✅ Pass | No new Python dependencies; sealed module is flat with minimal layers; pool generation is a thin wrapper over Forge's existing API |
| III. Data Integrity | ✅ Pass | Atomic writes (temp + rename) for `.npz` files; basic land filtering tested with known pool; name: line stripped deterministically |
| IV. DDD & Separation | ✅ Pass | `CardEncoder` in sealed domain (pure encoding logic); `EncodeCardsUseCase`/`GeneratePoolsUseCase` in application; file I/O and subprocess calls in infrastructure |
| V. Forge Interoperability | ✅ Pass | Pool generation extends the existing forge-connector JAR; no new interop mechanism needed |
| VI. Documentation | ✅ Pass | README update required: new `python -m sealed` module, `encode-cards` and `generate-pools` commands, workflow description, `.npz` and `pools.txt` artifact formats |

**Gate result: PASS. No violations.**

## Project Structure

### Documentation (this feature)

```text
specs/011-sealed-dataset/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── cli.md           # CLI subcommand contract
└── tasks.md             # Phase 2 output (/speckit.tasks — not yet created)
```

### Source Code

```text
# New Python package
src/sealed/
├── __init__.py
├── __main__.py                          # python -m sealed entry point
├── domain/
│   ├── __init__.py
│   └── card_encoder.py                  # CardEncoder: strip name, tokenize, pool
├── application/
│   ├── __init__.py
│   ├── encode_cards.py                  # EncodeCardsUseCase
│   └── generate_pools.py                # GeneratePoolsUseCase
└── infrastructure/
    ├── __init__.py
    ├── cli.py                           # argparse CLI (encode-cards, generate-pools)
    ├── embedding_store.py               # load/save .npz files, atomic write
    └── pool_connector.py                # subprocess call to forge-connector JAR

# Modified Python file (price_predictor)
src/price_predictor/infrastructure/transformer_model.py  # Add encode() method

# New Java class (forge-connector)
forge-connector/src/main/java/com/pricepredictor/connector/
└── PoolGenerator.java                   # Sealed pool generation via Forge API
forge-connector/src/main/java/com/pricepredictor/connector/
└── PoolMain.java                        # CLI entry point for pool generation

# New test files
tests/unit/sealed/domain/test_card_encoder.py
tests/unit/sealed/application/test_encode_cards.py
tests/unit/sealed/application/test_generate_pools.py
tests/unit/sealed/infrastructure/test_embedding_store.py
tests/integration/sealed/test_encode_cards_integration.py
forge-connector/src/test/java/com/pricepredictor/connector/PoolGeneratorTest.java
```

**Structure Decision**: The `sealed` package mirrors the existing `price_predictor` domain/application/infrastructure layout. It lives at `src/sealed/` (not inside `price_predictor/`) to respect the module boundary stated in the spec.

## Design

### CardEncoder (`sealed/domain/card_encoder.py`)

Pure domain class. Takes a loaded model + tokenizer; produces a numpy array.

```python
class CardEncoder:
    def __init__(self, model: CardPriceTransformerModel, tokenizer: MtgTokenizer,
                 max_seq_len: int, device: str = "cpu") -> None: ...

    def encode(self, card_text: str) -> np.ndarray:
        # 1. Strip the name: line
        #    lines = [l for l in card_text.splitlines() if not l.startswith("name:")]
        #    text = "\n".join(lines)
        # 2. Tokenize: input_ids, attention_mask = tokenizer.encode(text, max_seq_len)
        # 3. Run model.encode(input_ids, attention_mask) → (2*d_model,) tensor
        # 4. Return as numpy float32 array
```

### encode() method on CardPriceTransformerModel

New method on the existing class in `price_predictor/infrastructure/transformer_model.py`:

```python
def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the pooled card embedding (without meta or output head).

    Returns:
        (batch_size, 2 * d_model) — cat([max_pooled, mean_pooled])
    """
    # Reuses forward() internals up to the pooling step.
    # No meta, no output head.
```

### EncodeCardsUseCase (`sealed/application/encode_cards.py`)

```python
@dataclass
class EncodeCardsResult:
    processed: int
    skipped: int
    errors: list[str]

class EncodeCardsUseCase:
    def execute(self, cards_path: Path, encoder: CardEncoder,
                store: EmbeddingStore) -> EncodeCardsResult:
        # For each .txt in cards_path (recursive):
        #   npz_path = cards_path / script_path.with_suffix(".npz")
        #   if npz_path.exists(): skipped++; continue
        #   text = script_path.read_text()
        #   embedding = encoder.encode(text)
        #   store.save(npz_path, embedding)  # atomic
        #   processed++
        #   report progress every N cards
```

### EmbeddingStore (`sealed/infrastructure/embedding_store.py`)

```python
def save(path: Path, embedding: np.ndarray) -> None:
    # Write to temp file in same dir, then os.replace() for atomic rename

def load(path: Path) -> np.ndarray:
    # np.load(path)["embedding"]
```

### PoolGenerator (`forge-connector/.../PoolGenerator.java`)

```java
public class PoolGenerator {
    public List<List<String>> generate(String setCode, int poolCount) {
        // Use Forge's booster generation API to produce poolCount pools
        // Each pool = 6 boosters from setCode
        // Filter out basic land names
        // Return list of card name lists
    }
}
```

See `research.md` for the specific Forge API classes used.

### PoolMain (`forge-connector/.../PoolMain.java`)

New CLI entry point alongside `ConvertMain`:

```java
// args: --set RVR --size 10000 --pools-path output/sealed/pools/RVR/
// Streams progress lines to stdout: "Generated N/10000 pools"
// Writes pools.txt one line per pool, semicolon-separated card names
```

### GeneratePoolsUseCase (`sealed/application/generate_pools.py`)

```python
class GeneratePoolsUseCase:
    def execute(self, set_code: str, pool_count: int, pools_path: Path,
                connector: PoolConnector) -> None:
        # Ensure pools_path exists (mkdir parents)
        # Invoke connector (subprocess to PoolMain JAR)
        # Stream stdout for progress display
```

### PoolConnector (`sealed/infrastructure/pool_connector.py`)

Mirrors the existing Java invocation pattern in `price_predictor/infrastructure/cli.py`:

```python
class PoolConnector:
    def generate(self, set_code: str, pool_count: int, pools_path: Path) -> int:
        # Build classpath (same JAR resolution as existing convert command)
        # subprocess.run(["java", "-cp", ..., "com.pricepredictor.connector.PoolMain",
        #                 "--set", set_code, "--size", str(pool_count),
        #                 "--pools-path", str(pools_path)], ...)
```

### CLI (`sealed/infrastructure/cli.py`)

```
python -m sealed encode-cards
    --encoder-path   [default: models/price-predictor/transformer/latest.pt]
    --vocab-path     [default: models/price-predictor/transformer/vocab.txt]
    --cards-path     [default: output/cardsfolder/]

python -m sealed generate-pools
    --set            [default: RVR]
    --size           [default: 10000]
    --pools-path     [default: output/sealed/pools/{set}/]
```

## Complexity Tracking

> No constitution violations — table not required.
