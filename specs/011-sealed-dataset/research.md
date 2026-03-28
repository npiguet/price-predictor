# Research: Sealed Dataset Preparation

## 1. Forge Booster Generation API

**Decision**: Use `FModel.getMagicDb().getBoosters().get(setCode)` + `UnOpenedProduct.get()` to open individual boosters.

**Rationale**: This is the same API used by `SealedCardPoolGenerator` in `forge-gui`. It respects set-specific booster templates (slot distributions, rarities, foil rules) and returns `List<PaperCard>` directly. `FModel` is already initialized by the existing `ForgeEnvironmentInitializer`.

**Alternatives considered**:
- `SealedCardPoolGenerator` directly: too coupled to the GUI layer (requires `SGuiChoose`, `FModel.getDecks()`, etc.). Not suitable for headless batch use.
- `BoosterGenerator` directly: lower-level utility; `UnOpenedProduct` already wraps it correctly.

**Key classes and usage**:

```java
// forge-core: forge.item.generation
class UnOpenedProduct implements IUnOpenedProduct {
    UnOpenedProduct(SealedTemplate template)
    List<PaperCard> get()   // opens one booster, returns its cards
}

// forge-core: forge.model
class FModel {
    static MagicDb getMagicDb()
}

class MagicDb {
    IStorage<SealedTemplate> getBoosters()   // keyed by set code
}

// forge-core: forge.item
class PaperCard {
    String getName()                          // card name for pool line
    CardRules getRules()
}

class CardRules {
    ICardFace getMainPart()
}

interface ICardFace {
    CardType getType()
}

class CardType {
    boolean isBasicLand()                    // use this to filter basic lands
}
```

**Pool generation pattern**:

```java
ForgeEnvironmentInitializer.initialize();

SealedTemplate boosterTemplate = FModel.getMagicDb().getBoosters().get(setCode);
if (boosterTemplate == null) {
    throw new IllegalArgumentException("Unknown or unsupported set code: " + setCode);
}

List<String> poolCards = new ArrayList<>();
for (int b = 0; b < 6; b++) {
    List<PaperCard> boosterCards = new UnOpenedProduct(boosterTemplate).get();
    for (PaperCard card : boosterCards) {
        if (!card.getRules().getMainPart().getType().isBasicLand()) {
            poolCards.add(card.getName());
        }
    }
}
// poolCards is one pool; join with ";" and write to file
```

**Invalid set code handling**: `getBoosters().get(unknownCode)` returns `null`. Check for null and throw `IllegalArgumentException` before generating any output.

**Forge module dependency**: `forge-game` (already in `pom.xml`) transitively provides all required classes. No new Maven dependencies needed.

---

## 2. Card Encoder — Extracting Pooled Representation

**Decision**: Add an `encode()` method to `CardPriceTransformerModel` that returns `cat([max_pooled, mean_pooled])` without the output head or meta features.

**Rationale**: The existing `forward()` method requires `meta` (a 15-dim metadata vector) and returns a scalar logit. For the sealed deck encoder we need only the text-based pooled representation. Adding a dedicated `encode()` method avoids hacks like passing zero meta and extracting intermediate tensors.

**Alternatives considered**:
- Pass zero meta to `forward()` and ignore the scalar output: incorrect — zero meta changes the pooled tensor because the output head only affects the final linear projection, not the pooling. Actually, the pooling is done before meta is concatenated, so zeroing meta would still give the correct pooled vectors. But this approach is confusing and fragile.
- Create a separate encoder class in the `sealed` module: requires duplicating the architecture, which breaks if `CardPriceTransformerModel` is updated.

**Implementation**: Add to `transformer_model.py`:

```python
@torch.no_grad()
def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Return pooled card embedding without meta or output head.

    Returns:
        (batch_size, 2 * d_model) — cat([max_pooled, mean_pooled])
    """
    seq_len = input_ids.size(1)
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    x = self.token_embedding(input_ids) + self.position_embedding(positions)
    x = self.embed_dropout(x)
    padding_mask = attention_mask == 0
    x = self.encoder(x, src_key_padding_mask=padding_mask)
    padding_mask_3d = (attention_mask == 0).unsqueeze(-1)
    x_max = x.masked_fill(padding_mask_3d, float("-inf"))
    max_pooled = x_max.max(dim=1).values
    x_mean = x.masked_fill(padding_mask_3d, 0.0)
    lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean_pooled = x_mean.sum(dim=1) / lengths
    return torch.cat([max_pooled, mean_pooled], dim=-1)  # (batch, 2*d_model)
```

The `@torch.no_grad()` decorator is appropriate since `encode-cards` is inference-only. The `embed_dropout` is retained (consistent with `forward()`).

---

## 3. NPZ Format for Embeddings

**Decision**: Save each embedding as `np.savez_compressed(path, embedding=array)` and load with `np.load(path)["embedding"]`.

**Rationale**: `.npz` is the standard NumPy compressed format. Storing under the key `"embedding"` makes the array self-describing. Compressed saves ~30–50% disk space vs. uncompressed for float32 arrays.

**Atomic write pattern**:

```python
import tempfile, os
from pathlib import Path
import numpy as np

def save(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    np.savez_compressed(tmp, embedding=embedding)
    # np.savez_compressed adds .npz suffix automatically if not present
    # tmp path already ends in .tmp, so actual file is tmp.with_suffix(path.suffix)
    os.replace(str(tmp) + ".npz" if not str(tmp).endswith(".npz") else str(tmp), str(path))
```

**Note on np.savez_compressed naming**: `np.savez_compressed(path, ...)` always appends `.npz` if the path doesn't end in `.npz`. Since the final path already ends in `.npz`, write the temp file as `{stem}.tmp.npz` to avoid double-extension issues:

```python
tmp = path.parent / (path.stem + ".tmp.npz")
np.savez_compressed(tmp, embedding=embedding)
os.replace(str(tmp), str(path))
```

---

## 4. Module Layout for `sealed`

**Decision**: New top-level package `src/sealed/` with the same `domain/application/infrastructure` structure as `price_predictor`.

**Rationale**: Clean separation as required by the spec. The `sealed` module depends on `price_predictor` for the model/tokenizer; `price_predictor` never depends on `sealed`.

**`pyproject.toml` / `setup.py` consideration**: The `sealed` package must be discoverable. If `price_predictor` is installed as an editable package, `sealed` will need the same treatment. A `src/sealed/__main__.py` provides the `python -m sealed` entry point.

---

## 5. Progress Reporting

**Decision**: Print a single-line progress update every 100 cards (encode-cards) or every 100 pools (generate-pools), using `\r` overwrite for encode-cards and newline-per-update for generate-pools (since the Java process streams stdout).

**Rationale**: Simple, no new dependencies. The `\r` approach gives a live counter without scrolling for encode-cards. Generate-pools reads from the Java subprocess's stdout line by line and re-prints each line.
