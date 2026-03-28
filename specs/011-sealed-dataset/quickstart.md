# Quickstart: Sealed Dataset Preparation

## Prerequisites

1. **Forge connector JAR built**:
   ```bash
   cd forge-connector && mvn package -DskipTests
   ```

2. **Card scripts converted** (feature 006):
   ```bash
   python -m price_predictor convert --cards-path ../forge/forge-gui/res/cardsfolder/ --output-path output/cardsfolder/
   ```

3. **Pretrained encoder available** (feature 007/010):
   ```bash
   # Verify the model and vocab exist at their default locations:
   ls models/price-predictor/transformer/latest.pt
   ls models/price-predictor/transformer/vocab.txt
   ```

---

## Step 1 — Encode Card Embeddings

```bash
python -m sealed encode-cards
```

This scans `output/cardsfolder/` recursively and writes a `.npz` embedding file alongside each `.txt` card script. Already-encoded cards are skipped, so this is safe to re-run.

**Custom paths:**
```bash
python -m sealed encode-cards \
    --encoder-path models/price-predictor/transformer/latest.pt \
    --vocab-path   models/price-predictor/transformer/vocab.txt \
    --cards-path   output/cardsfolder/
```

**Expected output:**
```
Encoding cards in output/cardsfolder/
Progress: 100 encoded (0 skipped)
...
Done: 32000 processed, 0 skipped, 0 errors
```

---

## Step 2 — Generate Sealed Pools

```bash
python -m sealed generate-pools
```

This generates 10,000 RVR sealed pools and writes them to `output/sealed/pools/RVR/pools.txt`.

**Custom set and size:**
```bash
python -m sealed generate-pools --set MH3 --size 5000
# Output: output/sealed/pools/MH3/pools.txt
```

**Custom output path:**
```bash
python -m sealed generate-pools --set RVR --size 10000 --pools-path /data/sealed/RVR/
```

**Expected output:**
```
Generating 10000 RVR sealed pools...
Generated 1000/10000 pools
Generated 2000/10000 pools
...
Done: 10000 pools written to output/sealed/pools/RVR/pools.txt
```

---

## Notes

- The two commands are independent — run them in either order or in parallel.
- Re-running `encode-cards` after adding new card scripts only encodes the new ones.
- To re-encode all cards (e.g. after retraining the encoder), delete the `.npz` files first:
  ```bash
  find output/cardsfolder/ -name "*.npz" -delete
  ```
- To regenerate pools for a different set without touching existing ones, just run `generate-pools` with a different `--set`.
