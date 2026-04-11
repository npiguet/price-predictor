# Quickstart: Sealed Deck Scorer

## Prerequisites

- Features 011 (encode-cards, generate-pools) and 012 (match-outcomes) are complete
- Python 3.14+ with PyTorch installed (`pip install torch`)
- At least 10,000 match outcomes in `output/sealed/match-outcomes.txt`
- Card scripts in `output/cardsfolder/` with existing 512-dim embeddings
- Forge connector JAR built: `cd forge-connector && mvn package -DskipTests`

## Step 1: Re-encode cards to 544 dimensions

The existing 512-dimensional embeddings need to be replaced with 544-dimensional vectors that include deterministic game features:

```bash
python -m sealed encode-cards --clean
```

This deletes all existing `.npz` files and re-encodes every card. Takes ~10-30 minutes depending on card count.

## Step 2: Train the scorer (Phase A — frozen embeddings)

```bash
python -m sealed train-scorer \
    --outcomes-path output/sealed/match-outcomes.txt \
    --cards-path output/cardsfolder/ \
    --checkpoint-dir models/sealed/scorer/ \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3
```

Watch training and validation loss. Training is converging when validation loss plateaus.

## Step 3: Fine-tune embeddings (Phase B — optional)

Once Phase A validation loss plateaus, resume with unfrozen embeddings:

```bash
python -m sealed train-scorer \
    --resume models/sealed/scorer/best_l2_h4_s4_ff1088_mlp256.pt \
    --checkpoint-dir models/sealed/scorer/ \
    --unfreeze-embeddings \
    --embedding-lr 1e-5 \
    --epochs 50
```

Monitor embedding drift — if it climbs rapidly, lower `--embedding-lr`.

## Step 4: Evaluate against Forge baseline

```bash
python -m sealed evaluate-scorer \
    --checkpoint models/sealed/scorer/best_l2_h4_s4_ff1088_mlp256.pt \
    --pools 12 \
    --best-of 3 \
    --workers 4
```

This generates 12 pools, builds one scorer deck and one Forge deck from each pool, then plays
144 best-of-3 matches (every scorer deck vs every Forge deck). Prints per-pool win rate
comparisons and aggregate win rates for each builder group.

## Expected Outcomes

- **Validation accuracy**: >55% (above 50% random baseline)
- **Forge baseline aggregate win rate**: scorer decks win more than Forge decks across the shared opponent field
- **Per-pool comparison**: scorer decks outperform Forge decks from the same pool in the majority of pools
- **Sanity check**: Scorer ranks Forge-built decks higher than random decks >80% of the time
