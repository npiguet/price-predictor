# The card embedding cache

Rules to respect when running or changing `encode-cards`, or anything that reads the `.npz` cache it writes. Flags and defaults are each command's `--help`; the encoder stages are specified in `specs/2026-04-27-encoder-fine-tuning.md` and `specs/2026-05-03-card-winnability-pretraining.md`.

## The cache fixes every downstream width

- `encode-cards` strips the `name:` line from each card and writes a `.npz` for every `.txt` under `--cards-path`, beside it, or with `--embeddings-path DIR` at the card's place in a tree of the same layout under `DIR` (a `cache` run directory under the layout): a `float32` array `(pooled_dim + FEATURE_COUNT,)` under the key `"embedding"`, the pooled text vector followed by the deterministic game features, always in the trailing `FEATURE_COUNT` slots.
- `pooled_dim` depends on the encoder: `2·d_model` for the price encoder and for the sealed encoder's dual pool, `d_model` for the sealed encoder's attention-only pool.
- The scorer's input width is not a flag. It is the width of the `.npz` cache, and the picker's input width is the scorer's. So any encoder (price or sealed, any `--d-model` or `--pool-mode`) works as long as `encode-cards` was run with it.
- A scorer checkpoint, Phase B encoder or picker whose width disagrees with the cache fails fast with a clear error rather than a torch shape mismatch.
- Every reader of the cache (`build-decks`, `pick-decks`, `train-scorer`, `train-picker`, `evaluate-scorer`, and the draft commands) takes `--embeddings-path` to read it from where `encode-cards` wrote it, through `ConvertedCardLocator(embeddings_path=...)`; unset, it reads beside the cards.
- `encode-cards --clean` deletes every `.npz` under the tree it writes (`--embeddings-path`, else `--cards-path`). That is why the effect model's ability cache is never put in a converted card tree.
- `--encoder-checkpoint` and `--scorer-checkpoint` are mutually exclusive on `encode-cards`; the second extracts the encoder from a Phase B scorer checkpoint.
