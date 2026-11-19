# Vocabulary, encoder, scorer and picker training

Rules to respect when running or changing `build-vocab`, `train-encoder`, `train-scorer` or `train-picker`. Flags and defaults are each command's `--help`; what each stage specifies is in `specs/2026-03-28-sealed-deck-picker.md`, `specs/2026-04-27-encoder-fine-tuning.md`, `specs/2026-05-03-card-winnability-pretraining.md` and `specs/2026-05-19-one-shot-deck-picker.md`.

## Vocabulary and encoder

- The sealed vocabulary (`build-vocab --vocab-path`) is independent from the price-side one; building one does not modify the other. `[MASK]` is seeded for `train-encoder`'s MLM loss.
- `train-encoder` saves only the encoder weights; the regression and MLM heads are filtered out at save time. The pool mode is recorded in the checkpoint's config.
- Its train/validation split is card-disjoint.
- A card referenced in `cards-played.txt` with no converted `.txt` is warned about and dropped. It does not block the run.
- Every `train-encoder` run overwrites its win-rate table, `--win-rates-output` (by default `cards-win-rates.txt` in `--model-output`).

## Scorer and picker

- `train-scorer` Phase A (the default, `--embedding-lr 0`) reads the `.npz` cache with the encoder frozen. Phase B (`--embedding-lr` nonzero) fine-tunes the encoder jointly and needs either `--scorer-checkpoint <phaseA>.pt` (fresh start, encoder from the then-required `--encoder-checkpoint`) or `--resume <phaseB>.pt`.
- A Phase B checkpoint carries `encoder_state_dict` and `encoder_config`, so the file is self-contained.
- Architecture flags configure a fresh run only. On `--resume`, `--scorer-checkpoint` or `--picker-checkpoint` they are forbidden; the architecture comes from the checkpoint.
- The best scorer checkpoint is selected by validation accuracy; the best picker checkpoint by validation reward.
- `train-picker` scores each sampled deck's chosen spells and nonbasic lands only, never basics. `--kl-coef` requires `--picker-checkpoint`. Checkpoints hold picker weights only.
