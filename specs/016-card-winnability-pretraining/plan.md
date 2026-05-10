# Implementation Plan: Card Winnability Pretraining for Sealed Encoder

**Branch**: `016-card-winnability-pretraining` | **Date**: 2026-05-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/016-card-winnability-pretraining/spec.md`

## Summary

Replace the current single-`shrunk_label` sealed encoder pretraining target with a
9-head per-card winnability target plus an MLM auxiliary objective. Each card now
contributes (a) `score_play` and `score_draw` (per-side winning influence), (b)
`played_rate` (cast probability conditioned on being in deck), (c) `cast_lift`
(win-rate delta when actually played vs. dead in hand), and (d) `color_lift_X`
for each WUBRG color (color-conditioned winning influence) — totalling nine
regression cells driven by two passes over `cards-played.txt` (primary counters,
then per-color slices keyed off each card's `mana cost:` line). A masked-language
modeling head reads the contextualized token sequence and reconstructs masked
non-special tokens to act as a regularizer. Spec § Clarifications fixes optimizer
(AdamW + per-parameter-group max-norm 1.0 gradient clipping), LR schedule (linear
warmup over the first 5% of `--epochs × batches_per_epoch`, then constant),
best-checkpoint selection (full validation loss `L_reg + (--mlm-weight) · L_mlm`),
per-head per-batch sum-to-1 weight normalization, and a header row in
`cards-win-rates.txt`. The feature ships entirely inside the existing
`src/sealed/` package — no new modules — and the Java side already filters basic
lands at `cards-played.txt` write time (FR-004a, satisfied by
`PlayedCardCollector`/`MatchGenerator`), so this plan touches only Python.

## Technical Context

**Language/Version**: Python 3.14+ (`sealed` and `price_predictor` packages); Java 17+ for the unrelated `forge-connector` module — not touched by this feature (its `PlayedCardCollector` already implements FR-004a).
**Primary Dependencies**: `torch` (PyTorch with CUDA 12.6 wheels), `numpy`. No new dependencies. The `MtgTokenizer` and the price-side `build_vocabulary` utility are reused; the latter gains a `[MASK]` token in its seeded specials.
**Storage**: PyTorch `.pt` checkpoints under `models/sealed/encoder/` (extended `config` payload — `d_token`, `n_pool_queries`, plus the new auxiliary-loss config); flat-text `cards-played.txt` (already produced unchanged); new flat-text `cards-win-rates.txt` (overwritten per run, header + 23 columns); `models/sealed/encoder/vocab.txt` (extended with reserved `[MASK]` token, FR-009a).
**Testing**: `pytest` for Python; new unit tests under `tests/unit/sealed/application/test_train_encoder.py` cover aggregation passes, label arithmetic, per-batch sum-to-1 weight normalization, MLM masking shape/positions, stratification fallback (FR-018), and corpus-consistency error path. `tests/unit/sealed/application/test_build_vocab.py` is extended with a `[MASK]` presence assertion. Existing slow `tests/integration/sealed/test_train_encoder_smoke.py` is updated to assert (a) `latest.pt` carries no head/MLM keys, (b) the new `cards-win-rates.txt` header, and (c) full validation loss is reported per epoch.
**Target Platform**: Local development (Windows 11) and any OS with Python 3.14 + an optional CUDA-capable GPU. Training is CPU-feasible at small `--epochs` budgets and ~5–20× faster on a commodity GPU.
**Project Type**: CLI tool. Two existing subcommands — `python -m sealed train-encoder` and `python -m sealed build-vocab` — gain new flags / behaviors. No new top-level commands.
**Performance Goals**: With the default flags (`--n-layers 6`, `--n-heads 4`, `--batch-size 64`) and the current ~25K-card corpus, one training epoch should complete in well under a minute on CPU and a few seconds on a CUDA GPU; `--patience 20` typically caps a full run at <2 hours wall-clock. The MLM head adds one `Linear(d_token, vocab_size)` per token position; with `vocab_size ≈ 5000` and `max_seq_len ≈ 256` the extra cost is negligible compared to the transformer stack.
**Constraints**: `d_model` MUST be divisible by `--n-pool-queries` (existing `SealedEncoderConfig.__post_init__` invariant, FR-015). The vocabulary file MUST contain a reserved `[MASK]` token (FR-009a, FR-023a) — added by `build-vocab`. `train-encoder` MUST fail loud (with the exit codes already wired in `cli.run_train_encoder`) on every missing-prerequisite path enumerated in FR-023.
**Scale/Scope**: ~25K cards under `output/cardsfolder/`, vocab ~5K tokens, ~10⁵–10⁷ game lines in `cards-played.txt` after sustained self-play, 9 regression cells × N cards in the per-card label map (cards filtered out by FR-012 absent from the map; cells with zero slice denominator present-but-empty per FR-012). Per-batch loss involves one forward pass per card (no per-head architectural duplication; the heads are five tiny linear projections off the same pooled vector plus one linear projection off the contextualized token sequence).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Fast Automated Tests | **PASS** | New unit tests cover the two-pass aggregator (primary + per-color), the FR-011 label arithmetic in raw and shrunk form, FR-012 zero-denominator handling, FR-013a header + 23-column row format, FR-014b masking (probability + non-special-token-only), FR-015a MLM head shape, FR-017 per-batch sum-to-1 weight normalization, and FR-018 stratification fallback. The slow integration smoke is upgraded but stays marked `@pytest.mark.integration` so the fast suite remains fast. |
| II. Simplicity First | **PASS** | No new packages, no new modules, no new domain entities outside the existing `train_encoder.py` private dataclasses. The five regression-head families are realized as a single `nn.ModuleDict` of small linear projections off the shared pooled vector — no per-head encoder stack. The MLM head is one `nn.Linear`. The two-pass aggregator stays in `train_encoder.py` (single-file responsibility). The shared encoder `_BestCheckpoint` helper that already lives in `train_encoder.py` (mirroring `train_transformer.py`) is reused as-is — no premature extraction. |
| III. Data Integrity | **PASS** | The aggregator validates input via the existing `iter_rows` parser (rejects malformed lines mid-file, tolerates a final partial line for JVM-crash recovery). The corpus-consistency check (FR-023d) runs after pass 1 so the failure message can name the offending cards. The encoder checkpoint excludes regression and MLM heads at save time via the existing `_ENCODER_PREFIXES` filter on `SealedEncoderStore` — newly-added head modules use prefixes outside that set so no leakage is possible. Random seed is fixed at 42 (FR-022); aggregation is deterministic given a fixed input file. |
| IV. Domain-Driven Design | **PASS** | Layering preserved: `domain/encoder_model.py` extends architecture only; `application/train_encoder.py` owns the use case (aggregation, label map, train/val split, training loop); `infrastructure/encoder_store.py` and `cli.py` are extended for new payload fields and CLI flags. Aggregation never reaches into `infrastructure/`; corpus reads go through the existing `ConvertedCardLocator` adapter. The dependency graph stays inward-only (infra → app → domain). |
| V. MTG Forge Interoperability | **PASS** | No Forge-facing or remote-API changes. The Java `forge-connector` module is untouched; its existing `PlayedCardCollector.shouldRecord()` and `MatchGenerator.BASIC_LAND_NAMES` already implement FR-004a (basic-land filtering). The `cards-played.txt` schema documented by spec 014 is unchanged. |
| VI. Documentation | **PASS** | Two contract files (`contracts/cli.md` for the `train-encoder` and `build-vocab` CLI surface, `contracts/files.md` for the `cards-win-rates.txt` schema and the `{timestamp}.pt` / `latest.pt` payload) describe the user-visible artifacts. `quickstart.md` walks the build-vocab → match-outcomes → train-encoder → train-scorer pipeline end-to-end. CLAUDE.md already describes `train-encoder` and `build-vocab`; entries will be updated alongside implementation to mention the new heads, MLM head, and the new `cards-win-rates.txt` schema. |
| VII. Codebase-Aware Planning | **PASS** | Survey complete; outcome below. |

### Codebase Survey (Principle VII — required)

Full survey: [research.md#codebase-survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 11 existing concepts surveyed. 8 reused as-is or extended; 3 (`CardLabel`, `WinnabilityMap`, `_aggregate_counts`) replaced because the spec changes their schema (single-scalar → 9-cell label, single-pass → two-pass aggregation). 0 parallel concepts introduced.
- **Adjacent prior art**: 5 prior-art areas reused (cards-played streaming reader, converted-card text/mana-cost lookup, tokenizer + vocab loader, encoder save filter, linear-warmup schedule helper). The MLM head and the per-batch sum-to-1 weight normalization have no prior art; both are implemented inline in `train_encoder.py` as small private helpers.
- **Convention alignment**: Mirrors existing `sealed` package conventions for CLI registration (`_build_*_parser` + `set_defaults(func=run_*)`), application-layer signatures (`run(config: TrainEncoderConfig) -> Path`), dataclass-backed config, persistence helper (`SealedEncoderStore`), and timestamped logging (`_log`). No deviations.
- **Third-instance check**: `_BestCheckpoint` already exists in two trainers (price-side `train_transformer.py` and sealed `train_encoder.py`); `train_scorer.py` uses a different metric (val_acc, not val_loss) so it isn't a third instance of the *same* pattern. The existing follow-up note in `train_encoder.py` (extract when a third loss-driven trainer arrives) still stands; this feature does not introduce that third instance.

**Follow-up tasks from survey** (carry into `tasks.md`):

- *Replace* the single-scalar `CardLabel` dataclass with a 9-cell `CardCounters` + `CardLabels` pair (research.md §1.1).
- *Replace* `_aggregate_counts` with `_aggregate_pass_one` (primary + @play counters) and `_aggregate_pass_two` (per-color counters keyed off `mana cost:`).
- *Add* `[MASK]` to `_seed_special_tokens` in `price_predictor.application.build_vocabulary` (FR-009a) — a one-line addition, but it must precede `train-encoder` development so the vocab-build path produces the token.
- *Extend* `SealedEncoderModel` with a `nn.ModuleDict` of regression heads + an MLM head (`Linear(d_model, vocab_size)`) that reads the contextualized token sequence (pre-pool).
- *Update* `SealedEncoderStore._ENCODER_PREFIXES` documentation and assertion error message to name the new head prefixes that are intentionally filtered out (`regression_heads.`, `mlm_head.`).
- *Add* `--mlm-weight` (default 0.1) and `--mlm-mask-prob` (default 0.15) flags to `_build_train_encoder_parser` and the `TrainEncoderConfig` dataclass.
- *Update* `cards-win-rates.txt` writer: header row, 23 columns per row, sort key changes from raw_ratio to `shrunk_score_play`.

## Project Structure

### Documentation (this feature)

```text
specs/016-card-winnability-pretraining/
├── plan.md              # This file
├── research.md          # Codebase survey + Phase 0 design decisions
├── data-model.md        # Entity & artifact contracts
├── quickstart.md        # End-to-end build-vocab → train → use workflow
├── contracts/
│   ├── cli.md                    # train-encoder & build-vocab CLI surface
│   └── files.md                  # FR-013a cards-win-rates.txt schema +
│                                 #   {timestamp}.pt / latest.pt payload schema
└── tasks.md             # Generated by /speckit.tasks (NOT created here)
```

### Source Code (repository root)

```text
src/sealed/
├── application/
│   ├── train_encoder.py          # MODIFIED:
│   │                             #   - CardLabel (single-scalar) → CardCounters (8 primary +
│   │                             #     5 per-color counter sets) and CardLabels (9 raw + 9 shrunk)
│   │                             #   - WinnabilityMap → CardLabelMap (rename for clarity;
│   │                             #     payload changes)
│   │                             #   - _aggregate_counts → _aggregate_pass_one (8 primary
│   │                             #     counters incl. @play subset) + _aggregate_pass_two
│   │                             #     (per-color counters from mana cost: line)
│   │                             #   - _build_winnability_map → _build_label_map (FR-011 9-label
│   │                             #     formulae, raw + shrunk; FR-012 zero-denominator handling)
│   │                             #   - _split_cards: stratification key score_play (was
│   │                             #     shrunk_label) with FR-018 fallback
│   │                             #   - _WinnabilityDataset: returns (input_ids, attention_mask,
│   │                             #     labels[9], weights[9]) per FR-017a
│   │                             #   - _make_optimizer: keep AdamW + 5%-warmup; add per-group
│   │                             #     max-norm 1.0 clip in _train_epoch (FR-022)
│   │                             #   - _train_epoch: MLM mask draw (FR-014b), per-batch per-head
│   │                             #     sum-to-1 weighted MSE (FR-017), MLM CE at masked positions
│   │                             #     (FR-015a), full loss = L_reg + mlm_weight · L_mlm
│   │                             #   - _eval_loss: full loss for early stopping & best-checkpoint
│   │                             #     selection (FR-019)
│   │                             #   - _write_win_rates: header row + 23-column schema
│   │                             #     (FR-013a), sort by shrunk_score_play
│   │                             #   - TrainEncoderConfig: + mlm_weight (0.1), mlm_mask_prob
│   │                             #     (0.15)
│   └── build_vocab.py            # UNCHANGED — already delegates to price-side build_vocabulary;
│                                 #   the [MASK] seed lands in the price-side helper, not here
├── domain/
│   ├── encoder_model.py          # MODIFIED:
│   │                             #   - SealedEncoderModel: regression_head (single Sequential)
│   │                             #     replaced with nn.ModuleDict of five heads
│   │                             #     (score_play, score_draw, played_rate, cast_lift, color_lift)
│   │                             #     each Linear(2*d_model, 1 or 5); + mlm_head =
│   │                             #     nn.Linear(d_model, vocab_size) reading the
│   │                             #     contextualized token sequence
│   │                             #   - forward(): returns dict with per-head predictions +
│   │                             #     contextualized token sequence (training path); encode()
│   │                             #     stays the same (inference returns pooled vector only)
│   │                             #   - SealedEncoderConfig: no schema change (vocab_size already
│   │                             #     includes [MASK] when the vocab is built correctly)
│   └── (no other domain changes)
└── infrastructure/
    ├── cli.py                    # MODIFIED:
    │                             #   - _build_train_encoder_parser: + --mlm-weight,
    │                             #     --mlm-mask-prob; tighten --shrinkage-k help
    │                             #   - run_train_encoder: pass new fields into TrainEncoderConfig
    └── encoder_store.py          # MODIFIED:
                                  #   - documentation: list new head prefixes that are
                                  #     intentionally filtered out by _ENCODER_PREFIXES
                                  #   - error path: enrich the "non-encoder keys" message to
                                  #     point readers at FR-020

src/price_predictor/
└── application/
    └── build_vocabulary.py       # MODIFIED:
                                  #   - _seed_special_tokens: + "[MASK]" alongside [PAD]/[UNK]/cardname
                                  #   - VocabBuildResult: domain_token_count grows by 1 (specials)

tests/unit/sealed/
├── application/
│   ├── test_train_encoder.py     # NEW (currently no unit tests for this module):
│   │                             #   - aggregator pass 1 (primary + @play subset)
│   │                             #   - aggregator pass 2 (mana cost → colors;
│   │                             #     hybrid {W/U}, Phyrexian {W/P}, generic {2}, {X})
│   │                             #   - label arithmetic (raw vs shrunk for all 9 heads;
│   │                             #     k = 0 vs k = 20)
│   │                             #   - FR-012 zero-denominator handling (cell present-but-empty,
│   │                             #     contributes zero loss; card excluded if total in_deck = 0)
│   │                             #   - per-batch per-head sum-to-1 weight normalization
│   │                             #     (single-card batch, mixed-card batch with one head all-zero)
│   │                             #   - FR-014b mask draw (probability + non-special-token mask
│   │                             #     positions only; [PAD]/[UNK]/cardname/[MASK] never become [MASK])
│   │                             #   - stratification fallback for degenerate score_play
│   │                             #     distributions and for cards with empty score_play cell
│   │                             #   - cards-win-rates.txt header + 23 columns + sort order
│   │                             #   - corpus-consistency error names missing cards (capped)
│   └── test_build_vocab.py       # MODIFIED: assert [MASK] present in emitted vocab
├── domain/
│   └── test_encoder_model.py     # MODIFIED:
│                                 #   - forward returns the new dict shape under training=True
│                                 #   - encode() (inference) is unchanged: returns (B, 2*d_model)
│                                 #   - mlm_head logits shape (B, T, vocab_size)
│                                 #   - state_dict prefixes: regression_heads.* and mlm_head.*
│                                 #     are present in the live model and absent from the saved file
└── infrastructure/
    └── test_encoder_store.py     # MODIFIED: round-trip with new heads in the live model;
                                  #   assert the saved file has only token_encoder.*/card_encoder.*

tests/integration/sealed/
└── test_train_encoder_smoke.py   # MODIFIED:
                                  #   - tiny synthetic cards-played.txt + tiny corpus + tiny vocab
                                  #     (with [MASK])
                                  #   - run train-encoder for ~3 epochs
                                  #   - assert: latest.pt loads via SealedEncoderStore (no leaked
                                  #     head/MLM keys), full val_loss is monotonically reported,
                                  #     cards-win-rates.txt has the header + non-empty rows

forge-connector/                  # UNCHANGED — basic-land filtering already in
                                  #   PlayedCardCollector and MatchGenerator
```

**Structure Decision**: All Python changes live in the existing `sealed` package
(application + domain + infrastructure) and one shared utility in the
`price_predictor` package (`build_vocabulary._seed_special_tokens`). No new
modules or packages are introduced. The `forge-connector` Java module is
untouched — its existing basic-land filtering already satisfies FR-004a.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitution violations. No complexity justifications needed.
