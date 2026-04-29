# Implementation Plan: Encoder Fine-Tuning (Phase B) for Sealed Scorer

**Branch**: `015-encoder-fine-tuning` | **Date**: 2026-04-29 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/015-encoder-fine-tuning/spec.md`

## Summary

Replace the existing `--unfreeze-embeddings` lookup-table fine-tuning path
in `train-scorer` with proper encoder fine-tuning ("Phase B"). A
non-zero `--embedding-lr` puts the price-predictor encoder in the
training graph alongside the sealed scorer, with the encoder loaded
fresh from a Phase A checkpoint via the new `--scorer-checkpoint` flag
(scorer weights only; optimizer state and architecture come from the
checkpoint, training-loop state is reset). After Phase B finishes, the
new `encode-cards --scorer-checkpoint` flag refreshes every `.npz` under
`output/cardsfolder/` with the fine-tuned encoder, so downstream tools
see Phase B's improvements without any code change. Spec § Clarifications
locks the validation cadence at once-per-epoch, makes `--resume`
phase-locked (no cross-phase resume), inherits scorer architecture from
the checkpoint on bootstrap, requires `--scorer-checkpoint` or `--resume`
for any Phase B run, and switches both phases to AdamW with per-group
max-norm 1.0 gradient clipping.

## Technical Context

**Language/Version**: Python 3.14+ (`sealed` and `price_predictor` packages); Java 17+ for the unrelated `forge-connector` module — not touched by this feature.
**Primary Dependencies**: `torch` (PyTorch with CUDA 12.6 wheels), `numpy`, `scikit-learn` (for `train_test_split` only). No new dependencies.
**Storage**: PyTorch `.pt` checkpoints under `models/sealed/scorer/` and `models/price-predictor/transformer/`; `.npz` card embeddings under `output/cardsfolder/`; flat-text `match-outcomes.txt`. All file formats either unchanged (price-predictor encoder, `.npz`, match-outcomes) or extended additively (sealed scorer checkpoint gains an `encoder_state_dict` key + a richer `train_config` dict — no existing key removed or repurposed).
**Testing**: `pytest` for Python (unit tests under `tests/unit/sealed/`, slower integration tests under `tests/integration/`). Java tests are unaffected (no Java code changed).
**Target Platform**: Local development (Windows 11) and any OS with Python 3.14 + an optional CUDA-capable GPU. Phase B training is CPU-feasible but ~10× faster on GPU.
**Project Type**: CLI tool — two new flag groups added to existing `python -m sealed train-scorer` and `python -m sealed encode-cards` subcommands.
**Performance Goals**: Phase B epoch wall-clock ≤ 4× a Phase A epoch (per spec § Caching savings); the within-batch encoder cache (FR-007) is the lever. `--patience` keeps total run length within 5–15 epochs typically.
**Constraints**: Encoder gradient flow must accumulate from every duplicate-card reference in a batch (autograd handles this naturally if the cache hands out the same `Tensor` rather than a copy); per-batch graph must release between steps (no `retain_graph=True`). GPU memory budget: scorer + encoder + AdamW two-group state ≈ 200MB activation + ~50MB parameters per pass, within commodity-GPU range.
**Scale/Scope**: ~26K cards under `output/cardsfolder/` (each gets a `.txt` and a `.npz`); 1.2M card references in the match-outcomes corpus; 10K-50K training pairs after split. Phase B touches ~5–15 epochs over the same corpus; `encode-cards` re-runs once at the end.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Fast Automated Tests | **PASS** | New unit tests cover phase-detection and CLI rejection paths in `_resume_or_build_model`/`run_train_scorer`/`run_encode_cards`, the AdamW two-group optimizer assembly, the within-batch encoder cache (with mock encoder asserting one call per unique card), and the reference-batch drift metric. Existing fast tests in `tests/unit/sealed/` continue to pass with the `--unfreeze-embeddings` path removed. End-to-end Phase B is a slow integration test marked `@pytest.mark.integration`. |
| II. Simplicity First | **PASS** | No new packages or modules. Every new domain concept either extends an existing dataclass (`TrainScorerConfig`, `LoadedScorerCheckpoint`, `_TrainingContext`, `ResumeState`) or adds a sibling method to an existing module (`CardEncoder.encode_batch_text`, `EmbeddingTable.set_text_vectors`). The `--unfreeze-embeddings` flag, the `EmbeddingTable.unfreeze()` API, and the `--val-interval` flag are removed (FR-002 + Decision §6). |
| III. Data Integrity | **PASS** | Train/val split is deterministic (FR-011a, already enforced by `random_seed=42`). Phase A and Phase B checkpoints carry distinct keysets (`encoder_state_dict` present iff Phase B), giving an authoritative phase indicator. `train_config` makes every checkpoint self-describing for reproducibility (FR-009). The CLI rejects every cross-phase / mutually-exclusive combination with a clear error rather than silently doing the wrong thing. |
| IV. Domain-Driven Design | **PASS** | Layering preserved: `domain` (extended `CardEncoder`), `application` (extended `train_scorer.py`, `encode_cards.py` use cases), `infrastructure` (extended `scorer_store.py`, `cli.py`). The `EmbeddingTable` keeps its existing role (per-batch index → vector lookup); only its update mechanism changes (encoder writes fill rows in Phase B). No infrastructure leaks into the domain layer. |
| V. MTG Forge Interoperability | **PASS** | No Forge-facing or remote-API changes. The Java stub library (`forge-connector`) is untouched — Phase B affects only Python training and `.npz` artifacts, both of which downstream Forge consumers see only through the existing `.npz` contract. |
| VI. Documentation | **PASS** | The two CLI contracts (`contracts/train-scorer-cli.md`, `contracts/encode-cards-cli.md`) and the checkpoint format contract (`contracts/checkpoint-format.md`) are part of this plan. Quickstart in `quickstart.md` walks the full Phase A → Phase B → re-cache → evaluate workflow. CLAUDE.md describes both subcommands; entries will be updated alongside implementation. |
| VII. Codebase-Aware Planning | **PASS** | Survey complete; outcome below. |

### Codebase Survey (Principle VII — required)

Full survey: [research.md#codebase-survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 16 existing concepts surveyed. 12 reused as-is or with extension; 1 (`EmbeddingTable.unfreeze()`/`is_frozen()` API) explicitly removed; 0 parallel concepts introduced.
- **Adjacent prior art**: 4 prior-art areas surveyed (encoder loading via `transformer_store`, checkpoint persistence via `torch_checkpoint`, two-group optimizer assembly, CLI flag registration via `add_dataclass_arg`). All reused. The within-batch encoder cache has no prior art and is implemented as a per-step local dict in `train_scorer.py` (Decision §4).
- **Convention alignment**: Mirrors existing `sealed` package conventions for CLI registration, application-layer signatures, dataclass-backed config, persistence helper, end-of-epoch logging. No deviations.
- **Third-instance check**: No sub-problem already solved twice in the codebase. Encoder weight loading happens in two places (`encode-cards` and Phase B `train-scorer`), but they share the same shape (read `encoder_state_dict` slice, hand to `CardPriceTransformerModel.load_state_dict`) — a one-liner each, not a pattern warranting extraction.

**Follow-up tasks from survey** (carry into `tasks.md`):

- *Rename* `encode-cards --encoder-path` → `--encoder-checkpoint` (Decision §7).
- *Remove* `EmbeddingTable.freeze()` / `unfreeze()` / `is_frozen()` and the `--unfreeze-embeddings` flag (FR-002).
- *Remove* `--val-interval` from `train-scorer` and the dataclass (Decision §6).
- *Switch* `_build_optimizer` from `Adam` to `AdamW` (FR-005a) — both branches.

## Project Structure

### Documentation (this feature)

```text
specs/015-encoder-fine-tuning/
├── plan.md              # This file
├── research.md          # Codebase survey + design decisions
├── data-model.md        # Entity & artifact contracts
├── quickstart.md        # End-to-end Phase B workflow
├── contracts/
│   ├── train-scorer-cli.md
│   ├── encode-cards-cli.md
│   └── checkpoint-format.md
└── tasks.md             # Generated by /speckit.tasks (NOT created here)
```

### Source Code (repository root)

```text
src/sealed/
├── application/
│   ├── train_scorer.py          # MODIFIED:
│   │                            #   - TrainScorerConfig: + scorer_checkpoint, encoder_checkpoint, patience;
│   │                            #     - unfreeze_embeddings, val_interval; default embedding_lr=0.0;
│   │                            #     phase property + invariants
│   │                            #   - _build_optimizer: Adam → AdamW; second group is encoder.parameters()
│   │                            #     instead of embedding_table.parameters()
│   │                            #   - _resume_or_build_model: phase-lock check, scorer_checkpoint bootstrap path,
│   │                            #     encoder weight loading from --encoder-checkpoint or resumed checkpoint
│   │                            #   - _train_one_epoch: per-group clip_grad_norm_, encoder forward + within-batch
│   │                            #     cache, write into EmbeddingTable rows, reference-batch capture at step 0
│   │                            #   - _validate / _embedding_drift: re-encode reference batch each epoch
│   │                            #   - early stopping via --patience (epochs since last peak val_acc)
│   │                            #   - logging: encoder grad norm + embedding_drift in _print_epoch_report
│   └── encode_cards.py          # UNCHANGED — use case is encoder-agnostic; only the CLI assembly changes
├── domain/
│   └── card_encoder.py          # MODIFIED: add encode_batch_text(input_ids, attention_mask, *, with_grad)
│                                #   for Phase B's batched, gradient-tracking forward pass
└── infrastructure/
    ├── cli.py                   # MODIFIED:
    │                            #   - _build_train_scorer_parser: + --scorer-checkpoint, --encoder-checkpoint,
    │                            #     --patience; - --unfreeze-embeddings, --val-interval
    │                            #   - run_train_scorer: phase / mutual-exclusivity / architecture-flag rejection,
    │                            #     late-resolve --encoder-checkpoint default
    │                            #   - _build_encode_cards_parser: rename --encoder-path → --encoder-checkpoint;
    │                            #     + --scorer-checkpoint
    │                            #   - run_encode_cards: load encoder weights from scorer checkpoint when
    │                            #     --scorer-checkpoint set; reject Phase A scorer source per FR-014
    ├── scorer_store.py          # MODIFIED:
    │                            #   - LoadedScorerCheckpoint: + encoder_state_dict, train_config (both Optional)
    │                            #   - save_checkpoint: persist encoder_state_dict (when supplied) and train_config
    └── match_data_loader.py     # MODIFIED: remove EmbeddingTable.freeze/unfreeze/is_frozen;
                                 #   add EmbeddingTable.set_text_vectors(indices, text_vectors)

tests/unit/sealed/
├── application/
│   ├── test_train_scorer.py     # MODIFIED:
│   │                            #   - update existing fixtures from --unfreeze-embeddings → --embedding-lr
│   │                            #   - new: phase-lock cross-resume rejection
│   │                            #   - new: --scorer-checkpoint architecture-flag conflict rejection
│   │                            #   - new: Phase B requires --scorer-checkpoint xor --resume
│   │                            #   - new: AdamW dispatch (one group Phase A, two groups Phase B)
│   │                            #   - new: per-group clip_grad_norm_ called once per group
│   │                            #   - new: within-batch cache deduplicates encoder calls but accumulates grads
│   │                            #   - new: reference-batch drift metric over a tiny synthetic Phase B run
│   │                            #   - new: --patience early stopping
│   │                            #   - new: train_config persisted in checkpoint
│   └── test_encode_cards.py     # MODIFIED: existing tests stay (use case unchanged); CLI-layer tests below
├── infrastructure/
│   ├── test_scorer_store.py     # MODIFIED: round-trip Phase B save/load with encoder_state_dict + train_config
│   ├── test_match_data_loader.py # MODIFIED: drop freeze/unfreeze tests; add EmbeddingTable.set_text_vectors test
│   └── test_cli.py              # MODIFIED:
│                                #   - rename test cases referencing --encoder-path → --encoder-checkpoint
│                                #   - new: encode-cards mutual-exclusivity (only when both explicit)
│                                #   - new: encode-cards rejects Phase A scorer checkpoint
│                                #   - new: train-scorer rejects bare --embedding-lr (no scorer/resume)
│                                #   - new: train-scorer rejects architecture flag with --scorer-checkpoint
└── conftest.py                  # MODIFIED if needed: helpers for synthetic Phase A/Phase B checkpoints

tests/integration/sealed/
└── test_phase_b_smoke.py        # NEW (slow-suite, marked @pytest.mark.integration):
                                 #   tiny synthetic corpus + tiny encoder; run Phase B for 2 epochs;
                                 #   verify checkpoint contains encoder_state_dict; verify --clean
                                 #   re-cache produces .npz files differing from the pre-run cache

src/price_predictor/             # UNCHANGED (price-predictor encoder code not modified;
                                 #   transformer_model.py and transformer_store.py are reused as-is)

forge-connector/                 # UNCHANGED
```

**Structure Decision**: All changes live in the existing `sealed` package
(application + infrastructure + domain) with no new modules or packages.
The `price_predictor` package is reused as-is for tokenizer, transformer
model, and shared checkpoint helpers. The `forge-connector` Java module
is untouched.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitution violations. No complexity justifications needed.
