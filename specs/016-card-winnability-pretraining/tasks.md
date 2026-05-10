---
description: "Task list for feature 016 — Card Winnability Pretraining for Sealed Encoder"
---

# Tasks: Card Winnability Pretraining for Sealed Encoder

**Input**: Design documents from `/specs/016-card-winnability-pretraining/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md,
contracts/cli.md, contracts/files.md, quickstart.md.

**Tests**: Constitution Principle I requires automated tests for every
feature; test tasks below are MANDATORY.

**Organization**: Tasks are grouped by user story so each story can be
implemented and tested independently. Spec 016 already shipped a v1
single-`shrunk_label` implementation in master; this task list captures
the v2 deltas (multi-head + MLM + per-color counters + new file schema +
clarification 2026-05-10 amendments). User stories US1 ("per-game data
collection") and US3 ("scorer consumes sealed encoder defaults") were
fully delivered in v1 — they remain in scope for the spec's success
criteria but require **no new tasks** in this iteration; they are listed
explicitly with that note.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependencies).
- **[Story]**: User story label (US1…US5) for tasks inside a story phase.
- File paths are absolute or relative to repository root.

## Path Conventions

Single-project layout per `plan.md` § Project Structure. Source under
`src/sealed/` and `src/price_predictor/`; tests under
`tests/unit/application/...` (price-predictor unit tests),
`tests/unit/sealed/...` (sealed unit tests), and
`tests/integration/sealed/...` (sealed integration tests).

---

## Phase 1: Setup

No setup tasks. The repository, venv, dependencies, lint/test
configuration, and `forge-connector` Maven build are all already in
place from prior features. This iteration adds no new dependencies (per
`plan.md` § Technical Context).

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: One shared change that every other story depends on — the
`[MASK]` token must be present in any vocabulary `train-encoder` (US2)
or `build-vocab` (US4) consumes.

**⚠️ CRITICAL**: No US2 or US4 task may begin until this phase is
complete.

- [X] T001 Add `"[MASK]"` to `_seed_special_tokens` in `src/price_predictor/application/build_vocabulary.py` (after `cardname`, before domain-term seeding) so every freshly-built vocab contains the reserved token at a stable post-`cardname` ID. Update the function's docstring and the `domain_token_count` accounting; update CLAUDE.md's vocab-build description to mention the new special. (Prior art: `_seed_special_tokens` already exists at line 163; reuse — no parallel concept.) [Plan §Project Structure / Research D-1, FR-009a, FR-023a]

- [X] T002 [P] Update `tests/unit/application/test_build_vocabulary.py` to assert `[MASK]` is present in the returned vocab and that its ID falls between `cardname` and the first domain-term ID (regression guard against accidental removal). (Prior art: existing test cases for `[PAD]`/`[UNK]`/`cardname` presence — extend in place.)

- [X] T003 [P] Update `tests/unit/sealed/application/test_build_vocab.py` to assert that the file written by `BuildVocabConfig`-driven build contains the `[MASK]` token (read the file back and check membership). Complements T002 at the price-predictor layer: T002 guards the seeded specials in the shared `build_vocabulary` helper, T003 guards the file emitted by the sealed-side wrapper. (Prior art: existing test that asserts `[PAD]` and `[UNK]` exist after build — extend.)

**Checkpoint**: Foundational ready — US2 and US4 implementation can now proceed.

---

## Phase 3: User Story 1 — Per-game card-play data accumulation (Priority: P1) ✅ ALREADY DELIVERED

**Goal**: `cards-played.txt` accumulates one line per played game, with
basic lands excluded at write time.

**Independent Test**: Run `python -m sealed match-outcomes` for a few
matches and verify line counts and contiguous game ordering per the
spec's acceptance scenarios.

**Status**: All FR-001 through FR-006 are satisfied by the v1 Java code
shipped in prior commits — `CardsPlayedWriter`, `PlayedCardCollector`
(FR-003, FR-004a basic-land filter), `CardsPlayedRow`,
`MatchGenerator.BASIC_LAND_NAMES`, and the supervisor's existing
opens-writes-closes-per-line strategy. The spec amendments in 016 do
not change Java behavior. **No new tasks**; the story remains
verifiable via the spec's existing acceptance scenarios.

---

## Phase 4: User Story 2 — Train a sealed encoder from scratch (Priority: P1) 🎯 MVP

**Goal**: `python -m sealed train-encoder` reads `cards-played.txt`,
aggregates 9 per-card winnability labels in two passes, trains a token
encoder + card encoder + 5 regression heads + an MLM head from random
init under the FR-017 weighted MSE + MLM CE objective, and saves only
the encoder weights.

**Independent Test**: Run `train-encoder` against a small synthetic
`cards-played.txt` and verify (a) `models/sealed/encoder/latest.pt`
contains only `token_encoder.*` / `card_encoder.*` keys (no heads, no
MLM head); (b) the validation set is card-disjoint from the training
set; (c) the best checkpoint is selected by full validation loss
(`L_reg + (--mlm-weight) · L_mlm`); (d) `output/sealed/cards-win-rates.txt`
has the 24-column header + sorted data rows.

### Tests for User Story 2 (MANDATORY) ✅

> Write these first; they MUST fail before implementation lands.

- [X] T004 [US2] Create `tests/unit/sealed/application/test_train_encoder.py` with a minimal scaffold and aggregation unit tests covering `_aggregate_pass_one` (4 primary + 4 `@play` counters built from a tiny in-memory `CardsPlayedRow` list across both winning and losing sides; `starter` correctly drives the `@play` subset; multi-game matches accumulate correctly). (Prior art: `tests/unit/sealed/infrastructure/test_cards_played_reader.py` for fixture style.) [Research D-13]

- [X] T005 [US2] Extend `tests/unit/sealed/application/test_train_encoder.py` with `_aggregate_pass_two` tests (per-color counters from a fixture corpus mapping card → mana-cost line; hybrid `{W/U}`, Phyrexian `{W/P}`, mono-hybrid `{2/W}`, generic `{2}`, colorless `{C}`, and `{X}` all dispatched correctly; multi-color decks contribute to multiple slices) and a standalone test for the `_colors_from_mana_cost` helper covering the same edge cases. (Depends on T004 — same file.) [Research D-16]

- [X] T006 [US2] Extend `tests/unit/sealed/application/test_train_encoder.py` with label-arithmetic tests for `_build_label_map` (raw and shrunk values for all 9 head families with `k = 0` vs `k = 20`, FR-012 zero-denominator handling produces `None` cells, `wins_when_in_deck + losses_when_in_deck == 0` excludes the card entirely). (Depends on T005 — same file.) [Research D-15, FR-011, FR-012]

- [X] T007 [US2] Extend `tests/unit/sealed/application/test_train_encoder.py` with cards-win-rates writer tests (header row matches the FR-013a column list verbatim; data rows contain 24 fields; cells with `None` raw/shrunk values render as the empty string in both columns; rows sorted by `shrunk_score_play` descending with empty values at the end). (Depends on T006 — same file.) [Research D-15, FR-013a]

- [X] T008 [US2] Extend `tests/unit/sealed/application/test_train_encoder.py` with weighted-MSE + MLM mask + stratification tests: `_per_batch_weighted_mse` correctness on a single-card batch and on a mixed batch where one head has all-zero head_mask (returns 0 contribution, no NaN); `_draw_mlm_mask` masks at approximately the expected probability and never selects `[PAD]`/`[UNK]`/`cardname`/`[MASK]` positions; `_split_cards` stratification falls back through the FR-018 chain (`score_play` empty → `score_draw` → … → catch-all stratum) on a synthetic degenerate label distribution. (Depends on T007 — same file.) [Research D-10, D-11, D-4]

- [X] T009 [US2] Extend `tests/unit/sealed/application/test_train_encoder.py` with corpus-consistency tests: `_check_corpus_consistency` raises `CorpusInconsistencyError` naming up to 20 missing cards plus the total count; the error points the user at `python -m price_predictor convert`. (Depends on T008 — same file.) [FR-023d]

- [X] T010 [P] [US2] Update `tests/unit/sealed/domain/test_encoder_model.py` to assert (a) `forward()` returns a dict with keys `score_play`, `score_draw`, `played_rate`, `cast_lift`, `color_lift`, `mlm_logits` of the right shapes; (b) `encode()` (inference path) still returns `(B, 2*d_model)` unchanged; (c) `regression_heads.*` and `mlm_head.*` keys are present in the live model's `state_dict`. (Prior art: existing tests for `_encode_and_pool` shape.) [Research D-8, D-9]

- [X] T011 [P] [US2] Update `tests/unit/sealed/infrastructure/test_encoder_store.py` round-trip test to construct a `SealedEncoderModel` with the new heads + MLM head populated, save it, then assert the saved file's `model_state_dict` contains only `token_encoder.*` / `card_encoder.*` keys (no `regression_heads.*`, no `mlm_head.*`). (Prior art: existing prefix-filter test — extend to cover the new prefixes.) [FR-020]

- [X] T012 [P] [US2] Update `tests/integration/sealed/test_train_encoder_smoke.py` to use a tiny synthetic `cards-played.txt` + tiny corpus + tiny vocab (containing `[MASK]`); run `train-encoder` for ~3 epochs; assert (a) `latest.pt` loads via `SealedEncoderStore.load_encoder` without error, (b) the saved state_dict contains no head/MLM keys, (c) per-epoch log lines include the regression-only and MLM-only loss breakdowns, (d) the new `cards-win-rates.txt` header is present at line 1, (e) the saved `train_config` records no `init_from` / source-checkpoint field — guard against a future regression where `train-encoder` silently loads weights instead of training from random init (FR-016), (f) the LR schedule shape matches FR-022: with `total_steps = epochs × batches_per_epoch`, the optimizer's LR at step `ceil(0.05 × total_steps)` equals `--lr` to within float tolerance, the LR at every step `≥ ceil(0.05 × total_steps)` equals `--lr` (constant tail, no decay), and the LR at step 0 is strictly less than `--lr` (warmup is active). Drive the schedule directly via the `LambdaLR` (or equivalent) the trainer constructs; do not require a real 3-epoch run for (f).

- [X] T013 [P] [US2] Add CLI argparse coverage for the two new flags in `tests/unit/sealed/infrastructure/test_cli_train_encoder_argparse.py`: `--mlm-weight` defaults to 0.1 and accepts user values; `--mlm-mask-prob` defaults to 0.15 and accepts user values; both are surfaced as fields on the resolved `TrainEncoderConfig`. (Prior art: existing argparse tests in this file for `--shrinkage-k` etc.)

### Implementation for User Story 2

- [X] T014 [P] [US2] Replace the single `regression_head: nn.Sequential(Linear, Sigmoid)` on `SealedEncoderModel` with `regression_heads: nn.ModuleDict({"score_play": Linear(2*d_model, 1), "score_draw": Linear(2*d_model, 1), "played_rate": Linear(2*d_model, 1), "cast_lift": Linear(2*d_model, 1), "color_lift": Linear(2*d_model, 5)})` in `src/sealed/domain/encoder_model.py`. (Prior art: existing `regression_head` attribute is being replaced — no parallel concept.) [Research D-8, FR-014]

- [X] T015 [P] [US2] Add `mlm_head: nn.Linear(d_model, vocab_size)` to `SealedEncoderModel` in `src/sealed/domain/encoder_model.py`. The head must read the contextualized token sequence (output of the transformer-layer stack, before the pool layer); add a private path on `_CardEncoderBlock` (or `SealedEncoderModel._encode_and_pool`) to expose the pre-pool sequence to the caller. (Prior art: existing `_encode_and_pool` returns post-pool only — extend to return both.) [Research D-9, FR-015a]

- [X] T016 [US2] Update `SealedEncoderModel.forward(input_ids, attention_mask)` in `src/sealed/domain/encoder_model.py` to return a dict `{score_play: (B,), score_draw: (B,), played_rate: (B,), cast_lift: (B,), color_lift: (B, 5), mlm_logits: (B, T, vocab_size)}` with range-matched activations (tanh on the four signed heads + color_lift, sigmoid on `played_rate`). Keep `encode()` unchanged (it still returns the pooled vector via the no-grad path). Depends on T014, T015.

- [X] T017 [P] [US2] Update `SealedEncoderStore` docstring + `_ENCODER_PREFIXES` comment in `src/sealed/infrastructure/encoder_store.py` to name `regression_heads.*` and `mlm_head.*` as the prefixes filtered out at save time; update the `RuntimeError` message in `load_encoder` to point readers at FR-020 when a leak is detected. No code-behavior change. (Prior art: existing `_ENCODER_PREFIXES` filter — reuse; documentation only.)

- [X] T018 [P] [US2] Replace the v1 `CardLabel` dataclass in `src/sealed/application/train_encoder.py` with two new frozen dataclasses: `CardCounters` (4 primary + 4 `@play` ints + four `dict[str, int]` per-color counter dicts) and `CardLabels` (card_name, counters, raw + shrunk for the 9 heads as `float | None`). Define `CardLabelMap = dict[str, CardLabels]`; remove the old `WinnabilityMap` alias's references to the v1 schema. (Prior art: `CardLabel` is being replaced — no parallel concept.) [Research D-13, data-model.md]

- [X] T019 [P] [US2] Add the `_colors_from_mana_cost(line: str) -> set[str]` helper to `src/sealed/application/train_encoder.py`: regex over `\{[^}]+\}` symbols, union-collect WUBRG letters per symbol; return empty set when no symbol contains a letter. (No prior art — new helper; documented in research D-16.) [FR-010b]

- [X] T020 [US2] Replace `_aggregate_counts` in `src/sealed/application/train_encoder.py` with `_aggregate_pass_one(rows) -> dict[str, CardCounters]` that fills the 4 primary + 4 `@play` counters per card by iterating both sides of every game and bucketing by `winner` and by `starter`. Initialize the per-color counter dicts to zero — pass 2 fills them. Depends on T018. [Research D-13, FR-010a]

- [X] T021 [US2] Add `_aggregate_pass_two(rows, counters, locator) -> None` in `src/sealed/application/train_encoder.py` that iterates `cards-played.txt` a second time. For each game, build each side's deck-color set as the union over `_colors_from_mana_cost` of every card in the deck (card → cost line via `ConvertedCardLocator.load_text(name).mana_cost_line()` cached in a per-run dict); for each (card, color X in deck) pair, increment the four per-color counter dict entries on the card's `CardCounters`. Depends on T018, T019, T020.

- [X] T022 [US2] Update `_check_corpus_consistency` in `src/sealed/application/train_encoder.py` so it runs *between* pass 1 and pass 2 (FR-023d): the check reads the set of card names from pass 1's counter dict (not from the not-yet-built label map). The error message and capping behavior are unchanged. Depends on T020. (Prior art: existing function — reuse with a slight signature change.)

- [X] T023 [US2] Replace `_build_winnability_map` in `src/sealed/application/train_encoder.py` with `_build_label_map(counters: dict[str, CardCounters], shrinkage_k: float) -> CardLabelMap` that, per FR-011, computes 9 raw + 9 shrunk labels per card (cells with zero slice denominator stored as `None`; cards with `wins_when_in_deck + losses_when_in_deck == 0` excluded entirely per FR-012). Depends on T018, T020, T021. [FR-011, FR-012]

- [X] T024 [US2] Replace `_write_win_rates` in `src/sealed/application/train_encoder.py` with the FR-013a 24-column writer: header row first; one data row per included card with the column order from `contracts/files.md`; floats formatted to five decimals; `None` cells render as the empty string in both raw and shrunk columns; rows sorted by `shrunk_score_play` descending with empty `shrunk_score_play` values pushed to the end. Depends on T023. [Research D-15, FR-013a]

- [X] T025 [US2] Update `_split_cards` in `src/sealed/application/train_encoder.py` to (a) stratify on `score_play` quartile when present, (b) fall back through the FR-018 chain (`score_draw` → `cast_lift` → `color_lift_W/U/B/R/G`) when `score_play` is `None`, (c) place fully-degenerate cards in a single catch-all stratum. The 20% val split + `random_seed=42` shuffle inside each stratum stays. Depends on T023. [Research D-4, FR-018]

- [X] T026 [US2] Update `_WinnabilityDataset.__getitem__` in `src/sealed/application/train_encoder.py` to return `(input_ids, attention_mask, labels: Tensor[(9,)], weights: Tensor[(9,)], head_mask: Tensor[(9,)])`. `labels` carries the 9 shrunk values (with 0.0 substituted at empty cells); `weights` carries the FR-017a per-head weights computed from the card's `CardCounters` and the run's `shrinkage_k`; `head_mask` is 1.0 where the cell is non-empty and 0.0 elsewhere. Update `_WinnabilityDataset.__init__` to accept the per-card label map (now `CardLabelMap`). The dataset MUST preserve the v1 FR-014a behavior of stripping the `name:` line from each converted card before tokenizing — add a regression assertion in `test_train_encoder.py` (or T010) that the tokenized input never carries the cardname token at the head position, so the multi-head refactor cannot silently re-introduce a name→label shortcut. Depends on T018, T023.

- [X] T027 [P] [US2] Add `_draw_mlm_mask(input_ids, attention_mask, mask_prob, mask_token_id, special_token_ids) -> tuple[Tensor, Tensor]` to `src/sealed/application/train_encoder.py`: returns `(masked_ids, mask_positions)` where `masked_ids[mask_positions] == mask_token_id` and the original ids at those positions are returned for the CE target. Eligible positions: real (`attention_mask == 1`) AND not in `special_token_ids`. (No prior art — new helper; documented in research D-10.) [FR-014b]

- [X] T028 [P] [US2] Add `_per_batch_weighted_mse(predictions: dict, labels: Tensor, weights: Tensor, head_mask: Tensor) -> Tensor` to `src/sealed/application/train_encoder.py`: implements the FR-017 / D-11 per-head per-batch sum-to-1 weighted average across the 9 heads; sums the four signed-head terms unweighted plus `(1/5)` × the five color-lift terms; returns a scalar `L_reg`. If a head's total batch weight is zero (no card contributes to that head), the head's term MUST be exactly zero — short-circuit the head before the divide rather than relying on a denominator clamp, so a future numerator drift cannot leak through a tiny epsilon. (No prior art — new helper.)

- [X] T029 [US2] Rewrite `_train_epoch` in `src/sealed/application/train_encoder.py` to (a) draw an MLM mask per batch via T027; (b) run `model.forward(masked_ids, attention_mask)`; (c) compute `L_reg` via T028 and `L_mlm` as cross-entropy at masked positions only (denominator = `mask.sum().clamp(min=1)`); (d) backprop `L_reg + mlm_weight * L_mlm`; (e) call `clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()`. Track per-component losses for the per-epoch log line. Depends on T016, T026, T027, T028. [Research D-14, FR-017, FR-022]

- [X] T030 [US2] Rewrite `_eval_loss` in `src/sealed/application/train_encoder.py` to compute the same full loss (`L_reg_val + mlm_weight * L_mlm_val`) on the val set and return the scalar used by `_BestCheckpoint.update` and the patience counter. The MLM mask is drawn at val time too (with the same `--mlm-mask-prob`) so val numbers stay comparable across epochs. Depends on T027, T028. [Research D-12, FR-019]

- [X] T031 [US2] Update `TrainEncoderConfig` in `src/sealed/application/train_encoder.py`: add `mlm_weight: float = 0.1` and `mlm_mask_prob: float = 0.15` fields. Plumb both through `run(config)` into `_train_epoch` / `_eval_loss`. (Prior art: existing dataclass — extend.)

- [X] T032 [US2] Extend `_run_preflight` in `src/sealed/application/train_encoder.py` to read the vocab and assert `[MASK]` is present; raise `_PreFlightError` with exit code 2 and a message naming `python -m sealed build-vocab` if absent (FR-023a). Depends on T001 (foundational).

- [X] T033 [US2] Update `_log` per-epoch line format in `src/sealed/application/train_encoder.py` to include the `(reg=…, mlm=…)` breakdown alongside the full `train_loss` and `val_loss` numbers used for best-checkpoint selection. Depends on T029, T030.

- [X] T034 [P] [US2] Add `--mlm-weight` (type=float, default 0.1) and `--mlm-mask-prob` (type=float, default 0.15) flags to `_build_train_encoder_parser` in `src/sealed/infrastructure/cli.py`; thread both through `run_train_encoder` into the `TrainEncoderConfig` constructor. Depends on T031.

**Checkpoint**: At this point, US2 is fully functional. Running `python -m sealed train-encoder` against a populated `cards-played.txt` produces a multi-head + MLM-trained encoder whose saved file has only encoder weights, the new `cards-win-rates.txt` schema, and per-epoch logs that report both the regression and MLM components.

---

## Phase 5: User Story 3 — Sealed scorer consumes the sealed-trained encoder (Priority: P1) ✅ ALREADY DELIVERED

**Goal**: `train-scorer` and `encode-cards` default to
`models/sealed/encoder/latest.pt` for `--encoder-checkpoint`, with a
clear error pointing at `train-encoder` when missing.

**Independent Test**: Train any sealed encoder, then run `encode-cards`
followed by `train-scorer` with no `--encoder-checkpoint` flag and
verify the saved scorer's `train_config` records the sealed-encoder
path.

**Status**: All FR-024 through FR-027 are satisfied by the v1 default
flips (`_ENCODE_CARDS_DEFAULT_ENCODER` in
`src/sealed/infrastructure/cli.py:621`,
`TrainScorerConfig.encoder_checkpoint` default in
`src/sealed/application/train_scorer.py`, the missing-file error wired
in `run_train_scorer` and `run_encode_cards`). The spec amendments in
016 do not change downstream integration. **No new tasks.**

---

## Phase 6: User Story 4 — Build a sealed-specific vocabulary (Priority: P2)

**Goal**: `python -m sealed build-vocab` emits a vocab file that
contains the `[MASK]` token alongside `[PAD]`, `[UNK]`, and `cardname`,
without modifying the price-predictor vocab.

**Independent Test**: Run `python -m sealed build-vocab` against
`output/cardsfolder/`; the resulting `models/sealed/encoder/vocab.txt`
contains a `[MASK]` line at a deterministic position; no other vocab
file is touched.

**Status**: The original `build-vocab` subcommand is already in place
from v1 — only the `[MASK]` seeding is new for v2, and that change
lives in the foundational T001 (because US2 also depends on it). No
additional implementation is required. The remaining work for US4 is
verification.

- [X] T035 [US4] After T001/T002/T003 land, run `python -m sealed build-vocab` end-to-end against `output/cardsfolder/` and verify the emitted `models/sealed/encoder/vocab.txt` contains exactly one line `[MASK]` between `cardname` and the first domain-term token; record the verification in the PR description. (Manual quickstart-style validation; no code change.)

**Checkpoint**: US4 verified — every fresh `build-vocab` run produces a
vocab usable by US2's `train-encoder`.

---

## Phase 7: User Story 5 — Tune low-n shrinkage for noisy labels (Priority: P3)

**Goal**: `--shrinkage-k` shifts low-observation cards' labels visibly
across all 9 heads while leaving high-observation cards' shrunk values
within a few thousandths of their raw counterparts.

**Independent Test**: Run `python -m sealed train-encoder --shrinkage-k
0` and `python -m sealed train-encoder --shrinkage-k 20` against the
same corpus; diff `output/sealed/cards-win-rates.txt` between the two
runs.

**Status**: The `--shrinkage-k` flag is already present and fully wired
through `TrainEncoderConfig` from v1; the v2 label arithmetic
(implemented under US2) automatically applies the same `k` to all 9
heads via T023's `_build_label_map`. No production-code work is
required beyond US2.

- [X] T036 [US5] Add an integration test in `tests/integration/sealed/test_train_encoder_shrinkage.py` (new file) that runs `train-encoder` twice on a tiny synthetic `cards-played.txt` (one card with two observations, one card with a thousand observations), once with `--shrinkage-k 0` and once with `--shrinkage-k 20`, and asserts: (a) the low-observation card's shrunk values across all 9 head columns differ measurably between the two runs; (b) the high-observation card's shrunk values differ from raw by less than 0.005 in the `k=20` run. Mark `@pytest.mark.integration`. (Prior art: existing `test_train_encoder_smoke.py` for fixture style.) [SC-005]

**Checkpoint**: US5 verifiable — SC-005 is exercised by an integration
test.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T037 [P] Update `CLAUDE.md`'s `train-encoder` description to reflect (a) the 9 regression heads + MLM auxiliary head, (b) the new `--mlm-weight` / `--mlm-mask-prob` flags, (c) the new 24-column `cards-win-rates.txt` schema, (d) the AdamW + max-norm 1.0 + 5%-warmup-then-constant LR schedule + full-validation-loss best-checkpoint selection. Keep entries terse — match existing style.

- [X] T038 [P] Update `CLAUDE.md`'s `cards-win-rates.txt` paragraph (currently describes the v1 5-column schema) to describe the FR-013a 24-column schema with header row + sort by `shrunk_score_play`.

- [X] T039 Run `ruff check src/ tests/` on the entire repo and fix any issues introduced by this iteration.

- [X] T040 Run the full fast suite (`pytest -m "not integration"`) and the integration suite (`pytest tests/integration/sealed/`) and verify both pass.

- [ ] T041 Walk the `quickstart.md` end-to-end on a small corpus (Steps 2–4 are the v2-relevant ones; Step 1 is unchanged). Confirm the verification commands produce the expected outputs.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: empty.
- **Foundational (Phase 2)**: T001 → T002, T003 (T002 and T003 are [P]; both depend only on T001).
- **US1 (Phase 3)**: no tasks.
- **US2 (Phase 4)**: depends on Phase 2 completion (T001 must land before any US2 implementation can validate).
- **US3 (Phase 5)**: no tasks.
- **US4 (Phase 6)**: depends on Phase 2 completion + T035 is verification-only.
- **US5 (Phase 7)**: depends on US2 (T036 needs the label arithmetic from T023).
- **Polish (Phase 8)**: depends on US2 + US4 + US5 completion.

### User Story Dependencies

US1 and US3 are pre-shipped; they have no dependencies. US2 is the
critical-path P1. US4 and US5 are optional priority tiers (P2/P3) but
both share the foundational T001 and US2's implementation respectively.

### Within User Story 2

Tests are ordered by file:

- T004 → T005 → T006 → T007 → T008 → T009 (all in
  `test_train_encoder.py`, sequential because same file).
- T010, T011, T012, T013 are [P] — different test files.

Implementation has these dependency chains:

- Domain: T014 [P] T015 → T016 (forward shape).
- Aggregator + labels: T018 → T020 → T022 (corpus check) → T021
  (pass 2) → T023 (labels) → T024 (writer) → T025 (split) → T026
  (dataset).
- Helpers: T019 (colors), T027 (mask draw), T028 (weighted MSE) — all
  [P], independent.
- Loop: T029 (train epoch) needs T016, T026, T027, T028; T030 (eval)
  needs T027, T028.
- Config + CLI: T031 (config fields), T032 (preflight depends on T001),
  T033 (log line), T034 (CLI flags) — mostly independent.

Within US2, the suggested sequencing is:

1. Tests first: T004 → T005 → T006 → T007 → T008 → T009; then T010,
   T011, T012, T013 in parallel.
2. Domain layer: T014, T015 [P] → T016.
3. Application helpers: T018, T019, T027, T028 in parallel.
4. Aggregation chain: T020 → T022 → T021 → T023 → T024 → T025 → T026.
5. Training loop: T029 → T030.
6. CLI + config: T031, T032, T033, T034 in parallel where possible
   (T032 depends on T001; T034 depends on T031).
7. T017 [P] (docstring update) anywhere after T014/T015 land.

### Parallel Opportunities

Within Phase 2: T002 ‖ T003 once T001 is in.

Within US2 (after tests are in place):

- T014 ‖ T015 (different attributes on the same file but separate
  edits)
- T018 ‖ T019 ‖ T027 ‖ T028 (all add new private functions in
  `train_encoder.py` — sequential edits to the same file but logically
  independent; the [P] marker reflects logical independence, the actual
  file edits happen in sequence)
- T010 ‖ T011 ‖ T012 ‖ T013 (different test files)

Within Phase 8: T037 ‖ T038 (different sections of CLAUDE.md).

---

## Parallel Example: User Story 2 helper functions

Once T018 and T019 land, the loss helpers are independent:

```bash
# Same file (train_encoder.py), but logically independent additions:
Task: "T027 [US2] Add _draw_mlm_mask helper"
Task: "T028 [US2] Add _per_batch_weighted_mse helper"
```

Once these helpers are in place plus T016 (model forward) and T026
(dataset), T029 and T030 can both consume them.

---

## Implementation Strategy

### MVP Scope (User Story 2 only)

US2 is the load-bearing story for this iteration. US1 and US3 are
pre-shipped; US4 and US5 are small follow-ons. The MVP is:

1. Phase 2 (foundational): T001, T002, T003.
2. Phase 4 (US2): T004…T034.
3. **Stop and validate**: run `python -m sealed train-encoder` against
   a real `cards-played.txt`; inspect the new `cards-win-rates.txt`;
   confirm the saved encoder loads via `SealedEncoderStore`; confirm
   `train-scorer` (US3, already wired) consumes it.

### Incremental Delivery

1. Foundational (T001–T003) — produces vocabularies that include
   `[MASK]`.
2. US2 (T004–T034) — multi-head + MLM encoder pretraining works
   end-to-end.
3. US4 (T035) — manual verification of the foundational `[MASK]`
   seeding.
4. US5 (T036) — automated SC-005 integration test.
5. Polish (T037–T041) — CLAUDE.md updates, lint, full-suite pass,
   quickstart walkthrough.

Each step adds value without breaking the previous ones; US3's
already-shipped default flips continue to work because the new sealed
encoder's checkpoint shape is identical to v1's (same prefix layout).

---

## Notes

- `[P]` tasks = different files OR logically-independent additions to
  the same file (the latter still serialize on the file but can be
  authored in any order).
- `[Story]` label maps task to a specific user story for traceability.
- Each user story is independently completable; US2 is the only story
  with substantive new code.
- Per Constitution Principle I, every behavioral change in this
  iteration ships with a unit or integration test (T002–T013, T036).
- Per Constitution Principle VII, every new entity / helper task above
  cites the prior art it reuses or replaces (see plan.md § Codebase
  Survey and research.md). The new helpers `_colors_from_mana_cost`,
  `_draw_mlm_mask`, and `_per_batch_weighted_mse` have no prior art in
  the codebase; this is documented in research D-16, D-10, and D-11
  respectively.
- Commit after each phase or logical group; do not amend across
  user-story boundaries.
- Avoid: same-file conflicts inside a single PR (T004–T009 must be
  authored in series), cross-story dependencies that break US2's
  independence from US4/US5.
