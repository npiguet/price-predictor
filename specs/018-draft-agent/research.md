# Research: Draft agent — imitation policy + critic (generation 1)

**Feature**: `018-draft-agent` | **Date**: 2026-05-31
**Inputs**: `spec.md`, normative `specs/2026-05-28-draft-agent.md`, rationale
`experiments/2026-05-30-draft-agent-design.md`

This document resolves the open technical questions and records the
constitution-mandated codebase survey (Principle VII). The normative spec and
the design rationale already settle the *what* and most of the *why*; the
survey below grounds the *how* in concrete reused symbols.

## Codebase Survey

The feature lives in a **new top-level `draft` package** that mirrors `sealed`
and reuses its scorer, picker, greedy builder, embedding layout, card locator,
checkpoint plumbing, and the Forge supervisor/worker pattern. Almost nothing
here is genuinely new infrastructure — the new parts are the draft *state
representation*, the two-headed *model*, the *training loop*, and a Java
*draft worker*.

### Overlapping domain vocabulary

| Existing concept | File / symbol | Decision |
|---|---|---|
| `PickerConfig` / `PickerModel` (SAB trunk + per-card head + aux head, `embedding_dim`→`d_model`, optional input projection, fail-fast `d_model % n_heads`) | `sealed/domain/picker_model.py` | **Reuse as the template** for `DraftAgentConfig` / `DraftAgentModel`. The draft model is the same SAB-trunk-over-a-card-set shape with a per-`PACK`-token policy head (identical to the picker's per-card head) plus a `CONTEXT`-token critic head. The `d_model`/`n_heads` validation and optional-input-projection logic are copied verbatim in spirit. New sibling justified: the draft model adds typed tokens, recency/context embedding tables, and a second head — too divergent to extend `PickerModel` in place, but it follows its conventions exactly. |
| `SAB` self-attention block | `sealed/domain/scorer_model.py:26` | **Reuse** — import `from sealed.domain.scorer_model import SAB`, exactly as `picker_model.py:20` does. No re-implementation. |
| `ScorerConfig` / `SetTransformerScorer` | `sealed/domain/scorer_model.py` | **Reuse as a frozen labeler** via `ScorerStore`. The draft critic is a *separate* head; it does not subclass the scorer. The scorer's `score_decks` batched-forward helper (`evaluate_scorer.py:45`) is reused to label decks. |
| Deck "score" / pod-relative reward | `match_outcomes`, scorer outputs | **Extend** — the scorer's scalar is the per-seat `deck_score`; the new *pod-relative reward* (leave-one-out mean subtraction) is computed at load time in the trainer, not stored. New concept, but a thin derived value, not a parallel entity. |
| "pool" / drafted pool | sealed pools (`pool_file_reader.py`, `parse_pools`) | **Distinguish** — a sealed *pool* is 6 boosters' worth of cards in one file line; a draft *seat pool* is the 45 cards one seat drafted, reconstructed from the booster transcript. Same builder code path (`load_pool_embeddings` + picker/greedy) consumes both. Kept as distinct record shapes (sealed `pools.txt` vs draft `drafts.jsonl`) because their provenance and geometry differ. |

### Adjacent prior art

| Sub-problem | Existing solution | Decision |
|---|---|---|
| Forge supervisor that spawns/monitors/restarts a Java worker, stamps a `run_id`, reports throughput, handles SIGINT | `sealed/application/match_outcomes.py` (`MatchOutcomeSupervisor`) | **Reuse the pattern** for `generate-draft-data`'s supervisor. Gen-1 needs a single worker (not a pool), but the crash-restart + run-id + line-count-status loop is the same. The new supervisor additionally **reads** worker stdout (filter for `<<DRAFT-EVENT-JSON>>`), which the match supervisor does not (it discards stdout). |
| Launch a Forge JVM worker, build classpath, kill process tree | `price_predictor/infrastructure/forge_jvm.py` (`build_jvm_command`, `build_forge_classpath`, `run_forge_worker`, `kill_process_tree`) | **Reuse** directly — the draft worker connector calls these exactly like `MatchWorkerConnector`/`PoolConnector`. |
| Build a 40-card deck from a pool with the picker (forward → spell-quota walk → basics) | `sealed/application/pick_decks.py` + `deck_assembly.py` (`load_pool_embeddings`, `assemble_full_deck`) + `picker_model.decompose_picks` + `manabase.compute_basic_lands` | **Reuse** — the supervisor builds each seat's deck with this exact path. The picker runs on a 45-card pool unchanged (length-agnostic set transformer). |
| Build a 40-card deck from a pool with SA | `sealed/domain/greedy_deck_builder.py` (`GreedyDeckBuilder`) | **Reuse** for `--build-method greedy` and for the §5.3 builder-validation diagnostic's SA reference. |
| Score decks with the frozen scorer in one batched pass | `sealed/application/evaluate_scorer.py:45` (`score_decks`) | **Reuse** to label each seat's deck. |
| Load per-card `.npz` embeddings by Forge name, tolerate missing | `sealed/infrastructure/converted_card_locator.py` (`ConvertedCardLocator.load_embedding/load_text`) | **Reuse** — the loader and the supervisor both resolve card vectors through it; missing-embedding warning (≤20 names + total) mirrors existing pipelines. |
| Card-embedding feature layout (`FEATURE_COUNT`, `is_land_embedding`) | `sealed/domain/card_embedding_layout.py` | **Reuse** — `embedding_dim` (the `.npz` width) is read from a card vector; `is_land_embedding` flags drive the spell-quota walk during labeling. |
| Checkpoint save/load with dataclass config, `{timestamp}.pt` + `latest.pt` | `price_predictor/infrastructure/torch_checkpoint.py` (`save_checkpoint`/`load_checkpoint`); `PickerStore`/`ScorerStore` wrappers | **Reuse** — a new `DraftAgentStore` mirrors `PickerStore` (model weights + config + epoch + best-val + training metadata, here also the critic-target standardization mean/std). |
| AdamW + linear-warmup-then-constant LR schedule, per-group max-norm clipping | `train_encoder.py:1030` (`LambdaLR` warmup), `train_picker.py` clip helper | **Reuse the recipe** (same `LambdaLR` lambda; same `clip_grad_norm_` per group). |
| Resume / bootstrap-from-checkpoint guard, best-vs-latest persistence, early-stop on patience | `train_picker.py` `_resume_or_build_picker`, `_should_stop`; `train_scorer.py`; `train_encoder.py` | **Follow the convention** (see Third-instance check). |
| Pick-order / wheel geometry | *none — new* | The booster-transcript→state reconstruction (FR-016/FR-031) is genuinely new domain logic; it lives in `draft/domain`. |
| Resume an append-only output by counting/truncating partial lines | `pool_file_reader.count_complete_lines_and_truncate_partial`; `cards_played_reader` trailing-partial tolerance | **Reuse the idea** — JSONL reader tolerates a trailing partial line; `--resume` counts existing records. |

### Convention alignment

The feature mirrors **`sealed`** in every structural respect:

- **Layout**: `src/draft/{domain,application,infrastructure}/` + `__main__.py`
  + `infrastructure/cli.py`, exactly like `src/sealed/` (`sealed/__main__.py`,
  `sealed/infrastructure/cli.py` with `add_parser`/`set_defaults(func=…)`).
- **Dependency direction**: `draft` imports from `sealed` and
  `price_predictor`; never the reverse (FR-002), matching the existing
  one-way `sealed → price_predictor` rule documented in CLAUDE.md.
- **Domain purity**: model/state/geometry in `draft/domain` (no torch-free
  rule — `sealed/domain` already imports torch, so the draft model lives in
  `domain` like `picker_model.py`/`scorer_model.py`); supervisor, worker
  connector, stores, CLI in `application`/`infrastructure`.
- **Java worker**: `DraftWorkerMain` joins the existing
  `forge-connector` main classes (`PoolMain`, `MatchWorkerMain`,
  `ValidationWorkerMain`), reusing `ForgeEnvironmentInitializer`, the
  `MatchGenerator.computeEligibleSets()` random-set helper, and the
  sentinel-flush stdout style.
- **Test style**: fast fixture-based unit tests under `tests/unit/draft/`
  (mirroring `tests/unit/sealed/`); any Forge-dependent worker test tagged
  integration. Java tests under `forge-connector/src/test/java` with
  `@Tag("integration")` for Forge-dependent ones.

### Third-instance check

**Shared training scaffolding.** `train_encoder`, `train_scorer`, and
`train_picker` already share the resume/bootstrap/warmup/clip/best-checkpoint
/early-stop skeleton, and `train_picker.py:7` carries an explicit
`TODO(shared-trainer)` that weighs extraction and concludes it is *premature
today* because the three diverge on loss type, metric direction, and per-step
dataset shape — and resolves: "If a fourth REINFORCE-style trainer arrives,
extract the common scaffolding then." `train-draft-agent` is a fourth trainer
but **not** REINFORCE-style (it is supervised: CE + MC-regression MSE), and it
diverges again on dataset shape (per-`(draft,seat,pick)` typed-token states)
and on having two heads on two seat-subsets.

Decision: **follow the convention without extracting now.** Per Simplicity-First
(Principle II: an abstraction needs three *aligned* use cases) and the existing
author's recorded judgment, a generic training loop over four loss/metric/shape
variants would be speculative. The draft trainer copies the small, stable
helpers (warmup `LambdaLR` lambda, per-group clip, resume-guard shape) by
following the pattern. **Follow-up task (non-blocking, to land in `tasks.md`):**
re-evaluate extracting a `train_common` helper module (warmup schedule +
per-group clip + resume/bootstrap guard + best/latest persistence) once the
draft trainer is in place and the shared surface is concrete across four
trainers — but only the genuinely identical pieces, leaving the loss/metric
bodies per-trainer.

No other sub-problem reaches a third hand-coded instance: deck-building (picker
/greedy/score) is already centralized in `deck_assembly.py` + `score_decks` and
is reused, not re-copied.

## Technical decisions (resolving NEEDS CLARIFICATION)

All architecture-level choices are fixed by the normative spec; the items below
are the implementation-level resolutions the plan needs.

### D1 — Driving Forge's draft AI for all 8 seats (Java worker)

- **Decision**: `DraftWorkerMain` uses Forge's `BoosterDraft`
  (`forge.gamemodes.limited.BoosterDraft`) + `LimitedPlayer`
  (`getAllPlayers()`, `getPlayer(i)`, `getDeck()`, `getLastPick()`,
  `nextChoice()`/`setChoice()`/`chooseCard()`). Forge's AI fills every seat;
  the worker records, per booster and per pick, the card each seat took, in
  pick order. Random-override agents (`forge-r30`/`forge-r100`) replace a
  fraction of a seat's AI picks with a uniform-random legal pick from that
  seat's current pack.
- **Rationale**: `BoosterDraft` already models the full 8-seat pod, pack
  passing direction (L in packs 1 & 3, R in pack 2), the wheel, and AI
  picking. Reusing it avoids re-implementing draft topology. The
  per-seat `getDeck()` yields the drafted pool the supervisor needs to build a
  deck from.
- **Alternatives considered**: hand-rolling pod topology in Java (rejected —
  duplicates Forge's tested draft engine); driving the draft from Python via
  per-pick IPC (rejected — gen-1 model is not a participant, so there is no
  need for per-pick round trips; one transcript per completed draft suffices).
- **Open empirical item (non-blocking)**: the exact `BoosterDraft` factory and
  set-restriction entry points (`createDraft(LimitedPoolType, …)`,
  `setPodSize`, per-booster set codes for `--set` vs random) are confirmed
  during implementation against the sibling `../forge` checkout; the worker
  records `set_code` **per booster** (FR-015) so Chaos/mixed-set drafts and the
  random-per-draft default both serialize identically.

### D2 — Worker → supervisor transport

- **Decision**: sentinel-prefixed stdout. Worker prints one flushed line per
  completed draft: `<<DRAFT-EVENT-JSON>>` + compact newline-free JSON
  (boosters + per-seat agent ids, **no** deck/score). Supervisor filters for
  the sentinel, defensively `json.loads` the suffix, skips parse failures,
  pipes worker stderr to a log file, ignores Forge's incidental stdout.
- **Rationale**: matches the design rationale's chosen transport (robust
  against Forge log noise, simple, crash-tolerant at data-gen scale). Mirrors
  the existing forge-worker stdout-sentinel idioms.
- **Alternatives considered**: file-based pending area (extra FS coordination),
  Forge-logging-to-stderr reconfiguration (invasive/fragile) — both rejected in
  the rationale.

### D3 — Where the picker/scorer run

- **Decision**: on the **Python supervisor** side. The worker emits only the
  transcript; the supervisor reconstructs each seat's 45-card pool from the
  transcript, builds a deck with the picker (default) or SA (`--build-method`),
  scores it with the frozen scorer, and writes the completed record.
- **Rationale**: gen-1 model is not in the draft loop, so no per-pick IPC; the
  picker/scorer are PyTorch and already run in Python everywhere else. Keeps
  the Java worker zero-ML.

### D4 — State reconstruction geometry (the genuinely new logic)

- **Decision**: implement the FR-016/FR-031 conventions as a pure
  `draft/domain` function: given a parsed record and a target
  `(seat s, pack p, pick i)`, compute `PACK`/`POOL`/`PASSED`/`TAKEN` instance
  sets and per-instance `(packs_ago, pick_ago)` by walking the boosters the
  seat saw, applying wheel-diff and pack-end-flush `PASSED→TAKEN` transitions.
- **Rationale**: this is the one piece with no prior art; isolating it in a
  pure, table-tested domain function (SC-002 round-trip) is the cheapest way to
  get it right and keep the loader/model thin. Determinism + reproducibility
  (Principle III).
- **Validation**: golden tests assert reconstructed sets for hand-worked
  records (incl. the wheel and the pack-end flush) and that every observed
  instance is in exactly one type (FR-018).

### D5 — Feature widths and `d_model` divisibility

- **Decision**: default `d_model = embedding_dim + 4 + d(packs_ago) +
  d(pick_ago)` with `d(packs_ago)=4`, `d(pick_ago)=8` (rationale's suggested
  sizes). Validate `d_model % n_heads == 0` at startup (reuse the picker's
  `PickerArchitectureError` style). A non-default `--d-model` inserts one
  `Linear(concat_width, d_model)` (picker convention).
- **Rationale**: keeps the `.npz` width and the four type dims persistent in
  the residual stream (no projection) per the rationale; the recency tables are
  tiny. `embedding_dim` is already a multiple of common head counts; the added
  `4+4+8=16` keeps the default total a multiple of 8 (default `n_heads`).
- **Note**: the actual `embedding_dim` is read from a sample `.npz` at startup
  (as `pick_decks`/`build_decks` do), and recorded in the checkpoint config so
  reload reconstructs the exact architecture; `P` (pack size) is read from the
  corpus and recorded (sizes the `pick_number`/`pick_ago` tables).

### D6 — Critic target standardization

- **Decision**: z-score the leave-one-out pod-relative reward to zero
  mean/unit variance over the **training split** (clarification 2026-05-31);
  store mean/std in the checkpoint; de-standardize at inference back to raw
  scorer-score space. Both loss weights default to `1.0`.
- **Rationale**: the imitation CE (≈ ln P nats, O(1)) and the raw critic MSE
  (Bradley-Terry score differences, often ≪ 1) live on different scales;
  standardizing the target makes the 1:1 default balance meaningful and keeps
  the critic learning on every non-failed seat. Recorded mean/std preserve
  reproducibility (Principle III) and a raw-space inference contract.

### D7 — Builder-validation diagnostic delivery

- **Decision**: a one-off **script** (≈40 lines), not a CLI subcommand
  (FR-042), reusing `load_pool_embeddings` + picker + `GreedyDeckBuilder` +
  `score_decks`. Reads seat pools from an existing `drafts.jsonl` (or fresh
  pools). Reports picker-vs-SA Spearman, SA−picker gap median/spread, and the
  SA-vs-SA reference correlation across independent SA restarts.
- **Rationale**: it gates a one-time decision per checkpoint pair; a permanent
  subcommand would be over-engineering (Principle II). Lives under the feature's
  `scripts/` (or `experiments/`) and is documented in `quickstart.md`.

## Open questions deferred (non-blocking, gen-2)

RL/self-play, live Forge seat, opponent-archetype head, win-probability
calibration, encoder fine-tuning, picker fine-tuning on small pools — all
explicitly out of scope (spec §Out of Scope) and untouched here.
