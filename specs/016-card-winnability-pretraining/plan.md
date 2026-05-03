# Implementation Plan: Card Winnability Pretraining for Sealed Encoder

**Branch**: `016-card-winnability-pretraining` | **Date**: 2026-05-03 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/016-card-winnability-pretraining/spec.md`

## Summary

Replace the price-predictor transformer with a sealed-specific
encoder trained from scratch on per-card winnability — a `[0, 1]`
target derived from "did the winning side play this card during
the game?" aggregated across self-play matches. The feature has
four pieces: (1) a per-game card-play log written by the Java
match worker (`output/sealed/cards-played.txt`), (2) a sealed-side
vocabulary builder wrapping the existing price-side utility, (3) a
new `train-encoder` subcommand that aggregates labels inline,
trains a token+card encoder with multi-query attention pool plus
max pool against MSE on a Bayesian-shrunk target, and saves only
encoder weights, and (4) a default-flip in `encode-cards` and
`train-scorer` so the sealed pipeline picks up the new encoder
automatically. The price-predictor product itself remains live and
unaffected.

## Technical Context

**Language/Version**: Python 3.14 (project requirement); Java 17+
for the `forge-connector` Maven module.
**Primary Dependencies**: `torch` (with CUDA 12.6 wheels),
`numpy`, `scikit-learn` (for stratified split), existing
`price_predictor.application.build_vocabulary` and
`price_predictor.domain.tokenizer.MtgTokenizer` reuse. Java side
depends on `forge-game` / `forge-core` already pulled in by the
existing Forge connector.
**Storage**: filesystem only — no database. New artifacts:
`output/sealed/cards-played.txt`,
`output/sealed/cards-win-rates.txt`,
`models/sealed/encoder/{timestamp}.pt`,
`models/sealed/encoder/latest.pt`,
`models/sealed/encoder/vocab.txt`.
**Testing**: `pytest` for Python (`tests/unit/sealed/...`,
`tests/integration/...` for end-to-end flows); JUnit 5 for the
Java connector (`forge-connector/src/test/java/...`). Forge-
dependent Java tests carry the `@Tag("integration")` marker.
**Target Platform**: developer workstations (Windows 11 primary,
Linux secondary); CUDA-enabled GPUs accelerate
`train-encoder` but CPU fallback is acceptable for small corpora.
**Project Type**: CLI tool stack (single repository) — sealed
package on top of price-predictor, both within the `src/` tree.
The Java connector is a sibling Maven module.
**Performance Goals**: training is expected to fit in single-
digit hours on a typical sealed corpus (≤ 10⁵ games); the
aggregation pass is a single linear scan of `cards-played.txt`
and is O(games × deck_size).
**Constraints**: aggregation MUST run inline at train start
(FR-013, no separate command); `latest.pt` MUST contain only
encoder weights, no regression head (FR-020); train/val split
MUST be card-level disjoint (FR-018); per-game write in Java
MUST be line-buffered so a worker crash never produces a partial
line (FR-001).
**Scale/Scope**: ~10⁴–10⁵ games per `cards-played.txt` over a
typical data-collection campaign; ~3 × 10⁴ unique cards in the
converted corpus; vocabulary ~5000 tokens. `d_model = 256`,
`d_card = 512`, `n_layers = 6`, `n_heads = 4`,
`n_pool_queries = 4` by default.

## Constitution Check

*GATE: Pass before Phase 0 research; re-check after Phase 1 design.*

### Codebase Survey (Principle VII — required)

Findings recorded in [research.md § Codebase Survey](./research.md#codebase-survey).

- **Overlapping vocabulary**: 4 concepts reused
  (`MtgTokenizer`, `tokenizer_store.save_vocabulary`,
  `ConvertedCardLocator`, `card_name_corrections`); 2 parallel
  concepts introduced with explicit justification
  (`SealedEncoderConfig` parallel to `TransformerConfig`,
  `SealedEncoderModel` parallel to `CardPriceTransformerModel`)
  because the architectures and config shapes diverge by spec
  (FR-014, FR-015, FR-022). Both parallels reuse the masking and
  positional-encoding code via copy, not inheritance, since a
  shared base class would be a Principle II ("three concrete use
  cases") violation today; a follow-up extraction is queued for
  the next training entry point.
- **Adjacent prior art**: 6 utilities reused
  (`build_vocabulary`, `MtgTokenizer.encode`,
  `MatchResultWriter` pattern, `_BestCheckpoint` pattern,
  `ScorerStore` save pattern, `CardEncoder`); 2 new sibling
  classes in Java (`CardsPlayedRow` record + `CardsPlayedWriter`)
  follow the existing per-line write pattern; the
  `GameOutcome` record gains two fields (`cardsPlayedA`,
  `cardsPlayedB`) so the per-game observer can attach data
  without a parallel return type.
- **Convention alignment**: Sealed module conventions matched
  exactly — `domain/`/`application/`/`infrastructure/` split,
  `<Config> + run(config)` shape in application files, single CLI
  builder in `infrastructure/cli.py`, tests under
  `tests/unit/sealed/...` mirroring the source tree.
- **Third-instance check**: No third instance pattern triggered.
  One follow-up flagged: the `_BestCheckpoint` early-stopping
  helper becomes the second instance; extraction to a shared
  utility is recommended on the next training feature, not this
  one.

### Quality Gates

- **I. Fast Automated Tests**: tests are written alongside
  implementation (Python unit tests under `tests/unit/sealed/`
  for label aggregation, shrinkage math, train/val split,
  encoder save filtering, and CLI argument parsing; Java unit
  tests under `forge-connector/src/test/java/...` for the
  per-game observer, `CardsPlayedRow` formatting, and
  `CardsPlayedWriter`). The full training loop is exercised by
  one fast smoke test (1 epoch on a tiny synthetic corpus); the
  long-running cases are covered by a `@pytest.mark.integration`
  test that is excluded from the default fast suite.
- **II. Simplicity First**: no auxiliary `aggregate-labels`
  subcommand (FR-013). No separate config file format. No
  speculative resume capability (out of scope per spec). Bayesian
  shrinkage is the only low-n knob exposed.
- **III. Data Integrity**: `cards-played.txt` is line-buffered
  and tolerates trailing partial lines on read; `vocab.txt` is
  versioned only by overwrite (user responsibility per
  clarification); the encoder checkpoint stores its config dict
  alongside the state-dict so reload is fully reproducible
  given the same vocab.
- **IV. DDD & Separation of Concerns**: domain entities
  (`SealedEncoderModel`, `SealedEncoderConfig`, `CardLabel`)
  live in `src/sealed/domain/`; application use cases
  (`build_vocab`, `train_encoder`) in `src/sealed/application/`;
  I/O adapters (`encoder_store`, the cards-played reader) in
  `src/sealed/infrastructure/`. No domain entity imports
  framework or torch state; only domain modules `import torch`,
  consistent with sibling `scorer_model.py`.
- **V. MTG Forge Interoperability**: no impact on the public
  Java stub library (`PricePredictorClient`). The new Java
  components live in the existing connector package and are CLI-
  worker side, not consumed by Forge as a library.
- **VI. Documentation**: README/CLAUDE.md additions cover the
  new subcommands and file formats; `quickstart.md` is the
  copy-pastable end-to-end walkthrough.
- **VII. Codebase-Aware Planning**: see "Codebase Survey" subsection above.

No constitutional violations. Complexity tracking section omitted.

## Project Structure

### Documentation (this feature)

```text
specs/016-card-winnability-pretraining/
├── plan.md                # This file
├── research.md            # Codebase survey + design decisions
├── data-model.md          # Entities, fields, validation
├── quickstart.md          # End-to-end walkthrough
├── contracts/
│   ├── cli.md             # CLI command contracts
│   └── files.md           # On-disk file format contracts
├── spec.md                # Source spec (already exists)
└── tasks.md               # Generated by /speckit.tasks
```

### Source Code (repository root)

```text
src/
├── price_predictor/                  (UNCHANGED public surface)
│   ├── application/build_vocabulary.py        (REUSED)
│   ├── domain/tokenizer.py                    (REUSED)
│   └── infrastructure/tokenizer_store.py      (REUSED)
└── sealed/
    ├── domain/
    │   ├── encoder_model.py          NEW: SealedEncoderConfig + SealedEncoderModel
    │   │                                  (token_encoder, card_encoder, regression_head)
    │   ├── card_encoder.py           UNCHANGED (works against the new model.encode())
    │   ├── deterministic_features.py UNCHANGED
    │   └── scorer_model.py           UNCHANGED
    ├── application/
    │   ├── build_vocab.py            NEW: thin wrapper around price-side build_vocabulary
    │   ├── train_encoder.py          NEW: aggregation + train loop + save
    │   ├── encode_cards.py           UNCHANGED logic; see CLI default flip below
    │   ├── train_scorer.py           CHANGED: encoder_checkpoint default → sealed
    │   └── ...                       (other apps unchanged)
    └── infrastructure/
        ├── encoder_store.py          NEW: save/load encoder checkpoints with HEAD-filter
        ├── cards_played_reader.py    NEW: stream parser for cards-played.txt
        ├── cli.py                    CHANGED: register build-vocab + train-encoder; flip
        │                                       encoder defaults for encode-cards + train-scorer
        ├── card_name_corrections.py  REUSED
        ├── converted_card_locator.py REUSED
        └── scorer_store.py           UNCHANGED

forge-connector/src/main/java/com/pricepredictor/connector/
├── CardsPlayedRow.java               NEW: 11-field per-game record
├── CardsPlayedWriter.java            NEW: open-write-close per-line writer
├── PlayedCardCollector.java          NEW: IGameEventVisitor.Base<Void> subclass that
│                                          listens to GameEventCardChangeZone (→ Battlefield)
│                                          and GameEventSpellAbilityCast, with the four
│                                          filters from research.md D-3. Mirrors
│                                          ../jumpstart-tierlist CardCollector.
├── GamePlayer.java                   CHANGED: GameOutcome record gains 2 fields;
│                                              playMatch() creates a fresh PlayedCardCollector
│                                              per game and calls game.subscribeToEvents().
├── MatchGenerator.java               CHANGED: returns (MatchResult, List<CardsPlayedRow>)
├── MatchWorkerMain.java              CHANGED: writes both result files
├── MatchResultWriter.java            UNCHANGED
└── ...                               (other classes unchanged)

tests/
├── unit/sealed/
│   ├── application/
│   │   ├── test_build_vocab.py       NEW
│   │   └── test_train_encoder.py     NEW (aggregation, shrinkage, split, save filter)
│   ├── domain/
│   │   └── test_encoder_model.py     NEW (forward shape, encode shape, head present)
│   └── infrastructure/
│       ├── test_cards_played_reader.py  NEW
│       ├── test_encoder_store.py        NEW (round-trip save/load, head filter)
│       └── test_cli.py               EXTENDED (new subcommand argparse, default flips)
├── integration/
│   └── test_winnability_e2e.py       NEW (fixture corpus → train-encoder → encode-cards
│                                          → assert .npz shape & content)
└── fixtures/
    └── sealed/
        ├── cards-played.sample.txt   NEW (small synthetic corpus)
        └── cardsfolder/...           REUSED

forge-connector/src/test/java/com/pricepredictor/connector/
├── CardsPlayedRowTest.java           NEW (formatting, basic-land filter)
├── CardsPlayedWriterTest.java        NEW (line-buffered round-trip)
└── PerGameCardObserverTest.java      NEW (zone-change classification, controller bucketing)
```

**Structure Decision**: Single repository, two Python packages
(`price_predictor`, `sealed`) in `src/` plus the
`forge-connector` Maven module. The new domain
entities (`SealedEncoderConfig`, `SealedEncoderModel`,
`CardLabel`) and adapters (`encoder_store`, `cards_played_reader`)
slot into the existing `sealed/domain/` and
`sealed/infrastructure/` directories without introducing new
top-level layers. The Java additions (`CardsPlayedRow`,
`CardsPlayedWriter`, `PerGameCardObserver`) live alongside the
existing `MatchResult{Writer,Generator,…}` family in the same
package. CLI wiring is a single-file change
(`src/sealed/infrastructure/cli.py`) following the established
subparser pattern.

## Complexity Tracking

> No constitutional violations recorded; this section is intentionally empty.

## Phase 0 Output

- [research.md](./research.md) — codebase survey + 7 design decisions

## Phase 1 Output

- [data-model.md](./data-model.md) — entities, fields, validation
- [contracts/cli.md](./contracts/cli.md) — CLI command contracts
- [contracts/files.md](./contracts/files.md) — on-disk file schemas
- [quickstart.md](./quickstart.md) — end-to-end walkthrough

## Post-Design Constitution Re-check

After completing Phase 1 design:

- **No new principle violations introduced.** The design did not
  add a third instance of any pattern; the only follow-up
  surfaced (extract `_BestCheckpoint`) is deferred to a future
  feature when the third caller appears.
- **Domain layer remains framework-light.** The new
  `SealedEncoderModel` imports `torch` (consistent with
  `scorer_model.py`); no infrastructure leakage.
- **Data integrity boundaries are explicit.** Three error paths
  in `train-encoder` (missing vocab, missing cards-played, missing
  corpus card) each raise with a corrective-action message
  pointing at a specific subcommand. The corpus consistency check
  (FR-023d) runs after aggregation specifically so the error can
  enumerate offending names.
- **CLI defaults flip is reversible.** Passing
  `--encoder-checkpoint <price-encoder>` to either `encode-cards`
  or `train-scorer` restores the prior behavior in one step
  (SC-006), with no code changes.

Constitution Check: PASS.

## Follow-ups surfaced by the survey

- **Future**: when a third training entry point lands, extract a
  shared `BestCheckpoint` helper from the duplicated logic in
  `train_transformer.py` and `train_encoder.py` (Principle II,
  three-use-cases threshold).
- **Future**: rename the price-side `CardPriceTransformerModel`
  state-dict prefixes (`token_embedding`, `position_embedding`,
  `encoder`, `output_head`) into a child-module layout
  (`token_encoder.*`, `card_encoder.*`, `regression_head.*`) that
  matches the new sealed encoder, so a future shared base class
  has aligned state-dict keys. Not required by this feature.
