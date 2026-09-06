# Research: Ability effect model

**Feature**: `023-ability-effect-model` | **Date**: 2026-09-06
**Spec**: [spec.md](spec.md) | **Root spec**: [`../2026-09-05-ability-effect-model.md`](../2026-09-05-ability-effect-model.md)

The root spec pins every design decision this feature needs — flag defaults, the sampling mixture, the
architecture constants, all three gate thresholds — so Phase 0 has no open design questions to resolve.
What it does not pin is where the implementation should attach to the existing codebase. That is what
this document establishes, and it is the reason the Codebase Survey is the bulk of it.

## Codebase Survey

Constitution Principle VII. Every bullet cites a concrete file or symbol.

### Overlapping domain vocabulary

| Existing concept | Where | Decision |
|---|---|---|
| `ConvertedCardText` | `src/price_predictor/domain/card_text.py:33` | **Reuse.** It already parses the converted line format (`spell[n]:`, `static:`, `activated[n]:`) that the provenance sidecar indexes and the encoder tokenizes. The sidecar is a paired artifact next to the same `.txt`, not a competing representation of it. |
| `ConvertedCardLocator` | `src/sealed/infrastructure/converted_card_locator.py` | **Extend.** It already owns name sanitization, prefix-fallback lookup, the letter-keyed directory layout, and the `.txt`/`.npz` pairing. The effects cache needs identical resolution across three source trees (`cardsfolder`, `tokenscripts`, `variant-scripts`). Adding a source-tree parameter is strictly cheaper than a parallel locator, and a parallel one would duplicate `FILENAME_CORRECTIONS` and `sanitize_card_name`. |
| `card_embedding_layout` | `src/sealed/domain/card_embedding_layout.py` | **Parallel concept, justified.** That module describes the sealed `.npz`: one pooled card vector plus trailing deterministic features. The effects cache is a different artifact — an `(n_lines, e_dim)` matrix row-aligned to a sidecar. Shape, arity, and consumers all differ. No rename is proposed because the names do not collide: the new module is `effects/domain/ability_cache_layout.py`. |
| `DraftRecord` / `Booster` / `Seat` | `src/draft/domain/draft_geometry.py` | **Parallel concept, justified.** Same *shape of idea* — a self-contained JSONL record with an envelope and a payload — but a different domain. The effect record mirrors its split of pure domain dataclass from infrastructure serializer rather than reusing its types. |
| `compute_basic_lands`, `BASIC_LAND_NAMES` | `src/sealed/domain/manabase.py`, `src/sealed/domain/deck.py` | **Reuse.** The root spec mandates `compute_basic_lands` for coverage and variant decks. |
| `GreedyDeckBuilder` | `src/sealed/domain/greedy_deck_builder.py` | **Not used.** Coverage decks are weighted by record scarcity, not by a scorer; the root spec specifies the weighting directly. Recorded here so a reviewer does not assume it was overlooked. |

Two concepts reused, one extended, two parallel concepts introduced with justification, zero renames
required.

### Adjacent prior art

| Sub-problem | Prior art | Decision |
|---|---|---|
| Supervising a pool of Forge JVM workers | `ForgeWorkerPool`, `build_forge_classpath`, `build_jvm_command`, `kill_process_tree` in `src/price_predictor/infrastructure/forge_jvm.py`; used by `sealed/application/match_outcomes.py` and `draft/application/play_draft_games.py` | **Reuse unchanged.** `collect-coverage` and `collect-variants` are the third and fourth supervisors. The abstraction was already extracted at the second instance; its printed status lines are a documented operator contract. |
| Saving/loading torch checkpoints with dataclass configs | `save_checkpoint` / `load_checkpoint` in `src/price_predictor/infrastructure/torch_checkpoint.py`; used by `transformer_store`, `scorer_store`, `picker_store`, `draft_agent_store`, and the three draft trainers | **Reuse.** The effects checkpoint adds payload keys (held-out card list, `game_id` set, vocabulary and keyword-definition paths and hashes) but needs no new save/load machinery. |
| Per-parameter-group gradient clipping | `clip_per_group` in `src/price_predictor/infrastructure/torch_training.py` | **Reuse.** The root spec mandates per-parameter-group max-norm 1.0, which is exactly this helper. |
| Building a tokenizer vocabulary | `build_vocabulary` in `src/price_predictor/application/build_vocabulary.py`; wrapped by `sealed/application/build_vocab.py` with `_truncate_to_target_size` | **Wrap, mirroring sealed.** `effects build-vocab` is the third caller. It should call the shared utility and reuse sealed's truncation rather than re-deriving it — see the third-instance check. |
| Resolving a card name to files on disk | `ConvertedCardLocator` (above) | **Extend.** |
| Card-disjoint validation splitting | `_split_cards` in `src/sealed/application/train_encoder.py:545` | **Reimplement, documented reason.** The discipline (card-level disjointness, never row-level) carries over, but the stratification key does not: sealed splits by `score_play` quartile, while the effects split takes cards by newest first printing until they cover ≥ 8% of the corpus. Sharing the function would mean parameterizing the very thing that differs. |
| Ridge probe harness for the decodability battery | `scripts/scorer_probes/probe_lib.py` (`Probe`, `det_features`) | **Reuse.** The root spec cites this harness and its feature table by name. |
| Rendering converted ability lines from runtime Forge traits | `RulesParser.parseScript` in `forge-connector/.../RulesParser.java:61` | **Extend, additive.** The parser already holds the runtime trait object it renders each line from, which is exactly the provenance the sidecar records. Writing the sidecar is a new output, not a change to the rendered text. |
| Running instrumented Forge matches | `MatchWorkerMain` | **Extend.** Instrumentation sits behind `--effect-records`; absent the flag the worker is byte-for-byte the current one in behaviour. |
| Per-worker Java process launch from Python | `match_worker_connector.py`, `draft_worker_connector.py`, `pool_connector.py`, `evaluation_connector.py` | **Mirror.** Each Java main gets one thin connector; the effects collectors follow the same one-connector-per-main convention. |

### Convention alignment

Mirror **`src/draft/`**. It is the most recently added package and the closest structural analogue: a
third top-level package laid out `domain` / `application` / `infrastructure`, owning a JSONL corpus of
envelope records, importing downward from `sealed` and `price_predictor` and never the reverse, with
one `cli.py`, one `*_store.py` per model artifact, and one `*_connector.py` per Java worker main.

| `draft` | `effects` |
|---|---|
| `domain/draft_geometry.py` (pure record dataclasses) | `domain/records.py` |
| `domain/draft_state.py` (typed model input) | `domain/effect_head_input.py` |
| `domain/draft_agent_model.py` | `domain/effect_model.py`, `domain/ability_encoder.py` |
| `infrastructure/draft_record_io.py` | `infrastructure/record_io.py` |
| `infrastructure/draft_agent_store.py` | `infrastructure/effect_model_store.py` |
| `infrastructure/draft_worker_connector.py` | `infrastructure/collector_connector.py` |
| `infrastructure/cli.py` | `infrastructure/cli.py` |

Test style follows the repo's split: fast unit tests under `tests/unit/effects/` mirroring the source
tree, anything needing a JVM marked `integration` (the marker is declared in `pyproject.toml:36`).

No deviation from the sibling layout is proposed.

### Third-instance check

**Partial-line-tolerant reading of an append-only corpus is already implemented three times**, and the
effects record reader would be the fourth:

- `src/sealed/infrastructure/pool_file_reader.py:88` — `count_complete_lines_and_truncate_partial`
- `src/sealed/infrastructure/cards_played_reader.py:61` — tolerates a trailing partial line silently
- `src/draft/infrastructure/draft_record_io.py:81` — whose own docstring says it "Mirrors the
  partial-line tolerance of `sealed`'s `pool_file_reader` / `cards_played_reader`"

That docstring is the codebase admitting the duplication. Principle VII requires extraction rather
than a fourth copy. **Proposal**: extract the line-level primitive — iterate complete
(newline-terminated) lines, count them, and optionally truncate a trailing partial — into
`src/price_predictor/infrastructure/append_only.py`, the package both `sealed` and `draft` already
depend on, and refactor the three existing call sites onto it. Record parsing stays with each corpus;
only the completeness rule is shared. This becomes a required task in `tasks.md` and is a
prerequisite for the effects record reader, not a follow-up.

Three further patterns were checked and need no action, because extraction already happened at the
second instance: Forge worker supervision (`ForgeWorkerPool`), torch checkpoint IO
(`torch_checkpoint`), and the sealed delimited-line grammar (`sealed/infrastructure/delimited.py`,
used by three readers — not applicable here, since effect records are JSONL rather than
semicolon-delimited).

One near-miss to watch: `_truncate_to_target_size` currently lives only in
`sealed/application/build_vocab.py`. Effects is its second caller, so reusing it keeps the count at
one implementation; copying it would make the next feature a third instance.

## Decisions

### Corpus format: JSONL shards, not a single file

- **Decision**: one `{run_id}.{worker}.jsonl` shard per worker, all read as a directory.
- **Rationale**: the root spec fixes it. It also removes the cross-process append interleaving that
  `match-outcomes.txt` tolerates only because its rows are short enough to be atomic.
- **Alternatives considered**: a single append-only file (rejected — records are far larger than a
  match-outcome row, so interleaving is a real corruption risk); Parquet (rejected — the corpus is
  append-only under crash-prone workers, which is exactly what line-oriented formats survive).

### State stored as data, tensorized at training time

- **Decision**: snapshots carry names plus dynamic attributes; the tensor layout is derived in
  `domain/effect_head_input.py`.
- **Rationale**: the root spec's one collection principle. Every representation change becomes a code
  change instead of a re-collection, and the corpus cannot be rebuilt cheaply.
- **Alternatives considered**: baking the feature vector into the corpus (rejected explicitly by the
  design record, which cites the draft spec's having done exactly that).

### Two Python entry points into Java, not one

- **Decision**: instrumentation rides `sealed match-outcomes --effect-records`; `collect-coverage` and
  `collect-variants` are `effects` subcommands that spawn the same instrumented worker.
- **Rationale**: the root spec's split. It keeps the sealed corpus commands unchanged by default while
  giving the effects-only collectors their own supervisors and their own defaults.
- **Alternatives considered**: a single `effects collect` with a mode flag (rejected — the sealed
  command already exists and its outputs must stay untouched, so the opt-in has to live there).

### Degraded mode is a runtime detection, not a build flag

- **Decision**: workers probe for the patch hooks at startup and stamp `mode` on every record.
- **Rationale**: the patch lives in a sibling checkout that is rebuilt independently and re-patched by
  hand after every Forge upgrade. A build-time switch would silently mislabel a corpus collected after
  a lapsed patch.
- **Alternatives considered**: failing fast without the patch (rejected — stage one is defined as the
  patch-free stage); a bytecode agent (rejected in the design record as a heavier moving part).

## Technology notes

- **Python 3.14.3**, `requires-python >=3.14` (`pyproject.toml:9`); torch ≥ 2.2 on the CUDA 12.6 wheel
  index; numpy ≥ 1.26.
- **Java 17**, `forge-connector` compiling against forge-game / forge-core / forge-gui / forge-ai
  `2.0.15-SNAPSHOT` from the sibling checkout, pinned by the `forge.version` property in
  `forge-connector/pom.xml:22`.
- **Training GPU has 8 GB of VRAM.** This is the binding constraint on batch composition and is why
  `--context-cache` exists as a documented fallback rather than an optimization.
