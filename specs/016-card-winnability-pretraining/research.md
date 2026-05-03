# Phase 0 Research: Card Winnability Pretraining for Sealed Encoder

## Codebase Survey

This survey is the Principle VII gate. Each subsection cites concrete
files and symbols so a reviewer can verify reuse decisions without
re-deriving them.

### Overlapping domain vocabulary

- **Existing concept: `CardPriceTransformerModel`** —
  `src/price_predictor/infrastructure/transformer_model.py:11`. Owns
  the token + positional embeddings, an `nn.TransformerEncoder` stack,
  and `_encode_and_pool()` returning `cat([max_pool, mean_pool])` with
  shape `(batch, 2 * d_model)`.
  - **Decision: parallel concept (justified)** — name the new class
    `SealedEncoderModel` (or `CardWinnabilityModel`), living under
    `src/sealed/domain/` rather than `src/sealed/infrastructure/`
    because it is the trained domain artifact (mirrors the price-side
    arrangement: the model class lives in `price_predictor/infrastructure`
    only because the project predates that layering choice; the sealed
    side keeps the model in `domain/` per `sealed/domain/scorer_model.py`
    convention).
  - **Why parallel and not extend**: the spec requires (a) a different
    pool head — multi-query attention pool concatenated with max pool
    instead of mean pool — and (b) random initialization with a
    sigmoid regression head on a `[0, 1]` target, which is incompatible
    with the price model's `meta_dim`-side-channel architecture and
    log-price target. Sharing inheritance would require carving the
    overlap (token+positional embed, transformer stack) into a base
    class — useful but a larger refactor and an unjustified third
    instance until a third caller appears (Principle II "three concrete
    use cases").
  - **Reuse**: the `_encode_and_pool` masking pattern and positional-
    encoding shape (`nn.Embedding(max_seq_len, d_model)`) are copied
    verbatim into the new class.

- **Existing concept: `TransformerConfig`** —
  `src/price_predictor/domain/entities.py:103`. Holds `d_model`,
  `n_layers`, `n_heads`, `ff_dim`, `max_seq_len`, `vocab_size`,
  `dropout`, `meta_dim`, `regression_hidden_dim`.
  - **Decision: parallel concept** — introduce
    `SealedEncoderConfig` (sibling to `ScorerConfig` in
    `sealed/domain/scorer_model.py:N`) holding only the encoder-
    relevant fields (`d_model`, `n_layers`, `n_heads`, `ff_dim`,
    `max_seq_len`, `vocab_size`, `dropout`, `n_pool_queries`). The
    price-side fields `meta_dim` and `regression_hidden_dim` do not
    apply to the sealed encoder (no metadata side-channel; the
    regression head is intentionally tiny and not part of the saved
    artifact).

- **Existing concept: vocabulary file** —
  `models/price-predictor/transformer/vocab.txt`, written by
  `price_predictor.infrastructure.tokenizer_store.save_vocabulary`.
  - **Decision: separate file, shared utility**. The sealed
    vocabulary lives at `models/sealed/encoder/vocab.txt` (per
    FR-008). The same `save_vocabulary()` function writes both,
    and `MtgTokenizer.load(path)` reads both. This honors FR-008
    ("building one MUST NOT modify the other") while reusing the
    single source-of-truth read/write pair.

- **Existing concept: `MtgTokenizer`** —
  `src/price_predictor/domain/tokenizer.py:18`.
  - **Decision: reuse unchanged**. Construct a tokenizer from the
    sealed vocab file and pass it into the new training/encoding
    paths. No new tokenizer class is needed.

### Adjacent prior art

- **`price_predictor.application.build_vocabulary.build_vocabulary`** —
  `src/price_predictor/application/build_vocabulary.py:241`. Algorithm:
  seed special + domain tokens, optionally seed set-code fragments
  from `AllPrintings.json`, scan corpus, add tokens meeting
  `freq_threshold`. Returns `VocabBuildResult(vocab, coverage_stats)`.
  - **Decision: reuse via wrapper**. New module
    `src/sealed/application/build_vocab.py` calls this function with
    sealed-side defaults (`vocab-path = models/sealed/encoder/vocab.txt`)
    and writes via `save_vocabulary()`. No re-implementation. See
    decision **D-1** below for the `--target-size` ↔ `--freq-threshold`
    mismatch.

- **`MtgTokenizer.encode(text, max_seq_len)` returning
  `(input_ids, attention_mask)`** —
  `src/price_predictor/domain/tokenizer.py`. Already used by both the
  price training loop and `sealed.domain.card_encoder.CardEncoder`.
  - **Decision: reuse**. Used as-is in the new training Dataset and
    by the to-be-introduced sealed `CardEncoder` swap.

- **`CardEncoder` (sealed)** —
  `src/sealed/domain/card_encoder.py:34`. Strips the `name:` line via
  `ConvertedCardText.without_name_line()`, tokenizes, calls
  `model.encode()`, concatenates deterministic features.
  - **Decision: reuse, with the encoder swapped**. The new sealed
    encoder model exposes the same `encode(input_ids, attention_mask)`
    method shape `(batch, 2 * d_model)` so `CardEncoder` works
    unchanged. The only delta is at the loader level: a new
    `SealedEncoderStore.load(path) -> (SealedEncoderModel, SealedEncoderConfig)`
    is wired into `cli.run_encode_cards`.

- **`MatchResultWriter`** —
  `forge-connector/src/main/java/com/pricepredictor/connector/MatchResultWriter.java`.
  Append-only, one line per match, opens-writes-closes for each line
  so concurrent workers cannot corrupt the file.
  - **Decision: pattern-mirror, do not extend**. Add a sibling
    `CardsPlayedWriter` in the same package, identical write strategy,
    distinct file path and distinct line schema. Extending the existing
    writer would couple unrelated outputs (one is per-match, the other
    is per-game) and bloat its public surface.

- **`GamePlayer.playMatch()`** —
  `forge-connector/.../GamePlayer.java:55`. Currently returns
  `PlayedMatch(List<GameOutcome>, durationSeconds)` where
  `GameOutcome(winner, playFirst)`.
  - **Decision: extend `GameOutcome`** to also carry `Set<String>
    cardsPlayedA`, `Set<String> cardsPlayedB` — the set of non-basic
    card names that entered the battlefield or stack on each side
    during that game. The `GamePlayer` is the only place with the
    Forge `Game` object in scope while the game is being played, so
    instrumenting there avoids passing live game state out of the
    class. Existing call sites that read only `winner`/`playFirst`
    keep working.

- **`MatchGenerator.generateMatch()`** —
  `forge-connector/.../MatchGenerator.java`. Composes pool + decks +
  metadata, calls `GamePlayer.playMatch`, returns a `MatchResult`.
  - **Decision: extend the return value**. `MatchGenerator` produces
    a `(MatchResult, List<CardsPlayedRow>)` pair, where
    `CardsPlayedRow` is the per-game value object containing the
    eleven fields of `cards-played.txt`. The worker loop in
    `MatchWorkerMain.runForever` then writes the match line via
    `MatchResultWriter` and the game lines via `CardsPlayedWriter`.

- **`ScorerStore.save_checkpoint`** —
  `src/sealed/infrastructure/scorer_store.py:42`. Pattern: write
  timestamped `.pt` file then update `latest.pt`. Best-by-validation
  selection happens upstream in `train_scorer.py`.
  - **Decision: pattern-mirror**. New
    `src/sealed/infrastructure/encoder_store.py` adopts the same
    save/load pattern: `save_encoder(model, config, path)` writes
    `{timestamp}.pt` + `latest.pt` containing `model_state_dict` and
    `config` (only — no optimizer state, no regression-head weights,
    per FR-020). `load_encoder(path) -> (model, config)` is the
    counterpart used by `encode-cards` and `train-scorer`.

- **`_BestCheckpoint` early-stopping pattern** —
  `src/price_predictor/application/train_transformer.py:146`. Track
  best val loss, snapshot weights on improvement, restore on early
  stop or normal completion.
  - **Decision: reuse the pattern, not the class**. The price-side
    class is private (`_BestCheckpoint`) and tied to the price
    training Module; copy the ~30-line skeleton into the sealed
    training module. Promoting it to a shared utility is a
    Principle-II ("three use cases") candidate — there will now be
    two callers; mark a follow-up to extract on the next training
    feature.

- **`CardNameCorrections.FILENAME_CORRECTIONS`** —
  `src/sealed/infrastructure/card_name_corrections.py`. Maps known
  Forge filename typos to canonical names.
  - **Decision: reuse for the corpus consistency check** (FR-023d).
    When verifying that every card named in `cards-played.txt`
    exists under `output/cardsfolder/`, apply the corrections map so
    a Forge-side typo does not falsely flag a missing card.

- **`ConvertedCardLocator`** —
  `src/sealed/infrastructure/converted_card_locator.py`. Already
  resolves card name → on-disk `.txt` path, applying corrections.
  - **Decision: reuse**. The label-aggregation step uses this
    locator to (a) drop missing cards into the FR-023d error path
    and (b) feed text into the tokenizer.

### Convention alignment

The sibling module to mirror is `sealed/` itself, which is well-laid-out:

- Domain models live in `src/sealed/domain/*.py` (e.g.,
  `scorer_model.py`, `card_encoder.py`).
- Use cases live in `src/sealed/application/*.py` and follow the
  pattern `<config dataclass> + run(config) -> result`. See
  `train_scorer.py:38` for the canonical shape (`TrainScorerConfig`,
  `train_scorer(config)`, `_log()` for stdout progress lines, etc.).
- Infrastructure adapters live in `src/sealed/infrastructure/*.py`
  and own all I/O (`scorer_store.py`, `match_data_loader.py`,
  `embedding_store.py`).
- CLI wiring lives in one file:
  `src/sealed/infrastructure/cli.py:60` (`build_parser`). Each
  subcommand has a private `_build_<name>_parser(subparsers)` and a
  module-level `run_<name>(args)` handler.
- Tests mirror the source layout under `tests/unit/sealed/{domain,
  application, infrastructure}/test_*.py`. Java tests live under
  `forge-connector/src/test/java/.../`.

The new feature follows this layout exactly. No deviation.

### Third-instance check

No third instance pattern triggered.

- **Encoder loaders**: there is one (`load_model` in
  `transformer_store.py`). After this feature there will be two
  (price-side + sealed-side). Two is below the Principle-II threshold
  for extraction; the second is justified by config-shape divergence
  (the sealed config has no `meta_dim`).
- **Vocabulary builders**: there is one (`build_vocabulary`). The
  sealed wrapper *delegates* to it rather than reimplementing,
  keeping the count at one.
- **CLI subcommand registration**: there is one pattern
  (`subparsers.add_parser` in `sealed/infrastructure/cli.py`). The
  two new subcommands follow it.
- **Best-checkpoint early stopping**: there is one
  (`_BestCheckpoint` in `train_transformer.py`). The new training
  loop will become the second instance. **Follow-up task**: when a
  third training entry point arrives, extract a shared
  `BestCheckpoint` utility into `price_predictor.application` (or a
  neutral shared module). Not extracted in this feature to keep the
  blast radius small.

## Decisions

### D-1: `--target-size` mapped to a post-hoc truncation, not freq-threshold tuning

**Decision**: `python -m sealed build-vocab --target-size N` calls
`build_vocabulary(corpus, freq_threshold=2)` (very inclusive), then
truncates the resulting vocab to the top-N entries by frequency,
*always preserving* the seeded special tokens (`[PAD]`, `[UNK]`,
`cardname`), domain seed terms, and set-code fragments — those slots
count against the budget but are never evicted. If `N` is smaller
than the seed slot count, raise `ValueError`.

**Rationale**: The spec uses `--target-size` (FR-009) and reads the
default as "approximately 5000 tokens." The existing
`build_vocabulary` uses `--freq-threshold`. Three options were
considered:

1. **Rename to `--freq-threshold`** in the sealed wrapper — diverges
   from the spec and surprises users who expect "size" to mean
   "size."
2. **Binary-search freq-threshold to hit N tokens** — multi-pass over
   the corpus per `build-vocab` invocation, complex, and the seed
   tokens skew the search.
3. **Single-pass with post-hoc truncation** (chosen) — one pass at
   `freq_threshold=2`, sort the corpus tokens by frequency, keep the
   top `(N - seed_count)`. Simple, deterministic, single-pass.

**Alternatives rejected**:
- Option 1 above.
- Option 2 above.

### D-2: Per-card label map computed inline, not exposed as a CLI command

**Decision**: Aggregation runs inside `train-encoder` as the first
step. There is no `python -m sealed aggregate-labels` subcommand. The
human-readable snapshot (`output/sealed/cards-win-rates.txt`,
FR-013a) is the only persistence.

**Rationale**: The spec mandates this (FR-013): "Aggregation MUST run
inline at train start. The system MUST NOT expose a separate
aggregation subcommand." Honors Simplicity First (Principle II) — no
auxiliary command until a real second consumer of the label map
exists.

**Alternatives rejected**:
- Caching the label map in `models/sealed/encoder/labels.json` for
  cross-run reuse: rejected because aggregation is fast (single pass
  over a textfile), and a cache would create a stale-data hazard the
  first time `cards-played.txt` grows after a successful train run.
- A separate `aggregate-labels` subcommand: rejected per spec, and
  for the same Principle II reason.

### D-3: Java worker observes per-game card play via Forge eventbus (two events)

**Decision**: `GamePlayer.playMatch()` registers a per-game
observer with the Forge `Game` object via
`game.subscribeToEvents(visitor)` (Guava `EventBus` + `@Subscribe`).
The observer is an `IGameEventVisitor.Base<Void>` subclass that
captures **two** event types and unions their results:

1. `GameEventCardChangeZone` filtered by
   `event.to().getZoneType() == ZoneType.Battlefield`. Catches
   permanents that actually resolved (creatures, planeswalkers,
   artifacts, enchantments, lands). Misses instants/sorceries
   that never enter the battlefield.
2. `GameEventSpellAbilityCast` (any cast). Catches every spell
   regardless of whether it resolves, is countered, or fizzles.
   This is the "stack" half of FR-003 in code form — every cast
   produces this event because casting a spell puts it on the
   stack by definition.

For each event, the observer applies four filters before
recording the card:

- `card.getController() == card.getOwner()` — drops cards stolen
  via Threaten / Mind Control (FR-003: "controlled by the side
  that owns it").
- `!card.isToken()` — tokens are not deck cards.
- `!card.getType().isBasicLand()` — FR-004a basic-land exclusion
  applied at observation time, not at write time, so the
  observer's two sets are already write-ready.
- For copy effects (Clone, "enters as a copy of"), use
  `card.getPaperCard().getName()` instead of `card.getName()` so
  the *cast* card's identity is recorded, not the copied
  permanent's.

The recorded card is bucketed by `card.getOwner().getName()`
(lobby player name, e.g. `p1` / `p2`), then mapped to side
`A` / `B` via the `LOBBY_NAME_A` / `LOBBY_NAME_B` constants
already in `GamePlayer`. At end-of-game, the observer's two
sets are attached to the `GameOutcome` returned for that game.
The observer is created fresh per game (not per match) so
flicker effects, copies, and re-entries collapse to set
membership.

**Rationale**: This is a direct port of the working pattern in
`../jumpstart-tierlist/src/main/java/org/mtg/tierlist/JumpstartMatch.java`
(see the inner class `CardCollector`). That repository has been
running this approach against Forge in anger for sealed-adjacent
deck simulations and has resolved every Forge-specific corner
case empirically. Mirroring it removes guesswork about which
events fire, in what order, and against which `Card` instance
(token vs paper).

The two-event union is necessary because:
- `GameEventCardChangeZone` to `ZoneType.Battlefield` misses
  instants/sorceries entirely (they go to graveyard or exile,
  not battlefield).
- `GameEventSpellAbilityCast` covers every spell cast, but it
  does NOT fire when a land is played (land plays are not casts).
  Lands are caught by event 1 because they enter the battlefield.

**Alternatives rejected**:
- **Listen only to `ZoneType.Battlefield` zone transitions**:
  rejected because instants/sorceries never reach the
  battlefield and would be missed.
- **Listen only to `GameEventSpellAbilityCast`**: rejected
  because lands are played (zone change), not cast, and would
  be missed.
- **Listen also to `ZoneType.Stack` transitions**: rejected
  because every cast already produces the cast event, and
  triggered abilities would create false positives ("the card
  was played" should mean "the player chose to play it," not
  "it triggered something").
- **Snapshot the battlefield at end of game**: rejected because
  a card that is played and then dies (or is exiled) before the
  game ends would not appear on the final battlefield.
- **Diff `deck` against `library + hand + sideboard + exile` at
  end of game**: rejected because exile, graveyard, command,
  and stack are all valid "the card was played" destinations
  and the diff approach has to enumerate every "played" zone
  explicitly.
- **Instrument at the `MatchGenerator` level**: rejected because
  `MatchGenerator` does not have the `Game` instance — only
  `GamePlayer.playMatch()` does.

### D-4: Train/val split stratified by winnability quartile, not by raw count

**Decision**: After computing the per-card shrunk label, bucket cards
into four quantiles by label value, then sample 20% of each quartile
(rounded) into the validation set. If fewer than four distinct
quantile boundaries exist (small corpus, degenerate distribution),
fall back to as many strata as there are distinct boundaries and
preserve the 20% split per stratum.

**Rationale**: Spec FR-018 mandates card-level disjoint split
stratified by winnability quartile. Stratifying by quartile (not by
in-deck count) ensures each split spans the full label range so
the val-loss metric is comparable across runs even when the corpus
shifts toward one end of the distribution.

**Alternatives rejected**:
- **Stratify by `wins_when_in_deck`** (high vs low observation count):
  rejected because the model's signal is the label *value*, not the
  observation count, and the split should preserve the value
  distribution.
- **Random 80/20 with no stratification**: rejected because under a
  long-tail label distribution, a random val set can underrepresent
  high-winnability cards and inflate val loss noise.

### D-5: Encoder checkpoint format mirrors `transformer_store` minus regression head

**Decision**: `models/sealed/encoder/{timestamp}.pt` contains a torch
`.pt` payload with two keys: `model_state_dict` (only the token
encoder + card encoder weights, *not* the regression head) and
`config` (a serialized `SealedEncoderConfig` dataclass). `latest.pt`
is a copy of the best-by-val-loss checkpoint (not a symlink, since
this is Windows and symlinks are awkward; mirrors the price-side
convention in `transformer_store.save`).

**Rationale**: FR-020 requires "the regression head's weights MUST
NOT be written." The cleanest way is to filter the state-dict at
save time, keeping only keys whose first segment is `token_encoder`
or `card_encoder`. The downstream loader (`encode-cards`,
`train-scorer`) sees a state-dict it can drop straight into a
`SealedEncoderModel` without monkey-patching.

**Alternatives rejected**:
- **Save full state-dict and drop the head at load time**: rejected
  because a reader would have to know which keys to drop, and the
  artifact would unnecessarily carry training-only weights.
- **Save model + tokenizer config + vocab in one file**: rejected
  because the vocab is already at `models/sealed/encoder/vocab.txt`
  and the project convention is to keep them separate (matches the
  price-side `transformer/vocab.txt` + `transformer/latest.pt`
  split).

### D-6: `--encoder-checkpoint` default flip is a single-line config change

**Decision**: Switch
`TrainScorerConfig.encoder_checkpoint` (in
`src/sealed/application/train_scorer.py:45`) from
`Path("models/price-predictor/transformer/latest.pt")` to
`Path("models/sealed/encoder/latest.pt")`. Switch the
`_ENCODE_CARDS_DEFAULT_ENCODER` constant in
`src/sealed/infrastructure/cli.py` (currently around line 523)
similarly. Add a missing-file check that raises a clear error
naming `python -m sealed train-encoder` as the corrective action
(FR-026), placed in both call paths' default-resolution branches.

**Rationale**: The defaults are isolated to two locations, making
the swap trivial and reversible. The spec emphasizes this is a
default change, not a removal — explicit
`--encoder-checkpoint <path>` continues to work (FR-027).

**Alternatives rejected**:
- **Detect the sealed encoder's existence at runtime and fall back
  to the price encoder when missing**: rejected because the spec
  (FR-026, edge cases) is explicit that absence is a hard error
  pointing at `train-encoder`. Silent fallback would mask config
  bugs and produce a sealed scorer trained on the wrong encoder.

### D-7: `cards-played.txt` write happens in Java; Python supervisor unchanged

**Decision**: The Java match worker writes both `match-outcomes.txt`
(via `MatchResultWriter`) and `cards-played.txt` (via the new
`CardsPlayedWriter`). The Python supervisor (`match_outcomes.py`)
is unchanged — it still spawns workers, restarts crashes, and reads
neither file. The two files share an output directory but are
written by independent file handles, each opened-written-closed per
line.

**Rationale**: Per-game card play is observable only inside the
Forge JVM (Decision D-3). Writing from Java avoids serializing
`Set<String>` data over the supervisor stdout channel and mirrors
the existing per-match write pattern. Both files are flushed at
line granularity, so a JVM crash mid-write truncates at most one
line — tolerated by the downstream readers (Edge Cases section of
the spec).

**Alternatives rejected**:
- **Stream card-play data to Python over stdout, then have Python
  write the file**: rejected because it doubles the I/O path,
  requires a wire protocol, and the supervisor restart semantics
  would have to gain awareness of partial cards-played writes.
