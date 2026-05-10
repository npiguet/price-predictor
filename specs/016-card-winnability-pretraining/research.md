# Phase 0 Research: Card Winnability Pretraining for Sealed Encoder

**Feature**: 016-card-winnability-pretraining
**Date**: 2026-05-10

## Codebase Survey

This survey is the Principle VII gate. Each subsection cites concrete files
and symbols so a reviewer can verify reuse decisions without re-deriving
them. Spec 016's first iteration shipped a single-`shrunk_label` encoder; the
current iteration replaces that with a 9-head + MLM design (see spec
§ Clarifications, sessions 2026-05-03 and 2026-05-10), so several v1 entities
are explicitly *replaced* below.

### Overlapping domain vocabulary

| Existing concept | Location | Decision | Notes |
|---|---|---|---|
| `CardLabel` (single-scalar) | `src/sealed/application/train_encoder.py:64` | **Replace** | Holds three scalars (`wins_when_played`, `wins_when_in_deck`, `shrunk_label`). FR-010a/-011 require 8 primary + 4×5 per-color counters and 9 raw + 9 shrunk labels per card. Splitting into `CardCounters` (raw counts) + `CardLabels` (derived metrics) keeps each focused and matches the two-pass aggregator. |
| `WinnabilityMap` alias | `src/sealed/application/train_encoder.py:78` | **Reuse with rename** | The `dict[str, X]` shape is right; the value type changes. Rename to `CardLabelMap` to match the new entity name. |
| `_aggregate_counts` (winning-side, single-pass) | `src/sealed/application/train_encoder.py:81` | **Replace** | Counts only the winning side and tracks one denominator. FR-010a demands counters for both sides plus a `@play` subset (so `losses_*` and `*@draw` are derivable). FR-010b demands a second pass that resolves color identity from each card's `mana cost:` line and increments per-color counters. |
| `_shrink` (toward 0.5) | `src/sealed/application/train_encoder.py:106` | **Replace** | Hardcodes shrinkage toward 0.5 (a probability). FR-011 has five neutral points (0 for the four signed labels, 0.5 for `played_rate`); the closed-form expressions live inline in the new `_build_label_map` since each label has a slightly different numerator/denominator pattern. |
| `_build_winnability_map` | `src/sealed/application/train_encoder.py:111` | **Replace** | Computes one shrunk scalar per card. New version computes 9 raw + 9 shrunk labels per card with FR-012 zero-denominator handling (cell present-but-empty rather than dropped to neutral). |
| `_split_cards` | `src/sealed/application/train_encoder.py:178` | **Extend** | Card-level disjoint split with quartile stratification — exactly what FR-018 wants. The stratification key changes from `shrunk_label` to `score_play`, with FR-018's fallback chain (use whichever non-empty signed cell the card carries; catch-all stratum if none) added at the front. |
| `_WinnabilityDataset` | `src/sealed/application/train_encoder.py:215` | **Extend** | Already strips the `name:` line via `ConvertedCardText.without_name_line()` per FR-014a. Returns `(input_ids, attention_mask, label)` today; needs to return `(input_ids, attention_mask, labels[9], weights[9], head_mask[9])` so the per-head per-batch sum-to-1 weighted MSE has everything it needs (FR-017, FR-017a). |
| `_BestCheckpoint` | `src/sealed/application/train_encoder.py:359` | **Reuse** | Tracks best-by-val-loss state. Spec FR-019 keeps `min` selection; only the metric changes from regression-MSE-only to full loss `L_reg + (--mlm-weight) · L_mlm` (Clarification 2026-05-10). The existing follow-up note ("extract when a third loss-driven trainer arrives") still stands. |
| `_make_optimizer` | `src/sealed/application/train_encoder.py:383` | **Extend** | Already uses AdamW with linear warmup over `total_steps // 20` (= 5%) and constant after that — matches Clarification 2026-05-10 verbatim. The new requirement is per-parameter-group max-norm 1.0 gradient clipping (FR-022); a single `clip_grad_norm_(model.parameters(), max_norm=1.0)` call inside `_train_epoch` between `loss.backward()` and `optimizer.step()` covers it. |
| `SealedEncoderModel` | `src/sealed/domain/encoder_model.py:157` | **Extend** | Replace the single `regression_head` (Sequential[Linear, Sigmoid]) with `regression_heads: nn.ModuleDict` of five linear projections (`score_play`, `score_draw`, `played_rate`, `cast_lift` each `Linear(2*d_model, 1)`; `color_lift` `Linear(2*d_model, 5)`); add `mlm_head: nn.Linear(d_model, vocab_size)` reading the contextualized token sequence (output of the transformer-layer stack, before the pool layer — FR-015a). The existing `_encode_and_pool` stays; `forward` is widened to return both the pooled vector → heads predictions *and* the pre-pool token sequence → MLM logits. |
| `SealedEncoderConfig` | `src/sealed/domain/encoder_model.py:17` | **Reuse** | Existing fields cover the full new architecture. `vocab_size` already accommodates the `[MASK]` token because it's derived from the loaded vocab. The `d_model % n_pool_queries == 0` invariant (line 47) is the divisibility rule from FR-015. |
| `SealedEncoderStore._ENCODER_PREFIXES` | `src/sealed/infrastructure/encoder_store.py:23` | **Reuse** | The `("token_encoder.", "card_encoder.")` filter already excludes everything outside the saved encoder. The new heads (`regression_heads.*`, `mlm_head.*`) live outside the filter so the saved file remains head-free per FR-020. The error path docstring is updated to mention the new head prefixes for clarity. |

No parallel domain concepts introduced. Three concepts (`CardLabel`,
`_aggregate_counts`, `_build_winnability_map`) are *replaced* because the
spec mandates a different schema; their identifiers do not survive.

### Adjacent prior art

#### Cards-played streaming reader (already in use)

- `iter_rows(path)` (`src/sealed/infrastructure/cards_played_reader.py:75`) yields `CardsPlayedRow` records, tolerates a trailing partial line for JVM-crash recovery, and rejects mid-file malformed lines.
- **Reuse**: The new two-pass aggregator iterates twice over the same file; both passes consume `iter_rows`. No reader-side change needed.
- The existing reader already exposes `winner` and `starter`, which pass 1 needs for the win/loss split and the `@play` subset. Pass 2 reads both decks (so per-color counters increment for the loser's deck too).

#### Converted-card text and mana-cost lookup

- `ConvertedCardLocator` (`src/sealed/infrastructure/converted_card_locator.py:27`) resolves a card name to its on-disk `.txt` path with double-faced/split/adventure prefix-search fallback.
- `ConvertedCardText` (`src/price_predictor/domain/card_text.py`) parses the file and exposes `without_name_line()` (used in pass-2 tokenization input) and a property for individual lines.
- **Reuse**: Pass 2 reads each in-corpus card's `mana cost:` line through the locator. The line lookup is a one-time-per-card operation cached in a small dict so the per-game color computation is just a dict lookup. The locator already memoizes the directory listing per first-letter bucket.
- **Color extraction helper** (no prior art): a private `_colors_from_mana_cost(line)` regex helper inside `train_encoder.py` scans `\{[^}]+\}` matches and unions the WUBRG letters they contain (so `{W}`, `{W/U}`, `{W/P}`, `{2/W}` all contribute W; `{2}`, `{C}`, `{X}` contribute nothing). Five lines, no domain entity.

#### Tokenizer + vocabulary loader

- `MtgTokenizer` (`src/price_predictor/domain/tokenizer.py:18`) already handles `[PAD]` and `[UNK]` reserved tokens. Adding `[MASK]` is a vocab-side change only — the tokenizer is dict-driven; once `[MASK]` is in the vocab dict, both `encode` and `decode` work with no tokenizer change.
- `load_tokenizer` (`src/price_predictor/infrastructure/tokenizer_store.py`) reads the vocab file line by line.
- **Reuse**: Both consumers work for the new vocab without code changes.
- **Where `[MASK]` is added**: `_seed_special_tokens` in `src/price_predictor/application/build_vocabulary.py:163`. One line: `_add_token(vocab, "[MASK]")` after `cardname` (which keeps `[PAD]=0` and `[UNK]=1` stable, just slots `[MASK]` into the next available ID). FR-009a's "must not collide with corpus-derived tokens" is satisfied because square brackets are not part of converted-card text.

#### Encoder save filter

- `SealedEncoderStore.save_encoder` (`src/sealed/infrastructure/encoder_store.py:43`) saves only keys starting with `token_encoder.` / `card_encoder.` and copies the `.pt` to `latest.pt`.
- **Reuse**: Verbatim. The new regression heads (`regression_heads.*`) and MLM head (`mlm_head.*`) sit outside the filter and are dropped at save time. `load_encoder` continues to call `load_state_dict(strict=True)` on the encoder children only — the live model still owns freshly-initialized heads after load, but those are training-only and never used at inference (since `encode()` is `@torch.no_grad()` and bypasses every head).

#### Linear-warmup schedule helper

- The current `_make_optimizer` (`src/sealed/application/train_encoder.py:383`) already implements the exact "linear warmup over 5% of total steps, then constant" schedule from Clarification 2026-05-10, using `LambdaLR` with `warmup_steps = max(1, total_steps // 20)`.
- **Reuse**: No change. The Clarification matched the existing implementation, not the other way around.

#### Per-batch per-head sum-to-1 weight normalization (no prior art)

- Nothing in the codebase normalizes per-head loss contributions per batch. The price-side trainer uses unweighted `MSELoss`; the scorer trainer uses BCE on win/loss labels.
- **New helper**, lives inline in `train_encoder.py`. For a batch with B cards and 9 heads, build `(B, 9)` weights and `(B, 9)` head_mask tensors. For each head h:
  ```
  L_h = sum( per_card_mse[:, h] * weights[:, h] * head_mask[:, h] )
        / max( sum(weights[:, h] * head_mask[:, h]), 1e-8 )
  ```
  `L_reg` is the unweighted sum of the four signed-head averages plus `(1/5)` times the sum of the five color-lift averages (FR-017). The `1e-8` clamp covers FR-017's "if a head's total sample weight in a batch is zero, that head's term in `L_reg` is zero for that batch" — when the numerator is also zero, any non-zero denominator yields zero contribution.

#### MLM masking (no prior art)

- The price-side and existing sealed encoders never train an MLM objective.
- **New helper**, lives inline in `train_encoder.py`. Given `input_ids: (B, T)` and `attention_mask: (B, T)`, draw a Bernoulli mask of probability `--mlm-mask-prob` over real (non-pad) positions where the original token is a non-special token (i.e., not `[PAD]`/`[UNK]`/`cardname`/`[MASK]`); replace those positions with the `[MASK]` ID; remember the mask positions and their original IDs for the cross-entropy loss. Loss is reduced only over masked positions; `mask.sum().clamp(min=1)` guards the denominator on minibatches where no position got masked.

#### `MatchResultWriter` (already extended in v1)

- `forge-connector/src/main/java/com/pricepredictor/connector/MatchResultWriter.java` writes one line per match, opens-writes-closes for each line so concurrent workers cannot corrupt the file.
- **Reuse**: The sibling `CardsPlayedWriter` (already implemented under v1, identical write strategy, distinct file path and line schema) ships unchanged. FR-001 is already satisfied.

#### `GameEventCardChangeZone` + `GameEventSpellAbilityCast` observer (already extended in v1)

- `GamePlayer.playMatch()` registers a per-game `IGameEventVisitor.Base<Void>` subclass via `game.subscribeToEvents(visitor)`. The observer captures both event types, applies four filters (`!isToken`, `!isBasicLand`, `controller == owner`, `gamePieceType == CARD`), and unions the results into the `Set<String>` returned with each `GameOutcome`. See `PlayedCardCollector` and `MatchGenerator.BASIC_LAND_NAMES`.
- **Reuse**: Already in production. FR-003, FR-004, FR-004a are all satisfied today.

#### `card_name_corrections` typo map

- `src/sealed/infrastructure/card_name_corrections.py` maps Forge filename typos to canonical names; `ConvertedCardLocator` already applies the corrections during lookup.
- **Reuse**: Pass 1's corpus-consistency check (FR-023d) goes through the locator and benefits automatically — a Forge-side typo will not falsely flag a missing card.

### Convention alignment

The sibling module to mirror is `sealed/` itself:

- Domain models live in `src/sealed/domain/*.py` (e.g., `scorer_model.py`, `encoder_model.py`, `card_encoder.py`).
- Use cases live in `src/sealed/application/*.py` and follow the pattern `<config dataclass> + run(config) -> result`. See `train_encoder.py:448` and `train_scorer.py` for the canonical shape (`run(config)`, `_log()` for stdout progress, custom exit-code carrying exceptions for pre-flight failures).
- Infrastructure adapters live in `src/sealed/infrastructure/*.py` and own all I/O (`encoder_store.py`, `cards_played_reader.py`, `match_data_loader.py`).
- CLI wiring lives in `src/sealed/infrastructure/cli.py`. Each subcommand has a private `_build_<name>_parser(subparsers)` and a module-level `run_<name>(args)` handler. `run_train_encoder` (line 1085) maps custom exceptions to exit codes.
- Tests mirror the source layout under `tests/unit/sealed/{domain, application, infrastructure}/test_*.py`. Java tests live under `forge-connector/src/test/java/.../`.

The new feature follows this layout exactly. No deviation.

### Third-instance check

- **Best-checkpoint pattern**: there are two instances (`_BestCheckpoint` in `price_predictor.application.train_transformer` and the v1 copy in `sealed.application.train_encoder`). `train_scorer.py` uses a *different* metric (val_acc, not val_loss) and doesn't fit the same mold. Two-and-a-bit instances; the existing follow-up note ("extract on the next loss-driven trainer") still stands. This feature does not introduce a third copy.
- **Encoder loaders**: still two (`load_model` in `transformer_store.py`; `SealedEncoderStore.load_encoder`). Below the extraction threshold; the dispatch helper in `cli._load_encoder_for_encode_cards` already abstracts the runtime choice for callers.
- **Per-batch per-head normalization** and **MLM mask draw**: zero prior instances each. Inline implementation is the right call.

## Decisions

### D-1: `--target-size` mapped to a post-hoc truncation, not freq-threshold tuning

**Decision**: `python -m sealed build-vocab --target-size N` calls
`build_vocabulary(corpus, freq_threshold=2)` (very inclusive), then truncates
the resulting vocab to the top-N entries by frequency, *always preserving*
the seeded special tokens (`[PAD]`, `[UNK]`, `cardname`, `[MASK]`), domain
seed terms, and set-code fragments — those slots count against the budget
but are never evicted. If `N` is smaller than the seed slot count, raise
`ValueError`.

**Rationale**: Spec uses `--target-size`; reads the default as
"approximately 5000 tokens." Three options were considered:
1. Rename to `--freq-threshold` — diverges from spec.
2. Binary-search freq-threshold to hit N tokens — multi-pass, complex,
   skewed by seeds.
3. Single-pass with post-hoc truncation (chosen) — one pass at
   `freq_threshold=2`, sort corpus tokens by frequency, keep top
   `(N - seed_count)`. Simple, deterministic, single-pass.

`[MASK]` joins the seeded specials so it's never evicted by a too-small
`--target-size`.

### D-2: Per-card label map computed inline, not exposed as a CLI command

**Decision**: Aggregation runs inside `train-encoder` as the first step.
There is no `python -m sealed aggregate-labels` subcommand. The
human-readable snapshot (`output/sealed/cards-win-rates.txt`, FR-013a) is
the only persistence.

**Rationale**: Spec FR-013 is explicit. Honors Simplicity First (Principle
II) — no auxiliary command until a real second consumer of the label map
exists.

### D-3: Java worker observes per-game card play via Forge eventbus (already shipped in v1)

**Decision** (already implemented): `GamePlayer.playMatch()` registers an
`IGameEventVisitor.Base<Void>` subclass via `game.subscribeToEvents(visitor)`.
The observer captures `GameEventCardChangeZone` (filtered to
`ZoneType.Battlefield`) and `GameEventSpellAbilityCast` (any cast), applies
four filters (controller==owner, !isToken, !isBasicLand, gamePieceType==CARD),
records `card.getPaperCard().getName()` to credit the cast card on copies/clones,
and emits two `Set<String>` per game.

**Rationale**: Direct port of the working pattern from
`../jumpstart-tierlist/.../JumpstartMatch.java#CardCollector`. Two events
needed because:
- Battlefield zone-changes miss instants/sorceries (graveyard/exile, not
  battlefield).
- Cast events miss lands (lands are played, not cast).

This decision has shipped and the implementation now lives in
`forge-connector/.../PlayedCardCollector.java`.

### D-4: Train/val split stratified by `score_play` quartile with FR-018 fallback

**Decision**: After computing the per-card label map, stratify cards into
four quantiles by their `score_play` cell value. For cards whose
`score_play` cell is empty (zero `@play` denominator), fall back to
whichever non-empty signed-head cell the card carries, in order:
`score_play` → `score_draw` → `cast_lift` → `color_lift_W` →
`color_lift_U` → `color_lift_B` → `color_lift_R` → `color_lift_G`. Cards
with no non-empty signed cell at all go into a single catch-all stratum.
Sample 20% of each stratum (rounded) into the validation set with
`random_seed=42`.

**Rationale**: FR-018 mandates stratification by `score_play` quartile and
defines the fallback chain. The any-signed-cell fallback is robust against
cards that have data on some heads but not on `@play` (e.g., a card only
ever observed in decks on the draw). The catch-all stratum handles
fully-degenerate cards (all signed cells empty) without dropping them —
they still have signal on heads where their cell is non-empty.

**Alternatives rejected**:
- Stratify by `wins_when_in_deck` (high vs low observation count): rejected
  because the model's signal is the label *value*, not the observation
  count, and the split should preserve the value distribution.
- Random 80/20 with no stratification: rejected because under a long-tail
  label distribution, a random val set can underrepresent
  high-winning-influence cards and inflate val loss noise.
- Drop cards without a `score_play` value: rejected — they still have
  trainable signal on the heads where their cells are non-empty.

### D-5: Encoder checkpoint format excludes regression heads AND the MLM head

**Decision**: `models/sealed/encoder/{timestamp}.pt` contains a torch
`.pt` payload with two keys: `model_state_dict` (only token-encoder +
card-encoder weights — `regression_heads.*` and `mlm_head.*` filtered out)
and `config` (a serialized `SealedEncoderConfig` dataclass). `latest.pt`
is a copy (not a symlink) for Windows portability.

**Rationale**: FR-020 requires "the regression heads' weights and the MLM
head's weights MUST NOT be written." `SealedEncoderStore._ENCODER_PREFIXES`
already filters the state-dict by prefix — adding new heads under
`regression_heads.*` / `mlm_head.*` keeps them outside the filter
automatically. No save-side code change needed.

**Alternatives rejected**:
- Save full state-dict and drop heads at load time: rejected — a reader
  would have to know which keys to drop, and the artifact would carry
  training-only weights (a 5000-entry MLM head matrix is not negligible).
- Save model + tokenizer config + vocab in one file: rejected — vocab
  already lives at `models/sealed/encoder/vocab.txt` and the project
  convention is to keep them separate.

### D-6: `--encoder-checkpoint` default flip is a single-line config change (already shipped in v1)

**Decision** (already implemented): `TrainScorerConfig.encoder_checkpoint`
defaults to `Path("models/sealed/encoder/latest.pt")`;
`_ENCODE_CARDS_DEFAULT_ENCODER` in `cli.py` matches. A missing-file check
(FR-026) raises a clear error naming `python -m sealed train-encoder`.

**Rationale**: Defaults are isolated to two locations; explicit
`--encoder-checkpoint <path>` continues to work (FR-027).

### D-7: `cards-played.txt` write happens in Java; Python supervisor unchanged (already shipped in v1)

**Decision** (already implemented): The Java match worker writes both
`match-outcomes.txt` (via `MatchResultWriter`) and `cards-played.txt` (via
`CardsPlayedWriter`). The Python supervisor (`match_outcomes.py`) is
untouched.

**Rationale**: Per-game card play is observable only inside the Forge JVM.
Writing from Java avoids serializing `Set<String>` data over the supervisor
stdout channel and mirrors the existing per-match write pattern.

### D-8: Heads structure — `nn.ModuleDict` of five linear projections

**Decision**: The five regression-head families live in one `nn.ModuleDict`
attribute on `SealedEncoderModel`, keyed by head name. Four are
`Linear(2*d_model, 1)` (`score_play`, `score_draw`, `played_rate`,
`cast_lift`); one is `Linear(2*d_model, 5)` (`color_lift`, with one column
per WUBRG letter).

**Rationale**: A single `Linear(2*d_model, 9)` would fuse all heads but
make per-head sample weighting awkward (the gradient still flows through
every row of the weight matrix even when a head's mask is zero). Per-head
modules give clean per-head parameter groups for future per-head LR tuning
if it ever arrives, and they make state-dict prefix filtering trivial. An
`nn.ModuleList` would work but `ModuleDict` makes the head-name → tensor
mapping explicit and self-documenting.

**Alternatives rejected**:
- Single fused `Linear(2*d_model, 9)`: gradient leakage through zero-mask
  heads, harder to extend per-head later.
- Five separate top-level attributes (`self.score_play_head`, ...): the
  `_ENCODER_PREFIXES` filter on `SealedEncoderStore` would need a longer
  prefix list, vs. the single new prefix `regression_heads.` for the
  dict-based version.

### D-9: MLM head reads pre-pool token sequence

**Decision**: The MLM head reads the output of the transformer-layer stack
(shape `(B, T, d_model)`), *before* the pool layer. The pool layer's input
becomes a tee in `forward()`: one branch goes into the pool → regression
heads; the other branch goes into the MLM head.

**Rationale**: FR-015a is explicit. Pooling discards token-level
information that the MLM head needs, so reading post-pool is impossible.
Reading pre-stack (raw token+pos embeddings) would defeat the purpose —
the MLM head exists to push gradient through the transformer layers
themselves.

**Alternatives rejected**: None — the spec pins this.

### D-10: Mask draw — per-step, non-special tokens only, all-MASK replacement

**Decision**: Mask draw runs in `_train_epoch` for every batch (different
mask per epoch per card). Eligible positions: real (non-pad) tokens whose
ID is not one of the specials (`[PAD]`, `[UNK]`, `cardname`, `[MASK]`).
Probability: `--mlm-mask-prob` (default 0.15). All eligible drawn positions
are replaced with `[MASK]` (no 80/10/10 BERT-style stochastic schedule).

**Rationale**: FR-014b says "approximately `--mlm-mask-prob` fraction of
each card's non-special tokens are replaced with the reserved `[MASK]`
token". "Non-special" is interpreted to exclude every reserved token,
which is the natural reading and matches what the spec considers
signal-bearing. The 80/10/10 BERT schedule is a regularization detail not
mandated by the spec and not necessary at this scale; if a future ablation
needs it, that's a follow-up. The mask is drawn fresh each forward pass
so the same card sees many distinct masks across an epoch.

**Alternatives rejected**:
- 80/10/10 BERT schedule: rejected for simplicity; revisitable if MLM
  regularization underperforms.
- Mask at dataset level once per card per epoch: rejected because per-step
  variability is cheap and gives strictly more diverse gradients.

### D-11: Per-batch per-head sum-to-1 weight normalization with safe denominator

**Decision**: Per-head per-batch sum-to-1 weighted average (formula in
"Adjacent prior art § Per-batch per-head sum-to-1 weight normalization"
above). `1e-8` clamp on the denominator prevents NaN when no card in a
batch contributes to a head; the head_mask zeroes the numerator in the
same case so the result is exactly 0.

**Rationale**: Clarification 2026-05-10 selected per-head per-batch
sum-to-1. The `1e-8` clamp is a numerical safety net, not a result-shaping
decision.

**Alternatives rejected**:
- Per-epoch sum-to-1: more bookkeeping, and per-batch is already the right
  denoising granularity given gradient updates happen per-batch.
- Raw weights (no normalization): rejected by the clarification — the
  goal is `--lr` and `--mlm-weight` stability across batches with varying
  weight totals.

### D-12: Best-checkpoint metric includes the MLM term

**Decision**: Validation `loss = L_reg_val + (--mlm-weight) * L_mlm_val`.
The same loss drives both `_BestCheckpoint.update` and `--patience` early
stopping (FR-019, Clarification 2026-05-10).

**Rationale**: The user explicitly chose Option B in the clarification —
treating the auxiliary objective as part of the encoder's optimization
target rather than as a discardable regularizer.

**Alternatives rejected**: Option A (regression-only val loss) was the
recommended option but the user picked B. No further analysis needed.

### D-13: Two-pass aggregator — pass 1 streams; consistency check; pass 2 streams again

**Decision**: Pass 1 makes one full streaming sweep through
`cards-played.txt` and accumulates 8 counters per card (4 primary + 4
`@play`). The corpus-consistency check (FR-023d) runs between passes 1
and 2. Pass 2 makes a second full streaming sweep, looks up each card's
color set on first encounter (cached in a dict), computes each game's
deck-color set as the union over each side's deck contents, and
accumulates 4 counters × 5 colors per card.

**Rationale**: Two separate sweeps keep each pass's logic narrow (pass 1
needs only `winner`/`starter`; pass 2 needs deck content + color
identity). The cards-played file is line-oriented and read-once per pass —
both passes are O(rows × cards-per-row). Memory stays bounded by the
per-card counter dict, not by the file size. Running the consistency
check between passes lets the error message name missing cards before
any pass-2 work happens.

**Alternatives rejected**:
- Single pass with both kinds of counters: rejected — pass 2 needs pass
  1's output (the consistency check uses the "cards seen in the file" set
  to detect missing-corpus cards before doing color work).
- Pre-aggregate to an on-disk intermediate: rejected per FR-013.

### D-14: Optimizer = AdamW + per-parameter-group max-norm 1.0 gradient clipping

**Decision** (Clarification 2026-05-10): The `train-encoder` optimizer is
`torch.optim.AdamW` with per-parameter-group max-norm 1.0 gradient
clipping. With one parameter group (encoder + heads + MLM share one
optimizer state), this is a single `clip_grad_norm_(model.parameters(),
max_norm=1.0)` call between `loss.backward()` and `optimizer.step()`.

**Rationale**: Matches the existing `train-scorer` convention (spec 015,
FR-005a). Keeps cross-trainer behavior consistent. The current
`train_encoder.py:386` already constructs `torch.optim.AdamW`; only the
clipping call needs adding.

**Alternatives rejected**:
- Adam (no weight decay): rejected by clarification.
- SGD + momentum: rejected by clarification.
- No clipping: rejected — random-init transformer training is sensitive
  to early-step gradient spikes.

### D-15: `cards-win-rates.txt` schema — header row + 24 columns + sort by `shrunk_score_play`

**Decision** (FR-013a + Clarification 2026-05-10): The file begins with
one header row naming each column in declaration order, followed by N
data rows sorted by `shrunk_score_play` descending. Columns:

```
card_name; wins_when_played; wins_when_in_deck; losses_when_played; losses_when_in_deck;
raw_score_play; shrunk_score_play; raw_score_draw; shrunk_score_draw;
raw_played_rate; shrunk_played_rate; raw_cast_lift; shrunk_cast_lift;
raw_color_lift_W; shrunk_color_lift_W; raw_color_lift_U; shrunk_color_lift_U;
raw_color_lift_B; shrunk_color_lift_B; raw_color_lift_R; shrunk_color_lift_R;
raw_color_lift_G; shrunk_color_lift_G
```

(That's 1 (`card_name`) + 4 (primary counters) + 2×9 = 18 (raw/shrunk
per head) = **23 fields per row**.) Cells whose slice denominator is
zero are written
as the empty string in both the raw and shrunk columns.

**Rationale**: The clarification picked Option A (header row included);
FR-013a defines the column order and zero-denominator handling. Sorting
by `shrunk_score_play` descending puts the highest winning-influence
cards first, which is the natural inspection order for SC-005.

**Alternatives rejected**:
- No header row: rejected by clarification.
- Sort by raw vs shrunk: shrunk is the value the model actually trains
  against, so sorting by shrunk is the more useful inspection key.

### D-16: Color extraction — regex over `{...}` symbols on the `mana cost:` line

**Decision**: A 5-line helper `_colors_from_mana_cost(line)` scans the
line for `\{[^}]+\}` matches and union-collects the WUBRG letters they
contain. Hybrid (`{W/U}`), Phyrexian (`{W/P}`), and mono-hybrid (`{2/W}`)
all contribute. Generic (`{2}`), colorless (`{C}`), and `X` (`{X}`)
contribute nothing. Cards whose converted text has no `mana cost:` line
contribute no colors.

**Rationale**: Matches FR-010b verbatim. The regex+letter-set approach
sidesteps enumerating every mana symbol shape.

**Alternatives rejected**:
- Use `ManaCost` value object from `price_predictor.domain.value_objects`:
  the existing parser handles cost arithmetic but is overkill for the
  "which colors does this cost contain" question, and would create a
  cross-package dependency for a one-time-per-card lookup.
