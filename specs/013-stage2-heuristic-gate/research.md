# Research: Stage 2 Training — Heuristic Gate

**Feature**: 013-stage2-heuristic-gate | **Date**: 2026-03-31

## 1. Actual Card File Format (verified against output/)

**Decision**: Mana abilities use the `activated[N]:` prefix, matching FR-008.

**Rationale**: The actual converted card files in `output/` use `activated[N]: {T}: add ...` for
mana abilities (all lowercase). Verified against multiple cards:
- Island: `activated[1]: {T}: add {U}`
- Plains: `activated[1]: {T}: add {W}`
- Breeding Pool: `activated[1]: {T}: add {G} or {U}`
- Jungle Shrine (tri-land): `activated[1]: {T}: add {R}, {G}, or {W}.`
- Sol Ring: `activated[1]: {T}: add {C}{C}.`
- Wastes: `activated[1]: {T}: add {C}.`

Note: test fixtures under `tests/fixtures/converted_cards/` use a different format (`mana[N]:`)
which does NOT match production output. Implementation MUST be tested against the actual format.

**Key format observations**:
- All text is lowercase in actual card files
- Dual lands use "or" syntax: `add {G} or {U}` (not separate lines per color)
- Tri-lands use comma+or: `add {R}, {G}, or {W}.`
- Some abilities have trailing periods, some don't — parser must handle both
- Non-mana activated abilities also use `activated[N]:` but with different cost patterns
  (e.g., `{4}{W}: CARDNAME becomes...`). Only match `{T}: add` pattern per FR-008.
- "Add one mana of any color" has no color symbols → contributes +0 per FR-008 (counts
  "each color symbol that appears in its 'Add' clause")

**Phyrexian mana examples** (in mana cost lines):
- `mana cost: {3}{R/P}{R/P}` (Act of Aggression)
- `mana cost: {1}{W/P}` (Apostle's Blessing)

**Hybrid mana examples** (in mana cost lines):
- `mana cost: {3}{W/U}` (Aethertow)

**Multi-face card format**:
- `ALTERNATE` line separates faces
- `layout:` line at top (e.g., `layout: transform`, `layout: split`)
- Each face has its own `name:`, `mana cost:`, `types:`, etc.
- Transform back face typically has no mana cost
- Split cards have mana cost on each half (e.g., Fire // Ice: `{1}{R}` and `{1}{U}`)

## 2. Reuse Strategy for EpisodeRunner and PPOTrainer

**Decision**: Reuse both unchanged. Modify episode rewards after the episode completes in the
application layer.

**Rationale**: EpisodeRunner always produces per-step rewards (+1/-1 based on spell/land budgets)
and handles duplicate-pick termination. These behaviours are correct for both Stage 1 and Stage 2:
- **Completed episodes** (Stage 2): overwrite `episode.step_rewards` with uniform mana-score reward
  and `episode.reward` with the mapped score after the episode runs.
- **Terminated episodes** (Stage 2): keep the Stage 1 per-step rewards as-is (FR-003).

PPOTrainer reads `step_rewards` from Episode objects and normalises advantages across the batch
(FR-011). It doesn't know or care whether the rewards are per-step or uniform — the math is the
same. Terminal picks use -1.0 advantage regardless of stage.

**Alternatives considered**:
- Subclassing EpisodeRunner with a Stage 2 variant — rejected: unnecessary complexity since the
  episode mechanics (shuffling, picking, duplicate detection) are identical.
- Adding a `reward_fn` callback to EpisodeRunner — rejected: YAGNI, only two stages exist.

## 3. Embedding Adapter Extraction

**Decision**: Extract `_EmbeddingAdapter` from `train_stage1.py` to
`infrastructure/embedding_adapter.py` as public `EmbeddingAdapter`.

**Rationale**: Currently used by `train_stage1.py` and `sample_stage1.py` (which imports the
private class cross-module). Stage 2 adds two more consumers. Four use cases justifies extraction
per Constitution principle II (three concrete use cases threshold). The adapter wraps
`EmbeddingStore` (infrastructure) and implements `CardEmbeddingPort` (domain protocol), so it
belongs in the infrastructure layer.

**Alternatives considered**: Keeping it in `train_stage1.py` — rejected: four consumers of a
private class is a code smell. The underscore prefix implies internal use.

## 4. Card Text Access for Mana Analysis

**Decision**: Add `get_card_text(card_name: str) -> str` to `CardEmbeddingPort` protocol and
implement it in `EmbeddingAdapter` with caching.

**Rationale**: The mana scorer needs raw card text to extract `mana cost:` and `activated[N]:`
lines. The adapter already reads .txt files for `is_land()`. Adding `get_card_text()` with a cache
follows the same pattern. The domain mana_scorer module receives raw text and applies parsing rules
(MTG mana symbol semantics), keeping I/O in the adapter and domain logic in the domain layer.

**Alternatives considered**:
- Returning structured data (parsed mana costs, abilities) from the port — rejected: moves domain
  parsing logic into infrastructure.
- Having the application layer read files directly — rejected: bypasses the port/adapter pattern.

## 5. Stage 2 Training State

**Decision**: Reuse existing `TrainingState` dataclass with `best_run=MAX_PICKS` (fixed at 40)
and `episode_count=0` at Stage 2 start.

**Rationale**: Stage 2 has no curriculum advancement — `best_run` is always 40. The existing
`TrainingState(best_run, episode_count)` fits this model. When loading from `--init-from`, only
model weights are loaded; a fresh `TrainingState(best_run=40, episode_count=0)` is created
(clarification from spec session). When resuming from a Stage 2 checkpoint, the full state is loaded
as in Stage 1.

**Alternatives considered**: Creating a `Stage2TrainingState` — rejected: fields are identical,
no additional state needed for Stage 2.

## 6. CLI `--init-from` and Checkpoint Priority

**Decision**: Add `--init-from` argument to the train subcommand. When `--stage 2`:
- If `--model-path` exists: resume from it (ignore `--init-from`).
- If `--model-path` does not exist and `--init-from` exists: load model weights only, fresh
  optimizer and episode_count.
- If neither exists: error.

**Rationale**: This follows the natural priority: an existing Stage 2 checkpoint takes precedence
over re-initialising from Stage 1. Matches the pattern users expect from Stage 1 (where
`--model-path` existence triggers resume). The `--init-from` default is
`models/sealed/stage1/{set}/latest.pt` and `--model-path` default for Stage 2 is
`models/sealed/stage2/{set}/latest.pt`.

**Alternatives considered**: Always loading from `--init-from` — rejected: would prevent resuming
an interrupted Stage 2 run.

## 7. Mana Source Parsing Pattern

**Decision**: Match `activated[N]: {T}: add ...` lines where the cost portion is exactly `{T}`.
Extract color symbols from the "add" clause using regex `\{([WUBRGC])\}`.

**Rationale**: Not all `activated[N]:` lines are mana abilities. Some have additional costs
(e.g., `{4}{W}:`, `{T}, sacrifice CARDNAME:`, `{1}, {T}:`). Only `{T}: add` matches the
tap-for-mana pattern described in FR-008.

**Color extraction from Add clause**:
- `add {W}` → W
- `add {G} or {U}` → G, U
- `add {R}, {G}, or {W}` → R, G, W
- `add {C}{C}` → C, C (but counted as +1 C per FR-008: "each color symbol")
- `add one mana of any color` → no color symbols → +0 (literal reading of FR-008)

**Regex**: `r"activated\[\d+\]:\s*\{T\}:\s*add\s+(.+)"` to capture the add clause, then
`r"\{([WUBRGC])\}"` to extract color symbols from it. Use `set()` on results since each
distinct color contributes +1 (Sol Ring's `{C}{C}` = +1 C, not +2).
