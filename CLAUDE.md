# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An ML system that predicts Magic: The Gathering card EUR prices from game-visible attributes (mana cost, types, oracle text, power/toughness, keywords, printing metadata). Works for both real cards and hypothetical ones. Also hosts a sealed-format ML pipeline (card embeddings, pool generation, deck scorer) built on top of the price predictor's transformer encoder.

## Tech stack

- Sibling checkout of MTG Forge expected at `../forge` (built with `mvn install -DskipTests`). MTGJSON data files (`AllPrintings.json`, `AllPricesToday.json`) expected in `resources/`.

## Common commands

```bash
# Setup — PyTorch needs the CUDA 12.6 wheel index
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126

# Fat JAR required by convert / generate-pools / match-outcomes / generate-draft-data
cd forge-connector && mvn package -DskipTests
```

## Architecture

Three Python packages live under `src/`, each laid out in hexagonal (ports-and-adapters) style: `domain` → `application` → `infrastructure`. Per-package detail lives next to the code and loads when you work there:

- `src/price_predictor/CLAUDE.md` — price prediction (sklearn + transformer, `convert`, `vocabulary`, `serve`).
- `src/sealed/CLAUDE.md` — sealed pipeline (encoder, scorer, picker, self-play match generation) and its file formats.
- `src/draft/CLAUDE.md` — draft agent (gen-1 imitation, gen-2 RL, gen-3 online GRPO) and `drafts.jsonl`.

### `forge-connector` — Java Maven module

Zero-dependency (stdlib-only) Java 17+ library at `forge-connector/` with two roles:

1. **Client library for Forge**: `PricePredictorClient` + `CardAttributes` give Forge a 5-line API for hitting the `POST /api/v1/predict` endpoint. Used by Forge's deck-building heuristics.
2. **CLI workers invoked by the Python side**: fat JAR built with `mvn package -DskipTests` → `target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar`. Main classes invoked by Python subprocess:
   - `ConvertMain` — `python -m price_predictor convert`
   - `PoolMain` — `python -m sealed generate-pools`
   - `MatchWorkerMain` — one per worker in `python -m sealed match-outcomes`
   - `ValidationWorkerMain` — `python -m sealed evaluate-scorer`
   - `DeckBuilderMain` — used during scorer evaluation to build decks from pools
   - `DraftWorkerMain` — one per worker in `python -m draft generate-draft-data` (drives Forge's `BoosterDraft`/`LimitedPlayerAI` for all pod seats, streams one transcript per draft)

These workers import `forge-game` / `forge-core` from the sibling `../forge` checkout (classpath assembled in `run_convert` in `infrastructure/cli.py`). `ForgeEnvironmentInitializer` bootstraps Forge's static state in each JVM.

### Cross-package dependencies

`sealed` **imports from** `price_predictor` (tokenizer, vocabulary builder, transformer model/store) — `price_predictor` provides the shared MTG tokenizer, vocab-build utility, and the alternate price-trained encoder available via `--encoder-checkpoint`. Don't create the reverse dependency.

`draft` **imports from** `sealed` and `price_predictor` (scorer, picker, greedy builder, `deck_assembly`, `score_decks`, `ConvertedCardLocator`, card-embedding layout, `torch_checkpoint`, `torch_training`, `forge_jvm` worker helpers) — never the reverse. `draft` adds only the genuinely-new logic (booster→state geometry, typed-token state, the two-headed model, the Java draft worker).

## Corpus file formats

Append-only data contracts spanning the Java writers and the Python readers — no single source file teaches a whole one, and a mismatch silently corrupts a corpus that cannot easily be rebuilt.

Match-outcome file format (`output/sealed/match-outcomes.txt`): one line per match, ten semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s`. `timestamp` is ISO 8601 UTC, `run_id` is a UUID identifying the supervisor invocation, `set_code` is the MTG set both pools were drawn from, `method_A`/`method_B` are build-method tags (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or the per-deck label set by `build-decks --label` when the deck came from a generated-decks file), `deck_A`/`deck_B` are pipe-separated lists of 40 Forge canonical card names (duplicates repeat), `games` is the per-game winner sequence (`AA`, `ABA`, `ABABABA`, etc. — length depends on `--best-of`, between `ceil(N/2)` and `N` chars), `play` is the per-game play-first sequence (same length), and `duration_s` is the match wall-clock duration in whole seconds. See `specs/2026-03-28-sealed-deck-picker.md` §Phase 0 Step 4 for full details.

Cards-played file format (`output/sealed/cards-played.txt`): one line per played game, eleven semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter`. `timestamp`/`run_id`/`set_code`/`method_A`/`method_B` mirror the parent match's row (joinable by `run_id` plus positional offset within a `(run_id, set_code, method_A, method_B)` group). The four card-list columns are pipe-separated **sets** of distinct card names (no duplicates within a column; basic lands excluded by the writer). For a side X, `cards_played_X` and `cards_not_played_X` are disjoint — if any copy of a card was played, the name appears only in `cards_played_X`. `winner` and `starter` are single chars `A`/`B` matching the corresponding chars of the parent's `games`/`play` strings. Trailing partial lines are tolerated by readers (JVM-crash-mid-write recovery).

Cards-win-rates file format (`output/sealed/cards-win-rates.txt`): overwritten by every `train-encoder` run; one header row + N data rows, semicolon-separated, sorted by `shrunk_score_play` descending (cards with empty `shrunk_score_play` sort to the end). Schema (23 columns): `card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G`. Floats formatted to five decimals; counters as plain integers. Cells whose slice denominator is zero are written as the empty string in both raw and shrunk columns (distinguishes "no signal" from "neutral signal"). Cards with `wins_when_in_deck + losses_when_in_deck == 0` are excluded entirely.

Pool file format (`output/sealed/pools/*/pools.txt`): one pool per line, `SET_CODE;Card1|Card2|...|CardN`. The set-code prefix lets downstream tools (`build-decks`, self-play `match-outcomes`) honor same-set constraints.

Generated-decks file format (`output/sealed/generated-decks.txt`): one finished 40-card deck per line, `LABEL;SET_CODE;Card1|Card2|...|Card40`. `LABEL` is the value passed to `build-decks --label` and is recorded as the `method_A` / `method_B` tag whenever this deck is sampled into a self-play match. Concatenating multiple generated-decks files with different labels into one self-play corpus is supported.

Drafts file format (`output/draft/drafts.jsonl`): one self-contained JSON record per line, append-only; readers tolerate a trailing partial final line. Each record has `draft_id`, `run_id`, `timestamp` (ISO 8601 UTC), `seats` (length `pod_size`; each `{agent, deck (40 names incl. basics, or [] on failed build), deck_score (scorer scalar, or null on failed build)}`), and `boosters` (length `pod_size × packs`; each `{set_code, picks}` where `picks` is the cards drafted from that physical pack in pick order, fully drained to `pack_size`). The booster *ordering* pins all geometry: `pack_number(k)=⌊k/pod_size⌋+1`, `opening_seat(k)=k mod pod_size`, and the pick at offset `j` of `boosters[k]` was made by seat `(opening_seat + j·dir_p) mod pod_size` with `dir_p=+1` for packs 1 & 3 and `−1` for pack 2 — so any seat's full observation history reconstructs from the record alone (`draft/domain/draft_geometry.py`).

## Model artifact layout

Checkpoints live under `models/{price-predictor,sealed,draft}/`; every trainer writes a `{timestamp}` file plus a rolling `latest`. Best-checkpoint selection differs per model — scorer by val accuracy, picker by val reward, draft agent by val loss (gen-3 online: by anchor margin). See each package's `CLAUDE.md` and its store module.

Inputs the code expects to find on disk:
- `resources/AllPrintings.json`, `resources/AllPricesToday.json` — MTGJSON dumps.
- `../forge/forge-gui/res/cardsfolder/` — Forge card scripts (source for `convert`).
- `output/cardsfolder/` — converted card text files, each paired with a `.npz` after `encode-cards` runs.
- `output/sealed/pools/{set}/pools.txt` or `output/sealed/pools/pools.txt` — generated sealed pools (`SET_CODE;Card1|...` per line).
- `output/sealed/generated-decks.txt` — scorer-built 40-card decks from `build-decks` (`LABEL;SET_CODE;Card1|...|Card40` per line); input to `match-outcomes --side-a-decks` / `--side-b-decks`.
- `output/sealed/match-outcomes.txt` — append-only training data for the scorer.
- `output/sealed/cards-played.txt` — append-only per-game card-play log; the input to `train-encoder` after label aggregation.
- `output/draft/drafts.jsonl` — append-only labeled draft corpus from `generate-draft-data` (one self-contained JSON record per line); the input to `train-draft-agent` and (per-cycle, e.g. `drafts-genK.jsonl`) the on-policy corpus for `train-draft-agent-rl`, and the file `train-draft-agent-online` appends its streaming self-play rollouts to. Schema unchanged across gen-1/gen-2/gen-3 — RL pairs a corpus to its generating checkpoint by operator convention, not a stored field; gen-3 needs no pairing at all, being on-policy by construction.

## Specs, experiments & the feature workflow

Three kinds of design document, each with its own audience and location — keep them in their lanes:

- `experiments/*.md` — what happened and why; the **only** home for benchmark numbers, run logs, and rejected alternatives.
- `specs/YYYY-MM-DD-<name>.md` — hand-written normative specs (WHAT, not WHY), timeless present tense.
- `specs/NNN-name/` — spec-kit feature dirs (`spec.md` → `plan.md` → `research.md` → `tasks.md`) driving implementation.

Load the **feature-workflow** skill before writing or amending any of them. Before starting non-trivial work in an area, read the relevant `spec.md` / `plan.md` / `research.md`.
