# `sealed` — sealed-format ML pipeline

Entry point: `python -m sealed <subcommand>` (see `src/sealed/infrastructure/cli.py`). The package's domain terms are defined in `GLOSSARY.md` beside this file, and the terms it shares with other packages in the root `GLOSSARY.md`. The sealed pipeline is built around a sealed-trained card encoder; the price-predictor transformer is a valid alternate encoder source via `--encoder-checkpoint`. `sealed` imports from `price_predictor` only (the tokenizer, the vocabulary builder, the transformer model and store); never create the reverse dependency.

This file lists the rules to respect and the file formats. It does not describe the commands:

- Flags, defaults and what each run logs: each subcommand's `--help`.
- What each stage specifies: `specs/2026-03-28-sealed-deck-picker.md`, `specs/2026-04-23-distributed-match-generation.md`, `specs/2026-04-27-encoder-fine-tuning.md`, `specs/2026-05-03-card-winnability-pretraining.md`, `specs/2026-05-19-one-shot-deck-picker.md`.
- Why, and every benchmark: the sealed records under `experiments/`.

## Topic files — read on demand

None of these loads by itself. Read a topic file in full when its condition is met.

| Read | Before |
|---|---|
| `docs/embedding-cache.md` | running or changing `encode-cards`, or anything that reads the `.npz` card embeddings |
| `docs/match-outcomes.md` | running or changing `match-outcomes`, its supervisor or its workers |
| `docs/training.md` | running or changing `build-vocab`, `train-encoder`, `train-scorer` or `train-picker` |

## Subcommands

- `build-vocab` — writes the sealed-side tokenizer vocabulary (`docs/training.md`).
- `train-encoder` — trains the card encoder on per-card winnability labels aggregated from `cards-played.txt`, with an MLM auxiliary loss (`docs/training.md`).
- `encode-cards` — writes one `.npz` embedding per converted card, beside it or in the `--embeddings-path` tree (`docs/embedding-cache.md`).
- `generate-pools` — Java `PoolMain`; writes sealed pools (§ Pools and decks).
- `build-decks` — the scorer greedily builds one 40-card deck per pool (§ Pools and decks).
- `match-outcomes` — long-running supervisor over Java `MatchWorkerMain` workers; plays matches and appends the two corpora (`docs/match-outcomes.md`).
- `train-scorer` — trains the Set Transformer deck scorer on `match-outcomes.txt` (`docs/training.md`).
- `evaluate-scorer` — plays the scorer's decks against Forge's own best builder on fresh pools.
- `train-picker` — trains the one-shot deck picker by REINFORCE against a frozen scorer (`docs/training.md`).
- `pick-decks` — the picker's inference counterpart to `build-decks` (§ Pools and decks).
- `analyze-generated-decks` — composition statistics over generated-decks files.

## Rules for any work in the package

- `sealed` never imports `effects`. The two places they meet are plain data: the `--exclude-cards` name list that `python -m effects holdout-cards` writes, and the collection caps, whose values, defaults and property names are defined once in `domain/collection_caps.py` (`CollectionCaps`) and whose flags are in `infrastructure/effect_collection_flags.py`, both imported by `effects`.
- The composition engine in `application/analyze_generated_decks.py` is reused by `draft analyze-generated-decks`.
- Every subcommand that writes files keeps a `run.json` run manifest in its run directory (`.claude/skills/artifact-layout/SKILL.md`), declared in `RUN_MANIFESTS` in `infrastructure/cli.py`: `match-outcomes` records in `--effect-records` when given, else in `--outcomes-dir`, and `encode-cards` in its `--embeddings-path` directory, else in the `--cards-path` tree it writes beside.
- Paths follow the `artifact-layout` skill: every input and output path is a required flag, and no flag defaults to a location under `output/` or `models/`. A resumable trainer (`train-scorer`, `train-picker`) inherits the paths it needs from its `--resume` checkpoint, and refuses a fresh run without them (exit 2, naming the flags). `encode-cards` takes exactly one of `--encoder-checkpoint` and `--scorer-checkpoint`; `match-outcomes` takes `--outcomes-dir` or `--effect-records`.
- The scorer's and the picker's input width is the width of the `.npz` cache, never a flag (`docs/embedding-cache.md`).

## Pools and decks

- `--exclude-cards <file>` names cards no booster may contain. Each is redrawn within its own rarity slot, so pool size and rarity structure are unchanged.
- On `generate-pools` the flag depletes a pools file, for the consumers that read one. The effect model's depleted collection run needs the same flag on `match-outcomes`, whose workers open their own pool per match and read no pools file.
- `--label` is required on `build-decks` and `pick-decks`. It is read back by `match-outcomes` as the `method_A`/`method_B` tag of every match that plays the deck.
- `pick-decks` skips pools with fewer than 23 embeddable cards.

## Corpus file formats

Append-only data contracts spanning the Java writers and the Python readers — no single source file teaches a whole one, and a mismatch silently corrupts a corpus that cannot easily be rebuilt.

**Match outcomes** (`match-outcomes.txt` in `match-outcomes --outcomes-dir`): one line per match, ten semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s`. `timestamp` is ISO 8601 UTC, `run_id` is a UUID identifying the supervisor invocation, `set_code` is the MTG set both pools were drawn from, `method_A`/`method_B` are build-method tags (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or the per-deck label set by `build-decks --label` when the deck came from a generated-decks file), `deck_A`/`deck_B` are pipe-separated lists of 40 Forge canonical card names (duplicates repeat), `games` is the per-game winner sequence (`AA`, `ABA`, `ABABABA`, etc. — length depends on `--best-of`, between `ceil(N/2)` and `N` chars), `play` is the per-game play-first sequence (same length), and `duration_s` is the match wall-clock duration in whole seconds. See `specs/2026-03-28-sealed-deck-picker.md` §Phase 0 Step 4 for full details.

**Cards played** (`cards-played.txt`, beside `match-outcomes.txt`): one line per played game, eleven semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter`. `timestamp`/`run_id`/`set_code`/`method_A`/`method_B` mirror the parent match's row (joinable by `run_id` plus positional offset within a `(run_id, set_code, method_A, method_B)` group). The four card-list columns are pipe-separated **sets** of distinct card names (no duplicates within a column; basic lands excluded by the writer). For a side X, `cards_played_X` and `cards_not_played_X` are disjoint — if any copy of a card was played, the name appears only in `cards_played_X`. `winner` and `starter` are single chars `A`/`B` matching the corresponding chars of the parent's `games`/`play` strings. Trailing partial lines are tolerated by readers (JVM-crash-mid-write recovery).

**Pools** (`pools.txt` in `generate-pools --pools-path`): one pool per line, `SET_CODE;Card1|Card2|...|CardN`, basics excluded. The set-code prefix lets downstream tools (`build-decks`, self-play `match-outcomes`) honor same-set constraints.

**Generated decks** (`build-decks --output`, `pick-decks --output`): one finished 40-card deck per line, `LABEL;SET_CODE;Card1|Card2|...|Card40`. Concatenating several generated-decks files with different labels into one self-play corpus is supported.

**Cards win rates** (`train-encoder --win-rates-output`): a Python-only snapshot for inspection, with a header row. A cell whose slice denominator is zero is written as the empty string in both its raw and shrunk columns, which distinguishes "no signal" from "neutral signal". Cards never seen in a deck are excluded.

## Model artifacts

Each training run writes its checkpoints (a rolling `latest` beside the trainer's own best or timestamped files) to its run directory, `models/sealed/<gen>/<model>/<run>/` (`encoder`, `scorer`, `picker`): `train-encoder --model-output`, `train-scorer --checkpoint-dir`, `train-picker --checkpoint-dir`.
