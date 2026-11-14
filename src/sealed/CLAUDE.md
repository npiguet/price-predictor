# `sealed` — sealed-format ML pipeline

Entry point: `python -m sealed <subcommand>` (see `src/sealed/infrastructure/cli.py`). The package's domain terms are defined in `GLOSSARY.md` beside this file, and the terms it shares with other packages in the root `GLOSSARY.md`. The sealed pipeline is built around a sealed-trained card encoder; the price-predictor transformer is a valid alternate encoder source via `--encoder-checkpoint`.

Every subcommand that writes files keeps a `run.json` run manifest in its run directory (`.claude/skills/artifact-layout/SKILL.md`), declared in `RUN_MANIFESTS` in `infrastructure/cli.py`: `match-outcomes` records in `--effect-records` when given, else in `output/sealed/`, and `encode-cards` in the `--cards-path` tree it writes beside.

This file lists the rules to respect and the file formats. It does not describe the commands:

- Flags, defaults and what each run logs: each subcommand's `--help`.
- What each stage specifies: `specs/2026-03-28-sealed-deck-picker.md`, `specs/2026-04-23-distributed-match-generation.md`, `specs/2026-04-27-encoder-fine-tuning.md`, `specs/2026-05-03-card-winnability-pretraining.md`, `specs/2026-05-19-one-shot-deck-picker.md`.
- Why, and every benchmark: the sealed records under `experiments/`.

## Subcommands

- `build-vocab` — writes the sealed-side tokenizer vocabulary.
- `train-encoder` — trains the card encoder on per-card winnability labels aggregated from `cards-played.txt`, with an MLM auxiliary loss.
- `encode-cards` — writes one `.npz` embedding beside every converted card.
- `generate-pools` — Java `PoolMain`; writes sealed pools.
- `build-decks` — the scorer greedily builds one 40-card deck per pool.
- `match-outcomes` — long-running supervisor over Java `MatchWorkerMain` workers; plays matches and appends the two corpora.
- `train-scorer` — trains the Set Transformer deck scorer on `match-outcomes.txt`.
- `evaluate-scorer` — plays the scorer's decks against Forge's own best builder on fresh pools.
- `train-picker` — trains the one-shot deck picker by REINFORCE against a frozen scorer.
- `pick-decks` — the picker's inference counterpart to `build-decks`.
- `analyze-generated-decks` — composition statistics over generated-decks files.

## Dependencies

- `sealed` imports from `price_predictor` only (the tokenizer, the vocabulary builder, the transformer model and store). Never create the reverse dependency.
- `sealed` never imports `effects`. The two places they meet are plain data: the `--exclude-cards` name list that `python -m effects holdout-cards` writes, and the collection flags, which are defined here in `infrastructure/effect_collection_flags.py` and imported by `effects`.
- The composition engine in `application/analyze_generated_decks.py` is reused by `draft analyze-generated-decks`.

## Vocabulary and encoder

- The sealed vocabulary (`models/sealed/encoder/vocab.txt`) is independent from the price-side one; building one does not modify the other. `[MASK]` is seeded for `train-encoder`'s MLM loss.
- `train-encoder` saves only the encoder weights; the regression and MLM heads are filtered out at save time. The pool mode is recorded in the checkpoint's config.
- Its train/validation split is card-disjoint.
- A card referenced in `cards-played.txt` with no converted `.txt` is warned about and dropped. It does not block the run.
- Every `train-encoder` run overwrites `output/sealed/cards-win-rates.txt`.

## The embedding cache fixes every downstream width

- `encode-cards` strips the `name:` line from each card and writes a `.npz` beside every `.txt` under `output/cardsfolder/`: a `float32` array `(pooled_dim + FEATURE_COUNT,)` under the key `"embedding"`, the pooled text vector followed by the deterministic game features, always in the trailing `FEATURE_COUNT` slots.
- `pooled_dim` depends on the encoder: `2·d_model` for the price encoder and for the sealed encoder's dual pool, `d_model` for the sealed encoder's attention-only pool.
- The scorer's input width is not a flag. It is the width of the `.npz` cache, and the picker's input width is the scorer's. So any encoder (price or sealed, any `--d-model` or `--pool-mode`) works as long as `encode-cards` was run with it.
- A scorer checkpoint, Phase B encoder or picker whose width disagrees with the cache fails fast with a clear error rather than a torch shape mismatch.
- `encode-cards --clean` deletes every `.npz` under the tree. That is why the effect model's ability cache lives outside `output/cardsfolder/`.
- `--encoder-checkpoint` and `--scorer-checkpoint` are mutually exclusive on `encode-cards`; the second extracts the encoder from a Phase B scorer checkpoint.

## Pools and decks

- `--exclude-cards <file>` names cards no booster may contain. Each is redrawn within its own rarity slot, so pool size and rarity structure are unchanged.
- On `generate-pools` the flag depletes a pools file, for the consumers that read one. The effect model's depleted collection run needs the same flag on `match-outcomes`, whose workers open their own pool per match and read no pools file.
- `--label` is required on `build-decks` and `pick-decks`. It is read back by `match-outcomes` as the `method_A`/`method_B` tag of every match that plays the deck.
- `pick-decks` skips pools with fewer than 23 embeddable cards.

## `match-outcomes`

- Each match chooses its decks independently:
  - Side A is sampled from `--side-a-decks` if given, else built by one of the four Forge methods (weights 4:3:2:1) on a fresh pool from a random sealed-legal set.
  - Side B is rolled between the four Forge methods (combined weight 10) on a fresh pool of deck A's set and, if `--side-b-decks` is given, that file (weight `--side-b-decks-weight`) filtered to deck A's set.
- Mirror matches are excluded by equality of the card-name multiset. If no non-mirror file deck exists for deck A's set, the file roll falls back to the Forge methods.
- With both flags absent the command must reproduce the original behaviour: four Forge methods on both sides, random eligible set.
- `--best-of` must be a positive odd integer.
- The supervisor restarts crashed workers. Long Forge AI games can crash the JVM and that is expected. Why the workers are short-lived by design: `src/effects/CLAUDE.md` § Collection.
- With `--exclude-cards`, no collected game holds a held-out card. Without it, the same command collects the full-strength games that become the effect model's card-disjoint validation stratum.
- `--effect-records` has no default: collection is an opt-in, and without the flag the two sealed corpora keep their exact format and content.

## Scorer and picker training

- `train-scorer` Phase A (the default, `--embedding-lr 0`) reads the `.npz` cache with the encoder frozen. Phase B (`--embedding-lr` nonzero) fine-tunes the encoder jointly and needs either `--scorer-checkpoint <phaseA>.pt` (fresh start, encoder from `--encoder-checkpoint`) or `--resume <phaseB>.pt`.
- A Phase B checkpoint carries `encoder_state_dict` and `encoder_config`, so the file is self-contained.
- Architecture flags configure a fresh run only. On `--resume`, `--scorer-checkpoint` or `--picker-checkpoint` they are forbidden; the architecture comes from the checkpoint.
- The best scorer checkpoint is selected by validation accuracy; the best picker checkpoint by validation reward.
- `train-picker` scores each sampled deck's chosen spells and nonbasic lands only, never basics. `--kl-coef` requires `--picker-checkpoint`. Checkpoints hold picker weights only.

## Corpus file formats

Append-only data contracts spanning the Java writers and the Python readers — no single source file teaches a whole one, and a mismatch silently corrupts a corpus that cannot easily be rebuilt.

**Match outcomes** (`output/sealed/match-outcomes.txt`): one line per match, ten semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s`. `timestamp` is ISO 8601 UTC, `run_id` is a UUID identifying the supervisor invocation, `set_code` is the MTG set both pools were drawn from, `method_A`/`method_B` are build-method tags (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or the per-deck label set by `build-decks --label` when the deck came from a generated-decks file), `deck_A`/`deck_B` are pipe-separated lists of 40 Forge canonical card names (duplicates repeat), `games` is the per-game winner sequence (`AA`, `ABA`, `ABABABA`, etc. — length depends on `--best-of`, between `ceil(N/2)` and `N` chars), `play` is the per-game play-first sequence (same length), and `duration_s` is the match wall-clock duration in whole seconds. See `specs/2026-03-28-sealed-deck-picker.md` §Phase 0 Step 4 for full details.

**Cards played** (`output/sealed/cards-played.txt`): one line per played game, eleven semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter`. `timestamp`/`run_id`/`set_code`/`method_A`/`method_B` mirror the parent match's row (joinable by `run_id` plus positional offset within a `(run_id, set_code, method_A, method_B)` group). The four card-list columns are pipe-separated **sets** of distinct card names (no duplicates within a column; basic lands excluded by the writer). For a side X, `cards_played_X` and `cards_not_played_X` are disjoint — if any copy of a card was played, the name appears only in `cards_played_X`. `winner` and `starter` are single chars `A`/`B` matching the corresponding chars of the parent's `games`/`play` strings. Trailing partial lines are tolerated by readers (JVM-crash-mid-write recovery).

**Pools** (`output/sealed/pools/*/pools.txt`): one pool per line, `SET_CODE;Card1|Card2|...|CardN`, basics excluded. The set-code prefix lets downstream tools (`build-decks`, self-play `match-outcomes`) honor same-set constraints.

**Generated decks** (`output/sealed/generated-decks.txt`): one finished 40-card deck per line, `LABEL;SET_CODE;Card1|Card2|...|Card40`. Concatenating several generated-decks files with different labels into one self-play corpus is supported.

**Cards win rates** (`output/sealed/cards-win-rates.txt`): a Python-only snapshot for inspection, with a header row. A cell whose slice denominator is zero is written as the empty string in both its raw and shrunk columns, which distinguishes "no signal" from "neutral signal". Cards never seen in a deck are excluded.

## Model artifacts

Checkpoints live under `models/sealed/` (`encoder/`, `scorer/`, `picker/`).
