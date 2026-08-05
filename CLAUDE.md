# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An ML system that predicts Magic: The Gathering card EUR prices from game-visible attributes (mana cost, types, oracle text, power/toughness, keywords, printing metadata). Works for both real cards and hypothetical ones. Also hosts a sealed-format ML pipeline (card embeddings, pool generation, deck scorer) built on top of the price predictor's transformer encoder.

## Tech stack

- **Python 3.14+** (required — see `pyproject.toml`); manage with `venv` + `pip`. Python executable is `python`, pip is `pip`.
- **Java 17+** for the `forge-connector` Maven module (used for card script conversion and Forge-AI match simulation).
- Dependencies: `scikit-learn`, `pandas`, `numpy`, `joblib`, `ijson`, `fastapi`, `uvicorn`, `torch`, `transformers`. PyTorch is installed with CUDA 12.6 wheels via `--extra-index-url https://download.pytorch.org/whl/cu126`.
- Sibling checkout of MTG Forge expected at `../forge` (built with `mvn install -DskipTests`). MTGJSON data files (`AllPrintings.json`, `AllPricesToday.json`) expected in `resources/`.

## Common commands

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate    # Windows (git-bash: source .venv/Scripts/activate)
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126

# Python tests
pytest                                    # unit + integration (default)
pytest tests/unit/                        # fast unit tests only
pytest tests/integration/                 # integration tests only
pytest tests/unit/application/test_train.py::test_name   # single test
pytest -m "not integration"               # skip integration-marked tests

# Python lint
ruff check src/ tests/

# Java (forge-connector)
cd forge-connector && mvn package -DskipTests    # build fat JAR used by convert/pools/match-outcomes
cd forge-connector && mvn test                   # run JUnit 5 tests
```

Test/lint commands run from the project root. `pyproject.toml` configures `testpaths = ["tests"]` and an `integration` pytest marker.

## Architecture

Two independent Python packages live under `src/`, each laid out in hexagonal (ports-and-adapters) style: `domain` → `application` → `infrastructure`.

### `price_predictor` — the price prediction system

Entry point: `python -m price_predictor <subcommand>` (see `src/price_predictor/__main__.py` and `infrastructure/cli.py`).

Subcommands:
- **`convert`** — shells out to the Java `ConvertMain` to transform Forge's `.txt` card scripts (at `../forge/forge-gui/res/cardsfolder/`) into a compact LLM-friendly text format written to `./output/`. One output file per card with lowercase property lines and classified ability lines.
- **`check-convert`** — compares converted files against MTGJSON oracle text and flags low-similarity cards.
- **`vocabulary`** — scans the converted corpus and writes `models/price-predictor/transformer/vocab.txt` (custom MTG word-level tokenizer, ~5k tokens). **Required before training or running the transformer model.** Also seeds set-code fragments from `AllPrintings.json` when available.
- **`train sklearn`** — trains a `GradientBoostingRegressor` on `log(price)` using a 17-group feature pipeline (mana cost, types, keywords, TF-IDF oracle text, power/toughness, printing metadata). Saves to `models/price-predictor/sklearn/{timestamp}.joblib` + `latest.joblib`.
- **`train transformer`** — trains a custom transformer (tunable `--d-model/--n-layers/--n-heads/--ff-dim`) on tokenized card text. Uses log-price target with `--log-offset`, price-bucket oversampling (`--sampler-exponent`), and validation-accuracy-based best checkpoint selection. Saves to `models/price-predictor/transformer/`.
- **`predict {sklearn,transformer}`** — reads converted card text (from `--file` or inline `--card`) and prints JSON with `predicted_price_eur` + `model_version`. Attaches printing metadata from MTGJSON when the card name matches.
- **`evaluate {sklearn,transformer}`** — computes MAE/median % error/`top_20_overlap` on a held-out test split. Transformer adds per-bucket breakdown.
- **`serve`** — FastAPI + uvicorn REST service exposing `POST /api/v1/predict`. Loads the sklearn model (required), transformer (optional, graceful degradation), tokenizer, and MTGJSON metadata map for auto-fill. Accepts `text/plain` card text and returns predictions from every available model.

Key modules inside `price_predictor`:
- `domain/` — `entities.py` (Card, PriceEstimate, TrainingExample, TrainedModel, EvaluationMetrics), `value_objects.py` (ManaCost, PrintingData), `tokenizer.py` (custom MTG tokenizer).
- `application/` — one file per use case (`train.py`, `train_transformer.py`, `predict.py`, `predict_transformer.py`, `evaluate.py`, `evaluate_transformer.py`, `build_vocabulary.py`, `feature_engineering.py`, `check_convert.py`).
- `infrastructure/` — `cli.py` (argparse wiring), `server.py` (FastAPI app), `converted_card_parser.py`, `mtgjson_loader.py`, `model_store.py` (joblib), `transformer_model.py`, `transformer_store.py` (`.pt`), `transformer_dataset.py`, `tokenizer_store.py`, `metadata_encoder.py`.

The two model types have **different input contracts**:
- `sklearn` takes a parsed `Card` object and runs it through `FeatureEngineering` to produce a numeric vector.
- `transformer` takes raw converted card text, tokenizes it with the custom vocab, and receives `PrintingData` as a side-channel (not embedded in the text).

Prices are trained on `log(price + offset)` and exp-transformed back on inference — the skew of the EUR distribution would otherwise let a handful of expensive cards dominate the loss.

### `sealed` — sealed-format ML pipeline

Entry point: `python -m sealed <subcommand>` (see `src/sealed/infrastructure/cli.py`). The sealed pipeline is built around a sealed-trained card encoder; the price-predictor transformer is a valid alternate encoder source via `--encoder-checkpoint`.

Subcommands:
- **`build-vocab`** — scans the converted card corpus (default `output/cardsfolder/`) and writes a sealed-side tokenizer vocab to `models/sealed/encoder/vocab.txt`. Wraps `price_predictor.application.build_vocabulary`; `--target-size` (default 5000) post-truncates the corpus-frequency vocab while preserving seeded specials and domain tokens. The seeded specials include `[PAD]`, `[UNK]`, `cardname`, and `[MASK]` (the last reserved for `train-encoder`'s MLM auxiliary loss). Independent from the price-side `vocab.txt` — building one does not modify the other.
- **`train-encoder`** — reads `output/sealed/cards-played.txt` in one streaming pass, aggregating per-card winnability inline (primary + `@play` counters + per-color slices keyed off each card's `mana cost:` line all in the same pass), and computes nine per-card regression labels (`score_play`, `score_draw`, `played_rate`, `cast_lift`, and `color_lift_X` for each WUBRG letter) in raw and shrunk form (`--shrinkage-k`, default 20). Splits cards stratified by `score_play` quartile (20% val, card-disjoint, FR-018 fallback chain for cards with empty `score_play`), tokenizes the corpus once, and trains a token encoder + card encoder + five regression heads + an MLM head from random init on `L_reg + (--mlm-weight) · L_mlm`. The card encoder pools the token outputs into a `pooled_dim`-wide card vector that feeds the regression heads: `--pool-mode dual` (default) concatenates a multi-query attention pool with a max-pool (`pooled_dim = 2·d_model`); `--pool-mode attn` uses the attention pool only (`pooled_dim = d_model`). The chosen mode is recorded in the saved checkpoint's config. `L_reg` uses per-head per-batch sum-to-1 weighted MSE with FR-017a per-card weights; `L_mlm` is cross-entropy at the masked positions only (`--mlm-mask-prob` default 0.15, mask drawn fresh per batch over non-special, non-pad tokens; the MLM head is applied only at those positions). Batches are length-bucketed (similar-length cards grouped, batch order reshuffled per epoch) and padded per-batch to the batch's longest card. Optimizer: AdamW with per-parameter-group max-norm 1.0 gradient clipping; LR schedule: linear warmup over the first 5% of `--epochs × batches_per_epoch` scheduled steps, then constant `--lr`. Saves only the encoder weights to `models/sealed/encoder/{timestamp}.pt` plus `latest.pt`; the regression heads and MLM head are filtered out at save time. Writes the per-card label snapshot to `output/sealed/cards-win-rates.txt` (23 columns, sorted by `shrunk_score_play` desc) for SC-005 inspection. Best checkpoint selected by full validation loss (`L_reg + (--mlm-weight) · L_mlm`). Cards referenced in `cards-played.txt` with no converted `.txt` under `--cards-folder` are reported via a log warning (naming up to 20 + the total) and dropped — they don't block the run. Each epoch logs the loss decomposition plus, on the val set, MLM perplexity / top-1 masked-token accuracy and the per-head Pearson correlation between predictions and targets, and a one-batch gradient-norm probe of the regression-vs-MLM pull on the shared encoder (`‖∂L_reg/∂trunk‖` vs `‖∂(w·L_mlm)/∂trunk‖`, measured on the epoch's first batch via two extra `autograd.grad` passes). `d_model` (`--d-model`, default 256) and `ff_dim` (`--ff-dim`, default `4·d_model`) are tunable for architecture experiments; `val_fraction=0.2` and `random_seed=42` are hardcoded; `max_seq_len` (the position-embedding table size) is computed from the corpus.
- **`encode-cards`** — strips the `name:` line from each card and writes a `.npz` file next to every `.txt` in `output/cardsfolder/`. Each `.npz` stores a `float32` array of shape `(pooled_dim + FEATURE_COUNT,)` under key `"embedding"` — the encoder's pooled text vector followed by the deterministic game features (always the trailing `FEATURE_COUNT` slots). `pooled_dim` depends on the encoder: `2·d_model` for the price encoder (`cat([max_pool, mean_pool])`) and the sealed encoder's dual pool (`cat([attn_pool, max_pool])`, `--pool-mode dual`), or `d_model` for the sealed encoder's attention-only pool (`--pool-mode attn`); pooling is over the encoder's token outputs with padding masked. The summary line prints the resulting embedding width. Encoder weights default to `models/sealed/encoder/latest.pt` via `--encoder-checkpoint`; pass `--scorer-checkpoint <phaseB>.pt` instead to extract encoder weights from a Phase B sealed scorer checkpoint (the two flags are mutually exclusive when both are explicitly passed). The `--vocab-path` default is `models/sealed/encoder/vocab.txt`. Re-running is idempotent; `--clean` forces a full re-encode.
- **`generate-pools`** — invokes the forge-connector JAR (`PoolMain`) to generate N sealed pools (6 boosters each); writes `pools.txt` (one pool per line in `SET_CODE;Card1|Card2|...|CardN` format, basics excluded). With `--set RVR` all pools come from the given set and are written to `output/sealed/pools/{set}/`; without `--set`, each pool uses an independently-selected random sealed-legal set and output defaults to `output/sealed/pools/`.
- **`build-decks`** — reads a pools file (`SET_CODE;Card1|...` format), loads a trained scorer checkpoint, and greedily builds one 40-card deck per pool (23 nonlands chosen by the scorer + basic lands from the manabase heuristic). Writes `generated-decks.txt` (one deck per line in `LABEL;SET_CODE;Card1|...|Card40` format) to `output/sealed/` by default. `--label` is **required** and identifies the generation method (e.g. `gen-2`); it is later read back by `match-outcomes` as the `method_A` / `method_B` tag for self-play matches that play this deck. Consumed by `match-outcomes --side-a-decks` / `--side-b-decks` for self-play.
- **`match-outcomes`** — long-running supervisor that spawns Java `MatchWorkerMain` workers. Each match independently chooses how to produce deck A and deck B:
  - **Side A**: sampled from `--side-a-decks <path>` if given (`method_A` = the deck's `LABEL`), else built by the 4 Forge methods (weights 4:3:2:1) on a freshly-generated pool from a random sealed-legal set.
  - **Side B**: rolled between the 4 Forge methods (combined weight 10) on a fresh pool of deck A's set, and — if `--side-b-decks <path>` is given — sampling from that file (weight `--side-b-decks-weight`, default 4) filtered to deck A's set code. `--side-b-decks-weight` requires `--side-b-decks`. Mirror matches are excluded by content equality on the card-name multiset, so deck B never matches deck A. If no non-mirror file deck exists for deck A's set, the file roll falls back to Forge methods.

  Phase 0 (both flags absent) reproduces the original "4 Forge methods on both sides, random eligible set" behavior. `--best-of` (default 7) controls games per match; must be a positive odd integer.

  Match-level outcomes are appended to `output/sealed/match-outcomes.txt`; per-game card-play data (one line per played game) is appended to `output/sealed/cards-played.txt` by the same worker. The supervisor restarts crashed workers — long Forge AI games can crash the JVM and that's expected. Ctrl-C to stop.
- **`train-scorer`** — trains a Set Transformer deck scorer on `match-outcomes.txt` using AdamW with per-parameter-group max-norm 1.0 gradient clipping and `--patience`-driven early stopping (default 5). Architecture flags (`--n-layers/--n-heads/--n-seeds/--d-ff/--mlp-hidden/--dropout`) configure a fresh run; on `--resume` or `--scorer-checkpoint` they are forbidden (architecture inherits from the loaded checkpoint). The scorer's input width (`ScorerConfig.d_model`) is **not** a flag — it's derived from the width of the `.npz` embedding cache (encoder text-vector width + `FEATURE_COUNT`), so any encoder (price or sealed, any `--d-model`/`--pool-mode`) works as long as `encode-cards` was run with it; a loaded scorer checkpoint or Phase B encoder whose width disagrees with the cache fails fast with a clear error rather than a torch shape mismatch. Two phases: **Phase A** (default, `--embedding-lr 0`) consumes the `.npz` cache with the encoder frozen; **Phase B** (`--embedding-lr <nonzero>`) jointly fine-tunes the encoder alongside the scorer, requiring either `--scorer-checkpoint <phaseA>.pt` (fresh kickoff, encoder weights from `--encoder-checkpoint` defaulting to `models/sealed/encoder/latest.pt`) or `--resume <phaseB>.pt` (continuation, encoder weights from the resumed checkpoint). When the default sealed encoder is missing on a Phase B fresh kickoff, the run errors out and points the user at `train-encoder` (or at passing `--encoder-checkpoint <path>` explicitly). Checkpoints saved to `models/sealed/scorer/`; Phase B checkpoints carry an additional `encoder_state_dict` + `encoder_config` so the saved file is self-contained. Best checkpoint selected by validation accuracy.
- **`evaluate-scorer`** — generates fresh sealed pools, has the scorer greedily build a deck from each, and has Forge AI play matches between the scorer's deck and Forge's own optimal builder. Outputs win/loss stats. Spawns Java workers via `evaluation_connector.py`. With `--set BLB` all pools are from the given set; without `--set`, a random sealed-legal set is selected.
- **`train-picker`** — trains a one-shot deck **picker** (a policy transformer over a pool) from random init via REINFORCE against a frozen scorer (spec 017). Each step samples `--n-samples` decks per pool with a GPU-batched sequential without-replacement sampler, scores them with the frozen `--scorer-checkpoint` (chosen spells + nonbasic lands only, no basics), and optimizes a policy-gradient + entropy + auxiliary-pool-quality loss with a per-pool empirical-mean baseline (with optional `--normalize-advantage` GRPO-style division of the centered reward by the per-pool reward std). The picker's input width derives from the scorer's `ScorerConfig.d_model` (= `.npz` cache width); `--d-model` overrides the internal width (inserting an input projection). Architecture flags (`--d-model/--n-layers/--n-heads/--ff-dim/--dropout`) configure a fresh run and are forbidden on `--resume` / `--picker-checkpoint` (architecture inherits from the checkpoint). The entropy coefficient follows a val-reward-driven decay schedule; best checkpoint selected by validation reward (deterministic-walk decks scored by the training scorer); early stopping on `--patience`. `--auditor-scorer-checkpoint` enables a per-epoch cross-scorer Spearman audit; the per-epoch log also reports distributional summaries (color count, 5-bin CMC histogram, creature count, type balance). `--kl-coef` (requires `--picker-checkpoint`) adds a KL penalty against a frozen reference picker. Random seed hardcoded to 42. Checkpoints saved to `models/sealed/picker/` as `latest.pt` (every epoch) + `best_{timestamp}.pt` (on each new val-reward best); picker weights only.
- **`pick-decks`** — the inference counterpart to `build-decks`: loads a `--picker-checkpoint`, runs one deterministic picker forward + the § 1.1 pick-decomposition walk per pool (23 spells in ranked order + any nonbasic lands ranked above the 23rd spell), fills basic lands via `compute_basic_lands`, and writes a `generated-decks.txt` drop-in for `match-outcomes`. `--label` (required) is written verbatim as each line's method tag. `--resume` appends-and-skips like `build-decks`. Pools with fewer than 23 embeddable cards are skipped; picker/cache width mismatch fails fast.
- **`analyze-generated-decks`** — aggregate composition statistics over one or more generated-decks files (positional args, default `output/sealed/generated-decks.txt`): color presence, color-count distribution, pip-share-by-rank, mana curve, type balance, basic/nonbasic land split, pip distribution, and (unless `--no-rarity` or MTGJSON is absent) rarity distribution. Prints a global report plus a per-label breakdown when more than one label is loaded. The shared engine lives in `application/analyze_generated_decks.py` and is reused by `draft analyze-generated-decks`.

Match-outcome file format (`output/sealed/match-outcomes.txt`): one line per match, ten semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;duration_s`. `timestamp` is ISO 8601 UTC, `run_id` is a UUID identifying the supervisor invocation, `set_code` is the MTG set both pools were drawn from, `method_A`/`method_B` are build-method tags (`forge-best`, `forge-3sub`, `forge-8sub`, `random`, or the per-deck label set by `build-decks --label` when the deck came from a generated-decks file), `deck_A`/`deck_B` are pipe-separated lists of 40 Forge canonical card names (duplicates repeat), `games` is the per-game winner sequence (`AA`, `ABA`, `ABABABA`, etc. — length depends on `--best-of`, between `ceil(N/2)` and `N` chars), `play` is the per-game play-first sequence (same length), and `duration_s` is the match wall-clock duration in whole seconds. See `specs/2026-03-28-sealed-deck-picker.md` §Phase 0 Step 4 for full details.

Cards-played file format (`output/sealed/cards-played.txt`): one line per played game, eleven semicolon-separated fields `timestamp;run_id;set_code;method_A;method_B;cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter`. `timestamp`/`run_id`/`set_code`/`method_A`/`method_B` mirror the parent match's row (joinable by `run_id` plus positional offset within a `(run_id, set_code, method_A, method_B)` group). The four card-list columns are pipe-separated **sets** of distinct card names (no duplicates within a column; basic lands excluded by the writer). For a side X, `cards_played_X` and `cards_not_played_X` are disjoint — if any copy of a card was played, the name appears only in `cards_played_X`. `winner` and `starter` are single chars `A`/`B` matching the corresponding chars of the parent's `games`/`play` strings. Trailing partial lines are tolerated by readers (JVM-crash-mid-write recovery).

Cards-win-rates file format (`output/sealed/cards-win-rates.txt`): overwritten by every `train-encoder` run; one header row + N data rows, semicolon-separated, sorted by `shrunk_score_play` descending (cards with empty `shrunk_score_play` sort to the end). Schema (23 columns): `card_name;wins_when_played;wins_when_in_deck;losses_when_played;losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G`. Floats formatted to five decimals; counters as plain integers. Cells whose slice denominator is zero are written as the empty string in both raw and shrunk columns (distinguishes "no signal" from "neutral signal"). Cards with `wins_when_in_deck + losses_when_in_deck == 0` are excluded entirely.

Pool file format (`output/sealed/pools/*/pools.txt`): one pool per line, `SET_CODE;Card1|Card2|...|CardN`. The set-code prefix lets downstream tools (`build-decks`, self-play `match-outcomes`) honor same-set constraints.

Generated-decks file format (`output/sealed/generated-decks.txt`): one finished 40-card deck per line, `LABEL;SET_CODE;Card1|Card2|...|Card40`. `LABEL` is the value passed to `build-decks --label` and is recorded as the `method_A` / `method_B` tag whenever this deck is sampled into a self-play match. Concatenating multiple generated-decks files with different labels into one self-play corpus is supported.

Key modules inside `sealed`:
- `domain/` — `card_encoder.py` (wraps the encoder for inference), `encoder_model.py` (sealed transformer + multi-query attention pool), `scorer_model.py` (Set Transformer architecture), `picker_model.py` (one-shot picker: SAB trunk + per-card head + aux head, plus the deterministic pick-decomposition walk), `deterministic_features.py`.
- `application/` — `build_vocab.py`, `train_encoder.py`, `encode_cards.py`, `generate_pools.py`, `build_decks.py`, `match_outcomes.py`, `train_scorer.py`, `train_picker.py`, `pick_decks.py`, `evaluate_scorer.py`, `analyze_generated_decks.py` (deck-composition stats engine, shared with `draft`).
- `infrastructure/` — `cli.py`, `cards_played_reader.py`, `encoder_store.py`, `embedding_store.py`, `pool_connector.py`, `match_worker_connector.py`, `evaluation_connector.py`, `match_data_loader.py`, `scorer_store.py`, `picker_store.py`, `card_name_corrections.py`.

### `draft` — draft-agent ML pipeline

Entry point: `python -m draft <subcommand>` (see `src/draft/__main__.py` and `infrastructure/cli.py`). A generation-1 MTG-draft agent: a two-headed Set Transformer (imitation **policy** over the cards in the current pack + a Monte-Carlo **critic** on a context token) trained offline from a corpus of Forge-generated drafts. `draft` **imports from** `sealed` and `price_predictor` (scorer, picker, greedy builder, embedding layout, card locator, checkpoint plumbing, Forge worker pattern); never the reverse.

Subcommands:
- **`generate-draft-data`** — supervisor (Python) + Java `DraftWorkerMain` that drives Forge's draft AI for all pod seats. The worker streams one `<<DRAFT-EVENT-JSON>>` transcript per completed draft (boosters + per-seat agent); the supervisor reconstructs each seat's drafted pool from the booster geometry, builds a deck per seat with the frozen picker (`--build-method picker`, default) or SA greedy builder (`greedy`), scores the non-basic subset with the frozen scorer, and appends one self-contained record to `output/draft/drafts.jsonl`. One `run_id` (UUID) per invocation; crashed workers are restarted toward `--n-drafts` (Ctrl-C stops cleanly); `--resume` counts pre-existing records. `--set` restricts every draft to one set (else a random sealed-legal set per draft); `--agent-mix` (default `forge-full:6,forge-r30:1,forge-r100:1`) is a categorical sampled independently per seat — `forge-full` is pure Forge AI, `forge-r30`/`forge-r100` replace 30%/100% of that seat's picks with uniform-random legal picks. A trained agent can also **pilot live seats**: `--agent-checkpoint LABEL=PATH` (repeatable; bare `PATH` ⇒ label `draft-agent`) binds a mix label to a draft-agent checkpoint, and whenever the worker reaches a pick for such a seat it emits a `<<DRAFT-PICK-REQUEST>>` and blocks on stdin for the supervisor's `<<DRAFT-PICK-RESPONSE>>` (the trained policy's genuine choice, or `abort`). The supervisor reconstructs the live typed-token state incrementally with an `OnlineDraftStateTracker` (pinned to `build_state` by a gating equivalence test), embeds it, and picks via `--pick-mode {argmax,sample}` (default `argmax`), `--temperature`, and `--seed` (seeded `sample` is reproducible, SC-007). Every mix label must be a Forge built-in or bound (fail fast); each checkpoint's `packs`/`P` are validated against the live geometry. A **pick fault** — policy/tracker error, a protocol desync, or every legal action un-embeddable — abandons the whole in-flight draft with no substitute (clean corpus by construction); the worker emits `<<DRAFT-ABANDONED>>` (or the supervisor sends `abort`), the draft is not recorded or counted, and `--max-consecutive-faults` (default 5) consecutive abandonments abort the run with a nonzero exit. Worker stderr is redirected to a per-run log `output/draft/worker-<run_id>.log`. With no `--agent-checkpoint` the command is byte-for-byte the gen-1 behavior (no pick side-channel, stderr discarded). The corpus schema is unchanged: a model-piloted seat's `agent` is just its mix label, every seat is built + scored on the same scale, so a mixed pod yields a per-draft agent-minus-Forge `deck_score` delta.
- **`train-draft-agent`** — turns each `(draft, seat, pack, pick)` into a typed-token training example (`[CONTEXT][POOL][PACK][PASSED][TAKEN]` with learned `packs_ago`/`pick_ago` recency), then trains the imitation policy (cross-entropy over `--imitation-agents`-whitelisted seats' `PACK` tokens; default whitelist `forge-full`) and the critic (MSE over the standardized leave-one-out pod-relative reward of every non-failed seat) jointly: `L = imitation_weight·CE + critic_weight·MSE`. AdamW + linear-warmup-then-constant LR + per-group max-norm clip; draft-disjoint train/val split (`random_seed=42`); per-epoch log of the loss split + val imitation top-1/top-3 + per-`pack_number` critic MSE; best-by-val-loss checkpoint + `latest.pt` under `models/draft/agent/`. Optional ReduceLROnPlateau-style annealing (`--lr-decay-patience`, opt-in): after that many mini-epochs without a new best val_loss the LR is multiplied by `--lr-decay-factor` (default 0.1) down to `--min-lr` (default `lr·1e-3`), reusing the same strict-best counter as early stop — so `--lr-decay-patience` must be `< --patience` and early stop only fires once the LR floor is reached; the decay position (`lr_decay_count`) is checkpointed so `--resume` continues annealing (an explicit `--lr` override restarts it). `--d-model` default derives from the `.npz` width + feature widths (non-default inserts a `Linear`); `d_model % n_heads == 0` and architecture-flags-with-`--resume`/`--checkpoint` both fail fast. The critic-target standardization mean/std are stored in the checkpoint and de-standardized at inference back to raw scorer-score space.

- **`train-draft-agent-rl`** — gen-2 RL self-play fine-tuning: on-policy actor-critic (REINFORCE + GAE baseline + KL anchor + entropy bonus) over a corpus the reference checkpoint generated. `--checkpoint` warm-starts the actor **and** critic and is the KL anchor (xor `--resume` to continue an RL run); `--drafts-path` is the on-policy corpus and the **sole** source of the policy gradient; repeatable `--critic-corpus` adds off-policy corpora for critic regression/coverage only. `--learner-agents` (required) is the whitelist of mix labels whose seats feed the policy gradient; all non-failed seats feed the critic. `--rollout-temperature` (required, no default) is the temperature the corpus was sampled at — all policy distributions (logπ, entropy, KL, the behaviour-anomaly check) use it, and it must match the `generate-draft-data --temperature` that produced the corpus. Per learner pick the loss is `−A_t·logπ_T(a_t) + value_weight·MSE(V, R_std) − entropy_coef·H(π_T) + kl_coef·KL(π_T‖π_ref,T)`, where `R` is the pod-relative leave-one-out `deck_score` (terminal; γ=1) and `A_t` is the GAE(`--gae-lambda`, default 0.95) advantage over the seat's ordered 45-pick trajectory, recomputed each epoch in a batched no-grad critic pass. The critic stays in the reference checkpoint's standardized reward space (stored `critic_mean`/`std` reused, not recomputed). `--kl-coef`/`--entropy-coef` follow a val-driven decay schedule; AdamW + warmup-then-constant LR + per-group clip + optional LR-plateau annealing + `--patience` early stop, all mirroring `train-draft-agent`. Best-checkpoint/early-stop use the **held-out RL objective** (the same loss on a draft-disjoint val split) as an in-run guard — true cross-generation strength is judged externally by the yardstick. On-policy pairing of corpus↔checkpoint is the **operator's responsibility** (no corpus provenance is stored — FR-021); the only safeguard is a one-time behaviour-anomaly summary (mean `log π_ref` + fraction of learner picks the reference assigns near-zero probability) that **warns but never aborts**. Saves `{timestamp}.pt` + `latest.pt` under `models/draft/agent/` in the gen-1 checkpoint format plus `rl_metadata` (generation index, reference id, algorithm, `gae_lambda`/`kl_coef`/`entropy_coef`/`value_weight`/`rollout_temperature`) so a checkpoint is self-describing for the next cycle. No encoder weights (Phase A). The **cross-generation yardstick** and the **self-play loop** are operator runbooks over existing commands (no new code): generate a sample-mode corpus with `--pick-mode sample`, fine-tune with this command, then evaluate with one greedy `generate-draft-data --pick-mode argmax` fixed-mix co-seated run + `analyze-generated-decks --agent <each>` for per-agent mean `deck_score`; promotion is a manual judgment.
- **`analyze-generated-decks`** — the deck-composition diagnostic sourced from a `drafts.jsonl` corpus instead of a generated-decks file (`--drafts-path`, default `output/draft/drafts.jsonl`). A **required `--agent`** scopes the whole report to seats piloted by one agent/mix label (e.g. `draft-agent`, `forge-full`); run it once per agent to compare. Each seat's built `deck` is analyzed by the shared `sealed` engine (colors, curve, types, lands, pips, rarity), and — because the corpus also carries each seat's `deck_score` — a `deck_score` mean/median/n summary for that agent is printed above the composition report. Seats with a failed build (empty deck) are skipped; an `--agent` matching no built deck fails fast and lists the available agents.
- **`validate-builder`** — picker-vs-SA builder-validation diagnostic (FR-042): `python -m draft validate-builder --pools-from output/draft/drafts.jsonl` (or `--fresh-pools --set X --n-pools N`) builds each drafted pool with both the picker and SA, scores both with the frozen scorer, and prints the gating picker-vs-SA Spearman, the SA−picker score-gap median/IQR, and the SA-vs-SA reference ceiling — run once per picker/scorer pair to choose `--build-method`. (`--pools-from` and `--fresh-pools` are a required mutually-exclusive pair.)

Drafts file format (`output/draft/drafts.jsonl`): one self-contained JSON record per line, append-only; readers tolerate a trailing partial final line. Each record has `draft_id`, `run_id`, `timestamp` (ISO 8601 UTC), `seats` (length `pod_size`; each `{agent, deck (40 names incl. basics, or [] on failed build), deck_score (scorer scalar, or null on failed build)}`), and `boosters` (length `pod_size × packs`; each `{set_code, picks}` where `picks` is the cards drafted from that physical pack in pick order, fully drained to `pack_size`). The booster *ordering* pins all geometry: `pack_number(k)=⌊k/pod_size⌋+1`, `opening_seat(k)=k mod pod_size`, and the pick at offset `j` of `boosters[k]` was made by seat `(opening_seat + j·dir_p) mod pod_size` with `dir_p=+1` for packs 1 & 3 and `−1` for pack 2 — so any seat's full observation history reconstructs from the record alone (`draft/domain/draft_geometry.py`).

Key modules inside `draft`:
- `domain/` — `draft_geometry.py` (FR-016 booster↔seat/pick geometry + `DraftRecord`/`Seat`/`Booster` dataclasses + drafted-pool reconstruction), `draft_state.py` (typed-token state assembly with wheel-diff/pack-flush `PASSED→TAKEN` transitions + recency), `online_draft_state.py` (`OnlineDraftStateTracker`: the same typed-token state rebuilt incrementally from the live pick-request stream, pinned to `build_state`), `draft_agent_model.py` (`DraftAgentConfig` + `DraftAgentModel`: SAB trunk + policy head + critic head).
- `application/` — `agent_mix.py` (parse/sample `--agent-mix`), `generate_draft_data.py` (supervisor + sentinel parsing + record assembly + picker/greedy labelers + live pick-request routing), `agent_pick_service.py` (`AgentPickService` + `PickFault`: live policy inference for one checkpoint), `agent_registry.py` (`AgentRegistry`: label/geometry validation + per-label service map), `draft_pick_states.py` (`iter_seat_pick_states`: the shared per-pick typed-token state walk used by both the gen-1 imitation loader and the gen-2 RL loader), `train_draft_agent.py` (loader + joint training loop + val metrics), `train_draft_agent_rl.py` (gen-2 on-policy actor-critic loader + GAE advantage precompute + REINFORCE/KL/entropy loss + training loop behind `train-draft-agent-rl`), `rl_advantage.py` (`gae_advantages`: GAE(λ) over a terminal-reward trajectory), `validate_builder.py` (diagnostic logic behind `validate-builder`), `analyze_generated_decks.py` (drafts.jsonl→decks adapter + per-label deck-score summary feeding the shared `sealed` composition engine behind `analyze-generated-decks`).
- `infrastructure/` — `cli.py`, `draft_record_io.py` (JSONL read/write + resume count), `draft_worker_connector.py` (launches `DraftWorkerMain`, pipes the pick side-channel + per-run stderr log), `draft_agent_store.py` (checkpoint save/load, mirrors `PickerStore`).

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

`draft` **imports from** `sealed` and `price_predictor` (scorer, picker, greedy builder, `deck_assembly`, `score_decks`, `ConvertedCardLocator`, card-embedding layout, `torch_checkpoint`, `forge_jvm` worker helpers) — never the reverse. `draft` adds only the genuinely-new logic (booster→state geometry, typed-token state, the two-headed model, the Java draft worker).

## Tests

- `tests/unit/` — fast, fixture-based unit tests (`application/`, `domain/`, `infrastructure/`, `sealed/`).
- `tests/integration/` — end-to-end pipeline tests (convert → train → predict → serve) and transformer training smoke tests.
- `tests/fixtures/` — sample Forge card scripts and trimmed MTGJSON snippets.
- Java tests in `forge-connector/src/test/java/`; Forge-dependent ones are tagged `@Tag("integration")`.

## Model artifact layout

```
models/
  price-predictor/
    sklearn/        {timestamp}.joblib, latest.joblib
    transformer/    {timestamp}.pt, latest.pt, vocab.txt
  sealed/
    encoder/        {timestamp}.pt, latest.pt, vocab.txt
    scorer/         checkpoints (best_* selected by val accuracy)
    picker/         latest.pt (per epoch), best_{timestamp}.pt (best val reward)
  draft/
    agent/          {timestamp}.pt (best by val loss; gen-2 RL: best by held-out RL objective + rl_metadata), latest.pt (per epoch)
```

Inputs the code expects to find on disk:
- `resources/AllPrintings.json`, `resources/AllPricesToday.json` — MTGJSON dumps.
- `../forge/forge-gui/res/cardsfolder/` — Forge card scripts (source for `convert`).
- `output/cardsfolder/` — converted card text files, each paired with a `.npz` after `encode-cards` runs.
- `output/sealed/pools/{set}/pools.txt` or `output/sealed/pools/pools.txt` — generated sealed pools (`SET_CODE;Card1|...` per line).
- `output/sealed/generated-decks.txt` — scorer-built 40-card decks from `build-decks` (`LABEL;SET_CODE;Card1|...|Card40` per line); input to `match-outcomes --side-a-decks` / `--side-b-decks`.
- `output/sealed/match-outcomes.txt` — append-only training data for the scorer.
- `output/sealed/cards-played.txt` — append-only per-game card-play log; the input to `train-encoder` after label aggregation.
- `output/draft/drafts.jsonl` — append-only labeled draft corpus from `generate-draft-data` (one self-contained JSON record per line); the input to `train-draft-agent` and (per-cycle, e.g. `drafts-genK.jsonl`) the on-policy corpus for `train-draft-agent-rl`. Schema unchanged across gen-1/gen-2 — RL pairs a corpus to its generating checkpoint by operator convention, not a stored field.

## Specs, experiments & the feature workflow

Three kinds of design document, each with a distinct audience, purpose, tone, and location. Keep them in their lanes — don't blur experiment rationale into specs, or benchmark numbers into specs.

### `experiments/*.md` — experiment / design records (ADRs)

- **Audience:** the user (and Claude) reasoning about *what happened* and *what to try next*.
- **Purpose:** record the outcome of a run (results, metrics, what worked / failed) and the design rationale for the next iteration — the "why" behind a feature. This is the **only** place for benchmark numbers, run logs, post-mortems, rejected alternatives, and cross-references to prior results.
- **Format:** a fluent, discursive ADR. Prose is fine; explain reasoning, trade-offs, and the mechanism behind a result. Each design doc carries an **Outcome / Result** section to fill in once the experiment runs. Named `YYYY-MM-DD-<topic>-design.md` (design/plan) or `YYYY-MM-DD-<topic>.md` (results).
- Convert relative dates to absolute. Link related docs.

### `specs/YYYY-MM-DD-<name>.md` — root-level human-readable specs

- **Audience:** the user, to track and understand what is being built.
- **Purpose:** a **normative** specification — *what* to build and how it behaves from the outside (commands, CLI surface, contracts, records), based on the conclusions reached in the experiment doc. Stops short of an implementation manual.
- **Format & tone:** tight, direct, WHAT-not-WHY. Prefer **short bullet points and tables over long paragraphs**. Minimal, deliberate bold — only structural labels (list lead-ins, table columns, mini-headers), never mid-sentence emphasis. Keep rationale, gen-over-gen comparisons, and benchmark numbers **out** — those live in `experiments/` (link to them instead). Use timeless present tense.
- These are hand-written (not spec-kit), and are the source the speckit `spec.md` is derived from.

### `specs/NNN-name/` — spec-kit feature directories

- **Audience:** primarily Claude, to drive implementation (`spec.md` → `plan.md` → `research.md` → `tasks.md`).
- **Purpose:** the machine-workable spec that ultimately produces the code. Optimise these for implementation clarity, not for human browsing — do whatever is most useful for producing correct code (detailed FRs, acceptance scenarios, edge cases, cross-references).
- **Format:** follow the spec-kit templates. Invoke them via the `speckit.*` skills (`speckit.specify`, `speckit.clarify`, `speckit.plan`, `speckit.tasks`, `speckit.implement`) rather than editing the `.specify/` templates by hand. Numbered `NNN-name/`, next number = highest existing + 1.
- Before starting non-trivial work in an area, read the relevant `spec.md` / `plan.md` / `research.md`.

### The per-feature workflow

1. **Experiment doc** (`experiments/`) — record what happened in the last run and, based on those results, discuss and decide the next improvement.
2. **Root spec** (`specs/YYYY-MM-DD-*.md`) — write the normative, human-readable spec for that improvement, drawing its conclusions from step 1.
3. **Speckit** (`specs/NNN-name/`) — derive the spec-kit `spec.md` from the root spec and drive the implementation through `speckit.*`.
4. **Run & analyse** — run the experiment the spec enables, record the outcome back in the step-1 doc's Outcome section, and repeat from step 1.
