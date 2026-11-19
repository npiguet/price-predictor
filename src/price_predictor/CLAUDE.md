# `price_predictor` — the price prediction system

Entry point: `python -m price_predictor <subcommand>` (see `src/price_predictor/__main__.py` and `infrastructure/cli.py`). The terms this file uses are defined in `GLOSSARY.md` beside it, or, for those shared with other packages, in the root `GLOSSARY.md`.

Besides its two price models, the package owns what every other package reads cards through: the converted card format, the tokenizer and the vocabulary. `sealed`, `draft` and `effects` import from it; it imports from none of them.

Subcommands:
- **`convert`** — runs the Java `ConvertMain` to turn Forge card scripts (`../forge/forge-gui/res/cardsfolder/`) into converted cards under `--output-path`, one file per card: header lines, then ability lines, one block per face. Forge's token scripts (`../forge/forge-gui/res/tokenscripts/`) are converted by the same run into `--tokens-output-path`. Beside each converted card it writes a `<name>.provenance.json` provenance sidecar, which the effects package reads; the format is in `src/effects/docs/text-encoding.md` § Provenance sidecars. A card's text and sidecar are written together or not at all. An output tree that already holds converted files is refused, so a script that fails or was deleted cannot keep an earlier conversion's pair; `--replace` sweeps the converted files out of both trees first (`ConvertedTrees`), after the Forge hook is probed, pair by pair with the sidecar first. A `.txt` counts as converted only beside its sidecar, so `check-convert`'s `check_report.txt` and `run.json` are neither swept nor make a tree occupied. The run exits non-zero when any script failed.
- **`check-convert`** — the conversion check: scores each converted card's converted ability text against the Oracle text in its Forge card script by word-bag similarity, and lists the cards below `--threshold`.
- **`vocabulary`** — builds the vocabulary from the converted-card corpus into `vocab.txt` in `--output-dir`. **Required before training or running the price transformer**, and read by every package that tokenizes converted cards. Also seeds set-code fragments from `AllPrintings.json` when available.
- **`train sklearn`** — trains the sklearn model (a `GradientBoostingRegressor`) on the log of the card price over eight feature groups: mana cost, card types, supertypes, keywords, power and toughness, layout, printing data, and TF-IDF word frequencies of the converted ability text. Saves `{timestamp}.joblib` + `latest.joblib`.
- **`train transformer`** — trains the price transformer (tunable `--d-model/--n-layers/--n-heads/--ff-dim`) on tokenized converted cards, with printing data beside the tokens. It targets the shifted-log price (`--log-offset`), oversamples by sampling bucket (`--sampler-exponent`), and keeps the epoch with the lowest loss on the held-out split.
- **`predict {sklearn,transformer}`** — reads a converted card (from `--file` or inline `--card`) and prints JSON with `predicted_price_eur` + `model_version`. Looks up the card's printing data by canonical name. For a hypothetical card, the price transformer uses default printing data, while the sklearn model gets no printing data at all, so its printing-data columns are zero.
- **`evaluate {sklearn,transformer}`** — scores a model on the held-out split: mean absolute error, median percentage error, median log error and top-20 overlap. The transformer adds a breakdown by reporting bucket, with the signed log error.
- **`serve`** — FastAPI + uvicorn REST service exposing `POST /api/v1/predict`. Loads the sklearn model (required), the price transformer (optional, graceful degradation), the tokenizer and the MTGJSON printing data. Accepts a converted card as `text/plain` and returns a prediction from every model it loaded.

The two price models have **different input contracts**:
- the sklearn model takes a parsed `Card` and runs it through `FeatureEngineering` to produce its feature vector;
- the price transformer takes the converted card's text, tokenizes it with the vocabulary, and receives `PrintingData` beside the tokens, never inside the text.

Prices are learned in log space and turned back into euros at inference, because the skew of the euro price distribution would otherwise let a handful of expensive cards dominate the loss. The sklearn model uses the plain log of the card price; the price transformer uses the shifted-log price.

The price transformer's price encoder is the alternate card encoder the sealed pipeline accepts through `--encoder-checkpoint`.

## Shared infrastructure

Other packages also import this package's general-purpose infrastructure:

- `forge_jvm`: classpath assembly, JVM launch and `ForgeWorkerPool`, the Forge worker pool of `sealed match-outcomes`, `draft play-draft-games` and the `effects` collectors. Given a `GracefulStop` it runs the match workers' stop protocol (`specs/025-ability-effect-model-gen3/contracts/cli.md` § Match worker stop protocol; the flags `add_worker_stop_flags`, `WorkerStopSettings`, the ends counted in `WorkerStops`); without one, as for `draft`'s workers, it kills every worker at the end. With or without, it kills its longest-running worker every status interval (`_kill_oldest_worker`), which is containment, not tuning (`src/effects/docs/collection-runs.md` § Short-lived workers). Every worker and one-shot JVM it starts is bound to a Windows job object with kill-on-close, so a hard-killed supervisor takes its JVMs with it. The first JVM command a process builds removes the variant scripts a killed `collect-variants` left staged (`clear_stale_staging`). `worker_count` is the `--workers` type of every command that starts a worker pool and of the shard readers of `effects rule-counts` and `build-corpus`: 1 to 12, an explicit default of 12, never the CPU count.
- `domain/worker_failures`, `domain/worker_status` and `infrastructure/worker_log_scan`: the failure lines a Forge worker logs, by kind, the JSON status line each worker JVM prints (hooks, pinned preferences, hook-failure counts, fork refusals, directed fork tally) and the judgement of the engine it shows (`judge_engine`, `WorkerEngineRefused`), and the incremental scan of a pool's logs.
- `domain/directed_fork_probabilities`: the one formula of a directed rule's fork probability, shared by `effects collect-variants` and `sealed match-outcomes`, which never imports `effects`.
- `torch_checkpoint`, `torch_training`, `ridge_probes` (the linear-probe harness over card embeddings), `dataclass_defaults.field_default`.
- `append_only`: readers that tolerate a trailing partial line or a cut gzip member, and `has_shard_record`, which reads a run's effect-record shards no further than the first record.
- The run manifest: `run_manifest` writes the `run.json` every file-writing command and script keeps in its run directory (`.claude/skills/artifact-layout/SKILL.md`), `annotate_run` lets the code under a running command add fields to its entry in `invocations`, and `cli_run_manifest` wires it to a CLI through one table per CLI, `RUN_MANIFESTS`, dispatched from `main`; this package's own table is in `infrastructure/cli.py`.

## Inputs on disk

- `resources/AllPrintings.json`, `resources/AllPricesToday.json` — MTGJSON dumps, the source of card prices and printing data.
- `../forge/forge-gui/res/cardsfolder/` and `../forge/forge-gui/res/tokenscripts/` — Forge card and token scripts, the sources of `convert`.
- The converted trees go where `convert --output-path` and `--tokens-output-path` say, under the artifact layout a generation's `output/<module>/<gen>/converted/{cardsfolder,tokenscripts}/`; each module converts its own. The token tree is kept apart from the card tree because converted token and card filenames collide and the sealed pipeline reads the card tree as its card corpus. `convert` has no default output tree; every other flag default (`output/cardsfolder/`, `models/price-predictor/...`) predates the layout: pass every path.
