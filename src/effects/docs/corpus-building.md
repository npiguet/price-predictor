# Building the curated corpus

Rules to respect when running or changing `build-corpus` or `holdout-cards`, or anything that depends on the split, the strata, curation or the manifest. This file does not describe the build:

- Flags and defaults: `python -m effects build-corpus --help`.
- Placement, budgets and copies: `domain/stratum_placement.py`, `domain/corpus_curation.py`, `domain/rule_budgets.py`, `domain/budget_allocation.py`, `domain/effect_families.py`.
- What curation specifies: `specs/2026-10-30-ability-effect-model-gen3.md` § 12, `specs/025-ability-effect-model-gen3/contracts/cli.md`.
- Why: `experiments/2026-09-18-effect-model-gen2-improvements-design.md`, `experiments/2026-10-30-effect-model-gen3-design.md`.

## The dataset

- The output is `training/`, `validation/card-disjoint/` and `validation/game-disjoint/`, each an ordinary shard directory every reader loads unchanged, plus `manifest.json`.
- The dataset is a pure function of the raw corpus, the flags and `--seed`, all recorded in the manifest.
- A grown raw corpus is rebuilt whole, never extended in place: the split and the rarity table are corpus-wide quantities. The three directories are cleared before the write pass. `--verify` reports the manifest's drift against the corpus on disk and writes nothing.
- Placement depends on every game, so a rebuild over a grown corpus can move a game it placed before. Only the split a checkpoint records says where each game went.
- A failed build leaves no stale manifest: `manifest.json` is moved to `manifest.previous.json` before the strata are cleared, and the new one is written atomically, last.
- A checkpoint pins the manifest's digest, and `evaluate-effect-model` fails fast on a dataset rebuilt since. For that reason the legacy members `token_keys_remapped`, `token_keys_ambiguous` and `forge_tokenscripts` are still written (as `0`, `{}`, `""`), and a new manifest member is omitted at its gen-1 value. Removing or always writing one changes digests that checkpoints pinned.
- The manifest records the unique-ability-text count of each output. That count, not any record count, says whether curation preserved the tail.
- Only gen-3 shards are accepted; a records set holding any other exits 1 before a record is read.

## The split

- Training records are selected per record. Both validation strata are selected per game, the unit the split is decided in, so the card-disjoint cap admits or declines a whole game.
- The split runs over the card tree and the token tree alike. A token script printing a held-out text is a carrier, is held out by its script file, and a game holding it is held out.
- A game with a record naming a held-out card or token script goes to the card-disjoint stratum, random-seat games included. The game-disjoint stratum takes only games two Forge AIs played.
- A record of a training game that acts through a held-out text (a token or variant script printing it, or its template) is not trained on. It is counted as `dropped-held-out-text`, and the rest of its game still trains.
- The card-disjoint stratum holds at most `--card-disjoint-text-cap` games per held-out text. A game counts toward a text only through a resolution record acting through it, the only kind gate 1 scores, and a held-out game resolving no held-out text is dropped rather than trained on. This keeps every held-out text weighing comparably in gate 1 and in checkpoint selection.
- No evaluator resolves a probe's `mirror_of` within a stratum.
- A build that would produce a dataset gate 1 cannot be scored on (an empty holdout, an empty card-disjoint stratum) writes nothing and exits non-zero.
- After writing, the build exits 1 when a half's rule-test component misses its precision target. The corpus is still read after that exit. Only the rule tests are judged here; probes, grids and explained deviance wait for the selection-set freeze. The thresholds are config fields with no flag, so a test can lower them.

## Flags that must agree across commands

- `--holdout-permille`, `--holdout-max-carriers` and `--holdout-unit` must be passed with the same values to `holdout-cards`, `build-corpus`, `collect-variants` and `collect-coverage --only-cards`. A depleted corpus was composed against them and a checkpoint records them. All four select through the one rule (`HeldOutCards`), so they agree whenever the values do.
- `--vocab-path` decides the encoding surface the rarity table keys its texts on. Built on the other surface, the table keys every text differently while looking exactly as valid, which is why `train-effect-model --corpus` refuses a `--vocab-path` on the other one.
- `--variant-scripts` names the perturbed-script tree. Without it a variant line resolves to no ability text, and a resolution record acting through it is refused as `no-acting-text`.

## The holdout

- Membership depends on the text's own bytes (`crc32` of the normalized script text), so a Forge upgrade never reassigns an existing text. That is what lets a depleted corpus stay valid.
- `--holdout-unit template` (the default) holds out masked templates: digit runs outside chain labels, `CARDNAME` and description values are masked, so `NumDmg$ 2` and `NumDmg$ 3` fall on one side. `text` reproduces feature 023.
- Token scripts count as carriers, toward `--holdout-max-carriers` too.
- The trainer recomputes the holdout over its sidecars and refuses a disagreement with the manifest.
- A game counts toward the text `build-corpus` credits it to: its first acting key whose text is held out, through the one `HeldOutCards.text_acted_through` that `build-corpus` and `collect-coverage --only-cards` both call. Under `template`, a game whose first acting key prints a held-out template no listed card carries counts toward no listed text.

## `holdout-cards`

- Writes the depletion list `sealed generate-pools --exclude-cards` reads: one Forge canonical card name per line, a split card under its combined name ("Fire // Ice"), which is what Forge's exact-name depletion compares.
- It is the one coupling between `sealed` and `effects`, and deliberately a plain file rather than an import: `sealed` must never depend on `effects`.

## Curation

- The survey reads shards keyed by provenance key, not by ability text, and one process folds keys to texts afterwards. Resolving a key needs a `SidecarCache` over tens of thousands of files, and building one per worker would cost more than the scan.
- Only training games' records size the class, family and text quotas and the copy plans, so the plan never counts a record the write pass routes away. The rarity table, the legality mode counts, the rule and condition counts, the closed subtype and keyword lists and the held-out texts' game counts stay whole-corpus.
- Every selection choice is a function of the record's hash and `--seed`.
- Each class budget is split equally over effect families, then over outcome signatures. A short family's records repeat up to `--reuse-cap` and its unused share passes on. Real legality decisions fill up to half the legality class first.
- Rule budgets add copies to the rules the map marks `trained`. Every written copy carries a `curation` member, which the rule mask reads (`training.md`).
- A validation stratum is written in the training stratum's class mix and drops no record. Gate 1 takes a card-disjoint record's natural copy alone.
- A protection copy is written as a synthetic record `<record_id>~pc<i>`.
- `turn_action` records are written to no stratum. A permanent spell's resolution is kept, keyed to the permanent's card, although its implicit cast spell has no line. A decision whose candidate key is unjoinable is refused as `unjoinable-key`.
