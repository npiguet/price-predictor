# Building the curated corpus

Rules to respect when running or changing `build-corpus` or `holdout-cards`, or anything that depends on the split, the strata, curation or the manifest. This file does not describe the build:

- Flags and defaults: `python -m effects build-corpus --help`.
- Placement, budgets and copies: `domain/stratum_placement.py`, `domain/corpus_curation.py`, `domain/rule_budgets.py`, `domain/budget_allocation.py`, `domain/effect_families.py`.
- What curation specifies: `specs/2026-10-30-ability-effect-model-gen3.md` § 12, `specs/025-ability-effect-model-gen3/contracts/cli.md`.
- Why: `experiments/2026-09-18-effect-model-gen2-improvements-design.md`, `experiments/2026-10-30-effect-model-gen3-design.md`.

## The dataset

- The output is `training/`, `validation/card-disjoint/` and `validation/game-disjoint/`, each an ordinary shard directory every reader loads unchanged, plus `manifest.json`.
- The dataset is a pure function of the raw corpus, the flags and `--seed`, all recorded in the manifest.
- A grown raw corpus is rebuilt whole, never extended in place: the split and the rarity table are corpus-wide quantities. Every output directory is cleared before the write pass, and a directory the delete leaves (a shard another process holds open) stops the build naming what was left (`CorpusStore.clear_outputs`, exit 1): its shards would otherwise be read with the new dataset's. `--verify` reports the manifest's drift against the corpus on disk and writes nothing.
- Placement depends on every game, so a rebuild over a grown corpus can move a game it placed before. Only the split a checkpoint records says where each game went.
- A failed build leaves no stale manifest: `manifest.json` is moved to `manifest.previous.json` before the strata are cleared, and the new one is written atomically, last.
- A checkpoint pins the manifest's digest, and `evaluate-effect-model` fails fast on a dataset rebuilt since. Every member is written, and each is inside the digest. The manifest names its `generation` (`3`, `MANIFEST_GENERATION`). `CorpusManifest.from_dict` refuses a manifest with another or no `generation`, naming gen-1 or gen-2, and one missing a member (`ManifestError`, exit 1 from `build-corpus --verify`, `train-effect-model` and `evaluate-effect-model`); there is no default-filling for an older manifest.
- The manifest's `shards` lists every shard of each output stratum by its file name in the stratum's directory, with the records the write pass put in it, sorted by name: what a reader compares the shards it finds with. It sits outside the digest, since the repack's naming follows from members inside it.
- `vocab_path` and `variant_scripts` are recorded, and enter the digest, as POSIX paths relative to the project root, absolute outside it (`corpus_report.recorded_path`): one file typed relative or absolute, with either slash, pins one digest. A run taking the variant tree from the manifest opens a relative one from the working directory, so it runs from the project root.
- The manifest records the unique-ability-text count of each output. That count, not any record count, says whether curation preserved the tail.
- Only gen-3 shards are accepted; a records set holding any other exits 1 before a record is read.
- A gen-3 game whose shard holds no game-end marker was cut, and its records are incomplete. The survey and the write pass both drop it whole (`dropped-cut`), so no stratum, derived copy or selection set holds one. The manifest counts them in `cut_games`, inside the digest. A records set none of whose gen-3 games has a marker is refused before `--output` is touched.
- A record the write pass meets past what the survey read in its shard (a raw shard grown between the passes, a collection still writing) was never surveyed nor placed. It is written to no stratum and counted as `dropped-unsurveyed`, and the build names every raw shard whose size changed.
- Before anything under `--output` is touched, the build refuses a rules directory holding no scenario or no heads line, and a raw shard the survey cannot read whole. The rules directory is read relative to the working directory, so a build started elsewhere would tag and budget nothing.
- What the survey's reading met without stopping (`decode_off_list`, `damaged_tails`, `join_gaps`: one count per textless join answer, `JoinGaps`), `no_acting_text_scripts`, `unsupervised_events`, `code_version`, `sidecar_digest` and `tagged_nowhere_written` and each rule row's `drawn` sit outside the digest, so that neither where the raw corpus sits nor what ran before in the process moves it.
- `--workers` is 1 to 12 with a default of 12, never the CPU count.

## The split

- Training records are selected per record. Both validation strata are selected per game, the unit the split is decided in, so the card-disjoint cap admits or declines a whole game.
- The split runs over the card tree and the token tree alike. A token script printing a held-out text is a carrier, is held out by its script file, and a game holding it is held out.
- A variant script whose sidecar prints a held-out text (by template) is a carrier too: a game in which any entity or key names one, acting or on the board, is held out whole (`split.record_names_held_out_card`, the sidecar read through `WorkerContext.variant_prints_held_out`).
- A game with a record naming a held-out card or token script goes to the card-disjoint stratum, random-seat games included. The game-disjoint stratum takes only games two Forge AIs played.
- A record of a training game that acts through a held-out text (a token script printing it, or its template; a variant printing it holds its game out whole) is not trained on. It is counted as `dropped-held-out-text`, and the rest of its game still trains.
- The card-disjoint stratum holds at most `--card-disjoint-text-cap` games per held-out text. A game counts toward a text only through a resolution record acting through it, the only kind gate 1 scores, and a held-out game resolving no held-out text is dropped rather than trained on. This keeps every held-out text weighing comparably in gate 1 and in checkpoint selection.
- No evaluator resolves a probe's `mirror_of` within a stratum.
- A build that would produce a dataset gate 1 cannot be scored on (an empty holdout, an empty card-disjoint stratum) writes nothing and exits non-zero. So does one asking for a validation sample (`--validation-sample` above 0) whose game-disjoint stratum is empty, which would leave the sample the trainer validates on empty; both are refused in `_decide_or_refuse`, before anything under `--output` is touched.
- After writing, the build exits 1 when a half's rule-test component misses its precision target. The corpus is still read after that exit. Only the rule tests are judged here; probes, grids and explained deviance wait for the selection-set freeze. The thresholds are config fields with no flag, so a test can lower them.

## Flags that must agree across commands

- `--holdout-permille` and `--holdout-max-carriers` must be passed with the same values to `holdout-cards`, `build-corpus`, `collect-variants` and `collect-coverage --only-cards`. A depleted corpus was composed against them and a checkpoint records them. All of them select through the one rule (`HeldOutCards`), so they agree whenever the values do.
- `--vocab-path` decides the encoding surface the rarity table keys its texts on. Built on the other surface, the table keys every text differently while looking exactly as valid, which is why `train-effect-model --corpus` refuses a `--vocab-path` on the other one.
- `--variant-scripts` names the perturbed-script tree. A records set holding a key of a source tree the run has no root for (a variant key without `--variant-scripts`) is refused after the survey, naming the tree, before `--output` is touched (`JoinGaps.unconfigured` counts such keys by tree).

## The holdout

- Membership depends on the text's own bytes (`crc32` of the masked template of the normalized script text) and on the carrier count (a text more cards carry than `--holdout-max-carriers` is never eligible), so a Forge upgrade never reassigns an existing text. That is what lets a depleted corpus stay valid.
- The holdout unit is the masked template, the only unit and no flag: digit runs outside chain labels, `CARDNAME` and description values are masked, so `NumDmg$ 2` and `NumDmg$ 3` fall on one side.
- Token scripts count as carriers, toward `--holdout-max-carriers` too.
- The trainer recomputes the holdout over its sidecars and refuses a disagreement with the manifest.
- A game counts toward the text `build-corpus` credits it to: its first acting key whose text is held out, through the one `HeldOutCards.text_acted_through` that `build-corpus` and `collect-coverage --only-cards` both call. Under `template`, a game whose first acting key prints a held-out template no listed card carries counts toward no listed text.

## `holdout-cards`

- Writes the depletion list `sealed generate-pools --exclude-cards` reads: one Forge canonical card name per line, a split card under its combined name ("Fire // Ice"), which is what Forge's exact-name depletion compares.
- It is the one coupling between `sealed` and `effects`, and deliberately a plain file rather than an import: `sealed` must never depend on `effects`.

## Curation

- The survey reads shards keyed by provenance key, not by ability text, and one process folds keys to texts afterwards. Resolving a key needs a `SidecarCache` over tens of thousands of files, which each worker process builds once and reads every shard through (`converted_trees.process_sidecars`), so a sidecar is read once per process rather than once per shard.
- Only training games' records size the class, family and text quotas and the copy plans, so the plan never counts a record the write pass routes away. The rarity table, the legality mode counts, the rule and condition counts, the closed subtype and keyword lists and the held-out texts' game counts stay whole-corpus.
- Every selection choice is a function of the record's hash and `--seed`.
- In an outcome signature, a zone outcome the targets leave unknown reads `unknown`, never `stayed`; the acting spell's own move off the stack is left out; and a half that did not resolve names its outcome (`fizzled`, `declined`), so a fizzle never shares a cell with a kill.
- A record no stratum can hold (a refused record, a `turn_action` record) has its rule tags counted in the manifest's `rule_counts` under the source `refused`, never under a natural source; `refused` is a manifest-side source, not a `RuleSource` member. `tagged_nowhere_written` lists the rules the counts name that no `rules` row of the training or a validation stratum writes.
- A resolution is refused as `no-acting-text` when every acting key maps to no line; a trigger, continuous or decision record, when every acting key resolves `UNCONVERTED`, `UNCONFIGURED_TREE` or `AMBIGUOUS` (`record_quality.NO_SIDECAR`). `DROPPED` and `RUNTIME_ONLY` keys are answers the join gives on purpose and keep such a record. The refusals are counted per sampling class.
- A validation sample holds each `record_id` once, its natural copy where it has one, so a class's quota counts distinct records whatever the stratum's copies.
- `--text-cap` counts every natural copy of the text in the class, of tagged and untagged records alike.
- Each class budget is split equally over effect families, then over outcome signatures. A short family's records repeat up to `--reuse-cap` and its unused share passes on. Real legality decisions fill up to half the legality class first.
- Rule budgets add copies to the rules the map marks `trained`. Every written copy carries a `curation` member, which the rule mask reads (`training.md`).
- A validation stratum reaches the training stratum's class mix as a weight, never by repeating rows: each record's natural copy is written once (`natural_share` 1) and none is dropped, and every row carries its class's `curation.class_weight` (`corpus_curation.validation_multipliers`, at most `--reuse-cap`; 1 on a training row). Its rule budgets are planned as training's, a class's target being its records, so a validation record holds at most `--reuse-cap` copies, natural and added. A reader weighing validation rows multiplies by `class_weight`; one taking each record once ignores it. Gate 1 takes a card-disjoint record's natural copy alone.
- A protection copy is written as a synthetic record `<record_id>~pc<i>`. One is made only where the snapshot shows what the rule reads; the conditions per kind of copy are in `domain/protection_copies.py`. A prevented-damage copy undoes the whole death the damage caused (the `destroyed`, the move to the graveyard, the `unattached` of each Aura or Equipment that fell off and the Aura's own move); a target-illegal copy needs an acting chain the sidecar shows with exactly one targeting clause of `TargetMin$` 1 (a fight spell has two); a block-illegal copy forbids a changeling blocker as it does every creature type, and keeps a forbidden blocker in `must_block`, which lists every creature a requirement makes block, legal or not. A record offering several protection rules enters the budgets once, tagged with each, and each copy counts toward its drawn rule alone in the rows' `written` and `added`. A natural record's added copy counts toward every rule the record is tagged with, so a rule's `added` can exceed the copies written for it: each rule row's `drawn` is the part of `added` whose `curation.drawn_rule` is that rule (the copies to count against the written records), and the manifest reports it outside the digest, since it decides nothing (`corpus_manifest._leave_out_drawn`).
- `turn_action` records are written to no stratum. A permanent spell's resolution is kept, keyed to the permanent's card, although its implicit cast spell has no line. A decision whose candidate key is unjoinable is refused as `unjoinable-key`.
