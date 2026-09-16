# Corpus and trainer rework: design record

Written 2026-09-16 from the diagnosis of the first full training run on the
curated corpus (`models/effects/effect-model/10.08.log`, seed 2853911108).
Extends spec `specs/023-ability-effect-model/spec.md`; the FR amendments are
listed at the end. Three plans implement it:

- `docs/superpowers/plans/2026-09-16-build-corpus-rewrite.md`
- `docs/superpowers/plans/2026-09-16-trainer-consumes-curated-corpus.md`
- `docs/superpowers/plans/2026-09-16-token-script-keys.md`

## What the run showed

| epoch | train | card-disjoint | game-disjoint |
|---|---|---|---|
| 1 | 11.19 | 7.73 | 8.22 |
| 2 | 0.59 | 134.54 | 136.45 |
| 3 | -2.49 | 4.35 | 4.20 |
| 10 | -1.22 | 2.92 | 2.60 |
| 13 | -0.42 | 3.02 | 2.65 |

Findings, each verified against the corpus on disk or the code:

1. **The trainer's card-disjoint sample is not gate 1's slice.** The stratum
   on disk holds ~2.1M records over 5,841 games and roughly 8% of its
   resolution records act on a held-out text (~35k records). The trainer
   fills a 2,048-record sample per class with whatever arrives first and got
   20 unique-text records. Best-checkpoint selection and early stopping ran on
   in-distribution text.
2. **Whole-game event dumps are attributed to single records.** In a
   40-shard sample: 3% of resolution records have no ability, 10% carry events
   marked `attributed_to: unresolved`, and 16 records are whole games (one has
   278 events, 88 draws and a `player_won`). The build keeps records with no
   acting key uncapped by design (`_capped_class_records` docstring).
3. **Output shards mirror source shards one to one**, so 635 of 2,309 shard
   visits held 20 records or fewer and took 27% of all optimizer steps,
   replaying a handful of records 640 times each.
4. **The trainer re-applies the class mixture per shard with replacement**
   even though the manifest's `delivered_mix` already matches `class_mix`
   to within 0.5%. Combined with (3) this is what amplifies (2).
5. **Joint gradient clipping starves the encoder.** One optimizer group;
   head norms 5 to 20, encoder norms 0.1 to 0.3; clipping at 1.0 scales the
   encoder's update 5 to 20 fold below its own gradient.
6. **The epoch-2 validation spike is the curriculum.** Sparse fields enable
   at step 10,000, which is shard 244 of epoch 2; validation then scored 25
   freshly initialised heads. Losses before and after are not comparable.
7. **Poisson NLL with `full=False` has a data-dependent negative floor**
   (k - k ln k per count), so the printed train loss tracks batch composition,
   not model quality.
8. **Token abilities in forked games resolve to no text.** `GameCopier`
   rebuilds tokens from `TokenInfo`, which drops the `PaperToken`; the
   collector's `tokenScriptStem` then returns null and the key falls back to
   `cardsfolder/z/zombie_token.txt`. Affects context entities in ~0.6% of
   records, all forked combat records. Not the acting ability.
9. **A half-percent held-out leak in depleted games.** 254 of 52,822 depleted
   games played a held-out card; all but one opened The Hobbit or Marvel
   Super Heroes boosters. The exclusion list matches the manifest exactly, so
   the cause is in those two sets' booster slots. The routing rule handles
   it correctly (63 games went to the stratum). Not a design problem.

## Decisions

- **build-corpus owns every corpus-shaped decision.** Record quality,
  the gate-1 slice, the fixed validation samples, uniform shard size and the
  leak report are all build outputs recorded in the manifest.
- **The trainer reads and does not derive.** A curated corpus becomes the
  only training input. The startup sweep, per-shard mixture resampling and
  the raw-corpus split derivation go.
- **Trainer mechanics that stay training-time:** which shards an epoch draws,
  how batches are cut (weighted shuffle without replacement, rarity weights
  from the manifest), the schedule, clipping, and the curriculum, which moves
  to an epoch boundary.
- **Selection stays on card-disjoint loss (FR-089)**, but the card-disjoint
  sample's resolution classes are drawn from the gate-1 slice, so the number
  measures unseen text. Gate F1, zone accuracy and deviance are printed each
  epoch for reading, not selection.
- **The token fix is in the collector**, keyed off the copied card's image
  key, which survives `TokenInfo` and names the script stem.

## Spec amendments

| FR | change |
|---|---|
| FR-081 | unchanged; add: count losses use the full Poisson NLL (Stirling term) so every term is nonnegative at its optimum |
| FR-082 | `--curriculum-step` becomes `--curriculum-epoch` (default 3); sparse fields enable at the first step of that epoch; validation scores the field set the epoch trained with; the early stopper resets at the boundary |
| FR-086 | rarity weights apply inside a weighted shuffle without replacement over the resident shard; the per-batch class draw is removed |
| FR-088b | the thin-stratum warning reads `per_stratum["gate-one"]` from the manifest |
| FR-089 | unchanged; the card-disjoint sample is the build's `validation/samples/card-disjoint.jsonl.gz` |
| FR-095 | "per-parameter-group gradient clip 1.0" means two groups, `encoder` and `head` (plus `identity` for that variant) |
| FR-125 | withdrawn: the trainer no longer reserves shards or sweeps; `--corpus` is required |
| FR-135 | outputs gain `validation/gate-one/` and `validation/samples/{card-disjoint,game-disjoint}.jsonl.gz`; each output shard holds at least `--shard-records` records and closes at the first game boundary after that, so a game is never split and the last shard may be short |
| FR-148 (new) | build-corpus drops records with a quality defect: a resolution record with no acting ability, any **non-combat** record with an event attributed to `unresolved`, any record with more than `--max-events-per-record` events (default 64); counts by reason in the manifest. Combat is exempt from the unattributed rule because a damage step resolves nothing, so the collector attributes every combat event to `unresolved` by design and carries the cause in `cause` — the rule applied there refuses 98.9% of the class. The build stops rather than writing a dataset when the rules refuse more than half of any one sampling class |
| FR-149 (new) | build-corpus reports, per top-level source directory, games naming a held-out card, and warns when a directory routes fewer than 5% of its games (a leak, not a full-strength source) |
| FR-150 (new) | the collector resolves a token host without a `PaperToken` through its image key against the token database before falling back to a cardsfolder path |
