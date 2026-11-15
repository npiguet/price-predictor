# Evaluation: gates, probes and the selection set

Rules to respect when running or changing `evaluate-effect-model`, a gate, the knowledge or embedding probes, the control score or the selection set. This file does not describe the reports:

- Flags and what is printed: `python -m effects evaluate-effect-model --help`, `application/breakdowns.py`, `application/control_score.py`.
- What is measured: `specs/2026-10-30-ability-effect-model-gen3.md` § 14, `specs/024-ability-effect-model-gen2/contracts/knowledge-probes.md`.
- The selection-set file: `specs/025-ability-effect-model-gen3/contracts/selection-set.md`.
- Why: `experiments/2026-09-19-effect-knowledge-probes-design.md`, `experiments/2026-10-30-effect-model-gen3-design.md`.

## `evaluate-effect-model`

- Splits are never a flag. The held-out card list and both `game_id` sets come from `--checkpoint`: the corpus is append-only, and a recomputed split would score the gates partly on games the model trained on.
- `--corpus` defaults to the dataset the checkpoint recorded. A checkpoint recording none is refused, and so is a corpus whose manifest digest has changed. A `--variant-checkpoint` recording a different split or vocabulary hashes fails fast.
- It exits 0 after any gate verdict, printing `GATES FAILED: …`, and 1 on a corpus not wholly gen-3 or a checkpoint of another input layout.
- The three FR-116 checks never built (matched real-vs-fork agreement, the probe-diff re-check, the role-polarity probe) report `NOT_IMPLEMENTED`, printed `[todo]`, on every run. They are not skips: the corpus holds the records they would read.
- Every geometry check keys the cache on the checkpoint's encoding surface, so on a script checkpoint two lines sharing prose but not script stay two points.
- A line query that resolves to nothing is named in the report as unresolved, never dropped.
- Every breakdown and both gates read each record once. A validation stratum repeats a record's row for every added copy.
- The selection set's reporting half is scored once, through the control score's own scorer. The selection halves belong to the trainer.

## The gates

- Gate 1, the identity baseline, asks whether the encoder is reading text. Pass/fail.
  - It is scored on the card-disjoint split's unique-text stratum: resolution records whose acting text no record of the curated `training/` stratum acts with. That is the only slice where a baseline holding a free vector per text cannot simply recall the answer.
  - Three margins must all hold: at least 0.05 absolute affected-gate F1 gain, at least 0.05 absolute zone-outcome accuracy gain, at least 5% relative negative-binomial deviance reduction, each model's deviance read under its own learned dispersions.
  - Zone accuracy is class-balanced over the observed outcomes, so answering "stayed" every time scores no better than chance.
  - A margin that cannot be measured fails.
  - There are three because they fail differently: the F1 says whether the model knows that something happens, the zone accuracy what, the deviance how much.
- Gate 2, the damage-step keyword canary, only routes: its per-keyword verdict decides whether the probe machinery is worth building.
  - Per keyword: at least 200 qualifying records and at least 70% direction agreement under a model-side perturbation, the carrier stripped of the keyword (its kept flag cleared and every line of the keyword dropped, printed or granted).
  - The population is the real combats of the game-disjoint games. A probe fork or an intervention qualifies nothing.
  - A row qualifies a carrier only on damage dealt or taken in the record's own step (rule 510.4), and a phased-out creature or a last-known copy takes no part. The full qualification rules are in `domain/damage_step_keywords.py`, which the survey's keyword threshold and the `keyword_combat` detector read too.
  - The 200 counts distinct `record_id`s; agreement averages over (record, carrier) observations.
  - Fields are read through `HeadReading`, the gate's probability times the expected value, since a field head trains only where an entity was affected.
  - Under-sampled routes the same way as wrong. Treating untested as passed is how a canary stops being one.
- Gate 3, the collapse canaries, is pass/fail: mean pairwise cosine at most 0.5 over 10,000 random pairs, top principal component at most 30% of variance.
  - One vector per unique ability text, unique by the text on the checkpoint's encoding surface, never by the vector's bytes: one text encoded in two batches differs in float noise.
  - Over the card and token trees, variant scripts left out.
  - It is checked on the geometry rather than the loss, because a collapsed space passes every average-case metric.

## Probes

- The knowledge probes are `scripts/effect_knowledge_probes/`, the embedding probes `scripts/effect_embedding_probes/`; their entry scripts are run by path.
- `common.probe_context(checkpoint)` refuses a checkpoint with no corpus, a corpus rebuilt since, or another input layout. `halves.load_selection` refuses a selection set frozen against another corpus or edited.
- The line and pair ladders read training records. They are reported only and never feed a stratum score.
- AUC and R² are never averaged together.
- The rung-3 `legal_target` readouts read the `target_legal` field, because the gate trains on no decision record.
- A line counts as held out only when every card carrying it is held out.
- The observed-mana label sums each `mana_produced` event's `mana_by_color`. An empty map is unreadable mana and counts for nothing.
- Every rung-3 head read is gated through `HeadReading`, except the ladder's `damage_taken`, read as the amount alone because every one of its items took damage.

## Selection set

- `selection-set-<corpus digest>.json`: canonical JSON with a `digest` over the rest, frozen once per curated corpus and found by the corpus digest, in the directory every reader is given (`--reports-dir` on `train-effect-model` and `evaluate-effect-model`, `--probe-set-dir` on the probes; by default `output/effects/reports/`).
- Freeze it with `scripts/effect_knowledge_probes/run.py --freeze-probe-set --corpus DIR`; no checkpoint is needed. The freeze exits 1 when a half holds fewer than 300 games, a component's standard error passes 0.05 or a grid cell is empty.
- A freeze never replaces a set already there by another: the same set again is a no-op, another exits 1 and leaves the file alone.
- Its halves (`selection`, `reporting`) are the manifest's and are never reassigned.
- `train-effect-model` scores the selection halves each epoch and refuses a corpus with no set, or one frozen against another corpus or edited. `evaluate-effect-model` and the probes score the reporting halves.
