# Handoff: the gen-3 fixing round is done; next is a review pass, then the re-collection (2026-11-12)

Read `README.md` in this folder first: why the review exists, the process, the status table and every
user decision. `FIX-PLAN.md` says where each finding was fixed and what each audit found. This file
says where the work stands and what comes next.

## Where it stands
- Every confirmed finding of the three findings files is fixed, test-first (tasks.md T179, T180,
  T182–T193): Forge hooks (`WP1-HOOKS.md`), copier (`WP2-COPIER.md`), record moments, attribution and
  channels, snapshots, the Python supervisors, and the two variant rounds.
- The user's decisions since the review, all specced:
  - random-seat games naming a held-out card go to the card-disjoint stratum, and serve as rule-test
    records there (FR-059);
  - every stratum has one record composition (FR-059a); `build-corpus` did not give the validation
    strata the class mix and now repeats their records toward it;
  - two variant rounds, untargeted and steered (root § 8.15, FR-058a–g); scenario variants carry
    `script_edit` (207 of 260; the rest in `skipped-script-edits.txt`); every variant is written with
    its sidecar before play and kept;
  - the Forge AI's own loops (Chain of Smog copying forever, free no-op activations) are left alone;
    a per-game record cap (`--max-records-per-game`, 5000) and no zero perturbations contain them.
- Verification (2026-11-12): ruff clean; unit suite 5489 passed; Forge hook and copier tests pass;
  connector `mvn test` 1658 passed; integration suite 65 passed and 4 skipped (missing checkpoints,
  opt-in forks), its last failure fixed in d84bab69 (a charm that fizzled or was declined as a whole
  now writes one empty half per chosen mode); `validate-corpus` passes on the audit shards.
- Fork fidelity audit (T181): audits ran until the user stopped them (2026-11-12). The last
  (`output/effects/records-gen3-audit6`, 10 min) showed 6 of 103 forks differing (5.8%), from three
  rare copier gaps since fixed (a delayed trigger's remembered token that left the game, an emblem
  copied as a floating effect, a face-down card's hidden-face keywords: Forge d44cf28493c,
  936a41d719d, c9a71fb0650). Each audit draws different cards and turns up a few rare gaps; the
  `fork fidelity:` lines stay on in every collection, so the re-collection's logs show any left.
  The audit compares keyword, ability and line lists as multisets (order never matters, repeats do).
- Also changed after the first handoff, by user decision: no `continuous` record and no forced
  resolution at the untap boundary (FR-041b; `turn_action` records stay as they are);
  `untapped_production` counts only mana abilities that can be activated now, read as their
  controller's (delta contract; 74621cdb, 06210fe2). Not checked there: the ability's own mana cost
  (a filter land's {1}).
- `field-coverage`: `players[].rad` (Fallout, not in sealed pools) and `land_plays_unlimited` (rare)
  stay constant; check them again after the coverage round.
- Stock Forge, not fixed (fork cost only): statics with `IgnoreEffectCost` (Lost in Thought) add a
  command on every static check; Reconfigure adds a leaves-play command per attach.

## Next
1. The user runs a review pass with a stronger model before the re-collection: first the fixing
   round's diff (since 4c0b8acc here, and on `effect-record-hooks` since 04e72e0518e), in particular
   the record moments, `StaticSuppression` and the "without" reads, `ProvenanceResolver`'s trait
   counting, `ManaAffordability` (an approximation) and the steered round; then `src/effects` beyond
   collection and `rules/`/`scripts/`. Same process as before: slices, read-only reviewers with
   `review-brief*.md` and `review-hints.md`, an adversarial verifier per finding.
2. Fix what it confirms, test-first, and rerun a 10-minute audit.
3. Ask the user, then move `output/effects/records-gen3/` and `output/effects/variant-scripts/` aside
   and run stage 5 of `specs/025-ability-effect-model-gen3/quickstart.md` from scratch, the steered
   variant round last (T181).

## Session facts the next session needs
- Forge `../forge` is on `effect-record-hooks`; the connector fat JAR is built from this branch's
  HEAD. Nothing is pushed.
- Heavy jobs (pytest, mvn, JVM runs, audits) run one at a time through a lock wrapper: a script that
  creates a lock directory with `mkdir` (waiting while it exists), runs the job and removes it.
- Audit run: `match-outcomes --workers 6 --effect-records <dir> --random-seat-share 0.25
  --random-seat-probability 0.5 --interventions-per-game 4 --probes-per-game 4
  --directed-fork-rules <rules/directed-fork-rules.txt, comma-joined> --count-dropped-events`, stopped
  after 10–20 minutes; then count the `fork fidelity:` lines by field and caller.
- Audit directories `output/effects/records-gen3-audit*` are scratch, not corpus.
- User preferences: plain-text questions; collection bugs before code style; everything found
  before re-collecting; commit freely, never push; heavy jobs one at a time, at most 12 busy
  processes; never launch Fable agents unless the user says so.

## If a module review is resumed later
- Split a large module into slices. Each slice goes to a fresh reviewer with `review-brief.md` (plus
  `review-brief-forge.md` for Forge) and `review-hints.md`.
- Reviewers are read-only: no builds, tests, JVMs or pytest. They do not read `~/.claude`,
  `experiments/`, `diaries/` or git history. Bug hunting comes first; the `/review` rubric second.
- Save each report, then send it to a fresh verifier told to refute each finding. Write only verified
  results into a findings file laid out like `connector-findings.md`, then commit.
