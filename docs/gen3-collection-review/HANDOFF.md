# Handoff: the gen-3 collection review is done; next is the fixing round (2026-11-10)

Read `README.md` in this folder first: why the review exists, the process, the status table and every
user decision. This file says where the work stands and what comes next.

## Where it stands
- Done and verified, findings committed:
  - `connector-findings.md` (forge-connector: four slices);
  - `forge-hooks-findings.md` (Forge `effect-record-hooks`: hooks and copier);
  - `python-collection-findings.md` (Python collection supervisors: run control; decks, card choice
    and holdout).
- The user left the `src/effects`-beyond-collection review and the `rules/`/`scripts/` review out for
  now (README status table). One check from it is still owed: whether `build-corpus` gives every stratum
  the same record composition (README, "Every stratum has the same record composition").
- Decided by the user (README, "User decisions so far"):
  - the record-moments principle, already written into the gen-3 specs (commit 71c7e03b: root spec
    § 8.14, delta contract, spec.md FR-042a–f, tasks.md phase 12);
  - random-seat games naming a held-out card are kept, in the card-disjoint stratum (spec change to
    FR-059 and root § 12.4 still to write);
  - the same decision-sampling rates in every stratum, so the held-out round drops its override;
  - variants: an untargeted breadth round plus a rule-steered round, with interventions preferring the
    aimed variants. This is a new feature: it needs a spec section, plan and tasks before code.
- Do not run `collect-variants` before python R1 is fixed: the old `loadable.txt` in
  `output/effects/variant-scripts/` would poison every Forge startup on the machine.

## Next: the fixing round, in a fresh session
1. Load the feature-workflow skill before every edit under `specs/` or `experiments/`.
2. Write the spec changes the decisions need: FR-059 / root § 12.4 (random-seat card-disjoint games),
   stratum composition, and the variant rounds (untargeted, steered, intervention preference). Then add
   tasks to tasks.md beside phase 12 (T178–T181).
3. Plan one pass over every confirmed finding in the three findings files, test-first. Each finding names
   the test that pins it. Decide where the attribution findings are fixed (connector or Forge hook; README
   "The fixing round").
4. Verify before any long run (README "The fixing round", tasks.md T181), then move
   `output/effects/records-gen3/` aside and re-collect from scratch.

## Session facts the next session needs
- The machine is idle: no collection runs, no JVMs. The old full-strength run (7c8cb070) was stopped by
  the user and will not be resumed; everything is collected again from scratch after the fixes.
- Forge `../forge` is on `effect-record-hooks` at 04e72e0518e. The connector fat JAR is built from
  2acdc5d7 (later commits are docs only). Nothing is pushed.
- Forge's custom-cards directory is `../forge/forge-gui/data/custom/cards/`; on 2026-11-09 it held only
  an empty `.staged-by-effects.list`.
- The fork fidelity audit (`ForkFidelity`, `fork fidelity:` lines in the worker logs) is on in every
  collection. The last audit run showed about 5% of forks differing; see the table at the end of
  `connector-findings.md`.
- User preferences that apply here:
  - plain-text questions, never the question pop-up;
  - collection bugs matter more than code style;
  - the user wants everything found before re-collecting;
  - commit freely, never push unless asked;
  - heavy jobs one at a time; at most 12 busy processes.

## If a module review is resumed later
- Split a large module into slices. Each slice goes to a fresh general-purpose subagent on Opus (never
  Fable; at most 3 Opus agents at once) with `review-brief.md` (plus `review-brief-forge.md` for Forge)
  and `review-hints.md`.
- Reviewers are read-only: no builds, tests, JVMs or pytest. They do not read `~/.claude`,
  `experiments/`, `diaries/` or git history. Bug hunting comes first; the `/review` rubric second.
- Save each report to the session scratchpad, then send it to a fresh Opus verifier told to refute each
  finding. Write only verified results into a findings file laid out like `connector-findings.md`, then
  commit.
