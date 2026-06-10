# June 1, 2026 — Draft agent spec and tasks

**TL;DR:** Ran the full speckit planning and task-generation pipeline
for feature 018 (the draft agent). The design landed on heavy reuse
from `sealed` and a genuinely small footprint of new logic.

The day was entirely about getting feature 018 — the MTG draft agent —
from spec into a committed, ready-to-implement task list.

The first session ran `/speckit.plan`, which drove a full codebase
survey before committing to any design choices. The survey confirmed
that the supervisor/worker pattern, `SAB`, `PickerModel`, `score_decks`,
`GreedyDeckBuilder`, `ConvertedCardLocator`, and the Forge-JVM launch
helpers could all be reused wholesale. The genuinely new logic came
down to two things: the booster-to-typed-token state reconstruction
(`POOL/PACK/PASSED/TAKEN` + pick recency) and the two-headed model
(imitation policy + Monte-Carlo critic). That survey also surfaced a
pre-existing `TODO(shared-trainer)` in `train_picker.py`, which noted
that a 4th trainer sharing the same resume/warmup/clip/best-checkpoint
scaffolding could eventually justify a `train_common` extraction —
but the decision was to follow the existing pattern first and defer
the extraction until after this trainer lands.

The plan session produced seven artifacts: `plan.md`, `research.md`
(7 decisions, D1–D7), `data-model.md`, `quickstart.md`, and three
contracts (`cli.md`, `drafts-jsonl.md`, `worker-protocol.md`).

The second session ran `/speckit.tasks` (35 tasks, T001–T035) followed
by `/speckit.analyze`. The analysis found nine issues — one HIGH
(no README update, violating constitution Principle VI), two MEDIUMs
(crash-recovery untested, val-metrics untested), and six LOW/MEDIUM
underspecification gaps. I applied all remediations immediately, which
added two new test tasks and expanded several existing ones, bringing
the final count to 37 tasks.

The key structural decision encoded in tasks.md was to place the
FR-016 geometry module and the JSONL IO layer in a "Foundational"
phase rather than inside US1, so that US2 and US3 don't silently
depend on US1 completing first.
