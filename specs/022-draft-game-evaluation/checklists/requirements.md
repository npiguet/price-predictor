# Specification Quality Checklist: Draft agent game-played evaluation

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-09
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

- Items marked incomplete require spec updates before `/speckit.clarify` or `/speckit.plan`.

### Validation record

Checked against the root spec clause by clause. Every requirement in the root spec has a
corresponding FR, and every FR traces back to a root-spec clause; nothing was invented here
that the root spec does not settle.

Specific checks:

- Requirements name roles rather than paths and formats. FR-012, FR-017 and FR-020 defer
  the file formats to the draft-agent and sealed deck-picker specs, so concrete paths
  appear only in the root spec, Assumptions and Dependencies.
- Each success criterion is checkable without reading the implementation. SC-002 and SC-003
  are properties of the output file; SC-001 and SC-004 are checked by running the existing
  tally; SC-005 and SC-006 are properties of a run that encountered failures.
- The three user stories are independently deliverable: US1 alone produces a usable
  measurement, US2 filters it, US3 changes only when the run stops.

No [NEEDS CLARIFICATION] markers were written into the spec. The root spec settles every
question this feature raises, including the two that would otherwise have qualified — how
mirror pairs are handled, and whether repeated pairings are permitted.

### Clarification session, 2026-08-09

Three gaps the root spec does not reach were resolved interactively and are recorded in the
spec's Clarifications section:

1. **Reproducibility under concurrency.** The original SC-003 promised that a seed makes
   the pairing sequence repeat, which concurrent completion cannot honour. The criterion
   was dropped. The seed option itself was later removed as well, once sampling moved into
   the workers left it connected to nothing.
2. **Run reporting.** The spec required only a startup echo, so a run that had stopped
   producing looked like one that was working. FR-023 and FR-024 adopt the sealed
   match-generation supervisor's status line and worker-lifecycle lines unchanged, and
   FR-025 adds an end-of-run summary. SC-006 makes the failure mode observable through the
   status line's match rate and live-worker count rather than through a skip total, which
   the autonomous-worker design leaves nobody in a position to count.
3. **Worker-count default.** The Assumptions section deferred to "the sealed convention",
   which does not exist — `match-outcomes` defaults to 12 and `evaluate-scorer` to 4.
   Fixed at 12 in FR-009.

### Architecture revision, 2026-08-09

The Python/Java split was reconsidered after the plan was written. Workers no longer receive
assigned pairings; Python projects the corpus once into a flat seat table and each worker
samples pods and pairs from it on its own, mirroring `GeneratedDecksIndex`. Two spec
requirements changed as a result, neither of them cosmetic:

- **FR-007** counts matches recorded rather than pairings drawn, because the supervisor no
  longer issues pairings and so cannot count them.
- **FR-025 and SC-006** drop the skipped-match total. Nothing is positioned to count it, and
  nothing is lost by a failure: the worker that hit it draws another pairing immediately.

### Scope addition, 2026-08-09

User Story 4 and FR-026 through FR-030 add `--forge-native-fraction`, diverting a share of
`forge-full` seats to Forge's own sealed deck builder. One question was resolved with the
user: a diverted seat carries the label `forge-native`, not the recorded one, because the
tally groups by that column alone and a shared tag would average the rebuilt and recorded
decks into a single win rate — collapsing the very comparison the option exists to make.
