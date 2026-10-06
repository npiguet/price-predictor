# Specification Quality Checklist: Ability effect model — generation 2

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-10-06
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

- **"No implementation details" is read against this project's convention, not the generic
  template's.** `specs/NNN-name/` directories are written for Claude to drive implementation, so the
  flags, module names, file paths and class names in the FRs are contract carried from the
  [root spec](../../2026-10-06-ability-effect-model-gen2.md), not leakage. The same applies to
  "written for non-technical stakeholders": the audience is the implementation agent.
- **Two clarifications resolved with the user (2026-10-06)**, both gaps in the root spec:
  - a modal `option` line encodes its own mode's chain, and `convert` gives option lines
    provenance and `script_text` (FR-005a);
  - the random seat's draw at the play decision never passes while something is playable, whether
    the Forge AI would have played or passed (FR-022a).
- **`/speckit.clarify` session 2026-10-06 resolved four more points**, recorded in the spec's
  Clarifications section: the random seat's element-by-element draws (FR-022b), `--text-cap`
  counting repeated copies (FR-049), cost targets from `Cost$` only (FR-058a), and charms exposed
  as a root plus option lines and recorded per mode (FR-001, FR-005a, FR-029a–f, FR-063a). All four
  amend or extend the root spec, which has not yet been updated to match.
- **Requirements added beyond the root spec's text**, each a default the root spec implies but does
  not state. Review them during `/speckit.clarify`. FR-033 defines no default for absent envelope
  fields, because the user confirmed no gen-1 shard enters the gen-2 corpus.
  - FR-003: an SVar is emitted once per chain, and a missing SVar is reported and skipped;
  - FR-034: `CLAUDE.md` and feature 023's record-schema contract list the two new fields;
  - FR-042: a manifest or checkpoint without a holdout unit reads as `text`;
  - FR-052: selection is deterministic per corpus, flag set and seed;
  - FR-057: the encoder's fixed-σ noise is removed, since the batcher now applies the noise;
  - FR-061: a checkpoint without encoder-size fields loads at 4 layers and width 256, which is
    what lets the gen-1 checkpoint load.
