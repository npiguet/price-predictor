# Specification Quality Checklist: Ability effect model

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-06
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
  template's.** `CLAUDE.md` states that `specs/NNN-name/` directories are written primarily for Claude
  to drive implementation and should be optimised for implementation clarity — detailed FRs,
  file paths, CLI surface, and cross-references. The named paths, flags, and module locations in the
  FRs are therefore contract, carried verbatim from the
  [root spec](../../2026-09-05-ability-effect-model.md), not leakage.
  The same applies to "written for non-technical stakeholders": the audience here is the
  implementation agent.
- No clarification markers were needed. The root spec pins every delegated decision, including all
  flag defaults, the eight-class sampling mixture, the hardcoded architecture constants, and all three
  gate thresholds; nothing was invented here.
- Three items were tightened during initial validation rather than left failing:
  - Success criteria were rewritten from field-level metric names to outcome statements, so SC-001
    through SC-009 read as verifiable results rather than restatements of the gates.
  - Edge cases were extended to cover partial shard lines, sidecar merge/dedup cases, filename
    collisions between the card and token trees, and corpus growth between training and evaluation.
  - The assumptions section now names the Forge version this branch builds against, since the
    feasibility findings are version-sensitive.

## Fidelity review

Three independent review rounds checked this spec against the root spec and the design record for
contradictions, omissions, and invented requirements. Every finding was verified against the source
text before being folded in. The spec grew from 99 to 124 requirements across the three rounds; no
finding required changing a decision, only restoring contract the derivation had compressed away.

- **Round 1** — five definitional gaps where a requirement named a mechanism the file never defined:
  the four `--variant` baselines (gate 1 depends on `identity`), the sparse field group's membership,
  gate 2's perturbation, the sampling rule for the two record kinds that carry no acting ability text,
  and the per-kind input-variation table. Also removed a "byte-identical" acceptance criterion that
  could only ever fail, since match-outcome rows carry per-run timestamps, run ids, and durations.
- **Round 2** — the root spec's § Output heads was the one table neither enumerated nor delegated,
  which had silently dropped the whole player output group and the created-objects slot schema. Also
  disambiguated `no-state`, whose wording admitted a reading that inverted the control.
- **Round 3** — no blocking findings. Restored the four held-out stratum definitions (gate 1 is
  defined against one of them), the pooling and concatenation method behind the decodability battery
  and the scorer smoke test, the non-trainer CLI surface, and roughly ten smaller clauses.

One round-3 finding was rejected after checking the code: `--target-size` post-truncating the
corpus-frequency vocabulary is how the sibling `sealed build-vocab` already works
(`src/sealed/application/build_vocab.py`, `_truncate_to_target_size`), so it is a correct derivation
rather than an invented mechanism. The reviewer had checked the shared utility instead of the wrapper.
