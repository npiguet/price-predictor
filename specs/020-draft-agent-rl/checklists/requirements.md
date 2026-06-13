# Specification Quality Checklist: Draft Agent — RL Self-Play Fine-Tuning (Generation 2)

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-06-13
**Last validated**: 2026-06-13
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

## Build-Surface Honesty (feature-specific)

- [x] The new-code surface is explicitly scoped: only the RL trainer (US1) and the trainer's multi-corpus input — no corpus schema change / no provenance field (Assumptions; US1; clarified 2026-06-13).
- [x] US2 (cross-generation yardstick) is stated as a procedure over existing commands with no new code (US2 narrative, FR-022–026, Assumptions).
- [x] US3 (self-play loop) is stated as a manual operator runbook with no new orchestration code, retained for the quickstart (US3 narrative, FR-027, Assumptions).
- [x] Promotion is described as a manual operator judgment, not a system-enforced rule (FR-025; US2 §2; US3 §1).

## Notes

- **Validation outcome (2026-06-13)**: all items pass. No `[NEEDS CLARIFICATION]` markers; ambiguities were resolved into the **Assumptions** section and (via `/speckit.clarify`) the **Clarifications** section.
- **Clarifications applied (2026-06-13)**: (1) corpus↔checkpoint pairing is operator convention only — no schema change, no hard provenance check; (2) in-run best-checkpoint/early-stop uses the held-out RL objective; (3) the on-policy gating check warns-and-continues rather than aborting.
- **RL vocabulary vs. implementation detail**: the spec names the RL update contract (on-policy actor-critic, GAE(λ), KL anchor, entropy bonus, REINFORCE-with-baseline) because that contract *is* the reviewable, technology-agnostic behaviour of the new trainer command — it specifies *what must be computed*, not language/framework/library choices. This is intentional and does not count as leaked implementation detail.
- **Greedy yardstick pick-mode** is flagged in Assumptions as an inferred convention: the design rationale pins the fixed-mix randomized co-seating explicitly but not the pick-mode. If the operator wants sample-mode evaluation instead, only the Assumptions/FR-022/FR-028 wording changes — no requirement depends on the choice.
