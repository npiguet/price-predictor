<!--
  Sync Impact Report
  ==================
  Version change: 1.1.0 → 1.2.0
  Modified principles: None
  Added sections:
    - Core Principles: V. JVM Interoperability (MTG Forge)
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ No change needed
      (Technical Context section is dynamic placeholder)
    - .specify/templates/spec-template.md — ✅ No change needed
    - .specify/templates/tasks-template.md — ✅ No change needed
  Feature plans requiring re-evaluation:
    - ⚠️ specs/001-card-price-predictor/plan.md — VIOLATION
      Plan currently specifies Python 3.11+ with scikit-learn.
      Principle V requires Java 17+ / JVM. Plan MUST be revised
      to use JVM-based technologies (Java, Kotlin, or Scala)
      with JVM ML libraries (Smile, Tribuo, DL4J, or similar).
    - ⚠️ specs/001-card-price-predictor/research.md — STALE
      R2 (Technology Stack) chose Python. Must be re-researched
      for JVM alternatives.
    - ⚠️ CLAUDE.md — STALE
      Lists Python 3.11+ as active technology. Must be updated
      after plan revision.
  Follow-up TODOs:
    - Re-run /speckit.plan for 001-card-price-predictor with
      JVM stack
-->
# Price Predictor Constitution

## Core Principles

### I. Fast Automated Tests (NON-NEGOTIABLE)

All features MUST come with automated tests that run quickly to
keep the feedback loop short.

- Every new feature or behavior change MUST include automated
  tests before it is considered complete.
- Test suites MUST execute quickly; slow tests (integration,
  end-to-end) MUST be clearly separated so the fast suite can
  run independently.
- Flaky or slow tests MUST be fixed or quarantined immediately;
  they erode trust in the test suite and slow development.
- Tests MUST be written before or alongside implementation,
  never deferred to "later."

**Rationale**: A fast, reliable test suite is the foundation of
confident iteration. If tests are slow or missing, developers
avoid running them, bugs slip through, and refactoring becomes
risky.

### II. Simplicity First

Start with the simplest solution that meets the requirement.
Avoid premature abstraction, over-engineering, and speculative
features.

- YAGNI: Do not build functionality until it is actually needed.
- Prefer clear, readable code over clever code.
- New abstractions MUST justify their existence with at least
  three concrete use cases before extraction.
- Configuration and extensibility points MUST be driven by real
  requirements, not hypothetical ones.

**Rationale**: Complexity is the primary enemy of
maintainability. A price predictor involves inherently complex
domains (data pipelines, models, market signals); the codebase
itself MUST stay simple to compensate.

### III. Data Integrity

Data flowing through the system MUST be validated, traceable,
and reproducible.

- All external inputs (API responses, user-provided data, file
  imports) MUST be validated at system boundaries.
- Data transformations MUST be deterministic and tested with
  known input/output pairs.
- Price predictions MUST be reproducible given the same input
  data and model version.
- Schema changes MUST be versioned and backward-compatible, or
  accompanied by an explicit migration plan.

**Rationale**: A price predictor is only as good as its data.
Corrupt, unvalidated, or non-reproducible data undermines every
downstream decision and erodes user trust.

### IV. Domain-Driven Design & Separation of Concerns

The codebase MUST be organized around domain concepts with clear
boundaries between layers.

- Code MUST be structured into distinct layers: domain (core
  business logic and entities), application (use cases and
  orchestration), and infrastructure (external services, storage,
  APIs).
- Domain logic MUST NOT depend on infrastructure details. The
  domain layer MUST be free of framework imports, database
  drivers, and HTTP concerns.
- Each bounded context MUST have a clear, explicit boundary.
  Cross-context communication MUST go through well-defined
  interfaces, never direct internal access.
- Business rules MUST live in domain entities and value objects,
  not in controllers, handlers, or infrastructure code.
- Dependencies MUST point inward: infrastructure depends on
  application, application depends on domain, domain depends on
  nothing external.

**Rationale**: A price predictor spans multiple sub-domains
(market data ingestion, prediction models, user-facing results).
Without strict separation, changes in one area leak into others,
making the system fragile and hard to test in isolation.

### V. JVM Interoperability (MTG Forge)

All code MUST target the Java Virtual Machine to ensure
interoperability with MTG Forge
(https://github.com/Card-Forge/forge).

- Code MUST be written in Java 17+ (preferred), Kotlin, or
  another JVM language that produces standard JVM bytecode.
- Build tooling MUST use Maven or Gradle to align with Forge's
  Maven-based build system.
- Libraries and dependencies MUST be available as JVM artifacts
  (Maven Central, JitPack, or similar repositories).
- The price predictor MUST be packageable as a JAR that Forge
  can consume as a dependency or module without requiring a
  separate runtime (no Python, no subprocess calls).
- Public interfaces MUST use standard Java types and collections
  so Forge code can call them directly without adapters or
  serialization layers.

**Rationale**: The price predictor will be integrated into MTG
Forge, a Java 17 Maven multi-module application. Using non-JVM
technologies would require inter-process communication, adding
complexity, latency, and deployment friction that violates both
Simplicity First and the goal of seamless integration.

## Quality Gates

Every pull request and feature delivery MUST satisfy these gates:

- All automated tests pass (fast suite MUST complete quickly).
- No new warnings from linting or static analysis tools.
- Data validation covers all new external input paths.
- Domain logic MUST NOT introduce infrastructure dependencies.
- Code MUST compile and pass tests on Java 17+.
- Code has been reviewed by at least one other contributor (or
  self-reviewed with a structured checklist for solo work).

## Development Workflow

- Features are specified before implementation
  (`/speckit.specify`).
- Implementation follows the plan generated by `/speckit.plan`.
- Tasks are tracked and completed in priority order.
- Each user story is independently testable and deliverable.
- Commits are small, focused, and reference the relevant task ID.

## Governance

This constitution is the highest-authority document for the
Price Predictor project. All development practices, code reviews,
and architectural decisions MUST comply with the principles above.

- **Amendments**: Any change to this constitution MUST be
  documented with a clear rationale, reviewed, and versioned
  before adoption.
- **Versioning**: This document follows semantic versioning:
  - MAJOR: Principle removed or fundamentally redefined.
  - MINOR: New principle or section added, or material expansion.
  - PATCH: Clarifications, typo fixes, non-semantic refinements.
- **Compliance**: All pull requests MUST include a constitution
  compliance check. Reviewers MUST verify alignment with the
  principles defined here.
- **Disputes**: When a principle conflicts with a practical need,
  the conflict MUST be raised explicitly and resolved by amending
  the constitution, not by silently bypassing it.

**Version**: 1.2.0 | **Ratified**: 2026-02-26 | **Last Amended**: 2026-02-26
