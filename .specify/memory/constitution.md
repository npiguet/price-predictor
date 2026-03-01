<!--
  Sync Impact Report
  ==================
  Version change: 2.0.0 → 2.1.0
  Modified principles: None
  Added sections:
    - VI. Documentation (new principle requiring README,
      workflow descriptions, ML process rationale, and
      artifact documentation)
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ No change needed
      (Documentation deliverables can be captured in the
      existing Deliverables section of any plan.)
    - .specify/templates/spec-template.md — ✅ No change needed
      (Specs already have a scope section that can include
      documentation deliverables.)
    - .specify/templates/tasks-template.md — ✅ No change needed
      (Documentation tasks can be added as a Polish phase
      task using existing categories.)
  Quality Gates updated:
    - ✅ Added documentation completeness gate
  Follow-up TODOs:
    - Create README.md covering executables, workflows, ML
      rationale, and artifacts for feature 001
    - Carry forward from v2.0.0: Update plan.md to include
      Java stub library deliverable
    - Carry forward from v2.0.0: Define API contract between
      price predictor and stub
    - Carry forward from v2.0.0: Add Java 17+ / Maven to
      CLAUDE.md for stub library
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

### V. MTG Forge Interoperability (Java Stub + Remote API)

Interoperability with MTG Forge
(https://github.com/Card-Forge/forge) MUST be achieved by
providing a Java "Stub" library that calls the price predictor
application's remote API.

- The price predictor application itself MAY use any technology
  stack (e.g., Python, JVM, or other) for its core logic,
  provided it exposes a well-defined remote API.
- A Java 17+ stub library MUST be provided that MTG Forge can
  consume as a standard Maven/Gradle dependency.
- The stub library MUST use standard Java types and collections
  in its public API so Forge code can call it directly without
  adapters.
- The stub library MUST handle all remote communication details
  (HTTP calls, serialization, error handling) internally; Forge
  code MUST NOT need to know about the underlying remote API.
- The remote API contract MUST be versioned. Breaking changes
  to the API MUST be accompanied by a corresponding stub
  library update and a migration guide.
- The stub library MUST include graceful error handling for
  network failures, timeouts, and service unavailability so
  that Forge remains stable even when the price predictor
  service is unreachable.

**Rationale**: MTG Forge is a Java 17 Maven application. A Java
stub library provides seamless integration from Forge's
perspective while allowing the price predictor to use the most
appropriate technology for ML workloads (e.g., Python with
scikit-learn). The remote API approach cleanly separates the
prediction service from the game client, enabling independent
deployment, scaling, and technology evolution.

### VI. Documentation

The project MUST include documentation that enables users and
contributors to understand, operate, and extend the system
without reading the source code.

- A README file MUST exist at the project root explaining how
  to launch all executables and run all workflows (training,
  prediction, evaluation).
- A textual description of each application workflow MUST be
  provided, covering inputs, processing steps, and outputs.
- A textual description of the ML processes chosen during
  implementation MUST be included, with explicit rationale for
  why each approach was selected over alternatives.
- A description of all artifacts produced by the application
  MUST be maintained (trained model files, evaluation reports,
  prediction output formats, etc.).
- Documentation MUST be kept up to date: when a feature changes
  behavior, its documentation MUST be updated in the same
  commit or pull request.

**Rationale**: Code without documentation is accessible only to
its authors. A price predictor combines domain-specific ML
pipelines with data ingestion and CLI tooling; without clear
documentation, onboarding is slow, workflows are opaque, and
users cannot evaluate whether the system meets their needs.

## Quality Gates

Every pull request and feature delivery MUST satisfy these gates:

- All automated tests pass (fast suite MUST complete quickly).
- No new warnings from linting or static analysis tools.
- Data validation covers all new external input paths.
- Domain logic MUST NOT introduce infrastructure dependencies.
- Main application code MUST pass all tests in its native stack.
- Java stub library MUST compile and pass tests on Java 17+.
- Remote API contract tests MUST pass for both stub and server.
- Documentation MUST be complete for any new or changed
  workflows, CLI commands, artifacts, or ML processes.
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

**Version**: 2.1.0 | **Ratified**: 2026-02-26 | **Last Amended**: 2026-03-01
