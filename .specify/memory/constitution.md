<!--
  Sync Impact Report
  ==================
  Version change: 2.2.0 → 2.3.0
  Modified principles: None
  Added sections:
    - VIII. Performance-Conscious Implementation (new principle
      requiring any feature that moves data or runs model compute
      to be reviewed against a checklist of recurring performance
      pitfalls — I/O batching/caching, GPU placement, GPU
      batching, vectorization, streaming, load-once reuse — with
      optimization beyond the checklist gated on measurement so it
      stays bounded by Principle II)
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — ✅ updated
      (Constitution Check now includes a Performance Review
      subsection for Principle VIII, applied when the feature
      moves data or runs model compute)
    - .specify/templates/tasks-template.md — ✅ updated
      (Notes section references Principle VIII; the Polish-phase
      performance task is tied to the Principle VIII checklist)
    - .specify/templates/spec-template.md — ✅ No change
      (performance is a plan/implementation concern, not part of
      the problem statement — same rationale as Principle VII)
  Quality Gates updated:
    - ✅ Added performance review gate: features touching data
      loading or model compute must pass the Principle VIII
      checklist
  Follow-up TODOs: None
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

### VII. Codebase-Aware Planning

Before a plan or task list is finalized, the agent MUST deliberately
survey the existing codebase and ground its design in what is already
there. Planning in isolation is forbidden.

- The `/speckit.plan` workflow MUST include an explicit codebase
  survey step before the Technical Context section is filled in. The
  survey MUST look for:
  - domain entities, value objects, ports, and services whose
    vocabulary overlaps with the feature;
  - utilities, adapters, infrastructure, and CLI subcommands that
    already solve adjacent problems;
  - conventions used by sibling modules (folder layout, naming,
    dependency direction, test style).
- Survey findings MUST be recorded in `research.md` under a
  `## Codebase Survey` section, with concrete file and symbol
  references so reviewers can verify them. `plan.md` MUST link to
  that section under its Constitution Check and summarize the
  outcome; the detailed findings do not belong in `plan.md`.
- New domain concepts MUST NOT silently duplicate existing ones. If
  a concept with a similar name or responsibility already exists, the
  plan MUST either (a) reuse it, (b) extend it, or (c) explicitly
  justify why a parallel concept is warranted and propose a rename of
  the older concept so the codebase converges rather than diverges.
- Tasks in `tasks.md` that introduce a new entity, service, port, or
  adapter MUST reference the nearest prior art identified in the
  survey. A new sibling is acceptable only when its divergence is
  explained.
- Where an existing library, internal utility, or upstream API (e.g.,
  Forge, MTGJSON loader, transformer encoder) already covers a
  sub-problem, the plan MUST prefer reuse. Reimplementation is
  permitted only with a documented reason (behavior gap, licensing,
  required isolation).
- When the survey reveals that the feature is a third instance of a
  pattern already present twice in the codebase, the plan MUST
  propose extracting the shared abstraction rather than hand-coding
  another parallel copy.

**Rationale**: Every planning session that skips the codebase survey
risks reinventing an existing entity, adding a sideways variant of a
domain concept, or designing against assumptions that the rest of the
code has already contradicted. The price predictor and sealed modules
share transformer encoders, tokenizers, Forge adapters, and MTGJSON
loaders; without deliberate awareness, features drift into pigeonholed
local solutions that rot as the surrounding code evolves. Treating
the survey as a first-class planning step — on the same footing as
the Constitution Check itself — is the cheapest insurance against
duplication and domain decay.

### VIII. Performance-Conscious Implementation

Any feature that moves data or runs model compute MUST be reviewed
against a checklist of recurring performance pitfalls before it is
considered complete. These are concrete, repeatedly-observed problems
in this codebase — not speculative concerns — so the checks are
mandatory, while optimization *beyond* them stays evidence-driven.

- **Batch and cache I/O.** Repeated or per-item I/O — card embedding
  `.npz` loads, MTGJSON lookups, file reads — MUST be batched and/or
  cached rather than re-read inside a loop. Deterministic, reusable
  artifacts (e.g., card embeddings) MUST be computed once and
  persisted/reused across iterations and runs.
- **Use the GPU when it helps.** Compute that meaningfully benefits
  from the GPU (model forward/backward, large tensor ops) MUST run on
  the GPU when one is available, with the model and its inputs on the
  same device. Work the GPU can do MUST NOT silently run on the CPU.
- **Batch GPU operations.** GPU work MUST be batched where batching
  helps, to amortize kernel-launch and transfer cost and avoid
  CPU↔GPU ping-pong. Per-item host↔device transfers (`.item()`,
  `.cpu()`, `.numpy()`, Python-scalar reads) inside hot loops MUST be
  hoisted out or accumulated on-device.
- **Vectorize hot loops.** Element-wise work over arrays, tensors, or
  dataframes MUST use vectorized numpy/pandas/torch operations instead
  of per-row Python loops on hot paths.
- **Stream large inputs.** Large files MUST be processed in a
  streaming/single pass (as the MTGJSON and cards-played readers
  already do) rather than materialized wholesale in memory when a
  streaming pass suffices.
- **Load once, reuse.** Expensive-to-construct objects (models,
  tokenizers, metadata maps) MUST be loaded once and reused across
  requests/iterations, not reconstructed per call.
- **Measure before going further.** Optimization beyond this checklist
  MUST be justified by a profile or measurement that identifies the
  hot path. This keeps the principle bounded by Principle II: do not
  trade simplicity for speculative speedups on cold paths.

**Rationale**: ML and data pipelines live and die by I/O and device
efficiency. The dominant costs here are loading card embeddings and
MTGJSON data, moving tensors to and from the GPU, and the per-call
overhead of unbatched kernels — and these exact issues have
repeatedly required after-the-fact fixes once a feature was already
implemented. Encoding them as a standing checklist, applied during
planning and again at review, catches them while the design is still
cheap to change instead of as a follow-up patch. The closing
"measure first" clause prevents this principle from inviting the
premature optimization Principle II forbids: the listed items are
known hot spots; anything beyond them needs evidence.

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
- Plans introducing new domain concepts MUST cite prior art
  from the codebase survey (Principle VII) or document why a
  parallel concept is warranted.
- Features that move data or run model compute MUST pass the
  Principle VIII performance review (I/O batching/caching, GPU
  placement and batching, no per-item host↔device transfers in
  hot loops, streaming for large inputs, load-once reuse).
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

**Version**: 2.3.0 | **Ratified**: 2026-02-26 | **Last Amended**: 2026-06-10
