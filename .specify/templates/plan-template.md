# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]  
**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]  
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]  
**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]  
**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]
**Project Type**: [e.g., library/cli/web-service/mobile-app/compiler/desktop-app or NEEDS CLARIFICATION]  
**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]  
**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]  
**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

[Gates determined based on constitution file]

### Codebase Survey (Principle VII — required)

*GATE: Complete BEFORE filling in Technical Context below. Design must
be grounded in the existing codebase, not conceived in isolation.*

Before proposing any data model, module layout, or new domain concept,
survey what already exists in this repository and record the findings
in `research.md` under a `## Codebase Survey` section. Each bullet
MUST cite concrete files/symbols so a reviewer can verify.

The survey in `research.md` MUST contain these four subsections:

- **Overlapping domain vocabulary** — existing entities, value
  objects, ports, and services whose names or responsibilities
  overlap with the feature. For each, record the decision: reuse,
  extend, or (with justification) introduce a parallel concept. If a
  parallel concept is introduced, propose how the older concept
  should be renamed so the codebase converges.
- **Adjacent prior art** — existing utilities, adapters,
  infrastructure, or CLI subcommands that already solve adjacent
  sub-problems (e.g., MTGJSON loading, Forge bridging, transformer
  encoding, tokenization, model persistence). For each, record the
  decision: reuse, wrap, or reimplement (with documented reason).
- **Convention alignment** — the sibling module whose structure this
  feature should mirror (folder layout, naming, dependency direction,
  test style). Deviations MUST be justified.
- **Third-instance check** — if any sub-problem of this feature is
  already solved *twice* elsewhere in the codebase, the survey MUST
  propose extracting the shared abstraction rather than adding a
  third parallel implementation.

In this `plan.md`, under the Constitution Check, record only the
pointer and the outcome:

- Link to `research.md#codebase-survey` (or equivalent anchor).
- One-line status per subsection: e.g., "Overlapping vocabulary: 2
  concepts reused, 0 parallel concepts introduced."
- Any follow-up tasks the survey surfaced (e.g., a rename, an
  extraction) that must be added to `tasks.md`.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
