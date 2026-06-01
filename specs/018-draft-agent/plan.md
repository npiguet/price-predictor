# Implementation Plan: Draft agent — imitation policy + critic (generation 1)

**Branch**: `018-draft-agent` | **Date**: 2026-05-31 | **Spec**: [`spec.md`](./spec.md)
**Input**: Feature specification from `specs/018-draft-agent/spec.md`; normative
`specs/2026-05-28-draft-agent.md`; rationale
`experiments/2026-05-30-draft-agent-design.md`

## Summary

Build a generation-1 MTG-draft agent in a new top-level `draft` package: a
two-headed set transformer (imitation **policy** over `PACK` tokens + **critic**
on a `CONTEXT` token) trained offline from a corpus of Forge-generated drafts.
Three deliverables: a `generate-draft-data` CLI (Python supervisor + Java
`DraftWorkerMain`) that records complete drafts and labels each seat's deck with
the frozen sealed scorer; a `train-draft-agent` CLI that jointly trains both
heads (CE on whitelisted seats + standardized MC-regression MSE on all
non-failed seats); and a one-off picker-vs-SA builder-validation script. The
whole stack reuses `sealed`'s scorer, picker, greedy builder, embedding layout,
card locator, checkpoint plumbing, and the Forge supervisor/worker pattern; the
only genuinely new logic is the typed-token draft-state reconstruction
(`POOL/PACK/PASSED/TAKEN` + recency) and the two-headed model. No RL, no live
Forge seat (gen 2).

## Technical Context

**Language/Version**: Python 3.14+ (project requirement); Java 17+ for the
`forge-connector` worker.
**Primary Dependencies**: `torch` (model + training), `numpy`, the in-repo
`sealed` and `price_predictor` packages; Forge `forge-game`/`forge-gui`
(`BoosterDraft`, `LimitedPlayer`) on the worker classpath via the sibling
`../forge` checkout.
**Storage**: append-only JSONL corpus (`output/draft/drafts.jsonl`); torch
checkpoints (`models/draft/agent/{timestamp}.pt` + `latest.pt`); reuses the
`.npz` card-embedding cache (`output/cardsfolder/`).
**Testing**: pytest (`tests/unit/draft/`, `tests/integration/`); JUnit 5 for
`DraftWorkerMain` (Forge-dependent tests `@Tag("integration")`).
**Target Platform**: Linux/Windows dev workstation, single GPU (CUDA 12.6) or
CPU fallback.
**Project Type**: CLI + offline ML pipeline (hexagonal package, mirrors `sealed`).
**Performance Goals**: data-gen throughput bounded by Forge AI draft speed
(supervisor recycles/restarts workers); picker labeling ~5 ms/seat vs SA ~5 s
(throughput is why picker is the default builder). Training is a standard
supervised single-pass-per-epoch loop.
**Constraints**: `draft` MUST NOT be imported by `sealed`/`price_predictor`
(one-way dependency); `d_model % n_heads == 0` (fail fast); `.npz` width must
match scorer/picker checkpoints (fail fast); determinism/reproducibility
(`random_seed = 42`, stored standardization stats).
**Scale/Scope**: corpus of up to thousands of drafts × up to `pod_size ×
pack_size` (≈ 8 × 15 = 360) examples each; two CLI subcommands + one script +
one Java worker.

## Constitution Check

*GATE: passed (initial) — re-checked after Phase 1 design (still passing).*

- **I. Fast Automated Tests** — PASS. Fast unit tests for the new pure logic
  (state-reconstruction geometry, recency, loss masking, config validation)
  under `tests/unit/draft/`; slow Forge-dependent worker tests isolated as
  integration. No new flaky/slow tests in the fast suite.
- **II. Simplicity First** — PASS. Reuses existing scorer/picker/builder/
  embedding/checkpoint/supervisor infrastructure; introduces only the
  irreducible new concepts (typed draft state + two-headed model). The
  builder-validation diagnostic is a script, not a speculative subcommand. The
  shared-trainer abstraction is deliberately **not** extracted now (see
  research §Third-instance check).
- **III. Data Integrity** — PASS. JSONL records are self-contained and
  round-trippable (SC-002); `.npz`/checkpoint width mismatches fail fast;
  critic-target standardization mean/std are stored for reproducible inference;
  `random_seed = 42`; readers validate/skip partial lines.
- **IV. DDD & Separation of Concerns** — PASS. `draft/domain` (model, state
  geometry, recency), `draft/application` (supervisor, training, diagnostic),
  `draft/infrastructure` (CLI, stores, worker connector, JSONL IO). Dependencies
  point inward; `draft` depends on `sealed`/`price_predictor`, never the
  reverse.
- **V. MTG Forge Interoperability** — PASS (N/A to the remote API). This feature
  uses the existing Java-worker subprocess bridge (same as `generate-pools` /
  `match-outcomes`), not the price-predictor remote API; no stub-library change.
- **VI. Documentation** — PASS. CLAUDE.md gains the `draft` package + two
  subcommands + the `drafts.jsonl` format; `quickstart.md` + `contracts/`
  document workflows, artifacts, and the ML rationale (cross-links to the
  normative spec + experiments doc). To be updated in the implementing PR.
- **VII. Codebase-Aware Planning** — PASS. See survey below.

### Codebase Survey (Principle VII)

Full findings: [`research.md#codebase-survey`](./research.md#codebase-survey).

- **Overlapping vocabulary**: 4 concepts reused (`SAB`, `ScorerConfig`/scorer-as-
  labeler, `PickerModel` template, deck/seat-pool), 1 new derived value
  (pod-relative reward), 1 new parallel model (`DraftAgentModel`/`Config`) —
  justified (typed tokens + 2nd head diverge too far to extend `PickerModel`,
  but follow its conventions exactly). No silent duplication; no rename needed.
- **Adjacent prior art**: ~11 sub-problems reused (Forge supervisor/restart,
  JVM launch helpers, picker/SA deck build, `score_decks`, card locator,
  embedding layout, checkpoint store, warmup+clip recipe, resume/bootstrap
  guard, JSONL partial-line tolerance); 1 genuinely new (booster→state geometry).
- **Convention alignment**: mirrors `sealed` (folder layout, CLI wiring,
  one-way dependency, store/worker patterns, test style); `DraftWorkerMain`
  joins the existing forge-connector main classes. No unjustified deviation.
- **Third-instance check**: `train-draft-agent` is a 4th trainer sharing the
  resume/warmup/clip/best-checkpoint scaffolding. Decision: follow the pattern,
  do **not** extract yet (Simplicity-First + the existing
  `train_picker.py:7 TODO(shared-trainer)` judgment; the trainers diverge on
  loss/metric/shape). **Follow-up task surfaced** (non-blocking): re-evaluate a
  `train_common` helper extraction after this trainer lands — to be added in
  `tasks.md`.

**Gate result**: no violations; Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/018-draft-agent/
├── plan.md              # This file
├── spec.md              # Feature spec (input)
├── research.md          # Phase 0: codebase survey + technical decisions
├── data-model.md        # Phase 1: entities & rules
├── quickstart.md        # Phase 1: workflows
├── contracts/           # Phase 1: CLI, drafts.jsonl, worker protocol
│   ├── cli.md
│   ├── drafts-jsonl.md
│   └── worker-protocol.md
├── checklists/          # (pre-existing)
└── tasks.md             # Phase 2 (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

New top-level `draft` package mirroring `src/sealed/`, plus one Java worker and
a test tree:

```text
src/draft/
├── __init__.py
├── __main__.py                         # python -m draft <subcommand>
├── domain/
│   ├── draft_agent_model.py            # DraftAgentConfig + DraftAgentModel (SAB trunk + 2 heads)
│   ├── draft_state.py                  # typed-token assembly: CONTEXT/POOL/PACK/PASSED/TAKEN
│   └── draft_geometry.py               # FR-016 booster↔seat/pick geometry + recency walk
├── application/
│   ├── generate_draft_data.py          # supervisor: spawn worker, build+score decks, append JSONL
│   ├── train_draft_agent.py            # joint policy+critic training loop
│   └── validate_builder.py             # FR-042 diagnostic logic (driven by the script)
├── infrastructure/
│   ├── cli.py                          # argparse wiring (mirrors sealed/infrastructure/cli.py)
│   ├── draft_record_io.py              # JSONL read/write, trailing-partial tolerance, --resume count
│   ├── draft_worker_connector.py       # launches DraftWorkerMain (reuses forge_jvm helpers)
│   └── draft_agent_store.py            # checkpoint save/load (mirrors PickerStore)
└── scripts/
    └── validate_builder.py             # ~40-line entry calling application/validate_builder

forge-connector/src/main/java/com/pricepredictor/connector/
└── DraftWorkerMain.java                # Forge BoosterDraft over 8 seats → stdout sentinel transcript

tests/unit/draft/                       # geometry, state, recency, model, loss, config, IO
tests/integration/                      # worker/pipeline smoke (Forge-dependent)
forge-connector/src/test/java/...       # DraftWorkerMain tests (@Tag("integration") for Forge)
```

**Structure Decision**: single-project hexagonal package `src/draft/`, a direct
structural sibling of `src/sealed/` (same `domain`/`application`/`infrastructure`
split, same CLI-wiring and store/worker-connector idioms, same one-way
dependency rule). Reuse is via imports from `sealed` and `price_predictor`; no
new shared abstraction is extracted (see survey). The Java worker is added to
the existing `forge-connector` module rather than a new module.

## Complexity Tracking

> No constitution violations — table intentionally empty.
