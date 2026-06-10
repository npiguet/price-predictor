# Implementation Plan: Draft agent — live Forge integration

**Branch**: `019-draft-live-play` | **Date**: 2026-06-10 | **Spec**: [`spec.md`](./spec.md)
**Input**: Feature specification from `specs/019-draft-live-play/spec.md`; normative
design note `specs/2026-06-04-draft-agent-live-play.md`

## Summary

Let a trained draft agent **pilot a live seat** in a real Forge pod. The existing
`generate-draft-data` command is *extended* (no new subcommand): operators bind
mix labels to agent checkpoints; whenever the Java worker reaches a pick for a
model-piloted seat it asks the Python supervisor for the card over a new
request/response side-channel, and the supervisor answers with the trained
policy's genuine choice. Every finished draft is labeled and appended to the
unchanged `drafts.jsonl`, yielding a self-play corpus and an in-pod
agent-vs-Forge strength measurement at once.

The only genuinely new logic is (1) the **pick protocol** layered onto the
draft worker (`<<DRAFT-PICK-REQUEST>>` / `<<DRAFT-PICK-RESPONSE>>` on the
worker's stdout/stdin), (2) an **online incremental draft-state tracker** that
reconstructs, from the stream of packs a seat is shown, the *identical*
typed-token state the offline trainer builds for the same `(seat, pack, pick)`,
and (3) the **policy inference glue** that embeds that state and selects a card.
Everything else — deck labeling, scoring, run-id stamping, resume counting,
progress logging, JSONL append, crash-restart — is reused unchanged from gen-1.

## Technical Context

**Language/Version**: Python 3.14+ (project requirement); Java 17+ for the
`forge-connector` worker.
**Primary Dependencies**: `torch` (policy forward), `numpy`, the in-repo `draft`,
`sealed`, and `price_predictor` packages; Forge `forge-game`/`forge-gui`
(`BoosterDraft`, `LimitedPlayerAI`) on the worker classpath via the sibling
`../forge` checkout.
**Storage**: append-only JSONL corpus (`output/draft/drafts.jsonl`, **unchanged
schema**); reads existing agent checkpoints (`models/draft/agent/*.pt`) and the
`.npz` card-embedding cache (`output/cardsfolder/`). Writes **no** new model
artifacts (FR-015).
**Testing**: pytest (`tests/unit/draft/`, `tests/integration/`); JUnit 5 for the
worker pick-protocol (Forge-dependent tests `@Tag("integration")`).
**Target Platform**: Linux/Windows dev workstation, single GPU (CUDA 12.6) or
CPU fallback.
**Project Type**: CLI + offline ML pipeline (hexagonal `draft` package, mirrors
`sealed`).
**Performance Goals**: per-pick policy forward is a single tiny set-transformer
pass (≤ pack-size + pool tokens), inherently un-batchable under the strict
single-pick synchrony invariant; data-gen throughput stays bounded by Forge AI
draft speed. Model loaded once and kept on-device; card embeddings memoized by
the locator.
**Constraints**: `draft` MUST NOT be imported by `sealed`/`price_predictor`
(one-way dependency); the online-reconstructed state MUST equal the offline
`build_state` output at every `(seat, pack, pick)` (gating equivalence test,
SC-003); strict single-pick synchrony (≤ 1 outstanding request); a model-seat
pick fault abandons the whole draft with **no** substitute (SC-002); unknown
label / geometry mismatch fail fast (FR-011/FR-012); sampled picks reproducible
under `--seed` (SC-007); K-consecutive fault auto-abort with nonzero exit
(FR-016); **zero** behavior/format change with no model labels (SC-004).
**Scale/Scope**: one extended CLI subcommand, one new domain tracker, one new
inference service, the Java worker pick-protocol extension; corpus records and
record format identical to gen-1.

## Constitution Check

*GATE: passed (initial) — re-checked after Phase 1 design (still passing).*

- **I. Fast Automated Tests** — PASS. Fast unit tests for all new pure logic
  (online state tracker ↔ `build_state` equivalence, pick-mode determinism,
  un-embeddable dropping, `LABEL=PATH` parsing, label/geometry validation,
  supervisor pick-routing + abort + K-consecutive auto-abort with fake workers).
  Live-JVM end-to-end smoke is isolated as a Forge-dependent integration test.
- **II. Simplicity First** — PASS. Extends the existing command and worker
  rather than adding a subcommand or a second worker; reuses all gen-1 labeling/
  scoring/IO/restart machinery. The only new abstractions are the three
  irreducible pieces (pick protocol, online tracker, inference glue). No RL, no
  self-play loop, no critic-in-the-loop (all explicitly out of scope).
- **III. Data Integrity** — PASS. Corpus schema unchanged (FR-009); a model
  seat's recorded picks are *only* the policy's genuine choices — any fault
  abandons the draft, never substitutes (SC-002), so the corpus is clean by
  construction. Pick-response routing fields (`draft_id`/`seat`/`pack_number`/
  `pick_number`) are validated worker-side; sampled picks are seeded for
  reproducibility (SC-007); checkpoint geometry is validated against the live
  draft (FR-012).
- **IV. DDD & Separation of Concerns** — PASS. New online tracker is pure
  `draft/domain` logic (no torch, no IO); inference glue is `draft/application`;
  CLI/connector changes stay in `draft/infrastructure`; the Java protocol lives
  in the worker. Dependencies point inward; `draft` still imports `sealed`/
  `price_predictor`, never the reverse.
- **V. MTG Forge Interoperability** — PASS (N/A to the remote API). This uses the
  existing Java-worker subprocess bridge (same family as `match-outcomes` /
  `generate-pools`), extended with a stdin response channel; it is **not** the
  price-predictor remote stub library, so no stub/API-contract change.
- **VI. Documentation** — PASS. CLAUDE.md's `generate-draft-data` paragraph gains
  the model-pilot flags and the pick side-channel; `contracts/` document the
  extended CLI and the pick protocol; `quickstart.md` covers the self-play and
  strength-measurement workflows. Updated in the implementing PR.
- **VII. Codebase-Aware Planning** — PASS. See survey pointer below.
- **VIII. Performance-Conscious Implementation** — PASS. See performance review
  below.

### Codebase Survey (Principle VII)

Full findings: [`research.md#codebase-survey`](./research.md#codebase-survey).

- **Overlapping vocabulary**: 6 concepts reused (`DraftAgentModel`/`DraftAgentStore`,
  `DraftState`/`CardInstance`/`_recency`, `DraftGeometry`, `agent_mix`,
  `ConvertedCardLocator`, the gen-1 supervisor scaffolding); 1 new domain concept
  (`OnlineDraftStateTracker`) and 1 new application service (`AgentPickService`)
  — both justified (live request-stream input has no prior art; offline
  `build_state` and the loader walk both require a *finished* record). 0 renames.
- **Adjacent prior art**: ~8 sub-problems reused (deck labeler + scorer, JSONL
  append + resume count, run-id + crash-restart loop, JVM launch helpers, the
  `<<DRAFT-EVENT-JSON>>` transcript path, `--agent-mix` parse/format, checkpoint
  load, `.npz` memoized embedding load); 2 genuinely new (the pick
  request/response protocol, online state reconstruction).
- **Convention alignment**: mirrors `sealed`/`draft` idioms (CLI subparser +
  lazy imports, store/connector patterns, pure-domain tracker, test style);
  the worker extension stays inside the existing `DraftWorkerMain`.
- **Third-instance check**: `OnlineDraftStateTracker` is a **third** typed-token
  reconstruction alongside `draft_state.build_state` (full-record oracle) and
  `train_draft_agent._Loader._emit_seat` (full-record incremental). Decision:
  add the third as a thin sibling that **reuses** `CardInstance`/`DraftState`/
  `_recency` and is **locked to `build_state` by a gating equivalence test**;
  do **not** force-extract a shared core now because the *input contract differs*
  (partial request stream vs. complete `DraftRecord`). **Follow-up task
  surfaced** (non-blocking): after this lands, evaluate factoring the shared
  wheel-diff/flush/recency core out of the three call sites — to be added to
  `tasks.md`.

**Gate result**: no violations; Complexity Tracking left empty.

### Performance Review (Principle VIII)

The feature runs model compute (per-pick policy forward) and reads embeddings.

- **I/O batching & caching** — addressed. Card embeddings are loaded through the
  existing `ConvertedCardLocator.load_embedding` per-name memo cache (one `.npz`
  open per distinct card across the whole run); checkpoints load once at startup.
- **GPU placement** — addressed. The policy model is moved to CUDA-if-available
  once at startup and the single-example input tensors are built on that same
  device; co-location mirrors the gen-1 labelers.
- **GPU batching** — N/A-with-reason. Strict single-pick synchrony (≤ 1
  outstanding request, design §4) makes per-pick forwards inherently sequential;
  there is no batch to form. Each forward is a handful of tokens, so the
  unbatched cost is negligible and bounded by Forge AI pick latency, not by GPU
  throughput. The one unavoidable host↔device sync (reading the chosen logit)
  is once per pick, outside any tight loop.
- **Streaming & load-once** — addressed. The worker stdout is consumed
  line-by-line (already streaming); the model, store, locator, and agent
  registry are constructed once and reused for every pick of every draft.

No optimization beyond the checklist is proposed (no profile motivates it;
Principle II).

## Project Structure

### Documentation (this feature)

```text
specs/019-draft-live-play/
├── plan.md              # This file
├── spec.md              # Feature spec (input)
├── research.md          # Phase 0: codebase survey + technical decisions
├── data-model.md        # Phase 1: entities & rules
├── quickstart.md        # Phase 1: workflows
├── contracts/           # Phase 1: extended CLI + pick protocol
│   ├── cli.md
│   └── pick-protocol.md
├── checklists/          # (pre-existing)
└── tasks.md             # Phase 2 (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

Extends the existing `src/draft/` package and the existing `DraftWorkerMain`;
no new package or module is introduced for the worker.

```text
src/draft/
├── domain/
│   ├── online_draft_state.py          # NEW: OnlineDraftStateTracker (request stream → DraftState)
│   ├── draft_state.py                 # reused: CardInstance/DraftState/_recency (may export the core)
│   └── draft_geometry.py              # reused: pod/pack/pick conventions
├── application/
│   ├── agent_pick_service.py          # NEW: load agent ckpt, embed DraftState, argmax/sample pick
│   ├── agent_registry.py              # NEW: parse LABEL=PATH, validate labels/geometry, build services
│   └── generate_draft_data.py         # EXTENDED: pick-request routing, abort, K-consecutive auto-abort
├── infrastructure/
│   ├── cli.py                         # EXTENDED: --agent-checkpoint/--pick-mode/--temperature/--seed/...
│   ├── draft_worker_connector.py      # EXTENDED: pipe stdin, forward external-agent set + stderr log
│   └── draft_agent_store.py           # reused: checkpoint load (config carries packs/P/embedding_dim)

forge-connector/src/main/java/com/pricepredictor/connector/
└── DraftWorkerMain.java               # EXTENDED: external-seat pick request/response + draft abandon

tests/unit/draft/                      # online-tracker equivalence, pick-mode, registry, routing, auto-abort
tests/integration/                     # live-JVM model-seat smoke draft (Forge-dependent)
forge-connector/src/test/java/...      # pick-protocol formatting/validation (non-Forge) + @Tag("integration")
```

**Structure Decision**: continue the single-project hexagonal `src/draft/`
package established by gen-1 (018). The two new files are a pure-domain tracker
(`online_draft_state.py`) and an application service (`agent_pick_service.py`),
matching the existing domain/application/infrastructure split and the one-way
dependency on `sealed`/`price_predictor`. The worker change is an extension of
the existing `DraftWorkerMain`, not a new main class. No shared abstraction is
extracted now (see survey third-instance decision); the follow-up to revisit a
shared reconstruction core is recorded for `tasks.md`.

## Complexity Tracking

> No constitution violations — table intentionally empty.
