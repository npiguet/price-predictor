# Implementation Plan: Draft Agent — Online Self-Play GRPO Trainer (Generation 3)

**Branch**: `021-draft-online-grpo` | **Date**: 2026-08-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/021-draft-online-grpo/spec.md`

## Summary

Add one new application use case — an **online, critic-free GRPO trainer**
(`python -m draft train-draft-agent-online`) — that owns a streaming
generate → update → discard → regenerate loop in a single process. Each round it
drives **one resident Forge draft worker** for `--drafts-per-round` fresh drafts
whose learner seats are piloted by the live in-training policy, builds + scores
every seat's deck to get the reward, takes **one minibatch pass** of the single
term `−A·logπ_T(a|s)` over the learner picks, discards the batch, and drafts the
next round with the updated weights. There is no critic term, GAE, KL anchor,
entropy bonus, val split, or early stop.

The build is: the round loop + the update + four-axis per-round stdout
diagnostics (reward signal, exploration, movement, live anchor margin), plus five
small named extensions to existing components —
`GenerateDraftDataSupervisor.iter_records()` (the resident-worker record stream),
`AgentPickService.from_model()` (learner seats served by the live model),
`AgentRegistry.build(preloaded=…)`, `build_labeler(…, locator=…)` (so one
memoizing card locator serves the whole run), and a `-Ddraft.required.agent`
property in the Java worker — plumbed end-to-end through a
`GenerateDraftDataConfig.required_agent` field and the existing launcher — that
guarantees ≥1 learner seat per pod. Model, typed-token state, corpus schema,
checkpoint format, deck labeler, scorer, and the cross-generation yardstick are
reused unchanged.

## Technical Context

**Language/Version**: Python 3.14+ (`pyproject.toml`); Java 17+ for the
`forge-connector` worker change
**Primary Dependencies**: `torch` (CUDA 12.6 wheels), `numpy`; internal `draft`,
`sealed`, `price_predictor` packages (`DraftAgentModel`/`DraftAgentStore`,
`iter_seat_pick_states`, `GenerateDraftDataSupervisor` + deck labelers,
`AgentPickService`/`AgentRegistry`, `ConvertedCardLocator`, `draft_record_io`)
**Storage**: `.pt` checkpoints under `models/draft/agent/` (`latest.pt` + periodic
`{timestamp}.pt`); appends draft records to the shared
`output/draft/drafts.jsonl`; reads the `.npz` embedding cache. No new on-disk
schema — the corpus is unchanged and `rl_metadata` is an existing optional
checkpoint field
**Testing**: `pytest` — fast unit tests under `tests/unit/draft/`, one integration
smoke test under `tests/integration/` driven by a fake worker (no JVM); `ruff
check`; JUnit for the `forge-connector` change
**Target Platform**: local CUDA GPU with CPU fallback (Windows dev box) + a built
sibling Forge checkout at `../forge`
**Project Type**: single project, hexagonal (`domain` → `application` →
`infrastructure`); CLI / ML pipeline
**Performance Goals**: rounds of ~10 drafts alternating generation and training in
one process; Forge JVM startup (~20 s) paid **once per run**, not per round; the
round's ~1.8 k learner picks trained in length-bucketed minibatches and
diagnosed in one batched `no_grad` pass
**Constraints**: on-policy by construction (learner seats piloted by the live
model; batch used once, then discarded); batch size 32 within the **8 GB VRAM**
budget; every pod has ≥1 learner seat; frozen anchor never changes during a run;
frozen encoder (Phase A); architecture inherited from the `--learner` checkpoint
**Scale/Scope**: hours-long runs of hundreds of rounds × 10 drafts × 8 seats × 45
picks; only learner-label seats feed the gradient

No unresolved NEEDS CLARIFICATION. The two spec clarifications (corpus →
shared `drafts.jsonl`; checkpoints → shared `models/draft/agent/`) plus research
D1–D15 resolve the planning unknowns; the reproducibility question deferred
during `/speckit.clarify` is settled by D12.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Fast Automated Tests** — Pure/near-pure pieces each get a fast unit test:
  round-standardised advantage + degenerate-round guard, the single-term loss,
  the round loader (learner-only picks, failed-build drop), the diagnostics
  accumulators (entropy/perplexity/off-argmax/KL), the anchor-margin window,
  startup validation + exit codes, `rl_metadata` contents, and the
  `iter_records()` refactor (fake worker, no JVM). One integration smoke test
  runs two full rounds against a fake worker. Java: a `DraftWorkerMainTest` case
  for the forced-learner-seat rule. PASS (planned).
- **II. Simplicity First** — One new use case plus four additive extensions to
  existing components; no new abstraction invented for gen-3. The dropped gen-2
  machinery (critic/GAE/KL/entropy/val-split/plateau) is simply absent, and no
  `--resume` is added (D13). PASS.
- **III. Data Integrity** — Corpus schema unchanged and append-only via the
  existing partial-line-tolerant IO; failed builds excluded deterministically;
  startup validation of every external input (checkpoints, mix labels, cache
  width, temperature) before any update; `--seed` fixes everything Python-side,
  with the unseeded Forge-side randomness stated explicitly rather than implied
  (D12). PASS.
- **IV. DDD & Separation** — Reward/advantage/diagnostic math is pure and
  torch-only where it must be; the loop lives in `application`; worker launch,
  checkpoint IO, and the CLI stay in `infrastructure`. Dependency direction
  unchanged (`draft` → `sealed` → `price_predictor`); the extracted torch helpers
  land in `price_predictor` precisely so `sealed` may import them. PASS.
- **V. Forge Interop** — No API/stub surface change. The `forge-connector` worker
  gains one optional system property (`-Ddraft.required.agent`); absent, its
  behaviour is byte-for-byte today's. PASS.
- **VI. Documentation** — quickstart.md is the operator runbook; the new CLI
  command, its diagnostics, its artifacts, and the required JAR rebuild are added
  to README / CLAUDE.md in the same change. PASS (planned).
- **VII. Codebase-Aware Planning** — Codebase survey complete in
  [research.md#codebase-survey](research.md#codebase-survey).
  - Overlapping vocabulary: 14 concepts reused or extended, **0 parallel concepts
    introduced**, 0 renames needed.
  - Adjacent prior art: 9 utilities reused (supervisor/worker driving, pick
    protocol + fault handling, deck labelers, embedding locator, state walk,
    batched policy forward, checkpoint store, width fail-fast, CLI convention),
    **0 reimplemented**.
  - Convention alignment: mirrors the `train-draft-agent-rl` sibling (spec 020)
    for module layout, CLI wiring, test placement, and checkpoint location; the
    one deliberate deviation is no `--resume`/`cli_resume` (justified in D13).
  - Third-instance check: **4 helper groups extracted, not copied a third time** —
    `masked_log_softmax`/`policy_entropy`/`kl_divergence`/`clip_per_group` →
    new `price_predictor/infrastructure/torch_training.py`;
    `leave_one_out_rewards`/`length_bucketed_batches` → new
    `draft/application/draft_training_common.py`. Whole-loop extraction stays
    deferred (standing decision).
  - **Follow-up tasks the survey surfaced** (must appear in `tasks.md`):
    (a) the two extraction tasks above, including repointing the existing call
    sites in `train_picker`, `train_scorer`, `train_draft_agent`,
    `train_draft_agent_rl` and the four unit-test files that import the private
    names; (b) the `iter_records()` refactor of `GenerateDraftDataSupervisor`
    with its existing tests kept green; (c) the `forge-connector` JAR rebuild
    note in the docs.
- **VIII. Performance-Conscious Implementation** — Performance Review below.

### Performance Review (Principle VIII)

- **I/O batching & caching** — *addressed*: one memoizing `ConvertedCardLocator`
  shared by the deck labeler, all pick services, and the trainer, so each card's
  `.npz` is decompressed once per run — which is why `build_labeler` gains an
  optional `locator` parameter instead of constructing its own; per-round
  embedding table built from cache hits; corpus handle opened once in append mode
  and appended per record (research D14).
- **GPU placement** — *addressed*: learner model, the `prev_model` diagnostics
  copy, frozen pick-service models, scorer and (optional) picker all move to CUDA
  when available; batches are collated onto the device.
- **GPU batching** — *addressed*: the round's pass uses length-bucketed
  minibatches (size 32, the 8 GB VRAM budget); all exploration/movement
  diagnostics come from **one** batched `no_grad` sweep rather than per-pick math
  (research D9); the deck labeler already batches a whole pod through one
  builder + one scorer forward. The only per-item host↔device readout is
  `AgentPickService._select`'s single per-pick sync, inherent to the strictly
  synchronous pick protocol and unchanged from live-play; scalars are read once
  per round, not per step.
- **Streaming & load-once** — *addressed*: drafts are consumed as a suspended
  generator stream and never materialised as a corpus; models, locator, labeler,
  and the Forge worker are constructed once per run (the whole point of FR-005).
- No optimization beyond this checklist is proposed (Principle II). The obvious
  candidate — parallel Forge workers — is explicitly unnecessary (drafts play no
  games).

**Result**: Constitution Check PASS (pre- and post-design); Complexity Tracking
empty.

## Project Structure

### Documentation (this feature)

```text
specs/021-draft-online-grpo/
├── plan.md              # This file
├── research.md          # Phase 0 — codebase survey + decisions D1–D15
├── data-model.md        # Phase 1 — config, round batch, example, diagnostics, metadata
├── quickstart.md        # Phase 1 — operator runbook (setup → run → read logs → yardstick)
├── contracts/
│   ├── train-draft-agent-online.md   # Phase 1 — CLI + stdout contract
│   └── rollout-stream.md             # Phase 1 — resident-worker record stream + required-agent property
├── checklists/
│   └── requirements.md  # from /speckit.specify + /speckit.clarify
└── tasks.md             # Phase 2 — /speckit.tasks (NOT created here)
```

### Source Code (repository root)

```text
src/draft/
├── application/
│   ├── train_draft_agent_online.py   # NEW — config, round loader, GRPO update, loop, diagnostics
│   ├── draft_training_common.py      # NEW — extracted leave_one_out_rewards + length_bucketed_batches
│   ├── generate_draft_data.py        # EXTENDED — iter_records() generator (run() consumes it);
│   │                                 #   build_labeler(locator=…); config.required_agent forwarded
│   ├── agent_pick_service.py         # EXTENDED — AgentPickService.from_model(...)
│   ├── agent_registry.py             # EXTENDED — AgentRegistry.build(..., preloaded=...)
│   ├── draft_pick_states.py          # REUSED unchanged (per-pick typed-token walk)
│   ├── train_draft_agent.py          # EXTENDED — imports extracted helpers (no behaviour change)
│   └── train_draft_agent_rl.py       # EXTENDED — imports extracted helpers (no behaviour change)
├── domain/                           # REUSED unchanged (model, state, geometry, online tracker)
└── infrastructure/
    ├── draft_worker_connector.py     # EXTENDED — forward -Ddraft.required.agent
    ├── draft_agent_store.py          # REUSED unchanged (rl_metadata already free-form)
    └── cli.py                        # EXTENDED — train-draft-agent-online subparser + run_*

src/price_predictor/infrastructure/
└── torch_training.py                 # NEW — masked_log_softmax / policy_entropy / kl_divergence / clip_per_group

src/sealed/application/
├── train_picker.py                   # EXTENDED — imports extracted torch helpers
└── train_scorer.py                   # EXTENDED — imports extracted clip_per_group

forge-connector/src/main/java/com/pricepredictor/connector/
└── DraftWorkerMain.java              # EXTENDED — -Ddraft.required.agent forces one seat

tests/
├── unit/draft/
│   ├── test_online_advantage.py      # NEW — round standardisation + degenerate-round no-op
│   ├── test_online_loss.py           # NEW — single-term loss, PACK masking, temperature
│   ├── test_online_loader.py         # NEW — learner-only picks, failed-build drop, per-seat advantage
│   ├── test_online_diagnostics.py    # NEW — entropy/perplexity/off-argmax/KL + anchor-margin window
│   ├── test_online_cli.py            # NEW — startup validation, label wiring, exit codes
│   ├── test_record_stream.py         # NEW — iter_records(): yield/suspend/restart, fault routing
│   ├── test_agent_pick_service.py    # EXTENDED — from_model() shares live weights
│   ├── test_agent_registry.py        # EXTENDED — preloaded labels validate + geometry-check
│   ├── test_draft_loss.py            # EXTENDED — import repointed to draft_training_common
│   └── test_length_bucketing.py      # EXTENDED — import repointed to draft_training_common
├── unit/sealed/application/
│   ├── test_train_picker.py          # EXTENDED — imports repointed to torch_training
│   └── test_train_scorer.py          # EXTENDED — import repointed to torch_training
└── integration/
    └── test_train_draft_agent_online_smoke.py  # NEW — two rounds end-to-end against a fake worker

forge-connector/src/test/java/com/pricepredictor/connector/
└── DraftWorkerMainTest.java          # EXTENDED — forced learner seat when the mix draws none
```

**Structure Decision**: Single project, hexagonal. The feature is one new
`application` use case plus additive extensions to three existing `application`
modules, two `infrastructure` modules, and one Java worker class — mirroring the
`train-draft-agent-rl` sibling and respecting the
`draft` → `sealed` → `price_predictor` dependency direction. The two new shared
modules exist only to satisfy the Principle VII third-instance rule; they contain
extracted code, not new logic.

## Complexity Tracking

No Constitution Check violations — table intentionally empty.
