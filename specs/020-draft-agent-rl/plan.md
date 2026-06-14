# Implementation Plan: Draft Agent — RL Self-Play Fine-Tuning (Generation 2)

**Branch**: `020-draft-agent-rl` | **Date**: 2026-06-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/020-draft-agent-rl/spec.md`

## Summary

Add one new application use case — an **on-policy actor-critic RL trainer**
(`python -m draft train-draft-agent-rl`) — that fine-tunes the gen-1 two-headed
draft agent on a self-play corpus it generated, pushing the policy past the
imitation ceiling. The trainer warm-starts actor+critic from a reference
checkpoint, recomputes per-pick critic values and **GAE(λ) advantages** over each
learner seat's 45-pick trajectory from the recorded corpus, and applies a
KL-anchored policy-gradient + critic-MSE + entropy update (REINFORCE+GAE, not
PPO). It writes the next-generation checkpoint with self-describing RL metadata.

Everything else is reuse: the model, typed-token state, corpus schema, frozen
encoder/`.npz` cache, and the deck-score reward are unchanged. The
cross-generation **yardstick** (US2) and the **self-play loop** (US3) introduce
no code — they are operator runbooks over the existing `generate-draft-data`
(greedy vs sample) and `analyze-generated-decks`, with promotion as a manual
judgment (quickstart.md). The build is the trainer + its multi-corpus input + a
backward-compatible checkpoint RL-metadata extension.

## Technical Context

**Language/Version**: Python 3.14+ (`pyproject.toml`)
**Primary Dependencies**: `torch` (CUDA 12.6 wheels), `numpy`; internal `draft`,
`sealed`, `price_predictor` packages (reuses `DraftAgentModel`, `DraftAgentStore`,
`draft_state`/`draft_geometry`, `ConvertedCardLocator`, `cli_resume`, and the
REINFORCE helper patterns from `train_picker`)
**Storage**: `.pt` checkpoints under `models/draft/agent/`; reads `drafts.jsonl`
corpora and the `.npz` embedding cache. No new on-disk schema (corpus unchanged;
checkpoint gains optional `rl_metadata`)
**Testing**: `pytest` — fast unit tests under `tests/unit/draft/`, an integration
smoke test under `tests/integration/`; `ruff check`
**Target Platform**: local CUDA GPU with CPU fallback (Windows dev box)
**Project Type**: single project, hexagonal (`domain` → `application` →
`infrastructure`); CLI / ML pipeline
**Performance Goals**: offline batched training over a fixed corpus; GPU
forward/backward; shared embedding table (each card loaded once), length-bucketed
batching, per-epoch batched `no_grad` advantage precompute — no per-pick host↔device
sync in the loop
**Constraints**: on-policy gradient correctness (policy term only from the
`--checkpoint`-generated corpus; all distributions at the rollout temperature);
warm-started critic stays in gen-1's standardized reward space; frozen encoder
(Phase A); architecture inherited from the reference checkpoint
**Scale/Scope**: corpora of thousands of drafts × pod_size seats × 45 picks;
learner seats feed the policy gradient, all non-failed seats feed the critic

No unresolved NEEDS CLARIFICATION — the three spec clarifications (provenance =
operator convention; best-checkpoint = held-out RL objective; gating =
warn-and-continue) plus research D1–D9 resolve the planning unknowns.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Fast Automated Tests** — Pure helpers (GAE/advantage, loss decomposition,
  trajectory grouping, behaviour-anomaly summary, RL-metadata round-trip,
  startup validation, resume precedence) get fast unit tests under
  `tests/unit/draft/`; the end-to-end train-on-tiny-corpus path is one
  integration smoke test. PASS (planned).
- **II. Simplicity First** — One new use case mirroring `train_draft_agent`; no
  new abstraction beyond reused/extracted helpers. The shared-trainer extraction
  stays deferred (only two tiny atoms proposed for extraction; see survey). PASS.
- **III. Data Integrity** — Corpus read via the existing partial-line-tolerant
  reader; failed builds excluded deterministically; `random_seed=42`; the
  checkpoint extension is backward-compatible (gen-1 checkpoints load unchanged).
  PASS.
- **IV. DDD & Separation** — Reward/GAE/loss math is pure (`domain`/pure helpers,
  no torch in geometry/state); training orchestration in `application`;
  checkpoint IO + CLI in `infrastructure`. Dependency direction unchanged (`draft`
  → `sealed`/`price_predictor`). PASS.
- **V. Forge Interop** — N/A: no API/stub surface; rollouts reuse the existing
  Forge-driven `generate-draft-data` unchanged.
- **VI. Documentation** — quickstart.md is the operator runbook (US3); the new
  CLI command, its artifacts (RL-metadata checkpoint), and the RL process will be
  added to the README / CLAUDE.md in the same change. PASS (planned).
- **VII. Codebase-Aware Planning** — Codebase survey complete in
  [research.md#codebase-survey](research.md#codebase-survey).
  - Overlapping vocabulary: 6 concepts reused/extended, 0 parallel concepts
    introduced.
  - Adjacent prior art: 9 utilities reused (model, store, state/geometry,
    REINFORCE helpers, pick-service forward, `cli_resume`, record IO, `.npz`
    locator, analyze/deck-score), 0 reimplemented.
  - Convention alignment: mirrors the `train-draft-agent` sibling exactly.
  - Third-instance check: shared-trainer loop extraction **deferred** (documented
    standing decision); a non-blocking follow-up task proposes extracting only
    the two byte-identical atoms (warmup-LR lambda, per-group clip) now shared by
    5 trainers. Follow-up to add in tasks.md.
- **VIII. Performance-Conscious Implementation** — Performance Review below.

### Performance Review (Principle VIII)

- **I/O batching & caching** — addressed: `.npz` loaded once into a shared
  embedding table (gen-1 `_Loader` pattern); corpora streamed once via
  `read_records`.
- **GPU placement** — addressed: actor, frozen `π_ref`, and critic on CUDA when
  available; batches collated onto the device.
- **GPU batching** — addressed: length-bucketed training batches; per-epoch
  advantage precompute is a batched `no_grad` critic forward over trajectories;
  behaviour-log-prob recompute batched; no per-pick `.item()`/`.cpu()` in the
  loop (advantages cached as tensors; scalar reads only at eval boundaries).
- **Streaming & load-once** — addressed: corpus streamed; model/locator/table
  built once, reused across epochs.
- No optimization beyond the checklist (Principle II); further tuning would need
  a profile.

**Result**: Constitution Check PASS (pre- and post-design); no entries in
Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/020-draft-agent-rl/
├── plan.md              # This file
├── research.md          # Phase 0 — codebase survey + decisions D1–D9
├── data-model.md        # Phase 1 — config, trajectory, RLExample, loss, RL metadata
├── quickstart.md        # Phase 1 — operator runbook (US3 documentation)
├── contracts/
│   └── train-draft-agent-rl.md   # Phase 1 — CLI command contract
├── checklists/
│   └── requirements.md  # from /speckit.specify + /speckit.clarify
└── tasks.md             # Phase 2 — /speckit.tasks (NOT created here)
```

### Source Code (repository root)

```text
src/draft/
├── application/
│   ├── train_draft_agent.py        # gen-1 sibling (reused patterns; source of extracted state-walk)
│   └── train_draft_agent_rl.py     # NEW — RL use case (config, loader, GAE, loss, loop)
├── domain/
│   ├── draft_agent_model.py        # REUSED unchanged (actor + critic)
│   ├── draft_state.py, draft_geometry.py   # REUSED unchanged (state reconstruction)
│   └── (rl advantage/GAE pure helpers live with the use case or a small pure module)
└── infrastructure/
    ├── draft_agent_store.py        # EXTENDED — optional rl_metadata (backward compatible)
    └── cli.py                      # EXTENDED — train-draft-agent-rl subparser + run_* dispatch

tests/
├── unit/draft/
│   ├── test_rl_advantage.py        # NEW — GAE(λ)/return over a trajectory, γ=1, terminal reward
│   ├── test_rl_loss.py             # NEW — policy/value/entropy/kl decomposition + masking + NaN guards
│   ├── test_rl_loader.py           # NEW — trajectory grouping/order, learner/critic activation, failed-build drop
│   ├── test_rl_behaviour_anomaly.py# NEW — behaviour-logprob summary + warning threshold
│   ├── test_rl_cli.py              # NEW — startup validation / exit codes / resume precedence
│   └── test_draft_agent_store.py   # EXTENDED — rl_metadata round-trip + gen-1 back-compat load
└── integration/
    └── test_train_draft_agent_rl_smoke.py   # NEW — train one tiny on-policy corpus end to end
```

**Structure Decision**: Single project, hexagonal. The feature is one new
`application` use case (`train_draft_agent_rl.py`) plus minimal extensions to two
existing `infrastructure` files (`draft_agent_store.py`, `cli.py`), reusing the
unchanged `domain` model and state reconstruction. This mirrors the
`train-draft-agent` sibling and respects the `draft → sealed → price_predictor`
dependency direction.

## Complexity Tracking

No Constitution Check violations — table intentionally empty.
