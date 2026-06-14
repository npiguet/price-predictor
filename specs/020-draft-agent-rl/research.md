# Research: Draft Agent — RL Self-Play Fine-Tuning (Generation 2)

**Feature**: 020-draft-agent-rl | **Date**: 2026-06-13
**Inputs**: [spec.md](spec.md); normative source [`specs/2026-06-10-draft-agent-gen2-rl.md`](../2026-06-10-draft-agent-gen2-rl.md); rationale [`experiments/2026-06-10-draft-agent-gen2-design.md`](../../experiments/2026-06-10-draft-agent-gen2-design.md).

This feature adds **one new application use case** — an on-policy actor-critic
RL trainer (`train-draft-agent-rl`) — over the existing gen-1 draft model,
checkpoint format, corpus, state reconstruction, and frozen scorer. The
yardstick (US2) and the self-play loop (US3) are operator runbooks over existing
commands and add no code (spec Assumptions; quickstart.md). The build is the
trainer plus its multi-corpus input.

## Codebase Survey

### Overlapping domain vocabulary

| Existing concept | File / symbol | Decision |
|---|---|---|
| Two-headed agent (policy + critic) | `draft/domain/draft_agent_model.py` — `DraftAgentModel`, `DraftAgentConfig` | **Reuse unchanged.** The same model is the actor (policy head) + critic head. RL warm-starts both from a gen-1/champion checkpoint. No architecture change (spec Out of Scope). |
| Typed-token per-pick state | `draft/domain/draft_state.py` (`TYPE_POOL/PACK/PASSED/TAKEN`), `draft/domain/draft_geometry.py` (`DraftGeometry`, `DraftRecord/Seat/Booster`) | **Reuse unchanged.** Every RL training state is the same `(seat, pack, pick)` typed-token state the gen-1 loader and `agent_pick_service` build. |
| Per-pick example + corpus loader | `draft/application/train_draft_agent.py` — `DraftExample`, `_Loader`, `_emit_seat/_emit_example` | **Extend (new sibling loader).** The gen-1 loader already emits exactly the per-pick states + the leave-one-out reward. The RL loader reuses its state-walk logic but emits RL fields (per-seat trajectory grouping + ordering, `learner_active`, `critic_active`, terminal reward) instead of `(imitation_active, target_token)`. Shared state-walk pieces (`_emit_seat` body, `_leave_one_out_rewards`) are extracted to a small shared module rather than copied (third-instance check below). |
| Leave-one-out pod-relative reward | `train_draft_agent._leave_one_out_rewards` | **Reuse (extract).** Identical formula and failed-build handling the RL reward needs. |
| Draft-disjoint split / length-bucketed batches / critic standardization | `train_draft_agent.split_draft_disjoint`, `length_bucketed_batches`, `critic_standardization` | **Reuse (extract or copy).** Same split semantics (`random_seed=42`), same batching, and the RL critic reuses the gen-1 standardized reward space. |
| Agent / mix label | `draft/application/agent_mix.py`, `agent_registry.py`; corpus `Seat.agent` | **Reuse.** `--learner-agents` is a whitelist of mix labels, exactly like gen-1's `--imitation-agents`. |

No parallel concept is introduced; no rename needed.

### Adjacent prior art

| Sub-problem | Existing solution | Decision |
|---|---|---|
| REINFORCE policy-gradient + entropy + KL + per-pool baseline | `sealed/application/train_picker.py` — `_policy_entropy`, `_kl_penalty`, `_masked_log_softmax`, `_EntropySchedule`, `_compute_losses`, `_clip_per_group` | **Reuse the patterns (copy the small helpers).** The masked-softmax entropy/KL with the `0·-inf` NaN-gradient guards are exactly what the per-pick policy needs; `_EntropySchedule` is the val-driven coefficient decay for both the entropy and KL schedules. Copying these small, stable helpers is the established convention (train_picker docstring). |
| The policy forward → PACK-mask → temperature distribution | `draft/application/agent_pick_service.py` — `_select` | **Reuse as the reference.** Recomputing behaviour log-probs and the trained policy's log-prob is the *same* masked-softmax-over-PACK-at-temperature `_select` already does at inference; the RL trainer does it batched. |
| Joint policy+critic training loop scaffold | `train_draft_agent.py` — `TrainDraftAgentUseCase`, `_validate`, `run_eval`, warmup `_make_scheduler`, `_PlateauLR`, resume/best/early-stop | **Mirror structurally.** The RL use case copies this loop shape (warmup-then-constant LR, per-group clip, mini-epoch eval + best/latest + early stop, LR plateau annealing) and swaps the loss/metric bodies. |
| Checkpoint persistence | `draft/infrastructure/draft_agent_store.py` (`DraftAgentStore`) over `price_predictor.infrastructure.torch_checkpoint` | **Extend.** Add optional `rl_metadata` to `save_checkpoint`/`LoadedDraftAgentCheckpoint` (generation index, reference id, RL hyper-params). Backward compatible (defaults `None`). |
| Resume-precedence flag resolution | `sealed/infrastructure/cli_resume.py` — `resolve_resumable_args` | **Reuse.** The RL CLI registers resumable flags `default=None` and resolves CLI > resumed `train_config` > dataclass default, exactly like `train-draft-agent`. |
| Corpus read (streaming, partial-line tolerant) | `draft/infrastructure/draft_record_io.py` — `read_records` | **Reuse unchanged.** Multi-corpus = call it per `--drafts-path` / `--critic-corpus`. No schema change (spec FR-021). |
| Card embedding cache | `sealed/infrastructure/converted_card_locator.py` — `ConvertedCardLocator.load_embedding` (the `.npz` per card) | **Reuse.** Frozen encoder (Phase A); the trainer consumes cached card vectors, mirroring the gen-1 loader's shared-table memoization. |
| Deck-score yardstick + per-agent summary | `draft/application/analyze_generated_decks.py` (`deck_score_summary`); `generate-draft-data` | **Reuse (US2/US3 runbook).** The cross-generation yardstick is `generate-draft-data --pick-mode argmax` over a fixed mix + `analyze-generated-decks --agent <each>`. No new code. |

### Convention alignment

Mirror the `train-draft-agent` sibling exactly: new use case at
`src/draft/application/train_draft_agent_rl.py`; CLI subparser
`train-draft-agent-rl` in `src/draft/infrastructure/cli.py` (lazy import,
`set_defaults(func=…)`); checkpoints via `DraftAgentStore` to
`models/draft/agent/`; unit tests under `tests/unit/draft/` (one file per pure
helper, as `test_draft_loss.py` / `test_loader_walk.py` /
`test_draft_val_metrics.py` do) + an integration smoke test under
`tests/integration/`. Dependency direction unchanged: `draft` imports from
`sealed`/`price_predictor`, never the reverse.

### Third-instance check

The warmup-`LambdaLR` lambda and the per-group `clip_grad_norm_` helper now
recur across **five** trainers (`train_encoder`, `train_scorer`, `train_picker`,
`train_draft_agent`, and this RL trainer). The standing, documented project
decision (specs 017/018 research; `train_picker.py` docstring) is **defer
extraction of the whole loop** because each trainer's loss/metric/dataset-shape
bodies genuinely diverge, and a generic loop over five variants would be
speculative coupling (Simplicity-First).

However, two pieces are now *byte-identical* across five copies and stable: the
warmup schedule lambda and `_clip_per_group`. **Follow-up task proposed** (added
to tasks.md, non-blocking): extract just those two into a shared
`price_predictor/infrastructure/torch_training.py` (or `sealed`-side util) and
have all five trainers import them — extracting the genuinely-shared atoms while
leaving the divergent loop bodies per-trainer. The RL trainer is written to use
the shared helpers if that task lands first, else copies them like its siblings.
The REINFORCE-specific helpers (`_policy_entropy`, `_kl_penalty`) exist only in
`train_picker`; the RL trainer is their **second** instance, so they are copied
(not yet extracted) per the same convention.

## Technical decisions (resolving planning unknowns)

### D1 — On-policy estimator: REINFORCE + GAE baseline (no PPO)

**Decision**: Per learner pick, policy-gradient weight `ρ_t = log π(a_t|s_t)`
times a **detached GAE(λ) advantage** `A_t`; no clipped importance ratio.
**Rationale**: design doc — the strong warm start + KL anchor make destructive
updates unlikely, the regime where plain REINFORCE+baseline is adequate and
simplest, and the machinery already exists in `train_picker`. PPO/vine are out
of scope.

### D2 — GAE on a terminal-only reward; γ = 1; critic target = MC return R

**Decision**: The reward is terminal: `R` = the seat's pod-relative
leave-one-out `deck_score` (spec FR-010). Set `γ = 1` (fixed, not a flag — draft
reward is terminal so `G_t = R` for every pick, matching the gen-1 critic
target). The critic regresses `V(s_t) → R` (MC). The **advantage** is GAE(λ)
over the seat's ordered 45-pick value sequence: `δ_t = γ·V(s_{t+1}) − V(s_t)` for
non-terminal picks, `δ_T = R − V(s_T)`, `A_t = Σ_l (γλ)^l δ_{t+l}`. With the
default `--gae-lambda 0.95` (≈1) this is dominated by `R − V(s_t)` with light TD
smoothing from the next pick's value.
**Implication**: advantages need **per-seat trajectory ordering**, so the RL
loader groups examples by `(draft, seat)` and tags pick order; GAE is computed
over a full seat trajectory. **Rationale**: design doc "Credit assignment:
GAE(λ→1)"; reuses the gen-1 critic target (`R`) verbatim so the warm-started
critic stays calibrated.

### D3 — Advantages recomputed per epoch from the current critic (no_grad)

**Decision**: At the start of each epoch, run the critic over all learner
trajectories in a **batched `no_grad` GPU pass** to compute `V(s_t)` and cache
the detached `A_t` (and `R`); then do shuffled per-pick SGD where each step's
loss reads the cached `A_t`. **Rationale**: standard on-policy actor-critic
("compute advantages once per data pass with the value net at that point");
keeps the heavy critic forward batched (Principle VIII) and lets the
policy/value/entropy/KL terms batch over independently-shuffled picks (only the
advantage precompute needs trajectory order). REINFORCE's mild
cross-epoch policy staleness is bounded by the KL anchor; the operator keeps
`--epochs` modest, and regeneration each cycle (US3) restores strict on-policy
data. **Alternative rejected**: batching whole trajectories every step — more
complex batching for no accuracy gain at λ≈1.

### D4 — All policy distributions use the rollout temperature T

**Decision**: The behaviour log-prob, the trained policy's `log π(a_t)` in the
gradient, the entropy `H(π)`, and `KL(π‖π_ref)` are all computed on
`π_T = softmax(logits / T)` at the **rollout temperature** `T`
(`--rollout-temperature`, default `1.0`). **Rationale**: the actions were
sampled from `π_T`, so the on-policy gradient of `E_{a∼π_T}[R]` uses `∇log π_T`
(design doc "behaviour log-probs are recomputed at the same temperature, so the
policy gradient is exact"). Because no temperature is stored in the corpus
(operator-convention provenance, spec FR-021), `T` is supplied on the CLI and
MUST match the `generate-draft-data --temperature` used for the rollouts; a
mismatch is an operator error surfaced only by the D6 warning.

### D5 — Loss decomposition and which seats feed which term

**Decision** (spec FR-012/FR-013), per the loss
`L = −A_t·log π(a_t|s_t) + value_weight·MSE(V(s_t), R) − entropy_coef·H(π(·|s_t)) + kl_coef·KL(π(·|s_t)‖π_ref(·|s_t))`:

- **Policy-gradient + entropy + KL terms**: only **learner seats** (seat
  `agent ∈ --learner-agents`) and only at picks with an embeddable taken action
  (`learner_active`). Drawn **only** from the on-policy `--drafts-path` corpus.
- **Critic MSE term**: every **non-failed seat** (`critic_active`), from the
  on-policy corpus **and** any `--critic-corpus` (coverage). Failed builds
  (`deck=[]`, `deck_score=None`) are excluded from reward, pod mean, and all
  terms (spec FR-015).

This mirrors gen-1's `imitation_active` / `critic_active` split exactly, so the
loader and `_collate` are minimal adaptations of the gen-1 ones.

### D6 — Provenance: operator convention + a behaviour-anomaly warning

**Decision** (spec clarification 2026-06-13): no corpus schema change, no hard
provenance check. The trainer recomputes `log π_ref(a_t)` (at `T`) for learner
picks and **reports a summary**: mean behaviour log-prob and the fraction of
learner picks where the reference assigns the recorded action a probability
below a floor (`_PROB_FLOOR`, e.g. `1e-4`). If that fraction exceeds a small
threshold it logs a prominent warning ("corpus may not be on-policy for this
checkpoint / temperature") but **always proceeds** (spec FR-009, FR-016,
SC-004). On-policy pairing is the operator's responsibility (US3 names corpora
`drafts-genK.jsonl`). **Rationale**: the clarified, simplest safeguard that
still flags a gross mismatch.

### D7 — Warm-start, KL reference, and critic standardization

**Decision**: `--checkpoint <πₖ>` bootstraps actor+critic weights (like gen-1's
`--checkpoint`); a **frozen deep copy** of those weights is the KL reference
`π_ref` (like `train_picker`'s `reference_state`). The critic operates in the
loaded checkpoint's **standardized reward space**: reuse the stored
`critic_mean`/`critic_std` (do not recompute), and standardize `R` with them so
`V` and the target share a scale (advantages are in standardized-reward units;
absolute scale is absorbed by `--value-weight` / coefficients). `--resume`
continues an RL run (weights+optimizer+epoch+best+anneal position, like gen-1).
**Rationale**: keeps the warm-started critic calibrated; reuses gen-1 store
fields.

### D8 — Schedules for KL and entropy coefficients

**Decision**: Both `--kl-coef` and `--entropy-coef` are nonzero defaults that
**decay on a validation-driven schedule** reusing the `_EntropySchedule` pattern
from `train_picker` (held constant until val improves for N evals, then relaxed
on stalls) — "heavier early, relaxed as the critic proves out" (design doc).
`--gae-lambda` (0.95) and `--value-weight` (1.0) are constants.
**Rationale**: reuse the proven controller; matches the design doc's schedule
intent. Exact initial values / decay factor are tunable knobs (operator-set; not
correctness-bearing).

### D9 — In-run best checkpoint = held-out RL objective

**Decision** (spec clarification): best-checkpoint selection + early stopping use
the **held-out RL objective** (the same `policy + value + entropy + KL` sum) on a
draft-disjoint validation split, computed with the epoch's cached val
advantages. It is an in-run *guard* only; true cross-generation strength is
judged externally by the US2 yardstick. **Rationale**: reuses the gen-1
`best_val_loss`/early-stop/LR-plateau plumbing verbatim; the spec rules out a
within-run reward measure (the corpus is fixed; the policy can't be re-rolled
out inside the trainer).

## Performance Review (Principle VIII)

- **I/O batching & caching**: card vectors come from the `.npz` cache via
  `ConvertedCardLocator` memoized into one shared embedding table (gen-1
  `_Loader` pattern); each distinct card loaded once. Corpus read once via
  streaming `read_records`. **Addressed.**
- **GPU placement**: model (actor), `π_ref`, and the critic run on CUDA when
  available; inputs collated onto the device per batch (gen-1 `_collate`).
  **Addressed.**
- **GPU batching**: forward/backward batched via length-bucketed batches; the
  per-epoch advantage precompute is a batched `no_grad` critic forward over
  trajectories; behaviour-log-prob recompute is batched. No per-pick
  `.item()`/`.cpu()` in the training loop (advantages cached as tensors;
  logging reads scalars only at eval boundaries, as gen-1 does). **Addressed.**
- **Streaming & load-once**: corpus streamed; model/locator/embedding-table
  built once and reused across epochs. **Addressed.**
- No optimization beyond this checklist is proposed (Principle II); any further
  tuning would require a profile.
