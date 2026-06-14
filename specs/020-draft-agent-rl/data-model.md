# Data Model: Draft Agent — RL Self-Play Fine-Tuning (Generation 2)

**Feature**: 020-draft-agent-rl | **Date**: 2026-06-13

This feature adds **no persisted schema** beyond the gen-1 checkpoint's optional
RL-metadata fields. The corpus (`drafts.jsonl`) is unchanged (spec FR-021). The
entities below are in-memory training structures plus the checkpoint extension.

## 1. TrainDraftAgentRLConfig (use-case input)

Dataclass mirroring `TrainDraftAgentConfig`, in
`draft/application/train_draft_agent_rl.py`. New / changed fields in **bold**.

| Field | Type | Default | Notes |
|---|---|---|---|
| **`checkpoint`** | `Path` | _required_ | Reference πₖ: warm-starts actor+critic, is the KL anchor, and must have generated `drafts_path` (FR-002). Bootstrap-from-weights (not `--resume`). |
| `drafts_path` | `Path` | `output/draft/drafts.jsonl` | The on-policy corpus; **sole** source of the policy gradient (FR-003). |
| **`critic_corpus`** | `tuple[Path, ...]` | `()` | Extra off-policy corpora, critic regression only (FR-004, repeatable). |
| **`learner_agents`** | `tuple[str, ...]` | _required_ | Whitelist of mix labels whose seats feed the policy gradient (FR-005). Non-empty (FR-017). |
| **`rollout_temperature`** | `float` | _required_ | T the corpus was sampled at; all policy distributions use it (research D4). **Required, no default** — a forgotten flag must fail fast, not silently train at T=1.0. |
| **`gae_lambda`** | `float` | `0.95` | GAE λ (research D2). |
| **`kl_coef`** | `float` | `0.1`* | Initial KL-anchor coefficient (scheduled, research D8). |
| **`entropy_coef`** | `float` | `0.01`* | Initial entropy-bonus coefficient (scheduled, research D8). |
| `entropy_decay_after` | `int` | `5` | Evals of monotone val improvement before coefficient decay arms (picker pattern). |
| **`value_weight`** | `float` | `1.0` | Critic-MSE coefficient. |
| `cards_path` | `Path` | `output/cardsfolder/` | `.npz` cache. |
| `lr`, `warmup_frac`, `batch_size`, `max_grad_norm`, `epochs`, `val_fraction`, `evals_per_epoch`, `patience`, `lr_decay_*` | — | gen-1 defaults | Same semantics as `train-draft-agent`; resumable. |
| `resume` | `Path \| None` | `None` | Continue an RL run (xor `checkpoint`). |

*Initial schedule values are tunable, not correctness-bearing. `γ` is fixed at
`1.0` (not a flag, research D2). Architecture flags (`d_model` etc.) are **not**
accepted — architecture is inherited from `--checkpoint`/`--resume`.

**Validation (startup, FR-017; SC-003)** — fail fast, nonzero exit:
`checkpoint` (or `resume`) exists and `config.embedding_dim` matches the `.npz`
cache width (reuse gen-1 `_check_dims`); `learner_agents` non-empty;
`rollout_temperature` supplied and `> 0` (required flag — a missing value is a
usage error, not a silent default); `resume` xor `checkpoint`. No provenance
check (research D6).

## 2. Trajectory (per-seat ordered states for GAE)

Built by the RL loader; one per `(draft, seat)` that contributes any pick.

| Field | Type | Notes |
|---|---|---|
| `draft_index` | `int` | Groups the draft-disjoint split (FR-035 analogue). |
| `picks` | `list[RLExample]` | The seat's picks **in (pack, pick) order** (1→45), so `V(s_{t+1})` is `picks[t+1]`. |
| `reward` | `float \| None` | Terminal pod-relative leave-one-out reward `R` (`None` ⇒ failed build; the trajectory then contributes neither reward nor gradient). |
| `is_learner` | `bool` | `seat.agent ∈ learner_agents`. |

Lifecycle: built once from the corpus; `R` and `is_learner` are fixed; `V` and
the GAE advantage `A_t` are (re)computed per epoch into each `RLExample`
(research D3), not stored on disk.

## 3. RLExample (one `(draft, seat, pack, pick)` training state)

Adapts gen-1 `DraftExample`: the same typed-token tensors (the state is
identical), different per-example labels. Card embeddings are kept as int rows
into a shared table (gen-1 memory pattern), not materialized here.

| Field | Type | Notes |
|---|---|---|
| `draft_index` | `int` | Split key. |
| `card_idx` | `np.ndarray (N,) int32` | Rows into the shared embedding table. |
| `type_idx` | `np.ndarray (N,) int8` | `TYPE_POOL/PACK/PASSED/TAKEN`. |
| `packs_ago`, `pick_ago` | `np.ndarray (N,) int8` | Recency (unchanged). |
| `pack_number`, `pick_number` | `int` | CONTEXT token + pick ordering within the trajectory. |
| `action_token` | `int` | Absolute index of the **taken** PACK token (the action `a_t`), or `-1` if the taken card has no `.npz` (then `learner_active=False`). |
| `learner_active` | `bool` | `is_learner and action_token >= 0` — feeds policy-grad + entropy + KL (research D5). |
| `critic_active` | `bool` | Seat is non-failed — feeds critic MSE (any corpus). |
| `reward` | `float` | Raw `R` (pre-standardization); de-standardized targets computed in `_collate`. |
| `advantage` | `float` | Cached detached GAE `A_t`, refreshed per epoch (D3); `0.0` until first compute / for non-learner picks. |

`n_tokens = card_idx.shape[0]`. The CONTEXT token is added inside
`DraftAgentModel.forward` (not stored), as in gen-1.

## 4. Loss terms (per batch)

Computed in `_compute_loss` over an `_Batch` (collated like gen-1, plus
`advantage` and a per-token `pack_mask`). Distributions on `π_T` (logits / T):

```
policy = −( advantage[learner] · logπ_T(action)[learner] ).mean()
value  =  value_weight · MSE( V[critic_active], standardize(R)[critic_active] )
entropy = −entropy_coef · H(π_T(·|s))[learner].mean()
kl      =  kl_coef · KL(π_T(·|s) ‖ π_ref,T(·|s))[learner].mean()
total   = policy + value + entropy + kl
```

`logπ_T`, `H`, `KL` reuse `train_picker`'s masked-softmax helpers (PACK-masked
for `draft`), with the `0·-inf` NaN-gradient guards. `π_ref` is a frozen forward
(no_grad). Standardization uses the **loaded checkpoint's** `critic_mean/std`
(research D7).

## 5. Behaviour-anomaly summary (FR-009 / SC-004; research D6)

Per run, over a **seeded 1% subset of learner picks** (floored at
`_ANOMALY_SAMPLE_FLOOR` learner picks for stability on small corpora — it's a
gross-mispairing probe, not a full-corpus pass): `mean_behaviour_logprob` and
`frac_below_floor = fraction with π_ref,T(a_t) < _PROB_FLOOR`. Logged once as
`sampled n/N learner picks …`; if `frac_below_floor > _ANOMALY_FRACTION`, a
prominent "corpus may be off-policy for this checkpoint/temperature" warning is
emitted. Never aborts.

## 6. Checkpoint RL metadata (extends `DraftAgentStore`)

Backward-compatible additions to the saved `.pt` payload and
`LoadedDraftAgentCheckpoint` (default `None` ⇒ a gen-1 checkpoint reads cleanly):

| Field | Type | Notes |
|---|---|---|
| `rl_metadata` | `dict \| None` | Present on RL-produced checkpoints (FR-019). |
| `rl_metadata.generation` | `int` | `reference.generation + 1`; absent reference ⇒ reference is gen-1, so candidate is gen-2. |
| `rl_metadata.reference_checkpoint` | `str` | Path/id of `--checkpoint` (operator-convention provenance, research D6). |
| `rl_metadata.algorithm` | `str` | `"reinforce+gae"`. |
| `rl_metadata.gae_lambda`, `kl_coef`, `entropy_coef`, `value_weight`, `rollout_temperature` | floats | The RL hyper-parameters, so the checkpoint is self-describing for the next cycle. |

Existing fields unchanged: `model_state_dict`, `optimizer_state_dict`, `epoch`,
`best_val_loss` (now the held-out **RL objective**, research D9), `config`,
`critic_mean`, `critic_std`, `train_config`, `lr_decay_count`. Saved to
`models/draft/agent/{timestamp}.pt` + `latest.pt` (FR-020). No encoder weights
(Phase A).

## 7. Entity relationships

```
TrainDraftAgentRLConfig ──drives──> RL use case
   │ checkpoint (πₖ) ──► DraftAgentModel (actor+critic)  +  frozen π_ref copy
   │ drafts_path ──read_records──► DraftRecord ──loader──► Trajectory[] ──► RLExample[]
   │ critic_corpus[] ──read_records──► DraftRecord ──loader──► RLExample[] (critic_active only)
   ▼
per-epoch: critic no_grad forward over Trajectory ──► A_t cached on RLExample
per-step:  _Batch(RLExample…) ──► _compute_loss ──► total.backward()
   ▼
DraftAgentStore.save_checkpoint(..., rl_metadata) ──► models/draft/agent/{ts}.pt + latest.pt
```
