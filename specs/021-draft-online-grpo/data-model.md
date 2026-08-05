# Data Model: Draft Agent — Online Self-Play GRPO Trainer (Generation 3)

**Feature**: 021-draft-online-grpo | **Date**: 2026-08-05

In-memory entities only. Nothing on disk changes shape: the corpus keeps the
gen-1 `drafts.jsonl` schema and the checkpoint keeps the gen-1 `.pt` payload
(with the already-optional `rl_metadata` dict filled in). Reused entities
(`DraftRecord`/`Seat`/`Booster`, `RawPickState`, `DraftAgentConfig`) are not
restated here — see `draft/domain/draft_geometry.py`,
`draft/application/draft_pick_states.py`, `draft/domain/draft_agent_model.py`.

## 1. `TrainDraftAgentOnlineConfig`

The resolved run configuration (spec FR-001…FR-006, FR-024). Every field is
echoed at startup (FR-013). Home: `draft/application/train_draft_agent_online.py`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `learner_label` | `str` | — | required; the mix label piloted by the live policy |
| `learner_checkpoint` | `Path` | — | required; warm-starts the policy at round 0 |
| `frozen` | `dict[str, Path]` | `{}` | label → checkpoint; untrained references |
| `anchor` | `str \| None` | `None` | a `frozen` label; defaults to the sole frozen label |
| `mix` | `list[tuple[str, int]]` | `gen-3:5,gen-1:3,forge-r30:1,forge-r100:1` | parsed by `agent_mix.parse_agent_mix` |
| `scorer_checkpoint` | `Path` | `models/sealed/scorer/latest.pt` | must exist; produces `deck_score` = reward |
| `build_method` | `"greedy" \| "picker"` | `"greedy"` | pool → deck before scoring |
| `picker_checkpoint` | `Path` | `models/sealed/picker/latest.pt` | must exist only when `build_method == "picker"` |
| `cards_path` | `Path` | `output/cardsfolder/` | `.npz` cache; fixes the model input width |
| `rollout_temperature` | `float \| None` | `None` | **required, positive**; sampling *and* every policy distribution |
| `lr` | `float` | `1e-4` | AdamW |
| `drafts_per_round` | `int` | `10` | fresh drafts generated + trained on per round |
| `anchor_window` | `int` | `100` | sliding window (drafts) for the anchor margin |
| `snapshot_every` | `int` | `25` | rounds between timestamped snapshots |
| `max_rounds` | `int \| None` | `None` | optional budget; `None` ⇒ until interrupt |
| `set_code` | `str \| None` | `None` | restrict rollouts to one set |
| `batch_size` | `int` | `32` | fixed-and-forget; the 8 GB VRAM budget |
| `max_grad_norm` | `float` | `1.0` | per-group clip |
| `warmup_steps` | `int` | `200` | one LR ramp at run start, then constant (research D15) |
| `seed` | `int` | `42` | torch/numpy init, batch shuffling, pick-sampling RNG (research D12) |
| `output_path` | `Path` | `output/draft/drafts.jsonl` | shared corpus; the handle is opened **`"a"` unconditionally**, line-buffered, once per run — never `"w"`, which would truncate the canonical corpus (clarified 2026-08-05) |
| `max_consecutive_faults` | `int` | `5` | inherited pick-fault auto-abort |

**Absent by design** (spec FR-006, Out of Scope): `--value-weight`,
`--gae-lambda`, `--kl-coef`, `--entropy-coef`, any coefficient-decay knob,
`--val-fraction`, `--patience`, `--epochs`, `--lr-decay-*`, `--resume`,
`--pick-mode` (always `sample`).

### 1.1 Startup validation (FR-024) — all before any update, exit nonzero

1. Exactly one `--learner NAME=PATH`; its `PATH` exists and loads.
2. `learner_label` appears in `mix` and is **not** also a `--frozen` label.
3. Every `mix` label ∈ `FORGE_BUILTINS ∪ frozen ∪ {learner_label}`.
4. `anchor` (explicit or defaulted) is a `frozen` label **and** appears in `mix`;
   ambiguous when `len(frozen) > 1` and `--anchor` is omitted → error.
5. `rollout_temperature` supplied and `> 0`.
6. `scorer_checkpoint` exists; `picker_checkpoint` exists when
   `build_method == "picker"`.
7. Learner checkpoint architecture matches the `.npz` cache width
   (`_check_dims`) and the live geometry (`config.packs == PACKS`); same for
   every frozen checkpoint.
8. `drafts_per_round ≥ 1`, `anchor_window ≥ 1`, `batch_size ≥ 1`,
   `snapshot_every ≥ 1`, `max_rounds` (if given) `≥ 1`.

## 2. `RoundBatch`

One round's fresh, single-use data (spec FR-001, FR-011). Built after generation,
consumed by the update, then dropped.

| Field | Type | Notes |
|---|---|---|
| `index` | `int` | 0-based round index |
| `records` | `list[DraftRecord]` | exactly `drafts_per_round` completed drafts |
| `examples` | `list[OnlineExample]` | learner picks only, flattened |
| `table` | `np.ndarray (C, embedding_dim)` | per-round card table (research D14) |
| `learner_rewards` | `list[float]` | one per surviving learner seat (pre-standardisation) |
| `dropped_seats` | `int` | learner seats excluded (failed build / no LOO baseline) |
| `gen_seconds` | `float` | wall-clock spent generating |

**Invariants**: `len(records) == drafts_per_round`; every record has ≥1 learner
seat (guaranteed upstream, research D2); a record is appended to the corpus as
soon as it arrives, independent of whether the round later no-ops.

## 3. `OnlineExample`

One learner `(draft, seat, pack, pick)` training state. A trimmed sibling of
`train_draft_agent_rl.RLExample`: no critic fields, no GAE, no `learner_active`
flag (non-learner picks are never materialised).

| Field | Type | Notes |
|---|---|---|
| `card_idx` | `np.ndarray (N,) int32` | rows into `RoundBatch.table` |
| `type_idx` | `np.ndarray (N,) int8` | `TYPE_POOL/PACK/PASSED/TAKEN` |
| `packs_ago` | `np.ndarray (N,) int8` | learned-recency index |
| `pick_ago` | `np.ndarray (N,) int8` | learned-recency index |
| `pack_number` | `int` | 1-based |
| `pick_number` | `int` | 1-based |
| `action_token` | `int` | absolute index of the taken `PACK` token |
| `advantage` | `float` | the seat's shared, detached `A` (§4) |
| `n_tokens` | property | `card_idx.shape[0]`, for length bucketing |

Produced by walking each learner seat with
`draft_pick_states.iter_seat_pick_states` (research D6). A pick whose whole
`PACK` is un-embeddable yields no state; a pick whose *taken* card is
un-embeddable yields `action_position == -1` and is **dropped** (no usable
action).

## 4. Reward → advantage (spec FR-008, FR-009, FR-023)

```
R_i      = deck_score_i − mean({deck_score_j : j ≠ i, deck_score_j is not None})   # per seat, any label
A_i      = (R_i − mean(R over the round's learner seats)) / std(R over the round's learner seats)
```

- Terminal (γ = 1): `A_i` is shared by all of seat `i`'s picks.
- `deck_score is None` (failed build) ⇒ the seat is excluded from `R`, from every
  pod mean, and from the gradient (FR-022).
- A learner seat whose pod has **no other** non-failed seat has an undefined
  leave-one-out baseline ⇒ excluded (counts toward `dropped_seats`).
- **Degenerate round** (FR-023): fewer than 2 surviving learner rewards, or
  `std < 1e-8` ⇒ the round is a **no-op** — no optimizer step, so the weights do
  not move — logged as `skipped (no signal)`. The per-round `latest.pt` write,
  the corpus append, and the anchor-window update all still happen; the
  checkpoint is simply content-identical to the previous round's.

## 5. Loss (spec FR-010)

Per minibatch of learner picks, at `π_T = softmax(logits / T)` masked to `PACK`:

```
L = − mean_over_batch( A · log π_T(a | s) )
```

No critic/value term, no GAE, no KL, no entropy bonus. The critic head is
forwarded (the model returns it) and discarded.

## 6. `RoundDiagnostics`

Everything printed for one round (spec FR-014…FR-018, SC-003/SC-004). Computed
from the round's own data plus one post-update batched `no_grad` pass with the
pre-update weights `π_k` (research D9).

| Axis | Field | Source |
|---|---|---|
| identity | `index`, `drafts`, `total_drafts`, `learner_seats`, `dropped_seats`, `learner_picks`, `gen_seconds`, `train_seconds`, `skipped` | round bookkeeping |
| reward | `reward_mean`, `reward_std`, `adv_std`, `adv_near_zero_frac` (`\|A\|<0.1`), `adv_large_frac` (`\|A\|>0.5`), `adv_absmax` | §4 |
| exploration | `entropy`, `perplexity` = `exp(entropy)`, `off_argmax_rate` | `π_k` forward |
| movement | `mean_logp` (of taken actions under `π_k`), `policy_loss` (mean over the round's steps), `grad_norm` (pre-clip, mean over steps), `kl_prev_new` = `KL(π_k ‖ π_{k+1})` | steps + `π_k`/`π_{k+1}` forward |
| progress | `anchor_margin`, `label_means: dict[str, float]`, `window_drafts` | §7 |

`off_argmax_rate` = fraction of learner picks whose recorded (sampled) action is
not `argmax π_k` over the same `PACK` set — exact, because both come from the
same reconstructed state.

## 7. `AnchorWindow`

Sliding-window progress state (spec FR-017, FR-021).

| Field | Type | Notes |
|---|---|---|
| `window` | `deque[dict[str, list[float]]]` | `maxlen = anchor_window`; one entry per draft: label → that draft's non-`None` `deck_score`s |
| `anchor_label` | `str` | fixed for the whole run (FR-021) |
| `learner_label` | `str` | fixed for the whole run |

- `label_mean(label)` = mean of every score for `label` across the window.
- `margin` = `label_mean(learner) − label_mean(anchor)`; `None` until both labels
  have at least one scored seat in the window.
- `window_drafts` = `len(window)`.
- Best margin and its round index are tracked for the final summary (FR-019).

## 8. Checkpoint payload (spec FR-027, FR-028)

Written by the unchanged `DraftAgentStore.save_checkpoint`:

| Key | Value |
|---|---|
| `model_state_dict` | trunk + policy + **critic (carried unchanged)** + recency/context tables |
| `optimizer_state_dict` | AdamW state (snapshot only; there is no `--resume`) |
| `epoch` | the round index |
| `best_val_loss` | `inf` — no held-out metric exists (FR-026) |
| `critic_mean` / `critic_std` | carried through from the base checkpoint verbatim |
| `config` | `DraftAgentConfig`, inherited from the base checkpoint |
| `train_config` | the resolved §1 config (paths stringified) |
| `rl_metadata` | see below |

```python
rl_metadata = {
    "generation": base_generation + 1,       # base rl_metadata["generation"] or 1 (gen-1)
    "base_checkpoint": str(learner_checkpoint),
    "algorithm": "online-grpo",
    "lr": lr,
    "rollout_temperature": rollout_temperature,
    "drafts_per_round": drafts_per_round,
}
```

`generation` is the **lineage counter**, independent of the `--learner` mix
label: a run labeled `gen-3` in the mix but warm-started from a gen-1 base writes
`generation: 2`. The label names a kind of seat; the counter records how many
training generations deep the weights are.

No critic/GAE/KL/entropy hyper-parameters are stored (they do not exist); no
encoder weights (Phase A).

Paths: `models/draft/agent/latest.pt` every round;
`models/draft/agent/{timestamp}.pt` every `snapshot_every` rounds and once at
run end/interrupt.

## 9. Java worker input (research D2)

| Property | Type | Meaning |
|---|---|---|
| `-Ddraft.required.agent=<label>` | optional string | After the pod's per-seat mix draw, if no seat carries `<label>`, one uniformly-chosen seat is overwritten with it. Absent/blank ⇒ unchanged behaviour. |

Set by `DraftWorkerConnector.start(required_agent=…)` to the learner label; it is
the mechanism behind spec FR-003's "every generated draft has at least one
learner seat".

The value reaches the connector through the existing launcher rather than a
bespoke one: `GenerateDraftDataConfig` gains an optional
`required_agent: str | None = None` field (default `None` ⇒ today's behaviour
byte-for-byte), `GenerateDraftDataSupervisor._default_launch_worker` forwards it
into `DraftWorkerConnector.start`, and the online trainer sets it to
`learner_label` on the config it hands the supervisor. Without that last step the
property is never set and learner-free pods are played, so it is part of the
FR-003 chain, not an optimisation.
