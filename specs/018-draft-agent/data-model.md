# Data Model: Draft agent (generation 1)

**Feature**: `018-draft-agent` | **Date**: 2026-05-31

Entities are grouped by lifecycle stage: the **persisted corpus** (JSONL on
disk), the **derived training example** (built in memory by the loader), the
**model input** (typed token sequence), and the **checkpoint**. Field-level
rules trace to the FR numbers in `spec.md`.

## 1. Persisted corpus entities

### DraftRecord  (one JSONL line)

The self-contained unit of the corpus and of the draft-disjoint train/val split.

| Field | Type | Rules |
|---|---|---|
| `draft_id` | string (UUID) | Groups this record; the split unit (FR-035). |
| `run_id` | string (UUID) | Same for every record from one `generate-draft-data` invocation; lets a bad batch be filtered out (FR-005). |
| `timestamp` | string (ISO 8601 UTC) | Draft completion time (FR-013). |
| `seats` | array<Seat> | Length = `pod_size` (derived, default 8). Index = seat number. |
| `boosters` | array<Booster> | Length = `pod_size × packs`. Ordering pins all geometry (FR-016). |

Derived constants (read at load time, never stored as fields):
`pod_size = len(seats)`, `packs = len(boosters)/len(seats)`,
`pack_size P = len(boosters[0].picks)` (Assumptions; FR-016). A single draft's
boosters share one `P`.

Validation / reader rules:
- Readers tolerate a trailing partial final line (JVM-crash-mid-write) (FR-013).
- Card names are Forge canonical names (FR-013).
- Append-only file; `--resume` counts existing records toward `--n-drafts`
  (FR-012).

### Seat  (`seats[i]`)

One drafter in the pod. Source of imitation targets (when whitelisted) and of
the critic's pod-relative reward.

| Field | Type | Rules |
|---|---|---|
| `agent` | string | Free-form identifier (`forge-full`/`forge-r30`/`forge-r100`/…), sampled per seat from `--agent-mix` (FR-006). Read by `--imitation-agents` whitelist (FR-033). |
| `deck` | array<string> | 40-card built deck incl. basics, or `[]` on failed build (FR-014). |
| `deck_score` | number \| null | Frozen scorer scalar over the non-basic subset, or `null` on failed build (FR-014). |

State transition: failed build ⇒ `deck = []`, `deck_score = null` ⇒ excluded
from the pod mean and from critic training (FR-032, Edge Cases).

### Booster  (`boosters[k]`)

One opened pack; the geometry that lets any seat's observation history be
reconstructed.

| Field | Type | Rules |
|---|---|---|
| `set_code` | string | MTG set of this booster, **per-booster** (supports Chaos draft) (FR-015). |
| `picks` | array<string> | Cards in pick order, fully drained so `len(picks) == pack_size`; the multiset of `picks` is the booster's initial contents (FR-015). |

Geometry conventions (FR-016), fixed and external-state-free:
- `pack_number(k) = floor(k / pod_size) + 1`
- `opening_seat(k) = k mod pod_size`
- the pick at position `j` in `boosters[k]` was made by seat
  `(opening_seat + j · dir_p) mod pod_size`, with `dir_p = +1` for packs 1 & 3
  and `dir_p = −1` for pack 2.

## 2. Derived training entities (built by the loader, not persisted)

### TrainingExample  (one `(draft_id, seat s, pack p, pick i)`)

Up to `pod_size × pack_size` per draft (FR-030). Built from a `DraftRecord` via
the FR-016 geometry (FR-031).

| Component | Derivation | Rules |
|---|---|---|
| `state` | DraftState (below) | Reconstructed PACK/POOL/PASSED/TAKEN + recency (FR-031). |
| `imitation_target` | `boosters[k].picks[j]` index within `PACK` | The card actually taken at this pick (FR-031). Index into the deduped PACK action set. |
| `critic_target` | `seats[s].deck_score − mean({seats[j].deck_score : j≠s, j not failed})` | Leave-one-out pod-relative reward; shared by all of seat `s`'s states (Monte-Carlo) (FR-032). |
| `imitation_active` | `seats[s].agent ∈ imitation_whitelist` | Gates the policy loss for this example only; critic always active (unless failed build) (FR-033). |

Loader rules:
- A seat with a failed build contributes **no critic** target and is excluded
  from the pod mean (FR-032); its picks may still serve as imitation targets if
  whitelisted? No — failed-build seats have a degenerate label only for the
  critic; imitation eligibility depends solely on the agent whitelist. (A
  whitelisted seat that failed to build still has valid picks; it contributes
  imitation but not critic.)
- Cards with no `.npz` under `--cards-path` ⇒ warn (≤20 names + total) and drop
  those picks; do not block (FR-038).
- Critic targets are standardized (zero mean/unit variance) over the **training
  split** before MSE; mean/std recorded in the checkpoint; de-standardized at
  inference (FR-032).

## 3. Model-input entities

### DraftState  (the typed token sequence)

Layout `[CONTEXT] [POOL…] [PACK…] [PASSED…] [TAKEN…]` (FR-017). Order within a
group is insignificant; padded per batch, padding masked everywhere (FR-023).

### CardToken

One observed card instance, in exactly one of four mutually-exclusive types at
a time (FR-018).

| Part | Width | Source |
|---|---|---|
| `.npz` vector | `embedding_dim` | `ConvertedCardLocator.load_embedding(name)` (shared block across types). |
| type one-hot | 4 | `(POOL, PACK, PASSED, TAKEN)`, exactly one set; **sole differentiator of multiset membership** (FR-020). |
| `packs_ago` embed | `d(packs_ago)` (=4) | Learned table, index ∈ {0,1,2} (FR-021). |
| `pick_ago` embed | `d(pick_ago)` (=8) | Learned table, index ∈ {0,…,P−1} (FR-021). |

Type semantics & lifecycle (FR-018):

| Type | Dedup | Lifecycle |
|---|---|---|
| `POOL` | multiset | accumulates across all packs |
| `PACK` | distinct names (legal actions) | reset each pick/pack |
| `PASSED` | one per instance | emptied at every pack boundary |
| `TAKEN` | one per instance | accumulates across the draft |

Recency rules (FR-021):
- `packs_ago ∈ {0,1,2}` = packs since the card was last in the seat's pack
  (`0` = this pack / wheel-capable).
- `pick_ago ∈ {0,…,P−1}` = picks since the card was last in the seat's pack
  prior to this pick (`0` if never before), **frozen** at its end-of-pack value
  once `packs_ago ≥ 1`.

`PASSED → TAKEN` transitions (FR-019): (a) wheel diff on a pack's return —
missing cards (less the seat's own pick) become TAKEN, survivors re-enter PACK;
(b) pack-end flush — remaining PASSED flush to TAKEN when a pack is exhausted.

### ContextToken

| Part | Rule |
|---|---|
| value | sum of two learned `d_model`-wide embeddings: `pack_number ∈ {1,2,3}` and `pick_number ∈ {1,…,P}` (FR-022). |
| identity | no card embedding; no seat/set/agent identity (FR-022). |

## 4. Architecture & checkpoint entities

### DraftAgentConfig  (architecture record, stored in checkpoint)

Mirrors `PickerConfig` (`sealed/domain/picker_model.py`).

| Field | Default | Rule |
|---|---|---|
| `embedding_dim` | derived from `.npz` width | read from a sample card vector at startup. |
| `d_model` | `embedding_dim + 4 + d(packs_ago) + d(pick_ago)` | non-default inserts `Linear(concat_width, d_model)` (FR-025). |
| `n_layers` | 4 | SAB layers (FR-024). |
| `n_heads` | 8 | `d_model % n_heads == 0`, validated fast at startup (FR-026, SC-006). |
| `ff_dim` | `4 × d_model` | feed-forward width. |
| `dropout` | 0.0 | transformer dropout. |
| `d_packs_ago` | 4 | recency table width. |
| `d_pick_ago` | 8 | recency table width. |
| `P` (pack size) | derived from corpus | sizes `pick_number`/`pick_ago` tables (FR-040, Assumptions). |

### DraftAgentModel

| Component | Shape / rule |
|---|---|
| input projection | `Identity` by default, else `Linear(concat_width, d_model)` (FR-025). |
| trunk | `n_layers` × `SAB` (imported from `scorer_model`), all-tokens-attend-all (FR-024). |
| policy head | shared `Linear(d_model, 1)` over each `PACK` token → masked softmax over PACK only; `argmax` at inference (FR-027). |
| critic head | `Linear(d_model, 1)` over the `CONTEXT` token → scalar pod-relative reward (FR-028). |
| embedding tables | `packs_ago`, `pick_ago`, `pack_number`, `pick_number` (learned). Type is a one-hot, **not** a learned table (FR-040). |
| card embeddings | frozen (Phase A only) (FR-029). |

### DraftAgentCheckpoint  (`{timestamp}.pt` + `latest.pt` under `models/draft/agent/`)

| Key | Content (FR-040, FR-041) |
|---|---|
| `model_state_dict` | trunk + policy head + critic head + recency & context embedding tables. No encoder weights. |
| `config` | `DraftAgentConfig` (architecture + derived `embedding_dim` + `P`). |
| `epoch` | last completed epoch (resume counter). |
| `best_val_loss` | best validation `L` (selection metric, FR-036). |
| training metadata | incl. critic-target standardization **mean/std** (FR-032), optimizer state (for `--resume`). |

## Entity relationships

```
DraftRecord 1───* Seat            (pod_size seats)
DraftRecord 1───* Booster         (pod_size × packs boosters)
DraftRecord 1───* TrainingExample (≤ pod_size × pack_size examples; loader-derived)
TrainingExample 1───1 DraftState
DraftState   1───1 ContextToken
DraftState   1───* CardToken      (POOL ∪ PACK ∪ PASSED ∪ TAKEN, disjoint by instance)
DraftAgentModel ──uses── DraftAgentConfig
DraftAgentCheckpoint ──stores── {model_state_dict, DraftAgentConfig, standardization mean/std}
```

Split rule (FR-035): all `TrainingExample`s of one `draft_id` go entirely to
train or validation; first `--val-fraction` of distinct `draft_id`s (sorted
deterministically) form the held-out set with `random_seed = 42`.
