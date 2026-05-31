# Draft agent — imitation policy + critic (generation 1)

This spec is **normative** — it specifies what to build. Design rationale,
rejected alternatives, and the discussion that produced these choices live in
[`../experiments/2026-05-30-draft-agent-design.md`](../experiments/2026-05-30-draft-agent-design.md).

# Goal

Build a generation-1 draft agent: given the state of an MTG booster draft,
it picks the next card (imitation policy) and predicts the final deck's
quality (critic). Trained offline from Forge-generated draft data on a shared
transformer body. No RL and no live integration into Forge in gen 1; both
arrive in gen 2.

# Scope

**Gen 1 (this spec):**

- A CLI to generate a corpus of complete Forge drafts.
- A typed-token state representation for the draft.
- A two-headed model (policy + critic) trained jointly.
- A training CLI.
- Offline evaluation: imitation pick accuracy and critic regression error.

**Gen 2 (out of scope):**

- Integrating the trained policy as a live Forge draft seat (self-play).
- RL fine-tuning of the policy (actor-critic, GAE).
- Self-play data regeneration across generations.

# Reused artefacts

| Artefact | Role here |
|----------|-----------|
| Sealed scorer (`models/sealed/scorer/latest.pt`) | Labels each finished drafted deck with a scalar score. |
| One-shot picker (`models/sealed/picker/latest.pt`) | Builds a 40-card deck from each seat's 45-card drafted pool for scoring. |
| `.npz` cache from `encode-cards` (`output/cardsfolder/`) | Per-card vectors of width `embedding_dim = pooled_dim + FEATURE_COUNT`. |
| `GreedyDeckBuilder` + `compute_basic_lands` | Fallback (SA) deck builder for labeling. |
| Forge-connector pattern (Python supervisor + Java worker) | Same as `generate-pools` / `match-outcomes`. |

# Package layout

Top-level Python package `draft`, hexagonal style (`domain` / `application` /
`infrastructure`), entry point `python -m draft <subcommand>`. Imports from
`sealed` (scorer, picker, `GreedyDeckBuilder`, card-embedding-layout helpers)
and `price_predictor` (tokenizer); nothing in `sealed` or `price_predictor`
imports back from `draft`. Model artefacts live at `models/draft/agent/`;
data at `output/draft/`.

# 1. Draft state (model input)

A single decision is one seat at one pick. The state is a **typed token
sequence**: a `CONTEXT` token plus card tokens for everything the seat has
observed.

## 1.1 Card tokens

Each card token is the card's `.npz` vector with three concatenated draft
features: a 4-dim type one-hot and two recency embeddings (§ 1.4). The four
token types are mutually exclusive — every observed card *instance* is in
exactly one set at a time.

| Type | Contents | Dedup | Lifecycle |
|------|----------|-------|-----------|
| `POOL` | Cards this seat has drafted. | Multiset (copy count is part of the deck-building state). | Accumulates across all packs. |
| `PACK` | Cards in the pack now in front of the seat — the legal actions this pick. A `P`-card booster shows `P − pick_number + 1` cards. | Deduped to distinct card names (the action is a name; copies are equivalent picks). | Replaced every pick, reset to a fresh booster each pack. |
| `PASSED` | Cards the seat saw and passed *this pack*, whose fate it hasn't observed yet. | One token per card instance (distinct copies stay distinct). | Within-pack transient — empties at every pack boundary. |
| `TAKEN` | Cards the seat saw that ended up in an opponent's pool — known taken via wheel diff or pack-end flush. | One token per card instance. | Accumulates across the draft. |

**`PASSED` → `TAKEN` transitions** happen at two times:

- *Wheel diff:* in an 8-seat pod a `P`-card pack returns to the seat 8 picks
  after first sighting, with `P − 8` cards left. The cards missing on return
  (less the seat's own pick) move `PASSED → TAKEN`; the survivors enter
  `PACK` again and re-enter `PASSED` after the seat passes them.
- *Pack-end flush:* once a pack is exhausted, all remaining `PASSED`
  instances flush to `TAKEN` (every card from the pack is necessarily in
  some pool by then).

**Positive wheel signal.** A wheeled card re-entering `PACK` carries
`pick_ago ≈ 8` (below) — the "it came back, colours open" signal lives in
the recency on the `PACK` token; no separate type is needed.

## 1.2 Recency: `packs_ago`, `pick_ago`

Two concatenated learned embeddings per card token, measuring how long since
the card was last in the seat's pack.

- `packs_ago ∈ {0, 1, 2}` — packs since the card was last in the seat's pack.
  `0` = this pack (wheel-capable); `≥ 1` = a prior pack.
- `pick_ago ∈ {0, …, P − 1}` — picks since the card was last in the seat's
  pack **prior to the current pick** (`0` if never in the pack before now).
  **Frozen at the pack boundary**: once `packs_ago` becomes `≥ 1`, `pick_ago`
  stops advancing and holds its end-of-pack value.

| Token | `(packs_ago, pick_ago)` |
|-------|-------------------------|
| Fresh `PACK` card | `(0, 0)` |
| Wheeled `PACK` card just back | `(0, ≈ 8)` |
| `PASSED` card 1 pick ago | `(0, 1)`, growing as it sits |
| Card from a prior pack | `(≥ 1, frozen)` |
| `POOL` card | `(packs_ago, pick_ago)` of when the seat drafted it |

The embedding rows train normally from every token that selects them; no
stop-gradient for frozen indices (the index carries no gradient anyway).

## 1.3 Context token

One `CONTEXT` token, formed as the sum of two learned `d_model`-wide
embeddings: `pack_number ∈ {1, 2, 3}` and `pick_number ∈ {1, …, P}`. No
card embedding; carries no seat / set / agent identity.

## 1.4 Assembled sequence and `d_model`

Layout:

```
[CONTEXT] [POOL …] [PACK …] [PASSED …] [TAKEN …]
```

- Order within each type group is not significant (the trunk is
  permutation-equivariant; no positional encoding).
- Padded per batch to the longest sequence in the batch; padding positions
  masked from attention and every head.
- Sequences reach ~200–300 tokens by pack 3 (`TAKEN` accumulates across the
  draft, `PASSED` flushes into it each pack boundary).

Per the scorer/picker `FEATURE_COUNT` convention, the draft features are
**concatenated** onto the card embedding and `d_model` grows to include
them — no input projection:

```
d_model = embedding_dim + 4 (type one-hot) + d(packs_ago) + d(pick_ago)
```

- Type: 4-dim one-hot (no separate learned table — the first SAB projection
  learns the per-type interpretation).
- `packs_ago`, `pick_ago`: small learned embedding tables (e.g. 3 × 4 and
  P × 8).
- **Constraint:** `d_model` must be divisible by `n_heads`; size the feature
  widths so the *total* is a multiple of `n_heads` (the picker/scorer fail
  fast otherwise).

The `CONTEXT` token has no card embedding and is built directly to `d_model`
as the sum of its two metadata embeddings.

# 2. Model architecture

A set transformer over the § 1.4 sequence with two heads.

1. **Token assembly** as above. No input projection by default; a non-default
   `--d-model` (§ 5.2) inserts a `Linear` from the concatenated width to
   `d_model`.
2. **Trunk:** `n_layers` Set Attention Blocks (SAB, the scorer/picker/encoder
   primitive). All tokens attend to all tokens.
3. **Policy head:** shared `Linear(d_model, 1)` applied to each `PACK` token
   output → one logit per pack card → masked softmax over `PACK` positions
   only. `argmax` at inference; categorical sample during gen-2 rollouts.
4. **Critic head:** `Linear(d_model, 1)` applied to the `CONTEXT` token
   output → scalar predicted final pod-relative reward.

Card embeddings are frozen (Phase A only — joint encoder fine-tuning is out
of scope).

# 3. Training

Supervised, offline, single pass over the recorded draft corpus.

## 3.1 Training examples

Each `(draft_id, seat s, pack_number p, pick_number i)` is one example. From
the JSON record (§ 4) the loader locates the booster `k` the seat saw at
this pick — by the § 4.1 conventions, `s_open = (s − (i − 1) · dir_p) mod
pod_size`, `k = (p − 1) · pod_size + s_open`, offset `j = i − 1` — then:

- `PACK` = `boosters[k].picks[j:]`
- `POOL` = each prior pick of seat `s` (same geometry, prior `(p', i')`).
- `PASSED` / `TAKEN` = walk each prior booster the seat saw, partitioning
  passed cards by the wheel-diff and pack-end-flush rules (§ 1.1).
- Recency follows each instance's most recent in-pack pick prior to
  `(p, i)`, frozen at pack boundaries (§ 1.2).
- Context scalars = `(p, i)`.

Labels:

- **Imitation target**: `boosters[k].picks[j]`.
- **Critic target**: `seats[s].deck_score − mean({seats[j].deck_score : j ≠ s})`
  (leave-one-out pod-relative reward; § 4.2). All 45 states of a seat share
  this single label (Monte-Carlo regression).

A draft yields up to 8 × 45 = 360 examples.

## 3.2 Loss

The two heads train on **different seat subsets**: policy on the imitation
whitelist, critic on all seats.

```
L = imitation_weight · CE(policy_logits, taken_index)   [whitelisted-agent states only]
  +     critic_weight · MSE(critic_pred, seat_reward)    [all states]
```

- Imitation whitelist set by `--imitation-agents` (§ 5.2; default
  `forge-full`); the critic ignores it.
- Seats whose pool failed to build a legal deck (`deck = []`,
  `deck_score = null`) are excluded from critic training and from the
  pod mean used to compute the reward.

## 3.3 Optimisation and split

- **Optimiser:** AdamW, per-parameter-group max-norm 1.0 gradient clipping.
- **LR schedule:** linear warmup over the first `--warmup-frac` (default
  0.05) of scheduled steps, then constant `--lr`.
- **Split:** draft-disjoint — all picks of a `draft_id` go entirely to train
  or validation. First `--val-fraction` (default 0.2) of distinct
  `draft_id`s form the held-out set; `random_seed = 42`.
- **Batching:** states padded per batch (§ 1.4); length bucketing
  permitted.
- **Best checkpoint:** selected by validation `L`. Per-epoch log reports
  the loss decomposition plus validation imitation top-1 / top-3 accuracy
  and critic MSE sliced by `pack_number`.
- Cards in the corpus with no `.npz` under `--cards-path` are warned (up
  to 20 names + total) and their picks dropped; do not block the run.

# 4. Draft-event file format

One self-contained JSON record per line in an append-only JSONL file at
`output/draft/drafts.jsonl`. Card names are Forge canonical names. Readers
tolerate a trailing partial line (JVM-crash-mid-write recovery).

## 4.1 Record schema

```json
{
  "draft_id": "<uuid>",
  "run_id": "<uuid>",
  "timestamp": "<ISO 8601 UTC>",
  "seats": [
    {"agent": "forge-full", "deck": ["...40 names..."], "deck_score": 12.34}
  ],
  "boosters": [
    {"set_code": "BLB", "picks": ["...P names in pick-order..."]}
  ]
}
```

(One entry shown per array; a real record has `pod_size` seats and
`pod_size × packs` boosters.)

Conventions that pin everything else:

- `seats` has length `pod_size`; `seats[i]` is seat `i`.
- `boosters` has length `pod_size × packs`. Ordering: pack-1 boosters first
  (one per opening seat, in seat order), then pack-2, then pack-3. So for
  `boosters[k]`:
  - `pack_number = floor(k / pod_size) + 1`
  - `opening_seat = k mod pod_size`
- Inside `boosters[k].picks` (length `pack_size`, in pick-order), the pick
  at position `j` was made by seat
  `(opening_seat + j · dir_p) mod pod_size`, where `dir_p = +1` for
  L-passing packs (1 and 3) and `−1` for R-passing pack (2).
- Every booster is fully drained (`len(picks) == pack_size`); the multiset
  of `picks` is also the booster's initial contents.

Pod size, pack count, and pack size are read at load time from
`len(seats)`, `len(boosters) / len(seats)`, and `len(boosters[0].picks)`.

## 4.2 Field semantics

| Field | Meaning |
|-------|---------|
| `draft_id` | UUID grouping this record. |
| `run_id` | UUID of the `generate-draft-data` invocation; one run stamps the same `run_id` on every record it appends, so a bad batch can be located and filtered/deleted after the fact. |
| `timestamp` | ISO 8601 UTC at draft completion. |
| `seats[i].agent` | Free-form identifier (`forge-full`, `forge-r30`, `forge-r100`, `draft-agent-v0.1.2`, `human-nicolas`, …). Set by `--agent-mix` (§ 5.1); read by `--imitation-agents` (§ 5.2). |
| `seats[i].deck` | 40-card deck built from seat `i`'s 45-card pool by `--build-method` (§ 5.1); includes basics. |
| `seats[i].deck_score` | Frozen scorer's scalar for the non-basic subset of `deck` (basics not scored). The critic label is the **pod-relative** form `seats[i].deck_score − mean({seats[j].deck_score : j ≠ i})` — leave-one-out, computed at load time. Failed builds: `deck = []`, `deck_score = null`; excluded from the mean and from critic training. |
| `boosters[k].set_code` | MTG set of this booster. Per-booster (not per-draft) so Chaos draft is supported natively. |
| `boosters[k].picks` | Cards in pick-order. |

## 4.3 Building the deck

`deck` and `deck_score` are produced by `--build-method` (§ 5.1), defaulting
to `picker`. `greedy` is the higher-fidelity SA fallback, selected only if
the § 5.3 validation rejects the picker.

The picker is fed the 45-card pool with no padding or resizing — its set
transformer is length-agnostic, so the 45 cards run through the same code
path a ~80-card sealed pool takes. `(1, 45, embedding_dim)` tensor +
all-true mask → 45 logits → the spec-017 spell-quota walk selects the
23-spell deck; `compute_basic_lands` fills basics.

# 5. CLI

Two subcommands on `python -m draft <subcommand>`.

## 5.1 `generate-draft-data`

Spawns a single Java draft worker (`DraftWorkerMain`, analogous to
`PoolMain` / `MatchWorkerMain`) that runs Forge's draft AI for all eight
seats at the agents assigned by `--agent-mix`. The supervisor/worker split
is kept for JVM-crash recovery (the supervisor restarts the worker on
crash); a single worker suffices. The supervisor generates a fresh
`run_id` (UUID) at startup and stamps it on every record. For each
completed draft the supervisor receives the transcript from the worker,
builds a deck from each seat's 45-card pool, scores it, and appends one
complete JSONL record to `drafts.jsonl`.

**Worker → supervisor transport.** The worker emits each completed draft
as one line on stdout, prefixed with the sentinel `<<DRAFT-EVENT-JSON>>`
followed by the compact (no embedded newlines) JSON transcript (boosters +
per-seat agent identifiers, without `deck`/`deck_score`); `flush()` after
each line. The supervisor reads stdout line-by-line, filters for the
sentinel, defensively parses the suffix as JSON (skipping anything that
fails to parse), completes each record with the picker-built `deck` and
the scorer's `deck_score`, and appends to `drafts.jsonl`. Forge's
incidental stdout and the worker's own diagnostics on stderr are ignored;
stderr is piped to a log file. On supervisor crash the in-flight draft is
lost; on worker JVM crash the supervisor restarts and continues.

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-drafts` | _(required)_ | Number of complete drafts to generate. |
| `--set` | _(none → random per draft)_ | Restrict all drafts to one set code; otherwise each draft independently picks a random sealed-legal set. |
| `--agent-mix` | `forge-full:6,forge-r30:1,forge-r100:1` | Probability weights for assigning agents to pod seats. Each of the `pod_size` seats is sampled **independently** from this categorical distribution (default ≈ 6/8 chance `forge-full`, 1/8 each of `forge-r30`/`forge-r100`), so pod compositions vary draft-to-draft. The sampled identifier goes into `seats[i].agent`. Built-ins: `forge-full` = Forge draft AI; `forge-r30` / `forge-r100` = 30% / 100% of that seat's picks replaced by uniform-random legal picks. |
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer used to label finished decks. |
| `--build-method` | `picker` | `picker` (default) or `greedy`. See § 4.3. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Picker used when `--build-method picker`; ignored otherwise. |
| `--cards-path` | `output/cardsfolder/` | `.npz` cache. |
| `--output-path` | `output/draft/drafts.jsonl` | Destination JSONL; appended; created if missing. |
| `--resume` | off | Append to existing files and continue toward `--n-drafts`, counting drafts already present. |

## 5.2 `train-draft-agent`

Trains policy + critic jointly on a recorded corpus.

| Flag | Default | Meaning |
|------|---------|---------|
| `--drafts-path` | `output/draft/drafts.jsonl` | JSONL of recorded drafts (§ 4). |
| `--cards-path` | `output/cardsfolder/` | `.npz` cache. |
| `--d-model` | _derived: `embedding_dim` + feature widths_ | Default = concatenated token width, no input projection; a different value inserts a `Linear`. |
| `--n-layers` | `4` | SAB layers. |
| `--n-heads` | `8` | Attention heads; `d_model` must be divisible by this (fails fast at startup otherwise). |
| `--ff-dim` | `4 × d_model` | Feed-forward width. |
| `--dropout` | `0.0` | Transformer dropout. |
| `--imitation-weight` | `1.0` | CE coefficient. `0` = critic-only run. |
| `--critic-weight` | `1.0` | MSE coefficient. `0` = imitation-only run. |
| `--imitation-agents` | `forge-full` | Whitelist (comma-separated, or the flag repeated) of agent identifiers whose picks are imitation targets. Critic unaffected — trains on all seats. |
| `--lr` | `3e-4` | AdamW LR. |
| `--warmup-frac` | `0.05` | LR linear-warmup fraction. |
| `--batch-size` | `32` | States per gradient step. |
| `--max-grad-norm` | `1.0` | Per-group gradient-norm cap. |
| `--epochs` | `100` | Max epochs. |
| `--val-fraction` | `0.2` | Draft-disjoint validation fraction. |
| `--patience` | `10` | Early-stop epochs without val improvement. |
| `--resume` | _(none)_ | Continue a stopped run (weights + optimiser + epoch counter + best-val). Architecture flags forbidden. Mutually exclusive with `--checkpoint`. |
| `--checkpoint` | _(none)_ | Bootstrap a fresh run from this checkpoint's weights only. Architecture flags forbidden. Mutually exclusive with `--resume`. |

## 5.3 Picker-vs-SA builder validation (script)

A one-off diagnostic that decides whether `--build-method picker` is
trustworthy or whether to fall back to `greedy`. Not a CLI subcommand — a
short ad-hoc script (~40 lines), run once per picker/scorer checkpoint
pair before committing to a large `generate-draft-data` run.

**Procedure.**

1. Sample a few hundred drafted 45-card pools. The natural source is an
   existing `drafts.jsonl` (each seat's 45-card final pool is derivable by
   walking its picks across its three boosters, per § 4.1 conventions);
   fresh pools also work.
2. Build each pool two ways: the one-shot picker and the SA
   `GreedyDeckBuilder`.
3. Score both decks with the frozen scorer → `picker_score`, `sa_score`.
4. Report **Spearman rank correlation** of `picker_score` vs `sa_score`
   across pools (the gating number); plus the distribution of
   `sa_score − picker_score` (median + spread) for interpretation.

**Reference ceiling.** SA is stochastic, so compute the SA-vs-SA rank
correlation across independent restarts on the same pools as the threshold.

**Gating.**

- Picker tracks SA ≈ as well as SA tracks itself → use `--build-method picker`.
- Picker materially below the reference, or pool-composition-dependent gap →
  use `--build-method greedy`.

# 6. Model artifacts

Checkpoints at `models/draft/agent/`:

```
models/draft/agent/
  {timestamp}.pt
  latest.pt
```

Each checkpoint:

- `model_state_dict` — trunk + policy head + critic head + recency
  (`packs_ago`, `pick_ago`) and context (`pack_number`, `pick_number`)
  embedding tables. (Type is a one-hot, not a learned table.)
- `config` — architecture (`d_model`, `n_layers`, `n_heads`, `ff_dim`,
  `dropout`, derived `embedding_dim`) and booster size `P` (sizes the
  `pick_number` / `pick_ago` tables).
- `epoch`, `best_val_loss`, training metadata.

No encoder weights are embedded (Phase A only).

# Evaluation

Gen 1 is evaluated offline; the definitive head-to-head-vs-Forge scoreboard
is gen 2.

- **Policy (imitation).** Top-1 and top-3 pick-match accuracy against
  held-out Forge picks, sliced by `pack_number` / `pick_number`.
- **Critic.** Monte-Carlo regression error on held-out drafts, sliced by
  draft stage; ranked across model variants on a shared held-out set so
  shared luck cancels.

# Out of scope

- Live integration of our policy as a Forge draft seat (self-play). Gen 2.
- RL fine-tuning of the policy (actor-critic, GAE, REINFORCE/PPO). Gen 2.
- Surgical pack-2 forking / vine-style advantage estimation. Gen 2.
- Self-play data regeneration. Gen 2.
- Auxiliary opponent-archetype prediction head. Gen 2.
- Encoder fine-tuning alongside the drafter (Phase B analogue).
- Picker fine-tuning on variable / smaller pools (only if § 5.3 rejects the
  picker as label-builder; the SA fallback otherwise suffices).
- An inference / play-out CLI (gen 1 ships data-gen + training only).
