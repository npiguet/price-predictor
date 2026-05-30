# Draft agent — imitation policy + critic (generation 1)

# Goal

A model that drafts a Magic: The Gathering booster draft: at every pick it
chooses one card from the pack in front of it, conditioned on everything it
has seen so far, so that the 45 cards it ends with build into a strong deck.
Unlike sealed (a single fixed pool), draft is a sequential decision process
under partial observation — the agent sees only its own picks and the packs
that reach it, while seven other drafters deplete the same packs in parallel.

This spec covers **generation 1**: a supervised foundation built from Forge
draft data — an **imitation policy** (pick like a competent Forge drafter)
and a **critic** (predict the final deck quality from any mid-draft state),
trained jointly on a shared encoder. Generation 1 needs no reinforcement
learning and no live integration of our own model into Forge; it is trained
entirely offline from recorded drafts.

The picker (`017-one-shot-deck-picker`) and the draft agent solve different
problems: the picker selects 23 spells from a *complete* pool in one shot;
the draft agent decides *which card to acquire next* given an incomplete pool
and a depleting pack. The draft agent reuses the picker's and scorer's card
representation (the `.npz` embedding cache from `encode-cards`) and the same
set-transformer primitives, but adds the sequential draft state as input.

# Scope: generation 1 vs generation 2

**Generation 1 (this spec):**

- A CLI to generate a training corpus of complete Forge drafts, recorded in a
  draft-event file format, with a configurable agent mix across the eight
  seats.
- A draft state representation (typed token sequence) fed to a transformer.
- A model with a **policy head** (one logit per pack card → the pick) and a
  **critic head** (one scalar → predicted final deck score).
- A CLI to train both heads jointly on the recorded corpus.
- Offline evaluation: imitation pick accuracy and critic regression error.

**Generation 2 (future, separate spec) — explicitly out of scope here:**

- Integrating our trained policy as a live seat in a Forge draft (self-play).
- Reinforcement-learning fine-tuning of the policy (actor-critic with GAE,
  REINFORCE/PPO), optimising the pod-relative terminal reward.
- Surgical pack-2 forking / vine credit assignment with common random numbers.
- Self-play data regeneration across generations.

Generation 1 produces a competent drafter and a usable value estimator on
their own. Generation 2 refines them once self-play becomes possible. The RL
machinery is described in *Design rationale and rejected alternatives* so the
generation-1 artifacts are built to slot into it, but none of it is built now.

# Background

## Why draft is hard

Booster draft is a **partially observable** sequential game. Eight drafters
sit in a pod; each opens a 15-card booster, takes one card, passes the rest;
packs pass left in packs 1 and 3 and right in pack 2; this repeats until all
packs are empty (3 packs × 15 picks = 45 cards per drafter). A drafter sees
only the packs that reach it, never the other seats' pools. Three classic
difficulties follow:

1. **Hidden information** — seven other drafters consume the same cards
   simultaneously; you never observe their pools directly.
2. **Open colours / signal reading** — to avoid fighting over contested
   colours you must infer what is "open" from what reaches you late.
3. **Sparse, hard-to-attribute reward** — deck quality is only known at the
   end and is hard to attribute to any single pick.

The agent conditions on its whole observation history (pool so far +
cards seen-and-passed + pack/pick counters). That history is an implicit
belief about the hidden state, so hidden information and open-colour reading
emerge from the input representation without an explicit opponent model.

## What already exists and is reused

- **Sealed scorer** (`013`, gen4-512 and earlier): a Set Transformer that
  maps a deck to a scalar quality score, trained on pairwise Forge match
  outcomes with `binary_cross_entropy_with_logits` on the score delta — i.e.
  a Bradley-Terry model, so `sigmoid(S_A − S_B)` is already a win probability
  up to one temperature. Used here to label finished drafted decks.
- **One-shot picker** (`017`): SAB trunk + per-card head over a pool. The
  draft agent shares its architectural primitives and its `.npz` input
  contract.
- **Card embedding cache** (`encode-cards`): one `.npz` per card under
  `output/cardsfolder/`, a `float32` array of width `embedding_dim`
  (`pooled_dim + FEATURE_COUNT`, equal to the scorer's `ScorerConfig.d_model`).
  The draft agent looks cards up by name in this cache exactly as the picker
  and scorer do; `is_land_embedding` reads the land flag from the
  deterministic-feature block.
- **Forge connector**: Python supervisor + Java workers pattern
  (`generate-pools`/`PoolMain`, `match-outcomes`/`MatchWorkerMain`). Draft
  data generation follows the same pattern with a new draft worker.
- **Greedy deck builder** + `compute_basic_lands`: turn a drafted 45-card
  pool into a 40-card deck for scoring.

The draft agent is its own top-level Python package, `draft`, laid out in the
same hexagonal style as the other packages (`domain` → `application` →
`infrastructure`) with a `python -m draft <subcommand>` entry point. It is
**not** part of `sealed`. It **imports from** `sealed` (the scorer, picker,
`GreedyDeckBuilder`, the `.npz` embedding cache, and the card-embedding layout
helpers) and from `price_predictor` (the shared MTG tokenizer); nothing in
`sealed` or `price_predictor` imports back from `draft`. The objective is win
rate *when the Forge AI pilots the deck* — Forge's piloting tendencies are the
target distribution by design, not a bias to correct.

# Specification

This section is prescriptive. Rationale and rejected options are deferred to
*Design rationale and rejected alternatives*.

## 1. Draft state (model input)

A single decision is made by one seat at one pick. The state at that decision
is a **typed token sequence** built from card embeddings plus a context token.

### 1.1 Card tokens

Each card token starts from that card's cached embedding (`embedding_dim`
wide = `pooled_dim + FEATURE_COUNT`, looked up by Forge canonical name in the
`.npz` cache). Onto that context-free vector the draft state concatenates a
type one-hot and two recency embeddings — features that cannot live in the
cache because they depend on where the card sits in *this* draft. § 1.4 covers
the concatenation and the resulting `d_model`.

**Four token types**, mutually exclusive. Every card *instance* the seat has
observed is in exactly one set at a time — never two. As the draft proceeds an
instance moves between sets (e.g. a `PASSED` instance the wheel reveals was
taken moves to `TAKEN` and is removed from `PASSED`; see the transitions
below):

| Type | Contents |
|------|----------|
| `POOL` | Cards this seat has already drafted. Accumulates across all three packs. **Multiset** — two copies of a card are two tokens, since copy count changes the deck-building state. |
| `PACK` | Cards in the pack now in front of the seat — the legal actions this pick. A `P`-card booster shows `P − pick_number + 1` cards. **Deduped to one token per distinct card name**: the action is choosing a name, and duplicate copies (boosters can contain them; collation varies by set) are equivalent picks. Replaced every pick, reset to a fresh booster each pack; never accumulates. |
| `PASSED` | Cards the seat saw and passed *this pack*, whose fate it hasn't observed yet. One token per card instance (not deduped by name — two physical copies are two instances). A **within-pack transient**: it empties at each pack boundary (below) and never carries across packs. |
| `TAKEN` | Cards the seat saw that ended up in an **opponent's pool** — known taken, whether seen leaving via the wheel or inferred certain at the pack's end (below). One token per card instance (not deduped by name). The contested-colours signal. Accumulates across the draft. |

**The wheel, the round-end flush, and the PASSED/TAKEN split.** A card moves
`PASSED → TAKEN` once the seat knows an opponent took it, which happens at two
times. (1) *The wheel*: in an 8-seat pod a `P`-card pack returns to the seat 8
picks after it was first seen, with `P − 8` cards left; diffing what the seat
saw against what came back identifies exactly which cards the seven intervening
opponents took (those move `PASSED → TAKEN`), while the survivors move
`PASSED → PACK` (available again — the seat picks one and the rest return to
`PASSED`). (2) *The pack's end*: once a pack is exhausted every card from it is
in some pool, so anything still in `PASSED` is necessarily in an opponent's
pool — the whole remaining `PASSED` set flushes to `TAKEN`. So the wheel gives
early, within-pack detection (informing picks 9–15) and the flush resolves the
rest for use in later packs; `PASSED` is consequently a within-pack transient
that empties at every boundary, and `TAKEN` is the complete record of cards
opponents took. The *positive* half of the wheel (a strong card that came back,
signalling your colours are open) needs no separate type: it reappears as a
`PACK` token carrying a recency of ~8 (below).

**Duplicates, by role.** Whether a type keeps duplicate cards follows from
what the type *is*. `PACK` is the action space — the pick is one card *name*
and two physical copies are the same choice — so it is deduped to one token
per name; keeping both would only inflate that name's selection probability by
its copy count, which has nothing to do with the card's pick value. `POOL` is
the deck the seat holds, where a second copy is a materially different build
(you can run the playset), so it stays a multiset. `PASSED`/`TAKEN` are
observation history, kept one token per *card instance* — distinct physical
copies stay distinct (two copies taken from different packs are two `TAKEN`
tokens, the stronger contested signal), but re-observing the *same* instance
(e.g. a card that wheels back) updates that instance's status and recency
rather than spawning a second token. Each instance is in at most one set at a
time, so when the wheel reveals a `PASSED` instance was taken it moves to
`TAKEN` and leaves `PASSED`.

**Recency — `packs_ago` and `pick_ago`.** Every card token carries how long
since the card was last in the seat's pack, as two concatenated learned
embeddings:

- `packs_ago ∈ {0, 1, 2}` — packs since the card was last in the seat's pack.
  `0` = this pack (live, wheel-capable); `≥ 1` = a prior pack (stale colour
  history).
- `pick_ago ∈ {0 .. P−1}` — picks since the card was last in the seat's pack
  *prior to the current pick* (`0` if it has never been in the pack before
  now), **frozen at the pack boundary**: once `packs_ago` becomes `≥ 1` it
  stops advancing and holds its end-of-pack value, so it stays in the
  within-pack range and never re-encodes what `packs_ago` already says.

So a fresh `PACK` card is `(0, 0)`; a wheeled `PACK` card just back is
`(0, ~8)` — `pick_ago ≥ 1` on a `PACK` token is exactly the "it came back,
colours open" signal; a card passed one pick ago is `(0, 1)`, growing as it
sits; and a prior-pack card is `(≥1, frozen)`. When a wheeled card is passed
again `pick_ago` resets to `1` (its last in-pack pick is now the wheel pick) —
the reset falls out of the definition rather than being special-cased. Recency
matters most for `PACK`/`PASSED`/`TAKEN`; for `POOL` the same pair records when
the card was drafted (incidental, kept for uniformity).

Freezing fixes only which embedding *row* a stale card reads — that row still
trains normally from every token that selects it, live or stale (no
stop-gradient; the index carries no gradient in the first place, and the
shared row's meaning is disambiguated by the `packs_ago` it is paired with).

### 1.2 Context token

One additional token of type `CONTEXT`, carrying the per-state globals as a
sum of learned, `d_model`-wide embeddings (it has no card embedding of its
own):

- `pack_number` ∈ {1, 2, 3}.
- `pick_number` ∈ {1 .. P}, where P is the maximum booster size in the corpus
  (the embedding table is sized from the data, like `train-encoder`'s
  `max_seq_len`).

The token deliberately carries **no seat, set, or agent identity**: a draft pod
is a symmetric ring, so absolute seat position carries no signal; omitting the
set keeps the model **format-agnostic** — it reads the format from the cards
themselves, whose embeddings already encode it, so one model works across sets
and Chaos draft; and the per-seat agent identifier is a synthetic
data-generation artefact the model never needs (§ 3.4). The context token participates in attention so every card
token can condition on draft progress, and its trunk output is the pooled
representation the critic head reads (§ 2).

### 1.3 Assembled sequence

```
[CONTEXT] [POOL …] [PACK …] [PASSED …] [TAKEN …]
```

Order within each type group is not significant (the trunk is
permutation-equivariant; no positional encoding). Sequences are padded per
batch to the longest in the batch, with an attention mask; padding positions
never contribute to attention or to any head. `TAKEN` accumulates across the
whole draft (with `PASSED` flushing into it at each pack boundary, § 1.1), one
token per observed card instance, so by pack 3 a sequence can reach ~200–300
tokens — larger than the ~80-card sealed pools but
well within reach; recency (§ 1.1) lets the model discount the stale tail
rather than needing a hard reset, and the accumulation can be capped to a
recent window if sequence length becomes a throughput problem.

### 1.4 Token vector width and `d_model`

Per the convention the scorer and picker already use for their `FEATURE_COUNT`
block, the per-card draft features are **concatenated** onto the card
embedding (not projected into a fixed width), and `d_model` grows to include
them — no input projection:

```
d_model = embedding_dim + 4 (type one-hot) + d(packs_ago) + d(pick_ago)
```

- **Type** is a 4-dim one-hot; no separate learned type table is needed, since
  the first SAB's projection learns the per-type interpretation from the
  indicator dims.
- **`packs_ago`, `pick_ago`** are small learned embedding tables, concatenated.

`d_model` must stay divisible by `n_heads`. `embedding_dim` already is, so size
the feature widths to sum to a multiple of `n_heads` — e.g. with `n_heads = 8`,
type 4 + `packs_ago` 4 + `pick_ago` 8 = +16. The widths are tunable; only the
divisibility of the total is fixed.

The `CONTEXT` token (§ 1.2) has no card embedding, so it is assembled to this
same `d_model` purely as a sum of its (`d_model`-wide) metadata embeddings.

## 2. Model architecture and pick mechanism

The model is a set transformer over the § 1.3 sequence with two heads.

1. **Token assembly.** Each card token is its cache embedding with the type
   one-hot and the `packs_ago`/`pick_ago` embeddings concatenated (§ 1.1,
   § 1.4); `d_model` is the resulting concatenated width, with no input
   projection by default (a non-default `--d-model` inserts a projection from
   that width; § 5.2). The `CONTEXT` token is a `d_model`-wide sum of metadata
   embeddings (§ 1.2).
2. **Trunk.** A stack of `n_layers` **Set Attention Blocks** (SAB, the same
   primitive as the scorer/picker/encoder). All tokens attend to all tokens.
3. **Policy head.** A shared `Linear(d_model, 1)` applied to each `PACK` token
   output produces one logit per pack card; a masked softmax over the `PACK`
   positions only (`POOL`/`PASSED`/`TAKEN`/`CONTEXT` excluded) is the pick
   distribution directly. Because `PACK` is deduped to distinct card names
   (§ 1.1), every token is a distinct action — no name-aggregation is needed:
   imitation cross-entropy targets the picked card's token, and sampling draws
   one token. The pick is `argmax` at inference (deterministic) and a
   categorical sample during generation-2 rollouts.
4. **Critic head.** A `Linear(d_model, 1)` applied to the `CONTEXT` token
   output produces one scalar: the predicted final deck score for this seat
   (the reward defined in § 4.2 / § 3.4).

**Output at one pick:** a distribution over the pack (pick the card), and a
scalar value (what the eventual deck is expected to be worth from here). The
two heads share the trunk; this is the actor and critic on one body, ready
for generation-2 actor-critic without architectural change.

Card embeddings are frozen (consumed from the `.npz` cache). Jointly
fine-tuning the encoder is out of scope (analogous to the scorer/picker
Phase B).

## 3. Training

Supervised, offline, single pass over the recorded draft corpus. No
environment interaction, no live drafting.

### 3.1 Training examples

Each `(draft_id, seat s, pack_number p, pick_number i)` is one example. From
the JSON record (§ 4) the loader locates the booster `k` that reaches seat `s`
at this pick — by the conventions in § 4.1, `s_open = (s − (i − 1) · dir_p)
mod pod_size`, `k = (p − 1) · pod_size + s_open`, and the seat's offset into
that booster is `j = i − 1`. Then:

- **`PACK`** = `boosters[k].picks[j:]` — the cards still in the booster when
  seat `s` sees it.
- **`POOL`** = each prior pick of seat `s`, found by the same geometry
  applied to each prior `(p', i')`.
- **`PASSED` / `TAKEN`** = walk each prior booster the seat saw: the cards
  the seat passed are those at offsets after the seat's offset; for a booster
  that wheeled back to the seat, the cards strictly between the two seat
  offsets are observed `TAKEN`; remaining passed cards flush to `TAKEN` at
  each pack boundary (§ 1.1); the rest stay `PASSED` within the current pack.
- **Recency** (`packs_ago` / `pick_ago`) follows each instance's most recent
  in-pack pick prior to `(p, i)`, frozen at pack boundaries (§ 1.1).
- The context scalars `pack_number` and `pick_number` come from `(p, i)`.

The label pair is:

- **Imitation target**: `boosters[k].picks[j]` — the card the recorded agent
  took at this pick.
- **Critic target**: the pod-relative form of `seats[s].deck_score` —
  `seats[s].deck_score − mean({seats[j].deck_score : j ≠ s})`, i.e. the
  seat's own score minus the mean of the *other* pod seats' scores (leave-
  one-out) — computed at load time from the JSON record (§ 4.2). Every pick
  of a given seat shares this single reward (Monte-Carlo regression — the
  same label for all 45 states of a draft seat).

A draft yields up to 8 × 45 = 360 examples.

### 3.2 Loss

Two heads with **different training seats** (§ 3.4): the policy imitates only
seats whose `agent` (§ 4.1) is on the `--imitation-agents` whitelist (§ 5.2,
default `forge-full`), while the critic regresses on **all** seats.

```
L = imitation_weight · CE(policy_logits, taken_index)   [whitelisted-agent states only]
  +     critic_weight · MSE(critic_pred, seat_reward)    [all states]
```

- **Policy: whitelisted agents only.** Cross-entropy *copies* the
  demonstrator, so training it on a weaker agent's picks would teach the
  policy to reproduce bad picks. It imitates competent play only; steering
  *away* from bad picks is a generation-2 RL effect, not something imitation
  can express (§ 3.4).
- **Critic: all seats.** The critic's job is to value states, including bad
  ones, so weaker-agent seats give it the coverage of incoherent pools it
  needs to anchor the low end of the value scale (§ 3.4).
- The imitation whitelist is set by `--imitation-agents` (§ 5.2); the critic
  ignores it.

### 3.3 Optimisation and split

- **Optimiser**: AdamW with per-parameter-group max-norm 1.0 gradient
  clipping (project convention).
- **LR schedule**: linear warmup over the first `--warmup-frac` (default 0.05)
  of scheduled steps, then constant `--lr`.
- **Split**: draft-disjoint — all picks of a `draft_id` go entirely to train
  or validation. The first `--val-fraction` (default 0.2) of distinct
  `draft_id`s encountered form the held-out set; `random_seed = 42`.
- **Batching**: states padded per batch (§ 1.3); length bucketing (grouping
  similar-length states, as in `train-encoder`) is permitted for throughput
  but not semantically required.
- **Best checkpoint**: selected by validation `L`. The per-epoch log reports
  the loss decomposition plus validation imitation top-1 / top-3 accuracy and
  critic MSE sliced by draft stage (§ Evaluation).
- Cards referenced in the draft corpus with no `.npz` under `--cards-path` are
  reported via a log warning (naming up to 20 + the total) and their picks
  dropped, mirroring `train-encoder`; they do not block the run.

### 3.4 What each head trains on

**The policy imitates only seats whose `agent` is on a competent-demonstrator
whitelist** (default `forge-full`; § 5.2). Cross-entropy maximises the
probability of the demonstrated pick — it copies, with no notion of good or
bad — so feeding it a weaker agent's picks would teach it to reproduce them.
Avoidance ("steer away from this pick") requires an advantage signal that signs
the gradient by outcome, which is generation-2 RL; imitation can't express it.
So the policy learns from competent demonstrations only.

**The critic trains on all seats**, including those played by weaker Forge
variants. Its job is to value states, and that includes recognising bad ones,
so it needs to *see* incoherent pools — exactly what weaker agents supply.
Training it only on whitelisted-agent seats would leave it blind to the low
end of the value scale and prone to wild extrapolation there. The apparent
objection — that a weaker agent's final-deck label reflects a *bad*
continuation, not a competent one — is mostly benign here: (a) the critic is
used as a baseline for advantages, and *any* state-dependent value is a valid
baseline regardless of bias; (b) genuinely-bad pools score low whoever finishes
them, so their labels are about right anyway, and the continuation confound
only bites for middling pools finished sloppily. We accept that mild bias rather
than reintroducing an agent tag (dropped as artificial, § 1.2); generation 2
removes it outright by regressing the critic on the policy's *own* rollouts,
so the continuation is the policy's by construction, yielding `V^π`.

Together this is the generation-1 foundation for generation 2's "steer away
from bad picks": imitation establishes competent play, the broadly-trained
critic learns to recognise bad states, and gen-2 actor-critic wires the
critic's advantage into the policy gradient so it actually steers.

## 4. Draft-event file format

Each complete draft is recorded as **one self-contained JSON record per line**
in an append-only JSONL file at `output/draft/drafts.jsonl`. Card names are
Forge canonical names. Readers tolerate a trailing partial line
(JVM-crash-mid-write recovery).

### 4.1 Record schema

One JSON object per line:

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

(Each array has one entry shown; a real record has `pod_size` seats and
`pod_size × packs` boosters.)

Everything else is derived from these fields by convention:

- `seats` has length `pod_size`, and `seats[i]` is seat `i`.
- `boosters` has length `pod_size × packs` (24 for a standard 8-seat 3-pack
  draft). The convention is: pack-1 boosters first (one per opening seat, in
  seat order), then pack-2, then pack-3. So for `boosters[k]`:
  - `pack_number = floor(k / pod_size) + 1`
  - `opening_seat = k mod pod_size`
- Inside `boosters[k].picks` (length `pack_size`, in pick-order), the pick at
  position `j` was made by seat `(opening_seat + j · dir_p) mod pod_size`,
  where `dir_p = +1` for L-passing packs (1 and 3) and `−1` for R-passing
  pack (2).
- Every booster is fully drained (`len(picks) == pack_size`), so the multiset
  of `picks` is also the booster's initial contents.

Pod size, pack count, and pack size are read at load time from `len(seats)`,
`len(boosters) / len(seats)`, and `len(boosters[0].picks)` respectively.

### 4.2 Field semantics

- **`draft_id`** — UUID grouping the record.
- **`run_id`** — UUID identifying the `generate-draft-data` invocation that
  produced this record. One supervisor run stamps the same `run_id` on every
  draft it appends, so a bad batch can be located and deleted/filtered after
  the fact by filtering on this field.
- **`timestamp`** — ISO 8601 UTC string at the moment the draft completed.
- **`seats[i].agent`** — free-form identifier for whatever played seat `i`:
  Forge variants (`forge-full`, `forge-r30`, `forge-r100`), our model
  versions (`draft-agent-v0.1.2`), humans (`human-nicolas`), etc. Identifiers
  are set by `--agent-mix` at generation time (§ 5.1). The trainer's
  `--imitation-agents` flag (§ 5.2) decides which agents count as imitation
  targets; the critic ignores it.
- **`seats[i].deck`** — the 40-card deck built from seat `i`'s 45-card final
  pool by the configured `--build-method` (§ 5.1), with basics included.
- **`seats[i].deck_score`** — the frozen scorer's scalar for that deck's
  non-basic subset (the scorer's input contract; basics are not scored). The
  critic label per pick of `(draft_id, i)` is the **pod-relative** form:
  `seats[i].deck_score − mean({seats[j].deck_score : j ≠ i})` — the seat's
  own score minus the mean of the *other* pod seats' scores (leave-one-out;
  seat `i` is not in the mean, so it can't subtract itself). Computed at
  load time. The mean skips any seat whose pool failed to build a legal deck
  (its `deck` is `[]` and `deck_score` is `null`); such seats are also
  excluded from critic training.
- **`boosters[k].set_code`** — MTG set of this booster. Per-booster, not
  per-draft, so Chaos draft (each booster from a different set) is supported
  natively.
- **`boosters[k].picks`** — the cards in the order they were taken from this
  booster.

### 4.3 Building the deck (default: the picker)

The `deck` and `deck_score` fields are produced by `--build-method` (§ 5.1),
defaulting to the one-shot picker for throughput: labeling runs one build per
seat per draft (eight per draft), and the picker's ~5 ms forward versus the SA
builder's ~5 s is the difference between labeling costing days and costing
minutes across a large corpus. SA (`greedy`) is the higher-fidelity fallback,
used only if the § 5.3 validation rejects the picker.

**Feeding 45 cards to a picker trained on ~80 — the technique.** No padding,
resizing, or special handling. The picker is a set transformer (SAB layers, no
positional encoding, attention masked to the real cards) with a per-card head,
so its forward pass is intrinsically length-agnostic. The 45 drafted cards are
stacked into a `(1, 45, embedding_dim)` tensor with an all-true card mask and
run through *exactly* the same code path a ~80-card sealed pool takes, yielding
45 logits; the § 1.1 spell-quota walk then selects the deck. The only
difference from training is that 45 is below the picker's ~60–90 training range
— this is extrapolation in set size, not a different input format. Set
attention extrapolates gracefully in sequence length, and the dominant term in
each per-card logit is the card's own frozen embedding, which is pool-size
invariant; § 5.3 confirms empirically that this holds before the picker is
trusted as the labeler.

## 5. CLI

Two subcommands on the `draft` package's own entry point
(`python -m draft <subcommand>`).

### 5.1 `generate-draft-data`

Generates a corpus of complete Forge drafts. A Python supervisor spawns Java
a Java draft worker (a new `DraftWorkerMain`, analogous to `PoolMain` /
`MatchWorkerMain`) that runs Forge's draft AI for all eight seats at the agents
assigned by `--agent-mix`. A single worker is sufficient — Forge drafts are
fast (no game simulation, just pick decisions) — but the supervisor/worker
split is kept for JVM-crash recovery: the supervisor restarts a fresh worker
on each crash. The supervisor generates a fresh `run_id` (UUID) at startup and
stamps it on every record it writes. For each completed draft, the supervisor
receives the transcript from the worker (boosters + per-seat agent
identifiers), builds a deck from each seat's 45-card pool, scores it with the
frozen scorer, and appends one complete JSONL record to `drafts.jsonl` (§ 4).

**Worker → supervisor transport.** The worker emits each completed draft as a
single line on stdout, prefixed with the sentinel `<<DRAFT-EVENT-JSON>>`
followed by the compact (no embedded newlines) JSON transcript — boosters and
per-seat agent identifiers, without `deck`/`deck_score`. The worker `flush()`es
after each line. The supervisor reads stdout line-by-line, filters for the
sentinel, attempts to parse the suffix as JSON (silently skipping anything
that doesn't parse), completes each record with the picker-built `deck` and
the scorer's `deck_score`, and appends the finalised record to `drafts.jsonl`.
Forge's incidental stdout output and the worker's own diagnostics on stderr
are ignored by the parser (stderr is piped to a log file for post-mortems).
This avoids stdout-noise misparsing without the complexity of a file-based
pending area. If the supervisor crashes mid-record the in-flight draft is
lost; if the worker JVM crashes the supervisor restarts it and continues.

| Flag | Default | Meaning |
|------|---------|---------|
| `--n-drafts` | _(required)_ | Number of complete drafts to generate. |
| `--set` | _(none → random per draft)_ | Restrict all drafts to one set code; otherwise each draft uses an independently chosen random sealed-legal set. |
| `--agent-mix` | `forge-full:6,forge-r30:1,forge-r100:1` | Probability weights for assigning agents to pod seats. Each of the `pod_size` seats is sampled **independently** from this categorical distribution, normalised across the listed weights (the default gives each seat ≈ 6/8 chance of `forge-full`, 1/8 each of `forge-r30` / `forge-r100`), so pod compositions vary draft-to-draft. The sampled identifier is written into each seat's `agent` field (§ 4.1). Built-in identifiers: `forge-full` = Forge draft AI; `forge-r30` / `forge-r100` = 30% / 100% of that seat's picks replaced by uniform-random legal picks. |
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer used to label finished decks (scores the built deck under either build method). |
| `--build-method` | `picker` | How each seat's 45-card pool is built into a 40-card deck for scoring. `picker` runs `--picker-checkpoint` in one ~5 ms forward (§ 4.2); `greedy` runs `GreedyDeckBuilder`'s SA search — the higher-fidelity but ~1000× slower fallback, selected only when the § 5.3 validation shows the picker diverges from SA. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Picker used when `--build-method picker`; ignored for `greedy`. |
| `--cards-path` | `output/cardsfolder/` | `.npz` embedding cache used by the deck build + scorer. |
| `--output-path` | `output/draft/drafts.jsonl` | Destination JSONL file (§ 4). Records are appended; the file is created if missing. |
| `--resume` | off | Append to existing files and continue toward `--n-drafts`, counting drafts already present. |

The agent mix in each pod serves two purposes (§ 3.4): **opponent diversity**
— weaker agents vary the signals and open colours the strong seats see — and
**critic coverage** — the critic trains on the weaker-agent seats too, so it
learns to value the incoherent pools they reach. Weaker-agent seats are *not*
imitation targets, though; the policy learns only from agents on the
`--imitation-agents` whitelist (§ 5.2). Heavily-random pods are kept a minority
via the default mix, because all-random dynamics (colours wide open, signals
meaningless) are unrealistic and would teach the policy patterns that competent
pods punish.

### 5.2 `train-draft-agent`

Trains the policy + critic jointly on a recorded corpus.

| Flag | Default | Meaning |
|------|---------|---------|
| `--drafts-path` | `output/draft/drafts.jsonl` | The JSONL file of recorded drafts (§ 4); one record per line, each carrying every pick the policy and critic need. |
| `--cards-path` | `output/cardsfolder/` | `.npz` embedding cache. |
| `--d-model` | _(derived: `embedding_dim` + feature widths)_ | Model width. By default `d_model` is the concatenated token width (§ 1.4) and no projection is inserted; a different value inserts a `Linear` from the concatenated width to `d_model`. |
| `--n-layers` | `4` | SAB layers. |
| `--n-heads` | `8` | Attention heads; `d_model` must be divisible by this (fails fast at startup otherwise). |
| `--ff-dim` | `4 × d_model` | Feed-forward width, computed from the resolved `d_model`. |
| `--dropout` | `0.0` | Transformer dropout. |
| `--imitation-weight` | `1.0` | Coefficient on the cross-entropy term. `0` = critic-only run. |
| `--critic-weight` | `1.0` | Coefficient on the critic MSE term. `0` = imitation-only run (the "solidify Rung 1 first" mode). |
| `--imitation-agents` | `forge-full` | Whitelist (comma-separated, or the flag repeated) of agent identifiers whose picks are imitation (policy) targets (§ 3.2). Default `forge-full`: the policy learns only from `forge-full` seats. The critic is unaffected — it trains on all seats regardless. |
| `--lr` | `3e-4` | AdamW learning rate. |
| `--warmup-frac` | `0.05` | Fraction of scheduled steps for linear LR warmup. |
| `--batch-size` | `32` | States per gradient step. |
| `--max-grad-norm` | `1.0` | Per-parameter-group gradient-norm cap. |
| `--epochs` | `100` | Maximum epochs (one epoch = one pass over the training drafts). |
| `--val-fraction` | `0.2` | Draft-disjoint validation fraction (§ 3.3). |
| `--patience` | `10` | Early-stop after this many epochs without validation improvement. |
| `--resume` | _(none)_ | Continue a stopped run; loads weights, optimiser state, epoch counter, best-val metadata. Architecture flags forbidden (inherited from checkpoint). Mutually exclusive with `--checkpoint`. |
| `--checkpoint` | _(none)_ | Bootstrap a fresh run from this checkpoint's weights only (optimiser/epoch/metadata discarded). Architecture flags forbidden. Mutually exclusive with `--resume`. |

Setting `--critic-weight 0` reproduces the recommended build order's Rung 1
(imitation alone) for validating that the policy is solid before the critic is
added; the default jointly trains both heads on the same corpus (Rungs 1–2),
since the two labels are independent and share the encoder.

### 5.3 Picker-vs-SA builder validation (script, not a subcommand)

A one-off diagnostic that decides whether `--build-method picker` is
trustworthy for labeling, or whether a run must fall back to `greedy`. Run once
per picker/scorer checkpoint pair before committing to a large
`generate-draft-data` run. Like the picker spec's cold-start sanity check, this
is a short ad-hoc script (~40 lines), not a maintained CLI subcommand: it runs
a handful of times across the project's life and the CLI surface, tests, and
docs a subcommand would carry are not worth the upkeep.

**What it measures.** Whether the picker's builds rank pools the same way SA
does. The critic consumes *rank-consistent* labels — advantages are value
jumps, so a constant absolute gap between picker and SA builds cancels and is
harmless. The failure that *would* harm the critic is **pool-composition-
dependent** divergence (the picker building well on some pool types and badly
on others), which injects structured label noise and could teach the critic
spurious correlations. The gating metric is therefore rank correlation, not
absolute agreement.

**Procedure.**

1. Sample a few hundred drafted 45-card pools. An existing `drafts.jsonl` is
   the natural source — each seat's 45-card final pool is derivable from the
   record by walking its picks across its three boosters (§ 4.1 conventions),
   which is what makes the script "run on draft event outcomes"; freshly
   generated pools also work.
2. For each pool, build a deck two ways from the same `.npz` cache: the
   one-shot picker (`--picker-checkpoint`) and the SA `GreedyDeckBuilder`.
3. Score both decks with the frozen scorer → `picker_score`, `sa_score` per
   pool.
4. Report:
   - **Spearman rank correlation** of `picker_score` vs `sa_score` across the
     sampled pools — the gating number.
   - The distribution of `sa_score − picker_score` (median + spread) — the
     absolute gap, for interpretation only.
   - Optionally, the per-pool fraction of cards shared between the two builds,
     as a second view on agreement.

**Gating decision.** The threshold is read against the scorer's own noise:
because SA is stochastic, compute the correlation of SA builds with *themselves*
across independent restarts on the same pools as the reference ceiling.

- If the picker tracks SA about as well as SA tracks itself (picker-vs-SA
  rank correlation close to the SA-vs-SA reference) → use `--build-method
  picker`; the absolute gap is absorbed by the critic and the speedup stands.
- If the picker's rank correlation is materially below that reference, or the
  gap is visibly pool-composition-dependent → use `--build-method greedy`, or
  fine-tune the picker on variable smaller pools first (see Out of scope).

## 6. Model artifacts

Checkpoints saved to `models/draft/agent/`:

```
models/draft/agent/
  {timestamp}.pt
  latest.pt
```

Each checkpoint contains:

- `model_state_dict` — trunk + policy head + critic head + the recency
  (`packs_ago`/`pick_ago`) and context (`pack_number`/`pick_number`) embedding
  tables. (Type is a concatenated one-hot, not a learned table.)
- `config` — architecture (`d_model`, `n_layers`, `n_heads`, `ff_dim`,
  `dropout`, derived `embedding_dim`) and the booster size `P` (which sizes the
  `pick_number` and `pick_ago` tables) resolved from the training corpus.
- `epoch`, `best_val_loss`, training metadata.

No encoder weights are embedded (Phase A only): the drafter is paired at use
time with whichever encoder's `.npz` cache produced its input width, exactly
like the picker.

# Design rationale and rejected alternatives

This records the design discussion, including options not chosen, so the
decision shapes survive future iterations. Caveats from the originating
discussion that do not apply given this project's context have been dropped.

## Two-stage plan (imitation → RL)

Accepted. Stage 1 (imitation) is supervised, stable, and cheap, and yields
most of a working drafter. Stage 2 (RL self-play) refines it. Generation 1 is
Stage 1 plus the critic that Stage 2 will need; the RL itself is generation 2.
This is the "bootstrapping ladder": climb the cheap, certain rungs first
(imitation, then critic on the same data) so the actor and critic are
pre-aligned on one play distribution before any RL.

## State representation

Accepted: the typed token sequence (`POOL` / `PACK` / `PASSED` / `TAKEN` /
`CONTEXT`) over reused card embeddings. Open-colour reading is left **implicit**
— the `PASSED`/`TAKEN` tokens plus pack/pick counters carry the signal. `TAKEN`
records *observed* opponent picks read off the wheel (§ 1.1), which is factual
observation, not a learned model of any neighbour, so the policy still builds
no explicit opponent model. An auxiliary head predicting each neighbour's
archetype was considered and deferred to generation 2 (it needs self-play
ground truth to supervise).

**Rejected — pick-as-attention-op.** Treating the pick distribution *as* the
attention weights between a pool/context query and the pack keys is
parameter-efficient and interpretable, but forces the policy through a single
attention bottleneck that struggles to represent context-dependent value
("good *because* I lack fixing"). The per-card head over a jointly-attended
trunk (the picker's validated pattern) is more expressive at negligible extra
cost. The attention formulation remains available as an ablation, not the
default.

## Pool evaluator for a dense early signal

The originating discussion weighed four ways to evaluate an incomplete pool
for per-pick shaping:

- **Opt 1 — separate pool-quality regression model** (partial pool → final
  win rate).
- **Opt 2 — synthetic completion** (sample completions to 45 cards, run the
  sealed model, average). Rejected: expensive and noisy early.
- **Opt 3 — mask-trained sealed model** (retrain to accept partial pools).
- **Opt 4 — terminal reward only** (no dense signal).

Resolution: the **learned critic** (§ 2 critic head, trained by Monte-Carlo
regression) subsumes the pool-evaluator role and gives a value from pick 1
without touching the sealed scorer. It is effectively Opt 1, but as a head on
the shared trunk rather than a separate model, so the policy and the value
estimate share representation. The sealed scorer cannot itself serve as the
dense critic because it needs a full ~23-spell deck and so only becomes
meaningful late in the draft; the critic learns to predict that eventual score
from any state instead.

**Greedy pool-maximisation is the wrong decision rule.** Picking the card that
maximises current pool quality each step is a greedy climb that gets stuck:
it will not lower pool quality temporarily to switch out of a contested lane,
it undervalues speculative enablers and fixing, it ignores signalling/hate
picks, and it misses that deck quality is lumpy (a coherent 18-card archetype
beats 23 scattered strong cards). The objective is **expected final return**,
not current pool value; the critic is a shaping signal, never the objective.

## Builder for critic labels

The picker is the default label-builder (§ 4.2) over the SA builder purely for
throughput — one build per seat per draft, ~5 ms vs ~5 s, which at corpus scale
is the difference between labeling taking minutes and taking days. The picker
is out-of-distribution on a 45-card pool (trained on ~60–90), but that matters
far less for *labeling* than it would for production deckbuilding, for two
reasons. First, the critic needs only **rank-consistent** labels — a uniformly
slightly-worse builder is a near-monotone transform of pool value, which the
advantage (a value jump) is invariant to. Second, on a *focused* drafted pool
(already ~2 colours, mostly playables) the deck score is relatively
**builder-insensitive**: the high-impact decisions (play the bombs, stay in
colour) are easy and any decent ranker gets them, while the decisions the
picker might flub (the 23rd card, a marginal splash) move the score least — so
its degradation translates into small score error here, plausibly smaller than
on a wide sealed pool. The residual risk is *structured* (pool-composition-
dependent) divergence, which is exactly what the § 5.3 validation gates on,
with SA as the higher-fidelity fallback. It's the same continuation-matching
idea that recurs elsewhere (§ 3.4): label with whatever builder will actually
materialise the agent's decks downstream.

## Reward / fitness

Accepted: **pod-relative** reward — a seat's `deck_score` minus the **mean of
the other pod seats' `deck_score`s** (leave-one-out, so the seat doesn't
subtract itself), computed at load time from § 4.2's `deck_score` field. It is the only critic
target — there is no absolute-reward option. Subtracting a pod reference is a
variance-reducing baseline, and in draft it is more than that: drafters fight
over the same cards, so the pod-relative reward has a genuine two-sided
gradient (improving yours degrades theirs). The clinching case is the **hate
pick** — slightly lowering your own deck to deny a key card to a competitor.
Under absolute reward that looks bad (your `deck_score` drops); under
pod-relative reward it is correctly valued positive whenever the competitor's
deck loses more than yours does. The objective the gameplay actually optimises
is "win the pod," not "maximise my deck's absolute quality," and pod-relative
reward is what aligns the gradient with that objective.

- **Aggregate over opponents with the mean, not the max.** Max/argmax is
  high-variance and flips discontinuously through a single opponent; in
  self-play it makes policies pile onto whoever is ahead. The pod mean spreads
  denial across the field and is stable.
- **Aggregation encodes the goal (generation 2).** Mean of `P(beat i)` =
  expected match win rate; product of `P(beat i)` = probability of beating
  everyone, whose gradient saturates and concentrates on the closest
  matchups. Linear margin against the top deck rewards overkill and is blind
  to a second shaky matchup — not equivalent to "beat everyone."
- **Calibration is cheap here.** The scorer is a Bradley-Terry model
  (`binary_cross_entropy_with_logits` on the score delta), so scores are
  already logits: `sigmoid((S_A − S_B)/T)` is a win probability up to one
  temperature `T`, fit in an afternoon on held-out matches. Generation 1's
  critic regresses the pod-relative reward in raw score space; the
  temperature fit is what lets generation 2 switch to the product objective if
  the goal becomes robustly beating the whole pod rather than crushing one
  deck. Transitivity of the scalar score is treated as a working assumption,
  supported by this project's finding that per-pool score deltas track actual
  win-rate deltas.

## Credit assignment (generation 2)

The discussion's recursive-forking ("vine") idea and the root-level GRPO/RLOO
idea are recognised as the **same baseline trick at different granularities**
(pod-level, pick-level, trajectory-level) and compose. The textbook answer to
the heterogeneous per-pick importance they target is a **learned value
function + GAE(λ)**: dense, cheap per-step signal at every pick, with λ near 1
leaning on Monte-Carlo. That is the generation-2 workhorse, and it is exactly
why generation 1 builds the critic. **Forking** stays surgical — concentrated
in pack 2 where picks are pivotal and the critic is least trustworthy, with
common random numbers (fixed unopened packs + opponent seeds) across paired
branches — used to sharpen and audit the critic, not as the main estimator.
Root-only GRPO (one scalar advantage per full draft) remains the trivial
fallback. None of this is built in generation 1.

## Critic training

Accepted: **Monte-Carlo regression** — label every state with its draft's
final deck score and fit by MSE. Stable, stationary target, cannot really
fail, and reuses the same drafts as the imitation head (one dataset, two heads
on the shared encoder).

- **Rejected for now — TD / bootstrapping** (regress `V(s_t)` toward
  `V(s_{t+1}) + reward`): lower variance but biased with a moving target;
  not worth the instability on a solo / single-GPU budget. GAE(λ→1) leans on
  MC anyway; add TD only if measured variance forces it.
- **Optional — warm-start from human/17lands data**: pretrain the critic by
  regressing intermediate pool-states against final results. Cheap if the data
  is available; not required.

**Rejected — critic-only greedy actor** ("try each card, keep the biggest
value jump"). A fine bootstrap but not the endpoint, for three reasons:
(1) argmax over an imperfect critic actively hunts its largest overestimate;
(2) one critic pass per candidate per pick is far costlier than one policy
forward, compounding over millions of training picks; (3) the critic's value
assumes a continuation policy — if the greedy actor differs, it drifts
off-distribution and the guidance degrades. The resolution is **actor-critic**:
the policy produces picks and the rollouts the critic learns from, so the
critic's assumed continuation stays locked to the actual one. Generation 1
already trains both on the same body; generation 2 closes the loop.

## Agent-mixed data, and why the critic isn't agent-conditioned

Mixing agents in the pod (§ 5.1) does two jobs: opponent diversity for the
competent seats to read, and — since the **critic trains on all seats** (§ 3.4)
— coverage of the incoherent pools weaker agents reach, which anchors the low
end of the value scale. An earlier design also fed a per-seat agent tag to the
model so the critic could learn an agent-aware `V` and be queried with the
policy's tag for `V^π`. We dropped the *tag* — it's an artefact of the synthetic
data, and the actor never needs it (it always plays as the same agent) — but
kept training the critic broadly. The cost is that a weaker agent's label
reflects a weak continuation, so generation-1's `V` is mildly biased; we accept
that (harmless for a baseline, roughly correct for genuinely-bad pools, § 3.4)
rather than reintroducing the tag, and generation 2 removes it by regressing on
the policy's own rollouts. Opponent-seat randomness stays near pure upside and
is used generously; own-seat randomness and all-random pods are kept a
minority, so the model is not trained mostly on unrealistic dynamics.

## Self-play regeneration (generation 2)

The out-of-distribution fix is to regenerate data from a mix of the current
policy + Forge + laddered/random bots and retrain a next generation on old +
new data concatenated. Mixing preserves opponent diversity; concatenation
prevents catastrophic forgetting and bad-region coverage loss. A steady
minority of dumb/random bots is kept every generation, because as the policy
improves it stops visiting incoherent states and that coverage would otherwise
evaporate from fresh data. A frozen held-out set (carved out now) gives a
cross-generation yardstick. All deferred to generation 2.

# Evaluation

Generation 1 is evaluated offline; the definitive actor scoreboard (drafted
decks scored / played head-to-head against Forge) belongs to generation 2,
once our model can draft live.

**Policy (imitation).** Top-1 and top-3 pick-match accuracy against held-out
Forge picks, sliced by `pack_number` / `pick_number`. Early-pack accuracy is
expected to be lower (more ties between comparable cards); late-pack accuracy
high (forced picks).

**Critic.** Monte-Carlo regression error on held-out drafts, ranked on a
shared held-out set across model variants so shared luck cancels (absolute
error is large in isolation because a single future is the mean plus luck).
Slice by draft stage and confirm the expected shape: humble/flat predictions
early (≈ population mean at pack 1 pick 1, since the pick barely determines the
deck), spreading and sharpening late. Over-confident early predictions signal
overfitting to noise. Ranking checks, which matter more than absolute values
because only the per-pick value *jumps* are used downstream: a competent-
continued state should be valued above the same state with a random
continuation (competent-agent ordering), and where pack-2 forks exist the
critic's jump should agree with the branch that ends better.

# Out of scope

- **Live integration of our policy as a Forge draft seat** (self-play). The
  generation-1 corpus comes only from Forge's own draft AI and its weaker
  variants. (Generation 2.)
- **Reinforcement-learning fine-tuning** of the policy (actor-critic, GAE,
  REINFORCE/PPO, the pod-relative reward as an optimisation objective,
  calibration temperature applied to the reward). (Generation 2.)
- **Surgical pack-2 forking** with common random numbers, and any vine-style
  advantage estimation. (Generation 2.)
- **Self-play data regeneration** across generations. (Generation 2.)
- **Auxiliary opponent-archetype prediction head.** (Generation 2.)
- **Encoder fine-tuning** alongside the drafter (Phase B analogue).
- **Fine-tuning the picker on variable / smaller pools** (masked-pool
  augmentation) to make 45-card builds in-distribution. Only pursued if the
  § 5.3 validation rejects the picker as the label-builder; the SA fallback
  covers that case without it.
- **An inference / play-out CLI.** Generation 1 ships data generation and
  training only; the policy is exercised through the offline evaluation above
  until live drafting exists.

# Glossary

- **POMDP** — partially observable decision process; the agent acts on its
  observation history (its belief about hidden state).
- **Critic / value function `V(s)`** — predicted expected final return from a
  state; here, the predicted final deck score.
- **Advantage** — how much better an action is than the state's average; the
  per-pick value jump `V(s_{t+1}) − V(s_t)`.
- **GAE(λ)** — generalized advantage estimation; λ interpolates Monte-Carlo
  (unbiased, high variance) and TD/critic (biased, low variance).
- **Baseline** — a reference subtracted from reward to cut variance (the pod
  mean here).
- **GRPO / RLOO** — sample a group from a state; advantage = reward minus the
  group mean.
- **Vine (TRPO)** — branch at states and compare rollout returns for a
  low-variance advantage.
- **CRN** — common random numbers; hold the future fixed across compared
  branches.
- **`V^π`** — the value assuming the current policy continues (the target RL
  actually wants); in generation 1 only loosely approximated (the critic
  regresses on a mix of Forge continuations), and reached directly by
  generation 2's on-policy training.
