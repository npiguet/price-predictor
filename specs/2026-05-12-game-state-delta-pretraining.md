# MTG Forge Sealed Deck Builder AI — Ability Embedding Upgrade Specification

## Overview

This document specifies an upgrade to the existing MTG Forge sealed deck builder. The current deck builder operates on
per-card embeddings represented as single vectors trained on card-level metrics. This upgrade replaces that
representation with **ability-level embeddings** trained on game state deltas, giving the model a richer and more
principled understanding of what each card actually does in play.

The core change is that a card is no longer represented as a single vector but as a **mini-sequence** of ability
embeddings, allowing the pool transformer to reason about individual card effects and cross-card synergies rather than
opaque card-level summaries.

---

## 1. Goals

### Primary Goal

Improve sealed deck win rate over the current per-card-embedding baseline by replacing card-level vectors with
ability-level representations trained on game state deltas.

### Secondary Goals

- Produce ability embeddings that generalize to cards not seen during training, including new sets
- Handle keyword expansion so that new keyword mechanics receive reasonable embeddings on first sight without requiring
  retraining
- Validate that ability-level representations capture strategically meaningful card effects

---

## 2. Motivation for the Change

The current single-vector card embedding conflates everything a card does into one representation. This creates two
problems:

**No ability-level generalization.** A card with a novel combination of two well-understood abilities requires its own
training examples from scratch. With ability-level embeddings, the model already understands each ability independently
and can reason about their combination.

**No interaction signal.** The pool transformer cannot distinguish between two cards that share an ability, because
their card vectors are unrelated even if the underlying ability is identical. With shared ability embeddings, cards that
share abilities will attend to each other more strongly, surfacing synergies the current model cannot detect.

---

## 3. Card Representation

### 3.1 The Atomic Unit: Abilities

The fundamental unit of card representation is the **ability**, not the card. Each ability is a sequence of tokens in
the existing MTG-specific vocabulary describing a single game rule or effect. Cards are represented as sequences of
ability embeddings rather than a single card-level vector.

This choice is motivated by:

- Abilities are the natural resolution granularity in the MTG rules engine — the engine resolves one ability at a time,
  making ability-level game state deltas the natural training signal
- Sharing ability representations across cards enables generalization — a card with a novel combination of known
  abilities can be reasoned about without requiring its own training examples
- New keyword mechanics can be handled by expanding them to rules text, rather than requiring a new vocabulary entry

### 3.2 Keyword Expansion

All keyword abilities are expanded to their full rules text before encoding. "Flying" becomes "This creature can only be
blocked by creatures with flying or reach." This ensures:

- New keywords introduced in future sets receive reasonable embeddings on first sight, because their expanded text is
  structurally similar to semantically related existing abilities
- The model reasons from rules semantics rather than arbitrary keyword tokens
- No keyword lookup table is required

MTG Forge already contains this expansion logic. The expansion should be sourced from the rules engine, not from printed
reminder text on cards.

### 3.3 Ability Tokenization and Embedding

Each expanded ability text is tokenized using the **existing MTG-specific tokenizer** (approximately 5,000 tokens
covering MTG vocabulary: card types, subtypes, mana symbols, keyword terms, rules language, etc.).

Token embeddings are initialized as **random vectors** and trained from scratch during ability embedding pretraining (
Section 5.1). No pretrained language model is required or assumed.

The ability encoder is a small transformer that takes a token sequence and produces a single fixed-size ability
embedding via a `[CLS]`-style aggregation token:

```
[CLS] [token_1] [token_2] ... [token_N]  →  transformer  →  ability_embedding = output[0]
```

The ability encoder is run **offline once per unique ability text** after training, and the resulting embeddings are
cached to disk (see Section 5.2). The ability encoder never runs at inference time.

### 3.4 Card Representation as a Sequence

A card is not represented as a single vector but as a **mini-sequence** of tokens:

```
card_sequence = [CARD_token, ability1_emb, ability2_emb, ..., abilityN_emb]
shape: [1 + N_abilities, d]
```

The `[CARD]` delimiter token carries the existing structured per-card features projected to model dimension. Ability
tokens carry ability embeddings from the trained ability encoder.

This design:

- Eliminates the need for a separate card encoder with its own training signal
- Allows the pool transformer to attend across card boundaries and detect cross-card synergies
- Produces implicit card embeddings as the `[CARD]` token output positions after the pool transformer runs — no explicit
  aggregation step required
- Is a drop-in replacement for the current single-vector card representation at the pool transformer input level

### 3.5 The [CARD] Token

The `[CARD]` token input is a learned delimiter embedding concatenated with the existing structured per-card feature
vector, projected to model dimension via a learned linear layer. The structured features are unchanged from the current
implementation and are not redefined here.

Numeric structured features (power, toughness, mana cost pips) remain as raw integer values rather than passed through
transformer attention, so that arithmetic comparisons remain reliable in the downstream MLP layers.

### 3.6 Creature Type Representation

With approximately 300 creature subtypes, multi-hot encoding is too sparse. Because creature and card types are already
part of the MTG tokenizer vocabulary, each subtype has a token embedding in the shared embedding table. A card's
creature type representation is the mean of its subtype token embeddings:

```python
card_type_repr = token_embedding_table[card.subtype_token_ids].mean(dim=0)
```

This is concatenated onto the `[CARD]` token alongside the other structured features. Because creature type tokens are
part of the shared vocabulary, their embeddings are trained jointly with the rest of the ability encoder — a card's type
tokens and the type-referencing tokens in its ability text share the same embedding table, allowing the model to
naturally align tribal synergies without any special handling.

### 3.7 Positional Encoding

The pool sequence has two distinct ordering requirements that pull in opposite directions:

- **Across cards:** order is irrelevant — the pool is a set, not a sequence. This is the same property the current set
  transformer preserves.
- **Within a card:** order is meaningful for certain card types (Sagas, modal double-faced cards, Adventure cards).

Standard positional encodings cannot satisfy both requirements simultaneously — they encode absolute position within the
full sequence, which would impose a spurious ordering on cards relative to each other. The solution is **local
positional encoding**: position indices are reset to zero at each `[CARD]` token and count up only within that card's
ability slots.

```
[CARD] [ability] [ability] [ability]  [CARD] [ability] [ability]  [CARD] [ability]
  pos:0  pos:1    pos:2    pos:3        pos:0  pos:1    pos:2        pos:0  pos:1
```

The `[CARD]` token always receives position 0. Abilities are numbered from position 1 within their card. The same
position indices are reused across every card, so the transformer has no signal about which card appeared earlier in the
sequence. Cross-card attention is purely content-based — permutation invariant over cards, exactly as in the current set
transformer.

Implementation replaces the standard `torch.arange(sequence_length)` position id construction with a local version:

```python
def local_position_ids(card_lengths):
    # card_lengths: list of (1 + n_abilities) per card, in pool order
    ids = []
    for length in card_lengths:
        ids.extend(range(length))  # resets to 0 at each new card
    return torch.tensor(ids)
```

**What position encodes per card type:**

For most cards the positional signal is weak — abilities are unordered and the model will learn to treat position 1+ as
interchangeable ability slots. The `[CARD]` token at position 0 is always structurally distinct. Position carries
genuine meaning in three cases:

- **Sagas** — chapter I, II, III must be serialized in chapter order. Positions 1, 2, 3 directly encode the chapter
  sequence. The ability encoder can learn from game state delta data that abilities at earlier positions fire before
  abilities at later positions.
- **Multi-face cards (modal double-faced cards, Adventure cards, split cards, flip cards)** — face boundaries are marked
  with an explicit `[ALTERNATE]` separator token, mirroring the `alternateType` separator already present in Forge's
  card description format. The `[ALTERNATE]` token carries a `type` property drawn from Forge's existing `alternateType`
  vocabulary (e.g. `transform`, `adventure`, `split`, `flip`), encoded as a small learned embedding concatenated onto
  the delimiter:

```
[CARD] [ability] [ability] [ALTERNATE, type=transform] [ability] [ability]
  pos:0  pos:1    pos:2         pos:3                    pos:4    pos:5
```

Position indices continue monotonically across the full card — the only reset is at `[CARD]`, which is what enforces
permutation invariance across cards. Position does **not** reset after `[ALTERNATE]`: if it did, face 2 abilities would
receive the same position indices as face 1 abilities, and the model would lose the signal distinguishing which face an
ability belongs to. With continuing positions, abilities on face 2 always have higher indices than abilities on face 1,
and the `[ALTERNATE]` token's position implicitly encodes how many abilities the first face had.

This is preferable to encoding face boundaries implicitly via position or via a flag on the `[CARD]` token because it
mirrors the source data format directly, makes the boundary unambiguous to the transformer, and carries the face type as
explicit signal rather than requiring the model to infer it from context.

**Vanilla creatures** (no abilities) are the degenerate case: the card sequence is just `[CARD]` at position 0 with no
ability tokens. This is self-consistent — the structured features on the `[CARD]` token carry all the information for
that card.

This positional encoding design is a deliberate departure from standard practice and should be noted in code comments,
since the recycled position indices will look unusual to anyone unfamiliar with the intent.

#### Encoding Scheme Options

Two encoding schemes are compatible with local positions, with different tradeoffs:

**Option A: Additive local positional embeddings (simpler)**

A positional embedding vector is added to each token embedding before the first transformer layer:

```python
input_to_transformer = token_embedding + positional_embedding[local_position_id]
```

Positional information is baked into the representation from the start and flows through all layers via the residual
stream. The local position ids ensure that the same positional vector is added to ability slot 1 on every card
regardless of where that card appears in the pool sequence. This is simple to implement and sufficient for most cases —
the recycled indices already prevent the transformer from inferring cross-card order.

The limitation is that there is no clean way to express "these two tokens have no positional relationship." Two tokens
on different cards are assigned local positions 0 and 0 (or whatever their respective local indices are), which
technically implies a distance of zero between them — a weak but nonzero positional signal that is semantically
meaningless.

**Option B: Relative positional encodings (more principled)**

Rather than modifying input embeddings, positional information is encoded inside the attention operation itself as a
bias on the attention score between token pairs:

```
attention_score(i, j) = content_score(query_i, key_j) + position_bias(relative_distance(i, j))
```

For tokens within the same card, `relative_distance` is their local index difference (−N to +N). For tokens on different
cards, `relative_distance` is set to a special `NO_POSITION` value that adds zero bias to the attention score.
Cross-card attention then becomes purely content-based with no positional contribution at all — a cleaner expression of
permutation invariance than Option A.

RoPE (Rotary Position Embedding, Su et al., 2021) is the current standard implementation of relative positional
encodings and is used in most recent large language models (LLaMA, Mistral, etc.). Efficient implementations are widely
available. The tradeoff versus Option A is implementation complexity — modifying the attention kernel is more involved
than modifying input embeddings.

**Recommendation:** Option A is the pragmatic starting point. Option B is the architecturally cleaner solution and worth
adopting if cross-card positional bleed becomes a measurable problem, or if a RoPE implementation is already available
in the project's transformer stack.

---

## 4. Data Collection

### 4.1 What to Collect

One new category of data is required for this upgrade:

**Ability-level game state deltas** — used to pretrain ability embeddings (Section 5.1)

Sealed pool and deck example collection is already implemented and not described here.

### 4.2 Ability-Level Game State Delta Collection

For each ability resolution during a simulated game, record:

```
{
  ability_id:       unique identifier for the ability text
  card_id:          source card
  game_id:          game instance
  turn:             turn number
  controller:       which player controls the ability

  state_before:     GameStateSnapshot (see 4.3)
  state_after:      GameStateSnapshot (see 4.3)

  delta:            DeltaVector (see 4.4)
}
```

Record **immediately before** the ability begins resolving and **immediately after** it finishes. For triggered
abilities, capture the state at the moment the trigger resolves on the stack, not when it triggered.

### 4.3 Game State Snapshot

A snapshot captures all strategically relevant state at a point in time. Since ability embeddings are trained from the *
*controller's perspective**, the snapshot is player-relative (self vs opponent):

```
GameStateSnapshot {
  # Life totals
  self_life:                    int
  opponent_life:                int

  # Hand
  self_hand_size:               int

  # Battlefield — own side
  self_creatures:               List[CreatureState]
  self_lands:                   int
  self_untapped_lands:          int
  self_other_permanents:        int

  # Battlefield — opponent side
  opponent_creatures:           List[CreatureState]
  opponent_lands:               int
  opponent_other_permanents:    int

  # Other zones
  self_graveyard_size:          int
  opponent_graveyard_size:      int
  self_cards_in_exile:          int
  opponent_cards_in_exile:      int

  # Stack
  spells_on_stack:              int

  # Turn context
  turn_number:                  int
  is_my_turn:                   bool
  current_phase:                PhaseEnum
}

CreatureState {
  power:                        int
  toughness:                    int
  is_tapped:                    bool
  has_summoning_sickness:       bool
  keywords:                     List[KeywordEnum]
}
```

**What to omit from snapshots:** Card identities in graveyard or exile (too sparse), exact hand contents (not observable
to opponent), library contents. The goal is aggregate state features that capture strategic position, not full game
reconstruction.

### 4.4 Delta Vector

The delta is computed as state_after − state_before. Represent it as a fixed-size vector of named differences:

```
DeltaVector {
  # Life
  self_life_delta:                  int
  opponent_life_delta:              int

  # Card counts
  self_hand_delta:                  int
  opponent_hand_delta:              int
  self_graveyard_delta:             int
  opponent_graveyard_delta:         int
  self_exile_delta:                 int
  opponent_exile_delta:             int

  # Battlefield — creatures
  self_creatures_delta:             int
  opponent_creatures_delta:         int
  self_total_power_delta:           int
  self_total_toughness_delta:       int
  opponent_total_power_delta:       int
  opponent_total_toughness_delta:   int

  # Battlefield — other permanents
  self_lands_delta:                 int
  opponent_lands_delta:             int
  self_other_permanents_delta:      int
  opponent_other_permanents_delta:  int

  # Stack
  stack_size_delta:                 int

  # Tempo proxy
  opponent_mana_spent_wasted:       int
}
```

### 4.5 Forge Integration Hooks Required

The following callbacks must be added to the Forge rules engine if not already present:

- **Pre-ability-resolution callback:** emit game state snapshot immediately before an ability begins resolving
- **Post-ability-resolution callback:** emit game state snapshot immediately after an ability finishes resolving

All required state is already tracked by the rules engine; these hooks are instrumentation only.

---

## 5. Training Pipeline

Training proceeds in two sequential stages. Stage 1 is new. Stage 2 is the existing deck builder training loop with a
modified input format.

```
Stage 1: Ability Embedding Pretraining
            ↓ produces: trained ability encoder + cached ability embeddings
Stage 2: Deck Builder Training (existing pipeline, modified input)
            ↓ produces: improved pool transformer + card selection head
```

### 5.1 Stage 1 — Ability Embedding Pretraining

**Goal:** Train the ability encoder so that ability embeddings capture strategic game effects.

**Training task:** Given an ability's token sequence, predict the game state delta produced when that ability resolves.

```python
# For each (ability_tokens, delta) pair in the collected dataset:
ability_embedding = ability_encoder(ability_tokens)  # [d]
predicted_delta = delta_head(ability_embedding)  # [len(DeltaVector)]
loss = mse(predicted_delta, actual_delta)
```

The delta head is a small MLP (2–3 layers) projecting from embedding dimension to delta vector dimension. It is
discarded after pretraining — only the ability encoder weights are kept.

**Data:** Ability-level game state delta records from Section 4.2. Each unique ability text is one embedding; the same
ability appearing on multiple cards or firing multiple times across games contributes multiple training examples,
naturally weighting embeddings toward frequently relevant effects.

**Training notes:**

- Normalize delta vectors per feature to zero mean and unit variance before training
- Abilities that appear rarely (fewer than ~50 observations) will have noisy embeddings; consider a minimum observation
  threshold before including an ability in the validation set

### 5.2 Ability Embedding Caching

After Stage 1, run every card's expanded abilities through the trained ability encoder and save embeddings to disk keyed
by a hash of the ability token sequence:

```python
# Run once after Stage 1, before Stage 2
ability_cache = {}
for card in all_cards:
    for ability in card.expanded_abilities:
        key = hash(ability.tokens)
        if key not in ability_cache:
            ability_cache[key] = ability_encoder(ability.tokens)

torch.save(ability_cache, "ability_embeddings.pt")
```

The ability encoder is never invoked after this point.

### 5.3 Stage 2 — Deck Builder Training (Modified Input)

The existing deck builder training loop is unchanged. The only modification is the format of the input fed to the pool
transformer.

**Current input format:**

```
pool_input = [card_vector_1, card_vector_2, ..., card_vector_84]
shape: [84, d]
```

One vector per card, cards treated as independent items in a set.

**New input format:**

```
pool_input = [CARD_1, ability1, ability2, CARD_2, ability1, CARD_3, ability1, ability2, ability3, ...]
shape: [84 + total_abilities_in_pool, d]
```

A flat sequence of interleaved `[CARD]` delimiter tokens and ability embedding tokens. The pool transformer processes
this single sequence, attending freely within and across card boundaries.

Typical sequence length: 84 cards × (1 delimiter + ~3 average abilities) ≈ 336 tokens. This is well within standard
transformer context lengths and does not require architectural changes to the pool transformer beyond adjusting the
input projection layer.

**Why this improves cross-card synergy detection:** In the current format, the pool transformer can only detect that
card A and card B are both present in the pool. In the new format, it can detect that ability X on card A shares an
embedding with ability X on card B, or that effect text referencing a creature type uses the same token embedding as
that type appearing in another card's type line. Synergies that were previously invisible become directly visible
through shared embedding structure.

---

## 6. Evaluation

### 6.1 Ability Embedding Quality

Before starting Stage 2, validate that the ability embeddings are meaningful. This is important because a poorly trained
ability encoder will produce worse deck builder performance than the current card-level baseline, and diagnosing that
failure is much easier before committing to Stage 2 training.

#### Nearest Neighbor Inspection

The most direct check is manual inspection of nearest neighbors in embedding space. For a sample of abilities, retrieve
the top-K most similar abilities by cosine similarity and verify the results make intuitive sense:

```python
def nearest_abilities(query_text, k=10):
    query_emb = ability_cache[hash(tokenize(query_text))]
    similarities = cosine_similarity(query_emb, all_ability_embeddings)
    return top_k(similarities, k)
```

Expected results:

- "Deal 3 damage to target creature" → nearest neighbors include "Deal 2 damage to any target", "Deal 4 damage to target
  creature or planeswalker", not "Draw a card"
- "Flying" (expanded) → nearest neighbors include "Reach", "Can't be blocked except by creatures with flying or reach",
  not "Trample"
- "At the beginning of your upkeep, draw a card" → nearest neighbors include other upkeep draw triggers, not damage
  spells

Failure modes to watch for: embeddings that cluster primarily by syntactic structure rather than effect (all "At the
beginning of your upkeep..." abilities cluster together regardless of their actual effect), or near-uniform embeddings
indicating encoder collapse.

#### Delta Prediction Held-Out Loss

Keep 10–20% of ability resolution events as a held-out validation set during Stage 1. Track validation MSE per delta
dimension separately:

```
self_life_delta MSE:           should be low for damage/lifegain abilities
opponent_creatures_delta MSE:  should be low for removal abilities
self_hand_delta MSE:           should be low for draw abilities
```

If MSE is no better than predicting the mean delta across all abilities, the encoder is not learning effect-specific
representations and Stage 2 should not proceed.

#### Embedding Space Visualization

Visualize a sample of ability embeddings using **UMAP** reduced to 2D, colored by manually assigned effect category:

```python
import umap

reducer = umap.UMAP(n_components=2, metric='cosine')
coords = reducer.fit_transform(ability_embedding_matrix)
```

Color points by effect category: damage, draw, removal, token generation, pump, ramp, counterspell, lifegain. A
well-trained embedding space will show these categories forming loose clusters with meaningful gradients between them —
damage and removal should be adjacent, not interleaved with draw effects.

UMAP is preferred over t-SNE for this task because it better preserves global structure (inter-cluster relationships) in
addition to local structure (intra-cluster tightness), and scales more efficiently to larger embedding sets.

#### Delta Correlation per Ability Class

A more quantitative cluster check: group abilities by their dominant delta dimension (the delta dimension with the
largest mean absolute value across observations) and measure whether abilities in the same group are closer in embedding
space than abilities across groups:

```python
# For each ability, identify its dominant effect dimension
dominant_effect = argmax(abs(mean_delta_across_observations))

# Measure cluster separation
intra_class_sim = mean
cosine
similarity
between
abilities
sharing
dominant_effect
inter_class_sim = mean
cosine
similarity
between
abilities
with different dominant_effect
separation_ratio = intra_class_sim / inter_class_sim
```

A ratio above 1.5 indicates the embeddings are capturing effect categories reliably. Below 1.1 suggests the encoder is
not differentiating effect types and should be retrained or debugged before proceeding.

#### Implicit Card Embedding Sanity Check

Compute implicit card embeddings as the mean of each card's ability embeddings and run the same nearest neighbor check
at the card level. Verify that functional analogues are near each other in embedding space (Lightning Bolt and Shock,
Llanowar Elves and Elvish Mystic) and that cards with no functional relationship are not.

This serves as an integration test of the full ability→card representation pipeline before committing to Stage 2.

### 6.2 Deck Builder Quality

Primary metric: **win rate against the Forge AI** in simulated sealed tournaments (best of 3, Swiss rounds), compared to
the current per-card-embedding baseline.

Secondary metrics:

- Color consistency (deck plays ≤ 2 colors in >90% of outputs)
- Mana curve distribution vs the current baseline and vs human expert sealed decks
- Bomb inclusion rate (does the model correctly identify and prioritize high-impact cards)

---

## 7. References

### Core Architecture

**BERT** (Devlin et al., 2018)
The `[CLS]` token aggregation pattern used by the ability encoder and by `[CARD]` delimiter tokens in the pool sequence.
https://arxiv.org/abs/1810.04805

**Pointer Networks** (Vinyals et al., 2015)
The card selection head's attention-over-pool mechanism, unchanged from the current implementation.
https://arxiv.org/abs/1506.03134

**Hierarchical Attention Networks** (Yang et al., 2016)
The conceptual motivation for ability-level representation, and prior art for flattening a card/ability hierarchy into a
single sequence with delimiter tokens.
https://aclanthology.org/N16-1174/

**Relative Position Encodings** (Shaw et al., 2018)
Original formulation of encoding position as a bias on attention scores rather than an additive input embedding; the
conceptual foundation for Option B in section 3.7.
https://arxiv.org/abs/1803.02155

**RoPE: Rotary Position Embedding** (Su et al., 2021)
The current standard implementation of relative positional encodings, used in LLaMA, Mistral, and most recent LLMs. The
recommended implementation if Option B is adopted.
https://arxiv.org/abs/2104.09864

**Graphformer** (Ying et al., 2021)
Transformer architecture for graphs; encodes position locally within node neighborhoods while treating the set of nodes
as unordered. The closest published analogue to the local positional encoding scheme in section 3.7.
https://arxiv.org/abs/2106.05234

### Embedding Training

**Word2Vec** (Mikolov et al., 2013)
The general principle of training embeddings from context signals; foundational for understanding why game state deltas
are a strong training signal for ability embeddings.
https://arxiv.org/abs/1301.3666

**Item2Vec** (Barkan & Koenigstein, 2016)
Extension of Word2Vec to item co-occurrence in sessions; directly analogous to training ability embeddings from deck
co-occurrence as a supplementary signal.
https://arxiv.org/abs/1603.04259

### Embedding Evaluation

**UMAP: Uniform Manifold Approximation and Projection** (McInnes et al., 2018)
Dimensionality reduction for embedding space visualization; preferred over t-SNE for larger embedding sets due to better
global structure preservation and faster runtime.
https://arxiv.org/abs/1802.03426

### Card Game AI and Limited Data

**17lands.com**
Large public dataset of MTG Limited game data including ability resolution frequencies, deck lists, and win rates.
Useful supplementary source for ability delta statistics and deck builder training labels.
https://www.17lands.com

**ML approaches to MTG draft** (various, arxiv)
Directly related work on card evaluation and selection in limited formats.
https://arxiv.org/search/?searchtype=all&query=magic+the+gathering+draft