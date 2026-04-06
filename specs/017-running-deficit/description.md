# Goal

Improve Stage 2 sealed training by supplying the pool transformer with the information it
actually needs to optimize the mana-score reward: a per-color running deficit representing how
many more mana sources the current partial deck still needs.

Two changes are bundled together because they stem from the same insight about transformer
limitations:

1. **Running mana deficit features in the pool tensor**: At each pick step the pool slot
   already has 6 reserved zero-padding dimensions. Fill those with the current per-color
   deficit — (pip demand from spells picked so far) − (mana supply from lands picked so far)
   — so the transformer can see it directly instead of inferring it from pick history.

2. **Remove redundant auxiliary heads from the card encoder**: The pip count and mana value
   auxiliary heads added in feature 015 teach the transformer to reconstruct values that are
   already appended as explicit features to the card embedding. Keeping them wastes gradient
   capacity that is better spent on semantic features.

# Motivation

Stage 2 rewards the model for building a mana base that matches its spells' requirements: the
right number of white, blue, black, red, and green sources proportional to pip demand. The
reward (heuristic mana score) is correct, but a reward alone does not make a task learnable —
the model also needs to *see* the information required to make good decisions.

**The counting problem.** At any given pick step the model needs to know something like "I
have 2 white pips worth of demand, 1 white source, so I am short 1 white land." This requires:
summing pip counts over all spells picked so far, summing mana produced over all lands picked
so far, and computing the per-color difference. These are accumulation and subtraction
operations — exactly what transformers are poor at.

Transformers learn by comparing token representations to each other via attention. They can
recognize that a card *is* a Plains or a card *has* white pips, because those are pattern-
matching tasks. Summing values across a sequence of 40 cards and tracking a running total is a
different kind of task — one closer to a program running on a counter register. Without
explicit help, the model would have to learn that counting trick from scratch through trial and
error, which is slow and unreliable.

**The information is already there.** Feature 015 appended 15 explicit mana features to every
card embedding, including normalized pip counts (per color) and mana produced (per color). This
means the running demand and supply can be computed from data the model already has access to —
no new parsing infrastructure is needed.

**The slot is already reserved.** The pool tensor layout already includes 6 zero-padding
dimensions per slot at positions `embed_dim+2..embed_dim+7`. Filling them with the deficit
requires no change to the pool slot structure or to the pool transformer's `d_model`.

# How the Running Deficit Works

At the start of each episode the deficit is zero — no demand, no supply.

After each pick step the deficit is updated:
- **Spell picked**: add the card's raw pip counts (W/U/B/R/G/C) to the running demand total.
- **Land picked**: add the card's raw mana produced counts (W/U/B/R/G/C) to the running supply total.
- **Deficit** = demand − supply, normalized by dividing by 17 (the target number of mana
  sources in a sealed deck).

This normalized deficit is then broadcast identically to all 96 rows of the pool tensor. The
transformer sees the same global deck state in every slot and can use it while scoring each
card. A large positive deficit in one color means "this deck is short on this color — prioritize
lands that produce it." A deficit near zero means the deck is balanced for that color.

**Reading pip counts from the card embedding:**

The card embedding produced by feature 015 is structured as:
```
[transformer embedding (2×d_model)] + [15 explicit mana features]
```

The 15 mana features are ordered:
```
  [0-5]  pip counts W/U/B/R/G/C   (normalized: ÷8 for W/U/B/R/G, ÷3 for C)
  [6]    generic mana count        (normalized: ÷15)
  [7]    X count                   (normalized: ÷3)
  [8]    mana value                (normalized: ÷16)
  [9-14] mana produced W/U/B/R/G/C (normalized: ÷3, clamped ≤ 1)
```

To compute the deficit, the raw pip counts (indices 0-5) and raw mana produced (indices 9-14)
are recovered by reversing the normalization using the constants defined in `mana_scorer.py`.

# Removing Redundant Auxiliary Heads

Feature 015 added 20 auxiliary heads to card encoder training: 1 is-land, 6 card color,
6 pip count, 1 mana value, 6 mana produced. The purpose was to force the transformer to encode
mana-relevant features in its learned representation.

The pip count and mana value heads are now redundant for the same reason the counting problem
exists: the transformer is not good at exact arithmetic. Even with auxiliary supervision, the
transformer approximates pip counts rather than computing them exactly — the representation
captures "roughly 2 white pips" rather than precisely 2.0.

Since the explicit mana features appended to the embedding already provide exact pip counts
and mana value without going through the transformer, the auxiliary heads offer little value
beyond what the explicit features already deliver. Worse, they consume gradient capacity that
could be used to learn genuinely semantic features — things like "this card is a removal spell"
or "this card is an aggressive two-drop" — which the explicit features cannot encode.

The 7 redundant heads (pip count × 6, mana value × 1) should be dropped, keeping 13:
1 is-land, 6 card color, 6 mana produced. These 13 heads cover properties that benefit from
being in the transformer representation because downstream code will use them for pattern
matching rather than arithmetic (is this card castable in my colors?).

# Feature Division Principle

This change reflects a general principle for what belongs in the transformer versus what
belongs in explicit features:

**Use the transformer for:**
- Semantic meaning: what does this card *do*? (removal, card draw, aggression, evasion)
- Synergies: does this card combine well with cards of a particular type or strategy?
- Efficiency: is this effect good for its mana cost? (This comparison requires the transformer
  to see both the effect text and the mana cost together.)

**Use explicit features for:**
- Anything derivable from formal grammar: mana costs, P/T, type lines, keywords
- Anything requiring exact arithmetic downstream: pip counts, mana value, mana produced

Power/toughness is an interesting borderline case: recognizing that a 3/3 for {W} is
unusually efficient requires comparing the numbers to the mana cost, which is a semantic
judgment. But accumulating P/T across a deck to compute a curve is arithmetic. The right
answer depends on which use case matters more; for now P/T remains in the transformer text.

# Architectural Extensions (for future consideration)

The changes above are the minimum required to make Stage 2 tractable. Two more ambitious
architectural directions came out of this discussion:

**Wide & Deep (Google 2016).** The current card encoder feeds `pooled_embedding + meta_vector`
to the price regression head. A Wide & Deep variant would run the structured mana features
through a small MLP (the "Wide" path) separately from the transformer (the "Deep" path) and
concatenate the outputs before the price head. This gives structured features their own
expressive pathway rather than mixing them into the transformer's input.

**FiLM (Feature-wise Linear Modulation).** For the pool transformer, instead of broadcasting
the deficit into each slot, FiLM would pass the deficit vector through a small MLP to produce
per-channel scale (γ) and shift (β) parameters. These are then applied to the transformer's
hidden states after each layer: `h' = γ ⊙ h + β`. This lets the deficit influence the entire
depth of the network, not just the input layer. Architecturally it is cleaner than per-slot
injection but requires adding FiLM layers to the pool transformer.

Both extensions are deferred — the slot-filling approach is sufficient for Phase 1 and does
not require architecture changes.
