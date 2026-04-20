# April 3, 2026 — Probe results expose embedding blindspot

**TL;DR:** Implemented the `validate-embeddings` linear probe command and
ran it against the real card corpus. The results made clear that the
price-predictor embeddings do not encode mana cost or color at all, which
directly motivated a new feature spec for multitask retraining.

The day started by completing the feature 014 implementation — the
`python -m sealed validate-embeddings` command that trains 20 sklearn
probes on frozen card embeddings to check whether mana-relevant features
are linearly decodable. Getting to a clean implementation took a few
rounds of spec refinement along the way. The original design had
"land-only" probes for mana production, which I pushed back on: cards
like Sol Ring and Orzhov Signet produce mana without being lands, so the
correct approach is to label all cards and let non-producers get zero.
The skip logic for "zero-positive-example colors" went the same way —
there are always examples for all colors across 30k cards, so no need
for it.

The more interesting part of the day came when I actually ran the probes
against the embeddings that the price-predictor transformer had been
generating. The results split cleanly into two groups:

Is land scored 0.992. All six mana-produced probes scored around 0.99.
Card color (C) scored 0.999.

Card color (W/U/B/R/G) scored around 0.91. All six pip count probes
scored around 0.51. Mana value scored 0.411.

The interpretation is straightforward: the price predictor learned that
expensive cards have powerful effects. Mana cost is largely noise from a
price-prediction standpoint — a busted effect at {1}{W} and a busted
effect at {3}{W}{W} are both expensive for the same reason. The
transformer encoded what mattered for its task and nothing more. The
0.99 scores for mana production fit the same logic: Sol Ring and Mana
Crypt are expensive partly because of their mana output, so the model
had good reason to track that.

The practical consequence is that Stage 2 sealed training cannot rely on
these embeddings to reason about casting costs or color requirements. The
question was what to do about it. I raised the idea of adding auxiliary
supervised heads directly to the pooling layer output — the same 512-dim
vector stored in the .npz files — so that the transformer is forced to
encode mana-relevant features during training. This is standard multitask
learning: the auxiliary losses flow gradients back through the encoder,
and since the heads are linear projections on the embedding itself, any
feature the head can predict will also be linearly decodable by the
probes. The heads would exist only during training and be discarded
before saving the checkpoint.

Using linear heads is intentional, not a simplification. The property we
are testing for with the probes is linear decodability, so training
linear heads guarantees exactly that property if the auxiliary losses are
strong enough.

One non-obvious design question that came up: BCE and MSE operate at
very different scales. A pip count of 3 produces MSE=9 if predicted as
zero, while a misclassified binary label produces BCE around 0.7. Without
correction, the regression heads would dominate the gradient. The fix is
to standardize regression targets before computing MSE, which puts all
20 heads on roughly the same footing without adding hyperparameters.
Class imbalance in the binary heads (is-land is about 5% positive) gets
handled by `pos_weight = num_negatives / num_positives` in
`BCEWithLogitsLoss`.

The lambda weighting question had no clean analytical answer. Starting
around 0.2 and tuning experimentally is the plan, with the intuition
that it should be "as low as possible while the probes still pass." The
price accuracy tolerance for how much degradation is acceptable is
intentionally left as a manual judgment after training, because there is
no principled threshold to anchor it to in advance. Price prediction
matters not just as a sanity check but because price is the proxy for
card strength that Stage 3 of the training curriculum will depend on.

The spec discussion surfaced a subtle point about MTG color rules that
needed to get the definition right. Card color is not the same as pip
counts. A card with {C}{C} in its mana cost has colorless pips but its
color is colorless (C=1), not because it has {C} pips specifically, but
because it has no colored pips at all. The devoid mechanic makes cards
colorless regardless of what their mana cost contains, and lands have no
mana cost and are therefore also colorless. Getting this wrong in the
current code (features 013 and 014) would have produced wrong labels;
the correction is canonical and fine to propagate to shared methods.

By end of day the feature 015 spec, plan, tasks, and analysis were all
committed, with the implementation left for a future session.
