# Price predictor

Predicts a card's euro price from what is printed on it and how it has been printed, for real cards and hypothetical ones alike. It also owns the converter, the tokenizer and the vocabulary builder that the sealed, draft and effects packages read cards through. The card-text, card-name, vocabulary and training terms those packages share are in the root `GLOSSARY.md`.

## Language

### Card conversion

**Conversion check**:
A comparison of each converted card's converted ability text against the card's Oracle text, scored as word-bag similarity, that flags cards the converter rendered badly.
_Avoid_: oracle check, validation

**Word-bag similarity**:
The share of words two texts have in common, counted with repetition and ignoring order, after both are normalised to the same conventions (reminder text removed, the card's name replaced, landwalk spelled one way).
_Avoid_: oracle similarity, match score

### Cards and prices

**Hypothetical card**:
A card that has no canonical name, because it was never printed, and so has neither a card price nor printing data of its own. It can still be priced.
_Avoid_: custom card, fake card, unknown card

**Constructed-legal printing**:
A printing of a card in a set legal or restricted in at least one recognized format. Collector's Edition, 30th Anniversary Edition and similar printings are not constructed-legal.
_Avoid_: tournament printing, legal printing

**Card price**:
The lowest current Cardmarket retail price, normal or foil, over a card's constructed-legal printings, in euros and never below €0.01. It is a property of the card, not of any one printing.
_Avoid_: market price, value, printing price

**Cheapest printing**:
The constructed-legal printing whose price became the card price. The card's printing data is read off this printing.
_Avoid_: reference printing, main printing

### Printing data

**Printing data**:
What is known about how a card has been printed rather than what it does: whether it is on the reserved list, its rarity, its printings count, its cheapest-printing year, its legalities and whether it is ABU-only. A model receives it beside the card's text, never inside it.
_Avoid_: metadata, meta, side-channel, enrichment, printing metadata

**Default printing data**:
The printing data assumed for a card that has none of its own, such as a hypothetical card: rare, printed once, cheapest-printing year 1993, not on the reserved list, not ABU-only, legal in every recognized format.
_Avoid_: empty printing data, unknown printing data

**Printings count**:
The number of sets a card has been printed in.
_Avoid_: reprint count, printing count

**Cheapest-printing year**:
The year the set of the card's cheapest printing was released, never earlier than 1993. It is not the year the card was first printed.
_Avoid_: release year, first-printed year, print year

**ABU-only**:
True of a card whose every constructed-legal printing is in Alpha, Beta or Unlimited.
_Avoid_: is_abu, old-border-only

**Recognized format**:
One of the ten constructed formats whose legality printing data records: standard, pioneer, modern, brawl, legacy, vintage, pauper, commander, penny and oathbreaker. A card's legalities are the recognized formats it is legal in; restricted does not count as legal there.
_Avoid_: format (unqualified), supported format

### Models

**sklearn model**:
The price model built from gradient-boosted decision trees over a fixed vector of hand-built features computed from a card's characteristics, the word frequencies of its converted ability text, and its printing data.
_Avoid_: baseline model, GBR, feature model

**Feature group**:
One family of columns in the sklearn model's feature vector, computed from one aspect of the card: mana cost, card types, supertypes, keywords, power and toughness, layout, printing data, or the word frequencies of the converted ability text.
_Avoid_: feature block, feature set

**Price transformer**:
The price model that reads a converted card as a sequence of tokens and combines the result with the card's printing data to predict a shifted-log price.
_Avoid_: BERT, transformer (unqualified, where the sealed encoder is also in play), neural model

**Price encoder**:
The part of the price transformer that turns a converted card into a pooled card vector, by putting the element-wise maximum and the mean of its per-token outputs side by side, without printing data or the price output. The sealed package can use it as its card encoder in place of the sealed encoder.
_Avoid_: price-trained encoder, BERT

**Shifted-log price**:
The logarithm of a card price plus the log offset; what the price transformer is trained to predict, and turned back into euros by the inverse transform.
_Avoid_: log price (unqualified), target

**Log offset**:
The constant added to a price before taking its logarithm. A smaller offset widens the gaps between cheap cards' targets and so weighs them more in training.
_Avoid_: price offset, epsilon

### Price buckets and evaluation

**Price bucket**:
A half-open euro price range, such as €2–10, used to group cards by price.
_Avoid_: price tier, price band, price class

**Sampling buckets**:
The four coarse price buckets (below €2, €2–10, €10–50, above €50) by which training draws expensive cards more often than their share of the corpus.
_Avoid_: training buckets

**Reporting buckets**:
The six price buckets (below €0.10, €0.10–0.50, €0.50–2, €2–10, €10–50, above €50) by which evaluation breaks down its errors, finer at the cheap end where most cards are.
_Avoid_: evaluation buckets

**Sampler exponent**:
How strongly training oversamples rare price buckets: each card is drawn with weight one over its sampling bucket's size raised to this exponent, so zero samples uniformly and one gives every bucket an equal share.
_Avoid_: oversampling factor, sampling temperature

**Top-20 overlap**:
The share of the held-out split's most expensive fifth, by card price, that the model also places in its own most expensive fifth.
_Avoid_: top-k accuracy, ranking accuracy

**Median percentage error**:
The median, over held-out cards, of the absolute prediction error as a percentage of the card price.
_Avoid_: MAPE (that is the mean)

**Median log error**:
The median, over held-out cards, of the absolute difference between predicted and actual shifted-log prices.
_Avoid_: log MAE, median |log|

**Signed log error**:
Predicted minus actual shifted-log price, kept with its sign so that a negative median within a price bucket shows the model underprices that bucket.
_Avoid_: bias, log bias
