# Sealed

Builds 40-card sealed decks from sealed pools. It collects Forge-played sealed matches, learns from them which cards and which decks win, and builds decks with what it learned. Terms shared with other packages are in the root `GLOSSARY.md`.

## Language

### Pools and decks

**Sealed pool**:
The cards one player opens for a sealed event: six boosters of one set, recorded without their basic lands.
_Avoid_: card pool, booster pool

**Sealed-legal set**:
A set a sealed pool can be opened from: one whose draft booster holds at least twelve cards and that is not an Un-set.
_Avoid_: eligible set, sealed-viable set

**Excluded card**:
A card no booster may contain. Each one drawn is replaced by another card from the same rarity slot, so the pool keeps its size and rarity mix.
_Avoid_: banned card, held-out card (that is the effects package's reason for excluding one)

**Deck**:
Forty cards: cards chosen from one sealed pool plus as many basic lands as the deck needs. A deck is identified by its cards and their copy counts, not by their order.
_Avoid_: build, list, decklist

**Manabase heuristic**:
The rule that fills a deck's remaining slots with basic lands: at least two for each color with a pip among the nonland cards' mana costs when there is room, the rest shared out in proportion to each color's pips, and an even split over all five colors when there are no colored pips.
_Avoid_: land fill, basic-land balancer

### How decks are built

**Build method**:
How one deck in a match was built: one of the four Forge build methods, or, for a generated deck, its deck label.
_Avoid_: generation method, deck method, method tag

**Forge-built deck**:
A deck built by one of the four Forge build methods: forge-best, forge-3sub, forge-8sub or random.
_Avoid_: baseline deck, Forge deck (unqualified)

**forge-best**:
The deck Forge's own sealed deck builder makes from the pool.
_Avoid_: Forge's optimal deck

**forge-3sub, forge-8sub**:
The forge-best deck with three, or eight, of its nonland cards or nonbasic lands swapped for pool cards of the same card type.
_Avoid_: perturbed deck, mutated deck

**random** (build method):
Twenty-three nonland cards drawn at random from the pool, with basic lands matched to their pips.
_Avoid_: random baseline

**Generated deck**:
A deck a trained sealed model built, by deck search with a deck scorer or by a deck picker's pick walk, recorded with a deck label.
_Avoid_: scorer-built deck (pickers build them too), self-play deck

**Deck label**:
The name a batch of generated decks is recorded under, which stands as their build method in every match they play; usually the generation that built them, such as gen-2.
_Avoid_: method tag, generation tag

### Matches and corpora

**Match**:
A best-of-N series of games, seven by default, between two decks of the same set, both piloted by the Forge AI.
_Avoid_: game (one game of a match), duel

**Side A, side B**:
The two places in a match. Side A's deck is chosen first; side B's comes from a different sealed pool of side A's set and never has the same cards as side A's.
_Avoid_: player 1 and player 2, seat (that is a draft term)

**Self-play match**:
A match in which at least one deck is a generated deck, so that the next deck scorer learns from decks its predecessors built.
_Avoid_: refinement match, gen-N match

**Collection run**:
One invocation of match collection; every record it writes carries its run id.
_Avoid_: session, generation (one generation's data spans many runs)

**Match-outcomes corpus**:
The append-only record of every collected match: its set, both decks and their build methods, the winner of each game and which side played first in it.
_Avoid_: match data, training matches

**Cards-played corpus**:
The append-only record of every collected game: for each side, which of its deck's nonbasic cards were played and which were not, the winner and which side played first.
_Avoid_: card-play log

**Played card**:
A card of a side's deck at least one copy of which entered the stack or the battlefield during the game.
_Avoid_: cast card (lands are played without being cast), used card

### Winnability labels

**Winnability label**:
One of the nine numbers per card that sum up how its decks fared in the cards-played corpus, over the games in which the card was in a side's deck: the play score, the draw score, the played rate, the cast lift and one color lift per color. The sealed encoder learns to predict them.
_Avoid_: card label, card win rate, card score

**Play score**:
Over the games the card's side played first with the card in its deck, the games won with the card played minus the games lost with the card played, per game. A card never played scores zero, however its decks fared.
_Avoid_: win rate on the play, score (unqualified)

**Draw score**:
The play score's counterpart over the games the card's side drew first.
_Avoid_: win rate on the draw

**Played rate**:
The share of the games with the card in its deck in which the card was played.
_Avoid_: cast rate, play rate

**Cast lift**:
The card's side's win rate in games where the card was played, minus the win rate in games where it stayed in the deck unplayed.
_Avoid_: played lift, impact

**Color lift**:
For one color, the card's score over its games in decks that hold that color minus its score over all its games, a score being games won with the card played minus games lost with it played, per game. A deck holds a color when any of its nonbasic cards has that color in its mana cost.
_Avoid_: color synergy, color affinity

**Raw label, shrunk label**:
A winnability label as observed, and the same label pulled toward neutral by counting imaginary even games alongside the real ones, so that a rarely seen card cannot take an extreme value. The sealed encoder learns the shrunk labels. A label the corpus holds no games for is left empty, not neutral.
_Avoid_: smoothed label, regularised label

**Shrinkage strength**:
How many imaginary even games a shrunk label counts.
_Avoid_: prior strength, k (in prose)

### Card encoders

**Card encoder**:
Whichever encoder turns converted cards into the card embeddings the deck scorer and the deck picker read: the sealed encoder or, in its place, the price encoder.
_Avoid_: BERT, text encoder

**Sealed encoder**:
The card encoder the sealed package trains for itself, by predicting each card's winnability labels and the tokens masked out of its converted card text. It pools a card's tokens by attention, alone or beside an element-wise maximum.
_Avoid_: sealed BERT, winnability encoder, card encoder (when this one is meant)

**Card embedding**:
What the deck scorer and the deck picker read a card as: the card encoder's pooled card vector for the card without its name line, followed by the card's deterministic features.
_Avoid_: card vector, embedding (unqualified), npz

**Deterministic features**:
The fixed list of numbers read straight off a converted card rather than learned: whether it is a land; its mana cost's pips by color, colorless and generic mana, X count and mana value; its colors; the colors and amount of mana it produces; and its power, toughness and starting loyalty.
_Avoid_: game features, det features, hand-built features (that is the price package's sklearn model)

**Embedding cache**:
The stored card embeddings of every converted card, all made by one card encoder. Changing the card encoder means rebuilding it.
_Avoid_: npz cache, embeddings

### Deck scorer

**Deck scorer**:
The model that reads a deck, less its basic lands, as an unordered set of card embeddings and gives it a deck score. It is trained so that, in each collected match, the winner's deck scores above the loser's.
_Avoid_: Set Transformer (the architecture, not the model's role), evaluator, value model

**Deck score**:
A deck scorer's output for one deck: a number meaningful only against other decks' scores from the same scorer, the higher-scored deck predicted to win the match.
_Avoid_: rating, deck quality, win probability

**Match margin**:
How many more games a match's winner won than its loser, by which scorer training can weigh the match.
_Avoid_: game margin, win margin

**Scorer accuracy**:
The share of the held-out split's matches in which the deck scorer scores the winner's deck above the loser's.
_Avoid_: validation accuracy, val_acc

**Phase A**:
Deck scorer training on a fixed embedding cache, the card encoder left as it is.
_Avoid_: frozen phase, Phase 1 (that numbers the sealed pipeline's stages)

**Phase B**:
Deck scorer training that also fine-tunes the card encoder, starting from a Phase A scorer.
_Avoid_: unfrozen phase, joint training, Phase 2

### Building decks with a scorer

**Deck search**:
Building a deck from a sealed pool with a deck scorer. Starting from 23 random nonland cards and no nonbasic lands, the search keeps making the single-card change the scorer scores highest (swapping a nonland card or a nonbasic land for one from the pool, adding a nonbasic land or removing one), then the manabase heuristic fills the deck.
_Avoid_: greedy builder (the search anneals by default), greedy search

**Restart**:
One of several deck searches over the same pool from different starting decks, of which the highest-scored result is kept.
_Avoid_: retry, seed

**Color-pair restarts**:
One restart per two-color pair, each starting from nonland cards of only those two colors.
_Avoid_: color seeding, guild restarts

### Building decks with a picker

**Deck picker**:
The model that reads a whole sealed pool's card embeddings at once and ranks every card in it, a deck being read off the ranking by the pick walk. It is trained to maximise the picker reward.
_Avoid_: policy, one-shot builder, picker model

**Pick walk**:
Reading a deck off a deck picker's ranking: going down from the top, take every nonland card until there are 23 and every nonbasic land met before then, then let the manabase heuristic fill the deck.
_Avoid_: pick decomposition, § 1.1 walk

**Training scorer**:
The deck scorer, left unchanged, whose deck scores the deck picker is trained to maximise.
_Avoid_: reward model, frozen scorer

**Picker reward**:
The deck score the training scorer gives one of the deck picker's decks, less its basic lands.
_Avoid_: return, reward (unqualified)

**Held-out reward**:
The mean picker reward of the pick-walk decks of the held-out split's pools.
_Avoid_: validation reward, val reward

**Auditor scorer**:
A second deck scorer, never trained against, whose ranking of the deck picker's decks is compared with the training scorer's to catch a picker that pleases only its training scorer.
_Avoid_: audit scorer, cross-scorer

**Reference picker**:
An earlier deck picker, left unchanged, that a new picker is penalised for drifting away from.
_Avoid_: anchor picker, prior picker

### Evaluation

**Round-robin evaluation**:
Measuring a deck scorer by match play: from each of a number of fresh sealed pools, deck search with the scorer builds one deck and forge-best builds another, and every scorer deck plays every forge-best deck.
_Avoid_: eval run, scorer evaluation

**Pool delta**:
For one pool of a round-robin evaluation, its scorer deck's win rate minus its forge-best deck's, each over all the matches the deck played, counted in games or in matches.
_Avoid_: per-pool delta, win-rate delta
