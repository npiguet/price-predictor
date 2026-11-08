# MTG models

Machine-learning models for Magic: The Gathering built on the Forge engine. This glossary holds the terms more than one package uses; each package's own terms are in the `GLOSSARY.md` beside its code.

## Language

### Card text

**Forge card script**:
A card as Forge's own script describes it, in Forge's scripting syntax. The input to conversion, never to a model.
_Avoid_: raw card, Forge card, card file

**Converted card**:
A card rewritten from its Forge card script into the project's own plain-text form: one header line per characteristic, then one ability line per ability, one block per face. Every model in the project reads cards in this form.
_Avoid_: LLM-friendly format, output card, card text file

**Header line**:
A converted card line stating one characteristic: name, mana cost, types, power and toughness, loyalty, defense, colours or layout.
_Avoid_: property line, metadata line

**Ability line**:
A converted card line carrying one ability's rules text, tagged by its kind (spell, static, triggered, activated, replacement, mana, planeswalker, chapter, level, mode and so on).
_Avoid_: oracle line, ability text line

**Converted ability text**:
All of a converted card's ability lines taken together. It is the converter's rendering of the rules text, not the card's Oracle text, and the two differ in wording.
_Avoid_: oracle text (that term means the official Oracle wording), ability text (unqualified)

**CARDNAME**:
The placeholder that stands for the card's own name wherever the card refers to itself, so that a card's text says nothing about what the card is called.
_Avoid_: self-reference, ~

### Card names

**Canonical name**:
A card's name as MTGJSON spells it, with both halves for a split card. A card is found by its canonical name ignoring case, and one half's name finds the whole card.
_Avoid_: card key, display name

**Forge card name**:
A card's name as Forge spells it, which is the name every corpus collected from Forge games records the card under: both halves for a split card, the front face alone for a double-faced or adventure card.
_Avoid_: canonical name (that is MTGJSON's spelling), card id

### Tokens and vocabulary

**Vocabulary**:
A fixed list of word-level tokens built from the converted-card corpus, through which a model reads converted cards; a word outside it reads as unknown. The price transformer and the sealed encoder each read cards through their own.
_Avoid_: vocab (in prose), tokenizer dictionary

**Multi-word keyword**:
A keyword whose name holds a space, such as first strike, which the tokenizer joins into a single token.
_Avoid_: compound keyword, phrase token

**No-mana-cost marker**:
The `none` mana cost a card with no mana cost line is read as, so that a card without a mana cost (a land) is told apart from a card costing {0}.
_Avoid_: mana cost sentinel, null cost

**Pooled card vector**:
A single vector standing for a whole card, made by pooling an encoder's per-token outputs into one.
_Avoid_: CLS vector, card embedding (in the sealed package that term also carries the card's deterministic features)

### Training

**Held-out split**:
The examples a model is never trained on, drawn reproducibly from a fixed seed, on which the model is scored and, for a model trained in epochs, its best checkpoint chosen.
_Avoid_: test split, validation split, test set, val set

**Model version**:
The name a trained model is saved and reported under: the timestamp of its training run.
_Avoid_: model id, checkpoint name

### Forge play

**Forge AI**:
The Forge engine's own game-playing AI. It pilots every deck in every game the project collects, so every win the project learns from is a Forge-piloted win.
_Avoid_: Forge (unqualified, when the AI is meant), bot, engine

**Generation**:
One step in a trained model's lineage, numbered from zero. For the deck-building and drafting agents, generation 0 is the Forge AI and each later generation is the agent trained on games its predecessors took part in. Never the name of a feature or a pipeline stage.
_Avoid_: version, round, iteration
