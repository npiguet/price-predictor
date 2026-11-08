# MTG Forge models

Machine-learning models for Magic: The Gathering, built on the [MTG Forge](https://github.com/Card-Forge/forge) engine: a card price predictor, and agents that build sealed decks, draft, and predict what card abilities do in play.

## Setup

Prerequisites: Python 3.14+, Java 17+, Maven, and a CUDA GPU for the PyTorch models.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu126

# MTGJSON data, into resources/
#   https://mtgjson.com/api/v5/AllPrintings.json
#   https://mtgjson.com/api/v5/AllPricesToday.json

# Forge as a sibling checkout at ../forge, on its effect-record-hooks branch
(cd ../forge && mvn install -DskipTests)
(cd forge-connector && mvn install -DskipTests)

pytest                             # fast unit tests
```

## Modules

Each module's `CLAUDE.md` describes its commands, file formats and invariants, and its `GLOSSARY.md` (where one exists) defines its terms. Its runbooks are the quickstarts of the features that built it. A later quickstart supersedes an earlier one's commands and flags wherever the two differ.

| Module | What it does | Quickstarts, oldest first |
|---|---|---|
| [`src/price_predictor/`](src/price_predictor/) | Predicts a card's euro price from its converted card text and printing data; owns the converted card format, the tokenizer and the vocabulary. | [001](specs/001-card-price-predictor/quickstart.md), [002](specs/002-forge-api-integration/quickstart.md), [003](specs/003-cheapest-printing-price/quickstart.md), [004](specs/004-cardmarket-eur-pricing/quickstart.md), [005](specs/005-card-eval-endpoints/quickstart.md), [006](specs/006-card-script-parsing/quickstart.md), [007](specs/007-transformer-model-arch/quickstart.md), [008](specs/008-model-harmonization/quickstart.md), [009](specs/009-printing-metadata/quickstart.md), [010](specs/010-mtg-custom-tokenizer/quickstart.md) |
| [`src/sealed/`](src/sealed/) | Sealed-format pipeline: card encoder, deck scorer, deck picker, pool generation and self-play matches. | [011](specs/011-sealed-dataset/quickstart.md), [012](specs/012-sealed-training-data/quickstart.md), [013](specs/013-sealed-deck-scorer/quickstart.md), [014](specs/014-self-play-refinement/quickstart.md), [015](specs/015-encoder-fine-tuning/quickstart.md), [016](specs/016-card-winnability-pretraining/quickstart.md), [017](specs/017-one-shot-deck-picker/quickstart.md) |
| [`src/draft/`](src/draft/) | Draft agent: gen-1 imitation, gen-2 RL, gen-3 online GRPO, and evaluation by games won. | [018](specs/018-draft-agent/quickstart.md), [019](specs/019-draft-live-play/quickstart.md), [020](specs/020-draft-agent-rl/quickstart.md), [021](specs/021-draft-online-grpo/quickstart.md), [022](specs/022-draft-game-evaluation/quickstart.md) |
| [`src/effects/`](src/effects/) | Ability effect model: a per-ability embedding cache and a state-conditional effect head, trained on records collected from Forge games. | [023](specs/023-ability-effect-model/quickstart.md), [024](specs/024-ability-effect-model-gen2/quickstart.md), [025](specs/025-ability-effect-model-gen3/quickstart.md) |
| [`forge-connector/`](forge-connector/) | Java workers the Python packages run against Forge. | none: its workers run through the Python packages' commands |

Design records live in [`experiments/`](experiments/) and the normative specs in [`specs/`](specs/).
