# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ML models for Magic: The Gathering, built on the MTG Forge engine: a card price predictor, and agents that build sealed decks, draft, and predict what card abilities do in play. Each module has its own `CLAUDE.md` (commands, file formats, invariants) and, where one exists, a `GLOSSARY.md` of its domain terms; both load when you work in that module. Terms more than one module uses are in the root `GLOSSARY.md`. Setup instructions are in `README.md`.

## Modules

- `src/price_predictor/` — predicts a card's euro price from its converted card text and printing data. Also owns what every other package reads cards through: the converted card format, the tokenizer and the vocabulary.
- `src/sealed/` — the sealed-format pipeline: card encoder, deck scorer, deck picker, pool generation and self-play match generation.
- `src/draft/` — the draft agent (gen-1 imitation, gen-2 RL, gen-3 online GRPO), trained on Forge-generated drafts.
- `src/effects/` — the ability effect model: a per-ability embedding cache and a state-conditional effect head, trained on effect records collected from Forge games.
- `forge-connector/` — the Java Maven module holding the CLI workers the Python packages run against Forge.

The four Python packages are each laid out in hexagonal (ports-and-adapters) style, `domain` → `application` → `infrastructure`. Imports point one way: `sealed` imports from `price_predictor`, and `draft` and `effects` import from both, never the reverse.

## Layout

- `../forge` — sibling checkout of MTG Forge on its `effect-record-hooks` branch, built with `mvn install -DskipTests`: the card scripts and the game engine.
- `resources/` — MTGJSON dumps (`AllPrintings.json`, `AllPricesToday.json`).
- `rules/` — the rules catalogue the effects package reads.
- `output/` — generated data, never committed: converted cards (`cardsfolder/`, `tokenscripts/`) and one directory per package (`sealed/`, `draft/`, `effects/`). Its corpora are append-only and cannot easily be rebuilt.
- `models/{price-predictor,sealed,draft,effects}/` — checkpoints, never committed; every trainer writes a `{timestamp}` file plus a rolling `latest`.
- `scripts/` — analysis and probe scripts, run by path.
- `tests/` — unit and integration tests.
- `experiments/*.md` — experiment records: what happened and why; the **only** home for benchmark numbers, run logs and rejected alternatives.
- `specs/YYYY-MM-DD-<name>.md` — hand-written normative specs (what, not why), in timeless present tense.
- `specs/NNN-name/` — spec-kit feature directories (`spec.md` → `plan.md` → `research.md` → `tasks.md`) driving implementation. Read the relevant ones before non-trivial work in an area.
- `diaries/` — dated retrospective diary entries.
- `docs/`, `docker/` — working plans and the container setup.

Load the **feature-workflow** skill before every edit to a file under `experiments/` or `specs/`, including a one-sentence reword: `experiments/*.md` carries a binding prose style that applies to each edit, not just to new documents.
