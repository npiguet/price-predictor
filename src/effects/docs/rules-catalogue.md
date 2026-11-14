# Rules catalogue, detectors and `rule-counts`

Rules to respect when editing `rules/`, a detector family or rule tagging, and when running or reading `rule-counts`. This file does not describe the catalogue:

- The catalogue's own documentation: `rules/README.md`; its file formats: `specs/025-ability-effect-model-gen3/contracts/catalogue-extensions.md`.
- What is specified: `specs/2026-10-30-ability-effect-model-gen3.md` § 4, 5 and 7.
- Code: enums in `domain/rule_catalogue.py`, read-only loaders in `infrastructure/rules_catalogue_io.py`, one module per detector family in `domain/rule_detectors/`.

## Declarations

- Every scenario of `rules/scenarios/<unit>.json` declares a `detector` or `{"none": reason}`. Every non-base variant carries an `operation` or `{"unclassifiable": reason}`.
- An `operation` is one of the `ForkOperation`s of `domain/fork_operation.py`, pinned against the Java enum both ways.
- A detector's parameters name a rule's general condition (a check, a `Mode$`, a keyword, a verdict bit, an `Event$`), never a card. A declaration copied from a scenario's specifics tags only a small share of what its family finds; `scripts/rule_breadth.py` flags those for review.
- A declaration whose `needs` the record's generation lacks makes the rule not countable, not zero.
- `combat_keyword_effect` covers the combat keywords outside gate 2's table and leaves that table alone.

## Commands to run after a change

- `scripts/rules_catalogue.py validate` checks the declarations.
- `scripts/rules_catalogue.py directed-rules --counts` writes `rules/directed-fork-rules.txt` and `rules/coverage-wording-cards.txt`.
- `scripts/rules_catalogue.py gen3-fields` recomputes the learnable rules after a map rerun.
- `scripts/rule_entity_audit.py` measures, per rule and output, how often the outputs are supervised at each role's entities.

## Tagging

- `rule-counts`, the `build-corpus` survey and the selection-set freeze all tag through the one `tag_record`.
- A rule's scenarios merge into one tag: `fires` if any says so, else `does_not_fire`, else `other`.
- A tag names entities in two roles, and this is the one definition:
  - Condition entities are what the rule's condition reads: a keyword's carrier, the acting source with its targets and controller, a judged permanent, an anchor attacker.
  - Outcome entities are where its outcome falls as the record's targets place it: the creature fought, the defending player, a lifelinker's controller, the player who discarded or drew, every card and player a target legality is written at.
- A tag whose outcome is an absence names condition entities only. A `trigger` record's tags name no outcome entity, because the trigger's effect arrives as its resolution's record.
- The hard rule mask's gate trains at both roles. A rule test reads at the outcome entities.
- A tag whose `observed` disagrees with its polarity goes on the review list and is never a rule-test record.
- `RuleSource` (`random_seat`, `forge_ai`, `aimed`) is decided per record.

## `rule-counts`

- Writes `rule -> polarity -> source -> records` to `--output` (default `output/effects/rule-counts.json`), with the rules declared without a detector, the uncountable ones, the review list and the per-rule `carriers`.
- A rule no record tags is omitted, so absent does not mean "not countable".
- It reads gen-2 or gen-3 shards, never mixed. The file's `generation` is what the steered variant round checks.
