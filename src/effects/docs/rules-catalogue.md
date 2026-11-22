# Rules catalogue, detectors and `rule-counts`

Rules to respect when editing `rules/`, a detector family or rule tagging, and when running or reading `rule-counts`. This file does not describe the catalogue:

- The catalogue's own documentation: `rules/README.md`; its file formats: `specs/025-ability-effect-model-gen3/contracts/catalogue-extensions.md`.
- What is specified: `specs/2026-10-30-ability-effect-model-gen3.md` § 4, 5 and 7.
- Code: enums in `domain/rule_catalogue.py`, read-only loaders in `infrastructure/rules_catalogue_io.py`, one module per detector family in `domain/rule_detectors/`.

## Declarations

- Every scenario of `rules/scenarios/<unit>.json` declares a `detector` or `{"none": reason}`. Every non-base variant carries an `operation` or `{"unclassifiable": reason}`.
- An `operation` is one of the `ForkOperation`s of `domain/fork_operation.py`, pinned against the Java enum both ways.
- A detector's parameters name a rule's general condition (a check, a `Mode$`, a keyword, a verdict bit, an `Event$`), never a card. A declaration copied from a scenario's specifics tags only a small share of what its family finds; `scripts/rule_breadth.py` flags those for review.
- A near miss is counted only where the declaration carries a scoping parameter of its family, one naming which sources, stack objects or judged entities the rule speaks for: `SCOPES` in `continuous_contribution`, `resolution_outcome` and `legality_membership` (without a `restriction`), `api`, `keyword` or `source_keyword` in `outcome_event`, qualified declarations included. A declaration without one has a `fires` side only. An unscoped near miss is every record of the kind without the fact: it makes a rule look two-sided, so `directed-rules` never sends forks after it.
- A `source_keyword` is the candidate card's, so a `verdict_bit` or `legality_membership` declaration carrying one also carries `candidate` (`cast` or `activated`, read by `acting_line.ability_kind`): without it an Aura's activated or granted ability is taken for its cast, and `DetectorIndex.of` refuses the declaration. `resolution_outcome`'s `ability_kind` says the same of a stack object (701.5a-1 reads cast spells only).
- A tapped creature (`legality_restrictions._tapped`) and a carried `vigilance` (`combat_keyword_effect`) have no near miss whatever the parameters: nothing lets a tapped creature attack or block, and an attacker without vigilance is 508.1f's `fires` side. An `affordable` rule's near miss is a candidate failing no check but `cost` (`verdict_bit._near`); a `verdict_bit` rule about responding narrows its candidates with `stack_empty: false` (702.61a-1).
- `test_catalogue_on_data.py` fails on a declaration either of whose sides tags more than half of its record kind on the fixtures, unless it is in `GENERAL` (the kind's own general fact).
- A declaration whose `needs` a field the record lacks makes the rule not countable, not zero; a gen-3 record carries every field a declaration needs.
- `DetectorIndex.of` checks every declaration when it is built. An unregistered family, a `record_kind` outside the family's `kinds`, a parameter the family does not declare, or a value outside its closed list (`domain/rule_detectors/declared_values.py`, pinned to `../forge`) raises `DeclarationError` naming the scenario, so a misspelt value is refused rather than matching nothing.
- A declaration asking its record kind for an event the collector never writes there is untaggable (`domain/rule_detectors/emittable_events.py`, which lists the event types each record kind carries by the connector source that builds them; a combat record lists the clause-hook events too, `replacement_applied` among them, since a clause a replacement runs inside the damage step is filed there; a turn-action record lists the events of its six actions alone (`TURN_ACTIONS`), the fork premise's `fork_edit` record being read by no declaration). It tags nothing and `rule-counts` lists it. When a collector source gains an event, add it to that file.
- No two scenarios of different rules share one declaration. A scenario stating another's observable fact declares `{"same_as": id or [ids], "reason": …}` and is counted under the named scenarios' rules, never its own: `DetectorIndex.of` refuses a `same_as` naming a scenario of one of its own rules, so two scenarios of one rule stating one fact each declare the detector. What a record cannot show declares `none`.
- On a degraded worker's record, a missing hook-derived event reads as unknown (a forced resolution's fork forwards the hook events, so its missing event reads as real): `outcome_event` gives neither an absence's `fires` side nor a near miss.
- The game-end marker is tagged by no declaration.
- The game-end record (`EffectRecord.is_game_end_record`) is read by a declaration only for a game-end event (`player_won`, `player_lost`); the events the collector rescued into it are no resolution's, so `outcome_event` returns no tag on it for any other event, and `sba_game_loss` and `sba_zone_move` skip it and read every other effect half.
- A game-end rule is declared on `resolution` and on `combat` records (104.2a-1 and 104.2a-2): most games end in the combat record's damage step. A rule no combat can show (704.5b's draw from an empty library) has no combat declaration.
- `sba_game_loss` reads life loss from `life_change` only: damage dealt to a player is not life loss under infect. `keyword_combat`'s `observed` is whether the record shows the keyword's effect (`keyword_outcomes.py`), so a `fires` combat without it goes on the review list.
- A rewrite's `buffered_events` are a place of their own, never the rewrite's events a family reads. `rewrite_event` reads no record whose incoming event is unknown.
- `combat_keyword_effect` covers the combat keywords outside gate 2's table and leaves that table alone.

## Commands to run after a change

- `scripts/rules_catalogue.py validate` checks the declarations.
- `scripts/rules_catalogue.py directed-rules --counts <counts> --cards-folder <gen>/converted/cardsfolder` writes `rules/directed-fork-rules.txt`, `rules/coverage-wording-cards.txt` and `rules/directed-fork-counts.json`, the counts the list is decided from, so `test_rules_catalogue.py` can regenerate the list from checked-in files. `rules/directed-fork-variants.tsv` is the registry's pairs, kept by hand (`docs/forks.md` § Registering a directed rule). `--probabilities-output FILE` also writes each listed rule's records per registered side from the same counts, which `sealed match-outcomes --directed-fork-probabilities` computes the rule's directed-fork probability from (`docs/forks.md` § Directed forks).
- `scripts/rules_catalogue.py gen3-fields` recomputes the learnable rules after a map rerun.
- `scripts/rule_entity_audit.py` (`--corpus` and `--output` required) measures, per rule and output, how often the outputs are supervised at each role's entities.

## Tagging

- `rule-counts`, the `build-corpus` survey and the selection-set freeze all tag through the one `tag_record`.
- A rule's scenarios merge into one tag: `fires` if any says so, else `does_not_fire`, else `other`.
- A tag names entities in two roles, and this is the one definition:
  - Condition entities are what the rule's condition reads: a keyword's carrier, the acting source with its targets and controller, a judged permanent, an anchor attacker.
  - Outcome entities are where its outcome falls as the record's targets place it: the creature fought, the defending player, a lifelinker's controller, the player who discarded or drew, every card and player a target legality is written at.
- A tag whose outcome is an absence names condition entities only. A `trigger` record's tags name no outcome entity, because the trigger's effect arrives as its resolution's record.
- The hard rule mask's gate trains at both roles. A rule test reads at the outcome entities.
- A tag whose `observed` disagrees with its polarity goes on the review list and is never a rule-test record.
- `RuleSource` (`random_seat`, `forge_ai`, `aimed`, `fork`) is decided per record. `fork` is a probe's, an intervention's or a protection fork's record (`FORK_SOURCE_ORIGINS`), even one carrying `aimed_at`; it re-records a moment of its live game, so `directed-rules` and the budgets' `available` leave it out, or a combat would count once per probe. `directed-rules` decides the list from `forge_ai` and `random_seat` alone (`DECIDING_SOURCES` in `scripts/rules_catalogue.py`): counting the directed forks' own `aimed` records would take a rule off the list that makes them.

## `rule-counts`

- Writes `rule -> polarity -> source -> records` to `--output` (pass it: a file in an `inputs` run directory when a later run reads it, as the steered round does), with the rules declared without a detector, the uncountable ones, the review list and the per-rule `carriers`. Every source is written, `fork` included.
- A `playability` record's carrier is the candidate's own card with its line, and a judged card only through the keyword line the rule names (`tag_carriers`): naming every judged entity's card makes `carriers` most of the file and gives the steered round no line to vary.
- A rule with a taggable declaration that no record tags is listed under `declared_zero`. The file also lists `untaggable` (per scenario, why no record can show it) and `same_as`. Only a rule with no declaration is left out silently.
- A gen-3 game whose shard holds no game-end marker counts toward no rule, review entry or carrier; `cut_games` says how many were left out.
- It reads gen-3 shards only, as `scripts/rule_breadth.py` does; a set of an older generation is refused, naming it. The file's `generation` is what the steered variant round checks.
