# Comprehensive Rules catalogue

This folder holds the Magic: The Gathering Comprehensive Rules, split into small files, and a sidecar
per rule unit. The sidecars record what the effect model would need to learn each rule. It serves the
gen-3 effect model (`experiments/2026-XX-XX-effect-model-gen3-design.md`: the plan is the
Description's "Gen-3 indexes its corpus by the rules of the game", and the catalogue's findings are
in its Result).

The goal is to cover the **observable mechanical outcome** of every rule. **Player decisions are out
of scope**: a scenario takes every choice as already made (what is cast, targets, modes, attackers,
blockers, damage assignment order) and records only what the engine then does. Each scenario comes
with contrasting variants, built so that changing one fact flips the outcome: the creature dies or
survives, the spell resolves or fizzles, the trigger fires or does not.

`STATUS.md` is the checklist of where every unit stands. Read it first.

## Layout

| path | what | written by |
|---|---|---|
| `raw/MagicCompRules-YYYYMMDD.txt` | the rules as downloaded from magic.wizards.com/en/rules | download |
| `rules.jsonl` | one line per numbered rule: `id`, `unit`, `section`, `chapter`, `parent`, `text`, `examples`, `refs` | `split` |
| `cr/index.md` | table of every unit with its title, rule count and file | `split` |
| `cr/<unit>.md` | the text of one unit's rules, with their examples | `split` |
| `cr/glossary.md` | the glossary, one entry per paragraph | `split` |
| `packets.json` | units grouped into agent work packets | `packets` |
| `triage/<unit>.jsonl` | phase 1: one classification line per rule | triage agents |
| `scenarios/<unit>.json` | phase 2: scenarios for the unit's mechanical and legality rules, with their variants | scenario agents |
| `MODEL-FIELDS.md` | what the gen-2 effect model reads and is trained on, compiled from the code; the reference the heads pass checks against | by hand |
| `heads/<unit>.jsonl` | heads pass: for each mechanical or legality rule, whether an output is trained on its outcome and which deciding facts the model does not read | heads agents (`scripts/rules_heads_workflow.js`) |
| `gen3-fields.json` | the canonical gen-3 inputs, outputs, extensions, planned items and fixes, with every heads-pass name merged into one of them or dropped with a reason | by hand |
| `GEN3-FIELDS.md` | each canonical field's slot, shape and source, and the rules it serves | `gen3-fields` |
| `STATUS.md` | per-unit checklist and totals | `status` |
| `FINDINGS.md` | implementation notes: the rules that need new record fields, by family; why rules were skipped; the writers' Forge notes by kind and the suspected Forge departures to test first; the verifier's error types; limits of the scenarios as written | by hand, from the sidecars |
| `RUN-NOTES.md` | per chunk of units, what the scenario writer flagged (interactions whose outcome depends on Forge's implementation, rules skipped as unreachable) and what the verifier fixed | from the workflow results |

The subcommands are those of `python scripts/rules_catalogue.py`: `split`, `packets`,
`validate [units...] [--no-cards] [--warnings]` and `status`.

**A unit** is a three-digit section (`704`), except in sections 701 (keyword actions) and 702 (keyword
abilities), where each second-level rule is its own unit (`702.2` is deathtouch). Those two sections
hold a third of all rules. Rule ids follow the document: `704.5f`, `702.2b`, `100.1`.

## Phases

1. **Split.** The parser turns the rules text into `rules.jsonl` and `cr/`. This step is deterministic.
2. **Triage.** Every rule gets a category, an observability level, the record kinds that could show
   it, and a one-line observable outcome. Chapters 8 (multiplayer) and 9 (casual variants) are marked
   `out_of_scope` without an agent, because collection plays two-player games only.
3. **Scenarios.** Each mechanical or legality rule whose observability is not `none` gets a scenario
   with at least two variants. Alternatively it is listed under `skipped` with a reason, for example
   that another scenario in the unit already covers it.
4. **Verification.** A second agent independently checks each scenario file against the rules and
   against the Forge scripts of the cards it uses. It fixes what is wrong, then logs its changes and
   sets `verified`.

Later phases, not yet started, in order of preference for finding each rule's situations:

1. per-rule detectors that find each scenario's pattern, and its near miss, in the self-play corpus
   and count them;
2. directed forks that wait for a live board meeting a scenario's base setup and apply one
   variant's one-fact change, which needs each variant's `delta` classified into a fork operation;
3. puzzle-style starting states, only for rules neither reaches, which needs the `decisions` turned
   into structured actions.

Rule tests are then drawn from games kept out of training.

## Heads schema (`heads/<unit>.jsonl`)

One line per rule whose category is `mechanical` or `legality` and whose observability is not
`none`, checked against `MODEL-FIELDS.md`:

```json
{"id": "704.5f", "output": "trained", "outputs": ["zone_outcome"], "input": "complete", "inputs_missing": [], "note": ""}
```

- `output`: `trained` (an existing output is trained on the outcome), `recorded` (a record
  carries it, no output is trained on it), `missing` (no record carries it), `planned` (the gen-3
  design plans the output: timing or failed-check columns).
- `outputs`: existing output names as `MODEL-FIELDS.md` spells them, or tagged `record:<field or
  event:type>`, `new:<output>`, `planned:<output>`.
- `input`: `complete` when the model reads every fact that decides the outcome, else
  `incomplete`; `inputs_missing` lists those facts tagged `snapshot:` (recorded, not read),
  `planned:` (planned in gen-3) or `new:` (not recorded).

`gen3-fields` adds, from `gen3-fields.json`, the canonical ids each rule needs:
`gen3_inputs`, `gen3_outputs` (new outputs, extensions and planned verdict columns),
`gen3_fixes`, `gen3_dropped` (the names left out by design), and `learnable_after_gen3`. A name
is looked up in its own role, so `new:text_change` as a missing input and as an output can map to
different fields. Rerun it after editing either the registry or a heads line.

`validate --heads` checks completeness, the enums and the gen-3 ids. Rerun the pass whenever the input layout or
the heads change, to count what the change made learnable.

## Triage schema (`triage/<unit>.jsonl`)

One JSON object per line, one line for **every** rule in the unit, parents included:

```json
{"id": "704.5f", "category": "mechanical", "observability": "indirect", "record_kinds": ["resolution"], "outcome": "A creature with toughness 0 or less moves from the battlefield to its owner's graveyard during the next state-based check.", "note": ""}
```

`category`, exactly one of:

- `mechanical`: the decisions are already made and the engine decides an observable outcome. That
  outcome can be a zone change, damage, a life total, counters, power and toughness, control, whether
  a spell or ability resolves, fizzles or is countered, whether a trigger fires, what a replacement or
  prevention effect does, a state-based action, a layer result or combat damage. Legality *re-checked
  at resolution*, as in 608.2b, is mechanical, because it decides the resolution outcome.
- `legality`: the engine's verdict on whether an action is permitted: casting or activating timing,
  legal targets when chosen, attack and block restrictions and requirements, whether a cost can be
  paid. Playability records show these verdicts.
- `decision`: a choice or procedure a player carries out, such as choosing modes, targets or X,
  ordering triggers, assigning combat damage among blockers, choosing how to pay, or announcing. We
  assume these are already taken. A rule that describes a choice **and** its observable consequence
  is `mechanical`.
- `definitional`: defines a term, a characteristic, a card part or a naming convention, and has no
  outcome of its own. The rules that use it cover it.
- `out_of_scope`: cannot occur in a Forge two-player game, or has nothing to do with game mechanics.
  Examples are deck construction, sideboards, tournaments, multiplayer-only rules, casual variants,
  starting the game and mulligans, ante, subgames, physical card handling and silver-border rules.

`observability`, exactly one of:

- `direct`: a field of an existing effect record shows the outcome, such as a resolution's `outcome`
  or a playability verdict.
- `indirect`: a detector can derive it from fields that exist, usually the event list. An example is
  a state-based death after a resolution.
- `needs_new_fields`: the outcome is observable in a game, but no record field captures it today. An
  example is which layer applied an effect.
- `none`: nothing observable, which holds for every `decision`, `definitional` and `out_of_scope`
  rule.

`record_kinds`: the record kinds that could show the outcome, as a subset of the gen-2 record kinds
(`specs/023-ability-effect-model/contracts/record-schema.md`), and empty when `observability` is
`none`:

- `resolution`: a spell or ability resolving. Its cost half reports `resolved`, `fizzled`,
  `partially_fizzled` or `countered`, and its effect half lists the resulting events, including the
  state-based consequences of the check right after it.
- `rewrite`: a replacement or prevention effect applied to an event, with the incoming and outgoing
  event.
- `continuous`: the result of static effects on each permanent on a stable board: power and
  toughness, keywords, types, colors, name.
- `combat`: one combat damage step: attackers, blocks, assignment and the resulting events.
- `trigger`: whether a triggered ability fired on an event (`fired` true or false).
- `playability`: the legality verdicts for casting or activating each candidate (`can_play`,
  `has_legal_target`, legal targets, cost after adjustment) and for legal attackers and blockers.

`outcome`: one sentence naming the observable outcome. It is empty for `decision`, `definitional`
and `out_of_scope` rules. `note`: optional, for example why a borderline rule got its category.

## Scenario schema (`scenarios/<unit>.json`)

```json
{
  "unit": "704",
  "verified": false,
  "review": [],
  "skipped": [{"id": "704.3", "reason": "covered by 704.5f-1, whose resolution is followed by the check 704.3 describes"}],
  "scenarios": [
    {
      "id": "704.5g-1",
      "rules": ["704.5g"],
      "title": "Lethal damage destroys a creature during the state-based check",
      "record_kind": "resolution",
      "setup": {
        "active": "A",
        "step": "main1",
        "life": {"A": 20, "B": 20},
        "battlefield": {"A": ["Mountain"], "B": ["Grizzly Bears"]},
        "hand": {"A": ["Shock"]},
        "graveyard": {}, "library": {}, "exile": {},
        "notes": ""
      },
      "decisions": ["A casts Shock targeting Grizzly Bears, tapping Mountain.", "Both players pass; Shock resolves."],
      "variants": [
        {"label": "base", "polarity": "fires", "delta": "", "expected": "Grizzly Bears (2/2) is dealt 2 damage and moves to B's graveyard in the state-based check after Shock resolves."},
        {"label": "tougher", "polarity": "does_not_fire", "delta": "B's creature is Centaur Courser (3/3) instead of Grizzly Bears.", "expected": "Centaur Courser stays on the battlefield with 2 damage marked."}
      ],
      "cards": ["Mountain", "Shock", "Grizzly Bears", "Centaur Courser"],
      "forge_notes": ""
    }
  ]
}
```

- **Players** are `A` and `B` in a two-player game. `setup.active` names the active player, and
  `setup.step` is a step or phase name: `upkeep`, `draw`, `main1`, `beginning_of_combat`,
  `declare_attackers`, `declare_blockers`, `combat_damage`, `end_of_combat`, `main2`, `end`,
  `cleanup`.
- **Zones** map each player to a list. A list entry is a plain card name, or an object when the card's
  state is not the default: `{"card": "Grizzly Bears", "tapped": true, "counters": {"+1/+1": 1},
  "damage": 1, "attached_to": "Grizzly Bears", "summoning_sick": false, "face_down": true,
  "token": true}`. Use `stack` for objects already on the stack, bottom first, as
  `{"card": ..., "controller": "A", "targets": [...]}`. Anything else goes in `setup.notes`.
- **`decisions`** lists, in order, every choice and action the scenario needs: casts, activations,
  targets, modes, X, payments, attack and block declarations, damage assignment, responses and
  passes. Nothing is left to an AI.
- **`variants`**: at least two. The first is the base, with an empty `delta`. Every other variant
  changes **exactly one fact** of the setup or the decisions, named in `delta`, and the change flips
  or changes the outcome the rule decides. `polarity` is `fires` (the rule's effect happens),
  `does_not_fire` (the rule applies to the situation and its effect does not happen) or `other`
  (a third branch, for example "resolves but does nothing" next to "fizzles"). A scenario should have
  at least one `fires` and one `does_not_fire` variant.
- **`expected`** is the observable outcome only: zone moves, life totals, damage marked, counters,
  power and toughness, whether the stack object resolved, fizzled or was countered, whether a trigger
  fired, or whether an action was legal. It contains no reasoning.
- **Distinct mechanisms get distinct scenarios.** One case is a target that became illegal before
  resolution, so the spell fizzles under 608.2b. Red Elemental Blast's target stops being blue in
  response. Another case is a legal target that fails a condition checked on resolution, so the spell
  resolves and does nothing. Pyroblast targeting a non-blue spell is an example.
- **`cards`** lists every card the scenario names, lands included, spelled exactly as Forge's `Name:`
  line spells it. Prefer simple, well-known cards, vanilla where possible, so that no other ability
  interferes. `validate` checks every name against Forge's cardsfolder.
- **`forge_notes`**: optional, for anything Forge may implement differently or a setup Forge may have
  trouble reaching.
- **`rules`** may list several sibling rules when one scenario illustrates them all. A
  mechanical or legality rule that is neither in some scenario's `rules` nor in `skipped` fails
  `validate`.
- **`verified`** and **`review`** belong to the verification agent: `review` holds one string per
  change it made, and `verified` is true once it has checked the whole file.
