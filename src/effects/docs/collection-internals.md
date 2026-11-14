# Collection internals

What a record's board and events mean, and the rules the Java collectors keep. Read before changing a collector, a hook, a cap flag or the random seat, and before interpreting what a record's board, events, duration or absence of an event means.

- What is specified: `specs/2026-10-30-ability-effect-model-gen3.md` § 8 (record moments are § 8.14).
- Code: `forge-connector/src/main/java/com/pricepredictor/connector/effects/` (`BusBracketCollector`, `HeldResolutionWindow`, `ManaChannel`, `RewriteChannel`, `StatDiffer`, `StaticSuppression`, `PatchHooks`, `CollectionCaps`).
- Forks: `forks.md`. The record format: `record-format.md`.

## Record moments

Each record's board is read just before the moment it describes, and its events are what the moment changed.

- A spell's cost half reads before its card moves to the stack; an activated ability's before its costs are paid. It is written at the cast event with no `outcome`. A triggered ability or a copy writes none.
- The effect half reads when its resolution starts (a mode's half at its first clause) and carries `outcome`: `resolved`, `fizzled`, `partially_fizzled` or `declined`. A fizzled or declined half has no events. `outcome` is `null` on a mana record and the rescue record.
- A spell's effect half takes its card's move off the stack after its own events, before the state-based deaths it holds.
- A countered item writes no effect half. The countering clause's half holds its `spell_countered` and the card's move off the stack.
- A cost half's events are listed but have not happened yet, so no per-entity output reads them.
- What a cast's move onto the stack set off, and what a bracket gathered before its resolution started, precede the effect half's board and are counted as dropped.
- A mana record reads before its mana ability's costs and holds a `tapped` event per permanent its cost tapped, then `mana_produced`. Mana a resolving spell or trigger makes is that half's `mana_produced`, never a mana record.
- A permanent entering the battlefield tapped gets a `tapped` event after its `zone_change`.
- Every library card a snapshot holds renders as it was in the library: untapped, no counters, controlled by its owner, face up.
- A rewrite record reads before the replacement runs.
- `conditions_met` is Forge's own answer, and a clause whose effect did not run files no event.
- On a degraded worker, with no hook, both boards fall back to the one read at the cast event.

## The hook inventory

- `PatchHooks.REQUIRED` lists every class, method and the record channel it feeds. It is load-bearing, not documentation: `AttributionMode` probes `PatchHooks.SUB_ABILITY`, `PatchedCollectors.CAN_SUPPRESS` reads the three `…Without` entries, and `PatchHooks.report()` prints what is missing at worker startup.
- A partly-applied patch is the state that matters. The mode still reads `patched`, because that is one hook's answer, while a channel the run meant to collect is quietly empty. Read the report line, not just the mode.
- The connector builds against stock Forge and reaches every hook by name. A typo in a hook's name therefore shows up at runtime as `degraded`, not at compile time, which is why every record carries the detected mode and `tests/integration/test_effects_patched.py` asserts it.
- The tests read the checkout rather than assuming it. One that pinned `degraded` would fail the day the patch landed, with nothing wrong.

## Where the events come from

- Forge's bus publishes an event of its own only for outcomes that have one: damage, counters, zone changes, taps. A pump, an anthem recompute, an animation and a colour change all publish only "this card's stats changed".
- `StatDiffer` remembers each card's computed characteristics and describes the difference. That is the source of `pt_change`, `keyword_change`, `type_change`, `color_change` and `control_change`. A draw, a discard and a mill are classified from the zone pair. Without these the "gets +2/+2 and gains flying" family resolved into an empty event list.
- A record's stat changes are changes after its own snapshot. The differ re-reads the battlefield whenever a bracket opened at a cast, a mode's half, a damage step or a fork's forced cast takes its snapshot, because Forge can apply a change long before publishing it. A change the re-read swallows reaches no record and is counted as dropped. It does not re-read while an ability resolves.
- An event's `duration` is read from the acting script's `Duration$`. An undeclared one is what Forge applies: instant for an outcome with no lasting state, end of turn for the APIs Forge ends at end of turn by default (Pump, Animate, Debuff, Protection, Effect and their `All` forms, ChangeText, ExchangePower), permanent otherwise.
- `attributed_to` is the resolving sub-ability's chain index, counted upward from the clause via `getParent()`, never downward from the bracket's root: Forge resolves a copy, and the two chains are not the same objects.

## The effect half and the state-based loop

- A resolution's effect half is held across the state-based loop that follows it and written at the loop's end signal, not when Forge reports the spell resolved. Forge runs state-based actions after that report, so a record written at it lost every death its own effects caused: a burn spell's target appeared damaged and never dead.
- While held, the half takes every battlefield-leaving move between the loop's begin and end signals, every pass included. The check before the resolution found nothing to do, so whatever this one finds the resolution caused.
- The held half takes deaths only, never a stat change.
- A trigger reaching the stack between two passes opens its own bracket without closing the held half. A dies trigger's records therefore hold none of the check's deaths; a death its own resolution later causes is the trigger's.
- Moves after the loop stay out, so a Treasure the spell made and later sacrificed for the next spell does not join it.
- A state-based death with no held half and no open damage step reaches no record. Example: a creature a -1/-1 counter cost leaves at 0 toughness. The fork sink refuses it the same way.
- On a modal resolution a death goes to the mode half whose events named the permanent, otherwise to the last half.
- A combat damage step's record holds its own check: the loop's end closes it, not a trigger put on the stack inside the loop.
- On a worker without the loop hook the half falls back to closing at the next action, and a move counts as the check's while the tracker is frozen.
- Interventional forks run the check themselves.

## Dropped events

- Every event that reaches no record is counted per game by event type and engine phase (`FE/DroppedEvents.java`): one at a player's action at priority that is no cast, an orphan discarded at game end, a state-based death nothing took.
- The counts are written per game, not at exit, because workers are killed with their process tree and run no shutdown hook. File format: `record-format.md`.

## Continuous records

- A `continuous` record's snapshot is the board without its own static. The label is what the static does, so a snapshot carrying the static's work would hand the model the answer with the question.
- The board is recomputed with the static's layer entries dropped, not stripped of what it contributed. A creature printed with trample keeps trample when the anthem granting it is removed; subtracting the contribution would describe a board that never existed. Expect a few percent of contributions to survive their own snapshot for that reason.
- Every channel of the entity reads that board through one object, `StaticSuppression`, and no static lists what it affects.
- A contribution carries `set_pt` only when its static sets a base and `controller` only when it gives control, so a static whose only change is a set base (Godhead of Awe) or control (Control Magic) is a record of its own.
- A checkout whose patch predates the `…Without` methods collects no continuous records at all rather than leaking ones.
- Continuous records are read at every phase boundary but the untap step's. No player gets priority there, and the live game has not yet re-applied its statics for the new turn, so a record there could describe a board the rules do not hold.
- `types` and `colors` are each one flat token list: a bare token is added, `-x` removed, and a lone `=` marks a line the static sets outright.
- Subtypes and supertypes are recorded and then dropped at target derivation, because `types_gained` and `types_lost` cover the core types only. The record is the archive; a head reads a subset.
- A `=` on a gen-1 or gen-2 record leaves `types_lost` unset. On a gen-3 record the lost fields are the entity's own types and colours on the board without the static, less what the `=` sets.

## Token script ids

- `token_script_id` changed value space on 2026-09-17. Before, it was the token's printed name; after, the token's script stem (`c_a_food_sac`), falling back to the printed name only when the script cannot be named. A reader spanning shards from both sides sees both value spaces in one field.

## Caps and sampling

- The three collectors that repeat themselves (continuous effects, the AI's blocker queries, mana) are capped by identity, not by count. The repeats describe a board that did not move. A count cap would keep the first N and then nothing, concentrating a card's records in whichever games came first.
- `--mana-cap` is keyed on the mana produced as well as the ability, so a dual land records each colour it makes.
- Mana is reservoir-sampled, not taken first. A land's first tap is turn one against an empty board, so keeping it would make every mana record describe the same early game.
- Mana records are therefore held until `close()`, and a JVM that dies loses that game's reservoir. The state snapshot is taken when the mana was made, not at flush. A triggered mana ability's mana joins the record of the activation that set it off.
- Both samplers scramble their seed through SplitMix64's finalizer first. Games are seeded consecutively, and a linear congruential generator's first output is nearly a linear function of its seed, so without it consecutive games take nearly the same first sampling decision.
- The eight cap flags are per-worker-process quantities the Python supervisor cannot observe. They reach the JVM as `-Deffect.*` system properties, and `CollectionCaps.fromSystemProperties()` falls back to a default rather than failing.
- All collection flags are defined once, in `sealed.infrastructure.effect_collection_flags`, which both CLIs import (`sealed` never imports `effects`). `tests/unit/effects/domain/test_collection_caps.py` pins its defaults against `CollectionCaps` and both against the property literals in `CollectionCaps.java`. A property one side reads and the other never writes is invisible from both.
- `--probe-keywords` is spelled as the JVM compares it (lower case, spaces as `_`), and a keyword gate 2 does not check is refused at parse time.
- An empty `--probe-keywords` takes no probe fork at all, whatever `--probes-per-game` says. Every collecting command prints its probe state before the first worker spawns, and `validate-corpus` reports the probe count as a watched number. The JVM's own fallback, with no property set, is the empty list; every supervisor sets the property.
- Snapshot depth is not a flag. `SnapshotBuilder` renders all four inclusion tiers into every record. A depth that varied by collector made `state.tiers` a proxy for how a record was collected, which the schema bars from the model. The field is still written because older shards carry shorter vectors; readers treat an absent tier as uncollected, not empty.

## Random seat

- Under `--random-seat-share F --random-seat-probability P`, one seat in a share `F` of matches is a `RandomSeatController`, a `PlayerControllerAi` subclass. Forge itself is unchanged.
- It exists because the AI never casts some cards at all. A random seat casts them for real, where an intervention forks past their cost and choices.
- At every priority, attack, block and target decision it draws uniformly from the legal options with probability `P`. It never passes while a non-mana ability is playable, and it leaves the land drop, mana payment and mulligans to the Forge AI.
- Its matches write effect records only, never a match-outcome or cards-played row.
- A share above 0 without a probability is refused.
- Every real decision is written and no what-if is. The AI re-asks one combat many times while planning, and a query drawn before de-duplication survives in proportion to how often it was asked. Blocks against attacks the AI rejected reach the corpus instead as real decisions against the random seat.

## Dead fields

- A collector that writes `"blocking":[]` produces records byte-identical to ones for a creature that genuinely is not blocking. The shape is right, the contract tests pass, and the field is dead. `field-coverage` exists to name every field a corpus never varied.
- `KNOWN_CONSTANT_FIELDS` lists the fields known dead. The integration test fails when one of them starts carrying data, so the list can only shrink. It is asserted in that direction only, because a field that is merely rare looks exactly like one that is unwired.
- `pt_duration` and `type_color_duration` are the two head fields no target path writes.
