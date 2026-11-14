# Forks

Rules to respect when changing or diagnosing interventions, damage-step probes, protection forks or directed forks, and when registering a directed rule.

- What is specified: `specs/2026-10-30-ability-effect-model-gen3.md` § 6 (directed forks) and § 8.
- Code: `forge-connector/src/main/java/com/pricepredictor/connector/effects/` (`Forks`, `ForkFidelity`, `ForkBudget`, `InterventionScheduler`, `ProtectionFork`, `ForkPlayForward`) and its `directed/` package.
- Flag signatures of fork records: `record-format.md`.

## Rules for every fork

- Every fork is score-checked against the live game before any perturbation. A copy that already disagrees is measuring the copier.
- Both games are scored from one seeded random stream of the check's own (`Forks.score`). Before the second main phase the evaluator has the AI declare the coming combat in a copy, drawing from Forge's global `MyRandom`, so an exact copy scored from that stream simulated a different combat.
- The score weighs only part of the board, so every copy is also audited against the live game's snapshot by entity and field, lists compared as multisets. Differences are printed as `fork fidelity:` lines. The audit is a report: it keeps the fork.
- A discarded fork still counts against its budget. Otherwise a systematically failing copy retries until the game ends.
- At most two forks may target one real resolution. This is a fixed constant no flag can raise.
- The score check fails at the beginning-of-combat phase event, because the evaluator reads a combat Forge has not built yet. No scheduler takes a fork there (`Forks.scoreCheckable`).
- The real-versus-fork difference is computed at evaluation time and never becomes a training target.

## Interventional resolutions

- An intervention forks the game and forces an ability the AI never casts. It writes the effect half alone: there was no real activation, so there is no cost half to link to.
- The ability goes on the fork's stack directly, not through `ComputerUtil.handlePlayingSpellAbility`. The cost is usually the reason the record is missing, and routing through cost payment succeeds only on free plays; a first attempt that way produced nothing but land drops.
- The candidate is chosen at a phase boundary, never inside the AI's own evaluation of its candidates. Forking there copies a game mid-decision and re-enters the rules engine underneath it.
- The scheduler forces nothing at the untap boundary, where nobody can act.
- A line whose text branches on whose turn it is (`ConditionPlayerTurn$`, `ConditionOpponentTurn$`) is forced twice a game at most: once on its holder's own turn and once at a boundary of another player's turn, where it shows its other branch. `InterventionScheduler` offers those first at each boundary, because the other forced resolutions otherwise spend the whole per-game budget at the game's first boundary.

## Damage-step probes

- A probe re-runs one combat with a keyword stripped. It forks at the phase event, which fires before the turn-based action that deals the damage: the only moment the copy can diverge from a board where nothing has been dealt yet.
- The keyword comes off through the layer system, not the printed list, so a printed, equipped and until-end-of-turn keyword all go the way the rules would take them.
- The branch is held rather than written. It mirrors the real combat record, and that record has no id until the step the probe forked ahead of has finished; `BusBracketCollector.onCombatRecord` hands the id over. Reconstructing the pairing from board state would break on two identical-looking combats in one turn.
- A probe's payload names both the keyword and the carrier. The keyword alone is ambiguous on a board with two tramplers, and an evaluator reproducing the perturbation model-side would strip the wrong creature.
- `--probe-keywords` is the runtime switch, deliberately separate from the build decision gate 2 drives.

## Protection forks

- As a first main phase begins, a protection fork copies the game, gives one creature a rare protection through the `keyword` operation and plays the copy to the end of the turn through `ForkPlayForward`.
- The quality is one that an opposing non-land permanent, an opposing hand card or a card attached to the creature has, spelled as `rule_outcomes.ProtectionQuality.fragment` spells it.
- Its records are of any kind, cost halves linked to their effect halves included.
- It has a budget of its own: one a game, drawn at each first main phase with a creature out. The forced resolutions spend theirs before any creature is out.

## Directed forks

- A directed fork is played on, not stopped. `ForkPlayForward` plays from the phase the copy was taken in until its cleanup step leaves nothing to resolve or the game ends, with both seats choosing afresh.
- On stopping, the fork's bracket writes what no later event would close: the cleanup's turn-based actions and the held half, or everything the game's end writes when the fork ended the game. Forge publishes the game's end only from a match.
- It is recorded by a fork-bound `GameRecording` that installs no hook. The live game's `PatchedCollectors` hand every hook callback to the fork's collectors too, and each channel keeps its own game's. Dispatch is by game identity, because the AI evaluates on a thread of its own.
- The fork's lines are held and written only once it has played out, so a fork that throws leaves nothing.
- Its random seat draws from the game's fork seed. Nothing in a fork draws from the match's source, which would shift later games' seats.
- The copy carries the turn's pending state (delayed triggers, end-of-turn and leaves-the-battlefield commands, controller timestamps) through the `GameCopier` patch. A command whose closure holds a reference the copy cannot map is left out rather than carried with a live reference into the real game.
- The edit is applied before the fork's collectors start listening: the edits are silent to the rules but still publish bus events.
- The budget, the rule's one fork per game and the lifetime tally are all spent when the fork is taken, before the score check.
- `mirror_of` is set only when a live record was still open at the fork point, which a phase boundary has already closed.
- Untap-step `turn_action` records and directed forks are taken at the untap boundary as anywhere.

## Registering a directed rule

- A rule is registered in `FE/directed/DirectedRules.java` as a `DirectedRule`, and each rule is a class of its own in `FE/directed/rules/`, its precondition a hand-written read of the live board at a named phase boundary.
- `DirectedRulesContractTest` checks every registration against `rules/scenarios/*.json`. `RegisteredRulesTest` builds each scenario's base on a `TestBoard`: the precondition must bind there and not on a board missing the base, and the edited copy must differ from the live board by the variant's one fact.
- A precondition only finds the board where the rule can happen. It cannot force the live game's or the fork's decisions, so a precondition on a spell in hand holds only as long as the AI has not yet cast it.
- The supervisor's startup check (`directed_fork_error` in `sealed/infrastructure/effect_collection_flags.py`) requires, for each listed rule, a declared detector, a contrasting variant classified into an operation, and a registration.
- `sealed` may not import `effects`, so that check reads `rules/scenarios/` itself. `tests/unit/sealed/infrastructure/test_directed_fork_flags.py` pins it against `rules_catalogue_io` over the real catalogue.
