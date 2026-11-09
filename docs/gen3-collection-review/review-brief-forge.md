# Review brief: Forge `effect-record-hooks` branch (engine-side collection support)

Read the general brief first: `C:\Users\nicol\IdeaProjects\price-predictor\docs\gen3-collection-review\review-brief.md`.
Its rules, reading list, bug-hunting checklist and output format all apply. This file adds what is
specific to the Forge module.

## What this module is
`C:\Users\nicol\IdeaProjects\forge` is a fork of the MTG Forge engine. The branch `effect-record-hooks`
adds:
- engine hooks: static listener points (`EffectRecordHook`, `EffectRecordOutcomes`, `OutcomeCapture`,
  `AttributionScope`, `ReplacementObserver`, `CombatLegalityObserver`, and patched call sites in
  `AbilityUtils`, `TriggerHandler`, `PhaseHandler`, `Phase`, `AbilityManaPart`, `AiController`,
  `PlayerControllerAi`, several `*Effect` classes and others). The project's Java collectors in
  `price-predictor/forge-connector` register with these hooks, looked up reflectively through `PatchHooks`;
- copy support, so the AI's `GameCopier` can make faithful copies of a live game for counterfactual
  "forks": `GameCopier`, `GameCommandCopier`, and copy methods added to `Card`, `Player`, `Game`, `Zone`,
  `CardDamageHistory` and `Combat`.

## Seeing what the branch changed
You may use `git diff 53a103721d627ecb76a2ea52b2febe894844f288 effect-record-hooks -- <path>` (run in
`C:\Users\nicol\IdeaProjects\forge`) to see the branch's code changes to a file. Do NOT use `git log`,
`git show` or `git blame` (commit messages carry earlier verdicts). Each changed main-code file is listed in
your slice below.

## The questions that matter most
1. **Does every hook fire exactly once, at the moment its contract says, on every path?** Paths include:
   - the AI player, the random seat (`RandomSeatController` in the connector, a `PlayerControllerAi`
     subclass), and simulations or copies the AI makes while thinking (`GameSimulator`, `GameCopier`);
   - nested resolutions, sub-abilities, `WrappedAbility` triggers, copies of spells, replacement effects,
     mana abilities, state-based actions and the end of the game;
   - planned or what-if queries versus real actions.
2. **Do the patches change how the game plays?** A hook must observe, not alter. Look at every modified line
   outside pure listener calls. Check `Combat`, `StaticAbilityCantAttackBlock`, `AiController`,
   `PlayerControllerAi`, `Card`, `GameAction`, `Zone`, `TriggerHandler`, `PhaseHandler` and the `*Effect`
   files: changed conditions, order of operations, new exceptions, swallowed exceptions, changed return
   values, or a listener call placed where a throw or a side effect changes control flow.
3. **Are hook arguments right and safe?** Live objects versus last-known copies; parameter maps
   deep-copied where the contract says; objects from a copied game leaking into the live one or the
   reverse; static state shared between games (forks run while the live game is paused; the AI evaluates
   on its own thread).
4. **Copy support:** does each copy method carry the state faithfully, mapped into the copy, with no live
   object shared that the copy could mutate? Does copying change anything in the live game?

The connector review already found several problems at the hooks' semantics. Treat them as known, and
judge where the fix belongs (at the hook in Forge, or in the connector). See
`C:\Users\nicol\IdeaProjects\price-predictor\docs\gen3-collection-review\connector-findings.md`:
- A1: the clause-hook pair fires inside the parent clause's scope;
- A2: `WrappedAbility` passes through the clause hook twice;
- A3: the clause hook fires for clauses whose condition failed;
- C2: the mana hook fires after `manaPool.addMana`;
- the F section: copier gaps.

The fixing session will want a recommendation of where each should be fixed.
