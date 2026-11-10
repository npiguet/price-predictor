# WP1 result: the Forge hooks the connector must wire (Forge 8f105fb36cd)

WP1 of `FIX-PLAN.md` changed and added these hooks on `effect-record-hooks` (commits c59ec2ad24e …
8f105fb36cd, installed). WP3 wires the connector to them.

## Changed and new hooks
- `AbilityUtils.EffectRecordClauseListener.onClauseResolved(SpellAbility clause, ClauseRun run)`:
  argument 1 was a boolean `threw`; it is now the enum `forge.game.ability.ClauseRun` (`RAN`,
  `CONDITIONS_UNMET`, `UNLESS_COST_DECIDED`, `THREW`, method `ran()`). Read it by position as an enum
  `name()`. Until WP3, the connector's `Boolean.TRUE.equals(args.optional(1))` is always false.
- Clause hook order (A1): RESOLVED fires after the clause's own effect and before its sub-abilities,
  so chains, charm modes included, are sequential. A wrapper (`sa.isWrapper()`) sends no notification
  (A2); the inner ability does.
- `ReplacementHandler.EffectRecordListener.onReplacing(ReplacementEffect, Map<AbilityKey,Object>)`, a
  default method fired before the replacement runs. Each `onReplacing` gets exactly one
  `onReplacement`, nested strictly: keep a stack, the innermost open one closes first.
- `MagicStack.setEffectRecordCastListener(EffectRecordCastListener)` → `onCastStarting(SpellAbility)`.
  - A spell: fired in `GameAction.moveToStack(Card, SpellAbility, Map)` just before the move, when the
    cause is a spell, the card is its own host, not a copy and not already on the stack. That method
    is the only funnel to the stack (AI, human path, `playStack`, `playNoStack`, `addAndUnfreeze`,
    `GameSimulator`, the random seat's `RandomSpellPlayer`, forks).
  - An activated non-mana ability: at the top of `CostPayment.payComputerCosts` when the decision
    maker is not an effect. Triggered, mana and effect-requested payments are excluded. A cast can
    still be abandoned after this; a completed one is told by the stack push that follows.
  - `MagicStack.effectRecordCastStarting(SpellAbility)` is Forge's own notifier; never call or require it.
- `MagicStack.setEffectRecordResolutionListener(…)` → `onResolutionStarting(SpellAbility sa, boolean
  fizzled)`: in `resolveStack` after the fizzle check, `copyLastState` and the controller change,
  before the fizzle branch and `AbilityUtils.resolve`; the item is still on the stack; once per item,
  before a trigger's inner first clause. Per-mode moments need no hook: with A1, each mode's first
  clause RESOLVING comes after the earlier modes finished.
- `CostPayment.setEffectRecordPaymentListener(…)` → `onPayingCosts(SpellAbility, Player payer,
  boolean effect)` and `onCostsPaid(SpellAbility, Player payer, boolean paid)`, nested strictly;
  `paid` false on a failed decision, part or throw. A mana ability's `onManaProduced` comes after its
  own `onCostsPaid`, when the paid ability resolves (from `addAndUnfreeze`): keep the pending mana
  snapshot keyed by ability identity until `onManaProduced`, drop it on `paid == false`. Triggered
  mana abilities (Wild Growth) are never paid for and need the fallback. `predictMana`,
  `canPayCost`, `canPayManaCost` fire nothing.
- H1-4: the trigger `fired` flag is `runnable && the activation limit still has room`, counted over
  earlier waiting events; Forge's own collected list is unchanged.

## `PatchHooks` additions
- Owners: `MAGIC_STACK = "forge.game.zone.MagicStack"`, `COST_PAYMENT = "forge.game.cost.CostPayment"`.
- Callbacks: `ON_CAST_STARTING("onCastStarting", 1)`, `ON_RESOLUTION_STARTING("onResolutionStarting", 2)`,
  `ON_PAYING_COSTS("onPayingCosts", 3)`, `ON_COSTS_PAID("onCostsPaid", 3)`, `ON_REPLACING("onReplacing", 2)`.
- Hooks, each in `REQUIRED` (20 → 23): `CAST_LISTENER` (`setEffectRecordCastListener`,
  `ON_CAST_STARTING`), `RESOLUTION_LISTENER` (`setEffectRecordResolutionListener`,
  `ON_RESOLUTION_STARTING`), `PAYMENT_LISTENER` (`setEffectRecordPaymentListener`, `ON_PAYING_COSTS`,
  `ON_COSTS_PAID`).
- `REPLACEMENT_LISTENER`'s callbacks become `ON_REPLACING, ON_REPLACEMENT`.
- `ON_CLAUSE_RESOLVED` keeps arity 1; argument 1 is the `ClauseRun` enum. `ClauseRun` is detectable by
  `find("forge.game.ability.ClauseRun", "values")`; `onReplacing` has nothing static to find.

## Testing facts
- `MagicStack.add` stores a copy of an activated ability: use `peekAbility()` for the stacked object.
- A mana ability's mana is produced after `payComputerCosts` returns, from `addAndUnfreeze`.
- Known limits (accepted): H1-4 can misreport under Panharmonicon with an activation limit above 1, or
  when `runSingleTrigger` returns before `triggerRun()`.
