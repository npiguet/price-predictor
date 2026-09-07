r"""Snapshot of Forge's effect-API, trigger-type and bus-event names.

Checked in so the fast unit suite can assert event-vocabulary completeness with
no JVM and no ``../forge`` checkout. This is data, not logic: the mapping from
these names to the canonical event types lives in
:mod:`effects.domain.event_schema`, and
``tests/unit/effects/domain/test_event_schema_completeness.py`` asserts that
every name below is either mapped or explicitly excluded there.

Regenerate after a Forge upgrade with::

    python - <<'PY'
    import re
    from pathlib import Path
    game = Path("../forge/forge-game/src/main/java/forge/game")
    apis = sorted(p.stem.removesuffix("Effect")
                  for p in (game / "ability/effects").glob("*.java")
                  if p.stem.endswith("Effect"))
    body = (game / "trigger/TriggerType.java").read_text(encoding="utf-8")
    body = body.split("public enum TriggerType {", 1)[1]
    triggers = sorted(set(re.findall(r"^\s{4}(\w+)\(Trigger\w*\.class\)", body, re.M)))
    events = sorted(p.stem for p in (game / "event").glob("GameEvent*.java")
                    if p.stem != "GameEvent")
    body = (game / "replacement/ReplacementType.java").read_text(encoding="utf-8")
    body = body.split("public enum ReplacementType {", 1)[1]
    replacements = sorted(set(re.findall(r"^\s{4}(\w+)\(", body, re.M)))
    for name, values in [("EFFECT_APIS", apis), ("TRIGGER_TYPES", triggers),
                         ("GAME_EVENTS", events), ("REPLACEMENT_TYPES", replacements)]:
        print(name, len(values))
    PY

A Forge upgrade that adds an effect API makes the completeness test fail, which
is the point: a new API is a new thing an ability can do, and the corpus should
not silently stop describing it.

Captured from forge 2.0.15-SNAPSHOT.
"""

from __future__ import annotations

# forge-game/.../ability/effects/*Effect.java, suffix stripped (203 classes).
EFFECT_APIS: tuple[str, ...] = (
    "Abandon", "ActivateAbility", "AddPhase", "AddTurn", "AdvanceCrank", "Airbend",
    "AlterAttribute", "Amass", "Animate", "AnimateAll", "AssembleContraption",
    "AssignGroup", "Attach", "Balance", "BecomeMonarch", "BecomesBlocked", "BidLife",
    "BlankLine", "Blight", "Block", "Bond", "Branch", "Camouflage", "ChangeCombatants",
    "ChangeSpeed", "ChangeTargets", "ChangeText", "ChangeX", "ChangeZone",
    "ChangeZoneAll", "ChangeZoneResolve", "ChaosEnsues", "Charm", "ChooseCard",
    "ChooseCardName", "ChooseColor", "ChooseDirection", "ChooseEvenOdd",
    "ChooseGeneric", "ChooseNumber", "ChoosePlayer", "ChooseSector", "ChooseSource",
    "ChooseType", "ClaimThePrize", "Clash", "ClassLevelUp", "CleanUp", "Cloak", "Clone",
    "Connive", "ControlExchange", "ControlExchangeVariant", "ControlGain",
    "ControlGainVariant", "ControlPlayer", "ControlSpell", "CopyPermanent",
    "CopySpellAbility", "Counter", "CountersMove", "CountersMultiply", "CountersNote",
    "CountersProliferate", "CountersPut", "CountersPutAll", "CountersPutOrRemove",
    "CountersRemove", "CountersRemoveAll", "DamageAll", "DamageBase", "DamageDeal",
    "DamageEach", "DamagePrevent", "DamageResolve", "DayTime", "Debuff",
    "DelayedTrigger", "Destroy", "DestroyAll", "DetachedCard", "Detain", "Dig",
    "DigMultiple", "DigUntil", "Discard", "Discover", "Draft", "DrainMana", "Draw",
    "Earthbend", "Effect", "Encode", "EndCombatPhase", "EndTurn", "Endure", "Explore",
    "Fight", "FlipCoin", "FlipOntoBattlefield", "Fog", "GameDraw", "GameLoss",
    "GameWin", "Goad", "Haunt", "HealDamage", "Heist", "ImmediateTrigger", "Incubate",
    "Intensify", "InternalRadiation", "Investigate", "Learn", "LifeExchange",
    "LifeExchangeVariant", "LifeGain", "LifeLose", "LifeSet", "LookAt", "LosePerpetual",
    "MakeCard", "Mana", "ManaReflected", "Manifest", "ManifestBase", "ManifestDread",
    "Meld", "Mill", "MultiplePiles", "MustBlock", "Mutate", "OpenAttraction",
    "OwnershipGain", "PeekAndReveal", "Permanent", "PermanentCreature",
    "PermanentNoncreature", "Phases", "Planeswalk", "Play", "PlayLandVariant", "Poison",
    "PowerExchange", "Protect", "ProtectAll", "Pump", "PumpAll", "Radiation",
    "RearrangeTopOfLibrary", "Recruit", "Regenerate", "Regeneration",
    "RemoveFromCombat", "RemoveFromGame", "RemoveFromMatch", "ReorderZone", "Repeat",
    "RepeatEach", "Replace", "ReplaceCounter", "ReplaceDamage", "ReplaceMana",
    "ReplaceSplitDamage", "ReplaceToken", "RestartGame", "Reveal", "RevealHand",
    "ReverseTurnOrder", "RingTemptsYou", "RollDice", "RollPlanarDice", "RunChaos",
    "Sacrifice", "SacrificeAll", "Scry", "Seek", "SetInMotion", "SetState", "Shuffle",
    "SkipPhase", "SkipTurn", "StoreSVar", "Subgame", "Surveil", "SwitchBlock",
    "TakeInitiative", "Tap", "TapAll", "TapOrUntap", "TapOrUntapAll", "TextBoxExchange",
    "TimeTravel", "Token", "TwoPiles", "Unattach", "UnlockDoor", "Untap", "UntapAll",
    "Venture", "VillainousChoice", "Vote", "ZoneExchange",
)


# forge.game.trigger.TriggerType enum members (146).
TRIGGER_TYPES: tuple[str, ...] = (
    "Abandoned", "AbilityCast", "AbilityResolves", "AbilityTriggered", "Adapt",
    "Airbend", "Always", "Attached", "AttackerBlocked", "AttackerBlockedByCreature",
    "AttackerBlockedOnce", "AttackerUnblocked", "AttackerUnblockedOnce",
    "AttackersDeclared", "AttackersDeclaredOneTarget", "Attacks", "BecomeMonarch",
    "BecomeMonstrous", "BecomeRenowned", "BecomesCrewed", "BecomesPlotted",
    "BecomesSaddled", "BecomesTarget", "BecomesTargetOnce", "BlockersDeclared",
    "Blocks", "CaseSolved", "Championed", "ChangesController", "ChangesZone",
    "ChangesZoneAll", "ChaosEnsues", "ClaimPrize", "Clashed", "ClassLevelGained",
    "CollectEvidence", "CommitCrime", "ConjureAll", "Connives", "CounterAdded",
    "CounterAddedAll", "CounterAddedOnce", "CounterPlayerAddedAll", "CounterRemoved",
    "CounterRemovedOnce", "CounterTypeAddedAll", "Countered", "CrankContraption",
    "Crewed", "Cycled", "DamageAll", "DamageDealtOnce", "DamageDone", "DamageDoneOnce",
    "DamageDoneOnceByController", "DamagePreventedOnce", "Destroyed", "Devoured",
    "Discarded", "DiscardedAll", "Discover", "Drawn", "DungeonCompleted", "Earthbend",
    "ElementalBend", "Enlisted", "Evolved", "ExcessDamage", "ExcessDamageAll",
    "Exerted", "Exiled", "Exploited", "Explores", "FacesDilemma", "Fight", "FightOnce",
    "Firebend", "FlippedCoin", "Forage", "Foretell", "FullyUnlock", "GiveGift",
    "Immediate", "Investigated", "LandPlayed", "LifeGained", "LifeLost", "LifeLostAll",
    "LosesGame", "ManaAdded", "ManaExpend", "ManifestDread", "Mentored", "Milled",
    "MilledAll", "MilledOnce", "Mutates", "NewGame", "PayCumulativeUpkeep", "PayEcho",
    "PayLife", "Phase", "PhaseIn", "PhaseOut", "PhaseOutAll", "PlanarDice",
    "PlaneswalkedFrom", "PlaneswalkedTo", "Proliferate", "RingTemptsYou", "RolledDie",
    "RolledDieOnce", "RoomEntered", "Sacrificed", "SacrificedOnce", "Saddled", "Scry",
    "SearchedLibrary", "SeekAll", "SetInMotion", "Shuffled", "Specializes",
    "SpellAbilityCast", "SpellAbilityCopy", "SpellCast", "SpellCastOrCopy", "SpellCopy",
    "Stationed", "Surveil", "TakesInitiative", "TapAll", "Taps", "TapsForMana",
    "TokenCreated", "TokenCreatedOnce", "Trains", "Transformed", "TurnBegin",
    "TurnFaceUp", "Unattached", "UnlockDoor", "UntapAll", "Untaps", "VisitAttraction",
    "Vote", "Waterbend",
)


# forge.game.event.GameEvent* bus classes (58).
GAME_EVENTS: tuple[str, ...] = (
    "GameEventAddLog", "GameEventAnteCardsSelected", "GameEventAttackersDeclared",
    "GameEventBlockersDeclared", "GameEventCardAttachment", "GameEventCardChangeZone",
    "GameEventCardCounters", "GameEventCardDamaged", "GameEventCardDestroyed",
    "GameEventCardForetold", "GameEventCardModeChosen", "GameEventCardPhased",
    "GameEventCardPlotted", "GameEventCardRegenerated", "GameEventCardSacrificed",
    "GameEventCardStatsChanged", "GameEventCardTapped", "GameEventCombatChanged",
    "GameEventCombatEnded", "GameEventCombatUpdate", "GameEventDayTimeChanged",
    "GameEventDoorChanged", "GameEventFlipCoin", "GameEventGameFinished",
    "GameEventGameOutcome", "GameEventGameRestarted", "GameEventGameStarted",
    "GameEventLandPlayed", "GameEventManaBurn", "GameEventManaPool",
    "GameEventMulligan", "GameEventPlayerControl", "GameEventPlayerCounters",
    "GameEventPlayerDamaged", "GameEventPlayerLivesChanged", "GameEventPlayerPoisoned",
    "GameEventPlayerPriority", "GameEventPlayerRadiation",
    "GameEventPlayerShardsChanged", "GameEventPlayerStatsChanged", "GameEventRandomLog",
    "GameEventRollDie", "GameEventScry", "GameEventShuffle",
    "GameEventSnapshotRestored", "GameEventSpeedChanged", "GameEventSpellAbilityCast",
    "GameEventSpellRemovedFromStack", "GameEventSpellResolved",
    "GameEventSprocketUpdate", "GameEventSubgameEnd", "GameEventSubgameStart",
    "GameEventSurveil", "GameEventTokenCreated", "GameEventTurnBegan",
    "GameEventTurnEnded", "GameEventTurnPhase", "GameEventZone",
)


# forge.game.replacement.ReplacementType enum members (41).
REPLACEMENT_TYPES: tuple[str, ...] = (
    "AddCounter", "AssembleContraption", "AssignDealDamage", "Attached", "BeginPhase",
    "BeginTurn", "Cascade", "Connive", "CopySpell", "Counter", "CreateToken",
    "DamageDone", "DealtDamage", "DeclareBlocker", "Destroy", "Draw", "DrawCards",
    "Explore", "GainLife", "GameLoss", "GameWin", "Learn", "LifeReduced", "LoseMana",
    "Mill", "Moved", "PayLife", "PlanarDiceResult", "Planeswalk", "ProduceMana",
    "Proliferate", "RemoveCounter", "ReplacementType", "RollDice", "RollPlanarDice",
    "Scry", "SetInMotion", "Tap", "Transform", "TurnFaceUp", "Untap",
)
