"""The closed lists a detector declaration's free-text values are checked against.

``DetectorIndex.of`` refuses a declaration whose value no list holds, naming
the scenario, so a misspelt zone (``batlefield``), API (``Destory``), trigger
mode (``ChangeZone``), keyword (``Flyng``), card type, designation, face
state, counter type or cost symbol stops the command that builds the index
rather than tagging nothing. Forge's own names are its enums, checked in here
and pinned to ``../forge``'s sources by ``test_declared_values``; the wire's
are ``wire_closed_lists``' and the snapshot's enums.

Torch-free.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from effects.domain.forge_vocabularies import KEYWORD_COUNTERS, NAMED_COUNTERS, SUBTYPES
from effects.domain.keyword_names import DETAILED_KEYWORDS, normalize_keyword
from effects.domain.provenance import KEYWORD_API_TYPE
from effects.domain.rule_outcomes import CARD_TYPES
from effects.domain.state_snapshot import FaceState, normalize_counter_type
from effects.domain.wire_closed_lists import FORGE_ZONES

FORGE_API_TYPES: frozenset[str] = frozenset({
    "Abandon", "ActivateAbility", "AddOrRemoveCounter", "AddPhase", "AddTurn", "AdvanceCrank",
    "Airbend", "AlterAttribute", "Amass", "Animate", "AnimateAll", "AssembleContraption",
    "AssignGroup", "Attach", "Balance", "BecomeMonarch", "BecomesBlocked", "BidLife",
    "BlankLine", "Blight", "Block", "Bond", "Branch", "Camouflage", "ChangeCombatants",
    "ChangeSpeed", "ChangeTargets", "ChangeText", "ChangeX", "ChangeZone", "ChangeZoneAll",
    "ChangeZoneResolve", "ChaosEnsues", "Charm", "ChooseCard", "ChooseColor", "ChooseDirection",
    "ChooseEvenOdd", "ChooseNumber", "ChoosePlayer", "ChooseSector", "ChooseSource",
    "ChooseType", "ClaimThePrize", "Clash", "ClassLevelUp", "Cleanup", "Cloak", "Clone",
    "CompanionChoose", "Connive", "ControlPlayer", "ControlSpell", "CopyPermanent",
    "CopySpellAbility", "Counter", "DamageAll", "DamageResolve", "DayTime", "DealDamage",
    "Debuff", "DelayedTrigger", "Destroy", "DestroyAll", "Detain", "Dig", "DigMultiple",
    "DigUntil", "Discard", "Discover", "Draft", "DrainMana", "Draw", "EachDamage", "Earthbend",
    "Effect", "Encode", "EndCombatPhase", "EndTurn", "Endure", "ExchangeControl",
    "ExchangeControlVariant", "ExchangeLife", "ExchangeLifeVariant", "ExchangePower",
    "ExchangeTextBox", "ExchangeZone", "Explore", "Fight", "FlipCoin", "FlipOntoBattlefield",
    "Fog", "GainControl", "GainControlVariant", "GainLife", "GainOwnership", "GameDrawn",
    "GenericChoice", "Goad", "Haunt", "HealDamage", "Heist", "ImmediateTrigger", "Incubate",
    "Intensify", "InternalIgnoreEffect", "InternalLegendaryRule", "InternalRadiation",
    "Investigate", "Learn", "LookAt", "LoseLife", "LosePerpetual", "LosesGame", "MakeCard",
    "Mana", "ManaReflected", "Manifest", "ManifestDread", "Meld", "Mill", "MoveCounter",
    "MultiplePiles", "MultiplyCounter", "MustBlock", "Mutate", "NameCard", "OpenAttraction",
    "PeekAndReveal", "PermanentCreature", "PermanentNoncreature", "Phases", "Planeswalk",
    "Play", "PlayLandVariant", "Poison", "PreventDamage", "Proliferate", "Protection",
    "ProtectionAll", "Pump", "PumpAll", "PutCounter", "PutCounterAll", "Radiation",
    "RearrangeTopOfLibrary", "Recruit", "Regenerate", "Regeneration", "RemoveCounter",
    "RemoveCounterAll", "RemoveFromCombat", "RemoveFromGame", "RemoveFromMatch", "ReorderZone",
    "Repeat", "RepeatEach", "ReplaceCounter", "ReplaceDamage", "ReplaceEffect", "ReplaceMana",
    "ReplaceSplitDamage", "ReplaceToken", "RestartGame", "Reveal", "RevealHand",
    "ReverseTurnOrder", "RingTemptsYou", "RollDice", "RollPlanarDice", "RunChaos", "Sacrifice",
    "SacrificeAll", "Scry", "Seek", "SetInMotion", "SetLife", "SetState", "Shuffle",
    "SkipPhase", "SkipTurn", "StoreSVar", "Subgame", "Surveil", "SwitchBlock", "TakeInitiative",
    "Tap", "TapAll", "TapOrUntap", "TapOrUntapAll", "TimeTravel", "Token", "TwoPiles",
    "Unattach", "UnlockDoor", "Untap", "UntapAll", "Venture", "VillainousChoice", "Vote",
    "WinsGame",
})

FORGE_TRIGGER_MODES: frozenset[str] = frozenset({
    "Abandoned", "AbilityCast", "AbilityResolves", "AbilityTriggered", "Adapt", "Airbend",
    "Always", "Attached", "AttackerBlocked", "AttackerBlockedByCreature", "AttackerBlockedOnce",
    "AttackerUnblocked", "AttackerUnblockedOnce", "AttackersDeclared",
    "AttackersDeclaredOneTarget", "Attacks", "BecomeMonarch", "BecomeMonstrous",
    "BecomeRenowned", "BecomesCrewed", "BecomesPlotted", "BecomesSaddled", "BecomesTarget",
    "BecomesTargetOnce", "BlockersDeclared", "Blocks", "CaseSolved", "Championed",
    "ChangesController", "ChangesZone", "ChangesZoneAll", "ChaosEnsues", "ClaimPrize",
    "Clashed", "ClassLevelGained", "CollectEvidence", "CommitCrime", "ConjureAll", "Connives",
    "CounterAdded", "CounterAddedAll", "CounterAddedOnce", "CounterPlayerAddedAll",
    "CounterRemoved", "CounterRemovedOnce", "CounterTypeAddedAll", "Countered",
    "CrankContraption", "Crewed", "Cycled", "DamageAll", "DamageDealtOnce", "DamageDone",
    "DamageDoneOnce", "DamageDoneOnceByController", "DamagePreventedOnce", "DayTimeChanges",
    "Destroyed", "Devoured", "Discarded", "DiscardedAll", "Discover", "Drawn",
    "DungeonCompleted", "Earthbend", "ElementalBend", "Enlisted", "Evolved", "ExcessDamage",
    "ExcessDamageAll", "Exerted", "Exiled", "Exploited", "Explores", "FacesDilemma", "Fight",
    "FightOnce", "Firebend", "FlippedCoin", "Forage", "Foretell", "FullyUnlock", "GiveGift",
    "Immediate", "Investigated", "LandPlayed", "LifeGained", "LifeLost", "LifeLostAll",
    "LosesGame", "ManaAdded", "ManaExpend", "ManifestDread", "Mentored", "Milled", "MilledAll",
    "MilledOnce", "Mutates", "NewGame", "PayCumulativeUpkeep", "PayEcho", "PayLife", "Phase",
    "PhaseIn", "PhaseOut", "PhaseOutAll", "PlanarDice", "PlaneswalkedFrom", "PlaneswalkedTo",
    "Proliferate", "RingTemptsYou", "RolledDie", "RolledDieOnce", "RoomEntered", "Sacrificed",
    "SacrificedOnce", "Saddled", "Scry", "SearchedLibrary", "SeekAll", "SetInMotion",
    "Shuffled", "Specializes", "SpellAbilityCast", "SpellAbilityCopy", "SpellCast",
    "SpellCastOrCopy", "SpellCopy", "Stationed", "Surveil", "TakesInitiative", "TapAll", "Taps",
    "TapsForMana", "TokenCreated", "TokenCreatedOnce", "Trains", "Transformed", "TurnBegin",
    "TurnFaceUp", "Unattached", "UnlockDoor", "UntapAll", "Untaps", "VisitAttraction", "Vote",
    "Waterbend",
})

FORGE_REPLACEMENT_MODES: frozenset[str] = frozenset({
    "AddCounter", "AssembleContraption", "AssignDealDamage", "Attached", "BeginPhase",
    "BeginTurn", "Cascade", "Connive", "CopySpell", "Counter", "CreateToken", "DamageDone",
    "DealtDamage", "DeclareBlocker", "Destroy", "Draw", "DrawCards", "Explore", "GainLife",
    "GameLoss", "GameWin", "Learn", "LifeReduced", "LoseMana", "Mill", "Moved", "PayLife",
    "PlanarDiceResult", "Planeswalk", "ProduceMana", "Proliferate", "RemoveCounter", "RollDice",
    "RollPlanarDice", "Scry", "SetInMotion", "Tap", "Transform", "TurnFaceUp", "Untap",
})

FORGE_KEYWORDS: frozenset[str] = frozenset({
    "Absorb", "Affinity", "Afflict", "Afterlife", "Aftermath", "Amplify", "Annihilator",
    "Ascend", "Assist", "Aura swap", "Awaken", "Backup", "Banding", "Bands with other",
    "Bargain", "Battle cry", "Beam me up", "Bestow", "Blitz", "Bloodthirst", "Bushido",
    "Buyback", "Cascade", "Casualty", "Champion", "Changeling", "Choose a Background", "Cipher",
    "Companion", "Compleated", "Conspire", "Convoke", "Craft", "Crew", "Cumulative upkeep",
    "Cycling", "Dash", "Daybound", "Deathtouch", "Decayed", "Defender", "Delve", "Demonstrate",
    "Dethrone", "Devoid", "Devour", "Disguise", "Disturb", "Doctor's companion",
    "Double Strike", "Double agenda", "Double team", "Dredge", "Echo", "Embalm", "Emerge",
    "Enchant", "Encore", "Enlist", "Entwine", "Epic", "Equip", "Escalate", "Escape",
    "Eternalize", "Evoke", "Evolve", "Exalted", "Exploit", "Extort", "Fabricate", "Fading",
    "Fear", "Firebending", "First Strike", "Flanking", "Flash", "Flashback", "Flying",
    "For Mirrodin", "Foretell", "Fortify", "Freerunning", "Frenzy", "Fuse", "Gift", "Graft",
    "Gravestorm", "Harmonize", "Haste", "Haunt", "Hexproof", "Hidden agenda", "Hideaway",
    "Horsemanship", "Impending", "Improvise", "Increment", "Indestructible", "Infect", "Ingest",
    "Intimidate", "Job select", "Jump-start", "Kicker", "Landwalk", "Level up", "Lifelink",
    "Living Weapon", "Living metal", "Madness", "MayFlashCost", "MayFlashSac", "Mayhem",
    "Megamorph", "Melee", "Menace", "Mentor", "Miracle", "Mobilize", "Modular",
    "More Than Meets the Eye", "Morph", "Multikicker", "Mutate", "Myriad", "Nightbound",
    "Ninjutsu", "Offering", "Offspring", "Outlast", "Overload", "Paradigm", "Partner",
    "Partner with", "Persist", "Phasing", "Plot", "Poisonous", "Protection", "Prototype",
    "Provoke", "Prowess", "Prowl", "Rampage", "Ravenous", "Reach", "Read ahead", "Rebound",
    "Reconfigure", "Recover", "Reflect", "Reinforce", "Renown", "Replicate", "Retrace", "Riot",
    "Ripple", "Saddle", "Scavenge", "Shadow", "Shroud", "Skulk", "Sneak", "Soulbond",
    "Soulshift", "Space sculptor", "Specialize", "Spectacle", "Splice", "Split second", "Spree",
    "Squad", "Start your engines", "Starting intensity", "Station", "Storied", "Storm",
    "Strive", "Sunburst", "Surge", "Suspend", "Teamwork", "Tiered", "Toxic", "Training",
    "Trample", "Transfigure", "Transmute", "Tribute", "TypeCycling", "Umbra armor", "Undaunted",
    "Undying", "Unearth", "Unleash", "Vanishing", "Vigilance", "Ward", "Warp", "Web-slinging",
    "Wither",
})

#: Keyword lines Forge's scripts print (``K:Chapter``) that its ``Keyword``
#: enum does not list: an internal keyword the converter renders as a line.
KEYWORD_LINES_BEYOND_ENUM: frozenset[str] = frozenset({"Chapter", "Class", "Visit"})
#: The designations a gen-3 entity can hold (the connector's ``CardFacts``).
DESIGNATIONS: frozenset[str] = frozenset({
    "bestowed", "cloaked", "commander", "foretold", "goaded", "harnessed", "manifested",
    "monstrous", "phased_out", "plotted", "renowned", "ring_bearer", "saddled", "solved",
    "specialized", "suspected", "unearthed", "warped",
})
#: The kinds of Forge's Role tokens (``tokenscripts/role_<kind>.txt``): a
#: Role's kind is its name, no subtype of its own (``source_token_kind``).
ROLE_KINDS: frozenset[str] = frozenset({
    "cursed", "huntsman", "monster", "questing", "royal", "sorcerer", "virtuous", "wicked",
    "young_hero",
})
#: Card types a declaration may name, ``tribal`` being kindred's older name.
DECLARED_CARD_TYPES: frozenset[str] = frozenset({*CARD_TYPES, "tribal"})
FACE_STATES: frozenset[str] = frozenset(FaceState)
SUPERTYPES: frozenset[str] = frozenset({"basic", "legendary", "snow", "world", "ongoing",
                                        "elite", "host"})
_SUBTYPES = frozenset(s.lower() for s in SUBTYPES)
_COUNTERS = frozenset(normalize_counter_type(c) for c in (*NAMED_COUNTERS, *KEYWORD_COUNTERS,
                                                          "+1/+1", "-1/-1"))
#: A detail making a keyword of its own (``trample_over_planeswalkers``) is one too.
_KEYWORDS = frozenset({
    *(normalize_keyword(k) for k in (*FORGE_KEYWORDS, *KEYWORD_LINES_BEYOND_ENUM)),
    *DETAILED_KEYWORDS.values(),
})
_ZONES = frozenset(z.lower() for z in FORGE_ZONES if z) | {"any"}
_MANA = re.compile(r"^(\d+|[WUBRGCSXYZ]|[WUBRGC\d]+/[WUBRGCP](/P)?)$")
_SYMBOL_CLASSES = frozenset({"hybrid", "monohybrid", "phyrexian"})


def _refuse(family: str, name: str, value: Any, what: str) -> None:
    raise ValueError(f"{family}: {name} {value!r} is no {what}")


def keyword(family: str, name: str, value: Any) -> None:
    """A Forge keyword, as a keyword line displays it (``Flashback``,
    ``Umbra armor``), compared in ``keyword_names``' spelling."""
    if not isinstance(value, str) or normalize_keyword(value) not in _KEYWORDS:
        _refuse(family, name, value, "Forge keyword")


def api(family: str, name: str, value: Any) -> None:
    """A Forge ``ApiType``, or ``Keyword``, the API a sidecar gives a keyword's line."""
    if value not in FORGE_API_TYPES and value != KEYWORD_API_TYPE:
        _refuse(family, name, value, "Forge ApiType")


def trigger_mode(family: str, name: str, value: Any) -> None:
    if value not in FORGE_TRIGGER_MODES:
        _refuse(family, name, value, "Forge TriggerType")


def replacement_mode(family: str, name: str, value: Any) -> None:
    if value not in FORGE_REPLACEMENT_MODES:
        _refuse(family, name, value, "Forge ReplacementType")


def mode_of_rewrite(family: str, name: str, value: Any) -> None:
    """A rewrite's ``mode``: the replacement's Forge mode (``Moved``,
    ``DamageDone``) as its incoming event carries it."""
    if value not in FORGE_REPLACEMENT_MODES:
        _refuse(family, name, value, "Forge ReplacementType")


def forge_phase(family: str, name: str, value: Any) -> None:
    """A script's phase list (``Upkeep``, ``Main``, ``BeginCombat->EndCombat``)."""
    from effects.domain.rule_outcomes import forge_phases

    if not isinstance(value, str) or forge_phases(value) is None:
        _refuse(family, name, value, "Forge phase list")


def zone(family: str, name: str, value: Any) -> None:
    """A zone, or Forge's comma list of them, compared without case."""
    parts = value.split(",") if isinstance(value, str) else [value]
    if not all(isinstance(p, str) and p.strip().lower() in _ZONES for p in parts):
        _refuse(family, name, value, "zone")


def card_type(family: str, name: str, value: Any) -> None:
    if not isinstance(value, str) or value.lower() not in DECLARED_CARD_TYPES:
        _refuse(family, name, value, "card type")


def subtype(family: str, name: str, value: Any) -> None:
    if not isinstance(value, str) or value.lower() not in _SUBTYPES:
        _refuse(family, name, value, "Forge subtype")


def token_kind(family: str, name: str, value: Any) -> None:
    """A token's kind: a Forge subtype (``Treasure``), or a Role's name."""
    if not isinstance(value, str) or (value.lower() not in _SUBTYPES
                                      and value.lower().replace(" ", "_") not in ROLE_KINDS):
        _refuse(family, name, value, "token kind")


def designation(family: str, name: str, value: Any) -> None:
    if not isinstance(value, str) or value.lower() not in DESIGNATIONS:
        _refuse(family, name, value, "designation")


def face_state(family: str, name: str, value: Any) -> None:
    if value not in FACE_STATES:
        _refuse(family, name, value, "face state")


def counter_type(family: str, name: str, value: Any) -> None:
    if not isinstance(value, str) or normalize_counter_type(value) not in _COUNTERS:
        _refuse(family, name, value, "counter type")


def cost_symbol(family: str, name: str, value: Any) -> None:
    if not isinstance(value, str) or not (
            value.lower() in _SYMBOL_CLASSES or value.upper() in ("T", "Q")
            or _MANA.match(value.upper())):
        _refuse(family, name, value, "cost symbol")


def boolean(family: str, name: str, value: Any) -> None:
    if not isinstance(value, bool):
        _refuse(family, name, value, "boolean")


def integer(family: str, name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _refuse(family, name, value, "integer")


def one_of(*allowed: Any) -> Callable[[str, str, Any], None]:
    def check(family: str, name: str, value: Any) -> None:
        if value not in allowed:
            _refuse(family, name, value, f"one of {sorted(map(str, allowed))}")
    return check


def script_param(family: str, name: str, value: Any) -> None:
    """A parameter name, or a name -> value map."""
    if isinstance(value, str) and value:
        return
    if isinstance(value, Mapping) and value and all(isinstance(k, str) for k in value):
        return
    _refuse(family, name, value, "parameter name or name -> value map")


def check(family: str, params: Mapping[str, Any],
          checks: Mapping[str, Callable[[str, str, Any], None]]) -> None:
    """Every parameter of ``params`` ``checks`` names, against its check."""
    for name, value in params.items():
        if name in checks:
            checks[name](family, name, value)


def each(check_one: Callable[[str, str, Any], None]) -> Callable[[str, str, Any], None]:
    """``check_one`` over a value or a list of values."""
    def check_all(family: str, name: str, value: Any) -> None:
        values: Iterable[Any] = value if isinstance(value, list) else [value]
        for item in values:
            check_one(family, name, item)
    return check_all
