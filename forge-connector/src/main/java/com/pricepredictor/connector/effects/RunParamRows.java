package com.pricepredictor.connector.effects;

import forge.game.replacement.ReplacementEffect;

import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.ModeRow.Sign.AS_READ;
import static com.pricepredictor.connector.effects.ModeRow.Sign.GAIN;
import static com.pricepredictor.connector.effects.ModeRow.Sign.LOSS;
import static com.pricepredictor.connector.effects.RunParamSlots.count;
import static com.pricepredictor.connector.effects.RunParamSlots.counters;
import static com.pricepredictor.connector.effects.RunParamSlots.lifeDelta;
import static com.pricepredictor.connector.effects.RunParamSubjects.about;

/**
 * Each Forge trigger and replacement mode's row: its event type, its subject
 * and the slots it fills, each read off the keys that mode's run-parameter
 * maps hold (the {@code AbilityKey}s Forge puts at the mode's call sites).
 *
 * <p>Deliberately partial. The vocabulary names observable outcomes and
 * Forge names engine hooks, so only some pairs are the same thing; every
 * other mode is listed as unread and carries its mode alone, as a {@code
 * state_flag_change} ({@link #UNREAD}). Every {@code TriggerType} and {@code
 * ReplacementType} is a row or listed, never both, so a mode Forge adds is
 * never missing unsaid ({@code RunParamNormalizerTest}).
 */
final class RunParamRows {

    private RunParamRows() {
    }

    /**
     * The row of this trait's mode: its own row of the trigger table or of the
     * replacement table, else {@link #UNREAD}.
     *
     * <p>Checked for null before the lookup: a trait whose {@code getMode()}
     * is absent — a hook whose signature drifted, so the argument is not the
     * trait we expect — must not take an NPE inside a dynamic proxy, which the
     * engine would surface as an {@code UndeclaredThrowableException} in the
     * middle of {@code canRunTrigger}.
     */
    static ModeRow rowOf(Object trait, String mode) {
        Map<String, ModeRow> table = trait instanceof ReplacementEffect
                ? REPLACEMENT_MODES : TRIGGER_MODES;
        ModeRow row = mode == null ? null : table.get(mode);
        return row != null ? row : UNREAD;
    }

    /**
     * A mode with no row: the generic type, about the first entity its map
     * names under the usual subject keys, and nothing else read.
     */
    static final ModeRow UNREAD = ModeRow.of(EventType.STATE_FLAG_CHANGE, RunParamSubjects.UNREAD);

    private static final ModeRow.Slot[] DAMAGE = {
        RunParamSlots.DAMAGE_AMOUNT, RunParamSlots.COMBAT, RunParamSlots.DAMAGE_SOURCE};

    /** A replacement's map is about the object or player it would affect. */
    private static final ModeRow.Subjects AFFECTED = about("Affected");

    /** A per-card or per-counter trigger: the card, else (Forge's player-side call sites) the player. */
    private static final ModeRow.Subjects CARD_OR_PLAYER = about("Card", "Player");

    /** Forge's trigger modes ({@code TriggerType} names) read here. */
    static final Map<String, ModeRow> TRIGGER_MODES = Map.ofEntries(
            // Card, else (a map without the live object) its last-known copy.
            Map.entry("ChangesZone", ModeRow.of(EventType.ZONE_CHANGE,
                    about("Card", "CardLKI"), RunParamSlots.ZONES)),
            Map.entry("Destroyed", ModeRow.of(EventType.DESTROYED,
                    about("Card"), RunParamSlots.REGENERABLE)),
            Map.entry("Sacrificed", ModeRow.of(EventType.SACRIFICED, CARD_OR_PLAYER)),
            Map.entry("DamageDone", ModeRow.of(EventType.DAMAGE_DEALT,
                    about("DamageTarget"), DAMAGE)),
            // One source's damage to every target at once: DamageSource and a
            // DamageMap of target to amount (CardDamageTable.triggerDamageDoneOnce).
            Map.entry("DamageDealtOnce", ModeRow.of(EventType.DAMAGE_DEALT,
                    RunParamSubjects::damageMapTargets, RunParamSlots.DAMAGE_MAP_TOTAL,
                    RunParamSlots.COMBAT, RunParamSlots.DAMAGE_SOURCE)),
            // All of one damage event: a CardDamageTable of source, target and amount.
            Map.entry("DamageAll", ModeRow.of(EventType.DAMAGE_DEALT,
                    RunParamSubjects::damageTableTargets, RunParamSlots.DAMAGE_TABLE_TOTAL,
                    RunParamSlots.COMBAT, RunParamSlots.DAMAGE_TABLE_SOURCE)),
            Map.entry("DamagePreventedOnce", ModeRow.of(EventType.DAMAGE_PREVENTED,
                    about("DamageTarget"), RunParamSlots.DAMAGE_AMOUNT, RunParamSlots.DAMAGE_SOURCE)),
            // One target's damage from every source at once: a DamageMap of source to amount.
            Map.entry("DamageDoneOnce", ModeRow.of(EventType.DAMAGE_DEALT,
                    about("DamageTarget"), RunParamSlots.COMBAT, RunParamSlots.SOURCES_DAMAGE_MAP)),
            Map.entry("ChangesZoneAll", ModeRow.of(EventType.ZONE_CHANGE,
                    RunParamSubjects::zoneTableCards, RunParamSlots.ZONE_TABLE_MOVE)),
            // The creature sacrificed to the exploit, as it last was.
            Map.entry("Exploited", ModeRow.of(EventType.SACRIFICED, about("Exploited"))),
            Map.entry("DiscardedAll", ModeRow.of(EventType.CARD_DISCARDED,
                    about("Cards"), count(RunParamAmount.sizeOf("Cards")))),
            Map.entry("RolledDie", ModeRow.of(EventType.DICE_ROLLED,
                    about("Player"), RunParamSlots.ONE_DIE)),
            Map.entry("RolledDieOnce", ModeRow.of(EventType.DICE_ROLLED,
                    about("Player"), RunParamSlots.DICE)),
            // One player's counters on one object, every type at once.
            Map.entry("CounterPlayerAddedAll", ModeRow.of(EventType.COUNTER_CHANGE,
                    about("Object"), counters(RunParamAmount.NONE, GAIN))),
            // Drawn, Discarded, Milled and TokenCreated fire once per card;
            // Drawn's Number and TokenCreated's Num are the turn's ordinals,
            // not counts.
            Map.entry("Drawn", ModeRow.of(EventType.CARD_DRAWN, CARD_OR_PLAYER, count(RunParamAmount.ONE))),
            Map.entry("Discarded", ModeRow.of(EventType.CARD_DISCARDED,
                    CARD_OR_PLAYER, count(RunParamAmount.ONE))),
            Map.entry("Milled", ModeRow.of(EventType.CARD_MILLED, CARD_OR_PLAYER, count(RunParamAmount.ONE))),
            Map.entry("LifeGained", ModeRow.of(EventType.LIFE_CHANGE, about("Player"), lifeDelta(GAIN))),
            Map.entry("LifeLost", ModeRow.of(EventType.LIFE_CHANGE, about("Player"), lifeDelta(LOSS))),
            Map.entry("PayLife", ModeRow.of(EventType.LIFE_CHANGE, about("Player"), lifeDelta(LOSS))),
            Map.entry("CounterAdded", ModeRow.of(EventType.COUNTER_CHANGE,
                    CARD_OR_PLAYER, counters(RunParamAmount.ONE, GAIN))),
            Map.entry("CounterAddedOnce", ModeRow.of(EventType.COUNTER_CHANGE,
                    CARD_OR_PLAYER, counters(RunParamAmount.key("CounterAmount"), GAIN))),
            Map.entry("CounterRemoved", ModeRow.of(EventType.COUNTER_CHANGE,
                    CARD_OR_PLAYER, counters(RunParamAmount.ONE, LOSS))),
            Map.entry("CounterRemovedOnce", ModeRow.of(EventType.COUNTER_CHANGE,
                    CARD_OR_PLAYER, counters(RunParamAmount.key("CounterAmount"), LOSS))),
            // The card tapped; Player is the tapper and Attacker whether it attacked.
            Map.entry("Taps", ModeRow.of(EventType.TAPPED, about("Card", "Player", "Attacker"))),
            Map.entry("TapsForMana", ModeRow.of(EventType.MANA_PRODUCED,
                    about("Card"), RunParamSlots.MANA)),
            Map.entry("Untaps", ModeRow.of(EventType.UNTAPPED, about("Card"))),
            // The Aura or Equipment; AttachTarget and Object name what it hangs or hung on.
            Map.entry("Attached", ModeRow.of(EventType.ATTACHED,
                    about("AttachSource"), RunParamSlots.ATTACHED_TO)),
            Map.entry("Unattached", ModeRow.of(EventType.UNATTACHED,
                    about("AttachSource"), RunParamSlots.DETACHED_FROM)),
            // MagicStack puts CardLKI, Activator and SpellAbility and no Card at
            // all: the last-known copy is the only name the map has for the
            // spell that was cast.
            Map.entry("SpellCast", ModeRow.of(EventType.SPELL_CAST, about("CardLKI"))),
            Map.entry("SpellAbilityCast", ModeRow.of(EventType.SPELL_CAST, about("CardLKI"))),
            Map.entry("Countered", ModeRow.of(EventType.SPELL_COUNTERED, about("Card"))),
            Map.entry("TokenCreated", ModeRow.of(EventType.TOKEN_CREATED,
                    CARD_OR_PLAYER, count(RunParamAmount.ONE))),
            // The tokens made at once (CardZoneTable.triggerTokenCreatedOnce).
            Map.entry("TokenCreatedOnce", ModeRow.of(EventType.TOKEN_CREATED,
                    about("Cards"), count(RunParamAmount.sizeOf("Cards")))),
            Map.entry("AttackersDeclared", ModeRow.of(EventType.ATTACKERS_DECLARED,
                    about("Attackers"), RunParamSlots.DEFENDER)),
            // Attacks is one creature being declared as an attacker, which is
            // exactly what attackers_declared names, and one of the heaviest
            // trait-derived modes: as attackers_declared it carries the defender.
            Map.entry("Attacks", ModeRow.of(EventType.ATTACKERS_DECLARED,
                    about("Attacker"), RunParamSlots.DEFENDER)),
            // A Blocks map holds Blocker and Attackers (BlockEffect,
            // PhaseHandler), or Attacker and Blocker (SwitchBlockEffect, whose
            // event this reads as about the attacker).
            Map.entry("Blocks", ModeRow.of(EventType.BLOCKERS_DECLARED,
                    about("Attacker", "Blocker", "Attackers"), RunParamSlots.BLOCKED)),
            // Retargeted: the subject of an AttackerBlocked event is the
            // attacker, not the blocker, so blockers_declared would put the
            // wrong creature in the slot named "blocked". became_blocked is the
            // type whose subject is the attacker and whose param is the blockers.
            Map.entry("AttackerBlocked", ModeRow.of(EventType.BECAME_BLOCKED,
                    about("Attacker"), RunParamSlots.BLOCKERS)),
            Map.entry("AttackerBlockedByCreature", ModeRow.of(EventType.BECAME_BLOCKED,
                    about("Attacker"), RunParamSlots.BLOCKERS)),
            Map.entry("TurnFaceUp", ModeRow.of(EventType.FACE_CHANGE, about("Card"), RunParamSlots.FACE)),
            Map.entry("Transformed", ModeRow.of(EventType.FACE_CHANGE,
                    about("Card"), RunParamSlots.FACE)),
            Map.entry("Shuffled", ModeRow.of(EventType.LIBRARY_SHUFFLED, about("Player"))),
            Map.entry("Scry", ModeRow.of(EventType.CARD_LOOKED_AT,
                    about("Player"), count(RunParamAmount.key("ScryNum")))),
            // Player.surveil puts FirstTime and its caller's zone-table keys and
            // no number of cards, so a surveil carries no count.
            Map.entry("Surveil", ModeRow.of(EventType.CARD_LOOKED_AT, about("Player"))),
            Map.entry("PhaseOut", ModeRow.of(EventType.PHASED,
                    about("Card"), RunParamSlots.phased(true))),
            Map.entry("PhaseIn", ModeRow.of(EventType.PHASED,
                    about("Card"), RunParamSlots.phased(false))));

    /**
     * Forge's replacement modes ({@code ReplacementType} names) read here.
     *
     * <p>A replacement mode with no row carries nothing but its mode — and an
     * event with no value in it cannot differ from itself, so the record reads
     * as a rewrite of nothing however much the replacement changed: a rewrite
     * that halved a draw would be dropped as an identity pair.
     */
    static final Map<String, ModeRow> REPLACEMENT_MODES = Map.ofEntries(
            Map.entry("Moved", ModeRow.of(EventType.ZONE_CHANGE, AFFECTED, RunParamSlots.ZONES)),
            Map.entry("Destroy", ModeRow.of(EventType.DESTROYED, AFFECTED, RunParamSlots.REGENERABLE)),
            Map.entry("DamageDone", ModeRow.of(EventType.DAMAGE_DEALT, AFFECTED, DAMAGE)),
            Map.entry("DealtDamage", ModeRow.of(EventType.DAMAGE_DEALT, AFFECTED, DAMAGE)),
            Map.entry("Draw", ModeRow.of(EventType.CARD_DRAWN, AFFECTED, count(RunParamAmount.ONE))),
            Map.entry("DrawCards", ModeRow.of(EventType.CARD_DRAWN,
                    AFFECTED, count(RunParamAmount.key("Number")))),
            Map.entry("Mill", ModeRow.of(EventType.CARD_MILLED, AFFECTED, count(RunParamAmount.key("Number")))),
            Map.entry("GainLife", ModeRow.of(EventType.LIFE_CHANGE, AFFECTED, lifeDelta(GAIN))),
            Map.entry("LifeReduced", ModeRow.of(EventType.LIFE_CHANGE, AFFECTED, lifeDelta(LOSS))),
            Map.entry("PayLife", ModeRow.of(EventType.LIFE_CHANGE, AFFECTED, lifeDelta(LOSS))),
            Map.entry(RunParamNormalizer.ADD_COUNTER, ModeRow.of(EventType.COUNTER_CHANGE,
                    AFFECTED, counters(RunParamAmount.NONE, GAIN))),
            Map.entry("RemoveCounter", ModeRow.of(EventType.COUNTER_CHANGE,
                    AFFECTED, counters(RunParamSlots::removal, AS_READ))),
            Map.entry("CreateToken", ModeRow.of(EventType.TOKEN_CREATED,
                    AFFECTED, count(RunParamSlots::affectedTokens))),
            Map.entry("Scry", ModeRow.of(EventType.CARD_LOOKED_AT, AFFECTED, count(RunParamAmount.key("Num")))),
            Map.entry("Untap", ModeRow.of(EventType.UNTAPPED, AFFECTED)),
            Map.entry("Tap", ModeRow.of(EventType.TAPPED, AFFECTED)),
            Map.entry("Counter", ModeRow.of(EventType.SPELL_COUNTERED, AFFECTED)),
            Map.entry("CopySpell", ModeRow.of(EventType.SPELL_COPIED,
                    AFFECTED, count(RunParamAmount.key("Amount")))),
            Map.entry("ProduceMana", ModeRow.of(EventType.MANA_PRODUCED, AFFECTED, RunParamSlots.MANA)),
            // ManaPool puts Mana "C" as the replacement's default answer (the
            // mana becomes colourless instead), never the mana being lost.
            Map.entry("LoseMana", ModeRow.of(EventType.MANA_LOST, AFFECTED)),
            Map.entry("Attached", ModeRow.of(EventType.ATTACHED, AFFECTED, RunParamSlots.ATTACHED_TO)),
            Map.entry("Transform", ModeRow.of(EventType.FACE_CHANGE, AFFECTED, RunParamSlots.FACE)),
            Map.entry("TurnFaceUp", ModeRow.of(EventType.FACE_CHANGE, AFFECTED, RunParamSlots.FACE)),
            Map.entry("DeclareBlocker", ModeRow.of(EventType.BLOCKERS_DECLARED,
                    AFFECTED, RunParamSlots.BLOCKED)),
            Map.entry("RollDice", ModeRow.of(EventType.DICE_ROLLED, AFFECTED)),
            Map.entry("GameWin", ModeRow.of(EventType.PLAYER_WON, AFFECTED)),
            Map.entry("GameLoss", ModeRow.of(EventType.PLAYER_LOST, AFFECTED, RunParamSlots.LOSE_REASON)));

    /**
     * Forge's trigger modes this side reads no row for: each names an
     * outcome no declared event type holds a magnitude of (a phase begins, a
     * card becomes the target, a dungeon room is entered, a gift is given)
     * or a batch whose cards its own per-card mode already writes.
     */
    static final Set<String> TRIGGER_MODES_UNREAD = Set.of(
            "Abandoned", "AbilityCast", "AbilityResolves", "AbilityTriggered", "Adapt",
            "Airbend", "Always", "AttackerBlockedOnce", "AttackerUnblocked",
            "AttackerUnblockedOnce", "AttackersDeclaredOneTarget", "BecomeMonarch",
            "BecomeMonstrous", "BecomeRenowned", "BecomesCrewed", "BecomesPlotted",
            "BecomesSaddled", "BecomesTarget", "BecomesTargetOnce", "BlockersDeclared",
            "CaseSolved", "Championed", "ChangesController", "ChaosEnsues", "ClaimPrize",
            "Clashed", "ClassLevelGained", "CollectEvidence", "CommitCrime", "ConjureAll",
            "Connives", "CounterAddedAll", "CounterTypeAddedAll", "CrankContraption", "Crewed",
            "Cycled", "DamageDoneOnceByController", "DayTimeChanges", "Devoured", "Discover", "DungeonCompleted",
            "Earthbend", "ElementalBend", "Enlisted", "Evolved", "ExcessDamage",
            "ExcessDamageAll", "Exerted", "Exiled", "Explores", "FacesDilemma", "Fight",
            "FightOnce", "Firebend", "FlippedCoin", "Forage", "Foretell", "FullyUnlock",
            "GiveGift", "Immediate", "Investigated", "LandPlayed", "LifeLostAll", "LosesGame",
            "ManaAdded", "ManaExpend", "ManifestDread", "Mentored", "MilledAll", "MilledOnce",
            "Mutates", "NewGame", "PayCumulativeUpkeep", "PayEcho", "Phase", "PhaseOutAll",
            "PlanarDice", "PlaneswalkedFrom", "PlaneswalkedTo", "Proliferate", "RingTemptsYou",
            "RoomEntered", "SacrificedOnce", "Saddled", "SearchedLibrary", "SeekAll",
            "SetInMotion", "Specializes", "SpellAbilityCopy", "SpellCastOrCopy", "SpellCopy",
            "Stationed", "TakesInitiative", "TapAll", "Trains", "TurnBegin", "UnlockDoor",
            "UntapAll", "VisitAttraction", "Vote", "Waterbend");

    /**
     * Forge's replacement modes this side reads no row for, on the same
     * terms as {@link #TRIGGER_MODES_UNREAD}: {@code Explore} and {@code
     * Proliferate} carry a count no declared event type holds (an explore's
     * land or counter, a proliferation's chosen objects, are their own
     * events), the rest no magnitude at all.
     */
    static final Set<String> REPLACEMENT_MODES_UNREAD = Set.of(
            "AssembleContraption", "AssignDealDamage", "BeginPhase", "BeginTurn", "Cascade",
            "Connive", "Explore", "Learn", "PlanarDiceResult", "Planeswalk", "Proliferate",
            "RollPlanarDice", "SetInMotion");
}
