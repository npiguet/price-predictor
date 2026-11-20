package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CounterType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RunParamValues.boolOf;
import static com.pricepredictor.connector.effects.RunParamValues.firstInt;
import static com.pricepredictor.connector.effects.RunParamValues.firstRef;
import static com.pricepredictor.connector.effects.RunParamValues.intOf;
import static com.pricepredictor.connector.effects.RunParamValues.refList;
import static com.pricepredictor.connector.effects.RunParamValues.subjectOf;
import static com.pricepredictor.connector.effects.RunParamValues.sumOf;
import static com.pricepredictor.connector.effects.RunParamValues.textOf;

/**
 * The slots a mode's row fills: each a parameter the event vocabulary
 * declares, read off the keys Forge puts it under. A row lists the ones its
 * mode reads ({@link RunParamRows}); a key written outside its type's own
 * set is one no reader would look for.
 */
final class RunParamSlots {

    private RunParamSlots() {
    }

    // ── zones and damage ────────────────────────────────────────────────

    /** {@code from_zone} and {@code to_zone}: a move's {@code Origin} and {@code Destination}. */
    static final ModeRow.Slot ZONES = RunParamSlots::zones;

    /** {@code amount}: the damage dealt or prevented, as {@code DamageAmount}. */
    static final ModeRow.Slot DAMAGE_AMOUNT = RunParamSlots::damageAmount;

    /** {@code combat}: {@code IsCombatDamage} on the triggers, {@code IsCombat} on the replacement. */
    static final ModeRow.Slot COMBAT = RunParamSlots::combat;

    /** {@code source}: the {@code DamageSource}. */
    static final ModeRow.Slot DAMAGE_SOURCE = RunParamSlots::damageSource;

    /** {@code amount}: a {@code DamageMap}'s (damaged to amount) amounts summed. */
    static final ModeRow.Slot DAMAGE_MAP_TOTAL = RunParamSlots::damageMapTotal;

    /**
     * {@code DamageDoneOnce}: one target's damage from every source at once
     * ({@code DamageMap}, source to amount), its {@code amount} the sum and
     * its {@code source} named when one source dealt it.
     */
    static final ModeRow.Slot SOURCES_DAMAGE_MAP = RunParamSlots::sourcesDamageMap;

    /** {@code amount}: a {@code CardDamageTable}'s ({@code DamageMap}) amounts summed. */
    static final ModeRow.Slot DAMAGE_TABLE_TOTAL = RunParamSlots::damageTableTotal;

    /** {@code source}: a {@code CardDamageTable}'s one source, when one source dealt it all. */
    static final ModeRow.Slot DAMAGE_TABLE_SOURCE = RunParamSlots::damageTableSource;

    /** {@code regenerable}: a destruction's {@code Regeneration}. */
    static final ModeRow.Slot REGENERABLE = RunParamSlots::regenerable;

    /**
     * {@code ChangesZoneAll}: {@code from_zone} and {@code to_zone} where
     * every card of the batch ({@code Cards}) made the same move.
     */
    static final ModeRow.Slot ZONE_TABLE_MOVE = RunParamSlots::zoneTableMove;

    private static void zones(EffectEvent event, ModeRow.RunParams params) {
        event.param("from_zone", CardSpelling.zoneName(params.get("Origin")));
        event.param("to_zone", CardSpelling.zoneName(params.get("Destination")));
    }

    private static void combat(EffectEvent event, ModeRow.RunParams params) {
        Boolean combat = boolOf(params.get("IsCombatDamage"));
        event.param("combat", combat != null ? combat : boolOf(params.get("IsCombat")));
    }

    private static void damageMapTotal(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("DamageMap") instanceof Map<?, ?> map) {
            event.param("amount", sumOf(map.values()));
        }
    }

    private static void sourcesDamageMap(EffectEvent event, ModeRow.RunParams params) {
        if (!(params.get("DamageMap") instanceof Map<?, ?> map)) {
            return;
        }
        event.param("amount", sumOf(map.values()));
        if (map.size() == 1) {
            event.param("source", subjectOf(map.keySet().iterator().next()));
        }
    }

    private static void damageTableTotal(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("DamageMap") instanceof Table<?, ?, ?> table) {
            event.param("amount", sumOf(table.values()));
        }
    }

    private static void damageTableSource(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("DamageMap") instanceof Table<?, ?, ?> table
                && table.rowKeySet().size() == 1) {
            event.param("source", subjectOf(table.rowKeySet().iterator().next()));
        }
    }

    private static void zoneTableMove(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("Cards") instanceof Table<?, ?, ?> table && table.size() == 1) {
            // Its one move: origin, destination and the cards that made it.
            table.cellSet().forEach(only -> {
                event.param("from_zone", CardSpelling.zoneName(only.getRowKey()));
                event.param("to_zone", CardSpelling.zoneName(only.getColumnKey()));
            });
        }
    }

    // ── amounts ─────────────────────────────────────────────────────────

    /** {@code count}: the cards drawn, milled or discarded, the tokens made, the copies. */
    static ModeRow.Slot count(RunParamAmount amount) {
        return (event, params) -> event.param("count", amount.read(params.byName()));
    }

    /**
     * {@code delta}: a life total's change, signed as {@code sign} moves it.
     * {@code LifeAmount} on the triggers, {@code LifeGained} on the {@code
     * GainLife} replacement, {@code Amount} on {@code LifeReduced}'s and
     * {@code PayLife}'s.
     */
    static ModeRow.Slot lifeDelta(ModeRow.Sign sign) {
        return (event, params) -> event.param("delta", sign.apply(
                firstInt(params.byName(), "LifeAmount", "LifeGained", "Amount")));
    }

    /** {@code counter_type} and {@code delta}: see {@link CounterSlot}. */
    static ModeRow.Slot counters(RunParamAmount amount, ModeRow.Sign sign) {
        return new CounterSlot(amount, sign);
    }

    /**
     * What a {@code RemoveCounter} replacement's removal comes to, as a
     * signed change: the count it would leave ({@code Result}, never below 0)
     * less the affected object's count of that type now, the removal not
     * having happened yet; null where either is unreadable.
     */
    static Integer removal(Map<String, Object> byName) {
        Integer result = intOf(byName.get("Result"));
        if (result == null || !(byName.get("Affected") instanceof GameEntity affected)) {
            return null;
        }
        if (!(byName.get("CounterType") instanceof CounterType counterType)) {
            return null;
        }
        return Math.max(result, 0) - affected.getCounters(counterType);
    }

    /**
     * The tokens a {@code CreateToken} replacement's table ({@code Token}, a
     * {@code TokenCreateTable} of controller, prototype and number) makes for
     * the player it is replacing for ({@code Affected}). Forge runs the
     * replacement once per player with the same table, so its other rows are
     * other players' records.
     */
    static Integer affectedTokens(Map<String, Object> byName) {
        if (!(byName.get("Token") instanceof Table<?, ?, ?> table)) {
            return null;
        }
        Map<?, ?> row = table.rowMap().get(byName.get("Affected"));
        return row == null ? 0 : sumOf(row.values());
    }

    // ── mana, faces, attachments, combat, the game's end ────────────────

    /**
     * {@code mana_by_color}: the mana a mana event moved, Forge's own short
     * text ({@code "G"}, {@code "G G"}, {@code "1"}) under {@code Produced}
     * or {@code Mana}, read as {@link ProducedMana#byColor} reads
     * it for the activation channel, which keeps a tapped land and a replaced
     * mana production comparable. A value that is not text yields nothing:
     * {@code {}} would claim no mana moved.
     */
    static final ModeRow.Slot MANA = RunParamSlots::mana;

    /** {@code to_state}: the face the card shows now, in an entity's {@code face_state} spelling. */
    static final ModeRow.Slot FACE = RunParamSlots::face;

    /** {@code attached_to}: what an Aura or Equipment hangs on ({@code AttachTarget}). */
    static final ModeRow.Slot ATTACHED_TO = RunParamSlots::attachedTo;

    /** {@code detached_from}: the permanent or player it left ({@code Object}). */
    static final ModeRow.Slot DETACHED_FROM = RunParamSlots::detachedFrom;

    /**
     * {@code defender}: the entity attacked. {@code Attacked} first: {@code
     * CombatUtil.checkDeclaredAttacker} spells it that way and no other.
     */
    static final ModeRow.Slot DEFENDER = RunParamSlots::defender;

    /** {@code blocked}: the attacker a blocker blocks. */
    static final ModeRow.Slot BLOCKED = RunParamSlots::blocked;

    /** {@code blockers}: every creature blocking the attacker. */
    static final ModeRow.Slot BLOCKERS = RunParamSlots::blockers;

    /** {@code reason}: why a player lost ({@code LoseReason}). */
    static final ModeRow.Slot LOSE_REASON = RunParamSlots::loseReason;

    /** {@code out}: whether the permanent phased out (true) or in (false). */
    static ModeRow.Slot phased(boolean out) {
        return (event, params) -> event.param("out", out);
    }

    /** {@code RolledDie}: one die, its {@code sides} and its natural and final value. */
    static final ModeRow.Slot ONE_DIE = RunParamSlots::oneDie;

    /** {@code RolledDieOnce}: every die of one roll, its {@code sides} and their final values. */
    static final ModeRow.Slot DICE = RunParamSlots::dice;

    private static void mana(EffectEvent event, ModeRow.RunParams params) {
        Object produced = params.get("Produced");
        if (!(produced instanceof String)) {
            produced = params.get("Mana");
        }
        if (produced instanceof String text) {
            Map<String, Integer> mana = ProducedMana.byColor(text.trim());
            event.param("mana_by_color", mana.isEmpty() ? null : mana);
        }
    }

    /**
     * Read off the card rather than the mode: Forge runs the {@code
     * Transform} and {@code TurnFaceUp} replacements and the {@code
     * Transformed} and {@code TurnFaceUp} triggers once the card has turned
     * (a transform lands on either face), and no map carries a usable state
     * name: {@code AbilityKey.CardState} is a {@code CardState} object only
     * door unlocking puts there ({@code Card.unlockRoom}).
     */
    private static void face(EffectEvent event, ModeRow.RunParams params) {
        Object card = params.get("Card");
        if (!(card instanceof Card)) {
            card = params.get("Affected");
        }
        event.param("to_state", card instanceof Card turned ? CardFacts.faceState(turned) : null);
    }

    private static void oneDie(EffectEvent event, ModeRow.RunParams params) {
        event.param("sides", intOf(params.get("Sides")));
        Integer result = intOf(params.get("Result"));
        if (result == null) {
            return;
        }
        Map<String, Integer> die = new LinkedHashMap<>();
        Integer natural = intOf(params.get("NaturalResult"));
        if (natural != null) {
            die.put("natural", natural);
        }
        die.put("final", result);
        event.param("results", List.of(die));
    }

    private static void dice(EffectEvent event, ModeRow.RunParams params) {
        event.param("sides", intOf(params.get("Sides")));
        if (params.get("Result") instanceof Iterable<?> results) {
            List<Map<String, Integer>> dice = new ArrayList<>();
            for (Object result : results) {
                Integer value = intOf(result);
                if (value != null) {
                    dice.add(Map.of("final", value));
                }
            }
            event.param("results", dice);
        }
    }

    private static void damageAmount(EffectEvent event, ModeRow.RunParams params) {
        event.param("amount", intOf(params.get("DamageAmount")));
    }

    private static void damageSource(EffectEvent event, ModeRow.RunParams params) {
        event.param("source", subjectOf(params.get("DamageSource")));
    }

    private static void regenerable(EffectEvent event, ModeRow.RunParams params) {
        event.param("regenerable", boolOf(params.get("Regeneration")));
    }

    private static void attachedTo(EffectEvent event, ModeRow.RunParams params) {
        event.param("attached_to", firstRef(params.byName(), "AttachTarget"));
    }

    private static void detachedFrom(EffectEvent event, ModeRow.RunParams params) {
        event.param("detached_from", firstRef(params.byName(), "Object"));
    }

    private static void defender(EffectEvent event, ModeRow.RunParams params) {
        event.param("defender", firstRef(params.byName(),
                "Attacked", "Defender", "AttackedTarget", "DefendingPlayer", "Defenders"));
    }

    private static void blocked(EffectEvent event, ModeRow.RunParams params) {
        event.param("blocked", firstRef(params.byName(), "Attackers", "Attacker"));
    }

    private static void blockers(EffectEvent event, ModeRow.RunParams params) {
        event.param("blockers", refList(params.byName(), "Blockers", "Blocker"));
    }

    private static void loseReason(EffectEvent event, ModeRow.RunParams params) {
        event.param("reason", textOf(params.get("LoseReason")));
    }
}
