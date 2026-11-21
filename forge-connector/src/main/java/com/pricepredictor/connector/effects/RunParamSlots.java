package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;

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

    /**
     * {@code excess}: the damage beyond lethal damage (120.4a) the patch
     * measured before the damage was dealt and put in a {@code DamageDone}
     * trigger's map ({@code DamageExcess}), which is evaluated once the
     * damage is marked or the loyalty gone; absent on a player's damage and
     * on a checkout without it.
     */
    static final ModeRow.Slot DAMAGE_EXCESS = RunParamSlots::damageExcess;

    /**
     * {@code excess} of a damage a replacement is changing: the damage not
     * dealt yet, measured off the board now as the patch measures a dealt
     * damage, {@code DamageAmount} less {@code Card.getExcessDamageValue}
     * (deathtouch read off the {@code DamageSource}), never below 0; absent
     * on a player's damage and a map without an amount.
     */
    static final ModeRow.Slot WOULD_BE_EXCESS = RunParamSlots::wouldBeExcess;

    /** {@code regenerable}: a destruction's {@code Regeneration}. */
    static final ModeRow.Slot REGENERABLE = RunParamSlots::regenerable;

    /**
     * {@code ChangesZoneAll}: {@code from_zone} and {@code to_zone} where
     * every card of the batch ({@code Cards}) made the same move.
     */
    static final ModeRow.Slot ZONE_TABLE_MOVE = RunParamSlots::zoneTableMove;

    private static void damageExcess(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.EXCESS, intOf(params.get("DamageExcess")));
    }

    private static void wouldBeExcess(EffectEvent event, ModeRow.RunParams params) {
        Integer amount = intOf(params.get("DamageAmount"));
        Card damaged = damageable(params.get("Affected"));
        if (amount == null || damaged == null) {
            return;
        }
        boolean deathtouch = params.get("DamageSource") instanceof Card source
                && source.hasKeyword(Keyword.DEATHTOUCH);
        event.param(EventParam.EXCESS, Math.max(0, amount - damaged.getExcessDamageValue(deathtouch)));
    }

    /**
     * {@code affected} when it is a card damage can be dealt to (120.1a): a
     * creature, a planeswalker or a battle; null for a player or anything else.
     */
    private static Card damageable(Object affected) {
        if (!(affected instanceof Card card)) {
            return null;
        }
        boolean takesDamage = card.isCreature() || card.isPlaneswalker();
        return takesDamage || card.isBattle() ? card : null;
    }

    private static void zones(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.FROM_ZONE, CardSpelling.zoneName(params.get("Origin")));
        event.param(EventParam.TO_ZONE, CardSpelling.zoneName(params.get("Destination")));
    }

    private static void combat(EffectEvent event, ModeRow.RunParams params) {
        Boolean combat = boolOf(params.get("IsCombatDamage"));
        event.param(EventParam.COMBAT, combat != null ? combat : boolOf(params.get("IsCombat")));
    }

    private static void damageMapTotal(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("DamageMap") instanceof Map<?, ?> map) {
            event.param(EventParam.AMOUNT, sumOf(map.values()));
        }
    }

    private static void sourcesDamageMap(EffectEvent event, ModeRow.RunParams params) {
        if (!(params.get("DamageMap") instanceof Map<?, ?> map)) {
            return;
        }
        event.param(EventParam.AMOUNT, sumOf(map.values()));
        if (map.size() == 1) {
            event.param(EventParam.SOURCE, subjectOf(map.keySet().iterator().next()));
        }
    }

    private static void damageTableTotal(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("DamageMap") instanceof Table<?, ?, ?> table) {
            event.param(EventParam.AMOUNT, sumOf(table.values()));
        }
    }

    private static void damageTableSource(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("DamageMap") instanceof Table<?, ?, ?> table
                && table.rowKeySet().size() == 1) {
            event.param(EventParam.SOURCE, subjectOf(table.rowKeySet().iterator().next()));
        }
    }

    private static void zoneTableMove(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("Cards") instanceof Table<?, ?, ?> table && table.size() == 1) {
            // Its one move: origin, destination and the cards that made it.
            table.cellSet().forEach(only -> {
                event.param(EventParam.FROM_ZONE, CardSpelling.zoneName(only.getRowKey()));
                event.param(EventParam.TO_ZONE, CardSpelling.zoneName(only.getColumnKey()));
            });
        }
    }

    // ── amounts ─────────────────────────────────────────────────────────

    /** {@code count}: the cards drawn, milled or discarded, the tokens made, the copies. */
    static ModeRow.Slot count(RunParamAmount amount) {
        return (event, params) -> event.param(EventParam.COUNT, amount.read(params.byName()));
    }

    /**
     * {@code delta}: a life total's change, signed as {@code sign} moves it.
     * {@code LifeAmount} on the triggers, {@code LifeGained} on the {@code
     * GainLife} replacement, {@code Amount} on {@code LifeReduced}'s and
     * {@code PayLife}'s.
     */
    static ModeRow.Slot lifeDelta(ModeRow.Sign sign) {
        return (event, params) -> event.param(EventParam.DELTA, sign.apply(
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

    /**
     * {@code mana_by_color} of a {@code LoseMana} replacement: the mana the
     * affected player's pool would have lost ({@link LostMana#wouldBeLost}),
     * read off the pool as the replacement reports, before {@code
     * ManaPool.clearPool} converts it. The map's {@code Mana} is the
     * replacement's default answer (the colour it becomes, {@code "C"}),
     * never the mana lost.
     */
    static final ModeRow.Slot WOULD_BE_LOST_MANA = RunParamSlots::wouldBeLostMana;

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

    /**
     * {@code from_zone} and {@code without_paying} of a cast spell, read off
     * the cast {@code SpellAbility}: the zone its card was cast from ({@code
     * Card.getCastFrom} on {@code CardLKI}, else the spell's card), and
     * {@code true} only for a cast without paying its mana cost (118.9).
     * Nothing for an ability's activation, which casts no card.
     */
    static final ModeRow.Slot CAST = RunParamSlots::cast;

    /**
     * What {@code SpellAbility.copyWithNoManaCost} adds to the description of
     * the spell ability it makes, the one way Forge casts a spell without
     * paying its mana cost (a {@code Play} effect's {@code WithoutManaCost$},
     * cascade, discover, suspend, plot, a {@code MayPlayWithoutManaCost}
     * static). The {@code WithoutManaCost} parameter it also puts there does
     * not say it: a later copy of the spell ability (an optional cost added)
     * drops it, and four {@code Play} spells print it on their own line.
     */
    static final String WITHOUT_MANA_COST = " (without paying its mana cost)";

    /** {@code reason}: why a player lost ({@code LoseReason}). */
    static final ModeRow.Slot LOSE_REASON = RunParamSlots::loseReason;

    /** {@code out}: whether the permanent phased out (true) or in (false). */
    static ModeRow.Slot phased(boolean out) {
        return (event, params) -> event.param(EventParam.OUT, out);
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
            event.param(EventParam.MANA_BY_COLOR, mana.isEmpty() ? null : mana);
        }
    }

    private static void wouldBeLostMana(EffectEvent event, ModeRow.RunParams params) {
        if (params.get("Affected") instanceof Player player) {
            event.param(EventParam.MANA_BY_COLOR, LostMana.wouldBeLost(player));
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
        event.param(EventParam.TO_STATE, card instanceof Card turned ? CardFacts.faceState(turned) : null);
    }

    private static void oneDie(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.SIDES, intOf(params.get("Sides")));
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
        event.param(EventParam.RESULTS, List.of(die));
    }

    private static void dice(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.SIDES, intOf(params.get("Sides")));
        if (params.get("Result") instanceof Iterable<?> results) {
            List<Map<String, Integer>> dice = new ArrayList<>();
            for (Object result : results) {
                Integer value = intOf(result);
                if (value != null) {
                    dice.add(Map.of("final", value));
                }
            }
            event.param(EventParam.RESULTS, dice);
        }
    }

    private static void damageAmount(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.AMOUNT, intOf(params.get("DamageAmount")));
    }

    private static void damageSource(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.SOURCE, subjectOf(params.get("DamageSource")));
    }

    private static void regenerable(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.REGENERABLE, boolOf(params.get("Regeneration")));
    }

    private static void attachedTo(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.ATTACHED_TO, firstRef(params.byName(), "AttachTarget"));
    }

    private static void detachedFrom(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.DETACHED_FROM, firstRef(params.byName(), "Object"));
    }

    private static void defender(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.DEFENDER, firstRef(params.byName(),
                "Attacked", "Defender", "AttackedTarget", "DefendingPlayer", "Defenders"));
    }

    private static void blocked(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.BLOCKED, firstRef(params.byName(), "Attackers", "Attacker"));
    }

    private static void blockers(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.BLOCKERS, refList(params.byName(), "Blockers", "Blocker"));
    }

    private static void cast(EffectEvent event, ModeRow.RunParams params) {
        if (!(params.get("SpellAbility") instanceof SpellAbility spell) || !spell.isSpell()) {
            return;
        }
        Card card = params.get("CardLKI") instanceof Card lastKnown ? lastKnown : spell.getHostCard();
        Zone from = card == null ? null : card.getCastFrom();
        event.param(EventParam.FROM_ZONE, from == null ? null : CardSpelling.zoneName(from.getZoneType()));
        String description = spell.getDescription();
        if (description != null && description.contains(WITHOUT_MANA_COST)) {
            event.param(EventParam.WITHOUT_PAYING, true);
        }
    }

    private static void loseReason(EffectEvent event, ModeRow.RunParams params) {
        event.param(EventParam.REASON, textOf(params.get("LoseReason")));
    }
}
