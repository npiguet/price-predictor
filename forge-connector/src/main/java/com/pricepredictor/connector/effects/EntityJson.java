package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import com.pricepredictor.connector.json.Json;
import forge.card.CardStateName;
import forge.card.CardTypeView;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.card.CardTraitChanges;
import forge.game.card.CounterType;
import forge.game.card.ICardTraitChanges;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A snapshot's {@code state.entities} entry for one card: its computed
 * characteristics, its place in combat and on the stack, the card facts of
 * FR-024 ({@link CardFacts}), and the lines it carries, printed and granted
 * ({@link CardLines}).
 */
final class EntityJson {

    private EntityJson() {
    }

    /**
     * One entity, as {@code state.entities} carries it.
     *
     * @param board the board it is read on: as it stands, or without one
     *              static ({@link StaticSuppression})
     */
    static String render(Game game, Card card, StaticSuppression board) {
        return render(game, card, board, Beyond.NONE);
    }

    /**
     * Why an entity beyond the four tiers is in the snapshot, and what it is
     * rendered as there (FR-024).
     *
     * @param id   the id it is written under: its own, or a last-known copy's,
     *             which must not collide with the live card's
     * @param zone the zone it is written in, where the object Forge holds has
     *             moved on since (a library card an event later names), or
     *             null for the card's own
     * @param rank its {@code order_rank} where the zone it is written in is
     *             not the one it sits in now, or null for the card's own
     * @param role {@code library_card}, {@code exiled_with},
     *             {@code mutate_component} or {@code last_known}; null for an
     *             entity of the tiers
     */
    record Beyond(String id, String zone, Integer rank, String role) {
        static final Beyond NONE = new Beyond(null, null, null, null);
    }

    /** One entity, written as {@code beyond} says. */
    static String render(Game game, Card card, StaticSuppression board, Beyond beyond) {
        CardTypeView type = board.type(card);
        SpellAbility onStack = CardLines.stackAbilityOf(game, card);
        SpellAbility cast = CardLines.castSpellOf(card, onStack);
        return Json.object()
                .value("id", beyond.id() != null ? beyond.id() : Refs.of(card))
                .value("name", card.getName())
                .value("zone", beyond.zone() != null ? beyond.zone() : zoneName(card))
                .value("owner", CardFacts.ownerOf(card))
                .value("controller", Refs.ofNullable(board.controller(card)))
                .value("face", faceOf(card))
                .value("face_state", CardFacts.faceState(card))
                .value("copy_source", copySourceOf(card))
                .value("token_script_id", CardSpelling.tokenScriptIdOf(card))
                .raw("types", typeNamesJson(type.getCoreTypes()))
                .raw("subtypes", typeNamesJson(type.getSubtypes()))
                .raw("supertypes", typeNamesJson(type.getSupertypes()))
                .raw("colors", Json.stringArray(CardSpelling.colorLetters(board.colors(card))))
                .value("mana_value", card.getCMC())
                .raw("mana_cost", CardFacts.manaCostJson(card))
                .raw("base_pt", basePtJson(card, board))
                .raw("pt_boosts", ptBoostsJson(card, board))
                .value("printed_loyalty", CardFacts.printedLoyalty(card))
                .value("printed_defense", CardFacts.printedDefense(card))
                .value("tapped", card.isTapped())
                .value("sick", board.sick(card))
                .value("damage", card.getDamage())
                .raw("counters", counterJson(card))
                .raw("combat", combatJson(game, card))
                .value("attached_to", Refs.ofNullable(card.getAttachedTo()))
                .value("face_down", card.isFaceDown())
                .raw("real_face", CardFacts.realFaceJson(card))
                .raw("designations", Json.stringArray(CardFacts.designations(card)))
                .value("x_paid", CardFacts.xPaid(cast))
                .raw("chosen", CardFacts.chosenJson(card))
                .raw("costs_paid", CardFacts.costsPaidJson(card, cast))
                .raw("history", CardFacts.historyJson(game, card))
                .value("order_rank", beyond.rank() != null
                        ? beyond.rank() : CardFacts.orderRank(game, card))
                .raw("granted_temporary", grantedTemporaryJson(card, board))
                .raw("lines", CardLines.render(game, card, board))
                .raw("stack_extras", stackExtrasJson(onStack))
                .value("role", beyond.role())
                .render();
    }

    private static String zoneName(Card card) {
        var zone = card.getZone();
        return zone == null ? null : CardSpelling.zoneName(zone.getZoneType());
    }

    /**
     * Power and toughness before boosts: the base the printed value or a
     * setting effect (layer 7b, Ensoul Artifact's 5/5) gives, or null for a
     * card that is not a creature on this board.
     *
     * <p>Forge's own breakdown of the numbers the creature fights with, so
     * base, boosts and counters add up to its net power and toughness, a
     * switch included.
     */
    private static String basePtJson(Card card, StaticSuppression board) {
        if (!board.isCreature(card)) {
            return "null";
        }
        return Json.value(List.of(
                board.power(card).currentValue, board.toughness(card).currentValue));
    }

    /**
     * The boosts on top of the base, kept apart by source: a temporary boost
     * and a counter are different effects with the same total, and the head
     * has to predict which happened. Counters are every counter's
     * contribution (+1/+1, -1/-1, +1/+2, ...), as Forge sums them.
     */
    private static String ptBoostsJson(Card card, StaticSuppression board) {
        if (!board.isCreature(card)) {
            return "null";
        }
        Card.StatBreakdown power = board.power(card);
        Card.StatBreakdown toughness = board.toughness(card);
        return Json.object()
                .value("temporary", List.of(power.tempBoost, toughness.tempBoost))
                .value("counters", List.of(power.bonusFromCounters, toughness.bonusFromCounters))
                .render();
    }

    private static String counterJson(Card card) {
        Json.ObjectWriter counters = Json.object();
        for (com.google.common.collect.Multiset.Entry<CounterType> entry
                : card.getCounters().entrySet()) {
            counters.value(CardSpelling.counterName(entry.getElement()), entry.getCount());
        }
        return counters.render();
    }

    private static String combatJson(Game game, Card card) {
        var combat = game.getCombat();
        if (!CombatShape.isCombatant(combat, card)) {
            return "null";
        }
        List<String> blockedBy = new ArrayList<>();
        if (combat.isAttacking(card)) {
            CardCollectionView blockers = combat.getBlockers(card);
            if (blockers != null) {
                for (Card blocker : blockers) {
                    blockedBy.add(Refs.of(blocker));
                }
            }
        }
        List<String> blocking = new ArrayList<>();
        if (combat.isBlocking(card)) {
            // The other half of blocked_by. Without it a blocker's own entry
            // says only that it is somewhere in a combat, so the flag the head
            // reads for "is this creature blocking" would always be false — and the
            // blocking relationship is what trample and deathtouch act through.
            for (Card attacker : combat.getAttackersBlockedBy(card)) {
                blocking.add(Refs.of(attacker));
            }
        }
        Object defender = combat.isAttacking(card)
                ? combat.getDefenderByAttacker(card) : null;
        return Json.object()
                .value("attacking", defender == null ? null : String.valueOf(defender))
                .raw("blocking", Json.stringArray(blocking))
                .raw("blocked_by", Json.stringArray(blockedBy))
                .value("became_blocked", combat.isBlocked(card))
                .render();
    }

    private static String grantedTemporaryJson(Card card, StaticSuppression board) {
        return Json.object()
                .raw("keywords", Json.stringArray(CardSpelling.keywordNames(board.keywords(card))))
                .raw("abilities", ProvenanceKey.toJsonArray(grantedTemporaryAbilities(card, board)))
                .render();
    }

    /**
     * Abilities granted by a timestamped change: a static's, an attachment's
     * or a resolved effect's grant.
     *
     * <p>"Gains flying and 'T: draw a card' until end of turn" puts the second
     * half here. {@code lines} lists the same grants among every other line,
     * flagged {@code granted}; this list keeps them by themselves. A
     * suppressed static's grants are not on its board.
     */
    private static List<ProvenanceKey> grantedTemporaryAbilities(Card card, StaticSuppression board) {
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (Table.Cell<Long, Long, ICardTraitChanges> cell : card.getChangedCardTraits().cellSet()) {
            // Only the record carries granted lines; a keyword change sharing
            // the table has none to name.
            if (cell.getValue() instanceof CardTraitChanges granted && !board.wrote(cell.getColumnKey())) {
                addKeys(keys, granted.getAbilities());
            }
        }
        return List.copyOf(keys);
    }

    /** Which face is up, as an index; 0 is the printed front. */
    private static int faceOf(Card card) {
        var state = card.getCurrentStateName();
        return state == null || state == CardStateName.Original ? 0 : 1;
    }

    /** What this permanent is copying, or null when it is itself. */
    private static String copySourceOf(Card card) {
        Card copied = card.getCopiedPermanent();
        return copied != null ? Refs.of(copied) : Refs.ofNullable(card.getCloner());
    }

    /**
     * What a stack object announced, or null when the entity is not on the stack.
     *
     * <p>Divided damage and "up to N" counts are choices the caster made that no
     * other field records: a Fireball for 3 split two ways and one aimed at a
     * single creature have the same cost and the same text.
     */
    private static String stackExtrasJson(SpellAbility onStack) {
        if (onStack == null || onStack.getTargets() == null) {
            return "null";
        }
        List<String> targets = new ArrayList<>();
        Json.ObjectWriter amounts = Json.object();
        int chosen = 0;
        for (GameEntity target : onStack.getTargets().getTargetEntities()) {
            chosen++;
            String id = Refs.ofAny(target).orElse(null);
            if (id == null) {
                continue;
            }
            targets.add(id);
            Integer divided = onStack.getDividedValue(target);
            if (divided != null) {
                amounts.value(id, divided);
            }
        }
        Json.ObjectWriter upTo = Json.object();
        if (onStack.getTargetRestrictions() != null) {
            upTo.value("chosen", chosen);
        }
        return Json.object()
                .raw("targets", Json.stringArray(targets))
                .raw("per_target_amounts", amounts.render())
                .raw("up_to_counts", upTo.render())
                .render();
    }

    /**
     * Add every line's key that has one and is not already listed; shared with
     * the global block's emblems.
     */
    static void addKeys(Set<ProvenanceKey> into, Iterable<?> lines) {
        for (Object line : lines) {
            if (line instanceof CardTraitBase trait) {
                ProvenanceKey key = ProvenanceResolver.of(trait);
                if (key != null) {
                    into.add(key);
                }
            }
        }
    }

    /** One part of a type line, as the snapshot's names. */
    private static String typeNamesJson(Iterable<?> types) {
        List<String> names = new ArrayList<>();
        for (Object type : types) {
            names.add(CardSpelling.typeName(type));
        }
        return Json.stringArray(names);
    }
}
