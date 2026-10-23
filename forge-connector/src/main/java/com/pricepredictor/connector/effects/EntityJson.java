package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardStateName;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.card.CardTraitChanges;
import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;
import forge.game.card.ICardTraitChanges;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A snapshot's {@code state.entities} entry for one card: its computed
 * characteristics, its place in combat and on the stack, and the lines it
 * carries, printed and granted.
 */
final class EntityJson {

    private EntityJson() {
    }

    /**
     * One entity, as {@code state.entities} carries it.
     *
     * @param withoutStatic the id of a static whose changes are removed from
     *                      the entity, or null to show it as it stands
     */
    static String render(Game game, Card card, Long withoutStatic) {
        Suppressed without = Suppressed.of(card, withoutStatic);
        CardTypeView type = without == null ? card.getType() : without.type();
        return Json.object()
                .value("id", Refs.of(card))
                .value("name", card.getName())
                .value("zone", zoneName(card))
                .value("controller", Refs.ofNullable(card.getController()))
                .value("face", faceOf(card))
                .value("copy_source", copySourceOf(card))
                .value("token_script_id", CardSpelling.tokenScriptIdOf(card))
                .raw("types", typeNamesJson(type.getCoreTypes()))
                .raw("subtypes", typeNamesJson(type.getSubtypes()))
                .raw("supertypes", typeNamesJson(type.getSupertypes()))
                .raw("colors", Json.stringArray(CardSpelling.colorLetters(
                        without == null ? card.getColor() : without.colors())))
                .value("mana_value", card.getCMC())
                .raw("pt", ptJson(card, without))
                .value("tapped", card.isTapped())
                .value("sick", card.isSick())
                .value("damage", card.getDamage())
                .raw("counters", counterJson(card))
                .raw("combat", combatJson(game, card))
                .value("attached_to", Refs.ofNullable(card.getAttachedTo()))
                .value("face_down", card.isFaceDown())
                .raw("granted_attached", ProvenanceKey.toJsonArray(grantedAttached(card)))
                .raw("granted_temporary", grantedTemporaryJson(card, without))
                .raw("printed", ProvenanceKey.toJsonArray(printedKeys(card)))
                .raw("stack_extras", stackExtrasJson(game, card))
                .render();
    }

    private static String zoneName(Card card) {
        var zone = card.getZone();
        return zone == null
                ? null
                : zone.getZoneType().name().toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * Computed power and toughness, decomposed into the three sources.
     *
     * <p>Base plus boosts plus counters, kept apart because an ability that
     * adds a boost and one that adds a counter are different effects with the
     * same total, and the head has to predict which happened.
     */
    private static String ptJson(Card card, Suppressed without) {
        if (!card.isCreature()) {
            return "null";
        }
        int counterPower = card.getCounters(CounterEnumType.P1P1)
                - card.getCounters(CounterEnumType.M1M1);
        int basePower = card.getBasePower();
        int baseToughness = card.getBaseToughness();
        int boostPower = card.getNetPower() - basePower - counterPower;
        int boostToughness = card.getNetToughness() - baseToughness - counterPower;
        if (without != null) {
            boostPower -= without.power();
            boostToughness -= without.toughness();
        }
        return Json.object()
                .value("base", List.of(basePower, baseToughness))
                .value("boosts", List.of(boostPower, boostToughness))
                .value("counters", List.of(counterPower, counterPower))
                .render();
    }

    private static String counterJson(Card card) {
        Json.ObjectWriter counters = Json.object();
        for (com.google.common.collect.Multiset.Entry<CounterType> entry
                : card.getCounters().entrySet()) {
            counters.value(
                    entry.getElement().getName().toUpperCase(java.util.Locale.ROOT),
                    entry.getCount());
        }
        return counters.render();
    }

    private static String combatJson(Game game, Card card) {
        var combat = game.getCombat();
        if (combat == null || !combat.isAttacking(card) && !combat.isBlocking(card)) {
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

    private static String grantedTemporaryJson(Card card, Suppressed without) {
        return Json.object()
                .raw("keywords", Json.stringArray(CardSpelling.keywordNames(
                        without == null ? card.getKeywords() : without.keywords())))
                .raw("abilities", ProvenanceKey.toJsonArray(grantedTemporaryAbilities(card)))
                .render();
    }

    /**
     * Lines granted by a timestamped change rather than printed or attached.
     *
     * <p>"Gains flying and 'T: draw a card' until end of turn" puts the second
     * half here. Attachment grants ride {@code granted_attached} instead, and
     * the two stay apart because the model reads them differently: an entity's
     * ability tokens are its printed and attached lines, while a temporary
     * grant is part of what happened to it.
     */
    private static List<ProvenanceKey> grantedTemporaryAbilities(Card card) {
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (ICardTraitChanges change : card.getChangedCardTraits().values()) {
            // Only the record carries granted lines; a keyword change sharing
            // the table has none to name.
            if (change instanceof CardTraitChanges granted) {
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
    private static String stackExtrasJson(Game game, Card card) {
        SpellAbility onStack = null;
        for (var instance : game.getStack()) {
            SpellAbility candidate = instance.getSpellAbility();
            if (candidate != null && candidate.getHostCard() == card) {
                onStack = candidate;
                break;
            }
        }
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

    /** Printed lines, as provenance keys into the card's own script. */
    private static List<ProvenanceKey> printedKeys(Card card) {
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        addKeys(keys, card.getSpellAbilities());
        return List.copyOf(keys);
    }

    /**
     * Attachment-granted lines only.
     *
     * <p>Temporary grants ride the overlay instead, and the two channels stay
     * separate because the model reads them differently: an entity's ability
     * tokens are its printed and attachment-granted lines, while a grant until
     * end of turn is part of what happened to it rather than what it is.
     */
    private static List<ProvenanceKey> grantedAttached(Card card) {
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (Card attachment : card.getAttachedCards()) {
            addKeys(keys, attachment.getSpellAbilities());
        }
        return List.copyOf(keys);
    }

    /**
     * Add every line's key that has one and is not already listed; shared with
     * the global block's emblems.
     */
    static void addKeys(Set<ProvenanceKey> into, Iterable<?> lines) {
        for (Object line : lines) {
            if (line instanceof SpellAbility sa) {
                ProvenanceKey key = ProvenanceResolver.of(sa);
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


    /**
     * One entity's characteristics with an acting static's changes removed.
     *
     * <p>A {@code continuous} record's label is what its static does to the
     * board, so a snapshot that still contains the static's own work hands the
     * model the answer with the question. Every channel that static wrote has
     * to come back out.
     *
     * <p>Recombined by the engine rather than subtracted here, because
     * subtraction gets two cases wrong: another static granting the same
     * keyword or type would go with it, and a static that overwrites a type
     * line or strips a whole class of type cannot be inverted from what it
     * contributed. Power and toughness are the exception — boosts are summed
     * per static, so dropping one term is already exact.
     */
    private record Suppressed(
            CardTypeView type, ColorSet colors,
            Iterable<KeywordInterface> keywords, int power, int toughness) {

        /**
         * What the card would be without this static, or null for no change.
         *
         * <p>Null when the patch is absent, which is also when a continuous
         * record is never written — {@link ContinuousChannel#collect}
         * checks for these three before it collects, so a leaking snapshot is
         * not reachable by falling back to here.
         */
        static Suppressed of(Card card, Long staticId) {
            if (staticId == null) {
                return null;
            }
            if (!(PatchHooks.TYPE_WITHOUT.read(card, staticId) instanceof CardTypeView type)
                    || !(PatchHooks.COLOR_WITHOUT.read(card, staticId) instanceof ColorSet colors)
                    || !(PatchHooks.KEYWORDS_WITHOUT.read(card, staticId)
                            instanceof Iterable<?> keywords)) {
                return null;
            }
            int power = 0;
            int toughness = 0;
            for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell
                    : card.getPTBoostTable().cellSet()) {
                if (staticId.equals(cell.getColumnKey())) {
                    power += cell.getValue().getLeft();
                    toughness += cell.getValue().getRight();
                }
            }
            @SuppressWarnings("unchecked")
            Iterable<KeywordInterface> typed = (Iterable<KeywordInterface>) keywords;
            return new Suppressed(type, colors, typed, power, toughness);
        }
    }
}
