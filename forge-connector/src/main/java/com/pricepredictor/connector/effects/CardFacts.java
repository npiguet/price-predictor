package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.card.CardStateName;
import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.mana.Mana;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * The card facts of FR-024 a card slot carries beside its characteristics:
 * which face is up, its designations, what it was cast with and for, what it
 * remembers choosing, what it did this turn, and where it sits in its zone.
 *
 * <p>Each is a fact some rule reads and nothing else in the slot says: a
 * monstrous creature's ability does nothing a second time, a Cavern of Souls
 * creature cannot be countered, a creature that entered this turn cannot
 * attack, and the newer of two legends is not the one the legend rule asks
 * about.
 */
final class CardFacts {

    private CardFacts() {
    }

    /**
     * Which face is up: {@code front}, {@code back}, {@code flipped},
     * {@code melded} or {@code face_down}. A split card, a prepared spell and
     * a specialized card are on their front.
     */
    static String faceState(Card card) {
        if (card.isFaceDown()) {
            return "face_down";
        }
        CardStateName state = card.getCurrentStateName();
        if (state == null) {
            return "front";
        }
        return switch (state) {
            case FaceDown -> "face_down";
            case Backside, Secondary -> "back";
            case Flipped -> "flipped";
            case Meld -> "melded";
            default -> "front";
        };
    }

    /**
     * The designations on the card, sorted: Forge's closed list of marks a
     * permanent carries apart from its characteristics.
     */
    static List<String> designations(Card card) {
        List<String> held = new ArrayList<>();
        for (Map.Entry<String, Predicate<Card>> designation : DESIGNATIONS.entrySet()) {
            if (designation.getValue().test(card)) {
                held.add(designation.getKey());
            }
        }
        return held;
    }

    /** In wire-name order, so the list is written sorted. */
    private static final Map<String, Predicate<Card>> DESIGNATIONS = designationTable();

    private static Map<String, Predicate<Card>> designationTable() {
        Map<String, Predicate<Card>> table = new LinkedHashMap<>();
        table.put("bestowed", Card::isBestowed);
        table.put("cloaked", Card::isCloaked);
        table.put("commander", Card::isCommander);
        table.put("foretold", Card::isForetold);
        table.put("goaded", Card::isGoaded);
        table.put("harnessed", Card::isHarnessed);
        table.put("manifested", Card::isManifested);
        table.put("monstrous", Card::isMonstrous);
        table.put("phased_out", Card::isPhasedOut);
        table.put("plotted", Card::isPlotted);
        table.put("renowned", Card::isRenowned);
        table.put("ring_bearer", Card::isRingBearer);
        table.put("saddled", Card::isSaddled);
        table.put("solved", Card::isSolved);
        table.put("specialized", Card::isSpecialized);
        table.put("suspected", Card::isSuspected);
        table.put("unearthed", Card::isUnearthed);
        table.put("warped", Card::isWarped);
        return table;
    }

    /**
     * The X the spell this card is or was cast as paid, kept on a permanent
     * after it resolves; null when it was not cast, or not with X.
     */
    static Integer xPaid(SpellAbility cast) {
        return cast == null ? null : cast.getXManaCostPaid();
    }

    /**
     * What the card remembers choosing, each member only when set: colours,
     * creature or card types, a number, a player, cards, card names, a
     * direction, a mode, an even-or-odd choice, a sector, and stored die
     * results. Cavern of Souls's type and Gods Willing's colour are here; a
     * modal spell's modes are its line's {@code chosen}.
     */
    static String chosenJson(Card card) {
        Json.ObjectWriter chosen = Json.object();
        List<String> colors = new ArrayList<>();
        card.getChosenColors().forEach(colors::add);
        if (!colors.isEmpty()) {
            chosen.raw("colors", Json.stringArray(colors));
        }
        if (card.hasChosenType()) {
            chosen.value("type", card.getChosenType());
        }
        if (card.hasChosenType2()) {
            chosen.value("type2", card.getChosenType2());
        }
        if (card.hasChosenNumber()) {
            chosen.value("number", card.getChosenNumber());
        }
        if (card.getChosenPlayer() != null) {
            chosen.value("player", Refs.of(card.getChosenPlayer()));
        }
        if (card.hasChosenCard()) {
            List<String> cards = new ArrayList<>();
            for (Card each : card.getChosenCards()) {
                cards.add(Refs.of(each));
            }
            chosen.raw("cards", Json.stringArray(cards));
        }
        if (card.hasNamedCard()) {
            chosen.raw("names", Json.stringArray(card.getNamedCards()));
        }
        if (card.getChosenDirection() != null) {
            chosen.value("direction", card.getChosenDirection().name().toLowerCase(java.util.Locale.ROOT));
        }
        if (card.getChosenMode() != null && !card.getChosenMode().isEmpty()) {
            chosen.value("mode", card.getChosenMode());
        }
        if (card.hasChosenEvenOdd()) {
            chosen.value("even_odd", card.getChosenEvenOdd().name().toLowerCase(java.util.Locale.ROOT));
        }
        if (card.getChosenSector() != null && !card.getChosenSector().isEmpty()) {
            chosen.value("sector", card.getChosenSector());
        }
        if (card.getStoredRolls() != null && !card.getStoredRolls().isEmpty()) {
            chosen.value("stored_rolls", card.getStoredRolls());
        }
        return chosen.render();
    }

    /**
     * What paid for the spell this card is or was cast as: the mana spent per
     * symbol, the zone it was cast from, and whether any of that mana carried
     * a rider (Cavern of Souls's "can't be countered", a counter or a keyword
     * it adds, a trigger on spending it). Null when the card was not cast.
     *
     * <p>Optional and alternative costs are on the lines they belong to
     * ({@code cast_with}), not here.
     */
    static String costsPaidJson(Card card, SpellAbility cast) {
        if (cast == null) {
            return "null";
        }
        List<Byte> spent = new ArrayList<>();
        boolean rider = false;
        for (Mana mana : cast.getPayingMana()) {
            if (mana == null) {
                continue;
            }
            spent.add(mana.getColor());
            rider |= mana.addsNoCounterMagic(cast) || mana.addsCounters(cast)
                    || mana.addsKeywords(cast) || mana.triggersWhenSpent();
        }
        Zone from = card.getCastFrom();
        return Json.object()
                .raw("mana", ManaCounts.tally(spent).toJson())
                .value("cast_from", from == null ? null : CardSpelling.zoneName(from.getZoneType()))
                .value("mana_rider", rider)
                .render();
    }

    /**
     * What the card did this turn: entered its zone, attacked, blocked, and
     * activated a loyalty ability. The sources that dealt it damage are
     * links, not bits.
     */
    static String historyJson(Game game, Card card) {
        int turn = game.getPhaseHandler().getTurn();
        return Json.object()
                .value("entered_this_turn", card.getTurnInZone() == turn)
                .value("attacked_this_turn", card.getDamageHistory().getCreatureAttacksThisTurn() > 0)
                .value("blocked_this_turn", !card.getBlockedThisTurn().isEmpty())
                .value("loyalty_activated", card.getPlaneswalkerAbilityActivated() > 0)
                .render();
    }

    /**
     * The card's rank in its zone, 0 first: on the battlefield by timestamp
     * (the oldest permanent is 0, which is layer and world-rule order); on the
     * stack from the top (0 resolves next); in a library from the top; in a
     * graveyard, hand or exile in the order Forge lists the zone, the card put
     * there earliest first. Null outside a zone, and in the command zone.
     */
    static Integer orderRank(Game game, Card card) {
        Zone zone = card.getZone();
        if (zone == null) {
            return null;
        }
        ZoneType type = zone.getZoneType();
        switch (type) {
            case Battlefield -> {
                Comparator<Card> byTimestamp = Comparator
                        .comparingLong(Card::getLayerTimestamp).thenComparingInt(Card::getId);
                int rank = 0;
                for (Card other : game.getCardsIn(ZoneType.Battlefield)) {
                    if (other != card && byTimestamp.compare(other, card) < 0) {
                        rank++;
                    }
                }
                return rank;
            }
            case Stack -> {
                int rank = 0;
                for (var instance : game.getStack()) {
                    SpellAbility ability = instance.getSpellAbility();
                    if (ability != null && ability.getHostCard() == card && ability.isSpell()) {
                        return rank;
                    }
                    rank++;
                }
                return null;
            }
            case Command -> {
                return null;
            }
            default -> {
                int index = zone.getCards().indexOf(card);
                return index < 0 ? null : index;
            }
        }
    }

    /**
     * A face-down card's real face, as the front it would turn up as: its
     * name, types, colours, mana value and printed power and toughness.
     * Null for a card that is not face down. Its lines are the slot's lines
     * flagged inactive.
     */
    static String realFaceJson(Card card) {
        if (!card.isFaceDown() || !card.hasState(CardStateName.Original)) {
            return "null";
        }
        CardState real = card.getState(CardStateName.Original);
        List<String> types = new ArrayList<>();
        real.getType().getCoreTypes().forEach(type -> types.add(CardSpelling.typeName(type)));
        List<String> subtypes = new ArrayList<>();
        real.getType().getSubtypes().forEach(type -> subtypes.add(CardSpelling.typeName(type)));
        return Json.object()
                .value("name", real.getName())
                .raw("types", Json.stringArray(types))
                .raw("subtypes", Json.stringArray(subtypes))
                .raw("colors", Json.stringArray(CardSpelling.colorLetters(real.getColor())))
                .value("mana_value", real.getManaCost().getCMC())
                .value("base_pt", real.getType().isCreature()
                        ? List.of(real.getBasePower(), real.getBaseToughness()) : null)
                .render();
    }

    /** Printed loyalty of the face up; null when it prints none or prints X. */
    static Integer printedLoyalty(Card card) {
        return printedNumber(card.getCurrentState().getBaseLoyalty());
    }

    /** Printed defense of the face up; null when it prints none or prints X. */
    static Integer printedDefense(Card card) {
        return printedNumber(card.getCurrentState().getBaseDefense());
    }

    private static Integer printedNumber(String printed) {
        if (printed == null || printed.isEmpty()) {
            return null;
        }
        try {
            return Integer.parseInt(printed.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * The mana cost's symbols, counted per kind and only where non-zero:
     * {@code generic}, and each coloured, colourless, X, hybrid, Phyrexian
     * and snow symbol by its short spelling ({@code W}, {@code C}, {@code X},
     * {@code W/U}, {@code 2/W}, {@code G/P}, {@code S}). {@code {}} for a cost
     * of {0}; null for a card with no mana cost at all, such as a land.
     */
    static String manaCostJson(Card card) {
        ManaCost cost = card.getManaCost();
        if (cost == null || cost.isNoCost()) {
            return "null";
        }
        Map<String, Integer> symbols = new LinkedHashMap<>();
        if (cost.getGenericCost() > 0) {
            symbols.put("generic", cost.getGenericCost());
        }
        for (ManaCostShard shard : cost) {
            symbols.merge(shard.toShortString(), 1, Integer::sum);
        }
        return Json.value(symbols);
    }

    /** The owner of an entity, which control changes leave alone. */
    static String ownerOf(Card card) {
        Player owner = card.getOwner();
        return owner == null ? null : Refs.of(owner);
    }
}
