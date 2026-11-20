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
     * Which face is up. A split card, a prepared spell and a specialized card
     * are on their front.
     */
    static FaceState faceState(Card card) {
        if (card.isFaceDown()) {
            return FaceState.FACE_DOWN;
        }
        CardStateName state = card.getCurrentStateName();
        if (state == null) {
            return FaceState.FRONT;
        }
        return switch (state) {
            case FaceDown -> FaceState.FACE_DOWN;
            case Backside, Secondary -> FaceState.BACK;
            case Flipped -> FaceState.FLIPPED;
            case Meld -> FaceState.MELDED;
            default -> FaceState.FRONT;
        };
    }

    /**
     * The designations on the card, sorted: Forge's closed list of marks a
     * permanent carries apart from its characteristics.
     */
    static List<String> designations(Card card) {
        List<String> held = new ArrayList<>();
        for (Designation designation : Designation.values()) {
            if (designation.heldBy(card)) {
                held.add(designation.wireName());
            }
        }
        return held;
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
        addChosenBasics(chosen, card);
        addChosenCollections(chosen, card);
        addChosenLabels(chosen, card);
        addChosenSectorAndRolls(chosen, card);
        return chosen.render();
    }

    private static void addChosenBasics(Json.ObjectWriter chosen, Card card) {
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
    }

    private static void addChosenCollections(Json.ObjectWriter chosen, Card card) {
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
    }

    private static void addChosenLabels(Json.ObjectWriter chosen, Card card) {
        if (card.getChosenDirection() != null) {
            chosen.value("direction", card.getChosenDirection().name().toLowerCase(java.util.Locale.ROOT));
        }
        if (isSet(card.getChosenMode())) {
            chosen.value("mode", card.getChosenMode());
        }
        if (card.hasChosenEvenOdd()) {
            chosen.value("even_odd", card.getChosenEvenOdd().name().toLowerCase(java.util.Locale.ROOT));
        }
    }

    private static void addChosenSectorAndRolls(Json.ObjectWriter chosen, Card card) {
        if (isSet(card.getChosenSector())) {
            chosen.value("sector", card.getChosenSector());
        }
        if (card.getStoredRolls() != null && !card.getStoredRolls().isEmpty()) {
            chosen.value("stored_rolls", card.getStoredRolls());
        }
    }

    private static boolean isSet(String text) {
        return text != null && !text.isEmpty();
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
        return switch (zone.getZoneType()) {
            case Battlefield -> battlefieldRank(game, card);
            case Stack -> stackRank(game, card);
            case Command -> null;
            default -> zoneIndex(zone, card);
        };
    }

    private static Integer battlefieldRank(Game game, Card card) {
        Comparator<Card> byTimestamp = Comparator
                .comparingLong(Card::getLayerTimestamp).thenComparingInt(Card::getId);
        int rank = 0;
        for (Card other : game.getCardsIncludePhasingIn(ZoneType.Battlefield)) {
            if (other != card && byTimestamp.compare(other, card) < 0) {
                rank++;
            }
        }
        return rank;
    }

    private static Integer stackRank(Game game, Card card) {
        int rank = 0;
        for (var instance : game.getStack()) {
            if (isSpellOf(instance.getSpellAbility(), card)) {
                return rank;
            }
            rank++;
        }
        return null;
    }

    private static boolean isSpellOf(SpellAbility ability, Card card) {
        if (ability == null) {
            return false;
        }
        return ability.getHostCard() == card && ability.isSpell();
    }

    private static Integer zoneIndex(Zone zone, Card card) {
        int index = zone.getCards().indexOf(card);
        return index < 0 ? null : index;
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
        return manaCostJson(card.getManaCost());
    }

    /** {@link #manaCostJson(Card)} of a mana cost read elsewhere: a board without a static's. */
    static String manaCostJson(ManaCost cost) {
        Map<String, Integer> symbols = manaCostSymbols(cost);
        return symbols == null ? "null" : Json.value(symbols);
    }

    /** A mana cost's non-zero symbols as {@code mana_cost} spells them, or null for no cost. */
    static Map<String, Integer> manaCostSymbols(ManaCost cost) {
        if (cost == null || cost.isNoCost()) {
            return null;
        }
        Map<String, Integer> symbols = new LinkedHashMap<>();
        if (cost.getGenericCost() > 0) {
            symbols.put("generic", cost.getGenericCost());
        }
        for (ManaCostShard shard : cost) {
            symbols.merge(shard.toShortString(), 1, Integer::sum);
        }
        return symbols;
    }

    /** The owner of an entity, which control changes leave alone. */
    static String ownerOf(Card card) {
        Player owner = card.getOwner();
        return owner == null ? null : Refs.of(owner);
    }
}
