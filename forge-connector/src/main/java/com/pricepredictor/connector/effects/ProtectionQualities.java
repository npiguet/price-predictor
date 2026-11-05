package com.pricepredictor.connector.effects;

import forge.card.CardType;
import forge.card.CardTypeView;
import forge.game.card.Card;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * The rare protection qualities an object has (FR-058), spelled as the
 * keyword fragment Forge writes for each.
 *
 * <p>Rare is every quality but a colour: a card type, a creature type, a
 * condition, and everything (702.16a, 702.16j). The spellings are the ones
 * {@code effects.domain.rule_outcomes.ProtectionQuality.fragment} writes, so a
 * fork's granted line and a protection copy's read alike:
 * {@code Protection:Artifact}, {@code Protection:Goblin},
 * {@code Protection:Card.MultiColor:multicolored},
 * {@code Protection from everything}.
 */
final class ProtectionQualities {

    /** Protection from everything (702.16j), which covers every object. */
    static final String EVERYTHING = "Protection from everything";

    /** The card types protection can name, as Forge spells them; rule_outcomes.CARD_TYPES. */
    private static final List<CardType.CoreType> CARD_TYPES = List.of(
            CardType.CoreType.Artifact, CardType.CoreType.Battle, CardType.CoreType.Creature,
            CardType.CoreType.Enchantment, CardType.CoreType.Instant, CardType.CoreType.Kindred,
            CardType.CoreType.Land, CardType.CoreType.Planeswalker, CardType.CoreType.Sorcery);

    /**
     * A creature type the fragment can carry: one word, or words joined by
     * hyphens, each capitalised. A type with a space in it (Time Lord) would
     * not survive as a filter, and the corpus's spelling capitalises only at
     * the hyphens.
     */
    private static final Pattern CREATURE_TYPE = Pattern.compile("[A-Z][a-z]*(-[A-Z][a-z]*)*");

    /** A condition Forge's cards print as a protection quality; rule_outcomes.CONDITIONS. */
    private record Condition(String fragment, Predicate<Card> holds) {
    }

    private static final List<Condition> CONDITIONS = List.of(
            new Condition("Protection:Card.MultiColor:multicolored",
                    card -> card.getColor().countColors() >= 2),
            new Condition("Protection:Card.MonoColor:monocolored",
                    card -> card.getColor().countColors() == 1),
            new Condition("Protection:Card.cmcLE3:mana value 3 or less",
                    card -> card.getCMC() <= 3),
            new Condition("Protection:Card.cmcGE3:mana value 3 or greater",
                    card -> card.getCMC() >= 3),
            new Condition("Protection from each color",
                    card -> card.getColor().countColors() > 0));

    private ProtectionQualities() {
    }

    /**
     * The rare qualities {@code card} has, everything left out, in a fixed
     * order: its card types, its creature types where it is a creature or
     * kindred (none for a changeling, which has them all), the conditions it
     * meets.
     */
    static List<String> of(Card card) {
        List<String> found = new ArrayList<>();
        CardTypeView type = card.getType();
        for (CardType.CoreType core : CARD_TYPES) {
            if (type.getCoreTypes().contains(core)) {
                found.add("Protection:" + core.name());
            }
        }
        if (!type.hasAllCreatureTypes()) {
            type.getCreatureTypes().stream()
                    .filter(name -> CREATURE_TYPE.matcher(name).matches())
                    .sorted()
                    .forEach(name -> found.add("Protection:" + name));
        }
        for (Condition condition : CONDITIONS) {
            if (condition.holds().test(card)) {
                found.add(condition.fragment());
            }
        }
        return found;
    }
}
