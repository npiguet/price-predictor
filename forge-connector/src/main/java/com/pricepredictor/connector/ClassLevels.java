package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.TextAbility;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A Class face's ability list, rearranged so it reads as the printed card
 * does: one line per level, level one first, the levels in order.
 *
 * <p>Two steps, in this order: re-type the unlevelled abilities as level
 * one, then sort the levels ahead of everything else. A level's own traits
 * render no line of their own (the Class keyword prints the level), so no
 * line here duplicates a level.
 */
final class ClassLevels {

    /**
     * Levels first, in level order; every other ability after them, in the
     * order it already had. {@link List#sort} is stable, so the zero the
     * second key gives every non-level ability keeps that order.
     */
    private static final Comparator<Ability> LEVELS_FIRST =
            Comparator.comparing((Ability a) -> !isLevel(a))
                    .thenComparingInt(a -> isLevel(a) ? a.ordinal() : 0);

    /** The kinds of line a Class card's level one is made of. */
    private static final Set<AbilityType> LEVEL_ONE_KINDS =
            EnumSet.of(AbilityType.STATIC, AbilityType.TRIGGERED, AbilityType.REPLACEMENT);

    private ClassLevels() {}

    /**
     * {@code abilities} rearranged as a Class face prints them; a re-typed
     * line takes the keys of the one it replaces through {@code duplicates}.
     */
    static List<Ability> arrange(List<Ability> abilities, DuplicateLines duplicates) {
        List<Ability> levels = new ArrayList<>(abilities);
        retypeLevelOne(levels, duplicates);
        levels.sort(LEVELS_FIRST);
        return levels;
    }

    /**
     * Re-type each static, triggered or replacement ability as level one: on
     * a Class card, whatever no level keyword claims is the level the card
     * starts at. The level-one line is a fresh object, and the recorder keys
     * by identity, so it replaces the ability's line with its keys.
     */
    private static void retypeLevelOne(List<Ability> abilities, DuplicateLines duplicates) {
        for (int i = 0; i < abilities.size(); i++) {
            Ability a = abilities.get(i);
            if (LEVEL_ONE_KINDS.contains(a.type())) {
                abilities.set(i, duplicates.replace(
                        a, new TextAbility(AbilityType.LEVEL, a.descriptionText(), 1)));
            }
        }
    }

    private static boolean isLevel(Ability a) {
        return a.type() == AbilityType.LEVEL;
    }
}
