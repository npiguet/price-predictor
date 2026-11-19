package com.pricepredictor.connector.effects;

import forge.game.ability.ApiType;
import forge.game.card.CardState;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The copy effect of a Clone's own printed state that adds a keyword as an
 * exception (707.9b): Dack's Duplicate's "enter as a copy ..., except it has
 * haste and dethrone". Such a keyword is part of the copy's copiable values,
 * so a token copy of the Clone holds it too, and it is printed on the
 * Clone's own card: in its {@code Clone} clause's {@code AddKeywords$}, which
 * {@code CardFactory.getCloneStates} reads.
 *
 * <p>The Clone's printed state is read, not the one it shows, which a later
 * copy effect on the Clone (Mirrorweave) replaces.
 */
final class CopyExceptions {

    /** The {@code AddKeywords$} prefix that adds each keyword only when it is new. */
    private static final String IF_NEW = "IfNew ";

    private CopyExceptions() {
    }

    /**
     * The {@code Clone} clause of {@code printed} whose {@code AddKeywords$}
     * lists {@code keyword}: of a replacement's or trigger's effect, or of a
     * spell or activated ability, its sub-abilities included. Null for none,
     * and for no state.
     */
    static SpellAbility clauseAdding(CardState printed, String keyword) {
        if (printed == null) {
            return null;
        }
        for (SpellAbility effect : effectsOf(printed)) {
            for (SpellAbility clause = effect; clause != null; clause = clause.getSubAbility()) {
                if (adds(clause, keyword)) {
                    return clause;
                }
            }
        }
        return null;
    }

    /** Whether {@code clause} is a {@code Clone} clause whose {@code AddKeywords$} lists {@code keyword}. */
    private static boolean adds(SpellAbility clause, String keyword) {
        return clause.getApi() == ApiType.Clone && addedKeywords(clause).contains(keyword);
    }

    /** The keywords a clause's {@code AddKeywords$} lists, as {@code CardFactory.getCloneStates} splits them. */
    private static List<String> addedKeywords(SpellAbility clause) {
        String added = clause.getParam("AddKeywords");
        if (added == null) {
            return List.of();
        }
        String listed = added.startsWith(IF_NEW) ? added.substring(IF_NEW.length()) : added;
        return Arrays.asList(listed.split(" & "));
    }

    /** The abilities a state's replacements, triggers and spell abilities execute. */
    private static List<SpellAbility> effectsOf(CardState printed) {
        List<SpellAbility> effects = new ArrayList<>();
        for (ReplacementEffect replacement : printed.getReplacementEffects()) {
            addIfAny(effects, replacement.getOverridingAbility());
        }
        for (Trigger trigger : printed.getTriggers()) {
            addIfAny(effects, trigger.getOverridingAbility());
        }
        printed.getSpellAbilities().forEach(effects::add);
        return effects;
    }

    private static void addIfAny(List<SpellAbility> effects, SpellAbility effect) {
        if (effect != null) {
            effects.add(effect);
        }
    }
}
