package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Charm ability with nested OPTION sub-abilities. If the charm has no description,
 * the factory returns the choices as top-level OPTION abilities instead.
 */
public record CharmAbility(String descriptionText, List<Ability> subAbilities) implements Ability {

    public CharmAbility {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    @Override
    public AbilityType type() {
        return AbilityType.SPELL;
    }

    public static List<Ability> fromSpellAbility(SpellAbility sa) {
        String charmDesc = sa.getParam("SpellDescription");
        if (charmDesc != null && !charmDesc.isEmpty()) {
            charmDesc = AbilityDescription.stripReminderText(charmDesc);
        }
        if ((charmDesc == null || charmDesc.isEmpty()) && sa.hasParam("Pawprint")) {
            String total = sa.getParam("Pawprint");
            charmDesc = "Choose up to " + total + " {P} worth of modes.";
            if ("True".equals(sa.getParam("CanRepeatModes"))) {
                charmDesc += " You may choose the same mode more than once.";
            }
        }

        // Collect charm choices as sub-abilities
        List<Ability> choiceSubs = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices != null) {
            for (var choice : choices) {
                String choiceDesc = SpellAbilityUtils.findParamInChain(choice, "SpellDescription");
                if (choiceDesc != null) {
                    choiceDesc = AbilityDescription.stripReminderText(choiceDesc);
                }
                String pawprint = choice.getParam("Pawprint");
                if (pawprint != null) {
                    choiceDesc = "{P}".repeat(Integer.parseInt(pawprint))
                            + " \u2014 " + choiceDesc;
                }
                choiceSubs.add(new TextAbility(AbilityType.OPTION,
                        AbilityDescription.applyCasing(choiceDesc)));
            }
        }

        List<Ability> result = new ArrayList<>();
        if (charmDesc != null && !charmDesc.isEmpty()) {
            result.add(new CharmAbility(AbilityDescription.applyCasing(charmDesc), choiceSubs));
        } else {
            // No charm description — add choices as top-level abilities
            result.addAll(choiceSubs);
        }
        return result;
    }
}
