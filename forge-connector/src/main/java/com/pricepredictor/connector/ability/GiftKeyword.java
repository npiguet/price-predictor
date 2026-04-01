package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;

import static com.pricepredictor.connector.ability.SpellAbilityUtils.getAdditionalAbility;
import static com.pricepredictor.connector.ability.SpellAbilityUtils.getParam;

/**
 * Gift keyword — searches the card's spell abilities for the gift description parameter.
 */
public record GiftKeyword(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.STATIC;
    }

    public static GiftKeyword of(KeywordInterface ki, Card card) {
        String giftTitle = card.getSpellAbilities().stream()
                .flatMap(sa -> getAdditionalAbility(sa, "GiftAbility").stream())
                .findFirst()
                .flatMap(additional -> getParam(additional, "GiftDescription"))
                .map(d -> ki.getTitle() + " " + d)
                .orElse(ki.getTitle());
        return new GiftKeyword(AbilityDescription.applyCasing(giftTitle));
    }
}
