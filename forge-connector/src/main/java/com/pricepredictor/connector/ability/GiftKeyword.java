package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;

/**
 * Gift keyword — searches the card's spell abilities for the gift description parameter.
 */
public record GiftKeyword(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.STATIC;
    }

    public static GiftKeyword of(KeywordInterface ki, Card card) {
        String giftTitle = card.getSpellAbilities().stream()
                .filter(sa -> sa.hasAdditionalAbility("GiftAbility"))
                .findFirst()
                .map(sa -> sa.getAdditionalAbility("GiftAbility").getParam("GiftDescription"))
                .filter(d -> d != null && !d.isEmpty())
                .map(d -> ki.getTitle() + " " + d)
                .orElse(ki.getTitle());
        return new GiftKeyword(AbilityDescription.applyCasing(giftTitle));
    }
}
