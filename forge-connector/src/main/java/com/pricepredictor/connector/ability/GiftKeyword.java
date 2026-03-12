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
        String giftTitle = ki.getTitle();
        for (SpellAbility sa : card.getSpellAbilities()) {
            if (sa.hasAdditionalAbility("GiftAbility")) {
                String giftDesc = sa.getAdditionalAbility("GiftAbility")
                        .getParam("GiftDescription");
                if (giftDesc != null && !giftDesc.isEmpty()) {
                    giftTitle = giftTitle + " " + giftDesc;
                }
                break;
            }
        }
        return new GiftKeyword(AbilityDescription.applyCasing(giftTitle));
    }
}
