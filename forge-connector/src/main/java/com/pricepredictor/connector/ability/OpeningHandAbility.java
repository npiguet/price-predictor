package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * MayEffectFromOpeningHand / MayEffectFromOpeningDeck keyword.
 *
 * <p>The Forge game engine processes this keyword at game-start (GameAction.java)
 * rather than registering a standard trigger or spell, so it does not appear in
 * the card's ability lists. The description is recovered by building an SA from
 * the named SVar; Chancellor-cycle cards fall back to a "RevealCard" SVar when
 * the primary SVar lacks a SpellDescription.
 */
public final class OpeningHandAbility {

    private OpeningHandAbility() {}

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        String original = ki.getOriginal(); // "MayEffectFromOpeningHand:SvarName" or "...Deck:..."
        String svarName = original.split(":", 3)[1];
        Card hostCard = ki.getHostCard();
        if (hostCard == null) return List.of();

        String desc = resolveDescription(hostCard, svarName);
        if (desc == null || desc.isEmpty()) return List.of();

        String normalized = AbilityDescription.normalize(desc);
        if (normalized == null) return List.of();
        return List.of(new TextAbility(AbilityType.TRIGGERED,
                AbilityDescription.applyCasing(normalized)));
    }

    private static String resolveDescription(Card hostCard, String svarName) {
        if (hostCard.hasSVar(svarName)) {
            SpellAbility svarSa = AbilityFactory.getAbility(hostCard, svarName);
            if (svarSa != null) {
                String desc = svarSa.getParam("SpellDescription");
                if (desc != null && !desc.isEmpty()) return desc;
            }
        }
        // Chancellor-cycle fallback: primary SVar has no SpellDescription; use RevealCard SVar.
        if (hostCard.hasSVar("RevealCard")) {
            SpellAbility revealSa = AbilityFactory.getAbility(hostCard, "RevealCard");
            if (revealSa != null) return revealSa.getParam("SpellDescription");
        }
        return null;
    }
}
