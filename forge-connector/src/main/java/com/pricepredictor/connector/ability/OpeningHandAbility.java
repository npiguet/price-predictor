package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import java.util.List;
import java.util.Optional;

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
        String svarName = KeywordFields.from(ki, 3).field(1); // "MayEffectFromOpeningHand:SvarName" or "...Deck:..."
        Card hostCard = ki.getHostCard();
        if (hostCard == null) return List.of();

        return resolveDescription(hostCard, svarName)
                .flatMap(AbilityDescription::normalize)
                .map(n -> List.<Ability>of(new TextAbility(AbilityType.TRIGGERED, n)))
                .orElse(List.of());
    }

    private static Optional<String> resolveDescription(Card hostCard, String svarName) {
        // Chancellor-cycle fallback: primary SVar has no SpellDescription; use RevealCard SVar.
        return resolveFromSVar(hostCard, svarName)
                .or(() -> resolveFromSVar(hostCard, "RevealCard"));
    }

    private static Optional<String> resolveFromSVar(Card hostCard, String svarName) {
        if (!hostCard.hasSVar(svarName)) return Optional.empty();
        return Optional.ofNullable(AbilityFactory.getAbility(hostCard, svarName))
                .map(sa -> sa.getParam("SpellDescription"))
                .filter(d -> d != null && !d.isEmpty());
    }
}
