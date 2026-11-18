package com.pricepredictor.connector.ability;

import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;

/** A Haunt keyword's field, read from {@code Haunt:<SvarName>}: the SVar holding its effect. */
record Haunt(String svarName) {

    static Haunt of(KeywordInterface ki) {
        return new Haunt(new KeywordSpec(ki.getOriginal()).rest(1));
    }

    /** The {@code SpellDescription$} of the effect SVar on {@code card}, or null. */
    String effectDescription(Card card) {
        return svarName == null
                ? null : SpellAbilityUtils.svarParam(card.getSVar(svarName), "SpellDescription");
    }
}
