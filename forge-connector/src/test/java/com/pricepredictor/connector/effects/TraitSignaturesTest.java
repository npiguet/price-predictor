package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.CardTraitBase;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/** A trait's signature is kept only while the parameters it was built from still hold. */
@ExtendWith(ForgeExtension.class)
class TraitSignaturesTest {

    private static CardTraitBase aTraitOf(String card) {
        return TestCards.build(card).getSpellAbilities().get(0);
    }

    @Test
    void aTraitReadTwiceGivesTheSameSignature() {
        CardTraitBase trait = aTraitOf("Lightning Bolt");
        TraitSignatures signatures = new TraitSignatures();

        String first = signatures.of(trait);

        assertSame(first, signatures.of(trait));
    }

    @Test
    void aParameterRewrittenInPlaceGivesTheNewSignature() {
        CardTraitBase trait = aTraitOf("Lightning Bolt");
        TraitSignatures signatures = new TraitSignatures();
        String before = signatures.of(trait);

        trait.getOriginalMapParams().put("Announce", "X");

        String after = signatures.of(trait);
        assertNotEquals(before, after);
        assertEquals(new TraitSignatures().of(trait), after);
    }

    @Test
    void twoCopiesOfOneCardShareASignature() {
        TraitSignatures signatures = new TraitSignatures();

        assertEquals(signatures.of(aTraitOf("Lightning Bolt")), signatures.of(aTraitOf("Lightning Bolt")));
    }
}
