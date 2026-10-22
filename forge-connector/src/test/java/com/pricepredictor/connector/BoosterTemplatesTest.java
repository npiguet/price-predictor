package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.item.SealedTemplate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(ForgeExtension.class)
class BoosterTemplatesTest {

    /**
     * Every set Forge's booster map holds resolves to the same template here.
     *
     * <p>The map holds only Draft templates under a bare set code (other
     * booster types carry an affix), so the edition's Draft template is the
     * whole lookup, and no set the map knew is lost by reading the edition.
     */
    @Test
    void everySetTheBoosterMapHoldsResolvesToTheSameTemplate() {
        List<String> differing = new ArrayList<>();
        for (CardEdition edition : StaticData.instance().getEditions()) {
            String code = edition.getCode();
            SealedTemplate mapped = StaticData.instance().getBoosters().get(code);
            if (mapped != null && BoosterTemplates.forSet(code).orElse(null) != mapped) {
                differing.add(code);
            }
        }
        assertTrue(differing.isEmpty(), "resolved differently: " + differing);
    }

    /** An alias names the set it is an alias of: {@code CON} is Conflux, {@code CFX}. */
    @Test
    void anAliasResolvesToItsSetsTemplate() {
        SealedTemplate conflux = BoosterTemplates.forSet("CFX").orElseThrow();

        assertSame(conflux, BoosterTemplates.forSet("CON").orElseThrow());
    }

    @Test
    void anUnknownSetHasNoTemplate() {
        assertEquals(Optional.empty(), BoosterTemplates.forSet("XXXXUNKNOWN"));
        assertEquals(Optional.empty(), BoosterTemplates.forSet(null));
    }
}
