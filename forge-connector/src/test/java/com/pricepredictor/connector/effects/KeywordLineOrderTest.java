package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The order a face prints its keyword lines in (finding 1.15): the keywords'
 * original text, sorted, which is the order {@link FaceOrder#keywordIndex}
 * numbers them in, so the n-th keyword line carries {@code keyword n}. The
 * script's own {@code K:} order, and the order of Forge's keyword collection,
 * are no part of it.
 */
@ExtendWith(ForgeExtension.class)
class KeywordLineOrderTest {

    private final RulesParser parser = new RulesParser();

    /** Blightsteel Colossus scripts Trample, Infect, Indestructible: reverse alphabetical. */
    @Test
    void keywordLinesFollowTheSortedOriginalText() {
        ConvertedScript converted = ConvertedScript.fromFile(
                parser, ConvertedScript.GOLDEN_024_CARDS, "b/blightsteel_colossus.txt");

        List<String> keywordLines = new ArrayList<>();
        List<Integer> keywordKeys = new ArrayList<>();
        for (ProvenanceSidecar.Line line : converted.sidecar().lines()) {
            for (ProvenanceKey key : line.provenance()) {
                if (key.traitKind() == TraitKind.KEYWORD) {
                    keywordLines.add(converted.texts().get(line.lineIndex()));
                    keywordKeys.add(key.indexWithinKind());
                }
            }
        }
        assertEquals(List.of("static: indestructible", "static: infect", "static: trample"), keywordLines);
        assertEquals(List.of(0, 1, 2), keywordKeys);
    }
}
