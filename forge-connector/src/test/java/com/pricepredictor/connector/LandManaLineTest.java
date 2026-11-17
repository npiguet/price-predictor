package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.ProvenanceKey;
import com.pricepredictor.connector.effects.ProvenanceSidecar;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The synthetic land-mana line of a face with basic land types.
 *
 * <p>The line stands for the mana abilities Forge builds from the basic land
 * types alone ({@code CardState.getLandManaForColor}). A mana ability the
 * script writes is a line of its own, with its own text and key: folded into
 * the land-mana line, its text vanished and its key named words it never said.
 */
@ExtendWith(ForgeExtension.class)
class LandManaLineTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();
    private static final String BOSK = "m/murmuring_bosk.txt";

    private record Converted(List<RenderedLine> rendered, ProvenanceSidecar sidecar) {

        /** The sidecar line whose rendered text is {@code text}. */
        ProvenanceSidecar.Line lineWithText(String text) {
            List<String> texts = rendered.stream().map(RenderedLine::text).toList();
            int index = texts.indexOf(text);
            assertTrue(index >= 0, "no line " + text + " in " + texts);
            return sidecar.lines().stream().filter(line -> line.lineIndex() == index)
                    .findFirst().orElseThrow();
        }
    }

    private static Converted convert(String relativePath) {
        String scriptFile = "cardsfolder/" + relativePath;
        RulesParser.ParsedCard parsed = new RulesParser().parseScript(lines(relativePath), scriptFile);
        List<RenderedLine> rendered = parsed.card().renderLines();
        return new Converted(rendered, ProvenanceSidecar.build(
                parsed.card().faces().get(0).name(), scriptFile, rendered, parsed.recorders()));
    }

    private static List<String> lines(String relativePath) {
        try {
            return Files.readAllLines(CARDS_FOLDER.resolve(relativePath));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static List<String> spellKeys(ProvenanceSidecar.Line line) {
        return line.provenance().stream().map(ProvenanceKey::indexWithinKind)
                .map(index -> "spell " + index).toList();
    }

    @Test
    void aScriptedManaAbilityOfABasicTypedLandKeepsItsOwnLine() {
        Converted bosk = convert(BOSK);
        ProvenanceSidecar.Line painLand = bosk.lineWithText(
                "activated[1]: {T}: add {W} or {B}. CARDNAME deals 1 damage to you.");
        assertEquals(List.of("spell 1"), spellKeys(painLand));
        assertTrue(painLand.script().scriptText().startsWith("AB$ Mana | Cost$ T | Produced$ Combo W B"),
                painLand.script().scriptText());
    }

    @Test
    void theLandManaLineKeepsOnlyTheIntrinsicManaAbility() {
        ProvenanceSidecar.Line forest = convert(BOSK).lineWithText("activated[2]: {T}: add {G}.");
        assertEquals(List.of("spell 2"), spellKeys(forest));
        assertTrue(forest.script().scriptText().contains("Produced$ G"), forest.script().scriptText());
    }
}
