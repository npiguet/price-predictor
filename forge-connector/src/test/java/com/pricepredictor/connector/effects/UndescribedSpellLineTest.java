package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.card.CardStateName;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The line of an instant's or sorcery's spell whose script gives it no
 * {@code SpellDescription$} (finding 1.2, task A19): the trait's own
 * {@code Description$}, else the face's oracle sentences no other line of the
 * face states, keyed with the spell's key. A spell with no such text anywhere
 * keeps its key in {@code dropped_keys}.
 */
@ExtendWith(ForgeExtension.class)
class UndescribedSpellLineTest {

    private static final Path FORGE_CARDS = ForgeEnvironmentInitializer.findCardsFolder();

    private final RulesParser parser = new RulesParser();

    private ConvertedScript golden(String relativePath) {
        return ConvertedScript.fromFile(parser, ConvertedScript.GOLDEN_024_CARDS, relativePath);
    }

    private ConvertedScript forge(String relativePath) {
        return ConvertedScript.fromFile(parser, FORGE_CARDS, relativePath);
    }

    private static ProvenanceKey spell(String relativePath, int face, int index) {
        return new ProvenanceKey("cardsfolder/" + relativePath, face, TraitKind.SPELL, index);
    }

    /** The text of the one line declaring {@code key}. */
    private static String lineOf(ConvertedScript converted, ProvenanceKey key) {
        List<String> texts = converted.sidecar().lines().stream()
                .filter(line -> line.provenance().contains(key))
                .map(line -> converted.texts().get(line.lineIndex()))
                .toList();
        assertEquals(1, texts.size(), key.describe() + " in " + converted.texts());
        return texts.get(0);
    }

    /** The waterbend cost is the static's line; the spell takes the oracle's other sentence. */
    @Test
    void aSpellTakesTheOracleSentencesNoOtherLineStates() {
        ConvertedScript wave = golden("c/crashing_wave.txt");

        assertEquals("spell[1]: tap up to X target creatures, then distribute three stun counters"
                        + " among tapped creatures your opponents control.",
                lineOf(wave, spell("c/crashing_wave.txt", 0, 0)));
        assertEquals(List.of(), wave.sidecar().droppedKeys());
    }

    @Test
    void aSpellWithADescriptionOfItsOwnTakesIt() {
        ConvertedScript fire = golden("g/garbage_fire.txt");

        assertEquals("spell[1]: CARDNAME deals damage to target creature equal to the highest"
                        + " number you noted for cards named CARDNAME.",
                lineOf(fire, spell("g/garbage_fire.txt", 0, 0)));
    }

    /** The oracle no longer falls back to an unkeyed {@code text:} line. */
    @Test
    void theSpellLineTakesThePlaceOfTheUnkeyedOracleLine() {
        ConvertedScript invasion = golden("t/tyranid_invasion.txt");

        assertEquals("spell[1]: create a number of 3/3 green tyranid warrior creature tokens"
                        + " with trample equal to the number of opponents you have.",
                lineOf(invasion, spell("t/tyranid_invasion.txt", 0, 0)));
        assertTrue(invasion.texts().stream().noneMatch(text -> text.startsWith("text: ")),
                invasion.texts().toString());
    }

    /** Cascade and rebound are the card's whole text: its dummy spell states nothing. */
    @Test
    void aSpellWithNoTextAnywhereStaysDropped() {
        ConvertedScript vortex = forge("i/into_the_time_vortex.txt");

        assertEquals(List.of(spell("i/into_the_time_vortex.txt", 0, 0)),
                vortex.sidecar().droppedKeys());
        assertTrue(vortex.texts().stream().noneMatch(text -> text.startsWith("spell")),
                vortex.texts().toString());
    }

    /**
     * An adventure is keyed on its own face, where the collector keys it: the
     * main face, which lists it among its spells, declares no key of it.
     */
    @Test
    void anAdventureSpellIsKeyedOnItsOwnFaceOnly() {
        String path = "t/twice_upon_a_time_unlikely_meeting.txt";
        ConvertedScript twice = forge(path);
        Card card = TestCards.build("Twice Upon a Time");
        SpellAbility adventure = card.getState(CardStateName.Secondary).getFirstSpellAbility();

        assertEquals(spell(path, 1, 0), ProvenanceResolver.of(adventure));
        assertTrue(lineOf(twice, spell(path, 1, 0)).startsWith("spell[1]: search your library"));
        assertEquals(List.of(), twice.sidecar().droppedKeys());
    }
}
