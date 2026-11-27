package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.card.CardStateName;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A meld half's melded face keys into its own sidecar's face 1.
 *
 * <p>Forge reads the Meld state of a half whose script does not hold the
 * melded face from its meld partner's script, and the collector keys that
 * state's traits as face 1 of the half's own file. Each trait's key must
 * resolve in the half's sidecar exactly as the same trait's key resolves in
 * the partner's: to a line with the same text, or to a dropped key.
 */
@ExtendWith(ForgeExtension.class)
class MeldFaceTest {

    private final RulesParser parser = new RulesParser();

    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '"', value = {
            "Argoth, Sanctum of Nature    | Titania, Voice of Gaea",
            "Bruna, the Fading Light      | Gisela, the Broken Blade",
            "Fang, Fearless l'Cie         | Vanille, Cheerful l'Cie",
            "Hanweir Garrison             | Hanweir Battlements",
            "Midnight Scavengers          | Graf Rats",
            "Phyrexian Dragon Engine      | Mishra, Claimed by Gix",
            "The Mightstone and Weakstone | Urza, Lord Protector",
    })
    void aMeldHalfsMeldedFaceJoinsItsOwnFaceOneAsThePartnersDoes(String half, String partner)
            throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card borrower = board.battlefield(half, board.you);
        Card holder = board.battlefield(partner, board.you);

        List<String> borrowed = meldedJoins(borrower);
        assertEquals(meldedJoins(holder), borrowed);
        assertTrue(borrowed.stream().anyMatch(join -> join.contains("-> line")),
                "no melded trait of " + half + " joins a line: " + borrowed);
    }

    /**
     * How each trait of the card's Meld state joins the card's own sidecar:
     * its key without the script file, then the text of the line it joins,
     * {@code dropped}, or {@code undeclared}.
     */
    private List<String> meldedJoins(Card card) throws IOException {
        ConvertedScript script = convertedScriptOf(card);
        CardState melded = card.getState(CardStateName.Meld);
        List<String> joins = new ArrayList<>();
        for (ProvenanceKey key : keysOf(melded)) {
            assertEquals(script.sidecar().scriptFile(), key.scriptFile(), key.describe());
            assertEquals(1, key.face(), key.describe());
            joins.add(withoutScript(key) + " -> " + joinOf(script, key));
        }
        return joins;
    }

    private static List<ProvenanceKey> keysOf(CardState state) {
        List<CardTraitBase> traits = new ArrayList<>(state.getSpellAbilities());
        traits.addAll(state.getTriggers());
        traits.addAll(state.getStaticAbilities());
        traits.addAll(state.getReplacementEffects());
        List<ProvenanceKey> keys = new ArrayList<>();
        traits.forEach(trait -> keys.add(ProvenanceResolver.of(trait)));
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            keys.add(ProvenanceResolver.keyOfKeyword(state, keyword));
        }
        keys.removeIf(Objects::isNull);
        return keys;
    }

    private static String withoutScript(ProvenanceKey key) {
        String described = key.describe();
        return described.substring(described.indexOf(") ") + 2);
    }

    /** The line {@code key} names, or the line its bare key names, as the sidecar contract joins them. */
    private static String joinOf(ConvertedScript script, ProvenanceKey key) {
        for (ProvenanceKey candidate : List.of(key, key.withoutBuilt())) {
            for (ProvenanceSidecar.Line line : script.sidecar().lines()) {
                if (line.provenance().contains(candidate)) {
                    return "line " + script.texts().get(line.lineIndex());
                }
            }
        }
        return script.sidecar().droppedKeys().contains(key) ? "dropped" : "undeclared";
    }

    /** The card's converted script and sidecar as {@code python -m price_predictor convert} writes them. */
    private ConvertedScript convertedScriptOf(Card card) throws IOException {
        String scriptFile = ScriptLocator.scriptFileOf(card);
        List<String> source = Files.readAllLines(Path.of(card.getRules().getPath()));
        return ConvertedScript.of(parser.parseScript(source, scriptFile), scriptFile);
    }
}
