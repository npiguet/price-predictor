package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The keyword keys of token copies, mutated permanents and the traits a
 * copy builds fresh (finding 1.8).
 *
 * <p>A token copy's state holds the original's keywords plus the ones its
 * copy effect added, and numbering that state's keywords shifted each later
 * printed keyword one line on. A mutated permanent's keywords come from
 * several cards, each printed in its own file, as do its triggers, statics,
 * replacements and abilities, which keep the state of the card that printed
 * them. A trait Forge builds on a
 * copy's state (a land's mana ability) is no line of the copy effect's. Each
 * key is read against the real sidecar of the card it names.
 */
@ExtendWith(ForgeExtension.class)
class CopyKeywordKeyTest {

    private final RulesParser parser = new RulesParser();

    /**
     * Kiki-Jiki copies Moldering Karok (trample, lifelink): the token's
     * Lifelink and Trample join their own lines of Karok's script, and the
     * Haste the copy effect added has no key and is a granted line.
     */
    @Test
    void aTokenCopysKeywordsJoinTheOriginalsLinesAndAnAddedOneIsGranted() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card kiki = board.battlefield("Kiki-Jiki, Mirror Breaker", board.you);
        Card karok = board.battlefield("Moldering Karok", board.you);
        SpellAbility copy = board.ability(kiki, ApiType.CopyPermanent);
        copy.resetTargets();
        copy.getTargets().add(karok);
        board.resolve(copy, board.you);
        board.game.getAction().checkStaticAbilities();
        Card token = tokenNamed(board, "Moldering Karok");
        ConvertedScript karoks = convertedScriptOf(karok);

        assertJoins(karoks, keyOf(token, "Lifelink"), "lifelink");
        assertJoins(karoks, keyOf(token, "Trample"), "trample");
        assertNull(keyOf(token, "Haste"), "the haste the copy effect added");

        List<Map<String, Object>> lines = linesOf(board, token);
        Map<String, Object> haste = only(lines, "Haste");
        assertEquals(List.of(), haste.get("keys"), haste.toString());
        assertEquals(true, haste.get("granted"), haste.toString());
        assertEquals("effect", haste.get("grant"), haste.toString());
        for (String printed : List.of("Lifelink", "Trample")) {
            Map<String, Object> line = only(lines, printed);
            assertEquals(false, line.get("granted"), line.toString());
            assertFalse(line.containsKey("grant"), line.toString());
        }
    }

    /**
     * Dreamtail Heron (mutate, flying) mutates onto Serra Angel (flying,
     * vigilance): Mutate keys into the Heron's file, Vigilance into the
     * Angel's, and the one Flying both print into the file of the card on
     * top, whose keyword the merged permanent holds. Every line is printed.
     */
    @Test
    void aMutatedPermanentsKeywordsJoinTheirOwnCardsLines() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.you);
        Card heron = board.hand("Dreamtail Heron", board.you);
        SpellAbility mutate = heron.getSpellAbilities().stream()
                .filter(SpellAbility::isMutate).findFirst().orElseThrow();
        mutate.resetTargets();
        mutate.getTargets().add(angel);
        board.resolve(mutate, board.you);
        board.game.getAction().checkStaticAbilities();
        Card merged = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::isMutated).findFirst().orElseThrow();
        Card top = merged.getTopMergedCard();
        ConvertedScript herons = convertedScriptOf(heron);
        ConvertedScript angels = convertedScriptOf(angel);

        assertJoins(herons, keyOf(merged, "Mutate:3 U"), "mutate");
        assertJoins(angels, keyOf(merged, "Vigilance"), "vigilance");
        assertJoins(top.getName().equals("Dreamtail Heron") ? herons : angels,
                keyOf(merged, "Flying"), "flying");

        List<Map<String, Object>> keywords = linesOf(board, merged).stream()
                .filter(line -> "keyword".equals(line.get("kind"))).toList();
        assertEquals(3, keywords.size(), keywords.toString());
        keywords.forEach(line -> assertEquals(false, line.get("granted"), line.toString()));
    }

    /**
     * Dreamtail Heron merged under Serra Angel, as {@code MutateEffect} merges
     * it: the one Flying both print keys into the Angel's file, the card on
     * top, and Mutate still into the Heron's.
     */
    @Test
    void aKeywordTwoMergedCardsPrintKeysIntoTheTopCardsFile() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.you);
        Card heron = board.card("Dreamtail Heron", board.you);
        heron.setMergedToCard(angel);
        angel.addMergedCard(angel);
        angel.addMergedCard(heron);
        long timestamp = board.game.getNextTimestamp();
        angel.setMutatedTimestamp(timestamp);
        angel.addCloneState(CardFactory.getMutatedCloneStates(angel, null), timestamp);
        board.game.getAction().checkStaticAbilities();

        assertJoins(convertedScriptOf(angel), keyOf(angel, "Flying"), "flying");
        assertJoins(convertedScriptOf(heron), keyOf(angel, "Mutate:3 U"), "mutate");
    }

    /**
     * Dreamtail Heron, whose mutate trigger is its own, merged on top of
     * Marjhan, which prints a replacement, a static, a trigger and two
     * activated abilities: each trait joins its own line in the file of the
     * card that prints it, and is printed.
     */
    @Test
    void aMutatedPermanentsTraitsJoinTheLinesOfTheCardOnTop() throws IOException {
        assertMergedTraitsJoinTheirOwnLines(true);
    }

    /** As above, with Dreamtail Heron merged under Marjhan. */
    @Test
    void aMutatedPermanentsTraitsJoinTheLinesOfTheCardUnder() throws IOException {
        assertMergedTraitsJoinTheirOwnLines(false);
    }

    private void assertMergedTraitsJoinTheirOwnLines(boolean heronOnTop) throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card marjhan = board.battlefield("Marjhan", board.you);
        board.battlefield("Island", board.you);
        Card heron = merge(board, marjhan, "Dreamtail Heron", heronOnTop);
        Map<String, ConvertedScript> scripts = Map.of(
                "Dreamtail Heron", convertedScriptOf(heron), "Marjhan", convertedScriptOf(marjhan));
        Map<String, String[]> printers = Map.of(
                "draw a card", new String[] {"Dreamtail Heron", "draw a card"},
                "doesn't untap", new String[] {"Marjhan", "doesn't untap"},
                "can't attack", new String[] {"Marjhan", "can't attack"},
                "no Islands", new String[] {"Marjhan", "no islands"},
                "Untap this", new String[] {"Marjhan", "untap this"},
                "1 damage", new String[] {"Marjhan", "1 damage"});

        List<String> seen = new ArrayList<>();
        for (CardTraitBase trait : ownTraitsOf(marjhan.getCurrentState())) {
            String description = descriptionOf(trait);
            String fragment = printers.keySet().stream().filter(description::contains)
                    .findFirst().orElseThrow(() -> new AssertionError("unexpected trait " + trait));
            String[] printer = printers.get(fragment);
            assertJoins(scripts.get(printer[0]), ProvenanceResolver.of(trait), printer[1]);
            seen.add(fragment);
        }
        assertEquals(printers.keySet(), Set.copyOf(seen), seen.toString());
        for (Map<String, Object> line : linesOf(board, marjhan)) {
            assertEquals(false, line.get("granted"), line.toString());
        }
    }

    /**
     * {@code name} merged with {@code target} as {@code MutateEffect} merges
     * them, on top or under.
     */
    private static Card merge(CharacterizationBoard board, Card target, String name, boolean onTop) {
        Card mutating = board.card(name, board.you);
        mutating.setMergedToCard(target);
        target.addMergedCard(target);
        if (onTop) {
            target.addMergedCardToTop(mutating);
        } else {
            target.addMergedCard(mutating);
        }
        long timestamp = board.game.getNextTimestamp();
        target.setMutatedTimestamp(timestamp);
        target.addCloneState(CardFactory.getMutatedCloneStates(target, null), timestamp);
        board.game.getAction().checkStaticAbilities();
        assertEquals(onTop ? mutating : target, target.getTopMergedCard());
        return mutating;
    }

    /** A state's triggers, statics, replacements and activated abilities no keyword built. */
    private static List<CardTraitBase> ownTraitsOf(CardState state) {
        List<CardTraitBase> traits = new ArrayList<>();
        traits.addAll(state.getTriggers());
        traits.addAll(state.getStaticAbilities());
        traits.addAll(state.getReplacementEffects());
        state.getSpellAbilities().stream().filter(SpellAbility::isActivatedAbility).forEach(traits::add);
        return traits.stream().filter(trait -> trait.getKeyword() == null).toList();
    }

    private static String descriptionOf(CardTraitBase trait) {
        for (String param : List.of("TriggerDescription", "Description", "SpellDescription")) {
            if (trait.hasParam(param)) {
                return trait.getParam(param);
            }
        }
        return trait.toString();
    }

    /**
     * Vesuva enters as a copy of a Forest: the Forest's mana ability, which
     * Forge builds on Vesuva's copy state, keys to no line, not to Vesuva's
     * "enter as a copy" line.
     */
    @Test
    void aManaAbilityBuiltOnALandCopysStateIsRuntimeOnly() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Forest", board.opponent);
        Card inHand = board.hand("Vesuva", board.you);
        board.resolve(board.spell(inHand, board.you), board.you);
        Card vesuva = board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getId() == inHand.getId()).findFirst().orElseThrow();
        assertTrue(vesuva.isCloned(), "Vesuva entered as a copy of the Forest");
        assertEquals("Forest", vesuva.getName());

        List<SpellAbility> mana = vesuva.getManaAbilities().stream().toList();
        assertEquals(1, mana.size(), mana.toString());

        assertNull(ProvenanceResolver.of(mana.get(0)), "the Forest's mana ability on Vesuva");
    }

    // ── the board ───────────────────────────────────────────────────────

    private static Card tokenNamed(CharacterizationBoard board, String name) {
        return board.game.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isToken() && name.equals(card.getName()))
                .findFirst().orElseThrow();
    }

    /** The key of the card's one keyword whose text is {@code original}. */
    private static ProvenanceKey keyOf(Card card, String original) {
        List<KeywordInterface> found = card.getCurrentState().getIntrinsicKeywords().stream()
                .filter(keyword -> original.equals(keyword.getOriginal())).toList();
        assertEquals(1, found.size(), card + "'s " + original + ": "
                + card.getCurrentState().getIntrinsicKeywords());
        return ProvenanceResolver.keyOfKeyword(card.getCurrentState(), found.get(0));
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) RecordLines.parse(
                EntityJson.render(board.game, card, StaticSuppression.NONE)).get("lines");
    }

    private static Map<String, Object> only(List<Map<String, Object>> lines, String fragment) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> fragment.equals(line.get("fragment"))).toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }

    // ── the sidecar ─────────────────────────────────────────────────────

    /** The card's converted script and sidecar as {@code python -m price_predictor convert} writes them. */
    private ConvertedScript convertedScriptOf(Card card) throws IOException {
        String scriptFile = ScriptLocator.scriptFileOf(card);
        List<String> source = Files.readAllLines(Path.of(card.getRules().getPath()));
        return ConvertedScript.of(parser.parseScript(source, scriptFile), scriptFile);
    }

    /** That the key names a line of the script, and that line is the keyword's. */
    private static void assertJoins(ConvertedScript script, ProvenanceKey key, String keyword) {
        assertNotNull(key, keyword);
        assertEquals(script.sidecar().scriptFile(), key.scriptFile(), keyword + ": " + key);
        int line = -1;
        for (ProvenanceSidecar.Line each : script.sidecar().lines()) {
            if (each.provenance().contains(key)) line = each.lineIndex();
        }
        assertTrue(line >= 0, keyword + " joins no line: " + key + " in " + script.sidecar().lines());
        String text = script.texts().get(line).toLowerCase(Locale.ROOT);
        assertTrue(text.contains(keyword), keyword + " joins \"" + text + "\"");
    }
}
