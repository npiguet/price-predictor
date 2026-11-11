package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A trait Forge attaches to a land at runtime never takes a printed line's
 * key, and a printed line a change leaves keeps its own.
 *
 * <p>A basic land type's mana ability is not a line of the card: Forge's
 * layer 4 builds it from the land's types as they stand. A change of land
 * type (Blood Moon, Spreading Seas, an {@code Animate} that sets the type)
 * clears the land's printed lines and adds the new type's mana ability; a
 * land type added "in addition" (Urborg) adds its mana ability among the
 * printed type's, in colour order. Either way the live spell list no longer
 * matches the one the sidecar numbered, and a live index can name another
 * printed line. Every case here checks each line of the land's snapshot and
 * the acting key of its mana record against the land's own sidecar: a line
 * not flagged granted names a sidecar line, and a granted one names no
 * position the sidecar declares.
 */
@ExtendWith(ForgeExtension.class)
class RuntimeLandTraitKeysTest {

    private static final Method ON_MANA_PRODUCED =
            HookShapes.method(HookShapes.Mana.class, "onManaProduced");
    private static final Method ON_PAYING_COSTS =
            HookShapes.method(HookShapes.Payment.class, "onPayingCosts");
    private static final Method ON_COSTS_PAID =
            HookShapes.method(HookShapes.Payment.class, "onCostsPaid");

    private final RulesParser parser = new RulesParser();

    @TempDir
    Path tempDir;

    /**
     * Blood Moon takes every printed line of Mishra's Factory and gives it a
     * Mountain's mana ability: that ability is granted, and neither its line
     * nor its mana record names one of the Factory's printed lines.
     */
    @Test
    void theManaAbilityBloodMoonGivesANonbasicLandNamesNoPrintedLine() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card factory = board.battlefield("Mishra's Factory", board.you);
        board.battlefield("Blood Moon", board.you);
        board.game.getAction().checkStaticAbilities();
        SpellAbility red = manaAbility(factory, "R");

        Sidecar sidecar = sidecarOf(factory);
        List<Map<String, Object>> lines = lines(board, factory);
        assertJoins(sidecar, lines);
        assertEquals(List.of(true), granted(lines), "the Factory's lines under Blood Moon");
        assertNamesNoDeclaredLine(sidecar, ProvenanceResolver.of(red), "the Mountain's mana ability");

        String record = manaRecord(board, red, "R");
        for (Map<String, Object> key : RecordLines.abilityKeys(record)) {
            assertFalse(sidecar.declares(describe(key)),
                    "the mana record acts through a printed line: " + record);
        }
    }

    /**
     * Spreading Seas makes Tundra an Island only: its Island mana ability is
     * the one it prints, and keeps the sidecar's mana line; its Plains mana
     * ability is gone.
     */
    @Test
    void aPrintedLandTypesManaAbilityATypeChangeLeavesKeepsItsLine() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card tundra = board.battlefield("Tundra", board.you);
        board.battlefield("Spreading Seas", board.you).attachToEntity(tundra, null);
        board.game.getAction().checkStaticAbilities();
        SpellAbility blue = manaAbility(tundra, "U");

        Sidecar sidecar = sidecarOf(tundra);
        List<Map<String, Object>> lines = lines(board, tundra);
        assertJoins(sidecar, lines);
        assertEquals(List.of(false), granted(lines), "Tundra's lines under Spreading Seas");
        assertTrue(sidecar.names(ProvenanceResolver.of(blue).describe()),
                "Tundra's Island mana ability left its line: " + ProvenanceResolver.of(blue));

        String record = manaRecord(board, blue, "U");
        assertEquals(List.of(ProvenanceResolver.of(blue).describe()),
                RecordLines.abilityKeys(record).stream().map(RuntimeLandTraitKeysTest::describe)
                        .toList(), record);
    }

    /**
     * Spreading Seas makes a Forest an Island: the Island mana ability is
     * granted and names no line of the Forest's.
     */
    @Test
    void theManaAbilityATypeChangeGivesABasicLandNamesNoPrintedLine() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card forest = board.battlefield("Forest", board.you);
        board.battlefield("Spreading Seas", board.you).attachToEntity(forest, null);
        board.game.getAction().checkStaticAbilities();

        Sidecar sidecar = sidecarOf(forest);
        List<Map<String, Object>> lines = lines(board, forest);
        assertJoins(sidecar, lines);
        assertEquals(List.of(true), granted(lines), "the Forest's lines under Spreading Seas");
        assertNamesNoDeclaredLine(sidecar, ProvenanceResolver.of(manaAbility(forest, "U")),
                "the Island's mana ability");
    }

    /**
     * Urborg makes a Forest a Swamp in addition: Forge lists the Swamp's
     * mana ability before the Forest's own, colour order, yet the Forest's
     * keeps its line and the Swamp's is granted.
     */
    @Test
    void aLandTypeAddedInAdditionLeavesThePrintedManaLineItsKey() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card forest = board.battlefield("Forest", board.you);
        board.battlefield("Urborg, Tomb of Yawgmoth", board.you);
        board.game.getAction().checkStaticAbilities();
        SpellAbility green = manaAbility(forest, "G");
        SpellAbility black = manaAbility(forest, "B");

        Sidecar sidecar = sidecarOf(forest);
        List<Map<String, Object>> lines = lines(board, forest);
        assertJoins(sidecar, lines);
        assertTrue(sidecar.names(ProvenanceResolver.of(green).describe()),
                "the Forest's own mana ability: " + ProvenanceResolver.of(green));
        assertNamesNoDeclaredLine(sidecar, ProvenanceResolver.of(black), "the Swamp's mana ability");

        String record = manaRecord(board, green, "G");
        assertEquals(List.of(ProvenanceResolver.of(green).describe()),
                RecordLines.abilityKeys(record).stream().map(RuntimeLandTraitKeysTest::describe)
                        .toList(), record);
    }

    /**
     * An {@code Animate} effect with no static behind it, Glorious Sunrise's
     * mode, gives Mishra's Factory "{T}: Add {G}{G}{G}": the line is granted
     * and names none of the Factory's printed lines, which keep theirs.
     */
    @Test
    void anAbilityAnAnimateEffectGivesALandNamesNoPrintedLine() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card factory = board.battlefield("Mishra's Factory", board.you);
        Card sunrise = board.battlefield("Glorious Sunrise", board.you);
        SpellAbility animate = AbilityFactory.getAbility(sunrise, "Animate");
        animate.getTargets().add(factory);
        board.resolve(animate, board.you);
        board.game.getAction().checkStaticAbilities();
        SpellAbility threeGreen = manaAbility(factory, "G");

        Sidecar sidecar = sidecarOf(factory);
        List<Map<String, Object>> lines = lines(board, factory);
        assertJoins(sidecar, lines);
        assertEquals(List.of(false, false, false, true), granted(lines),
                "the Factory's three printed lines, then the granted one");
        assertNamesNoDeclaredLine(sidecar, ProvenanceResolver.of(threeGreen),
                "the ability Glorious Sunrise gave");

        String record = manaRecord(board, threeGreen, "G");
        for (Map<String, Object> key : RecordLines.abilityKeys(record)) {
            assertFalse(sidecar.declares(describe(key)),
                    "the mana record acts through a printed line: " + record);
        }
    }

    /**
     * An {@code Animate} effect that sets a land's type, Dream Thrush's, makes
     * a Forest an Island: the Island mana ability is granted. Made Tundra an
     * Island, its Island mana ability keeps its printed line.
     */
    @Test
    void theManaAbilityAnAnimateEffectsTypeChangeGivesNamesNoPrintedLine() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card forest = board.battlefield("Forest", board.you);
        Card tundra = board.battlefield("Tundra", board.you);
        Card thrush = board.battlefield("Dream Thrush", board.you);
        thrush.setChosenType("Island");
        for (Card land : List.of(forest, tundra)) {
            SpellAbility animate = AbilityFactory.getAbility(thrush, "DBAnimate");
            animate.getTargets().add(land);
            board.resolve(animate, board.you);
        }
        board.game.getAction().checkStaticAbilities();

        Sidecar forestSidecar = sidecarOf(forest);
        List<Map<String, Object>> forestLines = lines(board, forest);
        assertJoins(forestSidecar, forestLines);
        assertEquals(List.of(true), granted(forestLines), "the Forest made an Island");
        assertNamesNoDeclaredLine(forestSidecar, ProvenanceResolver.of(manaAbility(forest, "U")),
                "the Island's mana ability on the Forest");

        Sidecar tundraSidecar = sidecarOf(tundra);
        List<Map<String, Object>> tundraLines = lines(board, tundra);
        assertJoins(tundraSidecar, tundraLines);
        assertEquals(List.of(false), granted(tundraLines), "Tundra made an Island");
        assertTrue(tundraSidecar.names(ProvenanceResolver.of(manaAbility(tundra, "U")).describe()),
                "Tundra's Island mana ability left its line");
    }

    // ── the join ────────────────────────────────────────────────────────

    /**
     * A card's sidecar as {@code python -m price_predictor convert} writes it,
     * reduced to the keys it names on a line and the keys it declares.
     */
    private record Sidecar(String scriptFile, List<String> named, List<ProvenanceKey> declared) {

        /** Whether a line of the sidecar carries this key. */
        boolean names(String key) {
            return named.contains(key);
        }

        /**
         * Whether this key falls inside a range the sidecar declares: its
         * script's, at an index no greater than the last one the sidecar
         * lists, on a line or dropped, for its face and trait kind.
         */
        boolean declares(String key) {
            for (ProvenanceKey each : declared) {
                if (key.startsWith("(" + scriptFile + ") face" + each.face() + "/"
                        + each.traitKind().wireName() + "[")
                        && indexOf(key) <= each.indexWithinKind()) {
                    return true;
                }
            }
            return false;
        }

        private static int indexOf(String key) {
            int open = key.indexOf('[');
            return Integer.parseInt(key.substring(open + 1, key.indexOf(']', open)));
        }
    }

    private Sidecar sidecarOf(Card card) throws IOException {
        String scriptFile = ScriptLocator.scriptFileOf(card);
        List<String> source = Files.readAllLines(Path.of(card.getRules().getPath()));
        ProvenanceSidecar sidecar = ConvertedScript.of(
                parser.parseScript(source, scriptFile), scriptFile).sidecar();
        List<String> named = new ArrayList<>();
        List<ProvenanceKey> declared = new ArrayList<>(sidecar.droppedKeys());
        for (ProvenanceSidecar.Line line : sidecar.lines()) {
            for (ProvenanceKey key : line.provenance()) {
                named.add(key.describe());
                declared.add(key);
            }
        }
        return new Sidecar(scriptFile, named, declared);
    }

    /**
     * Every face-up line joins: one not flagged granted names a sidecar line
     * by each of its keys, a granted one names no position the sidecar
     * declares.
     */
    private static void assertJoins(Sidecar sidecar, List<Map<String, Object>> lines) {
        for (Map<String, Object> line : lines) {
            boolean granted = Boolean.TRUE.equals(line.get("granted"));
            for (Object each : (List<?>) line.getOrDefault("keys", List.of())) {
                String key = describe(asMap(each));
                if (granted) {
                    assertFalse(sidecar.declares(key),
                            "a granted line names a printed position: " + line);
                } else {
                    assertTrue(sidecar.names(key),
                            "a printed line names no sidecar line: " + line + " in " + sidecar);
                }
            }
        }
    }

    private static void assertNamesNoDeclaredLine(Sidecar sidecar, ProvenanceKey key, String what) {
        assertTrue(key == null || !sidecar.declares(key.describe()),
                what + " keys inside the sidecar's declared range: " + key);
    }

    /** A JSON key as {@link ProvenanceKey#describe} spells it. */
    private static String describe(Map<String, Object> key) {
        String described = "(" + key.get("script_file") + ") face" + key.get("face") + "/"
                + key.get("trait_kind") + "[" + key.get("index_within_kind") + "]";
        return key.get("option") == null ? described : described + "/option" + key.get("option");
    }

    /** The lines of a card's snapshot entity; every land here has one face. */
    private static List<Map<String, Object>> lines(CharacterizationBoard board, Card card) {
        Map<String, Object> entity = parse(EntityJson.render(board.game, card, StaticSuppression.NONE));
        List<Map<String, Object>> lines = new ArrayList<>();
        for (Object line : (List<?>) entity.get("lines")) {
            lines.add(asMap(line));
        }
        return lines;
    }

    private static List<Boolean> granted(List<Map<String, Object>> lines) {
        return lines.stream().map(line -> Boolean.TRUE.equals(line.get("granted"))).toList();
    }

    /** The card's one mana ability that produces {@code colour}. */
    private static SpellAbility manaAbility(Card card, String colour) {
        List<SpellAbility> found = card.getManaAbilities().stream()
                .filter(sa -> sa.getManaPart().getOrigProduced().contains(colour))
                .toList();
        assertEquals(1, found.size(), card + "'s mana abilities making " + colour + ": "
                + card.getManaAbilities());
        return found.get(0);
    }

    /** The mana record activating {@code ability} writes. */
    private String manaRecord(CharacterizationBoard board, SpellAbility ability, String produced)
            throws Throwable {
        Player player = board.you;
        List<String> records = ShardRun.records(tempDir, board.game, (bracket, collectors) -> {
            var payment = collectors.mana().paymentHandler();
            payment.invoke(null, ON_PAYING_COSTS, new Object[]{ability, player, false});
            payment.invoke(null, ON_COSTS_PAID, new Object[]{ability, player, true});
            collectors.mana().handler().invoke(null, ON_MANA_PRODUCED,
                    new Object[]{ability, player, produced});
        });
        assertEquals(1, records.size(), records.toString());
        return records.get(0);
    }
}
