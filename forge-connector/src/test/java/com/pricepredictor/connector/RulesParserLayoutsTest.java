package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.FaceAbilities.abilitiesOfType;
import static com.pricepredictor.connector.FaceAbilities.rendered;
import static org.junit.jupiter.api.Assertions.*;

/** The parser's layouts and faces: multi-face cards, copied and melded faces, land mana lines. */
@ExtendWith(ForgeExtension.class)
class RulesParserLayoutsTest {

    private final ConvertedCards cards = new ConvertedCards();

    @Test
    void battleCardWithDefense() {
        MultiCard result = cards.convertFromFile("i/invasion_of_kamigawa_rooftop_saboteurs.txt");
        assertEquals("transform", result.layout());
        assertEquals(2, result.faces().size());

        CardFace front = result.faces().get(0);
        assertEquals("invasion of kamigawa", front.name());
        assertEquals("4", front.defense());
        assertNull(front.powerToughness());
        assertTrue(rendered(front).contains("defense: 4"));

        CardFace back = result.faces().get(1);
        assertEquals("rooftop saboteurs", back.name());
        assertNull(back.defense());
        assertEquals("2/3", back.powerToughness());
    }

    @Test
    void layoutDetection() {
        assertEquals("transform", cards.convert("Name:A", "ManaCost:1", "Types:Creature Human", "PT:1/1",
                "AlternateMode:DoubleFaced", "Oracle:", "ALTERNATE",
                "Name:B", "Types:Creature Werewolf", "PT:2/2", "Oracle:").layout());
        assertEquals("split", cards.convert("Name:Fire", "ManaCost:1 R", "Types:Instant",
                "AlternateMode:Split", "Oracle:", "ALTERNATE",
                "Name:Ice", "ManaCost:1 U", "Types:Instant", "Oracle:").layout());
        assertEquals("adventure", cards.convert("Name:Bonecrusher Giant", "ManaCost:2 R", "Types:Creature Giant",
                "PT:4/3", "AlternateMode:Adventure", "Oracle:", "ALTERNATE",
                "Name:Stomp", "ManaCost:1 R", "Types:Instant Adventure", "Oracle:").layout());
        assertEquals("meld", cards.convert("Name:The Mightstone and Weakstone", "ManaCost:5",
                "Types:Legendary Artifact Powerstone",
                "MeldPair:Urza, Lord Protector", "AlternateMode:Meld", "Oracle:").layout());
    }

    /**
     * {@code CopyFaceFrom:} names a face another script defines rather than
     * writing it out. Forge's card database fills the face in once every script
     * is read, so a parser that reads the one script alone holds no face there:
     * with both halves copied it has no front face at all.
     */
    @Test
    void aSplitCardWhoseHalvesAreBothCopiedConvertsBothHalves() {
        MultiCard card = cards.convertFromFile("b/bind_liberate.txt");

        assertEquals("split", card.layout());
        assertEquals(List.of("bind", "liberate"),
                card.faces().stream().map(CardFace::name).toList());
    }

    /** A copied second face is part of the card, not left out of its text. */
    @Test
    void aCopiedSecondFaceIsConverted() {
        MultiCard card = cards.convertFromFile("e/emeritus_of_conflict_lightning_bolt.txt");

        assertEquals(List.of("emeritus of conflict", "lightning bolt"),
                card.faces().stream().map(CardFace::name).toList());
        assertTrue(rendered(card).contains("deals 3 damage to any target"), rendered(card));
    }

    /** A face no script defines fails naming it, not with a bare null. */
    @Test
    void aCopiedFaceNoScriptDefinesIsNamedInTheFailure() {
        RuntimeException e = assertThrows(RuntimeException.class, () -> cards.convert(
                "CopyFaceFrom:No Such Face Was Ever Printed", "AlternateMode:Split",
                "ALTERNATE", "CopyFaceFrom:Fire"));

        assertTrue(e.getMessage().contains("No Such Face Was Ever Printed"), e.getMessage());
    }

    /**
     * A meld half whose script does not hold the melded face takes it from its
     * meld partner, as Forge's {@code CardFactory} does: The Mightstone and
     * Weakstone's face 1 is Urza, Planeswalker, line for line as Urza, Lord
     * Protector's script prints it.
     */
    @Test
    void theMightstoneAndWeakstoneHoldsUrzaPlaneswalkerOnFaceOne() {
        MultiCard mightstone = cards.convertFromFile("t/the_mightstone_and_weakstone.txt");
        MultiCard urza = cards.convertFromFile("u/urza_lord_protector_urza_planeswalker.txt");

        assertEquals(List.of("the mightstone and weakstone", "urza, planeswalker"),
                mightstone.faces().stream().map(CardFace::name).toList());
        assertEquals(rendered(urza.faces().get(1)), rendered(mightstone.faces().get(1)));
    }

    /** Every meld half whose script does not hold the melded face converts with it as face 1. */
    @ParameterizedTest
    @CsvSource(delimiter = '|', quoteCharacter = '"', value = {
            "a/argoth_sanctum_of_nature.txt     | argoth, sanctum of nature    | titania, gaea incarnate",
            "b/bruna_the_fading_light.txt       | bruna, the fading light      | brisela, voice of nightmares",
            "f/fang_fearless_lcie.txt           | fang, fearless l'cie         | ragnarok, divine deliverance",
            "h/hanweir_garrison.txt             | hanweir garrison             | hanweir, the writhing township",
            "m/midnight_scavengers.txt          | midnight scavengers          | chittering host",
            "p/phyrexian_dragon_engine.txt      | phyrexian dragon engine      | mishra, lost to phyrexia",
            "t/the_mightstone_and_weakstone.txt | the mightstone and weakstone | urza, planeswalker",
    })
    void aMeldHalfBorrowsTheMeldedFaceFromItsPartner(String file, String half, String melded) {
        MultiCard result = cards.convertFromFile(file);
        assertEquals("meld", result.layout());
        assertEquals(List.of(half, melded), result.faces().stream().map(CardFace::name).toList());
    }

    @Test
    void basicLandHasImplicitManaAbility() {
        CardFace card = cards.face("f/forest.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertEquals("{T}: add {G}.", activated.get(0).descriptionText());
    }

    @Test
    void dualLandHasCombinedManaAbility() {
        CardFace card = cards.face("b/bayou.txt");
        var activated = abilitiesOfType(card, AbilityType.ACTIVATED);
        assertEquals(1, activated.size());
        assertEquals("{T}: add {B} or {G}.", activated.get(0).descriptionText());
    }

    @Test
    void aTriomesManaLineReadsLikeAScriptedTriLand() {
        // Ketria Triome's line is synthesized from its basic types; Arcane
        // Sanctum's comes from its script. The two print the same sentence.
        var triome = abilitiesOfType(cards.face("k/ketria_triome.txt"), AbilityType.ACTIVATED);
        assertTrue(triome.stream().anyMatch(a -> a.descriptionText().equals("{T}: add {U}, {R}, or {G}.")),
                triome.toString());
        var sanctum = abilitiesOfType(cards.face("a/arcane_sanctum.txt"), AbilityType.ACTIVATED);
        assertEquals("{T}: add {W}, {U}, or {B}.", sanctum.get(0).descriptionText());
    }

    @Test
    void joinWithOrUsesTheSerialCommaFromThreeItems() {
        assertEquals("a", FaceParser.joinWithOr(List.of("a")));
        assertEquals("a or b", FaceParser.joinWithOr(List.of("a", "b")));
        assertEquals("a, b, or c", FaceParser.joinWithOr(List.of("a", "b", "c")));
        assertEquals("a, b, c, or d", FaceParser.joinWithOr(List.of("a", "b", "c", "d")));
    }

    @ParameterizedTest
    @CsvSource({
            "b/bloom_tender.txt,    bloom tender",
            "f/faeburrow_elder.txt, faeburrow elder",
            "t/tarnation_vista.txt, tarnation vista",
    })
    void cardConvertsWithoutError(String file, String expectedName) {
        assertEquals(expectedName, cards.face(file).name());
    }

    @Test
    void adventureMainFaceExcludesAdventureSa() {
        // Bonecrusher Giant (main face) + Stomp (adventure face).
        // The main face should only contain Bonecrusher Giant's triggered ability,
        // NOT Stomp's spell ability ("damage can't be prevented this turn").
        MultiCard result = cards.convertFromFile("b/bonecrusher_giant_stomp.txt");
        assertEquals("adventure", result.layout());
        CardFace main = result.faces().get(0);
        assertEquals("bonecrusher giant", main.name());
        // Main face should have the triggered ability
        assertFalse(abilitiesOfType(main, AbilityType.TRIGGERED).isEmpty(),
                "Main face should have its triggered ability: " + main.abilities());
        // Main face must NOT contain Stomp's spell
        assertTrue(abilitiesOfType(main, AbilityType.SPELL).isEmpty(),
                "Main face must not include adventure spell: " + main.abilities());
        assertFalse(main.abilities().stream()
                        .anyMatch(a -> a.descriptionText().contains("damage can't be prevented")),
                "Stomp's effect must not leak onto main face: " + main.abilities());
        // Adventure face should have the spell
        CardFace adventure = result.faces().get(1);
        assertEquals("stomp", adventure.name());
        assertFalse(abilitiesOfType(adventure, AbilityType.SPELL).isEmpty(),
                "Adventure face should have its spell: " + adventure.abilities());
    }

    @Test
    void specializeFacesFollowTheStateEnumOrderTheCollectorsUse() {
        // CardRules holds a specialize card's parts in a HashMap, whose
        // iteration order varies between JVM runs, while the collectors
        // number faces in the EnumMap order of the live card's states
        // (FaceOrder.faceOrder). The two must agree, or a record's face index
        // resolves to another face's lines.
        MultiCard card = cards.convertFromFile("a/alora_rogue_companion.txt");
        assertEquals(
                List.of("alora, rogue companion", "alora, cheerful mastermind",
                        "alora, cheerful thief", "alora, cheerful assassin",
                        "alora, cheerful swashbuckler", "alora, cheerful scout"),
                card.faces().stream().map(CardFace::name).toList(),
                "faces must follow Original, then SpecializeW, U, B, R, G");
    }
}
