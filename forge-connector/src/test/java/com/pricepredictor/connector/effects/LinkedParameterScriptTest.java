package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A trait's script follows every parameter that names an SVar holding an
 * ability or a trait Forge builds from it, beyond the sub-ability links:
 * a die roll's {@code ResultSubAbilities$}, an exert cost's {@code
 * Trigger$}, a mana ability's {@code TriggersWhenSpent$}, a prevention
 * shield's {@code PreventionSubAbility$}, an extra turn's or phase's delayed
 * trigger and its effect, a returned permanent's {@code StaticEffect$}. Each
 * named SVar is renamed and is a segment of the chain, with its own chain.
 */
@ExtendWith(ForgeExtension.class)
class LinkedParameterScriptTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private record Converted(ConvertedScript card, MissingSVarReport report) {
    }

    private static Converted convert(String relativePath) {
        RulesParser parser = new RulesParser();
        ConvertedScript card = ConvertedScript.fromFile(parser, CARDS_FOLDER, relativePath);
        return new Converted(card, parser.missingSVars());
    }

    /** The script text of the only line of {@code kind}. */
    private static String scriptOfOnly(Converted converted, String kind) {
        List<ProvenanceSidecar.Line> lines = converted.card().sidecar().lines().stream()
                .filter(line -> line.lineKind().equals(kind)).toList();
        assertEquals(1, lines.size(), kind + " lines in " + converted.card().texts());
        assertTrue(converted.report().drain().isEmpty(), "nothing is reported missing");
        return lines.get(0).script().scriptText();
    }

    /** A die roll's result abilities, each after its range, are segments of its chain. */
    @Test
    void aDieRollsResultAbilitiesAreSegmentsOfItsChain() {
        String text = scriptOfOnly(convert("c/contact_other_plane.txt"), "spell");

        assertEquals("ResultSubAbilities$ 1-9:SV1,10-19:SV2,20:SV3 | SP$ RollDice | Sides$ 20"
                + " | SpellDescription$ Roll a d20."
                + " [SEG] SV1: DB$ Draw | NumCards$ 2"
                + " | SpellDescription$ 1—9 VERT Draw two cards."
                + " [SEG] SV2: DB$ Scry | ScryNum$ 2"
                + " | SpellDescription$ 10—19 VERT Scry 2, then draw two cards. | SubAbility$ SV1"
                + " [SEG] SV3: DB$ Scry | ScryNum$ 3"
                + " | SpellDescription$ 20 VERT Scry 3, then draw three cards. | SubAbility$ SV4"
                + " [SEG] SV4: DB$ Draw | NumCards$ 3", text);
    }

    /** An exert cost's reflexive trigger ability is a segment of the static's chain. */
    @Test
    void anExertCostsTriggerIsASegmentOfItsStaticsChain() {
        String text = scriptOfOnly(convert("a/ahn_crop_champion.txt"), "static");

        assertEquals("Cost$ Exert<1/CARDNAME>"
                + " | Description$ You may exert CARDNAME as it attacks."
                + " When you do, untap all other creatures you control."
                + " | Mode$ OptionalAttackCost | Trigger$ SV1 | ValidCard$ Card.Self"
                + " [SEG] SV1: DB$ UntapAll"
                + " | SpellDescription$ When you do, untap all other creatures you control."
                + " | ValidCards$ Creature.YouCtrl+StrictlyOther", text);
    }

    /** The trigger a mana ability's mana sets off when spent is a segment, its effect too. */
    @Test
    void aManaAbilitysSpentTriggerIsASegmentOfItsChain() {
        String text = scriptOfOnly(convert("a/andrios_roaming_explorer.txt"), "activated");

        assertEquals("AB$ Mana | Cost$ T | Produced$ W U B R G"
                + " | SpellDescription$ Add {W}{U}{B}{R}{G}. Creature spells you spend this mana"
                + " to cast have their base power and toughness become 4/3."
                + " | TriggersWhenSpent$ SV1"
                + " [SEG] SV1: Execute$ SV2 | Mode$ SpellCast"
                + " | TriggerDescription$ Creature spells you spend this mana to cast have their"
                + " base power and toughness become 4/3."
                + " | ValidActivatingPlayer$ You | ValidCard$ Creature"
                + " [SEG] SV2: DB$ Animate | Defined$ TriggeredCard | Duration$ Permanent"
                + " | Power$ 4 | Toughness$ 3", text);
    }

    /** One card's only line of a kind, and what its script holds. */
    private record Case(String file, String kind, List<String> fragments) {
    }

    /**
     * The other naming parameters: each SVar is renamed, its trait or
     * ability is a segment, and no author label survives.
     */
    @Test
    void everyOtherNamingParameterIsFollowed() {
        List<Case> cases = List.of(
                new Case("a/acolytes_reward.txt", "spell", List.of(
                        "PreventionSubAbility$ SV2", "[SEG] SV2: DB$ DealDamage")),
                new Case("f/final_fortune.txt", "spell", List.of(
                        "ExtraTurnDelayedTrigger$ SV1 | ExtraTurnDelayedTriggerExecute$ SV2",
                        "[SEG] SV1: Mode$ Phase", "[SEG] SV2: DB$ LosesGame")),
                new Case("m/moraug_fury_of_akoum.txt", "triggered", List.of(
                        "ExtraPhaseDelayedTrigger$ SV2 | ExtraPhaseDelayedTriggerExcute$ SV3",
                        "[SEG] SV2: Mode$ Phase", "[SEG] SV3: DB$ UntapAll")),
                new Case("a/abuelos_awakening.txt", "spell", List.of(
                        "StaticEffect$ SV1", "[SEG] SV1: AddKeyword$ Flying")),
                new Case("v/volraths_shapeshifter.txt", "static", List.of(
                        "GainTextAbilities$ SV1", "[SEG] SV1: AB$ Discard")));
        for (Case c : cases) {
            String text = scriptOfOnly(convert(c.file()), c.kind());
            List<String> absent = c.fragments().stream().filter(f -> !text.contains(f)).toList();
            assertEquals(List.of(), absent, c.file() + ": " + text);
        }
    }
}
