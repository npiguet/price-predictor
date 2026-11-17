package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeEnvironmentInitializer;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The script surface of a trait that grants other traits: a static's {@code
 * AddTrigger$}, an Animate's {@code Triggers$}, an Effect's {@code
 * Triggers$} name SVars holding traits, and each such trait is a segment of
 * the granting trait's chain, its own sub-abilities with it. So are the
 * keyword lines whose text is a built trait's: a Class level, a Visit,
 * Backup, an opening-hand effect.
 */
@ExtendWith(ForgeExtension.class)
class GrantedTraitScriptTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private record Converted(ConvertedScript card, MissingSVarReport report) {
    }

    /** The line of a kind whose text holds a fragment. */
    private record LineQuery(String kind, String text) {
    }

    /** What a line's script is: its API type and fragments its text holds. */
    private record Chain(String api, List<String> fragments) {
    }

    private static Converted convert(String relativePath) {
        RulesParser parser = new RulesParser();
        ConvertedScript card = ConvertedScript.fromFile(parser, CARDS_FOLDER, relativePath);
        return new Converted(card, parser.missingSVars());
    }

    private static List<ProvenanceSidecar.Line> linesOfKind(Converted converted, String kind) {
        return converted.card().sidecar().lines().stream()
                .filter(line -> line.lineKind().equals(kind)).toList();
    }

    private static ProvenanceSidecar.Line line(Converted converted, LineQuery query) {
        List<String> texts = converted.card().texts();
        return linesOfKind(converted, query.kind()).stream()
                .filter(line -> texts.get(line.lineIndex()).contains(query.text()))
                .findFirst()
                .orElseThrow(() -> new AssertionError(query + " in " + texts));
    }

    private static void assertNothingMissing(Converted converted) {
        assertTrue(converted.report().drain().isEmpty(), "nothing is reported missing");
    }

    /** {@code text} holds every one of {@code fragments}. */
    private static void assertHolds(String text, List<String> fragments) {
        List<String> absent = fragments.stream().filter(f -> !text.contains(f)).toList();
        assertEquals(List.of(), absent, text);
    }

    /**
     * {@code line}'s script is a chain with the expected API type, not the
     * keyword's raw text, and holds every expected fragment.
     */
    private static void assertChain(ProvenanceSidecar.Line line, Chain expected) {
        String text = line.script().scriptText();
        assertEquals(expected.api(), line.script().apiType(), text);
        assertHolds(text, expected.fragments());
    }

    @Test
    void eachClassLevelLineCarriesItsLevelUpAndWhatItsStaticGrants() {
        Converted ranger = convert("r/ranger_class.txt");
        ProvenanceSidecar.Line second = line(ranger, new LineQuery("level", "{1}{G}: "));
        ProvenanceSidecar.Line third = line(ranger, new LineQuery("level", "{3}{G}: "));

        assertChain(second, new Chain("ClassLevelUp", List.of("AB$ ClassLevelUp",
                "AddTrigger$ SV", "Mode$ AttackersDeclared", "DB$ PutCounter")));
        assertChain(third, new Chain("ClassLevelUp", List.of("AddStaticAbility$ SV2 & SV3",
                "MayLookAt$ You", "MayPlay$ True")));
        assertNotEquals(second.script().scriptText(), third.script().scriptText());
        assertFalse(second.script().scriptText().contains("TrigPutCounter"), "no author label survives");
        assertNothingMissing(ranger);
    }

    /** Chapters printing one line alike give the first chapter's chain, not one per chapter. */
    @Test
    void aSagaLinePrintedByEveryChapterCarriesTheFirstChaptersChainAlone() {
        String text = linesOfKind(convert("d/day_of_the_moon.txt"), "chapter").get(0)
                .script().scriptText();

        assertTrue(text.startsWith("Chapter$ 1 | "), text);
        assertFalse(text.contains("Chapter$ 2"), text);
    }

    @Test
    void anAnthemGrantingATriggerCarriesThatTriggersEffect() {
        Converted retaliation = convert("r/retaliation.txt");
        String text = linesOfKind(retaliation, "static").get(0).script().scriptText();

        assertTrue(text.startsWith("AddSVar$ RetaliationTrigPump | AddTrigger$ SV1 | "), text);
        assertHolds(text, List.of(" [SEG] SV1: Execute$ SV2 | Mode$ AttackerBlockedByCreature",
                " [SEG] SV2: DB$ Pump"));
        assertNothingMissing(retaliation);
    }

    @Test
    void aManlandsAnimateCarriesTheTriggerItGrants() {
        Converted ravine = convert("r/raging_ravine.txt");
        ProvenanceSidecar.Line animate = line(ravine, new LineQuery("activated", "becomes a 3/3"));

        assertChain(animate, new Chain("Animate",
                List.of("Triggers$ SV1", "Mode$ Attacks", "DB$ PutCounter")));
        assertNothingMissing(ravine);
    }

    @Test
    void aVisitLineCarriesItsTriggersEffect() {
        Converted cars = convert("b/bumper_cars.txt");

        assertChain(linesOfKind(cars, "triggered").get(0), new Chain("Pump",
                List.of("Mode$ VisitAttraction", " [SEG] SV1: ", "DB$ Pump")));
    }

    @Test
    void aBackupLineCarriesItsCounterAndTheAbilitiesItGrants() {
        Converted archpriest = convert("a/archpriest_of_shadows.txt");

        assertChain(line(archpriest, new LineQuery("triggered", "backup 1")), new Chain("PutCounter",
                List.of("DB$ PutCounter", "DB$ Animate", "Mode$ DamageDone", "DB$ ChangeZone")));
        assertNothingMissing(archpriest);
    }

    @Test
    void anOpeningHandLineCarriesItsEffectsChain() {
        Converted chancellor = convert("c/chancellor_of_the_annex.txt");

        assertChain(line(chancellor, new LineQuery("triggered", "opening hand")), new Chain("Reveal",
                List.of("DB$ Reveal", "DB$ Effect", "Triggers$ SV", "Mode$ SpellCast", "DB$ Counter")));
        assertNothingMissing(chancellor);
    }

    @Test
    void aGrantedSVarTheScriptLacksIsReported() {
        RulesParser parser = new RulesParser();
        String text = onlyScript(parser, List.of(
                "Name:Test Anthem",
                "ManaCost:2 G",
                "Types:Enchantment",
                "S:Mode$ Continuous | Affected$ Creature.YouCtrl | AddTrigger$ TrigMissing "
                        + "| Description$ Creatures you control have a trigger.",
                "Oracle:Creatures you control have a trigger."));

        assertTrue(text.contains("AddTrigger$ SV1") && !text.contains("[SEG]"), text);
        assertEquals(1, parser.missingSVars().drain().size(), "the missing trigger is reported");
    }

    /** A granted static that grants itself is one segment, not a loop. */
    @Test
    void aGrantNamingItselfIsRenderedOnce() {
        RulesParser parser = new RulesParser();
        String text = onlyScript(parser, List.of(
                "Name:Test Loop",
                "ManaCost:2 G",
                "Types:Enchantment",
                "S:Mode$ Continuous | Affected$ Card.Self | AddStaticAbility$ SLoop "
                        + "| Description$ This has a static.",
                "SVar:SLoop:Mode$ Continuous | Affected$ Card.Self | AddStaticAbility$ SLoop "
                        + "| Description$ This has a static.",
                "Oracle:This has a static."));

        assertEquals(1, RecordLines.count(text, "[SEG]"), text);
        assertTrue(text.endsWith("[SEG] SV1: AddStaticAbility$ SV1 | Affected$ Card.Self "
                + "| Description$ This has a static. | Mode$ Continuous"), text);
        assertTrue(parser.missingSVars().drain().isEmpty(), "nothing is reported missing");
    }

    /** The script of the one line an inline test card prints. */
    private static String onlyScript(RulesParser parser, List<String> script) {
        String scriptFile = new ScriptFile(SourceTree.CARDSFOLDER, "t/test_card.txt").key();
        ConvertedScript card = ConvertedScript.of(parser.parseScript(script, scriptFile), scriptFile);
        assertEquals(1, card.sidecar().lines().size(), card.texts().toString());
        return card.sidecar().lines().get(0).script().scriptText();
    }
}
