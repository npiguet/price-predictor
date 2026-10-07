package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The chained script text (FR-001 to FR-003, FR-005a), against the scripts
 * captured for feature 024 before the converter changed.
 *
 * <p>Every card here is a real Forge script copied into
 * {@code src/test/resources/golden-024/cardsfolder}, apart from the four
 * synthetic ones the README there lists, so the shapes asserted on are the
 * shapes the collectors will meet. The converted text of each is checked byte
 * for byte against the golden by {@code BatchConverterTest}; this class reads
 * the sidecar side of the same conversion.
 */
@ExtendWith(ForgeExtension.class)
class TraitScriptChainTest {

    private static final Path GOLDEN_CARDS =
            Path.of("src/test/resources/golden-024/cardsfolder");

    private record Converted(List<String> lines, ProvenanceSidecar sidecar,
                             MissingSVarReport report) {
    }

    private static Converted convert(String relativePath) {
        RulesParser parser = new RulesParser();
        Path file = GOLDEN_CARDS.resolve(relativePath);
        List<String> source;
        try {
            source = Files.readAllLines(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        String scriptFile = SourceTree.CARDSFOLDER + "/" + relativePath;
        RulesParser.ParsedCard parsed =
                parser.parseScript(source, file.getFileName().toString(), scriptFile);
        List<Ability> owners = new ArrayList<>();
        List<Integer> faceOfLine = new ArrayList<>();
        List<String> lines = parsed.card().renderLines(owners, faceOfLine);
        ProvenanceSidecar sidecar = ProvenanceSidecar.build(
                parsed.card().faces().get(0).name(), scriptFile, parsed.card(),
                lines, owners, faceOfLine, parsed.recorders());
        return new Converted(lines, sidecar, parser.missingSVars());
    }

    private static List<ProvenanceSidecar.Line> linesOfKind(Converted converted, String kind) {
        return converted.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals(kind)).toList();
    }

    private static ProvenanceSidecar.Line onlyLineOfKind(Converted converted, String kind) {
        List<ProvenanceSidecar.Line> lines = linesOfKind(converted, kind);
        assertEquals(1, lines.size(), kind + " lines in " + converted.lines());
        return lines.get(0);
    }

    private static int count(String haystack, String needle) {
        int count = 0;
        for (int at = haystack.indexOf(needle); at >= 0; at = haystack.indexOf(needle, at + 1)) {
            count++;
        }
        return count;
    }

    // ── Story 1, scenario 1: a trigger's Execute and its SubAbility chain ──

    @Test
    void aTriggersChainFollowsExecuteThenEachSubAbility() {
        Converted vestige = convert("a/anticausal_vestige.txt");
        String text = onlyLineOfKind(vestige, "triggered").script().scriptText();

        assertTrue(text.startsWith("Execute$ SV1 | Mode$ ChangesZone | "), text);
        assertTrue(text.contains(" [SEG] SV1: DB$ Draw | SubAbility$ SV2 [SEG] SV2: "), text);
        assertTrue(text.contains("SV2: ") && text.contains("DB$ ChangeZone"), text);
        assertEquals(2, count(text, "[SEG]"), text);
        assertFalse(text.contains("TrigDraw") || text.contains("DBChangeZone"),
                "no author label survives: " + text);
    }

    @Test
    void theRootSegmentKeepsItsParameterKeysAndApiType() {
        Converted vestige = convert("a/anticausal_vestige.txt");
        TraitScript script = onlyLineOfKind(vestige, "triggered").script();

        assertEquals(List.of("Execute", "Mode", "Origin", "TriggerDescription", "ValidCard"),
                script.paramKeys(), "script_param_keys stay the root segment's keys");
        assertEquals("Draw", script.apiType());
        assertEquals(1, script.subAbilityLinks().size());
        assertEquals("DBChangeZone", script.subAbilityLinks().get(0).label(),
                "sub_ability_links keep the author's label: attributed_to reads it");
    }

    // ── Story 1, scenarios 2 and 11: a charm's root and option lines ──────

    @Test
    void aCharmRootIsItsOwnSegmentWithRenamedChoicesAndNoModeInlined() {
        Converted cryptic = convert("c/cryptic_command.txt");
        ProvenanceSidecar.Line root = onlyLineOfKind(cryptic, "spell");

        assertEquals("CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm",
                root.script().scriptText());
        assertEquals("Charm", root.script().apiType());
        assertEquals(List.of(new ProvenanceKey(
                        "cardsfolder/c/cryptic_command.txt", 0, ProvenanceKey.KIND_SPELL, 0)),
                root.provenance());
    }

    @Test
    void eachCharmModeLineCarriesItsKeyItsChainAndItsOwnApiType() {
        Converted cryptic = convert("c/cryptic_command.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(cryptic, "option");
        assertEquals(4, options.size());
        List<String> apis = List.of("Counter", "ChangeZone", "TapAll", "Draw");

        for (int i = 0; i < options.size(); i++) {
            ProvenanceSidecar.Line option = options.get(i);
            assertEquals(List.of(new ProvenanceKey("cardsfolder/c/cryptic_command.txt",
                            0, ProvenanceKey.KIND_SPELL, 0, i)),
                    option.provenance(), "mode " + i);
            assertEquals(apis.get(i), option.script().apiType(), "mode " + i);
            String text = option.script().scriptText();
            assertTrue(text.startsWith("SV1: "), "mode " + i + " opens with SV1: " + text);
            assertTrue(text.contains("DB$ " + apis.get(i)), text);
            assertFalse(text.contains("[SEG]"), "a mode with no sub-ability is one segment: " + text);
        }
    }

    @Test
    void aModeKeySpellsItsOptionMemberAfterTheIndex() {
        Converted cryptic = convert("c/cryptic_command.txt");
        ProvenanceKey third = linesOfKind(cryptic, "option").get(2).provenance().get(0);

        assertEquals("{\"face\":0,\"trait_kind\":\"spell\",\"index_within_kind\":0,\"option\":2}",
                third.toSidecarJson());
    }

    @Test
    void optionLinesFollowTheirRootInChoicesOrder() {
        Converted cryptic = convert("c/cryptic_command.txt");
        ProvenanceSidecar.Line root = onlyLineOfKind(cryptic, "spell");
        List<ProvenanceSidecar.Line> options = linesOfKind(cryptic, "option");

        for (int i = 0; i < options.size(); i++) {
            assertEquals(root.lineIndex() + 1 + i, options.get(i).lineIndex());
        }
    }

    // ── Story 1, scenario 3: a replacement's ReplaceWith and its chain ────

    @Test
    void aReplacementsChainFollowsReplaceWithThenItsSubAbility() {
        Converted leader = convert("l/leader_super_genius.txt");
        TraitScript script = onlyLineOfKind(leader, "replacement").script();
        String text = script.scriptText();

        assertTrue(text.contains("ReplaceWith$ SV1"), text);
        assertTrue(text.contains(" [SEG] SV1: DB$ Draw | SubAbility$ SV2 [SEG] SV2: DB$ Connive"),
                text);
        assertEquals(2, count(text, "[SEG]"), text);
        assertEquals("Draw", script.apiType(),
                "a replacement's API type is its replacing ability's, no longer null");
    }

    @Test
    void aReplacementWithoutASubAbilityHasOneReplacingSegment() {
        Converted swans = convert("s/swans_of_bryn_argoll.txt");
        String text = onlyLineOfKind(swans, "replacement").script().scriptText();

        assertTrue(text.contains("ReplaceWith$ SV1"), text);
        assertTrue(text.contains(" [SEG] SV1: DB$ Draw"), text);
        assertEquals(1, count(text, "[SEG]"), text);
    }

    // ── Story 1, scenario 12: label names never reach the text ───────────

    @Test
    void twoChainsDifferingOnlyInLabelNamesRenderTheSameText() {
        // The sub-ability's own SpellDescription renders a second spell line,
        // which feature 023 left unkeyed and textless: the trait's line is the
        // first, and it now carries the whole chain.
        List<ProvenanceSidecar.Line> a = linesOfKind(convert("l/label_a.txt"), "spell");
        List<ProvenanceSidecar.Line> b = linesOfKind(convert("l/label_b.txt"), "spell");
        assertEquals(2, a.size());
        assertTrue(a.get(1).provenance().isEmpty());
        assertNull(a.get(1).script().scriptText());

        String textA = a.get(0).script().scriptText();
        String textB = b.get(0).script().scriptText();
        assertEquals(textA, textB);
        assertTrue(textA.contains("| SubAbility$ SV1 |"), textA);
        assertTrue(textA.endsWith(" [SEG] SV1: DB$ Draw | Defined$ You | NumCards$ 1"
                + " | SpellDescription$ Draw a card."), textA);
        assertFalse(textA.contains("DBDraw"), textA);
    }

    // ── FR-003: once per chain, and undefined labels reported ────────────

    @Test
    void anSVarTwoBranchesShareIsEmittedOnceAndTheChainTerminates() {
        Converted loop = convert("r/repeat_loop.txt");
        String text = onlyLineOfKind(loop, "spell").script().scriptText();

        // Root references DBDamage (SV1) then DBCleanup (SV2); DBDamage's
        // chain reaches DBCount (SV3), whose chain reaches DBCleanup again.
        assertTrue(text.contains("RepeatSubAbility$ SV1"), text);
        assertTrue(text.contains("| SubAbility$ SV2 [SEG] SV1: DB$ DealDamage"), text);
        assertEquals(3, count(text, "[SEG]"), "three defined SVars, three segments: " + text);
        assertEquals(1, count(text, "SV2: "), "DBCleanup rendered once: " + text);
        assertEquals(2, count(text, "SubAbility$ SV2"), "but referenced twice: " + text);
        assertTrue(text.indexOf("SV2: ") < text.indexOf("SV3: "),
                "segments follow their reference order: " + text);
    }

    @Test
    void anUndefinedSVarIsNumberedReportedAndLeftOut() {
        Converted undefined = convert("u/undefined_svar.txt");
        ProvenanceSidecar.Line trigger = onlyLineOfKind(undefined, "triggered");
        String text = trigger.script().scriptText();

        assertTrue(text.contains(" [SEG] SV1: DB$ Draw | Defined$ You | NumCards$ 1 | SubAbility$ SV2"),
                text);
        assertFalse(text.contains("SV2: "), "the undefined segment is left out: " + text);
        assertEquals(1, count(text, "[SEG]"), text);

        List<MissingSVarReport.Entry> entries = undefined.report().entries();
        assertEquals(1, entries.size(), entries.toString());
        MissingSVarReport.Entry entry = entries.get(0);
        assertEquals("Undefined SVar", entry.card());
        assertEquals("SubAbility", entry.parameter());
        assertEquals("DBMissing", entry.label());
        assertEquals(trigger.provenance().get(0), entry.line());
        assertEquals("Undefined SVar (cardsfolder/u/undefined_svar.txt) face0/trigger[0]:"
                + " SubAbility$ DBMissing", entry.format());
    }

    @Test
    void aCardWithEveryLabelDefinedReportsNothing() {
        assertTrue(convert("c/cryptic_command.txt").report().isEmpty());
        assertTrue(convert("a/anticausal_vestige.txt").report().isEmpty());
    }

    // ── charms: shared sub-abilities, no description, triggers, chapters ──

    @Test
    void twoModesNamingOneSubAbilityEachCarryItInTheirOwnChain() {
        Converted lonely = convert("l/lonely_end.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(lonely, "option");
        assertEquals(2, options.size());

        for (ProvenanceSidecar.Line option : options) {
            String text = option.script().scriptText();
            assertTrue(text.contains("| SubAbility$ SV2"), text);
            assertTrue(text.contains(" [SEG] SV2: "), text);
            assertTrue(text.contains("DB$ GainLife"), text);
            assertEquals(1, count(text, "[SEG]"), text);
        }
    }

    @Test
    void aDescriptionlessCharmDropsItsRootKeyAndKeysItsModes() {
        Converted doomsday = convert("d/doomsday_confluence.txt");
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/d/doomsday_confluence.txt", 0, ProvenanceKey.KIND_SPELL, 0);

        assertTrue(linesOfKind(doomsday, "spell").isEmpty(), doomsday.lines().toString());
        assertTrue(doomsday.sidecar().droppedKeys().contains(root),
                doomsday.sidecar().droppedKeys().toString());
        List<ProvenanceSidecar.Line> options = linesOfKind(doomsday, "option");
        assertEquals(3, options.size());
        for (int i = 0; i < options.size(); i++) {
            assertEquals(List.of(root.withOption(i)), options.get(i).provenance());
            assertTrue(options.get(i).script().scriptText().startsWith("SV1: "),
                    options.get(i).script().scriptText());
        }
    }

    @Test
    void aDieRollOutcomeLineKeysToNothingAndHasNoChain() {
        Converted sorcerer = convert("a/aberrant_mind_sorcerer.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(sorcerer, "option");
        assertEquals(2, options.size(), sorcerer.lines().toString());

        for (ProvenanceSidecar.Line option : options) {
            assertTrue(option.provenance().isEmpty(), option.provenance().toString());
            assertNull(option.script().scriptText());
            assertNull(option.script().apiType());
        }
        // The trigger line still chains through to the die roll itself.
        String trigger = onlyLineOfKind(sorcerer, "triggered").script().scriptText();
        assertTrue(trigger.contains("DB$ RollDice"), trigger);
    }

    @Test
    void aTriggeredCharmChainsToTheCharmAndKeysItsModesToTheTrigger() {
        Converted grace = convert("a/abiding_grace.txt");
        ProvenanceSidecar.Line trigger = onlyLineOfKind(grace, "triggered");
        String text = trigger.script().scriptText();

        assertTrue(text.startsWith("Execute$ SV1 | Mode$ Phase"), text);
        assertTrue(text.endsWith(" [SEG] SV1: Choices$ SV2,SV3 | DB$ Charm"),
                "the charm names its modes and inlines none: " + text);
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/a/abiding_grace.txt", 0, ProvenanceKey.KIND_TRIGGER, 0);
        assertEquals(List.of(root), trigger.provenance());

        List<ProvenanceSidecar.Line> options = linesOfKind(grace, "option");
        assertEquals(2, options.size());
        assertEquals(List.of(root.withOption(0)), options.get(0).provenance());
        assertEquals(List.of(root.withOption(1)), options.get(1).provenance());
        assertEquals("GainLife", options.get(0).script().apiType());
        assertEquals("ChangeZone", options.get(1).script().apiType());
        assertTrue(options.get(1).script().scriptText().startsWith("SV1: "),
                options.get(1).script().scriptText());
    }

    @Test
    void aChapterCharmKeysItsModesToTheChapterTrigger() {
        Converted saga = convert("l/life_of_toshiro_umezawa_memory_of_toshiro.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(saga, "option");
        assertEquals(3, options.size(), saga.lines().toString());
        List<String> apis = List.of("Pump", "Pump", "GainLife");

        ProvenanceKey root = null;
        for (int i = 0; i < options.size(); i++) {
            List<ProvenanceKey> keys = options.get(i).provenance();
            assertEquals(1, keys.size(), "mode " + i + ": " + keys);
            assertEquals(ProvenanceKey.KIND_TRIGGER, keys.get(0).traitKind());
            assertEquals(Integer.valueOf(i), keys.get(0).option());
            if (root == null) root = keys.get(0).root();
            assertEquals(root, keys.get(0).root(), "every mode names the same chapter");
            assertEquals(apis.get(i), options.get(i).script().apiType());
        }
    }

    @Test
    void aPawprintCharmKeepsItsBudgetOnTheRootAndItsCostsOnTheModes() {
        Converted season = convert("s/season_of_gathering.txt");
        String root = onlyLineOfKind(season, "spell").script().scriptText();

        assertEquals("CanRepeatModes$ True | CharmNum$ 5 | Choices$ SV1,SV2,SV3"
                + " | MinCharmNum$ 0 | Pawprint$ 5 | SP$ Charm", root);
        List<ProvenanceSidecar.Line> options = linesOfKind(season, "option");
        assertEquals(3, options.size());
        assertTrue(options.get(0).script().scriptText().contains("Pawprint$ 1"),
                options.get(0).script().scriptText());
        assertTrue(options.get(2).script().scriptText().contains("Pawprint$ 3"),
                options.get(2).script().scriptText());
    }

    @Test
    void aSelectorChoicesParameterIsNotALabelList() {
        // Season of Gathering's first mode is a PutCounter whose Choices$ is
        // the set of creatures to choose among, not a list of abilities.
        Converted season = convert("s/season_of_gathering.txt");
        String mode = linesOfKind(season, "option").get(0).script().scriptText();

        assertTrue(mode.contains("Choices$ Creature.YouCtrl"), mode);
        assertTrue(season.report().isEmpty(), season.report().entries().toString());
    }

    @Test
    void aNonModalChoosersChoicesArePartOfItsChain() {
        // Season of Gathering's second mode is a GenericChoice: not a charm, so
        // its two choices are sub-abilities of the mode and follow it.
        Converted season = convert("s/season_of_gathering.txt");
        String mode = linesOfKind(season, "option").get(1).script().scriptText();

        assertTrue(mode.startsWith("SV1: "), mode);
        assertTrue(mode.contains("Choices$ SV2,SV3 | DB$ GenericChoice"), mode);
        assertTrue(mode.contains(" [SEG] SV2: DB$ DestroyAll"), mode);
        assertTrue(mode.contains(" [SEG] SV3: DB$ DestroyAll"), mode);
        assertEquals(2, count(mode, "[SEG]"), mode);
    }

    @Test
    void aCharmsOwnSubAbilityFollowsItsRootSegment() {
        // Blood on the Snow: a charm root with a SubAbility$ of its own, and
        // two modes that share one sub-ability.
        Converted blood = convert("b/blood_on_the_snow.txt");
        String root = onlyLineOfKind(blood, "spell").script().scriptText();

        assertTrue(root.startsWith("Choices$ SV1,SV2 | SP$ Charm | SubAbility$ SV3 [SEG] SV3: "),
                root);
        assertEquals(1, count(root, "[SEG]"), "the modes are not inlined: " + root);
        for (ProvenanceSidecar.Line option : linesOfKind(blood, "option")) {
            assertTrue(option.script().scriptText().contains(" [SEG] SV2: "),
                    option.script().scriptText());
        }
    }

    // ── lines with no chain are untouched ────────────────────────────────

    @Test
    void aKeywordLineStillCarriesItsOriginalTextOnly() {
        Converted spawn = convert("a/aboleth_spawn.txt");
        List<String> keywordTexts = spawn.sidecar().lines().stream()
                .filter(line -> "Keyword".equals(line.script().apiType()))
                .map(line -> line.script().scriptText()).toList();

        assertTrue(keywordTexts.contains("Ward:2"), keywordTexts.toString());
        assertTrue(keywordTexts.contains("Flash"), keywordTexts.toString());
    }

    @Test
    void aStaticAbilityIsOneSegmentWithItsModeAsApiType() {
        Converted anthem = convert("g/glorious_anthem.txt");
        TraitScript script = onlyLineOfKind(anthem, "static").script();

        assertNotNull(script.scriptText());
        assertFalse(script.scriptText().contains("[SEG]"), script.scriptText());
        assertEquals("Continuous", script.apiType());
    }

    @Test
    void aLineWithNoParametersHasNoText() {
        assertNull(TraitScript.of(null).scriptText());
        assertEquals(List.of(), TraitScript.of(null).paramKeys());
    }
}
