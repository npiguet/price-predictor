package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

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

    private record Converted(List<String> lines, ProvenanceSidecar sidecar,
                             MissingSVarReport report) {
    }

    private static Converted convert(String relativePath) {
        RulesParser parser = new RulesParser();
        ConvertedScript converted =
                ConvertedScript.fromFile(parser, ConvertedScript.GOLDEN_024_CARDS, relativePath);
        return new Converted(converted.texts(), converted.sidecar(), parser.missingSVars());
    }

    private static Converted convertInline(String relativePath, List<String> scriptLines) {
        RulesParser parser = new RulesParser();
        String scriptFile = new ScriptFile(SourceTree.CARDSFOLDER, relativePath).key();
        RulesParser.ParsedCard parsed =
                parser.parseScript(scriptLines, scriptFile);
        return converted(parsed, scriptFile, parser);
    }

    private static Converted converted(RulesParser.ParsedCard parsed, String scriptFile,
                                       RulesParser parser) {
        ConvertedScript converted = ConvertedScript.of(parsed, scriptFile);
        return new Converted(converted.texts(), converted.sidecar(), parser.missingSVars());
    }

    /** The {@code line_kind} values the sidecar's lines carry. */
    private enum LineKind {
        SPELL, TRIGGERED, REPLACEMENT, OPTION, STATIC;

        final String label = name().toLowerCase(java.util.Locale.ROOT);
    }

    private static List<ProvenanceSidecar.Line> linesOfKind(Converted converted, LineKind kind) {
        return converted.sidecar().lines().stream()
                .filter(line -> line.lineKind().equals(kind.label)).toList();
    }

    private static ProvenanceSidecar.Line onlyLineOfKind(Converted converted, LineKind kind) {
        List<ProvenanceSidecar.Line> lines = linesOfKind(converted, kind);
        assertEquals(1, lines.size(), kind.label + " lines in " + converted.lines());
        return lines.get(0);
    }

    /** What a fragment of a script text should count, and why that is the right count. */
    private record Occurs(String fragment, int times, String why) {
    }

    /** A script text under assertion; each check returns it, so a test chains them. */
    private record Chain(String text) {

        Chain startsWith(String prefix) {
            assertTrue(text.startsWith(prefix), text);
            return this;
        }

        Chain endsWith(String suffix) {
            assertTrue(text.endsWith(suffix), text);
            return this;
        }

        Chain hasAll(List<String> fragments) {
            for (String fragment : fragments) {
                assertTrue(text.contains(fragment), text);
            }
            return this;
        }

        Chain lacksAll(List<String> fragments) {
            for (String fragment : fragments) {
                assertFalse(text.contains(fragment), text);
            }
            return this;
        }

        Chain lacksLabels(List<String> labels) {
            for (String label : labels) {
                assertFalse(text.contains(label), "no author label survives: " + text);
            }
            return this;
        }

        Chain segments(int expected) {
            assertEquals(expected, RecordText.count(text, "[SEG]"), text);
            return this;
        }

        Chain counts(List<Occurs> expected) {
            for (Occurs occurs : expected) {
                assertEquals(occurs.times(), RecordText.count(text, occurs.fragment()),
                        occurs.why() + ": " + text);
            }
            return this;
        }
    }

    // ── Story 1, scenario 1: a trigger's Execute and its SubAbility chain ──

    @Test
    void aTriggersChainFollowsExecuteThenEachSubAbility() {
        Converted vestige = convert("a/anticausal_vestige.txt");
        String text = onlyLineOfKind(vestige, LineKind.TRIGGERED).script().scriptText();

        new Chain(text)
                .startsWith("Execute$ SV1 | Mode$ ChangesZone | ")
                .hasAll(List.of(" [SEG] SV1: DB$ Draw | SubAbility$ SV2 [SEG] SV2: ",
                        "SV2: ", "DB$ ChangeZone"))
                .segments(2)
                .lacksLabels(List.of("TrigDraw", "DBChangeZone"));
    }

    @Test
    void theRootSegmentKeepsItsParameterKeysAndApiType() {
        Converted vestige = convert("a/anticausal_vestige.txt");
        TraitScript script = onlyLineOfKind(vestige, LineKind.TRIGGERED).script();

        assertEquals(List.of("Execute", "Mode", "Origin", "TriggerDescription", "ValidCard"),
                script.paramKeys(), "script_param_keys stay the root segment's keys");
        assertEquals("Draw", script.apiType());
        assertSoleSubAbilityLabelIsTheAuthors(script);
    }

    private static void assertSoleSubAbilityLabelIsTheAuthors(TraitScript script) {
        assertEquals(1, script.subAbilityLinks().size());
        assertEquals("DBChangeZone", script.subAbilityLinks().get(0).label(),
                "sub_ability_links keep the author's label: attributed_to reads it");
    }

    // ── Story 1, scenarios 2 and 11: a charm's root and option lines ──────

    @Test
    void aCharmRootIsItsOwnSegmentWithRenamedChoicesAndNoModeInlined() {
        Converted cryptic = convert("c/cryptic_command.txt");
        ProvenanceSidecar.Line root = onlyLineOfKind(cryptic, LineKind.SPELL);

        assertEquals("CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm",
                root.script().scriptText());
        assertEquals("Charm", root.script().apiType());
        assertEquals(List.of(new ProvenanceKey(
                        "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0)),
                root.provenance());
    }

    @Test
    void eachCharmModeLineCarriesItsKeyItsChainAndItsOwnApiType() {
        Converted cryptic = convert("c/cryptic_command.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(cryptic, LineKind.OPTION);
        assertEquals(4, options.size());
        List<String> apis = List.of("Counter", "ChangeZone", "TapAll", "Draw");

        for (int i = 0; i < options.size(); i++) {
            ProvenanceSidecar.Line option = options.get(i);
            assertEquals(List.of(new ProvenanceKey("cardsfolder/c/cryptic_command.txt",
                            0, TraitKind.SPELL, 0, i)),
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
        ProvenanceKey third = linesOfKind(cryptic, LineKind.OPTION).get(2).provenance().get(0);

        assertEquals("{\"face\":0,\"trait_kind\":\"spell\",\"index_within_kind\":0,\"option\":2}",
                third.toSidecarJson());
    }

    @Test
    void optionLinesFollowTheirRootInChoicesOrder() {
        Converted cryptic = convert("c/cryptic_command.txt");
        ProvenanceSidecar.Line root = onlyLineOfKind(cryptic, LineKind.SPELL);
        List<ProvenanceSidecar.Line> options = linesOfKind(cryptic, LineKind.OPTION);

        for (int i = 0; i < options.size(); i++) {
            assertEquals(root.lineIndex() + 1 + i, options.get(i).lineIndex());
        }
    }

    // ── Story 1, scenario 3: a replacement's ReplaceWith and its chain ────

    @Test
    void aReplacementsChainFollowsReplaceWithThenItsSubAbility() {
        Converted leader = convert("l/leader_super_genius.txt");
        TraitScript script = onlyLineOfKind(leader, LineKind.REPLACEMENT).script();
        String text = script.scriptText();

        new Chain(text)
                .hasAll(List.of("ReplaceWith$ SV1",
                        " [SEG] SV1: DB$ Draw | SubAbility$ SV2 [SEG] SV2: DB$ Connive"))
                .segments(2);
        assertEquals("Draw", script.apiType(),
                "a replacement's API type is its replacing ability's, no longer null");
    }

    @Test
    void aReplacementWithoutASubAbilityHasOneReplacingSegment() {
        Converted swans = convert("s/swans_of_bryn_argoll.txt");
        String text = onlyLineOfKind(swans, LineKind.REPLACEMENT).script().scriptText();

        new Chain(text).hasAll(List.of("ReplaceWith$ SV1", " [SEG] SV1: DB$ Draw")).segments(1);
    }

    // ── Story 1, scenario 12: label names never reach the text ───────────

    @Test
    void twoChainsDifferingOnlyInLabelNamesRenderTheSameText() {
        // The sub-ability's own SpellDescription is the second sentence of
        // the spell's one line, which carries the whole chain.
        String textA = onlyLineOfKind(convert("l/label_a.txt"), LineKind.SPELL).script().scriptText();
        String textB = onlyLineOfKind(convert("l/label_b.txt"), LineKind.SPELL).script().scriptText();

        assertEquals(textA, textB);
        new Chain(textA)
                .hasAll(List.of("| SubAbility$ SV1 |"))
                .endsWith(" [SEG] SV1: DB$ Draw | Defined$ You | NumCards$ 1"
                        + " | SpellDescription$ Draw a card.")
                .lacksAll(List.of("DBDraw"));
    }

    // ── FR-003: once per chain, and undefined labels reported ────────────

    @Test
    void anSVarTwoBranchesShareIsEmittedOnceAndTheChainTerminates() {
        Converted loop = convert("r/repeat_loop.txt");
        String text = onlyLineOfKind(loop, LineKind.SPELL).script().scriptText();

        // Root references DBDamage (SV1) then DBCleanup (SV2); DBDamage's
        // chain reaches DBCount (SV3), whose chain reaches DBCleanup again.
        new Chain(text)
                .hasAll(List.of("RepeatSubAbility$ SV1",
                        "| SubAbility$ SV2 [SEG] SV1: DB$ DealDamage"))
                .counts(List.of(
                        new Occurs("[SEG]", 3, "three defined SVars, three segments"),
                        new Occurs("SV2: ", 1, "DBCleanup rendered once"),
                        new Occurs("SubAbility$ SV2", 2, "but referenced twice")));
        assertTrue(text.indexOf("SV2: ") < text.indexOf("SV3: "),
                "segments follow their reference order: " + text);
    }

    @Test
    void anUndefinedSVarIsNumberedReportedAndLeftOut() {
        Converted undefined = convert("u/undefined_svar.txt");
        ProvenanceSidecar.Line trigger = onlyLineOfKind(undefined, LineKind.TRIGGERED);
        String text = trigger.script().scriptText();

        new Chain(text)
                .hasAll(List.of(" [SEG] SV1: DB$ Draw | Defined$ You | NumCards$ 1 | SubAbility$ SV2"))
                .segments(1);
        assertFalse(text.contains("SV2: "), "the undefined segment is left out: " + text);

        List<MissingSVarReport.Entry> entries = undefined.report().drain();
        assertEquals(1, entries.size(), entries.toString());
        MissingSVarReport.Entry entry = entries.get(0);
        assertEntryNamesTheMissingLabel(entry);
        assertEquals(trigger.provenance().get(0), entry.line());
        assertEquals("Undefined SVar (cardsfolder/u/undefined_svar.txt) face0/trigger[0]:"
                + " SubAbility$ DBMissing", entry.format());
    }

    private static void assertEntryNamesTheMissingLabel(MissingSVarReport.Entry entry) {
        assertEquals("Undefined SVar", entry.card());
        assertEquals("SubAbility", entry.parameter());
        assertEquals("DBMissing", entry.label());
    }

    @Test
    void aCardWithEveryLabelDefinedReportsNothing() {
        assertTrue(convert("c/cryptic_command.txt").report().drain().isEmpty());
        assertTrue(convert("a/anticausal_vestige.txt").report().drain().isEmpty());
    }

    // ── charms: shared sub-abilities, no description, triggers, chapters ──

    @Test
    void twoModesNamingOneSubAbilityEachCarryItInTheirOwnChain() {
        Converted lonely = convert("l/lonely_end.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(lonely, LineKind.OPTION);
        assertEquals(2, options.size());

        for (ProvenanceSidecar.Line option : options) {
            String text = option.script().scriptText();
            new Chain(text)
                    .hasAll(List.of("| SubAbility$ SV2", " [SEG] SV2: ", "DB$ GainLife"))
                    .segments(1);
        }
    }

    @Test
    void aDescriptionlessCharmPrintsItsSynthesizedRootKeyedWithTheSpellAndKeysItsModes() {
        Converted doomsday = convert("d/doomsday_confluence.txt");
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/d/doomsday_confluence.txt", 0, TraitKind.SPELL, 0);

        List<ProvenanceSidecar.Line> roots = linesOfKind(doomsday, LineKind.SPELL);
        assertEquals(1, roots.size(), doomsday.lines().toString());
        assertEquals(List.of(root), roots.get(0).provenance());
        assertTrue(doomsday.sidecar().droppedKeys().isEmpty(),
                doomsday.sidecar().droppedKeys().toString());
        List<ProvenanceSidecar.Line> options = linesOfKind(doomsday, LineKind.OPTION);
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
        List<ProvenanceSidecar.Line> options = linesOfKind(sorcerer, LineKind.OPTION);
        assertEquals(2, options.size(), sorcerer.lines().toString());

        for (ProvenanceSidecar.Line option : options) {
            assertTrue(option.provenance().isEmpty(), option.provenance().toString());
            assertNull(option.script().scriptText());
            assertNull(option.script().apiType());
        }
        // The trigger line still chains through to the die roll itself.
        String trigger = onlyLineOfKind(sorcerer, LineKind.TRIGGERED).script().scriptText();
        assertTrue(trigger.contains("DB$ RollDice"), trigger);
    }

    @Test
    void aTriggeredCharmChainsToTheCharmAndKeysItsModesToTheTrigger() {
        Converted grace = convert("a/abiding_grace.txt");
        ProvenanceSidecar.Line trigger = onlyLineOfKind(grace, LineKind.TRIGGERED);
        String text = trigger.script().scriptText();

        // The charm names its modes and inlines none.
        new Chain(text)
                .startsWith("Execute$ SV1 | Mode$ Phase")
                .endsWith(" [SEG] SV1: Choices$ SV2,SV3 | DB$ Charm");
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/a/abiding_grace.txt", 0, TraitKind.TRIGGER, 0);
        assertEquals(List.of(root), trigger.provenance());

        List<ProvenanceSidecar.Line> options = linesOfKind(grace, LineKind.OPTION);
        assertEquals(2, options.size());
        assertOptionKeyAndApi(options.get(0), root.withOption(0), "GainLife");
        assertOptionKeyAndApi(options.get(1), root.withOption(1), "ChangeZone");
        new Chain(options.get(1).script().scriptText()).startsWith("SV1: ");
    }

    private static void assertOptionKeyAndApi(ProvenanceSidecar.Line option, ProvenanceKey key,
                                              String apiType) {
        assertEquals(List.of(key), option.provenance());
        assertEquals(apiType, option.script().apiType());
    }

    @Test
    void aChapterCharmKeysItsModesToTheChapterTrigger() {
        Converted saga = convert("l/life_of_toshiro_umezawa_memory_of_toshiro.txt");
        List<ProvenanceSidecar.Line> options = linesOfKind(saga, LineKind.OPTION);
        assertEquals(3, options.size(), saga.lines().toString());
        List<String> apis = List.of("Pump", "Pump", "GainLife");

        // Each mode is claimed under every key of the chapter line: chapter
        // I's and chapter II's, both built by the Chapter keyword, since the
        // mode resolves through whichever chapter fired.
        String file = "cardsfolder/l/life_of_toshiro_umezawa_memory_of_toshiro.txt";
        for (int i = 0; i < options.size(); i++) {
            List<ProvenanceKey> keys = options.get(i).provenance();
            for (ProvenanceKey key : keys) {
                assertEquals(Integer.valueOf(i), key.option(), "mode " + i + ": " + keys);
            }
            assertEquals(List.of(builtTrigger(file, 0).withOption(i), builtTrigger(file, 1).withOption(i)),
                    keys);
            assertEquals(apis.get(i), options.get(i).script().apiType());
        }
    }

    // ── a keyword rendering one line per trait it generates (WP12 N1) ───

    /** The keys of the line whose text starts with {@code prefix}. */
    private static List<ProvenanceKey> keysOfLine(Converted converted, String prefix) {
        List<ProvenanceSidecar.Line> lines = converted.sidecar().lines().stream()
                .filter(line -> converted.lines().get(line.lineIndex()).startsWith(prefix))
                .toList();
        assertEquals(1, lines.size(), prefix + " in " + converted.lines());
        return lines.get(0).provenance();
    }

    /** The {@code index}-th trigger the card's first keyword built, as its line declares it. */
    private static ProvenanceKey builtTrigger(String file, int index) {
        return new ProvenanceKey(file, 0, TraitKind.KEYWORD, 0).withBuilt(TraitKind.TRIGGER, index);
    }

    /**
     * The Chapter keyword generates one trigger per chapter and the converter
     * one line per distinct chapter ability: each line declares its own
     * chapters' {@code built} keys and no other, so a chapter II record reads
     * chapter II's text. A line printed for two chapters ("I, II —") declares
     * both, and its mode lines the modes of both and of no third.
     */
    @Test
    void eachChapterLineCarriesItsOwnChaptersTriggersAlone() {
        Converted saga = convert("l/life_of_toshiro_umezawa_memory_of_toshiro.txt");
        String file = "cardsfolder/l/life_of_toshiro_umezawa_memory_of_toshiro.txt";

        assertEquals(List.of(builtTrigger(file, 0), builtTrigger(file, 1)),
                keysOfLine(saga, "chapter: I, II "));
        assertEquals(List.of(builtTrigger(file, 2)), keysOfLine(saga, "chapter: III "));
        for (ProvenanceSidecar.Line option : linesOfKind(saga, LineKind.OPTION)) {
            assertFalse(option.provenance().stream().anyMatch(
                    key -> builtTrigger(file, 2).equals(key.root())),
                    "chapter III has no modes: " + option.provenance());
        }
    }

    /** Three chapters of three abilities: one trigger a line, in chapter order. */
    @Test
    void aSagaOfThreeAbilitiesKeysOneTriggerToEachChapterLine() {
        Converted konda = convertInline("t/the_fall_of_lord_konda_fragment_of_konda.txt",
                List.of(
                        "Name:The Fall of Lord Konda",
                        "ManaCost:2 W",
                        "Types:Enchantment Saga",
                        "K:Chapter:3:DBExile,DBRetake,DBTransform",
                        "SVar:DBExile:DB$ ChangeZone | ValidTgts$ Creature.OppCtrl+cmcGE4"
                                + " | Origin$ Battlefield | Destination$ Exile | SpellDescription$ Exile"
                                + " target creature an opponent controls with mana value 4 or greater.",
                        "SVar:DBRetake:DB$ GainControlVariant | AllValid$ Permanent"
                                + " | ChangeController$ CardOwner"
                                + " | SpellDescription$ Each player gains control of all permanents they own.",
                        "SVar:DBTransform:DB$ ChangeZone | Origin$ Battlefield | Destination$ Exile"
                                + " | SpellDescription$ Exile this Saga, then return it transformed.",
                        "Oracle:Exile."));
        String file = "cardsfolder/t/the_fall_of_lord_konda_fragment_of_konda.txt";

        assertEquals(List.of(builtTrigger(file, 0)), keysOfLine(konda, "chapter: I "));
        assertEquals(List.of(builtTrigger(file, 1)), keysOfLine(konda, "chapter: II "));
        assertEquals(List.of(builtTrigger(file, 2)), keysOfLine(konda, "chapter: III "));
    }

    /**
     * Haunt renders a creature's two lines from five generated triggers: the
     * enters-or-haunted-dies line carries the two that run its ability, the
     * haunt line the one that haunts and the two that end the haunting.
     */
    @Test
    void aHauntingCreaturesTwoLinesEachCarryTheirOwnTriggers() {
        Converted thrull = convertInline("a/absolver_thrull.txt", List.of(
                "Name:Absolver Thrull",
                "ManaCost:3 W",
                "Types:Creature Thrull Cleric",
                "PT:2/3",
                "K:Haunt:TrigDestroy",
                "SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Enchantment | SpellDescription$ Destroy target enchantment.",
                "Oracle:Haunt"));
        String file = "cardsfolder/a/absolver_thrull.txt";

        assertEquals(List.of(builtTrigger(file, 2), builtTrigger(file, 4)),
                keysOfLine(thrull, "triggered: when "));
        assertEquals(List.of(builtTrigger(file, 0), builtTrigger(file, 1), builtTrigger(file, 3)),
                keysOfLine(thrull, "triggered: haunt"));
    }

    /** A haunting sorcery: its spell line keeps the spell, the haunt lines the triggers. */
    @Test
    void aHauntingSpellsSpellLineCarriesTheSpellAlone() {
        Converted moons = convertInline("b/benediction_of_moons.txt", List.of(
                "Name:Benediction of Moons",
                "ManaCost:W",
                "Types:Sorcery",
                "K:Haunt:TrigGainLife",
                "SVar:TrigGainLife:DB$ GainLife | Defined$ You | LifeAmount$ X"
                        + " | SpellDescription$ You gain 1 life for each player.",
                "SVar:X:PlayerCountPlayers$Amount",
                "Oracle:You gain 1 life for each player."));
        String file = "cardsfolder/b/benediction_of_moons.txt";

        assertEquals(List.of(new ProvenanceKey(file, 0, TraitKind.KEYWORD, 0).withBuilt(TraitKind.SPELL, 0)),
                keysOfLine(moons, "spell"));
        assertEquals(List.of(builtTrigger(file, 0), builtTrigger(file, 1), builtTrigger(file, 2)),
                keysOfLine(moons, "triggered: haunt"));
        assertEquals(List.of(builtTrigger(file, 3)), keysOfLine(moons, "triggered: when "));
    }

    @Test
    void aPawprintCharmKeepsItsBudgetOnTheRootAndItsCostsOnTheModes() {
        Converted season = convert("s/season_of_gathering.txt");
        String root = onlyLineOfKind(season, LineKind.SPELL).script().scriptText();

        assertEquals("CanRepeatModes$ True | CharmNum$ 5 | Choices$ SV1,SV2,SV3"
                + " | MinCharmNum$ 0 | Pawprint$ 5 | SP$ Charm", root);
        List<ProvenanceSidecar.Line> options = linesOfKind(season, LineKind.OPTION);
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
        String mode = linesOfKind(season, LineKind.OPTION).get(0).script().scriptText();

        assertTrue(mode.contains("Choices$ Creature.YouCtrl"), mode);
        List<MissingSVarReport.Entry> left = season.report().drain();
        assertTrue(left.isEmpty(), left.toString());
    }

    @Test
    void aNonModalChoosersChoicesArePartOfItsChain() {
        // Season of Gathering's second mode is a GenericChoice: not a charm, so
        // its two choices are sub-abilities of the mode and follow it.
        Converted season = convert("s/season_of_gathering.txt");
        String mode = linesOfKind(season, LineKind.OPTION).get(1).script().scriptText();

        new Chain(mode)
                .startsWith("SV1: ")
                .hasAll(List.of("Choices$ SV2,SV3 | DB$ GenericChoice",
                        " [SEG] SV2: DB$ DestroyAll", " [SEG] SV3: DB$ DestroyAll"))
                .segments(2);
    }

    @Test
    void aCharmsOwnSubAbilityFollowsItsRootSegment() {
        // Blood on the Snow: a charm root with a SubAbility$ of its own, and
        // two modes that share one sub-ability.
        Converted blood = convert("b/blood_on_the_snow.txt");
        String root = onlyLineOfKind(blood, LineKind.SPELL).script().scriptText();

        assertTrue(root.startsWith("Choices$ SV1,SV2 | SP$ Charm | SubAbility$ SV3 [SEG] SV3: "),
                root);
        assertEquals(1, RecordText.count(root, "[SEG]"), "the modes are not inlined: " + root);
        for (ProvenanceSidecar.Line option : linesOfKind(blood, LineKind.OPTION)) {
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
        TraitScript script = onlyLineOfKind(anthem, LineKind.STATIC).script();

        assertNotNull(script.scriptText());
        assertFalse(script.scriptText().contains("[SEG]"), script.scriptText());
        assertEquals("Continuous", script.apiType());
    }

    @Test
    void aLineWithNoTraitHasNoTextAndNoParameters() {
        TraitScript script = TraitScript.of(null, null, null);

        assertNull(script.scriptText());
        assertEquals(List.of(), script.paramKeys());
        assertEquals(TraitScript.EMPTY, script);
    }

    // ── a charm line carrying several roots ─────────────────────────────

    /**
     * Two triggers sharing one charm render one line that carries both keys,
     * and Forge resolves a mode through whichever trigger fired: each mode
     * line has to carry the mode under both.
     */
    @Test
    void aModeLineCarriesTheModeUnderEveryTriggerItsCharmLineMerged() {
        Converted appa = convertInline("a/appa_loyal_sky_bison.txt", List.of(
                "Name:Appa, Loyal Sky Bison",
                "ManaCost:4 W W",
                "Types:Legendary Creature Bison Ally",
                "PT:4/4",
                "K:Flying",
                "T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigCharm | TriggerDescription$ Whenever NICKNAME enters or attacks, ABILITY",
                "T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigCharm | TriggerZones$ Battlefield | Secondary$ True | TriggerDescription$ Whenever NICKNAME enters or attacks, ABILITY",
                "SVar:TrigCharm:DB$ Charm | Choices$ DBPump,DBAirbend",
                "SVar:DBPump:DB$ Pump | ValidTgts$ Creature.YouCtrl | KW$ Flying | SpellDescription$ Target creature you control gains flying until end of turn.",
                "SVar:DBAirbend:DB$ Airbend | ValidTgts$ Permanent.Other+nonLand+YouCtrl | SpellDescription$ Airbend another target nonland permanent you control.",
                "Oracle:Flying"));
        List<ProvenanceSidecar.Line> options = linesOfKind(appa, LineKind.OPTION);
        assertEquals(2, options.size(), appa.lines().toString());
        for (int i = 0; i < options.size(); i++) {
            String file = "cardsfolder/a/appa_loyal_sky_bison.txt";
            List<ProvenanceKey> keys = options.get(i).provenance();
            assertTrue(keys.contains(new ProvenanceKey(file, 0, TraitKind.TRIGGER, 0, i)), keys.toString());
            assertTrue(keys.contains(new ProvenanceKey(file, 0, TraitKind.TRIGGER, 1, i)), keys.toString());
        }
    }

    // ── the other chain keys Forge parses (C2 #10, V2 #19) ──────────────

    /**
     * Every parameter Forge parses into an ability of the chain ({@code
     * AbilityFactory.additionalAbilityKeys}) is a link the renderer follows.
     */
    @Test
    void everyChainKeyForgeParsesIsFollowed() {
        java.util.Set<String> followed = TraitScript.chainKeys();
        java.util.List<String> missing = forge.game.ability.AbilityFactory.additionalAbilityKeys
                .stream().filter(key -> !followed.contains(key)).toList();
        assertEquals(List.of(), missing);
    }

    /** A coin flip's win and lose abilities are segments of its chain. */
    @Test
    void aCoinFlipsOutcomesAreSegmentsOfItsChain() {
        Converted coin = convertInline("t/test_coin.txt", List.of(
                "Name:Test Coin",
                "ManaCost:1 R",
                "Types:Instant",
                "A:SP$ FlipCoin | WinSubAbility$ DBDraw | LoseSubAbility$ DBDamage | SpellDescription$ Flip a coin.",
                "SVar:DBDraw:DB$ Draw | NumCards$ 1",
                "SVar:DBDamage:DB$ DealDamage | Defined$ You | NumDmg$ 1",
                "Oracle:Flip a coin."));
        String text = onlyLineOfKind(coin, LineKind.SPELL).script().scriptText();

        new Chain(text)
                .hasAll(List.of("LoseSubAbility$ SV", "WinSubAbility$ SV",
                        "DB$ Draw", "DB$ DealDamage"))
                .segments(2)
                .lacksLabels(List.of("DBDraw", "DBDamage"));
        assertTrue(coin.report().drain().isEmpty(), "nothing is reported missing");
    }
}
