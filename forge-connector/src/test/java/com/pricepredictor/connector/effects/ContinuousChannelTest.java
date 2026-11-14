package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Which continuous records coalesce: per static and stable board, and per rendering. */
@ExtendWith(ForgeExtension.class)
class ContinuousChannelTest {

    @TempDir
    Path tempDir;

    /** A channel over no game, enough for what it decides before anything is written. */
    private static ContinuousChannel channel() {
        return new ContinuousChannel(
                new LiveGame(null), TestCollectors.context(null, "game-1"), new RecordDedup());
    }

    // ── continuous coalescing ───────────────────────────────────────────

    /**
     * A static's record is keyed by the static and the board: one repeating
     * both is refused, and a change in either is a record of its own.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource({
            "a static is recorded once per stable board, anthem, board-1, false",
            "a changed board gets a new record, anthem, board-2, true",
            "two statics on one board each get a record, crusade, board-1, true"})
    void aContinuousRecordIsKeyedByItsStaticAndBoard(
            String why, String staticKey, String board, boolean allowed) {
        ContinuousChannel collector = channel();
        assertTrue(collector.allowContinuousRecord("anthem", "board-1"));
        assertEquals(allowed, collector.allowContinuousRecord(staticKey, board), why);
    }

    /**
     * Two copies of one anthem render one record.
     *
     * <p>The per-board coalescer cannot see this: the two statics have different
     * ids, so it treats them as different questions, while their key, their
     * contributions and their suppressed board are all the same.
     */
    @Test
    void twoStaticsThatRenderIdenticallyAreWrittenOnce() throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Glorious Anthem", board.you);
        board.battlefield("Glorious Anthem", board.you);
        board.battlefield("Grizzly Bears", board.you);
        board.game.getAction().checkStaticAbilities();

        List<String> records = ShardRun.records(tempDir, board.game,
                (bracket, collectors) -> collectors.continuous().collect());

        assertEquals(1, records.size(), String.valueOf(records));
    }

    /**
     * A static an Animate grants is found at the first boundary after it is
     * granted, though an earlier boundary looked its card over without it
     * (C12): the static lookup caches what it found by id, and must not have
     * cached the card's want of it.
     */
    @Test
    void aStaticAnAnimateGrantsIsFoundAtTheNextBoundary() throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card host = board.battlefield("Ornithopter", board.you);
        board.battlefield("Grizzly Bears", board.you);
        host.setSVar("Anthem", "Mode$ Continuous | Affected$ Creature.YouCtrl+Other"
                + " | AddPower$ 1 | Description$ Other creatures you control get +1/+0.");
        SpellAbility animate = AbilityFactory.getAbility(
                "AB$ Animate | Cost$ 0 | Defined$ Self | staticAbilities$ Anthem"
                        + " | Duration$ Permanent", host);
        animate.setActivatingPlayer(board.you);
        board.game.getAction().checkStaticAbilities();

        List<String> records = ShardRun.records(tempDir, board.game, (bracket, collectors) -> {
            collectors.continuous().collect();
            animate.resolve();
            board.game.getAction().checkStaticAbilities();
            collectors.continuous().collect();
        });

        assertEquals(1, records.stream()
                .filter(r -> "continuous".equals(RecordLines.parse(r).get("kind"))).count(),
                String.valueOf(records));
    }

    /**
     * A static whose contribution changes on a board alike in ids, names
     * and P/T writes again (V3 #6): the coalescing key holds what the
     * static contributes, not the board hash alone.
     */
    @Test
    void aStaticContributingSomethingElseOnAnAlikeBoardIsWrittenAgain() throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card servant = board.battlefield("Painter's Servant", board.you);
        board.battlefield("Grizzly Bears", board.you);

        List<String> records = ShardRun.records(tempDir, board.game, (bracket, collectors) -> {
            for (String color : List.of("blue", "red")) {
                servant.setChosenColors(List.of(color));
                board.game.getAction().checkStaticAbilities();
                collectors.continuous().collect();
            }
        });

        assertEquals(2, records.stream()
                .filter(r -> "continuous".equals(RecordLines.parse(r).get("kind")))
                .filter(r -> r.contains("painters_servant")).count(),
                String.valueOf(records));
    }

    // ── what a contribution says ────────────────────────────────────────

    /** The continuous records {@code board}'s statics write now. */
    private List<String> continuousRecords(CharacterizationBoard board) throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        board.game.getAction().checkStaticAbilities();
        return ShardRun.records(tempDir, board.game,
                (bracket, collectors) -> collectors.continuous().collect()).stream()
                .filter(r -> "continuous".equals(RecordLines.parse(r).get("kind"))).toList();
    }

    /** The one record acting through {@code script}'s contribution for {@code entity}. */
    private static java.util.Map<String, Object> contributionOf(
            List<String> records, String script, Card entity) {
        String record = records.stream().filter(r -> script.equals(RecordLines.scriptOf(r)))
                .findFirst().orElseThrow(() -> new AssertionError(script + ": " + records));
        for (Object each : (List<?>) RecordLines.payload(record).get("contributions")) {
            java.util.Map<String, Object> contribution = RecordLines.asMap(each);
            if (Refs.of(entity).equals(contribution.get("entity"))) {
                return contribution;
            }
        }
        throw new AssertionError("no contribution for " + entity + ": " + record);
    }

    /**
     * A static granting only an ability (Cryptolith Rite) is a record, its
     * contribution naming the granted ability's line (V2 #15).
     */
    @Test
    void aStaticGrantingOnlyAnAbilityContributesIt() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Cryptolith Rite", board.you);
        Card bears = board.battlefield("Grizzly Bears", board.you);

        java.util.Map<String, Object> contribution = contributionOf(
                continuousRecords(board), "cardsfolder/c/cryptolith_rite.txt", bears);
        List<?> abilities = (List<?>) contribution.get("abilities");
        assertEquals(1, abilities.size(), contribution.toString());
        assertTrue(String.valueOf(abilities.get(0)).startsWith("(cardsfolder/c/cryptolith_rite.txt)"),
                contribution.toString());
    }

    /**
     * Humility's "loses all abilities" sets the keyword line outright and
     * takes Prodigal Sorcerer's ability away, named by its line (V2 #15);
     * the contribution used to say nothing but the base it sets.
     */
    @Test
    void losingAllAbilitiesIsSaid() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Humility", board.you);
        Card sorcerer = board.battlefield("Prodigal Sorcerer", board.opponent);

        java.util.Map<String, Object> contribution = contributionOf(
                continuousRecords(board), "cardsfolder/h/humility.txt", sorcerer);
        assertEquals(List.of("="), contribution.get("keywords"), contribution.toString());
        List<?> abilities = (List<?>) contribution.get("abilities");
        assertTrue(abilities.stream().anyMatch(a -> String.valueOf(a)
                        .startsWith("-(cardsfolder/p/prodigal_sorcerer.txt)")),
                contribution.toString());
    }

    /**
     * Errantry's AddHiddenKeyword$ is in its contribution, and its record's
     * board, which is the board without it, shows no hidden keyword on the
     * enchanted creature (V2 #14).
     */
    @Test
    void aStaticsHiddenKeywordIsItsContribution() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card errantry = board.battlefield("Errantry", board.you);
        errantry.attachToEntity(bears, null);

        List<String> records = continuousRecords(board);
        java.util.Map<String, Object> contribution = contributionOf(
                records, "cardsfolder/e/errantry.txt", bears);
        assertEquals(List.of("CARDNAME can only attack alone."),
                contribution.get("hidden_keywords"), contribution.toString());
        String record = records.stream()
                .filter(r -> "cardsfolder/e/errantry.txt".equals(RecordLines.scriptOf(r)))
                .findFirst().orElseThrow();
        assertTrue(RecordLines.stateOf(record).contains("\"hidden_keywords\":[]"), record);
        assertTrue(!RecordLines.stateOf(record).contains("can only attack alone"), record);
    }

    /**
     * Archetype of Imagination's board without it shows the opponent's
     * Serra Angel with flying: the can't-have row the static wrote is
     * left out with it (C2 #6).
     */
    @Test
    void theBoardWithoutACantHaveStaticGivesTheKeywordBack() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Archetype of Imagination", board.you);
        Card angel = board.battlefield("Serra Angel", board.opponent);
        board.game.getAction().checkStaticAbilities();
        assertTrue(!angel.hasKeyword(forge.game.keyword.Keyword.FLYING), "the angel lost flying");

        List<String> records = continuousRecords(board);
        String record = records.stream()
                .filter(r -> "cardsfolder/a/archetype_of_imagination.txt".equals(RecordLines.scriptOf(r)))
                .filter(r -> RecordLines.payload(r).toString().contains(Refs.of(angel)))
                .findFirst().orElseThrow(() -> new AssertionError(String.valueOf(records)));
        java.util.Map<String, Object> entity = ((List<?>) RecordLines.asMap(
                RecordLines.parse(record).get("state")).get("entities")).stream()
                .map(RecordLines::asMap).filter(e -> Refs.of(angel).equals(e.get("id")))
                .findFirst().orElseThrow();
        assertTrue(String.valueOf(RecordLines.asMap(entity.get("granted_temporary")).get("keywords"))
                .contains("flying"), entity.toString());
    }

    /**
     * A static setting a creature's name and mana cost: its continuous
     * record's board, the board without it, shows the creature's own name,
     * mana value and mana cost (follow-up item 9).
     */
    @Test
    void theBoardWithoutANameStaticShowsTheCardsOwnNameAndManaValue() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card host = board.battlefield("Ornithopter", board.you);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        host.setSVar("Rename", "Mode$ Continuous | Affected$ Creature.YouCtrl+Other"
                + " | SetName$ Pigeon | ManaCost$ 5 G | Description$ Others are Pigeons costing 5G.");
        SpellAbility animate = AbilityFactory.getAbility(
                "AB$ Animate | Cost$ 0 | Defined$ Self | staticAbilities$ Rename"
                        + " | Duration$ Permanent", host);
        animate.setActivatingPlayer(board.you);
        animate.resolve();
        board.game.getAction().checkStaticAbilities();
        assertEquals("Pigeon", bears.getName());
        assertEquals(6, bears.getCMC());

        String record = continuousRecords(board).stream()
                .filter(r -> RecordLines.payload(r).toString().contains("Pigeon"))
                .findFirst().orElseThrow(() -> new AssertionError("no record of the renaming static"));
        java.util.Map<String, Object> entity = ((List<?>) RecordLines.asMap(
                RecordLines.parse(record).get("state")).get("entities")).stream()
                .map(RecordLines::asMap).filter(e -> Refs.of(bears).equals(e.get("id")))
                .findFirst().orElseThrow();
        assertEquals("Grizzly Bears", entity.get("name"), entity.toString());
        assertEquals(2L, entity.get("mana_value"), entity.toString());
        assertEquals(java.util.Map.of("generic", 1L, "G", 1L), entity.get("mana_cost"), entity.toString());
    }

    /** A continuous record of the {@code index}-th static line of one anthem card. */
    private static EffectRecord anthem(int index, String contributions, String board) {
        return RecordDedupTest.ofKind(
                RecordKind.CONTINUOUS,
                new ProvenanceKey(
                        "cardsfolder/g/glorious_anthem.txt", 0, TraitKind.STATIC, index, null),
                contributions, board);
    }
}
