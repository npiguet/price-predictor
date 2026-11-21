package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordEvents.events;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A mana record holds what its own mana ability did besides the mana (WP3's
 * verification, for WP7): a pain land's damage to its controller, after the
 * mana; a creature sacrificed for mana, before it. Played through the engine:
 * the AI pays the ability's cost and resolves it, with the hooks installed.
 */
@ExtendWith(ForgeExtension.class)
class ManaRecordEffectsTest {

    private static final String WHY = "for the payment and clause hooks";

    @TempDir
    Path tempDir;

    private static String manaRecordOf(List<String> records, String script) {
        List<String> mana = records.stream()
                .filter(record -> record.contains("\"kind\":\"resolution\""))
                .filter(record -> record.contains(script))
                .toList();
        assertEquals(1, mana.size(), records.toString());
        return mana.get(0);
    }

    @Test
    void aPainLandsDamageIsItsManaRecords() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card wastes = TestBoard.inPlay(game, you, "Adarkar Wastes");
        SpellAbility pain = wastes.getManaAbilities().stream()
                .filter(sa -> sa.getSubAbility() != null).findFirst().orElseThrow();
        pain.setActivatingPlayer(you);
        int life = you.getLife();

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () ->
                assertTrue(ComputerUtil.playNoStack(you, pain, game, false), "the AI tapped it"));

        assertEquals(life - 1, you.getLife(), "the pain land dealt its damage");
        String record = manaRecordOf(records, "cardsfolder/a/adarkar_wastes.txt");
        assertEquals(List.of("tapped", "mana_produced", "damage_dealt", "life_change"),
                eventTypes(record), record);
        assertEquals(List.of(Refs.of(you)), events(record).get(2).get("subjects"), record);
    }

    @Test
    void aCreatureSacrificedForManaIsItsManaRecords() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card pet = TestBoard.inPlay(game, you, "Blood Pet");
        SpellAbility sacrifice = pet.getManaAbilities().get(0);
        sacrifice.setActivatingPlayer(you);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () ->
                assertTrue(ComputerUtil.playNoStack(you, sacrifice, game, false), "the AI paid it"));

        assertTrue(pet.getZone() == null || pet.getZone().is(ZoneType.Graveyard)
                || game.getCardsIn(ZoneType.Graveyard).stream().anyMatch(c -> c.getName().equals("Blood Pet")),
                "the pet was sacrificed");
        String record = manaRecordOf(records, "cardsfolder/b/blood_pet.txt");
        List<String> types = eventTypes(record);
        assertTrue(types.contains("zone_change"), record);
        assertEquals("mana_produced", types.get(types.size() - 1), record);
        assertTrue(types.indexOf("zone_change") < types.indexOf("mana_produced"), record);
    }

    @Test
    void aManaAbilityWithoutSubAbilityStillWritesItsStampedTapAndMana() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card mountain = TestBoard.inPlay(game, you, "Mountain");

        String record = manaRecordOf(
                activated(game, you, TestBoard.ability(mountain, ApiType.Mana)),
                "cardsfolder/m/mountain.txt");

        assertEquals(List.of("tapped", "mana_produced"), eventTypes(record), record);
        assertStamped(events(record).get(0), "unresolved", record);
        assertStamped(events(record).get(1), "root", record);
    }

    private static List<String> manaRecordsOf(List<String> records, String script) {
        return records.stream()
                .filter(record -> record.contains("\"kind\":\"resolution\""))
                .filter(record -> record.contains(script))
                .toList();
    }

    private static void assertStamped(
            java.util.Map<String, Object> event, String attributedTo, String record) {
        assertEquals(attributedTo, event.get("attributed_to"), record);
        assertEquals("instant", event.get("duration"), record);
    }

    /** The mana records of {@code ability} activated once by {@code you}, through the engine. */
    private List<String> activated(Game game, Player you, SpellAbility ability) throws Throwable {
        ability.setActivatingPlayer(you);
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () ->
                assertTrue(ComputerUtil.playNoStack(you, ability, game, false)));
    }

    @Test
    void aManaAbilityComputingZeroWritesItsTapsAndNoManaProduced() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card cradle = TestBoard.inPlay(game, you, "Gaea's Cradle");

        String record = manaRecordOf(
                activated(game, you, cradle.getManaAbilities().get(0)),
                "cardsfolder/g/gaeas_cradle.txt");

        assertEquals(List.of("tapped"), eventTypes(record), record);
        assertStamped(events(record).get(0), "unresolved", record);
    }

    @Test
    void aManaAbilityThatProducedNothingEndsItsActivation() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card cradle = TestBoard.inPlay(game, you, "Gaea's Cradle");
        SpellAbility green = cradle.getManaAbilities().get(0);
        green.setActivatingPlayer(you);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            assertTrue(ComputerUtil.playNoStack(you, green, game, false), "no creature: zero");
            cradle.untap();
            TestBoard.inPlay(game, you, "Grizzly Bears");
            assertTrue(ComputerUtil.playNoStack(you, green, game, false), "one creature: one");
        });

        List<String> cradleRecords = manaRecordsOf(records, "cardsfolder/g/gaeas_cradle.txt");
        assertEquals(2, cradleRecords.size(), records.toString());
        assertEquals(List.of("tapped"), eventTypes(cradleRecords.get(0)), cradleRecords.toString());
        assertEquals(List.of("tapped", "mana_produced"), eventTypes(cradleRecords.get(1)),
                cradleRecords.toString());
        assertStamped(events(cradleRecords.get(1)).get(1), "root", cradleRecords.get(1));
    }

    /**
     * Jeweled Amulet with a charge counter and nothing noted: its mana string
     * is empty, so Forge's mana effect skips the mana hook. No real card has a
     * {@code ProduceMana} replacement that prevents the mana (every one in the
     * card scripts updates it with {@code ReplaceMana}), so that skip is not
     * played here; both skips end at the same clause signal.
     */
    @Test
    void aManaAbilityWhoseManaStringIsEmptyStillWritesItsRecordOnce() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card amulet = TestBoard.inPlay(game, you, "Jeweled Amulet");
        amulet.addCounterInternal(forge.game.card.CounterEnumType.CHARGE, 1, you, false, null, java.util.Map.of());
        SpellAbility mana = amulet.getManaAbilities().get(0);
        mana.setActivatingPlayer(you);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () ->
                assertTrue(ComputerUtil.playNoStack(you, mana, game, false), "the AI paid it"));

        String record = manaRecordOf(records, "cardsfolder/j/jeweled_amulet.txt");
        assertTrue(eventTypes(record).contains("tapped"), record);
        assertFalse(eventTypes(record).contains("mana_produced"), record);
    }

    /**
     * Foraging Wickermaw's mana is a sub-ability's below a ChooseColor root:
     * Forge says the root resolved before the sub-ability makes the mana, and
     * the activation is still the one record, with the mana in it.
     */
    @Test
    void aManaAbilityWhoseManaIsASubAbilitysWritesItsOwnRecord() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        TestBoard.inPlay(game, you, "Forest");
        Card wickermaw = TestBoard.inPlay(game, you, "Foraging Wickermaw");
        SpellAbility choose = wickermaw.getManaAbilities().get(0);

        List<String> own = activated(game, you, choose).stream()
                .filter(r -> r.contains("\"ability\":[{\"script_file\":\"cardsfolder/f/foraging_wickermaw.txt"))
                .toList();
        assertEquals(1, own.size(), "the Wickermaw's own record");
        String record = own.get(0);

        assertTrue(eventTypes(record).contains("choice_made"), record);
        assertTrue(eventTypes(record).contains("mana_produced"), record);
    }

    /**
     * Nykthos's green and black are two observations of one line: with a cap
     * of one per mana key, each is kept. Under one shared key the second
     * would be dropped.
     */
    @Test
    void aSubAbilitysManaKeysItsCapToTheManaProduced() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        List<Card> forests = List.of(
                TestBoard.inPlay(game, you, "Forest"), TestBoard.inPlay(game, you, "Forest"));
        Card nykthos = TestBoard.inPlay(game, you, "Nykthos, Shrine to Nyx");
        SpellAbility devotion = nykthos.getManaAbilities().stream()
                .filter(sa -> sa.getApi() == ApiType.ChooseColor).findFirst().orElseThrow();
        devotion.setActivatingPlayer(you);
        TestBoard.inPlay(game, you, "Grizzly Bears");

        CollectionCaps oneEach = TestCaps.caps().manaCap(1).interventions(0).probes(0).build();
        List<String> records = ShardRun.readShard(ShardRun.recordedWithClauseHook(
                tempDir, game, oneEach, WHY, () -> {
            assertTrue(ComputerUtil.playNoStack(you, devotion, game, false), "green");
            nykthos.untap();
            forests.forEach(Card::untap);
            TestBoard.inPlay(game, you, "Black Knight");
            TestBoard.inPlay(game, you, "Black Knight");
            assertTrue(ComputerUtil.playNoStack(you, devotion, game, false), "black");
        }).path());

        List<String> nykthosRecords = records.stream()
                .filter(r -> r.contains("\"ability\":[{\"script_file\":\"cardsfolder/n/nykthos_shrine_to_nyx.txt"))
                .toList();
        assertEquals(2, nykthosRecords.size(), records.toString());
    }
}
