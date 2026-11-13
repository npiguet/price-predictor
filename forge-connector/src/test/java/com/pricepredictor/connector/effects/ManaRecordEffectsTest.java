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

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.events;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void aManaAbilityWithoutSubAbilityStillWritesTapAndMana() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card mountain = TestBoard.inPlay(game, you, "Mountain");
        SpellAbility red = TestBoard.ability(mountain, ApiType.Mana);
        red.setActivatingPlayer(you);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () ->
                assertTrue(ComputerUtil.playNoStack(you, red, game, false)));

        String record = manaRecordOf(records, "cardsfolder/m/mountain.txt");
        assertEquals(List.of("tapped", "mana_produced"), eventTypes(record), record);
    }
}
