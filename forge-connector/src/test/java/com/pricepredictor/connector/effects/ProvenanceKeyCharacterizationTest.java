package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

/**
 * A provenance key's two JSON forms, for the three shapes a key takes.
 *
 * <p>The record form names the script file and the sidecar form leaves it to
 * the sidecar's header; both write {@code option} only on a mode key. Each key
 * is resolved from a real card the way the collectors resolve one, rather
 * than constructed, so the script path is the one the resolver derives.
 */
@ExtendWith({ForgeExtension.class, Characterization.PinnedForgeIds.class})
class ProvenanceKeyCharacterizationTest {

    @Test
    void theRecordAndSidecarFormsOfEachKeyShape() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);

        ProvenanceKey printed = ProvenanceResolver.resolve(
                board.spell(board.hand("Lightning Bolt", board.you), board.you)).key();
        Card treasure = board.token("c_a_treasure_sac", board.you);
        ProvenanceKey token = ProvenanceResolver.resolve(
                board.ability(treasure, ApiType.Mana)).key();
        SpellAbility charm = board.ability(board.hand("Boros Charm", board.you), ApiType.Charm);
        charm.setActivatingPlayer(board.you);
        List<AbilitySub> chosen = new ArrayList<>(
                List.of(charm.getAdditionalAbilityList("Choices").get(2)));
        charm.setChosenList(chosen);
        CharmEffect.chainAbilities(charm, chosen);
        ProvenanceKey mode = ProvenanceResolver.resolve(charm.getSubAbility()).key();

        StringBuilder text = new StringBuilder();
        for (Labelled shape : List.of(
                new Labelled("cardsfolder", printed), new Labelled("tokenscripts", token),
                new Labelled("option", mode))) {
            text.append(shape.label()).append(" record: ")
                    .append(shape.key().toJson()).append('\n');
            text.append(shape.label()).append(" sidecar: ")
                    .append(shape.key().toSidecarJson()).append('\n');
        }
        text.append("array: ")
                .append(ProvenanceKey.toJsonArray(List.of(printed, token, mode))).append('\n');

        Characterization.assertMatches("provenance-keys.txt", text.toString());
    }

    private record Labelled(String label, ProvenanceKey key) {
    }
}
