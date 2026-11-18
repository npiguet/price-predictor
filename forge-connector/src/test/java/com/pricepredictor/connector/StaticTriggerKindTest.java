package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The line kind of a {@code Static$ True} trigger: a triggered mana ability
 * (rule 605.1b) reads {@code triggered:}, a trigger that changes how an event
 * happens ("as ... enters") reads {@code replacement:}.
 */
@ExtendWith(ForgeExtension.class)
class StaticTriggerKindTest {

    private static final Path CARDS_FOLDER = ForgeEnvironmentInitializer.findCardsFolder();

    private static String kindOfFirstLineWith(String file, String prose) {
        try {
            var lines = Files.readAllLines(CARDS_FOLDER.resolve(file));
            CardFace face = new RulesParser().parseScript(lines, null).card().faces().get(0);
            return face.abilities().stream()
                    .filter(a -> a.descriptionText().toLowerCase().contains(prose))
                    .map(a -> a.type().getOutputPrefix())
                    .findFirst().orElseThrow();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Test
    void aStaticManaTriggerIsATriggeredAbility() {
        assertEquals("triggered",
                kindOfFirstLineWith("w/wild_growth.txt", "tapped for mana"));
    }

    @Test
    void aStaticTriggerThatDoesNotOpenWithATriggerWordStaysAReplacement() {
        assertEquals("replacement",
                kindOfFirstLineWith("d/domesticated_mammoth.txt", "token copy of pacifism"));
    }
}
