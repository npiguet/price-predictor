package com.pricepredictor.connector;

import com.pricepredictor.connector.json.Json;
import forge.card.CardRules;
import forge.game.card.Card;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The coverage collector's castability consult.
 *
 * <p>Answers one question per card: could Forge ever put this on the stack at
 * all? A card with no castable spell ability — a land, unless an Adventure
 * half or a face-down keyword such as Morph gives it one — will never appear
 * in a resolution record however many games it is dealt into, so the coverage
 * collector ranks it lower and stops waiting for it sooner.
 *
 * <p>The verdict <b>only ranks</b>. It never drops a card from deck building,
 * because being in a game is the precondition a stage-three intervention forks
 * from — a card this consult calls uncastable is exactly the one an
 * intervention has to force into play.
 *
 * <p>Needs a live Forge: castability is a question about the engine's own
 * legality rules, not about the card's text.
 */
public class CastabilityMain {

    /** A card's answer; {@link #wireName()} is what the Python side reads. */
    enum Verdict implements WireNamed {
        CASTABLE("castable"),
        UNCASTABLE("uncastable");

        private final String wireName;

        Verdict(String wireName) {
            this.wireName = wireName;
        }

        @Override
        public String wireName() {
            return wireName;
        }
    }

    public static void main(String[] args) {
        ToolMain.Args cli = ToolMain.args(args, "--cards", "--output");

        ToolMain.run(() -> {
            Path cardsFile = Path.of(cli.require("--cards"));
            Path output = Path.of(cli.require("--output"));
            ForgeEnvironmentInitializer.initialize();
            List<String> names = CorpusFiles.nonBlankLines(cardsFile);
            Map<String, Verdict> verdicts = consult(names);
            if (output.getParent() != null) {
                Files.createDirectories(output.getParent());
            }
            Files.writeString(output, render(verdicts), StandardCharsets.UTF_8);
            long castable = verdicts.values().stream()
                    .filter(Verdict.CASTABLE::equals).count();
            System.out.println(
                    "Consulted " + verdicts.size() + " cards: " + castable
                            + " castable, " + (verdicts.size() - castable)
                            + " uncastable. Output: " + output);
        });
    }

    /** One verdict per name Forge recognizes; names it does not are omitted. */
    static Map<String, Verdict> consult(List<String> names) {
        Map<String, Verdict> verdicts = new LinkedHashMap<>();
        for (String name : names) {
            verdictFor(name).ifPresent(verdict -> verdicts.put(name, verdict));
        }
        return verdicts;
    }

    /**
     * Whether a card has any spell ability at all.
     *
     * <p>Deliberately coarse: the consult is a ranking input, so a false
     * "castable" costs a few wasted decks and a false "uncastable" costs a
     * slower run. Neither is a correctness failure, which is why this asks the
     * cheap question rather than simulating a cast.
     *
     * <p>The card is built against the {@link OfflineGame} {@code convert}
     * builds every card against: with no game at all, three cards' ability
     * setup throws and would be misread as uncastable.
     */
    private static Optional<Verdict> verdictFor(String name) {
        // allowAltNames: a double-faced card's back-face name still names the
        // card the coverage unit is keyed on.
        CardRules rules = forge.model.FModel.getMagicDb().getCommonCards()
                .getRules(name, true);
        if (rules == null) {
            return Optional.empty();
        }
        try {
            Card card = OfflineGame.card(rules);
            return Optional.of(card.getSpells().isEmpty() ? Verdict.UNCASTABLE : Verdict.CASTABLE);
        } catch (RuntimeException e) {
            // A card Forge cannot even instantiate certainly cannot be cast.
            return Optional.of(Verdict.UNCASTABLE);
        }
    }

    static String render(Map<String, Verdict> verdicts) {
        Json.ObjectWriter json = Json.object();
        for (Map.Entry<String, Verdict> entry : verdicts.entrySet()) {
            json.value(entry.getKey(), entry.getValue());
        }
        return json.render();
    }
}
