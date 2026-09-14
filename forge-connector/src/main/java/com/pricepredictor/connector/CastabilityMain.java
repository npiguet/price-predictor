package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.Json;
import forge.card.CardRules;
import forge.card.CardRarity;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.item.PaperCard;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

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

    public static final String CASTABLE = "castable";
    public static final String UNCASTABLE = "uncastable";

    public static void main(String[] args) {
        Path cardsFile = null;
        Path output = null;
        Path cardsPath = Path.of("output/cardsfolder");

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--cards" -> {
                    if (i + 1 < args.length) cardsFile = Path.of(args[++i]);
                }
                case "--output" -> {
                    if (i + 1 < args.length) output = Path.of(args[++i]);
                }
                case "--cards-path" -> {
                    if (i + 1 < args.length) cardsPath = Path.of(args[++i]);
                }
            }
        }
        if (cardsFile == null || output == null) {
            System.err.println("Error: --cards and --output are required");
            System.exit(2);
        }

        try {
            ForgeEnvironmentInitializer.initialize();
            List<String> names = Files.readAllLines(cardsFile, StandardCharsets.UTF_8)
                    .stream().map(String::trim).filter(s -> !s.isEmpty()).toList();
            Map<String, String> verdicts = consult(names);
            if (output.getParent() != null) {
                Files.createDirectories(output.getParent());
            }
            Files.writeString(output, render(verdicts), StandardCharsets.UTF_8);
            long castable = verdicts.values().stream()
                    .filter(CASTABLE::equals).count();
            System.out.println(
                    "Consulted " + verdicts.size() + " cards: " + castable
                            + " castable, " + (verdicts.size() - castable)
                            + " uncastable. Output: " + output
                            + " (cards from " + cardsPath + ")");
            System.exit(0);
        } catch (IOException | RuntimeException e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /** One verdict per name Forge recognizes; names it does not are omitted. */
    static Map<String, String> consult(List<String> names) {
        Map<String, String> verdicts = new LinkedHashMap<>();
        for (String name : names) {
            String verdict = verdictFor(name);
            if (verdict != null) {
                verdicts.put(name, verdict);
            }
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
     * <p>The card is built with an explicit, non-negative id. {@code CardFactory}
     * treats a negative id as "for display only" and then reads none of the
     * script's traits — no keywords, no triggers, no statics and no {@code A:}
     * lines, which is where an instant's or sorcery's spell lives — so all
     * that is left are the spells {@code CardState} synthesises from the type
     * line: a permanent's {@code SpellPermanent}, a land's {@code LandAbility}.
     * The three-argument {@code getCard(paper, owner, game)} derives the id
     * from the owner and passes -1 for a null one, so an ownerless card has to
     * be given its id here. The game it is built against is the same
     * player-less dummy {@code convert} builds every card against in
     * {@link RulesParser}: with no game at all, three cards' ability setup
     * throws and would be misread as uncastable.
     */
    private static String verdictFor(String name) {
        // allowAltNames: a double-faced card's back-face name still names the
        // card the coverage unit is keyed on.
        CardRules rules = forge.model.FModel.getMagicDb().getCommonCards()
                .getRules(name, true);
        if (rules == null) {
            return null;
        }
        try {
            PaperCard paper = new PaperCard(rules, "UNK", CardRarity.Common);
            Game game = ConsultGame.INSTANCE;
            Card card = CardFactory.getCard(paper, null, game.nextCardId(), game);
            return card.getSpells().isEmpty() ? UNCASTABLE : CASTABLE;
        } catch (RuntimeException e) {
            // A card Forge cannot even instantiate certainly cannot be cast.
            return UNCASTABLE;
        }
    }

    /** A player-less game to build cards against; created once, on first use. */
    private static class ConsultGame {
        static final Game INSTANCE = create();

        private static Game create() {
            GameRules rules = new GameRules(GameType.Constructed);
            return new Game(List.of(), rules,
                    new Match(rules, List.of(), "castability-consult"));
        }
    }

    static String render(Map<String, String> verdicts) {
        StringJoiner joiner = new StringJoiner(",", "{", "}");
        for (Map.Entry<String, String> entry : verdicts.entrySet()) {
            joiner.add(Json.string(entry.getKey()) + ":" + Json.string(entry.getValue()));
        }
        return joiner.toString();
    }
}
