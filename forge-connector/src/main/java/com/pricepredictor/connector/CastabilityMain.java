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

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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

    /** A card's answer; {@link #wireName()} is what the Python side reads. */
    enum Verdict {
        CASTABLE("castable"),
        UNCASTABLE("uncastable");

        private final String wireName;

        Verdict(String wireName) {
            this.wireName = wireName;
        }

        String wireName() {
            return wireName;
        }
    }

    public static void main(String[] args) {
        ToolMain.Args cli = ToolMain.args(args, "--cards", "--output");
        String cardsArg = cli.get("--cards");
        String outputArg = cli.get("--output");
        if (cardsArg == null || outputArg == null) {
            System.err.println("Error: --cards and --output are required");
            System.exit(2);
        }
        Path cardsFile = Path.of(cardsArg);
        Path output = Path.of(outputArg);

        ToolMain.run(() -> {
            ForgeEnvironmentInitializer.initialize();
            List<String> names = Files.readAllLines(cardsFile, StandardCharsets.UTF_8)
                    .stream().map(String::trim).filter(s -> !s.isEmpty()).toList();
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
    private static Optional<Verdict> verdictFor(String name) {
        // allowAltNames: a double-faced card's back-face name still names the
        // card the coverage unit is keyed on.
        CardRules rules = forge.model.FModel.getMagicDb().getCommonCards()
                .getRules(name, true);
        if (rules == null) {
            return Optional.empty();
        }
        try {
            PaperCard paper = new PaperCard(rules, "UNK", CardRarity.Common);
            Game game = ConsultGame.INSTANCE;
            Card card = CardFactory.getCard(paper, null, game.nextCardId(), game);
            return Optional.of(card.getSpells().isEmpty() ? Verdict.UNCASTABLE : Verdict.CASTABLE);
        } catch (RuntimeException e) {
            // A card Forge cannot even instantiate certainly cannot be cast.
            return Optional.of(Verdict.UNCASTABLE);
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

    static String render(Map<String, Verdict> verdicts) {
        StringJoiner joiner = new StringJoiner(",", "{", "}");
        for (Map.Entry<String, Verdict> entry : verdicts.entrySet()) {
            joiner.add(Json.string(entry.getKey()) + ":"
                    + Json.string(entry.getValue().wireName()));
        }
        return joiner.toString();
    }
}
