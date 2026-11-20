package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.ForcedPlayRun;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.TestForks;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A started two-seat game in {@code phase} of your turn: you (P0, the
 * scenarios' A) and the opponent (P1, B).
 */
final class RuleBoard {
    final Game game;
    final Player you;
    final Player opponent;
    /** The object the variant changes. */
    Card subject;
    /** The other object a precondition binds, where it binds two. */
    Card other;

    RuleBoard(PhaseType phase) {
        game = TestBoard.startedTwoPlayerGame(phase);
        you = game.getPlayers().get(0);
        opponent = game.getPlayers().get(1);
    }

    /** A permanent since the turn began, with a timestamp of its own. */
    Card card(Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        return card;
    }

    /** {@code count} permanents named {@code name} of {@code owner}; the first, or null for none. */
    Card cards(Player owner, String name, int count) {
        Card first = null;
        for (int i = 0; i < count; i++) {
            Card card = card(owner, name);
            first = first == null ? card : first;
        }
        return first;
    }

    /** {@code attacker} attacks the opponent, no blocker declared yet. */
    void attacking(Card attacker) {
        Combat combat = new Combat(you);
        combat.addAttacker(attacker, opponent);
        game.getPhaseHandler().setCombat(combat);
    }

    /** {@code names} at the bottom of {@code owner}'s library in order, so the first listed is the topmost of them. */
    List<Card> library(Player owner, String... names) {
        List<Card> cards = new ArrayList<>();
        for (String name : names) {
            cards.add(game.getAction().moveToLibrary(TestBoard.card(game, owner, name), -1, null, null));
        }
        return cards;
    }

    /** {@code library}'s cards at the bottom of {@code owner}'s library in order, as {@link #library(Player, String...)} puts them. */
    List<Card> library(Player owner, Library library) {
        return library(owner, library.names().toArray(String[]::new));
    }

    /**
     * Copy the game as a directed fork does, score-check the copy as every
     * fork is checked before it is changed, map the precondition's
     * binding into the copy and apply the rule's one variant, then compare
     * the copy with the live board. The live board is left as it was.
     *
     * <p>The edited copy is then put through the state-based check the
     * fork's first priority makes, which must move no card: an edit that
     * the check undoes (a creature dying of the toughness it was given,
     * an Aura falling off, the legend rule) would fork a board the rule's
     * variant does not describe.
     */
    RuleTwin twin(DirectedRule rule) {
        assertEquals(1, rule.variants().size(), rule.name() + " registers one variant");
        Binding binding = rule.precondition(game).orElseThrow(
                () -> new AssertionError(rule.name() + ": the precondition does not hold"));
        Map<String, Object> real = BoardStates.of(game);
        GameCopier copier = new GameCopier(game);
        Game fork = copier.makeCopy();
        assertEquals(Set.of(), changes(real, BoardStates.of(fork)), "the unedited copy differs");
        assertTrue(TestForks.scoreChecks(game, copier, fork), rule.name() + ": the copy fails its score check");

        Binding forked = binding.inFork(copier);
        Application application = rule.variants().values().iterator().next();
        assertEquals(Optional.empty(), ForcedPlayRun.dryRun(application, binding, game),
                rule.name() + ": the forced play's dry run on the live board");
        application.edit(forked).apply(fork);

        Map<String, Object> edited = BoardStates.of(fork);
        assertEquals(real, BoardStates.of(game), "the edit reached the live game");
        Map<String, String> placed = placed(fork);
        Set<String> tidied = keywordEffectCards(fork);
        fork.getAction().checkStateEffects(true);
        assertEquals(placed, placed(fork),
                rule.name() + ": the first state-based check moved a card of the edited board");
        Map<String, Object> checked = BoardStates.of(fork);
        assertEquals(Set.of(), withoutTidied(changes(edited, checked), tidied, edited, checked),
                rule.name() + ": the first state-based check changed the edited board");
        assertFalse(fork.isGameOver(), rule.name() + ": the first state-based check ended the fork's game");
        for (Player player : fork.getRegisteredPlayers()) {
            assertFalse(player.hasLost(), rule.name() + ": the first state-based check lost " + player);
        }
        assertEquals(real, BoardStates.of(game), "the check reached the live game");
        assertForcedPlayMayBeMade(rule, application, application.afterEdit(forked, fork), fork);
        return new RuleTwin(changes(real, edited), real, edited, forked, fork);
    }

    /**
     * The variant's forced play (FR-010b), if it forces one, can be made
     * on the edited copy: its source, line or attack is there, and a cast
     * or an activation is playable now by the rules with its named
     * targets, the line the forcing seat would choose at this priority.
     */
    private static void assertForcedPlayMayBeMade(
            DirectedRule rule, Application application, Binding forked, Game fork) {
        if (application.forced() == null) {
            return;
        }
        ForcedPlayRun run = ForcedPlayRun.start(application, forked, fork);
        if (run.play().playsALine()) {
            assertTrue(run.choose(run.actor()) != null,
                    rule.name() + ": the forced line is playable on the edited copy");
        }
    }

    /**
     * The effect card the engine tidies away at a state-based check once
     * its source is gone: a player's {@code Keyword Effects} card
     * ({@code Player.getKeywordCard}), which holds the keywords statics
     * give the player. No other effect card, emblem or command-zone card
     * leaves at a check, and a change to one fails the twin.
     */
    static final String KEYWORD_EFFECTS = "Keyword Effects";

    /** Whether {@code card} is a player's {@link #KEYWORD_EFFECTS} card. */
    private static boolean keywordEffectCard(Card card) {
        return card.isImmutable() && !card.isEmblem() && KEYWORD_EFFECTS.equals(card.getName());
    }

    /** The snapshot ids of the players' {@link #KEYWORD_EFFECTS} cards in {@code game} ({@code E<id>}). */
    private static Set<String> keywordEffectCards(Game game) {
        Set<String> ids = new TreeSet<>();
        for (Card card : game.getCardsInGame()) {
            if (keywordEffectCard(card)) {
                ids.add("E" + card.getId());
            }
        }
        return ids;
    }

    /** Whether a change names one of {@code ids}: the entity itself, its fields, or a link to it. */
    private static boolean names(String change, Set<String> ids) {
        return ids.stream().anyMatch(id -> change.equals("-" + id) || change.equals("+" + id)
                || change.startsWith(id + ".") || change.contains("=" + id + ",")
                || change.contains("=" + id + "}"));
    }

    /**
     * {@code changes} without what names one of the {@code tidied}
     * {@link #KEYWORD_EFFECTS} cards, and without {@code global.floating_effects}
     * when the two snapshots' floating effects differ only by entries
     * naming one of them.
     */
    static Set<String> withoutTidied(Set<String> changes, Set<String> tidied,
                                     Map<String, Object> edited, Map<String, Object> checked) {
        Set<String> kept = new TreeSet<>();
        for (String change : changes) {
            if (!names(change, tidied)) {
                kept.add(change);
            }
        }
        if (kept.contains("global.floating_effects")
                && floatingWithout(edited, tidied).equals(floatingWithout(checked, tidied))) {
            kept.remove("global.floating_effects");
        }
        return kept;
    }

    /** The snapshot's floating effects, as text, less each entry naming one of {@code ids}. */
    private static List<String> floatingWithout(Map<String, Object> snapshot, Set<String> ids) {
        Object global = snapshot.get("global");
        Object floating = global instanceof Map<?, ?> map ? map.get("floating_effects") : null;
        List<String> kept = new ArrayList<>();
        if (floating instanceof List<?> entries) {
            for (Object entry : entries) {
                String text = String.valueOf(entry);
                if (ids.stream().noneMatch(id -> text.matches(".*\\b" + id + "\\b.*"))) {
                    kept.add(text);
                }
            }
        }
        return kept;
    }

    /**
     * Every card of {@code game} by id, to its zone and what it is
     * attached to; not a player's {@link #KEYWORD_EFFECTS} card, which the
     * check tidies away once its source is gone.
     */
    private static Map<String, String> placed(Game game) {
        Map<String, String> placed = new java.util.TreeMap<>();
        for (Card card : game.getCardsInGame()) {
            if (keywordEffectCard(card)) {
                continue;
            }
            placed.put(card.getName() + "#" + card.getId(),
                    card.getZone() + " on " + card.getEntityAttachedTo());
        }
        return placed;
    }
}
