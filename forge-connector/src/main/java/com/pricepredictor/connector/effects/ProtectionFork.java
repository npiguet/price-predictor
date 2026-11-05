package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.operations.KeywordOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;
import java.util.function.Consumer;

/**
 * An interventional fork that grants a creature a rare protection quality
 * mid-game and plays the turn on (FR-058).
 *
 * <p>Protection from a colour is common in the corpus; protection from a card
 * type, a creature type, a condition or everything is not, and the cards that
 * print one are seldom in a game. So, as the first main phase begins, the fork
 * gives one creature such a protection through the {@code keyword} operation
 * and {@link ForkPlayForward} plays the copy to the end of the turn: what the
 * turn then does about the protection (an attack it makes unblockable, damage
 * it prevents, a spell it leaves without a target, an Aura that falls off) is
 * the record.
 *
 * <p>The quality is one an object that could meet the creature has: a
 * non-land permanent or a card in hand of an opponent of its controller, or a
 * card attached to it. A quality nothing on the board has changes nothing
 * the turn can show. Protection from everything is always among the draws.
 *
 * <p>An interventional fork, as a forced resolution is, and flagged so on
 * every record ({@code interventional = true}, {@code fork = true}): it is a
 * board the live game's decisions did not make. Unlike a forced resolution it
 * is played on, as a directed fork is (D4), so its records are of any kind.
 *
 * <p>Taken at most {@link ForkBudget#MAX_PROTECTION_FORKS_PER_GAME} times a
 * game and only on a run taking interventions, drawn at each first main phase
 * with probability {@value #PROBABILITY} from a generator seeded per game, so
 * a game's one fork spreads over its turns rather than landing on the first
 * board with a creature.
 */
final class ProtectionFork {

    /** The chance a first main phase with a creature on the battlefield is forked. */
    static final double PROBABILITY = 0.25;

    /**
     * The boundary forked at: before the main phase's priority, so the whole
     * turn's spells and its combat meet the protection.
     */
    static final PhaseType PHASE = PhaseType.MAIN1;

    /** Every record of a protection fork carries these flags. */
    static final Consumer<EffectRecord> FLAGS = record -> record.fork(true).interventional(true);

    /** Kept apart from the other streams the same game seed seeds. */
    private static final long SEED_SALT = 0x7F4A7C159E3779B9L;

    private final Game game;
    private final ForkBudget budget;
    private final ForkPlayForward playForward;
    private final Random random;

    /** @param seed the game's seed */
    ProtectionFork(Game game, ForkBudget budget, ForkPlayForward playForward, long seed) {
        this.game = game;
        this.budget = budget;
        this.playForward = playForward;
        this.random = new Random(GameRecording.scramble(seed ^ SEED_SALT));
    }

    /**
     * One grant: a live creature and the protection fragment it gains, as a
     * card script spells it.
     */
    record Grant(Card creature, String quality) {
    }

    /**
     * Take this boundary's protection fork, if it draws one.
     *
     * @return whether a fork was taken, kept or discarded
     */
    boolean collect(PhaseType phase) {
        if (phase != PHASE || game.isGameOver() || !budget.mayProtect()) {
            return false;
        }
        List<Card> creatures = creatures(game);
        // Drawn only where a creature stands, so a game's draws depend on
        // its own boards and nothing else.
        if (creatures.isEmpty() || random.nextDouble() >= PROBABILITY) {
            return false;
        }
        Card creature = creatures.get(random.nextInt(creatures.size()));
        List<String> qualities = qualitiesFor(creature);
        budget.spendProtection();
        take(new Grant(creature, qualities.get(random.nextInt(qualities.size()))));
        return true;
    }

    /**
     * Fork the live game, grant the protection on the copy and play it to the
     * end of the turn. The budget is the caller's to spend, before this.
     *
     * @return whether the fork was kept and its records written
     */
    boolean take(Grant grant) {
        Forks.ForkRun run = new Forks.ForkRun(
                game, () -> { }, () -> game.getPhaseHandler().getPlayerTurn());
        ForkPlayForward.Played played = run.take((verified, snapshots) -> {
            Card forked = Forks.mappedCard(verified.copier(), grant.creature());
            if (forked == null) {
                return null;
            }
            // Before the fork's collectors listen (ForkPlayForward.play opens
            // them): the grant publishes a stats change, which is the fork's
            // premise, not something its turn did.
            KeywordOperation.adding(forked, grant.quality()).apply(verified.fork());
            return playForward.play(verified.fork(), FLAGS);
        }, (verified, result) -> result);
        return played != null;
    }

    /** The creatures on the battlefield, in card-id order. */
    static List<Card> creatures(Game game) {
        List<Card> creatures = new ArrayList<>();
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (card.isCreature()) {
                creatures.add(card);
            }
        }
        creatures.sort(Comparator.comparingInt(Card::getId));
        return creatures;
    }

    /**
     * The rare qualities a grant to {@code creature} draws from: those of
     * every object that could meet it, each once and sorted, then
     * protection from everything.
     */
    static List<String> qualitiesFor(Card creature) {
        TreeSet<String> found = new TreeSet<>();
        for (Card object : objectsMeeting(creature)) {
            found.addAll(ProtectionQualities.of(object));
        }
        found.remove(ProtectionQualities.EVERYTHING);
        List<String> qualities = new ArrayList<>(found);
        qualities.add(ProtectionQualities.EVERYTHING);
        return qualities;
    }

    /**
     * What a protection on {@code creature} can stop: the non-land permanents
     * and the cards in hand of its controller's opponents, which block it, are
     * blocked by it, damage it or target it, and the cards attached to it.
     */
    private static List<Card> objectsMeeting(Card creature) {
        List<Card> objects = new ArrayList<>();
        Player controller = creature.getController();
        for (Player opponent : controller.getOpponents()) {
            for (Card permanent : opponent.getCardsIn(ZoneType.Battlefield)) {
                if (!permanent.isLand()) {
                    objects.add(permanent);
                }
            }
            objects.addAll(opponent.getCardsIn(ZoneType.Hand));
        }
        objects.addAll(creature.getAttachedCards());
        return objects;
    }
}
