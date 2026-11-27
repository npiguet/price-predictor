package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.RandomChoices;
import forge.ai.ComputerUtilCost;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.cost.Cost;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiPredicate;
import java.util.function.BooleanSupplier;

/**
 * The {@link RandomSeatController}'s attack and block draws (024 FR-022b), once
 * the seat has drawn to declare at random.
 *
 * <p>Each declaration is drawn until the engine accepts it, at most
 * {@link RandomChoices#MAX_DRAWS} times, and a seat that never drew an
 * acceptable one is handed back to the AI with nothing declared. The draws
 * come from the seat's own stream, in the order the creatures and their
 * options are listed: a seeded run picks the same declaration only while
 * they are listed the same way.
 */
final class RandomCombatDeclarations {

    private final Player player;
    private final Random random;

    /**
     * @param player the seat's player, whose creatures are declared
     * @param random the seat's draws
     */
    RandomCombatDeclarations(Player player, Random random) {
        this.player = player;
        this.random = random;
    }

    /**
     * Each creature that can attack attacks with probability one half, at a
     * defender drawn among those it may attack.
     *
     * <p>The engine validates the whole declaration — requirements, maximums,
     * "can't attack alone" — and a declaration it rejects is redrawn; after
     * {@link RandomChoices#MAX_DRAWS} the AI declares.
     *
     * @param fallBack declare as the AI would
     */
    void declareAttackers(Combat combat, Runnable fallBack) {
        CardCollection mine = player.getCreaturesInPlay();
        Map<GameEntity, List<Card>> legalByDefender = legalBy(
                combat.getDefenders(), mine,
                (defender, creature) -> CombatUtil.canAttack(creature, defender));
        drawUntilAccepted(
                combat::clearAttackers,
                () -> drawAttacks(combat, mine, legalByDefender),
                () -> CombatUtil.validateAttackers(combat),
                fallBack);
    }

    /**
     * Each creature that can block picks uniformly from no block plus the
     * attackers it may legally block.
     *
     * <p>{@code CombatUtil.validateBlocks} is the engine's own check and the
     * engine never calls it, so the seat calls it before returning; a
     * declaration it rejects is redrawn, then left to the AI.
     *
     * @param fallBack declare as the AI would
     */
    void declareBlockers(Combat combat, Runnable fallBack) {
        List<Card> candidates = blockCandidates(combat);
        Map<Card, List<Card>> legalByAttacker = legalBy(
                attackersAtThisSeat(combat), candidates,
                (attacker, blocker) -> CombatUtil.canBlock(attacker, blocker, combat));
        drawUntilAccepted(
                () -> clearBlocks(combat),
                () -> drawBlocks(combat, candidates, legalByAttacker),
                () -> CombatUtil.validateBlocks(combat, player) == null,
                fallBack);
    }

    /** The seat's creatures that can block at all, in play order. */
    private List<Card> blockCandidates(Combat combat) {
        List<Card> candidates = new ArrayList<>();
        for (Card creature : player.getCreaturesInPlay()) {
            if (CombatUtil.canBlock(creature, combat)) {
                candidates.add(creature);
            }
        }
        return candidates;
    }

    /** The attackers coming at whatever the seat defends, defender by defender. */
    private List<Card> attackersAtThisSeat(Combat combat) {
        List<Card> attackers = new ArrayList<>();
        for (GameEntity defended : combat.getDefendersControlledBy(player)) {
            attackers.addAll(combat.getAttackersOf(defended));
        }
        return attackers;
    }

    /**
     * For each of {@code keys}, the creatures that may legally be declared
     * against it, both in the order given.
     *
     * <p>Read once per declaration rather than per draw: legality does not
     * change between redraws.
     */
    private static <K> Map<K, List<Card>> legalBy(
            Iterable<? extends K> keys, Iterable<Card> creatures,
            BiPredicate<K, Card> legal) {
        Map<K, List<Card>> legalBy = new LinkedHashMap<>();
        for (K key : keys) {
            List<Card> declarable = new ArrayList<>();
            for (Card creature : creatures) {
                if (legal.test(key, creature)) {
                    declarable.add(creature);
                }
            }
            legalBy.put(key, declarable);
        }
        return legalBy;
    }

    /**
     * What one creature may be declared against: every key of {@code legalBy}
     * whose list holds it, in the map's order.
     *
     * <p>The order is part of the draw: a seeded run picks the same option
     * only while the options are listed the same way.
     */
    private static <K> List<K> optionsFor(Card creature, Map<K, List<Card>> legalBy) {
        List<K> options = new ArrayList<>();
        legalBy.forEach((key, legal) -> {
            if (legal.contains(creature)) {
                options.add(key);
            }
        });
        return options;
    }

    /**
     * Draw a combat declaration until the engine accepts one, at most
     * {@link RandomChoices#MAX_DRAWS} times, and hand a seat that never drew
     * an acceptable one back to the AI with nothing declared.
     *
     * @param clear    withdraw everything the seat has declared so far
     * @param drawOnce declare one random draw
     * @param accepted whether the engine accepts the declaration as it stands
     * @param fallBack declare as the AI would
     */
    private static void drawUntilAccepted(
            Runnable clear, Runnable drawOnce, BooleanSupplier accepted, Runnable fallBack) {
        for (int attempt = 0; attempt < RandomChoices.MAX_DRAWS; attempt++) {
            clear.run();
            drawOnce.run();
            if (accepted.getAsBoolean()) {
                return;
            }
        }
        clear.run();
        fallBack.run();
    }

    /** One draw of the attack declaration, before the engine validates it. */
    private void drawAttacks(
            Combat combat, CardCollection mine, Map<GameEntity, List<Card>> legalByDefender) {
        for (Card creature : mine) {
            List<GameEntity> options = optionsFor(creature, legalByDefender);
            if (!options.isEmpty() && random.nextBoolean()) {
                combat.addAttacker(creature, options.get(random.nextInt(options.size())));
            }
        }
        removeUnpayableAttackers(combat);
    }

    /**
     * Attackers whose attack cost the seat cannot pay are withdrawn before
     * validation, as the AI withdraws its own: the decision to pay a
     * Propaganda tax is made at declaration (CR 508.1d), and an attacker
     * left in that cannot pay is removed by the engine after the fact.
     */
    private void removeUnpayableAttackers(Combat combat) {
        Game game = player.getGame();
        for (Card attacker : new ArrayList<>(combat.getAttackers())) {
            Cost attackCost = CombatUtil.getAttackCost(
                    game, attacker, combat.getDefenderByAttacker(attacker));
            if (attackCost != null && !canPayAttackCost(attacker, attackCost)) {
                combat.removeFromCombat(attacker);
            }
        }
    }

    /** Whether the seat can pay {@code attacker}'s attack cost, as the AI judges it. */
    private boolean canPayAttackCost(Card attacker, Cost attackCost) {
        SpellAbility fakeSA = new SpellAbility.EmptySa(attacker, attacker.getController());
        fakeSA.setCardState(attacker.getCurrentState());
        fakeSA.setPayCosts(attackCost);
        fakeSA.setSVar("X", "0");
        return ComputerUtilCost.canPayCost(attackCost, fakeSA, player, true);
    }

    /**
     * One draw of the block declaration: each candidate picks from no block,
     * listed first, then the attackers it may block.
     */
    private void drawBlocks(
            Combat combat, List<Card> candidates, Map<Card, List<Card>> legalByAttacker) {
        for (Card blocker : candidates) {
            List<Card> options = new ArrayList<>();
            options.add(null);
            options.addAll(optionsFor(blocker, legalByAttacker));
            Card attacker = options.get(random.nextInt(options.size()));
            if (attacker != null) {
                combat.addBlocker(attacker, blocker);
            }
        }
    }

    private void clearBlocks(Combat combat) {
        for (Card blocker : new ArrayList<>(combat.getAllBlockers())) {
            if (blocker.getController() != player) {
                continue;
            }
            for (Card attacker : new ArrayList<>(combat.getAttackersBlockedBy(blocker))) {
                combat.removeBlockAssignment(attacker, blocker);
            }
        }
    }
}
