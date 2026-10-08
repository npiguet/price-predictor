package com.pricepredictor.connector.effects;

import forge.ai.ComputerUtilMana;
import forge.game.GameEntity;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.apache.commons.lang3.Range;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Uniform draws over the legal options of a play: its modes, its targets and
 * its X.
 *
 * <p>Two callers, one implementation. An interventional fork forces an ability
 * the AI never chose and wants what the effect <b>does</b> rather than what the
 * AI thinks is good, so it draws at random (and was the first home of these
 * draws, in {@code ForkCollector}); the random seat makes the same draws for a
 * play it picked at random, with no {@code P} gate (FR-022c). The fork's first
 * version ignored {@code MinCharmNum}, repeated modes and Pawprint, which both
 * FR-022c and FR-029f need, so one widened implementation fixes both.
 *
 * <p>Every draw is uniform over the legal elements, element by element
 * (FR-022b): a target count between the clause's minimum and maximum, then that
 * many distinct candidates; a mode count between the charm's minimum and
 * maximum, then that many modes, repeating only where the charm allows it; an
 * X between the smallest legal value and the most the payer's mana can reach.
 * A draw the engine rejects is the caller's to redraw, up to
 * {@link #MAX_DRAWS}: a uniform draw fails only on cross-element constraints,
 * and twenty draws make a persistent failure rare without stalling a priority
 * loop that runs thousands of times a game.
 */
public final class RandomChoices {

    /** Redraws before a caller gives up on a decision (FR-022b, FR-022c). */
    public static final int MAX_DRAWS = 20;

    /** X values above this are never drawn, which is also the engine's own ceiling. */
    private static final int MAX_X = 99;

    private final Random random;

    public RandomChoices(Random random) {
        this.random = random;
    }

    // ── modes ───────────────────────────────────────────────────────────

    /**
     * Draw a charm's modes, without setting or chaining them.
     *
     * <p>The chosen entries are the charm's own {@code Choices$} objects, as
     * {@code CharmEffect.makeChoices} expects them: the random seat sets them
     * as the chosen list and lets the cast's own {@code makeChoices} chain
     * clones of them, which is how a mode's targets, drawn on the original,
     * reach the clone that resolves.
     *
     * @return the modes, possibly empty where the charm's minimum is zero;
     *         null when the ability is not a charm or has no legal mode set
     */
    public List<AbilitySub> drawModes(SpellAbility charm) {
        if (charm.getApi() != ApiType.Charm) {
            return null;
        }
        List<AbilitySub> options = CharmEffect.makePossibleOptions(charm);
        if (options == null || options.isEmpty()) {
            return null;
        }
        if (charm.isEntwine()) {
            return new ArrayList<>(options);
        }
        Card host = charm.getHostCard();
        boolean repeat = charm.hasParam("CanRepeatModes");
        if (charm.hasParam("Pawprint")) {
            return drawPawprintModes(charm, options, repeat);
        }
        int max = AbilityUtils.calculateAmount(
                host, charm.getParamOrDefault("CharmNum", "1"), charm);
        int min = charm.hasParam("MinCharmNum")
                ? AbilityUtils.calculateAmount(host, charm.getParam("MinCharmNum"), charm)
                : max;
        if (!repeat) {
            max = Math.min(max, options.size());
        }
        min = Math.max(0, Math.min(min, max));
        if (max < 0) {
            return null;
        }
        int count = min + random.nextInt(max - min + 1);
        List<AbilitySub> chosen = new ArrayList<>(count);
        if (repeat) {
            for (int i = 0; i < count; i++) {
                chosen.add(options.get(random.nextInt(options.size())));
            }
        } else {
            List<AbilitySub> shuffled = new ArrayList<>(options);
            Collections.shuffle(shuffled, random);
            chosen.addAll(shuffled.subList(0, count));
        }
        return chosen;
    }

    /**
     * A Pawprint charm: one mode at a time, uniformly among those whose
     * pawprint cost still fits, until nothing fits (FR-022c).
     */
    private List<AbilitySub> drawPawprintModes(
            SpellAbility charm, List<AbilitySub> options, boolean repeat) {
        Card host = charm.getHostCard();
        int budget = AbilityUtils.calculateAmount(host, charm.getParam("Pawprint"), charm);
        List<AbilitySub> remaining = new ArrayList<>(options);
        List<AbilitySub> chosen = new ArrayList<>();
        while (true) {
            List<AbilitySub> fitting = new ArrayList<>();
            for (AbilitySub option : remaining) {
                if (pawprintCost(option) <= budget) {
                    fitting.add(option);
                }
            }
            if (fitting.isEmpty()) {
                return chosen;
            }
            AbilitySub pick = fitting.get(random.nextInt(fitting.size()));
            chosen.add(pick);
            budget -= pawprintCost(pick);
            if (!repeat) {
                remaining.remove(pick);
            }
        }
    }

    private static int pawprintCost(AbilitySub option) {
        return AbilityUtils.calculateAmount(
                option.getHostCard(), option.getParamOrDefault("Pawprint", "0"), option);
    }

    /**
     * Draw, set and chain a charm's modes, for a resolution that bypasses the
     * cast — the fork pushes its ability onto the stack directly, so nothing
     * else would chain them.
     *
     * @return false when the charm has no legal mode set; true for a
     *         non-charm, which needs nothing chosen
     */
    public boolean chooseModes(SpellAbility ability) {
        if (ability.getApi() != ApiType.Charm || ability.getChosenList() != null) {
            return true;
        }
        List<AbilitySub> chosen = drawModes(ability);
        if (chosen == null) {
            return false;
        }
        // Always set, even where chainAbilities alone would do: the chosen
        // list is how the per-mode collector maps a resolving clone back to
        // its Choices$ position when two modes share a fingerprint.
        ability.setChosenList(chosen);
        CharmEffect.chainAbilities(ability, chosen);
        return true;
    }

    // ── targets ─────────────────────────────────────────────────────────

    /**
     * Give every targeting clause of the chain legal targets.
     *
     * <p>The whole {@code getSubAbility()} chain, not just the root: a
     * sub-ability targets independently, and a chain whose second clause has no
     * target resolves into the same silence as one whose first has none.
     *
     * @return false when some clause cannot reach its minimum, which is a draw
     *         to retry or a play to abandon rather than one that did nothing
     */
    public boolean chooseTargets(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null;
                clause = clause.getSubAbility()) {
            if (!clause.usesTargeting()) {
                continue;
            }
            if (!chooseTargetsFor(clause)) {
                return false;
            }
        }
        return true;
    }

    /**
     * One clause's targets: a count uniform in [min, max], then that many
     * distinct legal candidates (FR-022b).
     */
    public boolean chooseTargetsFor(SpellAbility clause) {
        clause.resetTargets();
        List<GameEntity> candidates =
                new ArrayList<>(clause.getTargetRestrictions().getAllCandidates(clause));
        int min = Math.max(0, clause.getMinTargets());
        int max = Math.min(clause.getMaxTargets(), candidates.size());
        if (max < min) {
            return false;
        }
        int count = min + random.nextInt(max - min + 1);
        Collections.shuffle(candidates, random);
        for (int i = 0; i < count; i++) {
            if (!clause.canAddMoreTarget()) {
                break;
            }
            clause.getTargets().add(candidates.get(i));
        }
        return clause.isTargetNumberValid();
    }

    // ── X ───────────────────────────────────────────────────────────────

    /**
     * Announce an X, uniform between the smallest legal value and the most the
     * payer's available mana can reach (FR-022c).
     *
     * <p>Left alone where the ability has no X or already has one: the engine
     * asks the controller only for an unannounced X, and an announced one is
     * what both the random seat's cast and the fork's direct push read.
     *
     * @param payer the player whose mana bounds the draw, or null for a
     *              resolution nobody pays for, where {@code floor} is the
     *              answer
     * @param floor the least X worth announcing: zero for a seat that pays,
     *              one for a fork, where an X of zero resolves into the same
     *              silence a missing target does
     */
    public void announceX(SpellAbility ability, Player payer, int floor) {
        if (!ability.costHasX() || ability.getXManaCostPaid() != null) {
            return;
        }
        int min = floor;
        int max = floor;
        try {
            Range<Integer> bounds = AbilityUtils.getAnnouncementBounds(ability, "X");
            min = Math.max(floor, bounds.getMinimum());
            max = Math.min(MAX_X, bounds.getMaximum());
            if (payer != null) {
                max = Math.min(max, ComputerUtilMana.determineLeftoverMana(ability, payer, false));
            } else {
                max = min;
            }
        } catch (RuntimeException e) {
            // A bound the script computes from a board it has not been put on
            // yet; the floor is the one value that is always legal to announce.
            max = min;
        }
        ability.setXManaCostPaid(max < min ? min : min + random.nextInt(max - min + 1));
    }

    /** The seeded source, for a caller that shuffles alongside these draws. */
    public Random random() {
        return random;
    }
}
