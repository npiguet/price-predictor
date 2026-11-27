package com.pricepredictor.connector.effects;

import forge.ai.ComputerUtilMana;
import forge.game.GameObject;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.apache.commons.lang3.Range;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Random;

/**
 * Uniform draws over the legal options of a play: its modes, its targets and
 * its X.
 *
 * <p>Two callers, one implementation. An interventional fork forces an ability
 * the AI never chose and wants what the effect <b>does</b> rather than what the
 * AI thinks is good, so it draws at random; the random seat makes the same
 * draws for a play it picked at random, with no {@code P} gate (024 FR-022c). Both
 * need {@code MinCharmNum}, repeated modes and Pawprint honoured (024 FR-022c,
 * 024 FR-029f), so one implementation serves both.
 *
 * <p>Every draw is uniform over the legal elements, element by element
 * (024 FR-022b): a target count between the clause's minimum and maximum, then that
 * many distinct candidates; a mode count between the charm's minimum and
 * maximum, then that many modes, repeating only where the charm allows it; an
 * X between the smallest legal value and the most the payer's mana can reach.
 * A draw the engine rejects is the caller's to redraw, up to
 * {@link #MAX_DRAWS}: a uniform draw fails only on cross-element constraints,
 * and twenty draws make a persistent failure rare without stalling a priority
 * loop that runs thousands of times a game.
 */
public final class RandomChoices {

    /** Redraws before a caller gives up on a decision (024 FR-022b, FR-022c). */
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
     * <p>Only among the modes whose targets exist (603.3c): a mode any
     * clause of whose chain needs more targets than the board offers is no
     * legal choice. Forge's own list ({@code CharmEffect.makePossibleOptions})
     * reads a mode's first clause alone, so a mode targeting in a later
     * clause is read here, and a draw never lands on a mode set the targets
     * draw would then fail on.
     *
     * @return the modes, present and possibly empty where the charm's minimum
     *         is zero; empty when the ability is not a charm or has no legal
     *         mode set (no caller tells the two apart: both guard on the charm
     *         first, so only the no-legal-set case reaches them)
     */
    public Optional<List<AbilitySub>> drawModes(SpellAbility charm) {
        if (charm.getApi() != ApiType.Charm) {
            return Optional.empty();
        }
        List<AbilitySub> printed = CharmEffect.makePossibleOptions(charm);
        if (printed == null || printed.isEmpty()) {
            return Optional.empty();
        }
        List<AbilitySub> options = reachableModes(printed);
        if (charm.isEntwine()) {
            return options.size() == printed.size()
                    ? Optional.of(new ArrayList<>(options)) : Optional.empty();
        }
        boolean repeat = charm.hasParam("CanRepeatModes");
        if (charm.hasParam("Pawprint")) {
            return Optional.of(drawPawprintModes(charm, options, repeat));
        }
        return modeBounds(charm, printed.size(), options, repeat)
                .map(bounds -> drawModeSet(options, bounds, repeat));
    }

    /** A count range, both ends inclusive. */
    private record Bounds(int min, int max) {
    }

    /**
     * How many modes a non-Pawprint charm may take, or empty where no legal
     * mode set exists.
     */
    private static Optional<Bounds> modeBounds(
            SpellAbility charm, int printedSize, List<AbilitySub> options, boolean repeat) {
        int max = AbilityUtils.calculateAmount(
                charm.getHostCard(), charm.getParamOrDefault("CharmNum", "1"), charm);
        int min = CharmModes.minimumModes(charm);
        if (!repeat) {
            // A mode set of the printed size needs that many distinct legal
            // modes: fewer legal than the charm's minimum is no legal set.
            if (Math.min(min, printedSize) > options.size()) {
                return Optional.empty();
            }
            max = Math.min(max, options.size());
        } else if (options.isEmpty() && min > 0) {
            return Optional.empty();
        }
        min = Math.max(0, Math.min(min, max));
        return max < 0 ? Optional.empty() : Optional.of(new Bounds(min, max));
    }

    /** A count in {@code bounds}, then that many modes, repeating only where {@code repeat}. */
    private List<AbilitySub> drawModeSet(List<AbilitySub> options, Bounds bounds, boolean repeat) {
        int count = between(bounds.min(), bounds.max());
        if (!repeat) {
            return drawDistinct(options, count, random);
        }
        List<AbilitySub> chosen = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            chosen.add(options.get(random.nextInt(options.size())));
        }
        return chosen;
    }

    /**
     * The modes whose chains reach their minimums, counted with the leftover
     * targets of the AI's evaluation set aside and given back.
     */
    private static List<AbilitySub> reachableModes(List<AbilitySub> printed) {
        return TargetingClauses.withNoneChosenOnModes(printed, () -> {
            List<AbilitySub> reachable = new ArrayList<>();
            for (AbilitySub option : printed) {
                if (TargetingClauses.chainReachesMinimum(option)) {
                    reachable.add(option);
                }
            }
            return reachable;
        });
    }

    /**
     * A Pawprint charm: one mode at a time, uniformly among those whose
     * pawprint cost still fits, until nothing fits (024 FR-022c).
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
        Optional<List<AbilitySub>> drawn = drawModes(ability);
        if (drawn.isEmpty()) {
            return false;
        }
        // Always set, even where chainAbilities alone would do: the chosen
        // list is how the per-mode collector maps a resolving clone back to
        // its Choices$ position when two modes share a fingerprint.
        ability.setChosenList(drawn.get());
        CharmEffect.chainAbilities(ability, drawn.get());
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
     * distinct legal candidates (024 FR-022b), and a divided amount split among
     * them. The candidates are {@link TargetingClauses#candidatesOf}'s, so a
     * clause that targets a spell takes the spell on the stack, never its
     * card.
     *
     * <p>The candidates are offered in a random order and each is taken only
     * if the clause can still target it beside those already taken
     * ({@code SpellAbility.canTarget}), which is where Forge reads the
     * constraints across a clause's targets: different or the same
     * controllers, different names or mana values, a total mana value or
     * power. A clause that "divides as you choose" gets its amount
     * split at random over its targets, each at least one while the amount
     * lasts ({@link #divide}); without it the resolution reads a null share.
     */
    public boolean chooseTargetsFor(SpellAbility clause) {
        // clearTargets also sets the amount a divided clause splits.
        clause.clearTargets();
        List<GameObject> candidates = TargetingClauses.candidatesOf(clause);
        Optional<Bounds> bounds = targetBounds(clause, candidates.size());
        if (bounds.isEmpty()) {
            return false;
        }
        addTargets(clause, candidates, between(bounds.get().min(), bounds.get().max()));
        if (!clause.isTargetNumberValid()) {
            return false;
        }
        if (clause.isDividedAsYouChoose()) {
            divide(clause);
        }
        return true;
    }

    /** How many targets the clause may take, or empty where it cannot reach its minimum. */
    private static Optional<Bounds> targetBounds(SpellAbility clause, int candidateCount) {
        int min = Math.max(0, clause.getMinTargets());
        int max = Math.min(clause.getMaxTargets(), candidateCount);
        if (clause.isDividedAsYouChoose()) {
            // Each target takes at least one of the amount (601.2d): no more
            // targets than the amount, which an announced X decides, and at
            // least one target for an amount to divide at all.
            int amount = dividedAmount(clause);
            max = Math.min(max, amount);
            if (amount > 0) {
                min = Math.max(min, 1);
            }
        }
        return max < min ? Optional.empty() : Optional.of(new Bounds(min, max));
    }

    /** Offer the candidates in a random order, taking each the clause can still target, up to {@code count}. */
    private void addTargets(SpellAbility clause, List<GameObject> candidates, int count) {
        for (GameObject target : drawDistinct(candidates, candidates.size(), random)) {
            if (clause.getTargets().size() >= count || !clause.canAddMoreTarget()) {
                break;
            }
            if (clause.canTarget(target)) {
                clause.getTargets().add(target);
            }
        }
    }

    /** The amount a divided clause splits, with its X as announced; never below zero. */
    private static int dividedAmount(SpellAbility clause) {
        Integer amount = clause.getDividedValue();
        if (amount == null) {
            amount = AbilityUtils.calculateAmount(
                    clause.getHostCard(), clause.getParam("DividedAsYouChoose"), clause);
        }
        return Math.max(0, amount);
    }

    /**
     * Split a divided clause's amount among its targets at random: each
     * target one while the amount lasts, the rest one at a time to a
     * uniformly drawn target.
     */
    void divide(SpellAbility clause) {
        List<GameObject> targets = new ArrayList<>();
        clause.getTargets().getTargetEntities().forEach(targets::add);
        if (targets.isEmpty()) {
            return;
        }
        int[] shares = new int[targets.size()];
        int left = dividedAmount(clause);
        for (int i = 0; i < shares.length && left > 0; i++, left--) {
            shares[i] = 1;
        }
        while (left-- > 0) {
            shares[random.nextInt(shares.length)]++;
        }
        for (int i = 0; i < shares.length; i++) {
            clause.addDividedAllocation(targets.get(i), shares[i]);
        }
    }

    // ── X ───────────────────────────────────────────────────────────────

    /**
     * Announce an X, uniform between the smallest legal value and the most the
     * payer's available mana can reach (024 FR-022c).
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
        ability.setXManaCostPaid(max < min ? min : between(min, max));
    }

    // ── draws ───────────────────────────────────────────────────────────

    /** An integer uniform in [{@code min}, {@code max}]; one draw from the source. */
    private int between(int min, int max) {
        return min + random.nextInt(max - min + 1);
    }

    /**
     * {@code count} distinct elements of {@code items}, uniformly, as the
     * first {@code count} of a shuffled copy; {@code items} itself is left in
     * its order.
     *
     * <p>Shared with the random seat's cost payment so that every uniform
     * subset either caller draws consumes the source the same way.
     */
    public static <T> List<T> drawDistinct(Collection<? extends T> items, int count,
                                           Random random) {
        List<T> shuffled = new ArrayList<>(items);
        Collections.shuffle(shuffled, random);
        return new ArrayList<>(shuffled.subList(0, count));
    }
}
