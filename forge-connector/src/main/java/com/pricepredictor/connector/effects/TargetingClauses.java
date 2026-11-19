package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.GameObject;
import forge.game.ability.ApiType;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.spellability.TargetChoices;
import forge.game.spellability.TargetRestrictions;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;

/**
 * The targeting clauses of a spell ability, what each may target and whether
 * each can reach its minimum: the one enumeration ({@link #candidatesOf}) and
 * the one chain walk ({@link #chainReachesMinimum}) the verdict, the decision
 * record, the random seat and the forks share.
 *
 * <p>A clause is one link of a sub-ability chain. A charm's own chain is the
 * modes someone chose — the AI's last evaluation, or nobody's yet — not the
 * modes the rules offer, so each walk says which of the two it follows: the
 * verdict and the record follow the offered modes, the random seat's redraw
 * the chosen ones it is about to cast.
 *
 * <p>Read-only on the ability: counting a clause's candidates never chooses
 * one. It does read what is chosen, though: Forge's {@code canTarget} judges
 * a cross-target rule (unique targets, different controllers, a shared type,
 * a total mana value) against the targets already chosen on the clause and
 * its parents, so a reader that wants what the rules offer before any choice
 * counts inside {@link #withNoneChosen}.
 */
public final class TargetingClauses {

    private TargetingClauses() {
    }

    /**
     * Whether every targeting clause of the ability can reach its minimum.
     *
     * <p>The walk stops at a charm and asks whether enough of its offered
     * modes could be chosen, ignoring whatever chain was left on it.
     */
    public static boolean reachMinimum(SpellAbility ability) {
        for (SpellAbility clause = ability; clause != null; clause = clause.getSubAbility()) {
            if (!canReachMinimum(clause)) {
                return false;
            }
            if (clause.getApi() == ApiType.Charm) {
                return enoughChoosableModes(clause);
            }
        }
        return true;
    }

    /**
     * What the rules let one clause target, not what anyone picked: never
     * {@code getTargets()}, which is the chosen set.
     *
     * <p>Players and cards of the clause's target zone; for a clause whose
     * target zone is the stack (Counter, CopySpellAbility, ChangeTargets,
     * ControlSpell, {@code ChangeZone | Origin$ Stack}), players and the
     * stack items it can target ({@code canTarget}, which asks
     * {@code canTargetSpellAbility}), each as the {@code SpellAbility} on the
     * stack and never as its card. The effect reads spell targets only, so a
     * card there resolved into nothing, and {@code TargetType$ Spell} passes
     * an instant's or sorcery's card but never a permanent spell's. That is
     * how Forge's own target selection offers the stack. A clause that also
     * targets another zone (Venser's "spell or permanent") keeps Forge's card
     * candidates, which its effect reads as cards.
     */
    public static List<GameObject> candidatesOf(SpellAbility clause) {
        if (!clause.usesTargeting()) {
            return List.of();
        }
        TargetRestrictions restrictions = clause.getTargetRestrictions();
        if (!List.of(ZoneType.Stack).equals(restrictions.getZone())) {
            return new ArrayList<>(restrictions.getAllCandidates(clause));
        }
        List<GameObject> candidates = new ArrayList<>(restrictions.getAllCandidates(clause, true));
        for (SpellAbilityStackInstance item : clause.getHostCard().getGame().getStack()) {
            if (clause.canTarget(item.getSpellAbility())) {
                candidates.add(item.getSpellAbility());
            }
        }
        return candidates;
    }

    /**
     * What the rules let any clause the ability offers target, each entity
     * once, in the order the clauses first name them: the union a decision
     * record's {@code legal_targets} lists (FR-031). A stack item is listed
     * as its card.
     */
    public static List<GameEntity> legalTargets(SpellAbility ability) {
        Set<GameEntity> union = new LinkedHashSet<>();
        for (SpellAbility clause : offered(ability)) {
            candidatesOf(clause).stream().map(TargetingClauses::entityOf).forEach(union::add);
        }
        return new ArrayList<>(union);
    }

    private static GameEntity entityOf(GameObject candidate) {
        return candidate instanceof SpellAbility item ? item.getHostCard() : (GameEntity) candidate;
    }

    /**
     * Run {@code body} with no target chosen on any clause {@code ability}
     * offers, then give each clause back its own choices: the same
     * {@code TargetChoices} object, so a divided allocation returns with it.
     * The divided total is the ability's, not the choices', and is left as it
     * was.
     *
     * <p>For the decision a candidate is read at, taken before any target is
     * chosen: the AI's evaluation leaves its picks on a spell it did not cast,
     * and Forge's {@code canTarget} reads them and the parent clauses' for a
     * cross-target rule, so "two target creatures controlled by different
     * players" left with one of each side's would offer only those two. Never
     * around the random seat's own draw, which fills one clause after another
     * and needs the earlier clauses' picks.
     */
    public static <T> T withNoneChosen(SpellAbility ability, Supplier<T> body) {
        return withNoneChosenOn(offered(ability), body);
    }

    /**
     * {@link #withNoneChosen} over a charm's modes: each mode's whole chain has
     * its choices set aside, for the reachability count of the modes a draw
     * may pick from. Never around the draw of targets.
     */
    public static <T> T withNoneChosenOnModes(List<AbilitySub> modes, Supplier<T> body) {
        List<SpellAbility> clauses = new ArrayList<>();
        for (AbilitySub mode : modes) {
            for (SpellAbility link = mode; link != null; link = link.getSubAbility()) {
                clauses.add(link);
            }
        }
        return withNoneChosenOn(clauses, body);
    }

    private static <T> T withNoneChosenOn(List<SpellAbility> clauses, Supplier<T> body) {
        Map<SpellAbility, TargetChoices> chosen = new IdentityHashMap<>();
        for (SpellAbility clause : clauses) {
            if (clause.usesTargeting()) {
                chosen.putIfAbsent(clause, clause.getTargets());
                clause.resetTargets();
            }
        }
        try {
            return body.get();
        } finally {
            chosen.forEach(SpellAbility::setTargets);
        }
    }

    /**
     * Every clause the rules offer: the root chain up to a charm, then each
     * of the charm's offered modes with its own chain.
     */
    static List<SpellAbility> offered(SpellAbility ability) {
        List<SpellAbility> clauses = new ArrayList<>();
        for (SpellAbility clause = ability; clause != null; clause = clause.getSubAbility()) {
            clauses.add(clause);
            if (clause.getApi() == ApiType.Charm) {
                for (AbilitySub mode : clause.getAdditionalAbilityList("Choices")) {
                    for (SpellAbility link = mode; link != null; link = link.getSubAbility()) {
                        clauses.add(link);
                    }
                }
                break;
            }
        }
        return clauses;
    }

    /**
     * Whether every targeting clause of the chain from {@code chain} has as
     * many candidates as its minimum: each counted alone, so a constraint
     * across targets is left to the targets draw. A charm's mode is
     * choosable only when its whole chain passes (CR 700.2a).
     */
    public static boolean chainReachesMinimum(SpellAbility chain) {
        for (SpellAbility clause = chain; clause != null; clause = clause.getSubAbility()) {
            if (!canReachMinimum(clause)) {
                return false;
            }
        }
        return true;
    }

    /** Whether one clause is untargeted or has as many candidates as its minimum. */
    static boolean canReachMinimum(SpellAbility clause) {
        return !clause.usesTargeting() || candidatesOf(clause).size() >= clause.getMinTargets();
    }

    /**
     * Whether the charm's minimum number of modes can be met by modes whose
     * own targets are reachable (CR 700.2a: a mode with no legal target can't
     * be chosen).
     *
     * <p>Counted here rather than through {@code CharmEffect.makePossibleOptions},
     * which renumbers each offered mode's {@code CharmOrder} as it filters and
     * so would write to the game.
     */
    private static boolean enoughChoosableModes(SpellAbility charm) {
        int choosable = 0;
        for (AbilitySub mode : charm.getAdditionalAbilityList("Choices")) {
            if (chainReachesMinimum(mode)) {
                choosable++;
            }
        }
        int required = CharmModes.minimumModes(charm);
        if (charm.hasParam("CanRepeatModes")) {
            required = Math.min(required, 1);
        }
        return choosable >= required;
    }

    /**
     * The clauses of a play about to be cast that target: the root chain, or
     * for a charm the root plus the chosen modes' own chains — the clones the
     * AI chained are rebuilt from those originals at the cast.
     */
    public static List<SpellAbility> chosen(SpellAbility ability) {
        List<SpellAbility> clauses = new ArrayList<>();
        if (ability.getApi() == ApiType.Charm) {
            if (ability.usesTargeting()) {
                clauses.add(ability);
            }
            if (ability.getChosenList() != null) {
                for (AbilitySub mode : ability.getChosenList()) {
                    addTargeting(mode, clauses);
                }
            }
            return clauses;
        }
        addTargeting(ability, clauses);
        return clauses;
    }

    private static void addTargeting(SpellAbility chain, List<SpellAbility> into) {
        for (SpellAbility clause = chain; clause != null; clause = clause.getSubAbility()) {
            if (clause.usesTargeting()) {
                into.add(clause);
            }
        }
    }
}
