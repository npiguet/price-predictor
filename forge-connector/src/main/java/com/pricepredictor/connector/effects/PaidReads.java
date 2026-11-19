package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostPart;
import forge.game.cost.CostPartWithList;
import forge.game.cost.CostRemoveAnyCounter;
import forge.game.cost.CostRemoveCounter;
import forge.game.cost.CostRollDice;
import forge.game.keyword.Keyword;
import forge.game.replacement.ReplacementEffect;
import forge.game.staticability.StaticAbility;
import forge.game.spellability.SpellAbility;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Whether a line's resolution reads what its cost paid, which a forced line
 * never pays (finding 5.3): forced, it would resolve as no legal play does.
 * Forced Burnt Offering adds no mana, where a cast one always sacrifices a
 * creature and adds its mana value.
 *
 * <p>Read in the script's own parameters: every clause of the line's chain
 * ({@code SubAbility$}), and every SVar a parameter names, then the SVars
 * those name. A line reads what its cost paid where one of them
 * <ul>
 * <li>names a list its cost pays into, as a defined list or a count's
 * source: {@code Sacrificed$CardPower}, {@code ConditionDefined$ Sacrificed},
 * {@code Discarded$CardManaCost}, {@code Defined$ ExiledCards} (the keys
 * {@code CostPartWithList} files its payment under: {@code Sacrificed},
 * {@code Discarded}, {@code Exiled}, {@code Tapped}, {@code Revealed} and
 * the others, each with its {@code Cards} list);</li>
 * <li>counts the mana the line's own payment spent ({@code
 * Count$ManaColorsPaid}, {@code Count$Adamant}, {@code
 * Count$TotalManaSpent});</li>
 * <li>for a spell, whose cast is the forced one, counts the mana spent to
 * cast it ({@code Count$Converge}, {@code Count$CastTotalManaSpent}, {@code
 * Count$EachSpentToCast}, {@code Count$EachPhyrexianPaidWithLife}, {@code
 * ConditionManaSpent$}, {@code ManaSpent$}), or is a sunburst permanent's.</li>
 * </ul>
 * A spell's host also reads its cast mana outside the chain, on its
 * {@code K:etbCounter} line ({@code Converge}, {@code Adamant$}), its
 * {@code R:} and {@code S:} lines: their parameters, their overriding
 * abilities' and the SVars those name are read as well. A cost holding a
 * counter-removal part makes {@code CostCountersRemoved} a read, and a die
 * roll the SVar its result is set in.
 *
 * <p>A permanent's activated ability reading the mana spent to cast its host
 * reads a payment the live game made, and is forced. A description is no
 * read. A list the line's cost pays nothing into is empty live as well, so
 * naming it is no read either ({@code Defined$ Exiled} of a spell that pays
 * no exile).
 */
final class PaidReads {

    /** Counts of the mana the line's own payment spent. */
    private static final Pattern OWN_MANA =
            Pattern.compile("Count\\$(?:ManaColorsPaid|Adamant|TotalManaSpent)");

    /** Counts of the mana spent to cast the line's host. */
    private static final Pattern CAST_MANA = Pattern.compile(
            "Count\\$(?:Converge|CastTotalManaSpent|EachSpentToCast|EachPhyrexianPaidWithLife)");

    /** Parameters that test the mana spent to cast the line's host. */
    private static final Set<String> CAST_MANA_PARAMETERS =
            Set.of("ConditionManaSpent", "ConditionManaNotSpent", "ManaSpent", "Adamant");

    /** What ends a defined list's or a count source's name: {@code Sacrificed$}, {@code ExiledCards.Other}. */
    private static final Pattern NAME_END = Pattern.compile("[.$ &,/+]");

    /** A word of a parameter's value that may name an SVar. */
    private static final Pattern WORD = Pattern.compile("\\w+");

    /** One script parameter, {@code name$ value}; an SVar's expression has no name. */
    private record Parameter(String name, String value) {
    }

    private final Card host;
    /** Whether the line is a spell, whose cast is the forced one. */
    private final boolean spell;
    /** The lists the line's cost pays into. */
    private final Set<String> paidLists;
    /** The SVars the line's cost sets as it pays: the counters it removed, a die's result. */
    private final Set<String> costSVars;

    private PaidReads(SpellAbility root) {
        this.host = root.getHostCard();
        this.spell = root.isSpell();
        this.paidLists = paidLists(root.getPayCosts());
        this.costSVars = costSVars(root.getPayCosts());
    }

    /** Whether {@code line}'s chain or the SVars it names read what its cost paid. */
    static boolean readsWhatItsCostPaid(SpellAbility line) {
        SpellAbility root = line.getRootAbility();
        if (root.getHostCard() == null) {
            return false;
        }
        PaidReads reads = new PaidReads(root);
        return reads.sunburst() || reads.anyRead(root);
    }

    /** The keys {@code CostPartWithList} files {@code cost}'s payment under. */
    private static Set<String> paidLists(Cost cost) {
        Set<String> lists = new HashSet<>();
        if (cost == null) {
            return lists;
        }
        for (CostPart part : cost.getCostParts()) {
            if (part instanceof CostPartWithList withList) {
                lists.add(withList.getHashForLKIList());
                lists.add(withList.getHashForCardList());
            }
        }
        return lists;
    }

    /** The SVars {@code cost}'s counter-removal and die-roll parts set when paid. */
    private static Set<String> costSVars(Cost cost) {
        Set<String> names = new HashSet<>();
        if (cost == null) {
            return names;
        }
        for (CostPart part : cost.getCostParts()) {
            if (part instanceof CostRemoveCounter || part instanceof CostRemoveAnyCounter) {
                names.add("CostCountersRemoved");
            } else if (part instanceof CostRollDice roll) {
                names.add(roll.getResultSVar());
            }
        }
        return names;
    }

    /** Whether the line casts a sunburst permanent, whose counters are the colors spent on it. */
    private boolean sunburst() {
        return spell && host.hasKeyword(Keyword.SUNBURST);
    }

    /** Whether a parameter of the chain, or of an SVar it names, reads what the cost paid. */
    private boolean anyRead(SpellAbility root) {
        Deque<Parameter> pending = new ArrayDeque<>();
        addChain(pending, root);
        if (spell) {
            addHostTraits(pending);
        }
        Set<String> named = new HashSet<>();
        while (!pending.isEmpty()) {
            Parameter parameter = pending.poll();
            if (reads(parameter)) {
                return true;
            }
            pending.addAll(svarsNamedBy(parameter, named));
        }
        return false;
    }

    private static void addChain(Deque<Parameter> pending, SpellAbility first) {
        for (SpellAbility clause = first; clause != null; clause = clause.getSubAbility()) {
            clause.getMapParams().forEach((name, value) -> pending.add(new Parameter(name, value)));
        }
    }

    /** The parameters of the host's replacement and static abilities (an {@code etbCounter} is one), and of what a replacement does. */
    private void addHostTraits(Deque<Parameter> pending) {
        for (ReplacementEffect replacement : host.getReplacementEffects()) {
            replacement.getMapParams().forEach((name, value) -> pending.add(new Parameter(name, value)));
            addChain(pending, replacement.getOverridingAbility());
        }
        for (StaticAbility ability : host.getStaticAbilities()) {
            ability.getMapParams().forEach((name, value) -> pending.add(new Parameter(name, value)));
        }
    }

    private boolean reads(Parameter parameter) {
        if (describes(parameter.name())) {
            return false;
        }
        return readsAPaidList(parameter.value()) || readsACostSVar(parameter.value()) || OWN_MANA.matcher(parameter.value()).find()
                || spell && readsTheCastsMana(parameter);
    }

    /** A description or a prompt, which shows the player text and reads nothing. */
    private static boolean describes(String name) {
        return name.endsWith("Description") || name.endsWith("Desc")
                || name.equals("TgtPrompt") || name.equals("AnnounceTitle");
    }

    /** Whether {@code value} names a list the cost pays into: {@code Sacrificed$CardPower}, {@code Sacrificed}. */
    private boolean readsAPaidList(String value) {
        return paidLists.contains(NAME_END.split(value.trim(), 2)[0]);
    }

    /** Whether {@code value} names an SVar the line's cost sets: {@code SVar$CostCountersRemoved/Twice}, a die's {@code X}. */
    private boolean readsACostSVar(String value) {
        Matcher word = WORD.matcher(value);
        while (word.find()) {
            if (costSVars.contains(word.group())) {
                return true;
            }
        }
        return false;
    }

    private static boolean readsTheCastsMana(Parameter parameter) {
        return CAST_MANA_PARAMETERS.contains(parameter.name()) || CAST_MANA.matcher(parameter.value()).find();
    }

    /** The parameters of each SVar of the host {@code parameter}'s value names, not named before. */
    private List<Parameter> svarsNamedBy(Parameter parameter, Set<String> named) {
        List<Parameter> parameters = new ArrayList<>();
        if (describes(parameter.name())) {
            return parameters;
        }
        Matcher word = WORD.matcher(parameter.value());
        while (word.find()) {
            String name = word.group();
            if (host.hasSVar(name) && named.add(name)) {
                parameters.addAll(parametersOf(host.getSVar(name)));
            }
        }
        return parameters;
    }

    /**
     * An SVar's parameters: an ability's ({@code DB$ GainLife | ...}), each
     * {@code name$ value}; an expression ({@code Sacrificed$CardPower}) as one
     * value with no name.
     */
    private static List<Parameter> parametersOf(String svar) {
        if (!svar.contains("|") && !svar.matches("^(?:AB|DB|SP)\\$.*")) {
            return List.of(new Parameter("", svar));
        }
        List<Parameter> parameters = new ArrayList<>();
        for (String part : svar.split("\\|")) {
            int dollar = part.indexOf('$');
            if (dollar > 0) {
                parameters.add(new Parameter(part.substring(0, dollar).trim(), part.substring(dollar + 1).trim()));
            }
        }
        return parameters;
    }
}
