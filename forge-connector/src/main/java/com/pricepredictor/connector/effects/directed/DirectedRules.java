package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.rules.Aftermath;
import com.pricepredictor.connector.effects.directed.rules.AscendBlessing;
import com.pricepredictor.connector.effects.directed.rules.AuraSwap;
import com.pricepredictor.connector.effects.directed.rules.AuraSwapCantEnchant;
import com.pricepredictor.connector.effects.directed.rules.BestowIllegalTarget;
import com.pricepredictor.connector.effects.directed.rules.Bolster;
import com.pricepredictor.connector.effects.directed.rules.CantEnterBattlefield;
import com.pricepredictor.connector.effects.directed.rules.CantGainLifeExchange;
import com.pricepredictor.connector.effects.directed.rules.Champion;
import com.pricepredictor.connector.effects.directed.rules.ConniveWithoutDraw;
import com.pricepredictor.connector.effects.directed.rules.Dethrone;
import com.pricepredictor.connector.effects.directed.rules.DiscoverNothingFound;
import com.pricepredictor.connector.effects.directed.rules.DistinctTargets;
import com.pricepredictor.connector.effects.directed.rules.DrawFromEmptyLibrary;
import com.pricepredictor.connector.effects.directed.rules.ExploreEmptyLibrary;
import com.pricepredictor.connector.effects.directed.rules.FinalChapterResolves;
import com.pricepredictor.connector.effects.directed.rules.Improvise;
import com.pricepredictor.connector.effects.directed.rules.JumpStartFromHand;
import com.pricepredictor.connector.effects.directed.rules.LifeExchange;
import com.pricepredictor.connector.effects.directed.rules.LifeTotalWin;
import com.pricepredictor.connector.effects.directed.rules.ManaValueSearch;
import com.pricepredictor.connector.effects.directed.rules.Mayhem;
import com.pricepredictor.connector.effects.directed.rules.Meld;
import com.pricepredictor.connector.effects.directed.rules.MorphFaceUp;
import com.pricepredictor.connector.effects.directed.rules.MutagenToken;
import com.pricepredictor.connector.effects.directed.rules.NoDamageToAGonePlaneswalker;
import com.pricepredictor.connector.effects.directed.rules.OptionalSearch;
import com.pricepredictor.connector.effects.directed.rules.PartnerWithSearch;
import com.pricepredictor.connector.effects.directed.rules.PlottedFromHand;
import com.pricepredictor.connector.effects.directed.rules.PoisonLoss;
import com.pricepredictor.connector.effects.directed.rules.Prototype;
import com.pricepredictor.connector.effects.directed.rules.SearchUnderMindcensor;
import com.pricepredictor.connector.effects.directed.rules.SeveralExplore;
import com.pricepredictor.connector.effects.directed.rules.SharedKeywords;
import com.pricepredictor.connector.effects.directed.rules.SimultaneousDamage;
import com.pricepredictor.connector.effects.directed.rules.SimultaneousLoss;
import com.pricepredictor.connector.effects.directed.rules.SkulkBlocker;
import com.pricepredictor.connector.effects.directed.rules.SmallLibraryShuffle;
import com.pricepredictor.connector.effects.directed.rules.ToughnessZero;
import com.pricepredictor.connector.effects.directed.rules.TrampleOverPlaneswalkers;
import com.pricepredictor.connector.effects.directed.rules.TypeSettingAura;
import com.pricepredictor.connector.effects.directed.rules.Unearth;
import com.pricepredictor.connector.effects.directed.rules.UnleashCantBlock;
import com.pricepredictor.connector.effects.directed.rules.VirtuousRole;
import com.pricepredictor.connector.effects.directed.rules.WebSlinging;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The rules a directed fork can be taken for, by rule id (D1).
 *
 * <p>The registry lives in Java because a rule's precondition and its edits
 * are Java code; the supervisor learns which rules it holds by running
 * {@code DirectedRulesMain} once at startup (FR-014). A rule listed in
 * {@code --directed-fork-rules} and absent here exits the run before any
 * game.
 *
 * <p>Not {@code VariantRegistry}: that holds a set of card names, where a rule
 * id here maps to behaviour.
 */
public final class DirectedRules {

    /**
     * Every registration, in the catalogue's order of its first rule: one
     * class per registration in {@code directed/rules/}, or one class
     * parameterised for two ({@link ManaValueSearch}), each naming its rules'
     * scenarios, its board precondition and its variants. Rules sharing a
     * board, a binding and an edit are one registration: the three meld
     * rules ({@link Meld}), Aura swap's two ({@link AuraSwap}). Together they
     * cover the ten operations (story 3, acceptance 3).
     */
    private static final DirectedRules REGISTERED = of(
            LifeTotalWin.rule(),
            PoisonLoss.rule(),
            SimultaneousLoss.rule(),
            VirtuousRole.rule(),
            MutagenToken.rule(),
            OptionalSearch.rule(),
            CantGainLifeExchange.rule(),
            TypeSettingAura.rule(),
            MorphFaceUp.rule(),
            SimultaneousDamage.rule(),
            DistinctTargets.rule(),
            Champion.rule(),
            FinalChapterResolves.rule(),
            CantEnterBattlefield.rule(),
            AuraSwap.rule(),
            LifeExchange.rule(),
            SearchUnderMindcensor.rule(),
            SmallLibraryShuffle.rule(),
            Bolster.rule(),
            Meld.rule(),
            ExploreEmptyLibrary.rule(),
            SeveralExplore.rule(),
            ConniveWithoutDraw.rule(),
            DiscoverNothingFound.rule(),
            SharedKeywords.rule(),
            TrampleOverPlaneswalkers.rule(),
            NoDamageToAGonePlaneswalker.rule(),
            ManaValueSearch.transmute(),
            AuraSwapCantEnchant.rule(),
            ManaValueSearch.transfigure(),
            Unearth.rule(),
            UnleashCantBlock.rule(),
            BestowIllegalTarget.rule(),
            Dethrone.rule(),
            SkulkBlocker.rule(),
            PartnerWithSearch.rule(),
            Improvise.rule(),
            Aftermath.rule(),
            Mayhem.rule(),
            AscendBlessing.rule(),
            JumpStartFromHand.rule(),
            PlottedFromHand.rule(),
            WebSlinging.rule(),
            DrawFromEmptyLibrary.rule(),
            ToughnessZero.rule(),
            Prototype.rule());

    /** Each rule id to its registration: a registration of several rules under each. */
    private final Map<String, DirectedRule> rules;
    private final List<DirectedRule> registrations;

    private DirectedRules(Map<String, DirectedRule> rules, List<DirectedRule> registrations) {
        this.rules = Collections.unmodifiableMap(rules);
        this.registrations = List.copyOf(registrations);
    }

    /** The rules the worker forks for. */
    public static DirectedRules registered() {
        return REGISTERED;
    }

    /**
     * A registry of {@code rules}, for the registered set and for a test.
     *
     * @throws IllegalArgumentException when two registrations name one rule id
     */
    public static DirectedRules of(DirectedRule... rules) {
        Map<String, DirectedRule> byId = new LinkedHashMap<>();
        for (DirectedRule rule : rules) {
            for (String id : rule.rules()) {
                if (byId.putIfAbsent(id, rule) != null) {
                    throw new IllegalArgumentException(id + " is registered twice");
                }
            }
        }
        return new DirectedRules(byId, List.of(rules));
    }

    /** The registration naming the rule {@code id}, if any. */
    public Optional<DirectedRule> find(String id) {
        return Optional.ofNullable(rules.get(id));
    }

    /** Every registered rule id, a registration of several rules giving each, in registration order. */
    public List<String> ids() {
        return List.copyOf(rules.keySet());
    }

    /** The registrations, in registration order, each once however many rules it names. */
    public Collection<DirectedRule> rules() {
        return registrations;
    }
}
