package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.rules.Aftermath;
import com.pricepredictor.connector.effects.directed.rules.AttractionVisit;
import com.pricepredictor.connector.effects.directed.rules.AuraSwap;
import com.pricepredictor.connector.effects.directed.rules.AuraSwapCantEnchant;
import com.pricepredictor.connector.effects.directed.rules.AuraSwapExchange;
import com.pricepredictor.connector.effects.directed.rules.BattleDefense;
import com.pricepredictor.connector.effects.directed.rules.Bolster;
import com.pricepredictor.connector.effects.directed.rules.CantEnterBattlefield;
import com.pricepredictor.connector.effects.directed.rules.CantGainLifeExchange;
import com.pricepredictor.connector.effects.directed.rules.CantLoseLifeExchange;
import com.pricepredictor.connector.effects.directed.rules.Cascade;
import com.pricepredictor.connector.effects.directed.rules.Champion;
import com.pricepredictor.connector.effects.directed.rules.ConniveWithoutDraw;
import com.pricepredictor.connector.effects.directed.rules.Dethrone;
import com.pricepredictor.connector.effects.directed.rules.DiscoverNothingFound;
import com.pricepredictor.connector.effects.directed.rules.DistinctTargets;
import com.pricepredictor.connector.effects.directed.rules.DrawFromEmptyLibrary;
import com.pricepredictor.connector.effects.directed.rules.Enlist;
import com.pricepredictor.connector.effects.directed.rules.ExploreEmptyLibrary;
import com.pricepredictor.connector.effects.directed.rules.FinalChapterResolves;
import com.pricepredictor.connector.effects.directed.rules.FinalityCounter;
import com.pricepredictor.connector.effects.directed.rules.GraveyardReplacement;
import com.pricepredictor.connector.effects.directed.rules.Intimidate;
import com.pricepredictor.connector.effects.directed.rules.LifeExchange;
import com.pricepredictor.connector.effects.directed.rules.LifeTotalWin;
import com.pricepredictor.connector.effects.directed.rules.LimitedSearch;
import com.pricepredictor.connector.effects.directed.rules.Madness;
import com.pricepredictor.connector.effects.directed.rules.Meld;
import com.pricepredictor.connector.effects.directed.rules.MorphFaceUp;
import com.pricepredictor.connector.effects.directed.rules.MutagenToken;
import com.pricepredictor.connector.effects.directed.rules.OptionalSearch;
import com.pricepredictor.connector.effects.directed.rules.PoisonLoss;
import com.pricepredictor.connector.effects.directed.rules.PoisonousInstances;
import com.pricepredictor.connector.effects.directed.rules.ReplacedSacrificeCost;
import com.pricepredictor.connector.effects.directed.rules.RestartGame;
import com.pricepredictor.connector.effects.directed.rules.SeveralExplore;
import com.pricepredictor.connector.effects.directed.rules.SharedKeywords;
import com.pricepredictor.connector.effects.directed.rules.ShuffleAndPlaceOnTop;
import com.pricepredictor.connector.effects.directed.rules.SimultaneousDamage;
import com.pricepredictor.connector.effects.directed.rules.SimultaneousShuffles;
import com.pricepredictor.connector.effects.directed.rules.SkippedUpkeepTrigger;
import com.pricepredictor.connector.effects.directed.rules.SmallLibraryShuffle;
import com.pricepredictor.connector.effects.directed.rules.SorcererRole;
import com.pricepredictor.connector.effects.directed.rules.StunCounter;
import com.pricepredictor.connector.effects.directed.rules.SuspendProhibited;
import com.pricepredictor.connector.effects.directed.rules.TextBoxExchange;
import com.pricepredictor.connector.effects.directed.rules.TokenMeldPartner;
import com.pricepredictor.connector.effects.directed.rules.ToughnessZero;
import com.pricepredictor.connector.effects.directed.rules.Transfigure;
import com.pricepredictor.connector.effects.directed.rules.Transmute;
import com.pricepredictor.connector.effects.directed.rules.TypeSettingAura;
import com.pricepredictor.connector.effects.directed.rules.Unearth;
import com.pricepredictor.connector.effects.directed.rules.UnmeldableStay;
import com.pricepredictor.connector.effects.directed.rules.VanishingInstances;
import com.pricepredictor.connector.effects.directed.rules.VirtuousRole;

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
     * Every registered rule, in the catalogue's rule order: one class per
     * rule in {@code directed/rules/}, each naming its scenario, its board
     * precondition and its variants. Together they cover the ten operations
     * (story 3, acceptance 3).
     */
    private static final DirectedRules REGISTERED = of(
            LifeTotalWin.rule(),
            PoisonLoss.rule(),
            RestartGame.rule(),
            SorcererRole.rule(),
            VirtuousRole.rule(),
            MutagenToken.rule(),
            ReplacedSacrificeCost.rule(),
            OptionalSearch.rule(),
            CantGainLifeExchange.rule(),
            CantLoseLifeExchange.rule(),
            DrawFromEmptyLibrary.rule(),
            StunCounter.rule(),
            BattleDefense.rule(),
            FinalityCounter.rule(),
            TypeSettingAura.rule(),
            GraveyardReplacement.rule(),
            MorphFaceUp.rule(),
            AttractionVisit.rule(),
            DistinctTargets.rule(),
            Champion.rule(),
            FinalChapterResolves.rule(),
            SkippedUpkeepTrigger.rule(),
            CantEnterBattlefield.rule(),
            AuraSwap.rule(),
            LifeExchange.rule(),
            TextBoxExchange.rule(),
            LimitedSearch.rule(),
            SmallLibraryShuffle.rule(),
            SimultaneousShuffles.rule(),
            ShuffleAndPlaceOnTop.rule(),
            Bolster.rule(),
            Meld.rule(),
            TokenMeldPartner.rule(),
            UnmeldableStay.rule(),
            ExploreEmptyLibrary.rule(),
            SeveralExplore.rule(),
            ConniveWithoutDraw.rule(),
            DiscoverNothingFound.rule(),
            SharedKeywords.rule(),
            Intimidate.rule(),
            Madness.rule(),
            Transmute.rule(),
            SuspendProhibited.rule(),
            VanishingInstances.rule(),
            AuraSwapExchange.rule(),
            AuraSwapCantEnchant.rule(),
            PoisonousInstances.rule(),
            Transfigure.rule(),
            Unearth.rule(),
            Cascade.rule(),
            Dethrone.rule(),
            Aftermath.rule(),
            Enlist.rule(),
            SimultaneousDamage.rule(),
            ToughnessZero.rule());

    private final Map<String, DirectedRule> rules;

    private DirectedRules(Map<String, DirectedRule> rules) {
        this.rules = Collections.unmodifiableMap(rules);
    }

    /** The rules the worker forks for. */
    public static DirectedRules registered() {
        return REGISTERED;
    }

    /**
     * A registry of {@code rules}, for the registered set and for a test.
     *
     * @throws IllegalArgumentException when two share a rule id
     */
    public static DirectedRules of(DirectedRule... rules) {
        Map<String, DirectedRule> byId = new LinkedHashMap<>();
        for (DirectedRule rule : rules) {
            if (byId.putIfAbsent(rule.rule(), rule) != null) {
                throw new IllegalArgumentException(rule.rule() + " is registered twice");
            }
        }
        return new DirectedRules(byId);
    }

    /** The rule registered under {@code id}, if any. */
    public Optional<DirectedRule> find(String id) {
        return Optional.ofNullable(rules.get(id));
    }

    /** The registered rule ids, in registration order. */
    public List<String> ids() {
        return List.copyOf(rules.keySet());
    }

    /** The registered rules, in registration order. */
    public Collection<DirectedRule> rules() {
        return rules.values();
    }
}
