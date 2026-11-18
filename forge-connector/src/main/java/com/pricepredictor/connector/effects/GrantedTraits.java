package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.replacement.ReplacementHandler;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.TriggerHandler;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;

/**
 * The parameters naming SVars from which Forge builds a trait only when the
 * trait they are on applies or resolves, as Forge reads them.
 *
 * <p>The traits a trait grants: a static's {@code AddTrigger$}, {@code
 * AddStaticAbility$}, {@code AddReplacementEffect$}, {@code AddAbility$}
 * and {@code GainTextAbilities$} list their SVars joined by {@code " & "}
 * ({@code StaticAbilityContinuous}); an Animate's {@code Abilities$}, {@code
 * Triggers$}, {@code Replacements$} and {@code staticAbilities$}, an
 * Effect's {@code Abilities$}, {@code Triggers$}, {@code StaticAbilities$}
 * and {@code ReplacementEffects$}, and a copy's {@code AddTriggers$}, {@code
 * AddAbilities$}, {@code GainTextAbilities$} and {@code AddStaticAbilities$}
 * join theirs by commas.
 *
 * <p>The traits a trait sets up, one SVar each: an exert cost's reflexive
 * {@code Trigger$} ({@code StaticAbility.getPayingTrigSA}), the trigger a
 * mana ability's mana sets off when spent ({@code TriggersWhenSpent$}), a
 * prevention shield's {@code PreventionSubAbility$}, an extra turn's or
 * phase's delayed trigger and its effect ({@code ExtraTurnDelayedTrigger$},
 * {@code ExtraTurnDelayedTriggerExecute$}, {@code ExtraPhaseDelayedTrigger$}
 * and Forge's own spelling {@code ExtraPhaseDelayedTriggerExcute$}), and the
 * static a moved permanent gets ({@code StaticEffect$}, {@code GameAction}).
 *
 * <p>Forge builds each from its SVar only when the trait applies, so it is
 * built here the same way, from the naming trait's own SVars.
 */
final class GrantedTraits {

    private GrantedTraits() {
    }

    /** How Forge parses the script an SVar of one parameter holds. */
    private enum Kind { TRIGGER, STATIC, REPLACEMENT, ABILITY }

    /** How a parameter lists the SVar names it holds. */
    enum Listing {
        /** Names joined by {@code " & "}. */
        AND,
        /** Names joined by commas. */
        COMMA,
        /** The whole value is one name. */
        ONE;

        /** The text between two names. */
        String separator() {
            return switch (this) {
                case AND -> " & ";
                case COMMA -> ",";
                case ONE -> "";
            };
        }

        /** The names {@code value} lists. */
        String[] names(String value) {
            if (this == ONE) return new String[] {value};
            return value.split(Pattern.quote(separator()), -1);
        }
    }

    /** What one naming parameter names, how it lists it, and on which traits. */
    record Grant(Kind kind, Listing listing, Predicate<CardTraitBase> grantsOn) {

        /**
         * The trait the SVar {@code name} holds, built as this grant on
         * {@code granter} builds it, or null where the script does not define
         * the SVar or Forge cannot build what it holds.
         */
        CardTraitBase build(String name, CardTraitBase granter) {
            if (!granter.hasSVar(name)) return null;
            try {
                return parse(kind, granter.getSVar(name), granter);
            } catch (RuntimeException e) {
                return null;
            }
        }
    }

    private static final Predicate<CardTraitBase> STATIC =
            trait -> trait instanceof StaticAbility;
    private static final Predicate<CardTraitBase> ANIMATE =
            apiIn(Set.of(ApiType.Animate, ApiType.AnimateAll));
    private static final Predicate<CardTraitBase> EFFECT = apiIn(Set.of(ApiType.Effect));
    private static final Predicate<CardTraitBase> ANIMATE_OR_EFFECT = ANIMATE.or(EFFECT);
    private static final Predicate<CardTraitBase> ABILITY =
            trait -> trait instanceof SpellAbility;
    private static final Predicate<CardTraitBase> ADD_TURN = apiIn(Set.of(ApiType.AddTurn));
    private static final Predicate<CardTraitBase> ADD_PHASE = apiIn(Set.of(ApiType.AddPhase));

    /** Each parameter's grants: the first that applies to a trait is the one it makes. */
    private static final Map<String, List<Grant>> GRANTS = Map.ofEntries(
            Map.entry("AddTrigger", List.of(new Grant(Kind.TRIGGER, Listing.AND, STATIC))),
            Map.entry("AddStaticAbility", List.of(new Grant(Kind.STATIC, Listing.AND, STATIC))),
            Map.entry("AddReplacementEffect",
                    List.of(new Grant(Kind.REPLACEMENT, Listing.AND, STATIC))),
            Map.entry("AddAbility", List.of(new Grant(Kind.ABILITY, Listing.AND, STATIC))),
            Map.entry("Abilities",
                    List.of(new Grant(Kind.ABILITY, Listing.COMMA, ANIMATE_OR_EFFECT))),
            Map.entry("Triggers",
                    List.of(new Grant(Kind.TRIGGER, Listing.COMMA, ANIMATE_OR_EFFECT))),
            Map.entry("Replacements", List.of(new Grant(Kind.REPLACEMENT, Listing.COMMA, ANIMATE))),
            Map.entry("staticAbilities", List.of(new Grant(Kind.STATIC, Listing.COMMA, ANIMATE))),
            Map.entry("StaticAbilities", List.of(new Grant(Kind.STATIC, Listing.COMMA, EFFECT))),
            Map.entry("ReplacementEffects",
                    List.of(new Grant(Kind.REPLACEMENT, Listing.COMMA, EFFECT))),
            Map.entry("AddTriggers", List.of(new Grant(Kind.TRIGGER, Listing.COMMA, ABILITY))),
            Map.entry("AddAbilities", List.of(new Grant(Kind.ABILITY, Listing.COMMA, ABILITY))),
            Map.entry("GainTextAbilities", List.of(
                    new Grant(Kind.ABILITY, Listing.AND, STATIC),
                    new Grant(Kind.ABILITY, Listing.COMMA, ABILITY))),
            Map.entry("AddStaticAbilities", List.of(new Grant(Kind.STATIC, Listing.COMMA, ABILITY))),
            Map.entry("Trigger", List.of(new Grant(Kind.ABILITY, Listing.ONE, STATIC))),
            Map.entry("TriggersWhenSpent", List.of(new Grant(Kind.TRIGGER, Listing.ONE, ABILITY))),
            Map.entry("PreventionSubAbility",
                    List.of(new Grant(Kind.ABILITY, Listing.ONE, ABILITY))),
            Map.entry("StaticEffect", List.of(new Grant(Kind.STATIC, Listing.ONE, ABILITY))),
            Map.entry("ExtraTurnDelayedTrigger",
                    List.of(new Grant(Kind.TRIGGER, Listing.ONE, ADD_TURN))),
            Map.entry("ExtraTurnDelayedTriggerExecute",
                    List.of(new Grant(Kind.ABILITY, Listing.ONE, ADD_TURN))),
            Map.entry("ExtraPhaseDelayedTrigger",
                    List.of(new Grant(Kind.TRIGGER, Listing.ONE, ADD_PHASE))),
            Map.entry("ExtraPhaseDelayedTriggerExcute",
                    List.of(new Grant(Kind.ABILITY, Listing.ONE, ADD_PHASE))));

    private static Predicate<CardTraitBase> apiIn(Set<ApiType> apis) {
        return trait -> trait instanceof SpellAbility sa && apis.contains(sa.getApi());
    }

    /** Every parameter that can name a trait. */
    static Set<String> parameters() {
        return GRANTS.keySet();
    }

    /**
     * The grant {@code parameter} makes on {@code trait}, or null where it
     * names nothing on that trait (an {@code Abilities$} on an API that reads
     * it otherwise).
     */
    static Grant grantOf(String parameter, CardTraitBase trait) {
        return GRANTS.getOrDefault(parameter, List.of()).stream()
                .filter(grant -> grant.grantsOn().test(trait))
                .findFirst().orElse(null);
    }

    private static CardTraitBase parse(Kind kind, String script, CardTraitBase granter) {
        Card host = granter.getHostCard();
        return switch (kind) {
            case TRIGGER -> TriggerHandler.parseTrigger(script, host, false, granter);
            case STATIC -> StaticAbility.create(script, host, stateOf(granter, host), false);
            case REPLACEMENT -> ReplacementHandler.parseReplacement(script, host, false, granter);
            case ABILITY -> AbilityFactory.getAbility(script, host, granter);
        };
    }

    private static CardState stateOf(CardTraitBase granter, Card host) {
        CardState state = granter.getCardState();
        return state != null ? state : host.getCurrentState();
    }
}
