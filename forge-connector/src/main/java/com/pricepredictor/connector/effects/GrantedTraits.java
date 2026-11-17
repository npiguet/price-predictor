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

import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;

/**
 * The parameters through which a trait grants other traits, each naming the
 * SVars that hold them, as Forge reads them when the grant applies.
 *
 * <p>A static's {@code AddTrigger$}, {@code AddStaticAbility$}, {@code
 * AddReplacementEffect$} and {@code AddAbility$} list their SVars joined by
 * {@code " & "} ({@code StaticAbilityContinuous}); an Animate's {@code
 * Abilities$}, {@code Triggers$}, {@code Replacements$} and {@code
 * staticAbilities$}, an Effect's {@code Abilities$}, {@code Triggers$},
 * {@code StaticAbilities$} and {@code ReplacementEffects$}, and a copy's
 * {@code AddTriggers$}, {@code AddAbilities$}, {@code GainTextAbilities$} and
 * {@code AddStaticAbilities$} join theirs by commas. Forge builds the granted
 * trait from the SVar only when the grant applies, so it is built here the
 * same way, from the granting trait's own SVars.
 */
final class GrantedTraits {

    private GrantedTraits() {
    }

    /** How Forge parses the script an SVar of one parameter holds. */
    private enum Kind { TRIGGER, STATIC, REPLACEMENT, ABILITY }

    /** One granting parameter: what it grants, how it lists it, and on which traits. */
    private record Grant(Kind kind, String separator, Predicate<CardTraitBase> grantsOn) {
    }

    private static final String AND = " & ";
    private static final String COMMA = ",";

    private static final Predicate<CardTraitBase> STATIC =
            trait -> trait instanceof StaticAbility;
    private static final Predicate<CardTraitBase> ANIMATE =
            apiIn(Set.of(ApiType.Animate, ApiType.AnimateAll));
    private static final Predicate<CardTraitBase> EFFECT = apiIn(Set.of(ApiType.Effect));
    private static final Predicate<CardTraitBase> ANIMATE_OR_EFFECT = ANIMATE.or(EFFECT);
    private static final Predicate<CardTraitBase> ABILITY =
            trait -> trait instanceof SpellAbility;

    private static final Map<String, Grant> GRANTS = Map.ofEntries(
            Map.entry("AddTrigger", new Grant(Kind.TRIGGER, AND, STATIC)),
            Map.entry("AddStaticAbility", new Grant(Kind.STATIC, AND, STATIC)),
            Map.entry("AddReplacementEffect", new Grant(Kind.REPLACEMENT, AND, STATIC)),
            Map.entry("AddAbility", new Grant(Kind.ABILITY, AND, STATIC)),
            Map.entry("Abilities", new Grant(Kind.ABILITY, COMMA, ANIMATE_OR_EFFECT)),
            Map.entry("Triggers", new Grant(Kind.TRIGGER, COMMA, ANIMATE_OR_EFFECT)),
            Map.entry("Replacements", new Grant(Kind.REPLACEMENT, COMMA, ANIMATE)),
            Map.entry("staticAbilities", new Grant(Kind.STATIC, COMMA, ANIMATE)),
            Map.entry("StaticAbilities", new Grant(Kind.STATIC, COMMA, EFFECT)),
            Map.entry("ReplacementEffects", new Grant(Kind.REPLACEMENT, COMMA, EFFECT)),
            Map.entry("AddTriggers", new Grant(Kind.TRIGGER, COMMA, ABILITY)),
            Map.entry("AddAbilities", new Grant(Kind.ABILITY, COMMA, ABILITY)),
            Map.entry("GainTextAbilities", new Grant(Kind.ABILITY, COMMA, ABILITY)),
            Map.entry("AddStaticAbilities", new Grant(Kind.STATIC, COMMA, ABILITY)));

    private static Predicate<CardTraitBase> apiIn(Set<ApiType> apis) {
        return trait -> trait instanceof SpellAbility sa && apis.contains(sa.getApi());
    }

    /** Every parameter that can grant a trait. */
    static Set<String> parameters() {
        return GRANTS.keySet();
    }

    /**
     * The text between two SVar names {@code parameter} lists on {@code
     * trait}, or null where the parameter grants nothing on that trait (an
     * {@code Abilities$} on an API that reads it otherwise).
     */
    static String separator(String parameter, CardTraitBase trait) {
        Grant grant = GRANTS.get(parameter);
        return grant != null && grant.grantsOn().test(trait) ? grant.separator() : null;
    }

    /**
     * The trait the SVar {@code name} holds, built as {@code parameter} on
     * {@code granter} grants it, or null where the script does not define the
     * SVar or Forge cannot build what it holds.
     */
    static CardTraitBase build(String parameter, String name, CardTraitBase granter) {
        Grant grant = GRANTS.get(parameter);
        if (grant == null || !granter.hasSVar(name)) return null;
        try {
            return parse(grant.kind(), granter.getSVar(name), granter);
        } catch (RuntimeException e) {
            return null;
        }
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
