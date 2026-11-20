package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for healed damage, granted abilities,
 * effects made and retargeted spells.
 */
final class HealGrantAndRetargetApiRules {

    private HealGrantAndRetargetApiRules() {
    }

    static void healGrantAndRetargetRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.each(EventType.DAMAGE_HEALED, HealGrantAndRetargetApiRules::damageBefore,
                        (sa, host, before) -> healed(before)),
                "HealDamage");
        put(rules, Rule.shaped(EventType.ABILITY_CHANGE, HealGrantAndRetargetApiRules::animated,
                        HealGrantAndRetargetApiRules::grantedAbilities),
                "Animate", "AnimateAll");
        put(rules, Rule.each(EventType.CONTINUOUS_EFFECT_CREATED, HealGrantAndRetargetApiRules::commandZone,
                        HealGrantAndRetargetApiRules::effectsMade),
                "Effect");
        put(rules, Rule.one(EventType.TARGETS_CHANGED, null, (sa, host, memo) -> retargeted(sa)),
                "ChangeTargets");
    }

    /**
     * One {@code continuous_effect_created} per effect card that reached the
     * command zone while the clause ran, so none for an {@code Effect} clause
     * that made nothing and one per owner for one that made several.
     *
     * <p>{@code EffectEffect.resolve} returns before making anything on an
     * invalid duration, an empty remembered list it was told to forget on a
     * move (Planeswalker's Mischief having exiled nothing), or no effect
     * owner left, and makes one effect card per {@code EffectOwner$} player
     * (Chandra, Roaring Flame's emblem, one per player dealt damage). Counting
     * the cards themselves sidesteps replicating either.
     */
    private static List<EffectEvent> effectsMade(SpellAbility sa, Card host, Set<Integer> before) {
        List<EffectEvent> events = new ArrayList<>();
        if (before == null) {
            return events;
        }
        String layers = sa.getParamOrDefault("StaticAbilities", "");
        for (Card card : host.getGame().getCardsIn(ZoneType.Command)) {
            if (card.isImmutable() && !before.contains(card.getId())) {
                events.add(new EffectEvent(EventType.CONTINUOUS_EFFECT_CREATED).param("layers", layers));
            }
        }
        return events;
    }

    /** The ids of every card in a command zone. */
    private static Set<Integer> commandZone(SpellAbility sa, Card host) {
        Set<Integer> ids = new HashSet<>();
        host.getGame().getCardsIn(ZoneType.Command).forEach(card -> ids.add(card.getId()));
        return ids;
    }

    /**
     * One {@code damage_healed} per permanent the clause removed damage from,
     * naming it, with the damage removed: the damage before the clause less
     * the damage after, never the damage before alone.
     */
    private static List<EffectEvent> healed(Map<Card, Integer> before) {
        List<EffectEvent> events = new ArrayList<>();
        if (before == null) {
            return events;
        }
        before.forEach((card, damage) -> {
            int removed = damage - card.getDamage();
            if (removed > 0) {
                events.add(new EffectEvent(EventType.DAMAGE_HEALED)
                        .subject(Refs.of(card))
                        .param("amount", removed));
            }
        });
        return events;
    }

    /**
     * Damage on each of the clause's affected cards, before it heals any of it away.
     *
     * <p>Read through {@link ClauseTargets#affectedCards} rather than off {@code
     * sa.getTargets().getTargetCards()}, the clause's own {@code
     * TargetChoices}. {@code HealDamageEffect.resolve()} finds its cards
     * through {@code
     * SpellAbilityEffect.getTargetCards(sa)}, which consults {@code
     * sa.getTargets()} only when {@code sa.usesTargeting()} is true, and
     * neither real {@code HealDamage} card in the cardsfolder declares any
     * targeting on the clause itself -- both use {@code Defined$
     * Replaced*}, resolved through a replacement effect's {@code
     * getReplacingObject}, which never touches {@code sa.getTargets()}. A
     * direct read would find the clause's own (always-empty, per {@code
     * SpellAbility.targetChosen}'s eager initialisation) {@code
     * TargetChoices} on every real firing and find nothing healed.
     */
    private static Map<Card, Integer> damageBefore(SpellAbility sa, Card host) {
        Map<Card, Integer> damage = new LinkedHashMap<>();
        for (Card card : ClauseTargets.affectedCards(sa)) {
            damage.put(card, card.getDamage());
        }
        return damage;
    }

    /**
     * The traits an animation grants, by their script names, and whether it
     * removes all others, on each card it animated.
     *
     * <p>{@code abilities} joins the four lists an animation grants traits
     * through -- {@code Abilities$}, {@code Triggers$}, {@code
     * Replacements$} and {@code staticAbilities$}, each a comma-separated
     * list of SVar names -- because the common animations grant a trigger or
     * a static rather than an activated ability; keywords are not among
     * them, being the keyword change's own event.
     */
    private static EffectEvent grantedAbilities(
            EffectEvent event, SpellAbility sa, Card host, List<Card> animated) {
        List<String> granted = new ArrayList<>();
        for (String list : GRANT_LISTS) {
            String names = sa.getParamOrDefault(list, "");
            for (String name : names.split(",")) {
                if (!name.isBlank()) {
                    granted.add(name.trim());
                }
            }
        }
        if (animated != null) {
            Game game = host.getGame();
            for (Card card : animated) {
                Card now = game.getCardState(card, null);
                if (now != null && now.isInPlay()) {
                    event.subject(Refs.of(now));
                }
            }
        }
        return event
                .param("abilities", String.join(",", granted))
                .param("removed", sa.hasParam("RemoveAllAbilities"));
    }

    /** The parameters an animation names granted traits in, as {@code AnimateEffectBase} reads them. */
    private static final List<String> GRANT_LISTS =
            List.of("Abilities", "Triggers", "Replacements", "staticAbilities");

    /**
     * The cards an animation is about to animate, read before it resolves.
     *
     * <p>{@code Animate} acts on {@code getCardsfromTargets} ({@link
     * ClauseTargets#cardsFromTargets}); {@code AnimateAll} on every card in
     * its {@code Zone$} (the battlefield by default) matching {@code
     * ValidCards$}, of the targeted or defined players where it names any,
     * as {@code AnimateAllEffect.resolve} filters them.
     */
    private static List<Card> animated(SpellAbility sa, Card host) {
        if (sa.getApi() != ApiType.AnimateAll) {
            return ClauseTargets.cardsFromTargets(sa);
        }
        List<ZoneType> zones = ZoneType.listValueOf(sa.getParamOrDefault("Zone", "Battlefield"));
        CardCollection cards = new CardCollection();
        if (sa.usesTargeting() || sa.hasParam("Defined")) {
            for (Player player : ClauseTargets.affectedPlayers(sa)) {
                cards.addAll(player.getCardsIn(zones));
            }
        } else {
            cards.addAll(host.getGame().getCardsIn(zones));
        }
        List<Card> valid = new ArrayList<>();
        AbilityUtils.filterListByType(cards, sa.getParamOrDefault("ValidCards", ""), sa)
                .forEach(valid::add);
        return valid;
    }

    /**
     * The retargeted spells, named by their cards, and what they point at now.
     *
     * <p>Read after the clause rather than reported by it, because a spell it
     * retargeted is still on the stack carrying its new {@code TargetChoices}
     * -- this is the one outcome in this group that survives on an object
     * other than the host.
     *
     * <p>Reads {@code getTargetEntities()} rather than only {@code
     * getTargetCards()}. Redirecting a burn spell at a <em>player</em> is the classic use of
     * this mechanic -- Deflection, Misdirection and Bolt Bend are all real
     * and all reach {@code ChangeTargetsEffect}'s default "choose any new
     * legal target" branch, which retargets at a player as readily as a
     * card. {@code getTargetCards()} filters to {@code Card} instances
     * only, so a {@code Player} target would be silently dropped and the
     * event would still fire with an empty {@code targets} list -- a channel
     * that looks wired while reporting nothing distinguishable from
     * "retargeted at nothing." Mixed-entity targets are read through
     * {@link Refs#ofAny}.
     */
    private static EffectEvent retargeted(SpellAbility sa) {
        EffectEvent event = new EffectEvent(EventType.TARGETS_CHANGED);
        List<String> targets = new ArrayList<>();
        if (sa.getTargets() != null) {
            for (SpellAbility changed : sa.getTargets().getTargetSpells()) {
                event.subject(Refs.of(changed.getHostCard()));
                addTargets(changed, targets);
            }
        }
        return event.param("targets", targets);
    }

    /** What {@code changed} targets now, as refs, onto {@code targets}. */
    private static void addTargets(SpellAbility changed, List<String> targets) {
        if (changed.getTargets() != null) {
            for (GameEntity target : changed.getTargets().getTargetEntities()) {
                Refs.ofAny(target).ifPresent(targets::add);
            }
        }
    }
}
