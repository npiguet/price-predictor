package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.Map;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for titles: the monarch, the Ring's
 * temptation, the initiative.
 */
final class TitleApiRules {

    private TitleApiRules() {
    }

    static void titleRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.MONARCH_CHANGED,
                        (sa, host) -> host.getGame().getMonarch(), TitleApiRules::newMonarch),
                "BecomeMonarch");
        put(rules, Rule.shaped(EventType.RING_TEMPTS, null, TitleApiRules::tempted),
                "RingTemptsYou");
        put(rules, Rule.shaped(EventType.INITIATIVE_TAKEN, null, TitleApiRules::initiativeTaker),
                "TakeInitiative");
    }

    /**
     * monarch_changed: the new monarch, and only when the monarch genuinely
     * changed.
     *
     * <p>{@code GameAction.becomeMonarch} is a no-op (no state change, no
     * trigger) when the named player already is the monarch or {@code
     * canBecomeMonarch()} says no (a static ability, e.g. Crown of Vigor's
     * granted "monarch can't lose the crown" combined with a would-be usurper
     * who cannot become monarch some other way) -- replicating that condition
     * here would duplicate the engine's own classification, a second copy that
     * drifts. Reading the actual before/after monarch instead needs no such
     * duplication and is correct regardless of how many players {@code
     * BecomeMonarchEffect}'s loop iterates.
     *
     * @param before the monarch before the clause ran
     */
    private static EffectEvent newMonarch(
            EffectEvent event, SpellAbility sa, Card host, Player before) {
        Player after = host.getGame().getMonarch();
        if (after == null || after.equals(before)) {
            return null;
        }
        return event.subject(Refs.of(after));
    }

    /**
     * ring_tempts: the tempted player, always the activator --
     * {@code RingTemptsYouEffect.resolve()} reads {@code
     * sa.getActivatingPlayer()} unconditionally, never {@code
     * Defined$}/targeting.
     *
     * <p>The Ring-bearer is a separable second fact from the same clause
     * ({@code Card.setRingBearer}, read back post-resolution since it is only
     * known once the player's controller has chosen one) and is named as a
     * second subject when one was actually chosen -- null when the player
     * controls no creatures, per {@code chooseSingleEntityForEffect}'s own
     * contract on an empty candidate list.
     */
    private static EffectEvent tempted(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        Player p = sa.getActivatingPlayer();
        if (p == null) {
            return null;
        }
        event.subject(Refs.of(p));
        Card ringBearer = p.getRingBearer();
        if (ringBearer != null) {
            event.subject(Refs.of(ringBearer));
        }
        return event;
    }

    /**
     * initiative_taken: the player taking the initiative.
     *
     * <p>{@code GameAction.takeInitiative} fires its trigger unconditionally
     * -- "You can take the initiative even if you already have it" is in the
     * engine's own comment -- unlike {@code becomeMonarch}, which is a no-op
     * when the target already holds the title. So this is NOT a before/after
     * diff (that would wrongly suppress the already-has-it re-affirmation): it
     * reads the post-resolution holder and confirms they are one of this
     * clause's own targets, which is true whether the holder changed or was
     * reconfirmed.
     *
     * <p>Known limit (record-schema.md "initiative_taken can miss a
     * redirect"): a target who satisfies {@code isInGame()} but has also
     * already {@code hasLost()} triggers {@code GameAction.takeInitiative}'s
     * own redirect to the next player, which that method's own unconditional
     * fall-through then overwrites back to the original (lost) target before
     * this rule ever reads {@code game.getHasInitiative()} -- so the event
     * still fires, but can name the lost player instead of who the redirect
     * actually landed on. Not replicated here: doing so would mean
     * re-deriving {@code GameAction}'s own fall-through as a second copy of
     * it, which drifts the way any duplicated classification does.
     */
    private static EffectEvent initiativeTaker(
            EffectEvent event, SpellAbility sa, Card host, Object memo) {
        Player holder = host.getGame().getHasInitiative();
        if (holder == null || !ClauseTargets.affectedPlayers(sa).contains(holder)) {
            return null;
        }
        return event.subject(Refs.of(holder));
    }
}
