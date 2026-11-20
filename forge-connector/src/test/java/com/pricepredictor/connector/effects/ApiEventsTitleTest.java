package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The {@link TitleApiRules}: the monarch, the Ring's temptation, the
 * initiative.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTitleTest {

    private static final String BECOME_MONARCH = "SP$ BecomeMonarch | ValidTgts$ Player";
    private static final String TAKE_INITIATIVE = "SP$ TakeInitiative | ValidTgts$ Player";

    /** A clause and the one player it targets. */
    private record PlayerClause(SpellAbility sa, Player target) {
    }

    /** {@code script} built on Grizzly Bears, targeting a new player {@code name}. */
    private static PlayerClause playerClause(String script, String name, int playerId) {
        SpellAbility sa = AbilityFactory.getAbility(script, TestCards.build("Grizzly Bears"));
        Player target = new Player(name, TestCards.game(), playerId);
        sa.resetTargets();
        sa.getTargets().add(target);
        return new PlayerClause(sa, target);
    }

    /** The event {@code sa}'s rule reports when {@code resolution} runs while it is observed. */
    private static EffectEvent eventAcross(SpellAbility sa, Runnable resolution) {
        ApiEvents.ClauseObservation<?> observed = observe(sa);
        resolution.run();
        return TestCards.onlyEvent(observed);
    }

    /** {@code GameAction.becomeMonarch}, as {@code BecomeMonarchEffect.resolve()} calls it. */
    private static void becomeMonarch(Player player) {
        player.getGame().getAction().becomeMonarch(player, "M12");
    }

    /** {@code GameAction.takeInitiative}, as {@code TakeInitiativeEffect.resolve()} calls it. */
    private static void takeInitiative(Player player) {
        player.getGame().getAction().takeInitiative(player, "BRO");
    }

    // ── monarch_changed ────────────────────────────────────────────────────

    /**
     * {@code GameAction.becomeMonarch} directly, not a hand-rolled {@code
     * game.setMonarch(...)} -- the same public method {@code
     * BecomeMonarchEffect.resolve()} itself calls, guards included, so this
     * exercises the real production code path around the rule under test
     * rather than a state the engine never produces.
     */
    @Test
    void becomingMonarchNamesTheNewMonarch() {
        PlayerClause clause = playerClause(BECOME_MONARCH, "target", 93001);
        Player target = clause.target();

        EffectEvent event = eventAcross(clause.sa(), () -> becomeMonarch(target));

        assertEquals(EventType.MONARCH_CHANGED, event.type());
        assertEquals(List.of(Refs.of(target)), event.subjects());
    }

    /**
     * {@code becomeMonarch} is a real no-op (no state change, no trigger)
     * when the named player already is the monarch. The plausible wrong
     * implementation this falsifies is a rule that always names the clause's
     * own target instead of checking whether the monarchy actually changed.
     */
    @Test
    void reaffirmingTheSameMonarchNamesNoOne() {
        PlayerClause clause = playerClause(BECOME_MONARCH, "already-monarch", 93002);
        Player target = clause.target();
        becomeMonarch(target);

        EffectEvent event = eventAcross(clause.sa(), () -> becomeMonarch(target));

        assertNull(event, "the monarch did not change, so nothing should be reported");
    }

    // ── ring_tempts ────────────────────────────────────────────────────────

    /**
     * With no creature to choose, {@code RingTemptsYouEffect.resolve()} still
     * calls {@code setRingBearer(null)}, which {@code Player.setRingBearer}
     * makes a no-op (it returns early on null) -- so an existing Ring-bearer
     * survives a re-temptation and is reported again as the second subject.
     * This end state is therefore also that shape's. The plausible wrong
     * implementation this falsifies is one that reads the no-op as an
     * unconditional clear.
     */
    @Test
    void ringTemptsNamesTheActivatorAndTheChosenRingBearer() {
        SpellAbility sa = AbilityFactory.getAbility("DB$ RingTemptsYou", TestCards.build("Grizzly Bears"));
        Player activator = new Player("tempted", TestCards.game(), 93101);
        sa.setActivatingPlayer(activator);
        Card bearer = TestCards.build("Runeclaw Bear");
        activator.setRingBearer(bearer);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.RING_TEMPTS, event.type());
        assertEquals(
                List.of(Refs.of(activator), Refs.of(bearer)),
                event.subjects());
    }

    /**
     * {@code chooseSingleEntityForEffect} returns {@code null} on an empty
     * candidate list (no creatures to become the Ring-bearer), and {@code
     * RingTemptsYouEffect.resolve()} calls {@code setRingBearer(null)}
     * either way -- but {@code Player.setRingBearer} is {@code if
     * (bearer == null) return;}, a no-op, not an unconditional clear. This
     * fresh {@code Player} never had a Ring-bearer to begin with, so it
     * reports only the activator for a reason this test can actually
     * demonstrate -- {@code getRingBearer()} was null before {@code resolve()}
     * ran and stays null after a no-op call. It does NOT demonstrate "no
     * creatures clears an existing Ring-bearer", which is false; see {@link
     * #ringTemptsNamesTheActivatorAndTheChosenRingBearer} for the shape a
     * fresh player cannot reach.
     */
    @Test
    void ringTemptsWithNoRingBearerNamesOnlyTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility("DB$ RingTemptsYou", TestCards.build("Grizzly Bears"));
        Player activator = new Player("tempted-no-bearer", TestCards.game(), 93102);
        sa.setActivatingPlayer(activator);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.RING_TEMPTS, event.type());
        assertEquals(List.of(Refs.of(activator)), event.subjects());
    }

    // ── initiative_taken ─────────────────────────────────────────────────

    /**
     * {@code GameAction.takeInitiative} directly, the same public method
     * {@code TakeInitiativeEffect.resolve()} calls.
     */
    @Test
    void takingInitiativeNamesThePlayer() {
        PlayerClause clause = playerClause(TAKE_INITIATIVE, "initiative-taker", 93401);
        Player target = clause.target();

        takeInitiative(target);
        EffectEvent event = TestCards.onlyEvent(observe(clause.sa()));

        assertEquals(EventType.INITIATIVE_TAKEN, event.type());
        assertEquals(List.of(Refs.of(target)), event.subjects());
    }

    /**
     * Unlike {@code becomeMonarch}, {@code takeInitiative} fires its trigger
     * even when the named player already has the initiative -- "You can take
     * the initiative even if you already have it" is in the engine's own
     * comment. The plausible wrong implementation this falsifies is a
     * before/after diff copied from {@code BecomeMonarch}, which would wrongly
     * suppress this re-affirmation.
     */
    @Test
    void takingInitiativeAgainStillNamesThePlayer() {
        PlayerClause clause = playerClause(TAKE_INITIATIVE, "already-has-it", 93402);
        Player target = clause.target();
        takeInitiative(target);

        EffectEvent event = eventAcross(clause.sa(), () -> takeInitiative(target));

        assertEquals(EventType.INITIATIVE_TAKEN, event.type());
        assertEquals(List.of(Refs.of(target)), event.subjects());
    }

    /** No one processed by this clause ever took the initiative here. */
    @Test
    void noInitiativeTakenNamesNoOne() {
        PlayerClause clause = playerClause(TAKE_INITIATIVE, "never-took-it", 93403);
        Player target = clause.target();

        EffectEvent event = TestCards.onlyEvent(observe(clause.sa()));

        assertNull(event, "nobody targeted by this clause holds the initiative");
    }
}
