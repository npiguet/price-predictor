package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a clause's event says once the clause's own effect has run on a real
 * board: the phase it skipped as the snapshot spells it, one effect created
 * per effect card it made, and the object it healed, handed over or
 * retargeted named as the event's subject.
 */
@ExtendWith(ForgeExtension.class)
class ClauseOutcomeEventsTest {

    /** The clause's rule observed, its own effect run, its events built: what the clause hooks bracket. */
    private static List<EffectEvent> resolved(SpellAbility clause) {
        ApiEvents.ClauseObservation<?> observed = ApiEvents.observe(clause, ApiEvents::ruleFor);
        clause.getApi().getSpellEffect().resolve(clause);
        return observed.finish();
    }

    /** A clause of {@code host}'s script, by its SVar name. */
    private static SpellAbility clauseOf(Card host, String svar) {
        return AbilityFactory.getAbility(host.getSVar(svar), host);
    }

    // ── phase_skipped ───────────────────────────────────────────────────

    /** Elfhame Sanctuary: {@code DB$ SkipPhase | Defined$ You | Step$ Draw | Duration$ EndOfTurn}. */
    @Test
    void aSkippedStepIsNamedAsTheSnapshotSpellsIt() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.UPKEEP);
        SpellAbility skip = clauseOf(board.battlefield("Elfhame Sanctuary", board.you), "DBSkipDraw");
        skip.setActivatingPlayer(board.you);

        List<EffectEvent> events = resolved(skip);

        assertEquals(1, events.size(), events.toString());
        assertEquals(EventType.PHASE_SKIPPED, events.get(0).type());
        assertEquals("draw", events.get(0).param(EventParam.PHASE));
        assertEquals(List.of(Refs.of(board.you)), events.get(0).subjects());
    }

    /** Fatespinner's {@code Phase$ Main} is both main phases, as Forge's replacement reads it. */
    @Test
    void aSkippedMainPhaseNamesBothMainPhases() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.UPKEEP);
        SpellAbility skip = clauseOf(board.battlefield("Fatespinner", board.you), "FatespinnerSkipMain");
        skip.setActivatingPlayer(board.you);
        skip.setTriggeringObject(AbilityKey.Player, board.opponent);

        List<EffectEvent> events = resolved(skip);

        assertEquals("main1,main2", events.get(0).param(EventParam.PHASE));
    }

    // ── continuous_effect_created ───────────────────────────────────────

    /**
     * Planeswalker's Mischief: {@code DB$ Effect | ... | ExileOnMoved$ Exile
     * | RememberObjects$ Remembered}. With nothing remembered (the revealed
     * card was no instant or sorcery) {@code EffectEffect} makes nothing.
     */
    @Test
    void anEffectClauseThatMadeNothingWritesNoEffect() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        SpellAbility effect = clauseOf(board.battlefield("Planeswalker's Mischief", board.you), "DBEffect");
        effect.setActivatingPlayer(board.you);

        List<EffectEvent> events = resolved(effect);

        assertTrue(board.game.getCardsIn(ZoneType.Command).isEmpty(), "the premise: nothing was made");
        assertEquals(List.of(), events);
    }

    /**
     * Raging River: {@code DB$ Effect | EffectOwner$ Remembered | Name$
     * Raging River Left | StaticAbilities$ DBCantBlock | ...}. With both
     * players remembered it makes one effect card for each.
     */
    @Test
    void anEffectClauseWithTwoOwnersWritesTwoEffects() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Card river = board.battlefield("Raging River", board.you);
        river.addRemembered(board.you);
        river.addRemembered(board.opponent);
        SpellAbility effect = clauseOf(river, "DBDefLeftEffect");
        effect.setActivatingPlayer(board.you);

        List<EffectEvent> events = resolved(effect);

        assertEquals(2, board.game.getCardsIn(ZoneType.Command).size(), "the premise: two were made");
        assertEquals(2, events.size(), events.toString());
        for (EffectEvent event : events) {
            assertEquals(EventType.CONTINUOUS_EFFECT_CREATED, event.type());
            assertEquals("DBCantBlock", event.param(EventParam.LAYERS));
        }
    }

    // ── damage_healed ───────────────────────────────────────────────────

    /**
     * Wolverine, Fierce Fighter's replacement heals the damage already on
     * him: {@code DB$ HealDamage | Defined$ ReplacedTarget}. The event names
     * him and the damage the clause removed.
     */
    @Test
    void aHealNamesThePermanentAndTheDamageRemoved() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card wolverine = board.battlefield("Wolverine, Fierce Fighter", board.you);
        wolverine.setDamage(2);
        SpellAbility heal = clauseOf(wolverine, "Healing");
        heal.setActivatingPlayer(board.you);
        heal.setReplacingObject(AbilityKey.Target, wolverine);

        List<EffectEvent> events = resolved(heal);

        assertEquals(0, wolverine.getDamage(), "the premise: the damage was removed");
        assertEquals(1, events.size(), events.toString());
        assertEquals(EventType.DAMAGE_HEALED, events.get(0).type());
        assertEquals(List.of(Refs.of(wolverine)), events.get(0).subjects());
        assertEquals(2, events.get(0).param(EventParam.AMOUNT));
    }

    /** A heal of an undamaged permanent removed nothing and writes nothing. */
    @Test
    void aHealThatRemovedNoDamageWritesNothing() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card wolverine = board.battlefield("Wolverine, Fierce Fighter", board.you);
        SpellAbility heal = clauseOf(wolverine, "Healing");
        heal.setActivatingPlayer(board.you);
        heal.setReplacingObject(AbilityKey.Target, wolverine);

        assertEquals(List.of(), resolved(heal));
    }

    // ── ownership_change ────────────────────────────────────────────────

    /** Darkpact: {@code SP$ GainOwnership | ValidTgts$ Card | TgtZone$ Ante}: the card is its subject. */
    @Test
    void anOwnershipChangeNamesTheCardHandedOver() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card theirs = board.hand("Grizzly Bears", board.opponent);
        SpellAbility gain = board.ability(board.hand("Darkpact", board.you), ApiType.GainOwnership);
        gain.setActivatingPlayer(board.you);
        gain.resetTargets();
        gain.getTargets().add(theirs);

        List<EffectEvent> events = resolved(gain);

        assertEquals(board.you, theirs.getOwner(), "the premise: the card changed hands");
        assertEquals(1, events.size(), events.toString());
        assertEquals(List.of(Refs.of(theirs)), events.get(0).subjects());
        assertEquals(Refs.of(board.you), events.get(0).param(EventParam.OWNER));
    }

    // ── targets_changed ─────────────────────────────────────────────────

    /**
     * Deflection, aimed at a Lightning Bolt on the stack: the event names the
     * Bolt by its card. Read without resolving, as the retarget's own choice
     * is the AI's and is not what this pins.
     */
    @Test
    void aRetargetNamesTheSpellItRetargeted() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        SpellAbility bolt = board.cast(board.card("Lightning Bolt", board.you), board.you, bears);
        SpellAbility deflection = board.ability(
                board.hand("Deflection", board.opponent), ApiType.ChangeTargets);
        deflection.setActivatingPlayer(board.opponent);
        deflection.resetTargets();
        deflection.getTargets().add(bolt);

        EffectEvent event = TestCards.onlyEvent(ApiEvents.observe(deflection, ApiEvents::ruleFor));

        assertEquals(EventType.TARGETS_CHANGED, event.type());
        assertEquals(List.of(Refs.of(bolt.getHostCard())), event.subjects());
        assertEquals(List.of(Refs.of(bears)), event.param(EventParam.TARGETS));
    }
}
