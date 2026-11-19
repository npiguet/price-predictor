package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.zone.ZoneType;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * When a damage step is forked for a probe, and which real record each
 * branch mirrors.
 *
 * <p>Gate 2 perturbs the keyword <b>model-side</b> and asks whether the
 * prediction moves the right way. A probe answers the same question against
 * the engine: the same combat, actually re-run without the keyword, so the
 * model's response has a ground truth to be checked against. Every collecting
 * supervisor names all eight damage-step keywords unless a run narrows
 * {@code --probe-keywords}; a worker started without the property probes
 * none.
 */
final class ProbeScheduler {

    private final Game game;
    private final String gameId;
    private final CollectionCaps caps;
    private final DamageStepProbe forks;
    /** Probe branches waiting for the combat record they mirror. */
    private final ProbeQueue heldProbes = new ProbeQueue();

    ProbeScheduler(Game game, String gameId, CollectionCaps caps, DamageStepProbe forks) {
        this.game = game;
        this.gameId = gameId;
        this.caps = caps;
        this.forks = forks;
    }

    /**
     * Fork this damage step for each probed keyword a combatant carries.
     *
     * <p>The branches are held rather than written. Each mirrors the real combat
     * record, which does not exist until the step this ran ahead of has
     * finished.
     */
    void collect() {
        if (!caps.probesEnabled()) {
            return;
        }
        DamageStep step = DamageStep.of(game.getPhaseHandler());
        if (step == null) {
            return;
        }
        // Before anything else: a branch of a step left behind has nothing
        // to mirror whether or not this step is probed.
        discardStaleProbes(step);
        boolean firstStrike = step == DamageStep.FIRST_STRIKE;
        if (firstStrike && !anyFirstStriker()) {
            // Forge always enters the first-strike step and skips it when no
            // combatant has first or double strike, so the phase event alone is
            // no evidence that a step will happen. Probing anyway would spend
            // budget on forks that assign no damage, and since the real game
            // writes no combat record for a step that assigns nothing, those
            // branches would be completed against the *regular* step's record.
            return;
        }
        // One keyword per game rather than whichever the battlefield order
        // reaches first. With eight probed keywords and a budget of two,
        // battlefield order would probe the same few keywords game after game
        // and leave the rest unprobed -- and gate 2 is a per-keyword decision,
        // so an unprobed keyword gets no check whatever.
        Set<String> tried = new HashSet<>();
        if (probeCombatants(new ProbePass(probeFocus(), step, tried)) == 0) {
            // The focus is a rotation, not a restriction: a game whose combat
            // carries none of it still spends its budget rather than saving it
            // for a keyword that is not there. It passes over what the focused
            // pass tried: a fork discarded there would be taken again on the
            // same creature and keyword, and spend the budget twice on it.
            probeCombatants(new ProbePass(null, step, tried));
        }
    }

    /**
     * Hold a branch until the record it mirrors exists.
     *
     * <p>The one place a branch enters the queue, which is also what lets the
     * pairing rules below be exercised without a live combat to fork.
     */
    void hold(DamageStepProbe.HeldProbe held) {
        heldProbes.hold(held);
    }

    /** This game's turn in the probe-keyword rotation, or null when none. */
    private String probeFocus() {
        List<String> keywords = caps.probeKeywords();
        if (keywords.isEmpty()) {
            return null;
        }
        return keywords.get(
                Math.floorMod(String.valueOf(gameId).hashCode(), keywords.size()));
    }

    /**
     * One pass over the combatants: the keyword it spends on (null for any
     * probed one), the step, and the creature and keyword pairs this step's
     * passes have already forked.
     */
    private record ProbePass(String focus, DamageStep step, Set<String> tried) {

        /** Whether this pass forks {@code card} for {@code normalized}, the flag's checks aside. */
        boolean admits(Card card, String normalized) {
            if (focus != null && !focus.equals(normalized)) {
                return false;
            }
            if (step == DamageStep.REGULAR && FIRST_STRIKE.equals(normalized)) {
                return false;
            }
            return !tried.contains(triedKey(card, normalized));
        }

        /** Mark {@code card} and {@code normalized} forked at this step. */
        void tried(Card card, String normalized) {
            tried.add(triedKey(card, normalized));
        }

        private static String triedKey(Card card, String normalized) {
            return card.getId() + " " + normalized;
        }
    }

    /**
     * Take a probe for each combatant carrying a probed keyword.
     *
     * @return how many branches were taken
     */
    private int probeCombatants(ProbePass pass) {
        int taken = 0;
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (inCombat(card)) {
                taken += probeCombatant(card, pass);
            }
        }
        return taken;
    }

    /** Take a probe for each keyword of {@code card} this pass forks. */
    private int probeCombatant(Card card, ProbePass pass) {
        int taken = 0;
        for (Map.Entry<String, String> keyword : probedKeywords(card, pass).entrySet()) {
            pass.tried(card, keyword.getKey());
            DamageStepProbe.HeldProbe held = forks.probe(keyword.getValue(), card, pass.step());
            if (held != null) {
                hold(held);
                taken++;
            }
        }
        return taken;
    }

    /**
     * The keywords of {@code card} this pass forks, by their normalized form,
     * each in the card's own spelling.
     *
     * <p>One per keyword a creature carries, however many spellings it has
     * of it (trample, and trample over planeswalkers): the probe strips them
     * all, so a second spelling's branch is the first's again. The flag names
     * keywords in the corpus's spelling (first_strike); the card carries
     * Forge's (First Strike). Matched on the normalized form and stripped by
     * the card's own, so neither side has to guess the other's.
     */
    private Map<String, String> probedKeywords(Card card, ProbePass pass) {
        Map<String, String> spellings = new LinkedHashMap<>();
        for (KeywordInterface keyword : card.getKeywords()) {
            String original = keyword.getOriginal();
            if (original != null && !original.isEmpty()) {
                spellings.putIfAbsent(KeywordEdit.normalize(original), original);
            }
        }
        spellings.keySet().removeIf(normalized -> !probed(card, normalized, pass));
        return spellings;
    }

    private boolean probed(Card card, String normalized, ProbePass pass) {
        if (!caps.probeKeywords().contains(normalized) || !pass.admits(card, normalized)) {
            return false;
        }
        return strikesIn(card, pass.step()) || ACTS_ON_DAMAGE_DEALT_TO_IT.contains(normalized);
    }

    /**
     * First strike, as the flag spells it. Never probed at the regular
     * step: a creature with it dealt its damage in the first-strike step and
     * deals none in this one with it or without it (702.7b-c), so the branch
     * would be the real step again.
     */
    static final String FIRST_STRIKE = "first_strike";

    /** Double strike, as the flag spells it. */
    static final String DOUBLE_STRIKE = "double_strike";

    /**
     * The probed keyword that acts on the damage its carrier is dealt rather
     * than on the damage it deals. Every other one (trample, deathtouch,
     * lifelink, wither, infect, first and double strike) changes nothing on
     * a combatant that deals no damage in the step, and its branch would be
     * the real step again: trample on a creature without first strike at the
     * first-strike step, or on a first striker at the regular step.
     */
    private static final Set<String> ACTS_ON_DAMAGE_DEALT_TO_IT = Set.of("indestructible");

    /**
     * Whether {@code card} deals combat damage in {@code step} (510.4): first
     * and double strikers in the first-strike step, double strikers and the
     * creatures without first strike in the regular one. A creature that
     * changed its first strike between the two steps is read by what it
     * carries now, where Forge reads whether it dealt first-strike damage.
     */
    static boolean strikesIn(Card card, DamageStep step) {
        if (step == DamageStep.FIRST_STRIKE) {
            return card.hasFirstStrike() || card.hasDoubleStrike();
        }
        return card.hasSecondStrike();
    }

    /** Is anything in this combat going to deal first-strike damage? */
    private boolean anyFirstStriker() {
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (inCombat(card) && strikesIn(card, DamageStep.FIRST_STRIKE)) {
                return true;
            }
        }
        return false;
    }

    /** Is this creature in the combat about to deal damage? */
    private boolean inCombat(Card card) {
        return CombatShape.isCombatant(game.getCombat(), card);
    }

    /**
     * Complete the branches this combat record is the counterfactual for.
     *
     * <p>The caller does not say which damage step it just wrote, so the oldest
     * held substep is taken to be the one: branches are held in the order the
     * steps happen, and a first-strike branch is therefore completed against the
     * first-strike record and not against the regular one that follows it.
     * Naming the substep outright is one parameter better and is what
     * {@link #writeHeldProbes(String, DamageStep)} is for.
     */
    void writeHeldProbes(String combatRecordId) {
        if (heldProbes.isEmpty()) {
            return;
        }
        writeHeldProbes(combatRecordId, heldProbes.oldestSubstep());
    }

    /**
     * Complete only the branches taken at this damage step.
     *
     * <p>A branch and the record it mirrors have to describe the same step. A
     * branch forked at the first-strike step — where nothing may have been
     * assigned — written against the regular step's record would make gate 2's
     * real-versus-fork difference compare a step that did not happen with one
     * that did.
     */
    void writeHeldProbes(String combatRecordId, DamageStep substep) {
        if (heldProbes.isEmpty()) {
            return;
        }
        for (DamageStepProbe.HeldProbe held : heldProbes.take(substep)) {
            forks.writeHeldProbe(held, combatRecordId);
        }
    }

    /**
     * Drop branches from a step that has been left behind.
     *
     * <p>A branch whose step produced no real record has nothing to mirror, and
     * writing it against the next record is worse than not writing it: the
     * fork's budget is already spent either way, and a mispaired counterfactual
     * is read as a difference the keyword caused.
     */
    int discardStaleProbes(DamageStep substep) {
        return heldProbes.discardOtherThan(substep);
    }
}
