package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.zone.ZoneType;

import java.util.List;

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
    private final ForkCollector forks;
    /** Probe branches waiting for the combat record they mirror. */
    private final ProbeQueue heldProbes = new ProbeQueue();

    ProbeScheduler(Game game, String gameId, CollectionCaps caps, ForkCollector forks) {
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
        boolean firstStrike = step == DamageStep.FIRST_STRIKE;
        if (firstStrike && !anyFirstStriker()) {
            // Forge always enters the first-strike step and skips it when no
            // combatant has first or double strike, so the phase event alone is
            // no evidence that a step will happen. Probing anyway spent half the
            // budget on forks that assigned no damage: every empty probe fork in
            // the sampled corpus was a first-strike-step fork, and every
            // first-strike-step fork was empty. Worse, the real game writes no
            // combat record for a step that assigns nothing, so those branches
            // survived and were completed against the *regular* step's record.
            return;
        }
        discardStaleProbes(step);
        // One keyword per game rather than whichever the battlefield order
        // reaches first. With eight probed keywords and a budget of two, the
        // first corpus probed lifelink and trample eight times each and
        // double strike, indestructible, wither and infect not at all -- and
        // gate 2 is a per-keyword decision, so an unprobed keyword gets no
        // check whatever.
        String focus = probeFocus();
        if (probeCombatants(focus, step) == 0) {
            // The focus is a rotation, not a restriction: a game whose combat
            // carries none of it still spends its budget rather than saving it
            // for a keyword that is not there.
            probeCombatants(null, step);
        }
    }

    /**
     * Hold a branch until the record it mirrors exists.
     *
     * <p>The one place a branch enters the queue, which is also what lets the
     * pairing rules below be exercised without a live combat to fork.
     */
    void hold(ForkCollector.HeldProbe held) {
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
     * Take a probe for each combatant carrying a probed keyword.
     *
     * @param focus the one keyword to spend on, or null for any probed one
     * @return how many branches were taken
     */
    private int probeCombatants(String focus, DamageStep step) {
        int taken = 0;
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (!inCombat(card)) {
                continue;
            }
            for (KeywordInterface keyword : card.getKeywords()) {
                String original = keyword.getOriginal();
                if (original == null || original.isEmpty()) {
                    continue;
                }
                // The flag names keywords in the corpus's spelling
                // (first_strike); the card carries Forge's (First Strike).
                // Matched on the normalized form and stripped by the card's own,
                // so neither side has to guess the other's.
                String normalized = normalizeKeyword(original);
                if (!caps.probeKeywords().contains(normalized)
                        || (focus != null && !focus.equals(normalized))) {
                    continue;
                }
                ForkCollector.HeldProbe held =
                        forks.probe(original, card, step);
                if (held != null) {
                    hold(held);
                    taken++;
                }
            }
        }
        return taken;
    }

    /** Is anything in this combat going to deal first-strike damage? */
    private boolean anyFirstStriker() {
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (inCombat(card) && (card.hasFirstStrike() || card.hasDoubleStrike())) {
                return true;
            }
        }
        return false;
    }

    /** Is this creature in the combat about to deal damage? */
    private boolean inCombat(Card card) {
        var combat = game.getCombat();
        return combat != null
                && (combat.isAttacking(card) || combat.isBlocking(card));
    }

    /**
     * Forge's spelling of a keyword in the one every reader uses.
     *
     * <p>Mirrors {@code effects.domain.state_snapshot.normalize_keyword}: the
     * parameter after {@code :} goes, spaces become underscores. Two spellings
     * of first strike is exactly the mismatch that made gate 2 report zero
     * observations of it.
     */
    static String normalizeKeyword(String original) {
        int parameter = original.indexOf(':');
        String base = parameter < 0 ? original : original.substring(0, parameter);
        return base.trim().toLowerCase(java.util.Locale.ROOT).replace(' ', '_');
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
     * <p>A branch and the record it mirrors have to describe the same step. In
     * the first corpus they did not: a branch forked at the first-strike step —
     * where nothing was ever assigned — outlived that step and was written
     * against the regular step's record, so gate 2's real-versus-fork
     * difference compared a step that did not happen with one that did.
     */
    void writeHeldProbes(String combatRecordId, DamageStep substep) {
        if (heldProbes.isEmpty()) {
            return;
        }
        for (ForkCollector.HeldProbe held : heldProbes.take(substep)) {
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
