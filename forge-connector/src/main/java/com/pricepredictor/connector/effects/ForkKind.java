package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The kind of fork that wrote a record: its envelope's {@code fork_kind},
 * which a live record does not carry.
 *
 * <p>The wire names are also the names the worker's failure report, its
 * refusal tally and the dropped-event counts give each kind ({@code fork
 * probe body}, {@code protection:main1}), so one spelling serves them all.
 */
public enum ForkKind implements WireNamed {
    /** A forced resolution ({@link InterventionFork}). */
    INTERVENTION("intervention"),
    /** A damage-step probe's branch ({@link DamageStepProbe}). */
    PROBE("probe"),
    /** A protection fork played to the end of the turn ({@link ProtectionFork}). */
    PROTECTION("protection"),
    /** A directed fork played to the end of the turn ({@link DirectedFork}). */
    DIRECTED("directed");

    private final String wireName;

    ForkKind(String wireName) {
        this.wireName = wireName;
    }

    @Override
    public String wireName() {
        return wireName;
    }
}
