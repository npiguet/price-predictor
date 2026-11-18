package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.AbilityType;
import forge.game.keyword.Keyword;

import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/** The line kind a keyword renders as. */
final class KeywordTypes {

    private KeywordTypes() {}

    private static final Map<Keyword, AbilityType> COST_TYPES = buildCostTypes();

    private static Map<Keyword, AbilityType> buildCostTypes() {
        Map<Keyword, AbilityType> map = new EnumMap<>(Keyword.class);
        for (Keyword kw : EnumSet.of(
                Keyword.BESTOW, Keyword.BLITZ, Keyword.DASH, Keyword.DISTURB,
                Keyword.EMERGE, Keyword.ESCAPE, Keyword.EVOKE, Keyword.FLASHBACK,
                Keyword.FORETELL, Keyword.FREERUNNING, Keyword.MADNESS,
                Keyword.MEGAMORPH, Keyword.MORPH, Keyword.DISGUISE,
                Keyword.MORE_THAN_MEETS_THE_EYE, Keyword.MUTATE,
                Keyword.OVERLOAD, Keyword.PLOT, Keyword.PROTOTYPE,
                Keyword.PROWL, Keyword.RETRACE, Keyword.SNEAK,
                Keyword.SPECTACLE, Keyword.SURGE,
                Keyword.AWAKEN, Keyword.EMBALM, Keyword.ENCORE,
                Keyword.ETERNALIZE, Keyword.HARMONIZE, Keyword.IMPENDING,
                Keyword.JUMP_START, Keyword.MIRACLE, Keyword.NINJUTSU,
                Keyword.OFFERING, Keyword.SUSPEND, Keyword.UNEARTH)) {
            map.put(kw, AbilityType.ALTERNATE_COST);
        }
        for (Keyword kw : EnumSet.of(
                Keyword.BUYBACK, Keyword.ENTWINE, Keyword.KICKER,
                Keyword.MULTIKICKER, Keyword.REPLICATE, Keyword.SQUAD,
                Keyword.STRIVE, Keyword.CASUALTY, Keyword.OFFSPRING,
                Keyword.BARGAIN, Keyword.CONSPIRE, Keyword.ESCALATE,
                Keyword.MAYFLASHCOST, Keyword.SPLICE, Keyword.SPREE,
                Keyword.TIERED)) {
            map.put(kw, AbilityType.ADDITIONAL_COST);
        }
        for (Keyword kw : EnumSet.of(
                Keyword.AFFINITY, Keyword.ASSIST, Keyword.CONVOKE,
                Keyword.DELVE, Keyword.IMPROVISE, Keyword.UNDAUNTED)) {
            map.put(kw, AbilityType.COST_REDUCTION);
        }
        return map;
    }

    /** Keywords always classified as STATIC regardless of trigger presence. */
    private static final Set<Keyword> FORCE_STATIC = EnumSet.of(
            Keyword.WARD  // Ward is effectively a static ability in oracle, not triggered
    );

    /** Classify a keyword into an AbilityType. Cost keywords get a fixed type, others by trait presence. */
    static AbilityType classify(Keyword kw, boolean hasAbilities, boolean hasTriggers) {
        AbilityType costType = COST_TYPES.get(kw);
        if (costType != null) return costType;
        if (FORCE_STATIC.contains(kw)) return AbilityType.STATIC;
        if (hasAbilities) return AbilityType.ACTIVATED;
        if (hasTriggers) return AbilityType.TRIGGERED;
        return AbilityType.STATIC;
    }
}
