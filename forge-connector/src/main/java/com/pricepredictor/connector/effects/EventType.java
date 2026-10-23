package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The outcome types an event names, as its {@code type} spells them.
 *
 * <p>The full vocabulary is owned by {@code EventType} in the Python side's
 * {@code event_schema.py}; these are the members this collector reaches, and
 * the completeness test on that side is what keeps the two in step, so a
 * spelling is changed on both sides or neither.
 *
 * <p>Types name <b>observable outcomes</b>, not the script API that produced
 * them: two effects that both move a card to the graveyard emit
 * {@code zone_change}, and the model learns one field group for it.
 */
public enum EventType implements WireNamed {
    ZONE_CHANGE("zone_change"),
    DESTROYED("destroyed"),
    SACRIFICED("sacrificed"),
    REGENERATED("regenerated"),
    PHASED("phased"),
    TOKEN_CREATED("token_created"),
    PERMANENT_COPIED("permanent_copied"),
    LIFE_CHANGE("life_change"),
    DAMAGE_DEALT("damage_dealt"),
    DAMAGE_PREVENTED("damage_prevented"),
    POISON_CHANGE("poison_change"),
    ENERGY_CHANGE("energy_change"),
    RADIATION_CHANGE("radiation_change"),
    COUNTER_CHANGE("counter_change"),
    PT_CHANGE("pt_change"),
    TAPPED("tapped"),
    UNTAPPED("untapped"),
    ATTACHED("attached"),
    UNATTACHED("unattached"),
    CONTROL_CHANGE("control_change"),
    FACE_CHANGE("face_change"),
    /** The generic outcome, for an engine mode the vocabulary has no member for. */
    STATE_FLAG_CHANGE("state_flag_change"),
    KEYWORD_CHANGE("keyword_change"),
    TYPE_CHANGE("type_change"),
    COLOR_CHANGE("color_change"),
    CARD_DRAWN("card_drawn"),
    CARD_DISCARDED("card_discarded"),
    CARD_MILLED("card_milled"),
    CARD_LOOKED_AT("card_looked_at"),
    CARD_REVEALED("card_revealed"),
    LIBRARY_REORDERED("library_reordered"),
    LIBRARY_SHUFFLED("library_shuffled"),
    MANA_PRODUCED("mana_produced"),
    MANA_LOST("mana_lost"),
    SPELL_CAST("spell_cast"),
    SPELL_COUNTERED("spell_countered"),
    SPELL_COPIED("spell_copied"),
    ATTACKERS_DECLARED("attackers_declared"),
    BLOCKERS_DECLARED("blockers_declared"),
    BECAME_BLOCKED("became_blocked"),
    COMBAT_ENDED("combat_ended"),
    COIN_FLIPPED("coin_flipped"),
    CLASH_RESOLVED("clash_resolved"),
    VOTE_TAKEN("vote_taken"),
    PILES_MADE("piles_made"),
    DUNGEON_VENTURED("dungeon_ventured"),
    DICE_ROLLED("dice_rolled"),
    DAY_NIGHT_CHANGED("day_night_changed"),
    PLAYER_WON("player_won"),
    PLAYER_LOST("player_lost"),
    SPEED_CHANGED("speed_changed"),
    TURN_ADDED("turn_added"),
    TURN_SKIPPED("turn_skipped"),
    PHASE_ADDED("phase_added"),
    PHASE_SKIPPED("phase_skipped"),
    X_CHANGED("x_changed"),
    OWNERSHIP_CHANGE("ownership_change"),
    CARD_MADE("card_made"),
    RESTRICTION_CHANGE("restriction_change"),
    CHOICE_MADE("choice_made"),
    DAMAGE_HEALED("damage_healed"),
    /**
     * What an {@code Animate}/{@code AnimateAll} clause was scripted to grant.
     *
     * <p>Carries the <b>scripted</b> text of the clause, not what the layer
     * system finally applied -- that is not knowable at the clause. The
     * applied result is recorded separately, in a {@code continuous} record's
     * contributions, so a reader of this event alone should not mistake it
     * for the applied result; the pair is complete only across both record
     * kinds.
     */
    ABILITY_CHANGE("ability_change"),
    /**
     * What an {@code Effect} clause was scripted to add as a continuous
     * effect.
     *
     * <p>Same limitation as {@link #ABILITY_CHANGE}: this is the scripted
     * {@code StaticAbilities$} text, not the applied result, which lives in a
     * {@code continuous} record's contributions instead.
     */
    CONTINUOUS_EFFECT_CREATED("continuous_effect_created"),
    TARGETS_CHANGED("targets_changed"),
    DELAYED_TRIGGER_CREATED("delayed_trigger_created"),
    /**
     * A strict subset of the {@code rewrite} record for the same replacement --
     * see the comment above {@code EventType.REPLACEMENT_APPLIED} in
     * {@code event_schema.py} for why the two must not be read as independent
     * evidence of the same replacement.
     */
    REPLACEMENT_APPLIED("replacement_applied"),
    MONARCH_CHANGED("monarch_changed"),
    RING_TEMPTS("ring_tempts"),
    REMOVED_FROM_COMBAT("removed_from_combat"),
    INITIATIVE_TAKEN("initiative_taken"),
    TEXT_CHANGE("text_change"),
    TURN_ENDED("turn_ended");

    private static final Map<String, EventType> BY_WIRE_NAME = Stream.of(values())
            .collect(Collectors.toUnmodifiableMap(EventType::wireName, Function.identity()));

    private final String wireName;

    EventType(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code type} an event carries. */
    @Override
    public String wireName() {
        return wireName;
    }

    /**
     * The member a wire name spells, or empty for a name this side does not
     * have -- the patched engine names an outcome's type as a plain string,
     * which nothing here compiles against.
     */
    public static Optional<EventType> fromWire(String wireName) {
        return wireName == null ? Optional.empty() : Optional.ofNullable(BY_WIRE_NAME.get(wireName));
    }
}
