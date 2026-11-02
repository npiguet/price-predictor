package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The relation families of root spec § 10.3, in its table's order: what a
 * snapshot's {@code links[].type} spells (FR-026, FR-050).
 *
 * <p>{@code FAMILIES} in {@code relation_types.py} is the other half of this
 * vocabulary, in the same order; {@code LinkFamilyContractTest} reads it. A
 * link is written in the family's forward direction, {@code from} to
 * {@code to}, as each constant's Javadoc says; the reader takes the reverse
 * type from the other end.
 *
 * <p>Four families are never written, because the slots they join are the
 * reader's own layout (FR-050): {@link #LINE_OF}, {@link #LINE_OF_THE_SAME_CARD},
 * {@link #MODE_OF} and {@link #HAS_THE_SAME_NAME}.
 */
public enum LinkFamily implements WireNamed {
    /** A card's line to the card; built by the reader. */
    LINE_OF("line_of", false),
    /** One line of a card to another of the same card; built by the reader. */
    LINE_OF_THE_SAME_CARD("line_of_the_same_card", false),
    /** A charm's mode to its root line; built by the reader. */
    MODE_OF("mode_of", false),
    /** A mutated permanent's component card, or the card a melded permanent melded with, to the permanent. */
    MERGED_WITH("merged_with", true),
    /** An Aura, Equipment or Fortification to the permanent or player it is attached to. */
    ATTACHED_TO("attached_to", true),
    /** An attacker to the player, planeswalker or battle it attacks. */
    ATTACKS("attacks", true),
    /** A blocker to each attacker it blocks. */
    BLOCKS("blocks", true),
    /** Two attackers of one band, written once per pair, from the lower card id. */
    IN_BAND_WITH("in_band_with", true),
    /** A creature to an attacker it must block. */
    MUST_BLOCK("must_block", true),
    /** A goaded creature to a player who goaded it. */
    GOADED_BY("goaded_by", true),
    /** A stack object's line to each of its targets, with the target's index. */
    TARGETS("targets", true),
    /** A stack object's source card to the stack object's line. */
    SOURCE_OF("source_of", true),
    /** A triggered ability's line on the stack to each card or player that triggered it. */
    TRIGGERED_BY("triggered_by", true),
    /** A static line, or a timestamped floating effect, to each permanent it applies to. */
    AFFECTS("affects", true),
    /** A command-zone effect, floating effect or delayed trigger to the line that made it. */
    CREATED_BY("created_by", true),
    /** A pending delayed trigger to each card or player it holds. */
    PENDING_ON("pending_on", true),
    /** The line (or card) that exiled, imprinted or remembers an object, to the object. */
    LINKED_BY_TEXT("linked_by_text", true),
    /** An exiled card that acts through a permanent (cipher, haunt) to the permanent. */
    BOUND_TO("bound_to", true),
    /** A player to each card they control. */
    CONTROLS("controls", true),
    /** A player to each card they own. */
    OWNS("owns", true),
    /** A card whose control an effect gave away to the card whose effect gave it. */
    CONTROL_GRANTED_BY("control_granted_by", true),
    /** A copy to what it copies. */
    COPY_OF("copy_of", true),
    /** Two paired creatures, written once per pair, from the lower card id. */
    PAIRED_WITH("paired_with", true),
    /** A battle's protector to the battle. */
    PROTECTS("protects", true),
    /** The card that chose an object, or promised a gift, to the object. */
    CHOSE("chose", true),
    /** A card to each card or player it dealt damage to this turn. */
    DAMAGED_THIS_TURN("damaged_this_turn", true),
    /** A creature to each attacker it blocked this turn. */
    FOUGHT_THIS_TURN("fought_this_turn", true),
    /** A Vehicle or Mount to each creature that crewed or saddled it this turn. */
    CREWED_BY("crewed_by", true),
    /** Two slots whose names match; built by the reader. */
    HAS_THE_SAME_NAME("has_the_same_name", false),
    /** A dungeon to the player venturing into it. */
    IN_DUNGEON("in_dungeon", true),
    /** The pending event's subject as it last existed ({@code L<id>}) to the card now. */
    LAST_KNOWN_STATE_OF("last_known_state_of", true);

    private final String wireName;
    private final boolean collected;

    LinkFamily(String wireName, boolean collected) {
        this.wireName = wireName;
        this.collected = collected;
    }

    /** The {@code type} a link carries. */
    @Override
    public String wireName() {
        return wireName;
    }

    /** Whether the collector writes links of this family; the rest are the reader's. */
    public boolean collected() {
        return collected;
    }
}
