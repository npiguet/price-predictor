package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.NonBlankString;

import java.util.Optional;

/**
 * Named constants for Forge-internal sentinel values and string markers that the
 * connector must recognise and filter out when extracting oracle text.
 *
 * <p>These are NOT oracle-text strings. They are implementation details of the Forge
 * game engine that leak into description params (SpellDescription, StackDescription,
 * TriggerDescription, etc.) and must be excluded from any output seen by a player.
 *
 * <p>Centralising them here makes their forge-internal meaning explicit and ensures
 * that all filtering points stay in sync when Forge changes these values.
 */
public final class ForgeParams {

    private ForgeParams() {}

    // --- SpellAbility mode sentinels ---

    /**
     * SpellAbility {@code Mode$} value identifying a replacement effect that is also
     * registered in the card's replacement-effects list.
     * Spell abilities with this mode are skipped in {@code collectSpellAbilities()}
     * to avoid duplicating the ReplacementAbilityEntry emitted by
     * {@code collectStaticsAndReplacements()}.
     */
    public static final String ETB_REPLACEMENT_MODE = "ETBReplacement";

    // --- StackDescription sentinels ---

    /**
     * {@code StackDescription$} value meaning "don't show this ability on the stack."
     * When present, the StackDescription carries no oracle text and must be ignored.
     */
    public static final String NO_DISPLAY = "None";

    /**
     * {@code StackDescription$} value meaning "display the SpellDescription param instead."
     * This is a level-of-indirection marker, not literal oracle text.
     */
    public static final String STACK_DESC_REF = "SpellDescription";

    // --- Template / formatting tokens ---

    /**
     * Prefix found in {@code StackDescription$} strings built from replacement-effect
     * templates (e.g. {@code "REP draws a card"}). These are Forge rendering artifacts,
     * not oracle text, and must be rejected as description sources.
     */
    public static final String REP_PREFIX = "REP ";

    /**
     * Token found in {@code StackDescription$} strings that reference a player object
     * (e.g. {@code "{p:you}"}). These are Forge template expressions, not oracle text.
     */
    public static final String PLAYER_TOKEN = "{p:";

    /**
     * Substring found in display strings that refer to the "effect source" — a Forge
     * internal rendering concept. Never valid oracle text.
     */
    public static final String EFFECT_SOURCE = "effectsource";

    // --- Description absence sentinels ---

    /**
     * {@code TriggerDescription$} value used as a placeholder meaning "no description
     * provided." Triggers with this value carry no human-readable text and must be skipped.
     */
    public static final String BLANK_DESC = "Blank";

    // --- SVar structure markers ---

    // --- Oracle text cost prefix ---

    /**
     * Prefix in cost descriptions built from the standard "as an additional cost" oracle pattern.
     * Stripped before emitting the cost as an ADDITIONAL_COST ability so the output starts with
     * the cost itself rather than this preamble.
     */
    public static final String ADDITIONAL_COST_PREFIX = "as an additional cost to cast this spell, ";

    /** Strip {@link #ADDITIONAL_COST_PREFIX} from {@code text} if present; otherwise return {@code text} unchanged. */
    private static String stripAdditionalCostPrefix(String text) {
        return text.startsWith(ADDITIONAL_COST_PREFIX)
                ? text.substring(ADDITIONAL_COST_PREFIX.length())
                : text;
    }

    /**
     * Strip {@link #ADDITIONAL_COST_PREFIX} from {@code text} if present.
     * Returns empty if stripping results in a blank string.
     */
    public static Optional<NonBlankString> stripAdditionalCostPrefix(NonBlankString text) {
        String result = stripAdditionalCostPrefix(text.value());
        return NonBlankString.of(result);
    }

    /**
     * Key present in trigger SVars that fire only from a specific zone
     * (e.g. {@code TriggerZones$ Battlefield}).
     * Such triggers are also registered as top-level triggers on the card via
     * {@code createTraits()}, so their descriptions are already emitted by
     * {@code RulesParser.collectTriggers()}. Including them again from
     * {@code SpellEffect.collectEffectDescriptions()} would produce duplicates.
     */
    public static final String TRIGGER_ZONES_MARKER = "TriggerZones$";
}
