package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.KeywordRendering;
import forge.game.ability.ApiType;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.List;
import java.util.Objects;

/**
 * Chapter ability from a Saga. One ChapterAbility per trigger, with uppercased Roman prefix.
 */
public record ChapterAbility(String descriptionText, List<Ability> subAbilities) implements Ability {

    /** What opens each mode in Forge's own description of a charm. */
    private static final char MODE_BULLET = '\u2022';
    /** What ends a chapter's numerals ("I, II —"). */
    private static final char EM_DASH = '\u2014';

    public ChapterAbility {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    /** Convenience constructor: no sub-abilities. */
    public ChapterAbility(String descriptionText) {
        this(descriptionText, List.of());
    }

    @Override
    public AbilityType type() {
        return AbilityType.CHAPTER;
    }

    /**
     * The chapter lines, each with the triggers that printed it.
     *
     * <p>Forge generates one trigger per chapter, the chapters sharing an
     * ability grouped together, the first of a group primary and the rest
     * secondary ("I, II —" is one line from two triggers): a secondary trigger
     * belongs to the line its group's primary printed.
     *
     * @param oracle the face's Oracle text, where a chapter whose script
     *               states nothing finds its sentence
     */
    public static KeywordRendering fromKeyword(KeywordInterface ki, FaceOracle oracle) {
        KeywordRendering.Builder rendering = new KeywordRendering.Builder();
        Ability groupLine = null;
        for (var trigger : ki.getTriggers()) {
            if (!trigger.isSecondary()) {
                groupLine = chapter(trigger, oracle);
                if (groupLine != null) rendering.add(groupLine);
            }
            rendering.printed(trigger, groupLine);
        }
        return rendering.build();
    }

    /** The line {@code trigger} prints, or null if it prints none. */
    private static Ability chapter(Trigger trigger, FaceOracle oracle) {
        String trigDesc = trigger.getParam("TriggerDescription");
        if (trigDesc == null) return null;

        // If TriggerDescription is just a chapter header (nothing after the em-dash),
        // resolve the execute SVar's SpellDescription to fill chapter content.
        String stripped = AbilityDescription.stripReminderText(trigDesc).trim();

        // If TriggerDescription contains the ABILITY placeholder and execute is a Charm,
        // replace with charm header and attach OPTION sub-abilities.
        SpellAbility execute = trigger.getOverridingAbility();
        if (DescriptionMarkers.hasAbilityPlaceholder(stripped) && isCharm(execute)) {
            return charmChapter(stripped, execute, trigger);
        }

        if (isCharm(execute) && stripped.indexOf(MODE_BULLET) >= 0) {
            return modesChapter(stripped, execute, trigger);
        }

        String normalized = AbilityDescription.normalize(
                withOracleText(withEffectText(stripped, execute), oracle));
        if (normalized == null) return null;
        return new ChapterAbility(AbilityType.CHAPTER.formatDescription(normalized));
    }

    /**
     * A chapter whose description spells out its charm: Forge writes the
     * chapter's numerals, the charm's header and every mode into the one
     * description, so the line is the numerals and the charm's choice
     * sentence ({@link CharmHeader}), and the modes become the chapter's
     * options.
     */
    private static Ability modesChapter(String stripped, SpellAbility execute, Trigger trigger) {
        String numerals = stripped.substring(0, stripped.indexOf(EM_DASH) + 1);
        String normalized = AbilityDescription.normalize(
                (numerals + " " + CharmHeader.of(execute)).trim());
        if (normalized == null) return null;
        return new ChapterAbility(AbilityType.CHAPTER.formatDescription(normalized),
                CharmAbility.optionsFrom(execute, trigger));
    }

    private static boolean isCharm(SpellAbility execute) {
        return execute != null && execute.getApi() == ApiType.Charm;
    }

    /** A header-only chapter completed with its execute chain's text. */
    private static String withEffectText(String stripped, SpellAbility execute) {
        if (execute == null || !isHeaderOnly(stripped)) return stripped;
        String effectText = SpellAbilityUtils.chainText(execute);
        return effectText.isEmpty() ? stripped : stripped + " " + effectText;
    }

    /**
     * A header-only chapter whose chain states nothing completed with the
     * Oracle line opening with its numerals (Fall of the Thran's "II, III —"
     * effect has no {@code SpellDescription$}), as an undescribed instant's
     * spell takes its Oracle sentences.
     */
    private static String withOracleText(String chapter, FaceOracle oracle) {
        if (!isHeaderOnly(chapter)) return chapter;
        String sentence = oracle.lineAfter(chapter);
        return sentence == null ? chapter : chapter + " " + sentence;
    }

    /** A chapter whose ABILITY placeholder names a charm: its header and options. */
    private static Ability charmChapter(String stripped, SpellAbility execute, Trigger trigger) {
        // Owned by the trigger: two chapters naming one charm SVar
        // share the charm object, and only the trigger tells
        // chapter I's option lines from chapter II's.
        CharmAbility.Expanded expanded =
                CharmAbility.expandPlaceholder(stripped, execute, trigger);
        if (expanded.description() == null) return null;
        return new ChapterAbility(
                AbilityType.CHAPTER.formatDescription(expanded.description()),
                expanded.options());
    }

    /**
     * Returns true if the trigger description is only a chapter header (e.g. "I —" or "I, II —")
     * with no effect text after the em-dash.
     */
    private static boolean isHeaderOnly(String desc) {
        int dashIdx = desc.indexOf(EM_DASH);
        if (dashIdx < 0) return false;
        return desc.substring(dashIdx + 1).trim().isEmpty();
    }
}
