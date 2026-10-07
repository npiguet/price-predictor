package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.List;
import java.util.Objects;

/**
 * One mode of a modal ability, rendered as an {@code option} line.
 *
 * <p>Rendered exactly as the {@link TextAbility} it replaces, so the converted
 * text does not change. What it adds is the join the sidecar needs: which
 * charm the mode belongs to and which position in that charm's
 * {@code Choices$} it occupies. Forge gives a mode no identity of its own — a
 * chosen mode resolves as a clone with no back-reference — so the position is
 * the only name a record and a sidecar can agree on, and it has to be carried
 * from the one place that knows it, the loop over the charm's choices.
 *
 * <p>A die-roll outcome is also an {@code option} line but is not a mode of
 * anything; it stays a {@link TextAbility} and keys to nothing.
 *
 * @param owner the trait that executes the charm where the caller knows it (a
 *              trigger whose {@code Execute$} is the charm), or null when the
 *              charm is the trait itself. Needed because Forge builds one
 *              {@code SpellAbility} per SVar <em>name</em> per card state, so
 *              two chapters naming one charm SVar share the charm object and
 *              only the trigger says which chapter a mode line belongs to.
 */
public record OptionAbility(
        String descriptionText,
        int modeIndex,
        SpellAbility charm,
        CardTraitBase owner
) implements Ability {

    public OptionAbility {
        Objects.requireNonNull(descriptionText, "descriptionText must not be null");
        if (descriptionText.isEmpty()) {
            throw new IllegalArgumentException("descriptionText must not be empty");
        }
        Objects.requireNonNull(charm, "charm must not be null");
    }

    @Override
    public AbilityType type() {
        return AbilityType.OPTION;
    }

    /** The mode's own ability: the {@code modeIndex}-th of the charm's choices. */
    public AbilitySub mode() {
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");
        return modeIndex < choices.size() ? choices.get(modeIndex) : null;
    }

    /** The SVar label the charm referenced this mode by, as the script wrote it. */
    public String label() {
        String choices = charm.getParam("Choices");
        if (choices == null) return "";
        String[] labels = choices.split(",");
        return modeIndex < labels.length ? labels[modeIndex].trim() : "";
    }

    /**
     * Whether {@code trait} is the printed line this mode belongs to.
     *
     * <p>The owner decides where one was recorded. Otherwise the charm is
     * climbed to its root ability: the trait itself for a modal spell, or the
     * trigger or replacement that executes the root. Both directions of the
     * trigger link are checked because {@code setOverridingAbility} points a
     * shared ability at the <em>last</em> trigger that claimed it.
     */
    public boolean belongsTo(CardTraitBase trait) {
        if (trait == null) return false;
        if (owner != null) return owner == trait;
        SpellAbility root = charm.getRootAbility();
        if (root == trait) return true;
        if (trait instanceof Trigger trigger) {
            return root.getTrigger() == trigger || trigger.getOverridingAbility() == root;
        }
        if (trait instanceof ReplacementEffect replacement) {
            return root.getReplacementEffect() == replacement
                    || replacement.getOverridingAbility() == root;
        }
        return false;
    }
}
