package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * A snapshot's {@code refs} block: targets, source and announced values for
 * the acting ability.
 *
 * <p>Rendered apart from the rest of the snapshot because the refs block is
 * the one part of it that is not read at the same moment as the rest: a
 * collector re-reads it at resolution and hands it to {@link Snapshot#withRefs}.
 */
final class RefsJson {

    private RefsJson() {
    }

    /** The refs block for {@code acting}, or the empty shape for null. */
    static String render(SpellAbility acting) {
        return render(acting, null);
    }

    /**
     * The refs block for {@code acting}, naming {@code sourceOverride} as its
     * source when there is no acting ability.
     *
     * @param sourceOverride the host card of an acting line that is not a spell
     *                       ability; used only when {@code acting} is null, so
     *                       an acting ability still names its own host
     */
    static String render(SpellAbility acting, Card sourceOverride) {
        List<String> targets = new ArrayList<>();
        String source = Refs.ofNullable(sourceOverride);
        Integer x = null;
        if (acting != null) {
            if (acting.getHostCard() != null) {
                source = Refs.of(acting.getHostCard());
            }
            if (acting.getTargets() != null) {
                for (var target : acting.getTargets()) {
                    Refs.ofAny(target).ifPresent(targets::add);
                }
            }
            x = acting.getXManaCostPaid();
            if (x == null) {
                x = announcedX(acting);
            }
        }
        return Json.object()
                .raw("targets", Json.stringArray(targets))
                .value("source", source)
                .raw("modes", modesJson(acting))
                .value("x", x)
                .raw("choices", choicesJson(acting))
                .render();
    }

    /**
     * X when it was not paid into the mana cost.
     *
     * <p>{@code getXManaCostPaid()} answers only for an X the caster paid as
     * mana, and that is the minority form. The common one is
     * {@code SVar:X:Count$...} with the parameter key {@code NumDmg} or
     * {@code Amount} holding the literal {@code X}: the ability has no {@code X}
     * parameter at all and the announced value lives only in the SVar. So an X
     * read from the mana payment alone is absent on exactly the cards whose X is
     * the interesting part of the effect.
     *
     * <p>Computed rather than read, because a {@code Count$} SVar is an
     * expression over the board and only the engine knows what it comes to. One
     * that cannot be answered from this record's moment is recorded as absent
     * rather than as zero -- a zero would say the caster announced nothing,
     * which is a different claim.
     */
    private static Integer announcedX(SpellAbility acting) {
        Card host = acting.getHostCard();
        if (host == null) {
            return null;
        }
        String amount = acting.getSVar("X");
        if (amount == null || amount.isEmpty()) {
            return null;
        }
        try {
            return AbilityUtils.calculateAmount(host, amount, acting);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /**
     * The modes chosen on a modal spell.
     *
     * <p>A charm's three modes are one card and three effects, and which was
     * chosen is nowhere else in the record: the ability key is the same line
     * either way.
     */
    private static String modesJson(SpellAbility acting) {
        List<String> modes = new ArrayList<>();
        if (acting != null && acting.getChosenList() != null) {
            // Forge keeps the chosen modes as the sub-abilities they select,
            // so the mode is named by the clause it turns on.
            for (var mode : acting.getChosenList()) {
                modes.add(mode.getApi() == null ? "?" : mode.getApi().name());
            }
        }
        return Json.stringArray(modes);
    }

    /**
     * Resolution-time choices the engine asked the controller to make.
     *
     * <p>A named card, a chosen colour, a chosen type or number — each turns
     * one line into a different effect, and the head reads them from
     * {@code [ACT]} alongside the announced values.
     */
    private static String choicesJson(SpellAbility acting) {
        Json.ObjectWriter choices = Json.object();
        if (acting == null || acting.getHostCard() == null) {
            return choices.render();
        }
        Card host = acting.getHostCard();
        addChoice(choices, "named_card", host.getNamedCard());
        addChoice(choices, "chosen_color", String.join("|", host.getChosenColors()));
        addChoice(choices, "chosen_type", host.getChosenType());
        if (host.hasChosenNumber()) {
            choices.value("chosen_number", host.getChosenNumber());
        }
        return choices.render();
    }

    private static void addChoice(Json.ObjectWriter into, String name, String value) {
        if (value != null && !value.isEmpty()) {
            into.value(name, value);
        }
    }
}
