package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

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
        return renderNow(acting, sourceOverride, announcedX(acting), "null");
    }

    /**
     * The refs block for {@code acting} with an X read earlier: the cast's,
     * read once when its costs were paid (B3).
     */
    static String renderWithX(SpellAbility acting, Integer x) {
        return renderNow(acting, null, x, "null");
    }

    /**
     * One effect half's refs block, read at the resolution's end,
     * with the die results and coin flips the resolution's own events report
     * ({@code random}) and the X read before the resolution (at the cast for
     * a cast item, at the resolution's start otherwise: an X computed now would
     * count the board the resolution has already changed), with the half's own targets: the ones
     * its links held as its resolution started, after Forge's check for
     * illegal targets removed the illegal ones ({@code targets}), and the ones
     * they held before that check ({@code targets_at_cast}, null where they
     * are unknown).
     */
    static String renderHalf(
            SpellAbility acting, List<EffectEvent> resolved, Integer x,
            List<String> targets, List<String> targetsAtCast, List<String> onStack) {
        return render(acting, null, x, randomJson(resolved), targets, targetsAtCast, onStack);
    }

    /**
     * A refs block already rendered at the resolution's end, with its
     * {@code random} filled from the resolution's events; for a caller that
     * holds the rendered block rather than the ability.
     */
    static String withRandom(String refs, List<EffectEvent> resolved) {
        String unread = "\"random\":null}";
        if (refs == null || !refs.endsWith(unread)) {
            return refs;
        }
        return refs.substring(0, refs.length() - unread.length())
                + "\"random\":" + randomJson(resolved) + "}";
    }

    /**
     * A refs block already rendered, with its {@code targets} and {@code
     * targets_at_cast} replaced by one effect half's; for a caller that holds
     * the rendered block rather than the ability (a forced resolution's).
     */
    static String withTargets(
            String refs, List<String> targets, List<String> targetsAtCast, List<String> onStack) {
        String opening = "{\"targets\":";
        int rest = refs == null ? -1 : refs.indexOf(",\"source\":");
        if (rest < 0 || !refs.startsWith(opening)) {
            return refs;
        }
        return "{\"targets\":" + Json.stringArray(targets)
                + ",\"targets_at_cast\":"
                + (targetsAtCast == null ? "null" : Json.stringArray(targetsAtCast))
                + ",\"targets_on_stack\":" + Json.stringArray(onStack)
                + refs.substring(rest);
    }

    /**
     * The random results a resolution came to, taken as given like a
     * decision ({@code in.random_results}): every die's final value in the
     * order rolled, and the coin flips won and lost (heads counts as won on an
     * uncalled flip, as Forge's coin effects read it).
     */
    static String randomJson(List<EffectEvent> events) {
        List<Object> dice = new ArrayList<>();
        int won = 0;
        int lost = 0;
        for (EffectEvent event : events) {
            if (event.type() == EventType.DICE_ROLLED
                    && event.param(EventParam.RESULTS) instanceof Iterable<?> results) {
                for (Object die : results) {
                    if (die instanceof java.util.Map<?, ?> rolled && rolled.get("final") != null) {
                        dice.add(rolled.get("final"));
                    }
                }
            } else if (event.type() == EventType.COIN_FLIPPED
                    && event.param(EventParam.RESULTS) instanceof Iterable<?> results) {
                for (Object flip : results) {
                    String result = String.valueOf(flip);
                    if (result.equals("win") || result.equals("heads")) {
                        won++;
                    } else if (result.equals("loss") || result.equals("tails")) {
                        lost++;
                    }
                }
            }
        }
        return Json.object()
                .value("dice", dice)
                .raw("coin_flips", Json.object().value("won", won).value("lost", lost).render())
                .render();
    }

    /**
     * Every target the acting ability's chain holds now, root first, each
     * once: a charm's modes' and a {@code SubAbility$}'s as well as the
     * root's; a spell or ability on the stack by its card.
     */
    /** A refs block whose targets are the ones the acting ability's chain holds now. */
    private static String renderNow(
            SpellAbility acting, Card sourceOverride, Integer x, String random) {
        if (acting == null) {
            return render(null, sourceOverride, x, random, List.of(), null, List.of());
        }
        ChainTargets now = ChainTargets.now(acting);
        List<String> targets = now.all();
        return render(acting, sourceOverride, x, random, targets, null,
                ChainTargets.onStackOf(targets, now));
    }

    private static String render(
            SpellAbility acting, Card sourceOverride, Integer x, String random,
            List<String> targets, List<String> targetsAtCast, List<String> onStack) {
        String source = Refs.ofNullable(sourceOverride);
        if (acting != null && acting.getHostCard() != null) {
            source = Refs.of(acting.getHostCard());
        }
        return Json.object()
                .raw("targets", Json.stringArray(targets))
                .raw("targets_at_cast",
                        targetsAtCast == null ? "null" : Json.stringArray(targetsAtCast))
                .raw("targets_on_stack", Json.stringArray(onStack))
                .value("source", source)
                .raw("modes", modesJson(acting))
                .value("x", x)
                .raw("choices", choicesJson(acting))
                .raw("random", random)
                .render();
    }

    /**
     * A parameter value naming X: the bare variable ({@code NumDmg$ X},
     * {@code NumAtt$ +X}) or a property comparing against it
     * ({@code Creature.cmcLEX}).
     */
    private static final Pattern NAMES_X =
            Pattern.compile("(?<![A-Za-z])X(?![A-Za-z])|(?:LT|LE|EQ|GE|GT|NE)X(?![A-Za-z])");

    /**
     * The X the acting ability announced, read now; null when it has none.
     *
     * <p>The X paid into the mana cost when there is one. Otherwise the
     * ability's {@code SVar:X}, but only when a parameter of its chain names
     * X: {@code getSVar} falls back to the card's own SVars, so without the
     * check every ability of a card whose static defines X — a Cranial
     * Plating's attach, a trigger, a permanent's cast — would carry that
     * static's count. Computed rather than read, because a {@code Count$}
     * SVar is an expression over the board; one that cannot be answered is
     * absent rather than zero. A {@code Count$Random} X is never evaluated:
     * each evaluation draws from {@code MyRandom}, so it names no announced
     * value and would change what the game draws next.
     */
    static Integer announcedX(SpellAbility acting) {
        if (acting == null) {
            return null;
        }
        Integer paid = acting.getXManaCostPaid();
        if (paid != null) {
            return paid;
        }
        Card host = acting.getHostCard();
        if (host == null || !chainNamesX(acting)) {
            return null;
        }
        String amount = acting.getSVar("X");
        if (amount == null || amount.isEmpty() || amount.contains("Random")) {
            return null;
        }
        try {
            return AbilityUtils.calculateAmount(host, amount, acting);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** Whether a parameter of the ability's root or one of its sub-abilities names X. */
    private static boolean chainNamesX(SpellAbility acting) {
        SpellAbility root = acting.getRootAbility() != null ? acting.getRootAbility() : acting;
        for (SpellAbility clause = root; clause != null; clause = clause.getSubAbility()) {
            for (String value : clause.getMapParams().values()) {
                if (value != null && NAMES_X.matcher(value).find()) {
                    return true;
                }
            }
        }
        return false;
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
