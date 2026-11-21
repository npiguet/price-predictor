package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.keyword.KeywordInterface;
import forge.game.staticability.StaticAbility;

import java.util.List;

/**
 * What keeps a creature out of a combat declaration's legal set, and the
 * {@code forbidden} entry a legality record writes for it.
 *
 * <p>The blockers record ({@link PlayabilityChannel}) and the attackers
 * record ({@link AttackDeclaration}) both name the line responsible: a
 * static ability's, else a can't-attack or can't-block keyword line of the
 * creature's own that Forge reads as a keyword and no static, else none,
 * where the rules themselves keep it out. A blocker may instead be kept out
 * by a requirement to block another attacker ({@link #requiredElsewhere}).
 */
final class BlockLegality {

    private BlockLegality() {
    }

    /**
     * The line that keeps a creature out of a legal set: a static
     * ability's (Pacifism, a flyer's keyword line through the static it
     * builds), else a keyword line of the creature's own that Forge reads
     * as a keyword and no static ("CARDNAME can't attack or block."), with
     * its key, or the reason it has none ({@link ProvenanceKey.Resolved});
     * and whether that line is the creature's own (a static it hosts, or
     * a keyword it prints) rather than another card's.
     */
    record Responsible(ProvenanceKey.Resolved line, boolean own) {

        /**
         * The static when there is one, else the creature's own keyword
         * line, else null: the rules themselves keep the creature out
         * (tapped, summoning sick), or a hidden keyword with no line does
         * (Panic's "can't block this turn", read off the entity's
         * {@code granted_temporary}).
         */
        static Responsible of(Card card, StaticAbility responsible, KeywordInterface ownLine) {
            if (responsible != null) {
                ProvenanceKey.Resolved line = ProvenanceResolver.resolve(responsible);
                return new Responsible(line == null ? ProvenanceKey.Resolved.UNKNOWN : line,
                        card.equals(responsible.getHostCard()));
            }
            if (ownLine != null) {
                // A keyword a static grants is that static's line, not the creature's.
                return new Responsible(keywordLine(ownLine), ownLine.getStatic() == null);
            }
            return null;
        }

        /** A keyword line's key: printed on the creature, else its granting static's. */
        private static ProvenanceKey.Resolved keywordLine(KeywordInterface keyword) {
            Card host = keyword.getHostCard();
            ProvenanceKey key = ProvenanceResolver.keyOfKeyword(
                    host == null ? null : host.getCurrentState(), keyword);
            if (key != null) {
                return new ProvenanceKey.Resolved(key, null);
            }
            return new ProvenanceKey.Resolved(null, keyword.getStatic() == null
                    ? UnresolvedReason.UNINDEXABLE : UnresolvedReason.GRANTED_KEYWORD);
        }
    }

    /** The keyword lines Forge reads as keywords, not statics, to keep a creature from attacking. */
    private static final List<String> CANT_ATTACK_KEYWORDS = List.of(
            "CARDNAME can't attack or block.", "CARDNAME can't attack.");
    /** The same, from blocking at all ({@code CombatUtil.canBlock(Card, boolean)}). */
    private static final List<String> CANT_BLOCK_KEYWORDS = List.of(
            "CARDNAME can't attack or block.", "CARDNAME can't block.");
    /** The same, from blocking while its controller has no other creature. */
    private static final List<String> CANT_BLOCK_ALONE_KEYWORDS = List.of(
            "CARDNAME can't attack or block alone.", "CARDNAME can't block alone.");

    /**
     * The creature's own keyword line (not a hidden keyword) that keeps it
     * from blocking at all, as {@code CombatUtil.canBlock(Card, boolean)}
     * reads it, or null: "can't attack or block", "can't block", and the
     * "... alone" forms while its controller has no other creature in play.
     */
    static KeywordInterface ownBlockingKeyword(Card blocker) {
        boolean alone = blocker.getController() != null
                && blocker.getController().getCreaturesInPlay().size() < 2;
        for (KeywordInterface keyword : blocker.getKeywords()) {
            if (keepsFromBlocking(keyword.getOriginal(), alone)) {
                return keyword;
            }
        }
        return null;
    }

    /** Whether keyword {@code line} keeps its creature from blocking, {@code alone} or not. */
    private static boolean keepsFromBlocking(String line, boolean alone) {
        return CANT_BLOCK_KEYWORDS.contains(line)
                || (alone && CANT_BLOCK_ALONE_KEYWORDS.contains(line));
    }

    /**
     * The creature's own keyword line (not a hidden keyword) that keeps it
     * from attacking at all, as {@code StaticAbilityCantAttackBlock.cantAttack}
     * reads it, or null.
     */
    static KeywordInterface ownAttackingKeyword(Card attacker) {
        for (KeywordInterface keyword : attacker.getKeywords()) {
            if (CANT_ATTACK_KEYWORDS.contains(keyword.getOriginal())) {
                return keyword;
            }
        }
        return null;
    }

    /**
     * One entity kept out of a legal set, and the line responsible.
     *
     * <p>The line is often absent, and that is a real answer rather than a
     * gap: a creature that is tapped or summoning sick is stopped by the
     * rules themselves and no line is to blame.
     */
    static String forbiddenJson(Card card, Responsible responsible) {
        return forbiddenJson(card, responsible, false);
    }

    /**
     * The same, for a blocker: {@code required_elsewhere: true}, last,
     * when what keeps it from blocking this attacker is a requirement to
     * block another (a lure, 509.1c), not a prohibition; absent otherwise.
     *
     * <p>{@code responsible_static} holds the responsible line's key: a
     * static's line, or the creature's own keyword line. After it,
     * {@code responsible_own: true} when that line is the creature's own (a
     * static it hosts, a keyword it prints), absent when it is another
     * card's; then {@code responsible_unresolved}, the reason, when the
     * responsible line has no key (as a record's {@code ability_unresolved}
     * does), leaving {@code responsible_static} empty: kept out by
     * something, though nothing names its line.
     */
    static String forbiddenJson(Card card, Responsible responsible, boolean requiredElsewhere) {
        ProvenanceKey.Resolved line = responsible == null ? null : responsible.line();
        var entry = Json.object()
                .value("entity", Refs.of(card))
                .raw("responsible_static", ProvenanceKey.toJsonArray(
                        line == null ? List.of() : line.keys()));
        if (responsible != null && responsible.own()) {
            entry.value("responsible_own", true);
        }
        if (line != null && line.key() == null) {
            entry.value("responsible_unresolved", line.reason().wireName());
        }
        if (requiredElsewhere) {
            entry.value("required_elsewhere", true);
        }
        return entry.render();
    }

    /**
     * Whether a requirement to block another attacker is what keeps
     * {@code blocker} from blocking {@code attacker}, though the
     * restrictions alone would let it (509.1c: requirements bind only
     * within what the restrictions allow). Read off Forge's own check, so
     * its whole requirement set counts (a lure, provoke, "must be blocked
     * by", and an attacker that "must be blocked if able", "by exactly one
     * creature" or "by two or more creatures" while that is unmet:
     * {@code CombatUtil.mustBlockAnAttacker}): every other gate of
     * {@code CombatUtil.canBlock(attacker, blocker, combat)} lets the block
     * through, and that check refuses it. A blocker the restrictions keep
     * from {@code attacker} anyway (it flies, the blocker does not) is not.
     */
    static boolean requiredElsewhere(Card blocker, Card attacker, Combat combat) {
        return combat != null
                && CombatUtil.canBlock(attacker, blocker)
                && CombatUtil.canBlock(blocker, combat)
                && CombatUtil.canBeBlocked(attacker, combat, blocker.getController())
                && !combat.isBlocking(blocker, attacker)
                && !CombatUtil.canBlock(attacker, blocker, combat);
    }

    /**
     * The static keeping {@code blocker} from blocking {@code attacker}: one
     * aimed at the pair, else one keeping it from blocking at all (Pacifism,
     * {@code cantBlockStatic}).
     */
    static StaticAbility blockingStatic(Card attacker, Card blocker) {
        StaticAbility pair = cantBlockByStatic(attacker, blocker);
        return pair != null ? pair : (StaticAbility) PatchHooks.CANT_BLOCK_STATIC.invoke(
                new Class<?>[]{Card.class}, blocker);
    }

    /** Reflective, because the engine exposes these only on a patched checkout. */
    private static StaticAbility cantBlockByStatic(Card attacker, Card blocker) {
        return (StaticAbility) PatchHooks.CANT_BLOCK_BY_STATIC.invoke(
                new Class<?>[]{Card.class, Card.class}, attacker, blocker);
    }
}
