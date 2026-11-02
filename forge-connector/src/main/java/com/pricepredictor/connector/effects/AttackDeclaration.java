package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.json.Json;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.combat.AttackRequirement;
import forge.game.combat.Combat;
import forge.game.event.GameEventAttackersDeclared;
import forge.game.event.GameEventCardTapped;
import forge.game.player.Player;
import forge.game.staticability.StaticAbility;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * One declaration of attackers, pivoted to one {@code attackers} record per
 * creature the attacking player controls (FR-037, C6).
 *
 * <p>The engine asks once per defender who may attack it
 * ({@code CombatLegalityObserver.attackersDeclarable}), all before the
 * attacking player declares. Each answer is gathered here, and the records
 * are written per attacker instead: the anchor, the defenders it may attack
 * ({@code legal_defenders}), and whether a requirement makes it attack
 * ({@code must_attack}). The declaration-wide lists the gen-2 record carried,
 * {@code legal_attackers} and {@code forbidden}, are kept on each record, the
 * same on every one: a creature is legal when it may attack at least one
 * defender.
 *
 * <p>The records are held until the declaration is made
 * ({@code GameEventAttackersDeclared}), because a declared anchor's record
 * also says whether declaring the attack tapped it ({@code tapped_by_attack}),
 * and the engine taps the attackers between the two
 * ({@code GameEventCardTapped}). The snapshot is the board before anyone
 * declared, taken when the engine first asks.
 *
 * <p>A declaration ends at the engine's event, at the next declaration (a
 * defender asked about twice, or another attacking player), or with the
 * game: a declaration no event ended wrote nobody declared, so its records
 * carry no {@code tapped_by_attack}.
 *
 * <p>Public, with public {@code @Subscribe} methods only, because Forge's bus
 * reflects over a subscriber's public methods; subscribed to the live game's
 * bus alone, so a fork's taps and declarations never reach it.
 */
public final class AttackDeclaration {

    /** Writes one finished attackers record. */
    interface Emitter {
        void emit(Player decider, String state, String payload);
    }

    private final LiveGame live;
    private final Emitter emitter;
    /** The static keeping a creature from attacking a defender, or null. */
    private final BiFunction<Card, GameEntity, StaticAbility> cantAttackStatic;

    /** The declaration being asked about, or null between two. */
    private Pending pending;

    AttackDeclaration(
            LiveGame live, Emitter emitter,
            BiFunction<Card, GameEntity, StaticAbility> cantAttackStatic) {
        this.live = live;
        this.emitter = emitter;
        this.cantAttackStatic = cantAttackStatic;
    }

    /**
     * The engine's answer for one defender: who among {@code candidates} may
     * attack it.
     *
     * @param attacking the attacking player the engine names, written as
     *                  {@code actor_player}
     */
    void declarable(Player attacking, GameEntity defender, List<Card> candidates, List<Card> legal) {
        if (candidates == null || candidates.isEmpty()) {
            return;
        }
        if (pending != null && !pending.continuedBy(attacking, defender)) {
            write(Set.of());
        }
        if (pending == null) {
            pending = new Pending(attacking, List.copyOf(candidates),
                    live.snapshots().ofBoard().render(), requirements(candidates));
        }
        pending.defenders.put(defender, legal == null ? List.of() : List.copyOf(legal));
    }

    /** A creature tapped while the declaration is being made. */
    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        if (pending != null && event.tapped() && event.card() != null) {
            pending.tapped.add(event.card().getId());
        }
    }

    /** The declaration made: its records are written, a declared anchor's with its tap. */
    @Subscribe
    public void onAttackersDeclared(GameEventAttackersDeclared event) {
        if (pending == null) {
            return;
        }
        Set<Integer> declared = new HashSet<>();
        for (CardView attacker : event.attackersMap().values()) {
            declared.add(attacker.getId());
        }
        write(declared);
    }

    /** The game is over: a declaration still open is written, with nobody declared. */
    void close() {
        if (pending != null) {
            write(Set.of());
        }
    }

    /**
     * Whether each candidate must attack, from the attack constraints the
     * engine initialized for this declaration just before asking.
     */
    private Map<Integer, Boolean> requirements(List<Card> candidates) {
        Combat combat = live.game().getPhaseHandler().getCombat();
        Map<Integer, Boolean> must = new LinkedHashMap<>();
        for (Card card : candidates) {
            AttackRequirement requirement = combat == null || combat.getAttackConstraints() == null
                    ? null : combat.getAttackConstraints().getRequirements().get(card);
            must.put(card.getId(), requirement != null && requirement.hasRequirement());
        }
        return must;
    }

    /** One record per candidate, in the engine's order; {@code declared} by card id. */
    private void write(Set<Integer> declared) {
        Pending declaration = pending;
        pending = null;
        String legalAttackers = declaration.legalAttackersJson();
        String forbidden = declaration.forbiddenJson(cantAttackStatic);
        for (Card anchor : declaration.candidates) {
            Json.ObjectWriter payload = Json.object()
                    .value("anchor_attacker", Refs.of(anchor))
                    .raw("legal_defenders", Json.stringArray(declaration.legalDefenders(anchor)))
                    .value("must_attack", declaration.mustAttack.get(anchor.getId()));
            if (declared.contains(anchor.getId())) {
                payload.value("tapped_by_attack", declaration.tapped.contains(anchor.getId()));
            }
            emitter.emit(declaration.attacking, declaration.state, payload
                    .raw("legal_attackers", legalAttackers)
                    .raw("forbidden", forbidden)
                    .render());
        }
    }

    /** What the engine has answered so far for one declaration. */
    private static final class Pending {

        final Player attacking;
        final List<Card> candidates;
        final String state;
        final Map<Integer, Boolean> mustAttack;
        /** Each defender asked about, in the engine's order, with who may attack it. */
        final Map<GameEntity, List<Card>> defenders = new LinkedHashMap<>();
        /** The card ids tapped since the engine first asked. */
        final Set<Integer> tapped = new HashSet<>();

        Pending(Player attacking, List<Card> candidates, String state,
                Map<Integer, Boolean> mustAttack) {
            this.attacking = attacking;
            this.candidates = candidates;
            this.state = state;
            this.mustAttack = mustAttack;
        }

        /** Whether an answer for {@code defender} belongs to this declaration rather than a new one. */
        boolean continuedBy(Player attacker, GameEntity defender) {
            return attacker == attacking && !defenders.containsKey(defender);
        }

        /** The defenders {@code anchor} may attack, as refs, in the engine's order. */
        List<String> legalDefenders(Card anchor) {
            List<String> refs = new ArrayList<>();
            for (Map.Entry<GameEntity, List<Card>> entry : defenders.entrySet()) {
                if (entry.getValue().contains(anchor)) {
                    Refs.ofAny(entry.getKey()).ifPresent(refs::add);
                }
            }
            return refs;
        }

        boolean legal(Card card) {
            for (List<Card> legal : defenders.values()) {
                if (legal.contains(card)) {
                    return true;
                }
            }
            return false;
        }

        String legalAttackersJson() {
            return Json.array(candidates.stream().filter(this::legal).toList(),
                    card -> Json.string(Refs.of(card)));
        }

        /**
         * The candidates that may attack no defender, each with the first
         * static, in defender order, that keeps it from one; none where the
         * rules themselves do.
         */
        String forbiddenJson(BiFunction<Card, GameEntity, StaticAbility> cantAttackStatic) {
            List<String> forbidden = new ArrayList<>();
            for (Card card : candidates) {
                if (legal(card)) {
                    continue;
                }
                StaticAbility responsible = null;
                for (GameEntity defender : defenders.keySet()) {
                    if (defender != null) {
                        responsible = cantAttackStatic.apply(card, defender);
                    }
                    if (responsible != null) {
                        break;
                    }
                }
                forbidden.add(PlayabilityChannel.forbiddenJson(card, responsible));
            }
            return Json.array(forbidden, Function.identity());
        }
    }
}
