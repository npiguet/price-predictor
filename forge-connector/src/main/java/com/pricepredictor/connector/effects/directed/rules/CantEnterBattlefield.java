package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 614.17d, an effect saying a permanent can't enter the battlefield stops it
 * entering, on scenario {@code 614.17d-1}: A casts Zombify on Grizzly Bears
 * in its graveyard with no Grafdigger's Cage on the battlefield, and the Bears
 * enters; with the Cage it would stay in the graveyard.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, the active player holds a spell it has the mana for that returns a
 * card from a graveyard to the battlefield ({@link #REANIMATION}), a creature
 * card in a graveyard is one that spell can return ({@link #returnsACreature}),
 * and no permanent on the battlefield has a replacement effect that stops a
 * card entering the battlefield from a graveyard. It binds the active player
 * and that spell's card; the fork casts that spell, the half of a split card
 * that reanimates. The live game, where the creature
 * would enter, is the common side; the fork adds the rare fact.
 *
 * <p>Variant {@code cage} ({@code add_permanent}): Grafdigger's Cage is on
 * the battlefield under the active player's control, so a creature card the
 * spell returns from the graveyard stays there: the rewrite record of the
 * prevented move is what 614.17d-1's declaration reads. The fork forces the
 * spell's cast, its target drawn (FR-010b): Forge's AI does not cast it
 * under the Cage.
 */
public final class CantEnterBattlefield {

    public static final String RULE = "614.17d";
    static final String SCENARIO = "614.17d-1";
    static final String CAGE = "cage";
    static final String GRAFDIGGERS_CAGE = "Grafdigger's Cage";
    /** A spell of the card that reanimates ({@link #reanimates}): what the precondition finds and the fork casts. */
    static final ForcedPlay.Line REANIMATION = ForcedPlay.Line.spell("reanimation", CantEnterBattlefield::reanimates);

    private CantEnterBattlefield() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CantEnterBattlefield::bind, CAGE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(GRAFDIGGERS_CAGE, fork.player("player"), false))
                        .forcing(ForcedPlay.cast("player", "card").playing(REANIMATION)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        if (live.getCardsIn(ZoneType.Battlefield).stream().anyMatch(CantEnterBattlefield::stopsEntering)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return Boards.castableSpellsInHand(active, REANIMATION).stream()
                .filter(spell -> returnsACreature(spell, active))
                .findFirst()
                .map(spell -> Binding.empty().with("player", active).with("card", spell.getHostCard()));
    }

    /**
     * Whether {@code spell}, cast by {@code caster}, can return a creature
     * card now in a graveyard: one it can target, or for a spell that does
     * not target, one of the caster's whose type it names
     * ({@code ChangeType}). A spell returning a card it defines otherwise
     * (itself, a remembered card) returns none this board can tell.
     */
    static boolean returnsACreature(SpellAbility spell, Player caster) {
        SpellAbility cast = Boards.asPlayedBy(spell, caster);
        if (cast.usesTargeting()) {
            return caster.getGame().getCardsIn(ZoneType.Graveyard).stream()
                    .anyMatch(card -> card.isCreature() && cast.canTarget(card));
        }
        String types = cast.getParam("ChangeType");
        return types != null && caster.getCardsIn(ZoneType.Graveyard).stream().anyMatch(card ->
                card.isCreature() && card.isValid(types.split(","), caster, cast.getHostCard(), cast));
    }

    /** Whether one of the permanent's replacement effects stops a card entering the battlefield from a graveyard. */
    static boolean stopsEntering(Card permanent) {
        for (ReplacementEffect replacement : permanent.getReplacementEffects()) {
            if (replacement.getMode() == ReplacementType.Moved
                    && Boards.lists(replacement.getParam("Destination"), "Battlefield")
                    && Boards.lists(replacement.getParam("Origin"), "Graveyard")
                    && "True".equals(replacement.getParam("Prevent"))) {
                return true;
            }
        }
        return false;
    }

    /** Whether the spell puts a card from a graveyard onto the battlefield. */
    static boolean reanimates(SpellAbility spell) {
        return spell.getApi() == ApiType.ChangeZone
                && Boards.lists(spell.getParam("Origin"), "Graveyard")
                && Boards.lists(spell.getParam("Destination"), "Battlefield");
    }
}
