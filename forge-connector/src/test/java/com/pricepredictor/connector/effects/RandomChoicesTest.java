package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The uniform draws (FR-022b, FR-022c), on real charms and real targets.
 *
 * <p>Each draw is checked for its range over many seeds rather than for one
 * seed's answer: what the random seat promises is that every legal option is
 * reachable and nothing illegal is, and one seed can show neither.
 */
@ExtendWith(ForgeExtension.class)
class RandomChoicesTest {

    private static final int SEEDS = 200;

    /**
     * The draws for one seed, scrambled the way production seeds are: a
     * linear congruential generator's first output is nearly linear in its
     * seed, so consecutive raw seeds all make nearly the same first draw and
     * a range test over them sees one value.
     */
    private static RandomChoices drawsFor(int seed) {
        return new RandomChoices(new Random(GameRecording.scramble(seed)));
    }

    // ── modes ───────────────────────────────────────────────────────────

    /** A "choose one" charm draws exactly one mode, and every mode is reachable. */
    @Test
    void aChooseOneCharmDrawsOneModeAndReachesEveryMode() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = TestBoard.charmOf(TestBoard.card(game, caster, "Boros Charm"), caster);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");

        Set<Integer> seen = new HashSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(charm);
            assertEquals(1, modes.size(), "CharmNum 1: " + modes);
            seen.add(choices.indexOf(modes.get(0)));
        }
        assertEquals(Set.of(0, 1, 2), seen);
    }

    /** A count uniform in [MinCharmNum, CharmNum], repeats only where allowed. */
    @Test
    void aMinimumOfZeroAndRepeatsAreHonoured() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card host = TestBoard.card(game, caster, "Grizzly Bears");
        host.setSVar("A", "DB$ GainLife | Defined$ You | LifeAmount$ 1 | SpellDescription$ a");
        host.setSVar("B", "DB$ GainLife | Defined$ You | LifeAmount$ 2 | SpellDescription$ b");
        SpellAbility charm = AbilityFactory.getAbility(
                "SP$ Charm | Choices$ A,B | CharmNum$ 3 | MinCharmNum$ 0 | CanRepeatModes$ True",
                host);
        charm.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        boolean repeated = false;
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(charm);
            counts.add(modes.size());
            repeated |= new HashSet<>(modes).size() < modes.size();
        }
        assertEquals(Set.of(0, 1, 2, 3), counts, "every count between the minimum and maximum");
        assertTrue(repeated, "a charm that may repeat modes sometimes does");

        SpellAbility noRepeat = AbilityFactory.getAbility(
                "SP$ Charm | Choices$ A,B | CharmNum$ 2", host);
        noRepeat.setActivatingPlayer(caster);
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(noRepeat);
            assertEquals(2, modes.size());
            assertEquals(2, new HashSet<>(modes).size(), "no repeat without CanRepeatModes");
        }
    }

    /** A Pawprint charm spends its budget one mode at a time and never overspends. */
    @Test
    void aPawprintCharmDrawsUntilNothingFits() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.card(game, caster, "Season of Gathering"), caster);
        int budget = 5;

        Set<Integer> spent = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            List<AbilitySub> modes = drawsFor(seed).drawModes(charm);
            int total = 0;
            for (AbilitySub mode : modes) {
                total += Integer.parseInt(mode.getParam("Pawprint"));
            }
            assertTrue(total <= budget, "overspent: " + total);
            // Nothing fits any more: the cheapest mode costs one, so a draw
            // stops only at a budget of zero.
            assertEquals(budget, total, "stopped with budget left: " + modes);
            spent.add(modes.size());
        }
        assertTrue(spent.size() > 1, "different mode mixes reach the same budget: " + spent);
    }

    /** The fork's form sets the chosen list and chains, so a direct push resolves them. */
    @Test
    void chooseModesSetsTheChosenListAndChains() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = TestBoard.charmOf(TestBoard.card(game, caster, "Boros Charm"), caster);

        assertTrue(new RandomChoices(new Random(1)).chooseModes(charm));
        assertNotNull(charm.getChosenList());
        assertEquals(1, charm.getChosenList().size());
        assertNotNull(charm.getSubAbility(), "the chosen mode is chained");
    }

    @Test
    void aNonCharmNeedsNoModes() {
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));
        assertNull(new RandomChoices(new Random(1)).drawModes(gain));
        assertTrue(new RandomChoices(new Random(1)).chooseModes(gain));
    }

    // ── targets ─────────────────────────────────────────────────────────

    /**
     * A clause that divides its amount gets it split over its targets, every
     * target a share and the shares summing to the amount (V3 #3): Arc
     * Lightning's resolution reads a share per target and threw on a null.
     */
    @Test
    void aDividedAmountIsSplitOverTheDrawnTargets() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inPlay(game, other, "Grizzly Bears");
        TestBoard.inPlay(game, other, "Hill Giant");
        SpellAbility arc = TestBoard.card(game, caster, "Arc Lightning").getFirstSpellAbility();
        arc.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            assertTrue(drawsFor(seed).chooseTargets(arc));
            int total = 0;
            for (Object target : arc.getTargets().getTargetEntities()) {
                Integer share = arc.getDividedValue((forge.game.GameObject) target);
                assertNotNull(share, "every target has its share");
                assertTrue(share >= 1, "at least one each");
                total += share;
            }
            assertEquals(3, total, "the shares make the amount");
            counts.add(arc.getTargets().size());
        }
        assertEquals(Set.of(1, 2, 3), counts);
    }

    /**
     * A fixed amount smaller than the targets the clause allows is drawn
     * over no more targets than the amount, each taking at least one
     * (601.2d): 2 damage over up to four targets.
     */
    @Test
    void aDividedAmountSmallerThanTheMaximumLimitsTheTargets() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        for (String name : List.of("Grizzly Bears", "Hill Giant", "Gray Ogre", "Craw Wurm")) {
            TestBoard.inPlay(game, other, name);
        }
        SpellAbility split = AbilityFactory.getAbility(
                "SP$ DealDamage | ValidTgts$ Creature | NumDmg$ 2 | TargetMin$ 1 | TargetMax$ 4"
                        + " | DividedAsYouChoose$ 2",
                TestBoard.card(game, caster, "Forked Bolt"));
        split.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            assertTrue(drawsFor(seed).chooseTargets(split));
            int total = 0;
            for (Object target : split.getTargets().getTargetEntities()) {
                int share = split.getDividedValue((forge.game.GameObject) target);
                assertTrue(share >= 1, "seed " + seed + ": every target takes one");
                total += share;
            }
            assertEquals(2, total);
            counts.add(split.getTargets().size());
        }
        assertEquals(Set.of(1, 2), counts);
    }

    /**
     * A constraint across a clause's targets is honoured (V3 #4): Cannibalize
     * takes two creatures one player controls, never one of each player's.
     */
    @Test
    void aConstraintAcrossTargetsIsHonoured() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Grizzly Bears");
        TestBoard.inPlay(game, other, "Hill Giant");
        TestBoard.inPlay(game, other, "Gray Ogre");
        SpellAbility cannibalize = TestBoard.card(game, caster, "Cannibalize").getFirstSpellAbility();
        cannibalize.setActivatingPlayer(caster);

        for (int seed = 0; seed < SEEDS; seed++) {
            if (!drawsFor(seed).chooseTargets(cannibalize)) {
                continue;
            }
            Set<Player> controllers = new HashSet<>();
            cannibalize.getTargets().getTargetCards().forEach(c -> controllers.add(c.getController()));
            assertEquals(Set.of(other), controllers, "seed " + seed);
        }
    }

    /** "Up to two" over three candidates: a count in {0, 1, 2}, distinct targets. */
    @Test
    void targetCountIsUniformBetweenMinimumAndMaximumAndTargetsAreDistinct() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        TestBoard.inPlay(game, other, "Grizzly Bears");
        TestBoard.inPlay(game, other, "Runeclaw Bear");
        TestBoard.inPlay(game, other, "Hill Giant");
        SpellAbility pump = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | TargetMin$ 0 | TargetMax$ 2 | NumAtt$ +1",
                TestBoard.card(game, caster, "Giant Growth"));
        pump.setActivatingPlayer(caster);

        Set<Integer> counts = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            assertTrue(drawsFor(seed).chooseTargets(pump));
            int chosen = pump.getTargets().size();
            counts.add(chosen);
            Set<Object> distinct = new HashSet<>();
            pump.getTargets().getTargetEntities().forEach(distinct::add);
            assertEquals(chosen, distinct.size(), "targets are distinct");
        }
        assertEquals(Set.of(0, 1, 2), counts);
    }

    /** A clause that cannot reach its minimum is a failed draw, not a silent resolution. */
    @Test
    void aClauseShortOfCandidatesFailsTheDraw() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility bolt = AbilityFactory.getAbility(
                "SP$ DealDamage | ValidTgts$ Creature | NumDmg$ 3",
                TestBoard.card(game, caster, "Lightning Bolt"));
        bolt.setActivatingPlayer(caster);

        assertFalse(new RandomChoices(new Random(1)).chooseTargets(bolt));
    }

    /** A line that targets nothing needs nothing chosen. */
    @Test
    void aLineThatTargetsNothingIsReadyToResolve() {
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));

        assertTrue(new RandomChoices(new Random(1)).chooseTargets(gain));
    }

    // ── X ───────────────────────────────────────────────────────────────

    /**
     * Spec Story 3 scenario 15: with five Mountains and a cost of {X}{R}, X
     * falls uniformly between zero and four — every value reached, none past
     * what the mana can pay.
     */
    @Test
    void xIsUniformBetweenTheSmallestLegalValueAndWhatTheManaCanPay() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        for (int i = 0; i < 5; i++) {
            TestBoard.inPlay(game, caster, "Mountain");
        }
        SpellAbility drain = AbilityFactory.getAbility(
                "AB$ DealDamage | Cost$ X R | ValidTgts$ Any | NumDmg$ X",
                TestBoard.card(game, caster, "Fireball"));
        drain.setActivatingPlayer(caster);

        Set<Integer> seen = new TreeSet<>();
        for (int seed = 0; seed < SEEDS; seed++) {
            drain.setXManaCostPaid(null);
            drawsFor(seed).announceX(drain, caster, 0);
            seen.add(drain.getXManaCostPaid());
        }
        assertEquals(Set.of(0, 1, 2, 3, 4), seen);
    }

    /** With nobody paying, the floor is the whole answer: the fork's one. */
    @Test
    void withNoPayerTheFloorIsAnnounced() {
        SpellAbility drain = AbilityFactory.getAbility(
                "AB$ LoseLife | Cost$ X B | Defined$ Player.Opponent | LifeAmount$ X",
                TestCards.build("Fountain of Youth"));
        new RandomChoices(new Random(1)).announceX(drain, null, 1);
        assertEquals(1, drain.getXManaCostPaid());
    }

    @Test
    void anAbilityWithNoXIsLeftAlone() {
        Player caster = TestBoard.twoPlayerGame(PhaseType.MAIN1).getPlayers().get(0);
        SpellAbility gain = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1",
                TestCards.build("Fountain of Youth"));

        new RandomChoices(new Random(1)).announceX(gain, caster, 1);

        assertNull(gain.getXManaCostPaid());
    }
}
