package forge.gamemodes.limited;

import forge.item.PaperCard;

import java.util.List;

/**
 * Rebuilds the colour pair a drafting seat committed to, by replaying its picks.
 *
 * <p>Forge builds a drafted pool with {@link BoosterDeckBuilder}, which is handed the
 * {@link DeckColors} the drafter accumulated <em>during</em> the draft rather than deriving
 * them from the finished pool. {@link LimitedPlayerAI} grows that object one pick at a time
 * and feeds it straight to the builder, so the pool and the colour pair are one artefact.
 * A drafted pool is already concentrated in two colours because of the commitment; re-deriving
 * colours from it afterwards, as {@link SealedDeckBuilder} does, throws away the half of that
 * artefact the draft produced.
 *
 * <p>Games replayed from a recorded corpus have no live drafter to ask, but they do have the
 * pick order, and {@link DeckColors#addColorsOf} is a pure function of it. Replaying the picks
 * through a fresh {@code DeckColors} therefore reproduces exactly what the drafting seat held.
 * The one divergence is {@code LimitedPlayerAI}'s Archdemon Curse branch, which takes a forced
 * pick without updating the colours.
 *
 * <p>This class lives in Forge's package on purpose: {@code DeckColors}' constructor is
 * package-private, and the alternative — reimplementing {@code addColorsOf} against the public
 * API — would fork the colour rule that has to stay identical to Forge's for the comparison to
 * mean anything. {@link FullDeckColors} is public but overrides {@code addColorsOf} to widen on
 * colour identity rather than deckbuilding colours, so it is not the same rule.
 */
public final class DraftedColorCommitment {

    private DraftedColorCommitment() {}

    /**
     * Replay a seat's picks and return the colours it would have committed to.
     *
     * <p>Mirrors {@link LimitedPlayerAI#chooseCard}: guard on {@code canChoseMoreColors()},
     * then {@code addColorsOf} the pick. The guard is not just an optimisation — it calls
     * {@code getChosenColors()}, which primes the lazily-built {@code ColorSet} that
     * {@code addColorsOf} dereferences on its first line.
     *
     * @param picksInPickOrder the seat's drafted cards, earliest pick first
     * @return the committed colours; empty if every pick was colourless
     */
    public static DeckColors replay(List<PaperCard> picksInPickOrder) {
        DeckColors colors = new DeckColors();
        for (PaperCard pick : picksInPickOrder) {
            if (colors.canChoseMoreColors()) {
                colors.addColorsOf(pick);
            }
        }
        return colors;
    }
}
