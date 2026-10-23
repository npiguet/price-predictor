package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import forge.StaticData;
import forge.card.CardEdition;
import forge.deck.Deck;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.util.MyRandom;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;

/**
 * Generates one complete sealed match outcome per call.
 *
 * <p>Each match is composed by independently choosing how to produce deck A
 * and deck B, with a same-set constraint: deck B is always built from a pool
 * (or sampled from a deck file) matching deck A's set code.
 *
 * <p><b>Side-A source</b> (a {@link DeckSource}):
 * <ul>
 *   <li>{@link DeckSource.ForgeBuilt} (default): pick a random eligible set,
 *       generate a fresh 6-booster pool, and call {@link DeckBuilder#buildDeck}
 *       which rolls the 4 Forge methods with weights 4:3:2:1.</li>
 *   <li>{@link DeckSource.FileSampled}: sample a deck uniformly at random from
 *       the index; the deck's recorded {@code label} becomes {@code method_A}.</li>
 * </ul>
 *
 * <p><b>Side-B source</b> (a {@link DeckSource} and {@code sideBWeight}):
 * <ul>
 *   <li>{@link DeckSource.ForgeBuilt}: deck B is always Forge-built from a
 *       fresh pool of deck A's set, via {@link DeckBuilder#buildDeck}
 *       (weights 4:3:2:1).</li>
 *   <li>{@link DeckSource.FileSampled}: roll between Forge methods (weights
 *       4:3:2:1, total 10) and sampling from the index (weight
 *       {@code sideBWeight}). On the file-sample branch, {@link
 *       GeneratedDecksIndex#randomDeckFromSet} filters to deck A's set code
 *       and excludes mirror matches by content equality. If no non-mirror
 *       candidate exists for that set, fall back to Forge methods — unless
 *       the round is decks-only, see below.</li>
 *   <li>Decks-only ({@link FileDecks#ONLY}): forces the roll to always take
 *       the file-sample branch, regardless of {@code sideBWeight}. For decks
 *       that belong to no set (coverage, variant) and carry a sentinel set
 *       code the Forge method branch would fail to resolve against Forge's
 *       booster/edition tables — so a decks-only non-mirror miss falls back
 *       to {@link GeneratedDecksIndex#randomAnyDeckFromSet} (accepting a
 *       mirror) rather than to Forge methods; only an empty index for that
 *       set reaches an exception. See {@link #pickDeckB} for why this case is
 *       ordinary rather than rare.</li>
 * </ul>
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class MatchGenerator {

    /** Where one side's decks come from. */
    sealed interface DeckSource {

        /** Built by {@link DeckBuilder#buildDeck} from a fresh 6-booster pool. */
        record ForgeBuilt() implements DeckSource {}

        /** Sampled from a generated-decks file; each deck's label is its method tag. */
        record FileSampled(GeneratedDecksIndex index) implements DeckSource {
            public FileSampled {
                Objects.requireNonNull(index, "index");
            }
        }

        DeckSource FORGE_BUILT = new ForgeBuilt();
    }

    /** What a round's decks files are to it: one source among Forge's, or the only one. */
    enum FileDecks {
        /**
         * Sealed self-play: a file is one source beside the Forge methods. A
         * side-B miss falls back to Forge, and a name Forge cannot resolve is
         * only logged, since a self-play deck's names came out of Forge in
         * the first place.
         */
        ALONGSIDE_FORGE(DeckMaterializer.UnresolvedPolicy.WARN),
        /**
         * Decks-only, for coverage and variant rounds: the files are the only
         * source, because a deck's set code can be a sentinel no Forge edition
         * resolves. Side B always samples its file, a miss accepts a mirror
         * rather than reach Forge, and a name Forge cannot resolve breaks the
         * round — see {@link #DECKS_ONLY_UNRESOLVED}.
         */
        ONLY(DeckMaterializer.UnresolvedPolicy.FAIL);

        /** What a file-sampled deck's unresolvable name does in this round. */
        final DeckMaterializer.UnresolvedPolicy unresolved;

        FileDecks(DeckMaterializer.UnresolvedPolicy unresolved) {
            this.unresolved = unresolved;
        }
    }

    private final List<String> eligibleSetCodes;
    private final DeckBuilder deckBuilder;
    private final GamePlayer gamePlayer;
    private final String runId;
    private final DeckSource sideA;
    private final DeckSource sideB;
    private final int sideBWeight;
    private final Random random;
    private final HoldoutCards excludedCards;
    private final FileDecks fileDecks;

    private MatchGenerator(Builder builder) {
        validate(builder);
        this.eligibleSetCodes = List.copyOf(builder.eligibleSetCodes);
        this.deckBuilder = builder.deckBuilder;
        this.gamePlayer = builder.gamePlayer;
        this.runId = builder.runId;
        this.sideA = builder.sideA;
        this.sideB = builder.sideB;
        this.sideBWeight = builder.sideBWeight;
        this.random = builder.random;
        this.excludedCards = builder.excludedCards;
        this.fileDecks = builder.fileDecks;
    }

    /** Refuse a configuration that would build matches other than the ones asked for. */
    private static void validate(Builder builder) {
        if (builder.eligibleSetCodes.isEmpty()) {
            throw new IllegalArgumentException("Eligible set list must not be empty");
        }
        if (builder.runId == null || builder.runId.isBlank()) {
            throw new IllegalArgumentException("runId must be non-empty");
        }
        if (builder.sideB instanceof DeckSource.FileSampled && builder.sideBWeight < 1) {
            throw new IllegalArgumentException(
                    "sideBWeight must be >= 1 when side B samples a decks file, got: "
                            + builder.sideBWeight);
        }
        boolean sideAFromFile = builder.sideA instanceof DeckSource.FileSampled;
        boolean sideBFromFile = builder.sideB instanceof DeckSource.FileSampled;
        if (builder.fileDecks == FileDecks.ONLY && !(sideAFromFile && sideBFromFile)) {
            // Both holes are silent, which is why they are refused rather
            // than tolerated. With side B Forge-built, rollIsFileSample()
            // returns false for every match and the whole round plays
            // ordinary sealed self-play while reporting success. With side A
            // Forge-built there is no exception anywhere: pickDeckA builds
            // deck A from a fresh Forge pool of a random eligible set, so
            // half of every match is a deck the round never asked for.
            throw new IllegalArgumentException(
                    "decksOnly requires a decks file on both sides, got side A "
                            + (sideAFromFile ? "from a file" : "Forge-built")
                            + ", side B " + (sideBFromFile ? "from a file" : "Forge-built")
                            + ". A missing file does not fail a decks-only match, it "
                            + "quietly turns it back into sealed self-play.");
        }
    }

    /**
     * A generator for {@code runId} drawing Forge-built decks from
     * {@code eligibleSetCodes}; the builder's setters change what differs.
     */
    static Builder builder(List<String> eligibleSetCodes, String runId) {
        return new Builder(eligibleSetCodes, runId);
    }

    /**
     * How a generator is configured. Every setting but the eligible sets and
     * the run id has a default: Forge-built decks on both sides, no holdout,
     * not decks-only, a best-of-3 {@link GamePlayer} with no effect records,
     * and Forge's shared random.
     */
    static final class Builder {
        private final List<String> eligibleSetCodes;
        private final String runId;
        private final DeckBuilder deckBuilder = new DeckBuilder();
        private GamePlayer gamePlayer = new GamePlayer(3);
        private DeckSource sideA = DeckSource.FORGE_BUILT;
        private DeckSource sideB = DeckSource.FORGE_BUILT;
        private int sideBWeight;
        private Random random = MyRandom.getRandom();
        private HoldoutCards excludedCards = HoldoutCards.NONE;
        private FileDecks fileDecks = FileDecks.ALONGSIDE_FORGE;

        private Builder(List<String> eligibleSetCodes, String runId) {
            this.eligibleSetCodes = eligibleSetCodes;
            this.runId = runId;
        }

        Builder gamePlayer(GamePlayer gamePlayer) {
            this.gamePlayer = gamePlayer;
            return this;
        }

        Builder sideA(DeckSource sideA) {
            this.sideA = sideA;
            return this;
        }

        /**
         * Side B's source, and when it samples a decks file, that file's
         * weight against the Forge methods' {@link BuildMethod#TOTAL_WEIGHT}.
         */
        Builder sideB(DeckSource sideB, int weight) {
            this.sideB = sideB;
            this.sideBWeight = weight;
            return this;
        }

        Builder random(Random random) {
            this.random = random;
            return this;
        }

        Builder excludedCards(HoldoutCards excludedCards) {
            this.excludedCards = excludedCards;
            return this;
        }

        /** Whether the decks files are the round's only source ({@link FileDecks#ONLY}). */
        Builder decksOnly(boolean decksOnly) {
            this.fileDecks = decksOnly ? FileDecks.ONLY : FileDecks.ALONGSIDE_FORGE;
            return this;
        }

        MatchGenerator build() {
            return new MatchGenerator(this);
        }
    }

    /**
     * Minimum booster size for a set to be considered sealed-viable. Matches the
     * rule Forge itself uses in {@code AdventureEventData.isValidDraftBlock()};
     * excludes historical sets (DRK, FEM, …) whose 8-card boosters are too small
     * to yield a real sealed pool.
     */
    static final int MIN_BOOSTER_CARDS = 12;

    /**
     * Compute the list of eligible sealed-format set codes.
     * Eligible sets have a draft booster template, are not un-sets (Type.FUNNY),
     * and have at least {@link #MIN_BOOSTER_CARDS} cards per booster.
     */
    public static List<String> computeEligibleSets() {
        List<String> eligible = new ArrayList<>();
        for (CardEdition edition : StaticData.instance().getEditions().getOrderedEditions()) {
            if (edition.getType() == CardEdition.Type.FUNNY) {
                continue;
            }
            SealedTemplate template = edition.getBoosterTemplate("Draft");
            if (template == null) {
                continue;
            }
            if (template.getNumberOfCardsExpected() < MIN_BOOSTER_CARDS) {
                continue;
            }
            eligible.add(edition.getCode());
        }
        return eligible;
    }

    /**
     * Generate one complete match outcome.
     *
     * <p>Returns both the per-match {@link MatchResult} (written to
     * {@code match-outcomes.txt}) and one {@link CardsPlayedRow} per played
     * game (written to {@code cards-played.txt}).
     */
    public MatchGenerationResult generateMatch() {
        DeckSelection a = pickDeckA();
        DeckSelection b = pickDeckB(a.setCode, a.cardNames);

        GamePlayer.PlayedMatch played = gamePlayer.playMatch(a.deck, b.deck);

        MatchResult matchResult = MatchResult.of(
                runId, a.setCode, a.method, b.method, a.cardNames, b.cardNames, played);

        List<CardsPlayedRow> rows = new ArrayList<>(played.games().size());
        for (GamePlayer.GameOutcome game : played.games()) {
            rows.add(CardsPlayedRow.of(matchResult, game));
        }
        return new MatchGenerationResult(matchResult, rows, played.randomSeat());
    }

    /**
     * Pick deck A. Forge-built: roll the 4 Forge methods
     * on a fresh pool from a random eligible set. From a file: sample
     * uniformly from the index.
     */
    DeckSelection pickDeckA() {
        if (sideA instanceof DeckSource.FileSampled file) {
            GeneratedDeck sampled = file.index().randomDeck(random);
            return DeckSelection.fromFile(sampled, sampled.setCode(), fileDecks);
        }
        String setCode = eligibleSetCodes.get(random.nextInt(eligibleSetCodes.size()));
        return forgeBuilt(setCode);
    }

    /**
     * Pick deck B for a match whose deck A has the given set code and card
     * list. Roll between the 4 Forge methods (weights 4:3:2:1, total
     * {@link BuildMethod#TOTAL_WEIGHT}) and — if side B samples a decks
     * file — sampling from it (weight {@code sideBWeight}). The
     * file-sample branch falls back to Forge methods when no non-mirror
     * candidate exists for {@code setCode} — <b>except</b> in a decks-only
     * round, which has no Forge-legal fallback to reach (its set
     * code can be a sentinel like {@code COVERAGE} that resolves against no
     * real Forge edition, so {@link #forgeBuilt} would find no booster
     * template for it and throw). A decks-only round's live-card pool shrinks
     * monotonically as cards are satisfied, so a round with exactly one card
     * left produces byte-identical decks, deterministically — an ordinary
     * tail state, not a rare one — and every match attempt would hit that
     * throw: each restarted worker gives up after {@link WorkerLoop}'s cap,
     * and the round never appends another progress line. So a decks-only
     * round instead accepts a mirror ({@link #mirrorFor}): a mirror match
     * still puts every card in the round's decks into a game, which is all a
     * coverage or variant round requires.
     */
    DeckSelection pickDeckB(String setCode, List<String> deckACards) {
        if (!(sideB instanceof DeckSource.FileSampled file) || !rollIsFileSample()) {
            return forgeBuilt(setCode);
        }
        Optional<GeneratedDeck> pick = file.index().randomDeckFromSet(
                setCode, deckACards, random);
        if (pick.isEmpty() && fileDecks == FileDecks.ONLY) {
            pick = Optional.of(mirrorFor(file.index(), setCode));
        }
        // Otherwise no non-mirror deck is available for this set; Forge builds it.
        return pick.map(deck -> DeckSelection.fromFile(deck, setCode, fileDecks))
                .orElseGet(() -> forgeBuilt(setCode));
    }

    /**
     * Any deck of {@code setCode}, mirror or not, for a decks-only side B
     * whose non-mirror search came up empty.
     *
     * @throws IllegalStateException when the file has no deck of the set at
     *         all. Not reachable from CollectorSupervisor, which always points
     *         both sides at the same file: a decks-only side-B index with no
     *         deck of deck A's set is a configuration error, and silently
     *         falling through to Forge here would reintroduce
     *         {@link #pickDeckB}'s whole bug for a rarer trigger.
     */
    private GeneratedDeck mirrorFor(GeneratedDecksIndex index, String setCode) {
        return index.randomAnyDeckFromSet(setCode, random).orElseThrow(
                () -> new IllegalStateException(
                        "decksOnly is set but the side-B decks file has no "
                                + "deck of set " + setCode + "; cannot build a "
                                + "match without reaching the Forge-methods "
                                + "path decksOnly exists to avoid"));
    }

    /**
     * Build a deck via the 4 Forge methods (weights 4:3:2:1) on a fresh
     * 6-booster pool of {@code setCode}.
     */
    private DeckSelection forgeBuilt(String setCode) {
        List<PaperCard> pool = generatePool(setCode);
        DeckBuilder.BuiltDeck built = deckBuilder.buildDeck(pool);
        return new DeckSelection(
                built.deck(), DeckMaterializer.names(built.deck()), built.method().label(),
                setCode);
    }

    /**
     * Roll {@code true} with probability {@code sideBWeight / (10 + sideBWeight)}
     * — the "sample from side B's file" branch. Always returns {@code false}
     * when side B is Forge-built, and always returns {@code true} (once
     * side B has a file) in a decks-only round.
     */
    boolean rollIsFileSample() {
        if (!(sideB instanceof DeckSource.FileSampled)) {
            return false;
        }
        // A decks-only round has no set to open a booster for: both sides come
        // from the decks file or the match cannot be built at all.
        if (fileDecks == FileDecks.ONLY) {
            return true;
        }
        double total = BuildMethod.TOTAL_WEIGHT + sideBWeight;
        return random.nextDouble() < sideBWeight / total;
    }

    List<PaperCard> generatePool(String setCode) {
        SealedTemplate template = BoosterTemplates.forSet(setCode).orElseThrow(
                () -> new IllegalArgumentException("No booster template for set " + setCode));
        // Delegated so that depletion has one implementation: this is the pool
        // every collected effect record is played from, and a second copy of
        // the redraw rule here would be the copy that silently stopped matching.
        return new PoolGenerator().openSinglePool(template, excludedCards);
    }

    /**
     * Why a decks-only round refuses a deck with a name Forge cannot resolve.
     *
     * <p>A sealed self-play deck's names came out of Forge in the first place,
     * so a miss there is only logged. A coverage or variant deck's names come
     * from {@code output/cardsfolder/} and from {@code VariantRegistry}: a name
     * Forge misses is dropped, the deck materializes as 17 basic lands, a game
     * plays, a progress line is appended, effect records are written — and the
     * card the round exists to reach never enters a game, then retires as
     * "castable but short". For {@code collect-variants} this can be the whole
     * run: if the staged custom scripts do not land in {@code getCommonCards()}
     * under exactly the names the decks file spells, every deck is basics and
     * the command exits 0 having collected nothing.
     */
    static final String DECKS_ONLY_UNRESOLVED =
            "A decks-only deck's names come from the converted tree or from "
                    + "VariantRegistry rather than from Forge, so dropping them "
                    + "would play a deck of basic lands and collect nothing "
                    + "while reporting a completed match.";

    /**
     * One side of a match: the playable {@link Deck}, its card-name list, the
     * method tag, and the set code. Bundled together so {@link #pickDeckA}
     * and {@link #pickDeckB} can return both file-sampled and Forge-built
     * decks through a single shape.
     */
    record DeckSelection(Deck deck, List<String> cardNames, String method, String setCode) {

        /**
         * Build a {@code DeckSelection} from a generated-decks-file entry,
         * materializing the card list into a Forge {@link Deck} and using
         * the deck's recorded {@code label} as the method tag. {@code setCode}
         * is taken explicitly so callers can either propagate the file's
         * own set code (side A) or substitute deck A's set code (side B,
         * where same-set pairing is enforced upstream). {@code fileDecks}
         * decides what a name Forge cannot resolve does: in a decks-only
         * round a coverage or variant deck's names did not come from Forge,
         * so one it cannot resolve is a broken round rather than a curiosity
         * — see {@link #DECKS_ONLY_UNRESOLVED}.
         */
        static DeckSelection fromFile(GeneratedDeck deck, String setCode, FileDecks fileDecks) {
            return new DeckSelection(
                    DeckMaterializer.materialize(
                            deck.cardNames(), fileDecks.unresolved, DECKS_ONLY_UNRESOLVED),
                    deck.cardNames(),
                    deck.label(),
                    setCode);
        }
    }
}
