package com.pricepredictor.connector;

import forge.gui.GuiBase;
import forge.model.FModel;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Shared Forge initialization — used by all entry points (ConvertMain, PoolMain, MatchWorkerMain) and tests.
 *
 * <p>Uses FModel.initialize() + GuiHeadless to set up the full Forge environment including
 * StaticData, AI subsystems, and the game rules engine. Auto-detects the Forge installation
 * by searching for a "forge" directory starting from the current working directory and walking
 * up to parent directories.
 */
public class ForgeEnvironmentInitializer {
    private static volatile boolean initialized = false;

    private static final String RES_SUBPATH = "forge-gui/res";
    private static final String GUI_SUBPATH = "forge-gui";

    public static synchronized void initialize() {
        initialize(null);
    }

    /**
     * Initialize Forge, optionally loading an extra directory of card scripts.
     *
     * <p>Stage four's synthetic variants live in
     * {@code output/effects/variant-scripts/}, and Forge cannot see them
     * otherwise: it resolves only {@code forge-gui/res/cardsfolder} plus the
     * fixed {@code ForgeConstants.USER_CUSTOM_CARDS_DIR}, neither of which this
     * project writes to. Without this, {@code collect-variants} could generate
     * perturbed scripts and never get one into a game.
     *
     * <p>The extra directory is staged into Forge's own custom-cards directory
     * rather than passed to the card reader, because the reader's source path
     * is a constant the engine resolves at class-init time. Staging is a copy,
     * so the variant tree stays the artifact the trainer and the encoder read.
     *
     * <p>Every staged name is also registered with {@code VariantRegistry},
     * which is what makes a variant's provenance keys name the
     * {@code variant-scripts} tree rather than {@code cardsfolder}. Registration
     * happens here because this is the last point at which the two are still
     * distinguishable — after {@code FModel.initialize} a variant is an
     * ordinary card.
     *
     * @param extraCardSource an additional card-script directory, or null
     */
    public static synchronized void initialize(Path extraCardSource) {
        initialize(extraCardSource, true);
    }

    /**
     * As {@link #initialize(Path)}, but optionally skipping the copy.
     *
     * <p>{@code stage} exists because the custom-cards directory is a single
     * shared location and a collection run has a dozen workers starting at
     * once. Each copying thousands of scripts into it while the others walk it
     * during {@code FModel.initialize} is a plain race: a worker reads a
     * partial directory, its card database is missing variants the decks file
     * names, and every deck holding one is refused. A real run lost 350 of its
     * 354 workers that way, reporting names as unresolvable that were on disk
     * the whole time.
     *
     * <p>So exactly one process stages: {@code VariantSidecarMain}, which runs
     * to completion before the first worker starts. Workers pass
     * {@code false} and register only, which is all they need — registration
     * is what makes a variant's provenance keys name the {@code
     * variant-scripts} tree, and the scripts themselves are already in place.
     *
     * @param stage whether to copy the scripts, or only register their names
     */
    public static synchronized void initialize(Path extraCardSource, boolean stage) {
        if (initialized) return;

        Path forgeDir = findForgeDir();
        Path forgeGuiDir = forgeDir.resolve(GUI_SUBPATH);

        // The GUI base comes first, staging second, FModel.initialize last, and
        // the window between the first two is exactly one class-init wide.
        // stageCustomCards resolves ForgeConstants.USER_CUSTOM_CARDS_DIR, and
        // ForgeConstants' static initializer calls
        // GuiBase.getInterface().getAssetsDir() -- so staging before this line
        // throws ExceptionInInitializerError on a null interface, and every
        // worker of a variant run dies at startup before playing a game.
        GuiBase.setInterface(new GuiHeadless(forgeGuiDir + File.separator));

        if (extraCardSource != null) {
            com.pricepredictor.connector.effects.VariantRegistry
                    .registerAll(extraCardSource);
            if (stage) {
                stageCustomCards(extraCardSource);
            }
        }

        FModel.initialize(null, null);

        initialized = true;
    }

    /**
     * Copy an extra card-script directory into Forge's custom-cards directory.
     *
     * <p>Must run <b>before</b> {@code FModel.initialize}: the card database is
     * read once at startup, and a script that appears afterwards is invisible
     * for the life of the JVM.
     */
    private static void stageCustomCards(Path source) {
        if (!Files.isDirectory(source)) {
            return;
        }
        Path target = Path.of(forge.localinstance.properties.ForgeConstants
                .USER_CUSTOM_CARDS_DIR);
        try {
            Files.createDirectories(target);
            int removed = clearPreviouslyStaged(target);
            java.util.List<String> staged = new java.util.ArrayList<>();
            try (var scripts = Files.walk(source)) {
                for (Path script : scripts.filter(Files::isRegularFile).toList()) {
                    if (!script.toString().endsWith(".txt")) {
                        continue;
                    }
                    Files.copy(
                            script, target.resolve(script.getFileName()),
                            java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                    staged.add(script.getFileName().toString());
                }
            }
            Files.write(target.resolve(STAGED_MANIFEST), staged);
            System.out.println(
                    "Staged " + staged.size() + " custom card scripts from "
                            + source + " into " + target
                            + (removed > 0
                               ? " (removed " + removed + " from a previous run)"
                               : ""));
        } catch (java.io.IOException e) {
            System.err.println(
                    "Could not stage custom cards from " + source + ": "
                            + e.getMessage() + "; variant cards will not load");
        }
    }

    /** Names the last staging copied in, so the next one can take them out. */
    // Deliberately not ".txt": this file lives in the directory Forge
    // reads card scripts from, and Forge reads every .txt in it. A list of
    // filenames has no Name: line, so CardRules comes out with a null
    // mainPart and StaticData throws while building the whole database --
    // one manifest killing every Forge startup on the machine.
    static final String STAGED_MANIFEST = ".staged-by-effects.list";

    /**
     * Remove what a previous staging left in Forge's custom-cards directory.
     *
     * <p>That directory is persistent and shared, and staging used to only
     * ever add to it — so every variant run's scripts piled on top of the
     * last's. One machine reached 17,700 files from runs whose variant trees
     * were long gone, and a single malformed script among them is not a
     * skipped card: {@code StaticData}'s constructor throws on it, so every
     * Forge startup afterwards dies, including runs that have nothing to do
     * with variants.
     *
     * <p>Only files this code copied are removed, named by the manifest it
     * wrote. A card a person put there by hand is not ours to delete.
     *
     * @return how many files were removed
     */
    private static int clearPreviouslyStaged(Path target) throws java.io.IOException {
        Path manifest = target.resolve(STAGED_MANIFEST);
        if (!Files.isRegularFile(manifest)) {
            return 0;
        }
        int removed = 0;
        for (String name : Files.readAllLines(manifest)) {
            if (name.isBlank()) {
                continue;
            }
            if (Files.deleteIfExists(target.resolve(name))) {
                removed++;
            }
        }
        return removed;
    }

    public static Path findCardsFolder() {
        return findForgeDir().resolve(RES_SUBPATH).resolve("cardsfolder");
    }

    private static Path findForgeDir() {
        Path dir = Path.of("").toAbsolutePath();
        while (dir != null) {
            Path candidate = dir.resolve("forge").resolve(RES_SUBPATH);
            if (Files.isDirectory(candidate)) {
                return dir.resolve("forge");
            }
            dir = dir.getParent();
        }
        throw new IllegalStateException(
                "Could not find forge/" + RES_SUBPATH + " in any parent of " + Path.of("").toAbsolutePath());
    }
}
