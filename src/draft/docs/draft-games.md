# Draft games

Rules to respect when running or changing `play-draft-games`, comparing agents by games won, or touching `forge-native`. Flags and exit codes are in `--help`; the command is specified in `specs/2026-08-09-draft-game-evaluation.md`.

## How a run works

- `play-draft-games` is the first measure of a draft agent that is not a function of the frozen scorer: games won.
- The corpus is projected once into a flat seat table, and the Java `DraftGameWorkerMain` workers are autonomous. Each draws a pod, draws two distinct seats of it, rejects a mirror pair unless `--include-mirrors`, plays one best-of-`--best-of` match and appends a row. Python issues no work and never parses a game result.
- Unless `--include-mirrors`, a pod whose seats all share one label is dropped when the table is loaded (`SeatTableIndex.canYieldPairing`). That is what makes the mirror-rejection loop terminate.
- Sampling is with replacement, and matchup frequencies follow the corpus composition; they are not equalised. There is no seed and a run is not reproducible.
- `--n-pairings` counts the rows this invocation recorded, not a pre-existing file. Without it the run plays until Ctrl-C.
- A pairing lost to a failure or a recycle is simply redrawn. Nothing was reserved for it, so there is no skipped count.
- Progress, restart and recycling come from the shared `ForgeWorkerPool`.
- Reporting is not part of the command: tally with `python scripts/analyze_winrates.py <run dir>/draft-games.txt --cards-path <converted cardsfolder>`.
- The fat JAR must be rebuilt first (`cd forge-connector && mvn install -DskipTests`).

## `forge-native`

- `--forge-native-fraction` diverts that share of `forge-full` seats to Forge's own deck builder. The choice is made once as the table is written, so a seat is diverted in every pairing or in none.
- A diverted seat reports under the distinct label `forge-native`. A shared label would average the two builders into one win rate and destroy the comparison. A pod holding one of each pits the two builders against each other on the same drafted cards.
- The builder is `BoosterDeckBuilder`, not `SealedDeckBuilder`. The two differ only in where the colour pair comes from: derived from the pool for a sealed pool, taken from the drafter for a drafted one, because a drafted pool was assembled under a two-colour commitment that steered every later pick.
- `DeckBuilder.buildDrafted` recovers that commitment by replaying the seat's picks, in pick order, through a `DeckColors` (`forge.gamemodes.limited.DraftedColorCommitment`, which lives in Forge's package to reach a package-private constructor).
- Re-deriving colours from the finished pool can land on a pair the pool was never drafted for, leaving too few on-colour playables and a deck padded out with basic lands.
