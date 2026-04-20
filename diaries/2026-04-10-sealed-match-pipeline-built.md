# April 10, 2026 — Sealed match pipeline built

**TL;DR:** Spec 012 went from plan to running code in a single day. The main
work was wiring Forge's game engine into a supervisor/worker pipeline that
generates sealed match outcomes, then fixing two non-obvious bugs that only
showed up when actual matches ran.

The day started with three planning sessions — speckit tasks generation, spec
analysis, and the full planning session that worked out which Forge APIs were
actually reachable. The key research finding was that `SealedDeckBuilder`,
`FModel`, `LobbyPlayerAi`, and `GuiBase` all live in `forge-gui`, not
`forge-game`, so two new Maven dependencies (`forge-gui` and `forge-ai`) had
to be added to the connector pom.xml. The existing `ForgeEnvironmentInitializer`
also had to be rewritten: the old manual `StaticData` setup was not enough to
play games; the full `FModel.initialize() + GuiHeadless` path was required.

That rewrite uncovered a subtle double-counting bug in `RulesParser`. With
`FModel` producing fully-resolved `Card` objects, basic land faces now had
their mana ability present in `SpellAbilities` AND added again by the
synthetic land-mana block — giving Forest two green mana abilities and Bayou
three. The fix was a guard in `parseFace()` to skip mana SAs for faces whose
subtype triggers the synthetic block.

The land rebalancing algorithm for deck construction methods 2-4 turned out to
be wrong in a less subtle way. The plan said to delegate to Forge's
`LimitedDeckBuilder.addLands()`, but that method is private. The workaround
was to pass the nonland cards to `SealedDeckBuilder` and let it pick a
deck — except `SealedDeckBuilder` is a card-selection algorithm, not a
land-adder; it discarded most of the nonland cards and filled the rest with
lands, producing 27-32 land decks. The fix was to implement the pip-
proportional land distribution directly: count WUBRG mana symbols from the
spells, guarantee a minimum of two basics per required color, then distribute
remaining slots proportionally. That brought land counts to 16-19, centered
on 17 — the standard sealed configuration.

A third issue appeared after real matches started running: worker JVMs
degraded over time. Long Forge AI games can push a JVM into a slow state
without crashing it. The solution was to have the supervisor kill the
longest-running worker every 60 seconds and restart it with a fresh JVM. On
Windows, `subprocess.terminate()` only kills the parent process, so the fix
used `taskkill /F /T /PID` to kill the entire process tree.

The Forge set eligibility filter also got cleaner during planning. The initial
plan was to filter by booster pack size (14-15 cards), but Claude found that
`CardEdition.getBoosterTemplate("Draft") != null` is the semantically correct
check — draft and play boosters both map to the `"Draft"` key in Forge's
metadata, and collector boosters with 15 cards would otherwise sneak through
the size filter.

At the end of the day the pipeline was running and producing 40-card sealed
decks from randomly selected sets with correct land distributions.
