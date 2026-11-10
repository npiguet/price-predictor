# Fixing round plan (2026-11-10)

One pass over every confirmed finding in the three findings files, plus the record moments (tasks.md
T179) and the review decisions (T182–T191). Every fix is test-first: write the test the finding names,
see it fail, then fix. Line numbers in the findings files are from 2026-11-09; re-check them.

Finding ids: connector `A*`, `B*`, `C*`, `F*`, `L*`, `X1` (`connector-findings.md`); Forge `H1-*`,
`H2-*` (`forge-hooks-findings.md`); Python `R*`, `D*` (`python-collection-findings.md`).

## Where the attribution findings are fixed

The Forge review's verifier checked each placement; all are taken. Upstream footprint is not a
criterion (`effect-record-hooks` is our branch).

| finding | fixed in | how |
|---|---|---|
| A2 (a trigger's clause hook fires twice) | Forge | `AbilityUtils.resolveApiAbility` skips the clause notification for `sa.isWrapper()` but keeps the scope |
| A3 (clauses that did not run emit events) | Forge + connector | Forge evaluates `metConditions()` once at its stock place and passes `ran` to `onClauseResolved` (true on `handleUnlessCost`'s `cost == null` path); the connector skips `finish()` when `!ran` and stops calling `metConditions()` itself |
| A1 (charm modes' events in the last mode's half) | Forge | RESOLVED fires after the clause's own effect and before `resolveSubAbilities`, in the normal and both unless-cost paths, with a `finally` around the clause's own effect only; mode chains become sequential |
| C2 (mana record board after the mana) | Forge + connector | a hook at the top of `CostPayment.payComputerCosts`; the connector snapshots there, pairs by mana-ability identity (activations nest), discards on a failed payment, and falls back for triggered mana abilities (Wild Growth) |
| C3 (mana made by a resolving spell) | connector | `ManaChannel` takes `isManaAbility()` only; the resolution's half gets `mana_produced` |
| H1-2 (rewrite record shows the substitute) | Forge + connector | `ReplacementObserver` gains `onReplacing` before `call.run`; the connector snapshots there, stacked per call |
| record moments: effect half at resolution start (FR-042b) | Forge + connector | a "resolution starting" hook in `MagicStack.resolveStack` before the first clause |
| record moments: cost half before casting (FR-042a) | Forge + connector | a "casting starting" hook before the card moves to the stack, placed (like C2's) so the AI, the random seat, `playNoStack` and forks all pass it; the agent verifies the placement in the Forge source first |

## Work packages

Packages touching the same files run in one stream, in order. Forge changes are installed
(`mvn install -DskipTests` in `../forge`) before the connector builds against them.

### WP1 — Forge hooks (`../forge`, forge-game; stream J)
A2, A3, A1, H1-2's callback, C2's payment hook, the resolution-start and casting-start hooks, H1-4
(report `fired` from a count of already-collected positives, not changing Forge's list), H1-1 (blockers
for attackers of a battle, through `getDefenderPlayerByAttacker`), H1-3 (`spellCountered` only when
`tgtSA.isSpell()`), H1-5 (params built inside `contain`, batching only when observed), revert the dead
`AiController` playability hook to stock. Update `PatchHooks.REQUIRED` for each new hook.
Tests: Forge `EffectRecord*Test`s named in each finding.

### WP2 — Forge copier (`../forge`, forge-ai `GameCopier`, `Combat`, `Card`, `Player`; stream J after WP1)
- One structural ability mapper (H2-MAP: root role, spell-ability or trigger index on the host, sub-ability
  depth) replacing `counterpartAbility`, `findSAInCard` and the connector's `Forks.findInFork`.
- One `copyCardState(orig, copy)` for zone-independent card state (F5, F6): designations, ring-bearer and
  `theRing`, `classLevel`, `targetedFromThisTurn`, goad/detain/bestow, counters, SVars and every chosen
  value off the battlefield too, `markedColors`.
- H2-1 activation, resolution and chosen-mode tables, keys mapped (needs H2-MAP).
- F1 (`combatantsThatDealtFirstStrikeDamage`), F2 (null monarch/initiative field on removal, or
  `mapEffectCard` checks membership), F8 (state, `backside`, `transformedTimestamp`, SVars after
  `setState`, keyword cache), F3 (dungeon piece type and room), F4 (mutate/prototype timestamps, merged
  pile, `timesMutated`, Merged zone), F7 (phased-out permanents copied with their flags), F12 (phase
  handler counters), F11 (attackers' last-known copies), H2-4 (unmapped damage histories kept),
  H2-2 (`copy(newCard, false, true)` and `setCardState`), H2-9 (keep the copy's effect source), H2-8
  (resolved-effect layer rows), H2-3 (refuse pools holding special mana, or map its source), H2-11
  (remembered SpellAbility), H2-6 (delete the double count).
- The NEEDS-TEST Robot token (+1/+0, haste) test from `forge-hooks-findings.md`; fix if it fails.
- A per-copy "no counterpart" tally that `ForkFidelity` prints (H2 code quality).
Tests: each finding's copy test, in `GameSimulationTest` or a new `GameCopierFidelityTest`.

### WP3 — connector record moments and attribution (`forge-connector`, stream J after WP1)
T179 (FR-042a–f) with A7 (decided by the moments), B5, A4 (re-check: with the cost half snapshotted
before the cast, its stat events fall after the snapshot), A3/A2/A1's connector sides, C2's and C3's
mana records, H1-2's rewrite snapshot. Then A5 (drop `CARD_MILLED` from surveil), L1 (a draw only when
Forge's draw counter moved), A8 (a push while `StackBrackets.finishing` is no next action), A6 (count
segments from the mode clone), X1 (the test; fix if a trigger's events read `unresolved`), C1 (a fork
that ends the game runs `finishGame`), C4 (decision refs: source only), C5 (rules-only affordability,
no `MyRandom`, no AI memory side effects), C7 (reservoir per line or cap negatives only), C11 (count
failures per hook; investigate the `isCardAddedThisTurn` NPE), C6 (guard each boundary step), C9
(`AtomicLong`), C12 (test the `staticById` cache), F10 (stop a play-forward at cleanup with an empty
stack), the connector half of H2-MAP (`Forks.findInFork` replaced).

### WP4 — connector snapshots (`forge-connector`, stream J after WP3)
B1 with L2 (one `StaticSuppression` answering every channel: type, colour, keywords, set P/T and boosts,
controller, traits added and removed; Forge "without" reads where needed), B3 (`refs.x` only when the
chain references X, read once at the cast, never `Count$Random`), B7 (tier-1 cards mapped to
`game.getCardState`, last-known copy only as `L<id>`), B4 (fork notes seeded from the live game's;
`ForkFidelity` uses the fork's builder), B9 (`OptionalCost` → `Keyword` map), B6 (face order over the
converter's states; null for an absent state), B8 (the clone line-key test).

### WP5 — Python collection supervisors (stream P, parallel with stream J)
R2 (check the JAR, `java` and `--exclude-cards` before starting; stop after N consecutive failed starts;
backoff), R4 (re-check shutdown after appending a spawned worker), R5 (`workers=config.workers` to the
shard counter), R6 (validate and normalise `--probe-keywords`), D1 and D5 (coverage counts only the
record kinds `build-corpus` keeps; a continuous record credits its host), D4 (split cards by their
combined name, or `HoldoutCards.contains` tests each face; Java side in stream J's next build), D6
(credit through provenance `script_file`; the split-name deck test), D8 (deal decks round-robin), R9
(the SIGINT test), run-scoped deck file names (`coverage-decks.<run_id>.txt`), the counters reading
`generation`, and the perturbation oddities (`+N` values, `-N` floors, `DeckHints` swaps). Then T182
(random-seat held-out games), T183 (stratum composition), T190 (holdout source required).

### WP6 — variant rounds (T184–T189, T191; after WP5, Java parts in stream J)
T186 first (R1, R3, R8: `loadable.lst`, hash names, unstaging), then T185 (`rule-counts` carriers), T184
(`script_edit` field and the catalogue pass), T187 (untargeted round), T188 (steered round), T189
(intervention pass, Java), T191 (the 100,073-record variant).

### WP6-Java — the variant rounds' Java halves (stream J, after WP4)
`VariantSidecarMain` writes `loadable.lst`; `ScriptFiles.under` stages only `.txt` files with a `Name:`
line (fix `VariantRegistryTest.aNonScriptFileIsIgnored`); unstaging at round end if Python cannot do it;
per-game `aimed_at` from the steered deck file's rule column (`CollectorContext`); T189's intervention
pass. The Python agent's requirements (its halves landed in e1ca65a8, 76872cf3):
1. `VariantSidecarMain.LOADABLE_FILE = "loadable.lst"`; then delete the strict-xfail mark on
   `test_the_list_is_the_one_variant_sidecar_main_writes`.
2. `ScriptFiles.under` stages only `.txt` files holding a `Name:` line; fix
   `VariantRegistryTest.aNonScriptFileIsIgnored`; R1's test: `initialize(tree)` succeeds twice running.
3. Unstaging: none in Java (Python's `variant_staging.py` empties `.staged-by-effects.list`'s files).
4. Steered deck lines are `LABEL;SET_CODE;Card1|…|Card40;RULE` (label `variant`, set code `COVERAGE`);
   other decks keep three fields. `GeneratedDeck.parse` splits with limit 4 and holds a nullable rule
   (today's limit-3 split would fold `;RULE` into the last card).
5. `CollectorContext`: when either deck of a game has a rule, every record of that game carries
   `aimed_at` = both decks' rules, de-duplicated and sorted, instead of the run-wide `effect.aimed.at`;
   already-aimed records (directed forks) keep theirs.
6. A game is steered when either deck carries a rule (no new cap property).
7. T189: in a steered game, `InterventionScheduler`'s first pass forces the lines of the variant cards
   each player holds, before FR-057's pass; then T188's integration game (records carry the decks' rules).

### T191's loops (Forge AI; not fixed, awaiting the user)
Chain of Smog Variant 5437 wrote 100,083 gen-2 records (55,006 in one game): with both hands empty the
two AIs copy it forever (`SpecialCardAi.ChainOfSmog` copies whenever its own hand is empty), and the real
card loops the same way. Sage of Hours (`CounterNum$ 0`, 36,588), Crypt Rats and the en-Kor variants
re-activate free abilities. The Python-caused loops (a selector swap on `Duration$ Permanent`: Opal
Avenger, Hidden Predators, Veiled Crocodile) are fixed in 93906a7d.

## Progress
- WP1 done: Forge 8f105fb36cd; hook contract in `WP1-HOOKS.md`.
- WP5 done: commits 16c7a568 … 892ebb30, 8003c9a5. `build-corpus` did not give the validation strata the
  class mix; it now repeats validation records toward it (c2e7de82), never dropping one.
- WP2 done: Forge cfb2ae28564; copier API in `WP2-COPIER.md`.
- WP6 Python and catalogue parts done (T184, T185, T187 ticked; T186, T188 Python halves; 207 of 260
  card-level variants carry a `script_edit`, the rest listed with reasons in the agent's scratchpad).
- Running: WP3a (record moments, connector side of the new hooks).

## Order and gates

1. Stream J: WP1 → install Forge → WP2 → install Forge → WP3 → WP4 → connector install. Stream P:
   WP5, then WP6's Python parts. Heavy jobs (pytest, mvn, JVM runs) never overlap across streams:
   each runs through one machine-wide lock, a wrapper that creates a lock directory with `mkdir`
   (waiting while it exists), runs the job and removes it.
2. After each package: its targeted tests, the module's full suite, `ruff check` on touched files.
3. After all: tasks.md T174's full verification, then T181 (fork fidelity audit, `validate-corpus`,
   `field-coverage`, score-discard count) before any long run.
4. Before the first variant run: move `output/effects/variant-scripts/` aside (old `loadable.txt`).

## Left out

- Code-quality items are done only where they are the vehicle of a fix: the structural ability mapper,
  `copyCardState`, `StaticSuppression`, the clause-hook `ran` value, the per-copy no-counterpart tally.
  The rest of the code-quality lists wait.
- Not-a-bug items (A9, L3, L4, C8, C10, F9, F13, F14, H2-5, H2-7, H2-10, B2, B10, the depleted
  match-outcomes runs, the flags accepted without `--effect-records`, `sum_dropped_events`).
