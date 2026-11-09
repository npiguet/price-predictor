# Where collection bugs have hidden before (hints for reviewers)

These are bug CLASSES found in past collections of this pipeline, with one instance each. They say where to
look and what kind of check catches such bugs. They are not a list of open bugs, and the instances named
here have been fixed unless marked open. Look for NEW members of each class in the code you review.

## Found on 2026-11-09 (fork copies)
- **The score check compared noise.** The AI's position evaluator simulates the coming combat in a copy,
  with attack and block choices drawn from Forge's global `MyRandom`. Two scorings of the same board
  differed. Lesson: anything that draws from a global random stream inside a check, or inside the
  collector, perturbs the live game and makes the check non-deterministic.
- **The fork copy (`GameCopier`) silently dropped state the model reads.** About 28% of forks differed
  from the live board. Missing:
  - per-card attack, block and damage history this turn;
  - the game's damage table and the cards that left the battlefield this turn;
  - each zone's cards added this turn (the copy logged every card it placed as just arrived from the stack);
  - per-player turn counts (surveil, sacrifices, discards and others);
  - effect cards' triggers, replacement effects and colour (monarch, regeneration shields);
  - what made an effect card;
  - must-block requirements;
  - a disguised or cloaked card's face-down ward;
  - delayed trigger ids (changed on copy).

  Still open: dungeon cards are unmapped in the copy ("Couldn't map The Initiative"); unearth; Max Speed's
  face; mutate components; "becomes X/Y" animate effects; a "doesn't untap" flag; one land's chosen mana
  colour. Lesson: any per-turn or per-game state the engine keeps outside the zones is suspect. Compare
  against what `Player.onCleanupPhase`, `Card` and `Game` reset or hold.
- **Interventions were taken at a boundary where their fork could never pass the check.** Each one used up
  budget and wrote nothing.
- **Snapshot code compared players with `==`.** A copy carries the live game's last-known card copies,
  whose controller is the live game's player object, so identity comparisons fail in forks. Look for other
  `==` or identity comparisons of Players and Cards in snapshot and collector code.
- **A test staged files into Forge's shared custom-cards directory and never removed them.** Every later
  worker loaded the stray card.

## Fix commits since 2026-10-21 (each subject names a bug that existed)
Event attribution and brackets:
- one bracket per stack item;
- file each event by its cause;
- a cast made mid-resolution waits for the resolution to end;
- a mid-resolution cast's payment and the check after it are nobody's half;
- the item below gets nothing of a resolution's finishing;
- nothing in an item above's aftermath joins the forced record;
- a forced record keeps what happens while it tops the stack;
- a recast's window ignores old marks;
- a land ends a given-up cast;
- an abandoned cast stops counting;
- a combat record holds its own step's assignment and only its own events;
- outcome notes name only their own subjects;
- record a card's first stat change;
- a record's stat changes are measured from its own snapshot.

Forced and fork records:
- forced resolutions read their state where live does;
- a forced resolution's state is read once its targets and X are drawn;
- a fork files a repeated state transition once, as the live bracket does;
- a modal fork recognises a repeat within one mode;
- a forced death is checked for a repeat in the half it joins;
- only the forced charm's own modes cut a forced record's halves;
- the hand forces only what is played from the hand.

Other:
- dedup records by their acting line, and cap a trigger line before its snapshot;
- one referenced tier for observed and forked snapshots;
- record combat legality from the engine's declarations;
- one rules-only playability verdict for both seats;
- a failed zone decides the decision bucket;
- every hook comes down even when closing a recording throws;
- a game's recording opens inside the try that closes it;
- a null nested in an event param renders as JSON null;
- colourless mana in snapshots;
- the held-out coverage round credits a game as build-corpus does.

## From the gen-1 and gen-2 collections
Collectors and brackets:
- An event arrives with no record open and is silently dropped. Example: gen-1 lost every state-based death
  after a resolution, so burn targets were damaged and never dead. Caught by probes and the dropped-event
  counter.
- A held resolution half is closed early by an unrelated next action. Example: a dies-trigger stacked
  between state-based passes stole later deaths.
- A state-based death with no held half (a -1/-1 counter shrinking a creature to 0) joined whatever cast
  bracket was open.
- A phase test by substring. `contains("COMBAT_DAMAGE")` misses `COMBAT_FIRST_STRIKE_DAMAGE`, so the whole
  first-strike step was lost for two corpus generations.
- Planned or what-if engine queries recorded as real:
  - legality asked while the AI was still planning;
  - a trigger for a tap the AI was only planning;
  - the "which triggers would fire" query notified the hook;
  - statics evaluated twice.
- The wrong bracket receives an effect:
  - a combat trigger's draw or token was written into the combat record;
  - a combat record kept absorbing events after a spell went on the stack.
- An effect note is attributed to the wrong subject:
  - a note with no subject (coin flip, token, vote) marked the acting card;
  - one note for an effect that made several kinds of object;
  - a 0 note for a prevention that prevented nothing;
  - prevented damage did not name its victim;
  - a reveal or scry did not name the library owner.
- A snapshot taken at the wrong moment. A combat record's snapshot at the first damage event already shows
  the first hit. Stats Forge applies before announcing them need a re-read.
- A per-path assumption about event sources. Probe-fork damage events lacked `params.cause` and reported
  `combat: false`.
- Verdict bits recorded the AI's choices instead of the rules:
  - `can_play` without timing;
  - a hook after the AI's own filter, so the AI never wrote `can_play = false`;
  - target legality read from what the AI chose;
  - affordability drew from the shared random stream.

Snapshots and fields:
- A field is written but never varies: for example `"blocking":[]`, or a duration setter nothing calls.
  Caught by `field-coverage`.
- A field the reader expects but the collector never writes: mana events lacked `params.amount`, and some
  `mana_by_color` maps are empty (open).
- A snapshot misreports derived state. A set-base P/T was folded into the boosts (Gigantoplasm).
- A field's value space changed inside one corpus: `token_script_id` went from the printed name to the
  script stem.
- Snapshot depth varied by collector. Tier 4 appeared only on interventional records, a perfect proxy for
  a hidden flag.
- Duration defaults were wrong. A pump with no `Duration$` was recorded as instant.
- A `continuous` record leaked its own static's work into its snapshot.

Forks and copier:
- Fork-copy fidelity: delayed triggers, end-of-turn commands and controller timestamps were not carried.
  A command closure holding a live reference can leak into the real game.
- A fork taken at a bad point:
  - forking inside the AI's evaluation re-enters the engine;
  - paying costs through the AI produced only land drops;
  - at the beginning-of-combat boundary the score check always fails.
- Mirror pairing:
  - a probe's `mirror_of` needs the real combat record's id;
  - reconstructing the pairing from board state breaks on two identical combats;
  - a probe naming only the keyword is ambiguous.
- Fork side effects on the live match:
  - drawing from the match's random source shifts later seats;
  - a discarded fork must still spend its budget;
  - a fork that throws must leave no lines;
  - hook dispatch must go by game identity, because the AI runs on its own thread.

Sampling, caps and the random seat:
- Sampling before de-duplication. A question asked K times survived with probability 1-(1-r)^K.
- Take-first bias. The first mana activation is always turn one; reservoir sampling needs the snapshot
  taken at activation time.
- Correlated seeds. Consecutive LCG seeds gave near-identical first draws, hence SplitMix64.
- Runaway play:
  - the random seat never passes while a free ability is activatable (2,190 activations in one turn);
  - power reached +2,098,760.
- A flag read by one side only. `effect.legality.rate` was read by the JVM and settable by no command. In
  the JVM probe keywords default to empty, and an empty string disables probes.

Ids, sharding and restarts:
- Ids not unique across JVM restarts. Counters were per worker slot, not per lifetime, so 71-75% of
  `record_id` values repeated.
- A file or shard boundary treated as a game boundary:
  - games are cut by kills;
  - the last 255 records of a lifetime are lost;
  - `taskkill` runs no shutdown hook;
  - a held reservoir dies with its JVM.
- Leftover or misfiled run data after a crash or restart; two drivers running at once.

Coverage and pool generation:
- Cards that cannot be put in a deck were treated as top priority. Planes and schemes made 71% of decks
  fail, and retirement then charged ordinary cards.
- Variant count too low for the number of deck slots.
- Collection-time guards fired on benign data: Undercity triggers, draw- and mill-heavy decks.

Corpus build and join:
- A record key does not join the converted text:
  - a charm cast through Flashback;
  - abilities a clone gains through `AddAbilities$`;
  - variant keys under `cardsfolder/` (Alchemy "A-" cards);
  - multi-trigger charm lines keyed under only the first trigger;
  - an unjoinable decision candidate.
- Converter numbering unstable across runs (specialize faces).
- Engine strings and model enums mismatch: combat phase names, counter types, protection from a colour,
  landwalk.
- Holdout leaks: tokens and variants carrying held-out texts; a coverage round over-covering.
- The build plan counted records the build would not train on.
- A check ignored a dimension (gate 2 ignored the damage step); a count went negative.

## From the gen-3 sessions (2026-10-30 to 2026-11-09)
Collectors and brackets:
- The state-based check window ended at the wrong time. A dies-trigger between passes stole later deaths.
  Fix: end the window when Forge's view tracker unfreezes (`HeldResolutionWindow`, `isStateBasedLeave`).
- End-of-game `player_won` and `game_lost` land in a separate rescue record, not the lethal resolution's.
- Event kinds never written for a record kind. Gen-2 resolution records never held `sacrificed`,
  `destroyed`, `face_change`, `player_lost` or `spell_cast`; `RunParamNormalizer` writes them only to
  trigger and rewrite records.
- Bus zone changes carry no library position.
- Turn-action events (draw, untap, day/night) had no record of their own.
- Candidate enumeration was pre-filtered by the engine's timing filter, so "can't play, wrong turn" was
  never seen. `getAllPossibleAbilities` drops unplayable abilities. The random seat never draws a land.
- Open worry: two-stage decision sampling may drop real attack declarations.

Snapshots:
- A snapshot leaked the answer into its own input. `CardLines.addTrait` compared a `Long` with Forge's
  `int` id using `equals`, which is always false. Look for other boxed-type comparisons.
- `StatDiffer` double-counted or kept stale stat changes. A state-based pass applies statics before Forge
  publishes them; changes that set base P/T emit no `pt_change`.
- A card that was never read lost its first change. The fix was a re-read at every bracket open.

Forks, `GameCopier` and the score check:
- The copier lost per-card or per-turn state. Before today:
  - each card's "turn it arrived" reset, so everything read `entered_this_turn` and graveyard permanents
    descended twice;
  - draw counts;
  - the discarded flag;
  - a card's cast-with spell and cast-from zone, so `costs_paid` was null on every cast permanent in a fork;
  - completed dungeons;
  - gained-control links.
- The copier lost pending effects: delayed triggers, until-leaves closures (Sneak Attack's sacrifice never
  fired in a fork) and `ExiledWith`.
- The copier dropped the meld link, so every fork with a melded permanent was discarded. `Card.equals`
  compares id and class only, so the card map can match a stale last-known copy.
- The score check threw at a boundary (declare blockers with an unset blocked flag), so every fork there
  was discarded. Fixed today in Forge `f0643b77987`.
- Forks that throw write nothing, by design. Such losses are invisible unless logged.
- Fork randomness:
  - copies drew seeds from the match's random source;
  - fork AI draws from the shared `MyRandom`;
  - `clonePlayer` kept the plain AI, so the fork's random seat was never attached.
- Readers refused valid protection-fork records (interventional records of any kind, with `link_id`).

Directed forks and interventions:
- Preconditions rarely or never satisfiable (Eon Hub skips, unleash).
- `is_probe` was too broad: a directed fork on a combat record matched it.
- Interventions only resolve the stack and do not play the turn on.

Sampling, counting and provenance:
- The holdout counter credited only held-out texts while the round targeted a superset, so units could
  never satisfy.
- Random-seat matches write no sealed row, so "matches completed" undercounts them.
- Rule counts folded aimed (directed and coverage) records into natural counts.

Rules catalogue and detectors:
- One scenario's labels applied to several rules' detectors (33 scenarios).
- Detectors tag the wrong records (open):
  - a battle with a pending trigger at defense 0;
  - counters the record places not added to toughness;
  - wither and infect damage counted as marked damage;
  - umbra armor, regeneration and deaths redirected to hand or library missed;
  - modal deaths filed in the last mode's half;
  - entity-level rules tagging the whole record.
- The rules-to-fields map names outputs no tagged record carries. These are silently skipped or read as
  zero.

Corpus build:
- The codec ignored an old key (`rule_entities`), so old corpora are stale.
- `count_shard_records` skips blank lines for `.gz` shards but not for plain ones.
- A Forge AI null pointer in `SpellAbility.getDividedValue` failed two match generations.
- Damage-grid amount edits changed only `NumDmg$`, which the prose surface never reads.
