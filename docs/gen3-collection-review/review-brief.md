# Review brief: forge-connector, effect-record collection (feature 025, "gen-3")

You are one of several independent reviewers of the Java code that collects effect records from Forge games
for training an ML model of what Magic: The Gathering abilities do. The records are the training data, and a
multi-day re-collection is about to start, so **the main goal is to find bugs that make collected records
wrong, missing or misattributed before that collection runs.** The secondary goal is a code-quality review
under the project's review rubric.

## Rules
- **Read-only.** Edit, create or delete no file anywhere. Run no builds, tests, JVMs, Maven or pytest. Other
  heavy jobs share this machine. Read-only shell commands (grep, find, cat) are fine.
- **Fresh eyes.** Do not read anything under `C:\Users\nicol\.claude\` (memories, plans, transcripts), the
  repo's `experiments/` or `diaries/` folders, or git history (no `git log`, `blame`, `show` or `diff`).
  These hold earlier verdicts that would bias you. Exception: the hints file named below.
- **Allowed reading:**
  - the code under review, its tests (`forge-connector/src/test/java/...`) and any other code in either
    repo;
  - the Forge engine source in the sibling checkout `C:\Users\nicol\IdeaProjects\forge`
    (`forge-game`, `forge-ai`, `forge-core`), branch `effect-record-hooks`, which carries this project's
    engine hooks and its `GameCopier` changes;
  - `CLAUDE.md` and `GLOSSARY.md` files, and `specs/`.
- You may delegate mechanical lookups (finding call sites, reading Forge methods) to Sonnet or Haiku
  subagents. Never use the Fable model. Give such helpers the same read-only and fresh-eyes rules.

## Read first (domain and contracts)
1. `src/effects/CLAUDE.md`: sections "Collection" (and its sub-sections on short-lived workers, the
   engine hooks and forks) and "File formats". This is the authoritative description of what the
   collectors must do and why.
2. `forge-connector/CLAUDE.md`.
3. `specs/025-ability-effect-model-gen3/contracts/record-schema-delta.md` and
   `specs/023-ability-effect-model/contracts/record-schema.md`, the record schema the code must produce.
4. The hints file `C:\Users\nicol\IdeaProjects\price-predictor\docs\gen3-collection-review\review-hints.md`.
   It lists bug CLASSES found in past collections of this pipeline, one instance each, with how each was
   noticed. Treat it as a map of where bugs have hidden before. Look for new members of each class in
   your slice. The listed instances are fixed unless marked open, so don't report them again unless the
   code shows they are not fixed.

## Bug hunting (primary)
For your slice, look for defects where the code would produce records that disagree with what actually
happened in the Forge game, or with the record schema and the `CLAUDE.md` contracts. Check in particular:
- **Attribution:** does every event reach the record whose ability caused it, and only that one? Think
  about:
  - nested stack items, triggers stacked during state-based passes, replacement effects, modal (charm)
    resolutions and copies of spells;
  - mid-resolution casts, costs and payments, combat damage steps (first strike and regular);
  - game end, and a JVM killed mid-game.
- **Snapshot timing and content:** is the state read at the moment the contract says, and does it contain
  the answer it is supposed to predict (leakage)?
- **Forge semantics:** check claims in code comments against the Forge source. Look at phase and step
  names, `PhaseType` comparisons, ids (int versus long, boxed `equals`), `Card.equals` (id and class,
  which matches last-known copies), and `==` on `Player` and `Card` objects that may come from a copied
  game.
- **Live-game side effects:** does collection change the game it observes? Look for draws from Forge's
  global `MyRandom`, mutation of live objects, and triggers fired by queries.
- **Silent loss:** caught exceptions that drop records without a trace, caps and budgets spent with
  nothing written, early returns and null paths.
- **Determinism and threading:** the AI evaluates on its own thread, static and thread-local state, and
  seeding.
- **Fields never varying or never written:** a schema field the code path can never set.
- Off-by-one errors and boundaries: the first or last record of a game, an empty stack, an empty board,
  zero-damage cases.

For each suspected bug, trace it concretely: name the inputs or game situation that trigger it, what the
record would wrongly contain, and the file:line evidence on both the connector and Forge sides. Rate each:
- **severity**: `corrupts` (wrong content written), `loses` (records or events silently missing),
  `crash` (worker dies or fork discarded) or `minor`;
- **confidence**: `high` (traced end to end), `medium` or `low` (needs a test to confirm).

Prefer a few well-traced findings over many speculative ones, but do list plausible `low` ones briefly
too, since a verifier will try to confirm or refute each.

## Code-quality review (secondary)
Read `C:\Users\nicol\IdeaProjects\price-predictor\.claude\commands\review.md` and apply its rubric to your
slice. Ignore its instructions to enter plan mode and to ask the user. Just produce the findings. Keep this
part brief: the strongest findings only, with location, problem and suggestion.

## Output (your final message)
1. One paragraph summarising the state of the slice.
2. **Bugs**, ordered by severity then confidence. Each with: title; severity and confidence; file:line;
   the trigger scenario; what goes wrong in the record; the evidence chain, including the Forge source
   where relevant; and a suggested fix in one or two lines.
3. **Code-quality findings** under the rubric: location, problem, suggestion.
4. Anything you could not resolve that a verifier with test access should check.
