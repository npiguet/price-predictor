# Draft agent — game-played evaluation

Normative spec for `play-draft-games`: a command that plays Forge matches between decks
drafted in the same pod and writes them in the sealed match-outcome format, so that
`scripts/analyze_winrates.py` reports per-agent win rates over them. Rationale, the
measurement gap this closes, and the sizing arithmetic live in
[`../experiments/2026-08-09-draft-agent-gen4-online-grpo.md`](../experiments/2026-08-09-draft-agent-gen4-online-grpo.md).

# 1. The command

`play-draft-games` samples pairs of decks drafted in the same pod from a `drafts.jsonl`
corpus, plays a match between each pair in Forge, and appends the results.

It does not draft, build, or score decks; it consumes a corpus produced by
`generate-draft-data`. It does not report: tallying is `scripts/analyze_winrates.py`,
unchanged (§ 8). The one deck it does build is the Forge-native case of § 5.

# 2. Scope

In scope

- The command, its sampling contract (§ 4), the Forge-native deck source (§ 5), and its
  match contract (§ 6).
- The output file (§ 7).

Reused unchanged

- `drafts.jsonl`, and the drafting and scoring paths that produce it.
- Forge's own sealed deck builder, and the pool reconstruction the draft package already
  performs, both used as-is by § 5.
- Forge match execution, from the sealed evaluation path.
- `scripts/analyze_winrates.py` and the sealed `match-outcomes.txt` format.

# 3. Inputs

| Input | Meaning |
|---|---|
| `--drafts-path` | Corpus to sample from; default `output/draft/drafts.jsonl` |
| `--run-id` | Optional, repeatable; restricts sampling to records with a matching `run_id`. Absent, the whole corpus is used |

The `drafts.jsonl` format is specified in
[`2026-05-28-draft-agent.md`](2026-05-28-draft-agent.md) § 4. A record's boosters share one
`set_code`, which is the set of every match drawn from that record.

The startup echo names the `run_id`s selected, or states that the whole corpus is in scope.

# 4. Pair sampling

- A pairing is drawn by choosing a record uniformly at random, then two distinct seats of
  that record uniformly at random. Pairs never span records.
- Mirror pairs, where both seats carry the same `agent` label, are excluded by default.
  `--include-mirrors` retains them. When they are excluded, a drawn mirror is discarded and
  the draw repeated.
- With mirrors excluded, records whose seats all carry one label are skipped, so the retry
  terminates.
- Draws are independent, so a pairing may be drawn more than once within a run or across
  runs, and the number of pairings per matchup follows the corpus composition rather than
  being equalised.
- `--n-pairings` sets how many matches this run records before it stops. Absent, it plays
  until interrupted.
- A run is not reproducible. Pairings are drawn concurrently and complete out of order, and
  no seed is offered.

# 5. Forge-native decks

A seat's deck normally comes from the corpus. `--forge-native-fraction` diverts a share of
the Forge reference seats to Forge's own sealed deck builder instead, so the same drafted
cards can be compared under two builders.

- `--forge-native-fraction` is a fraction between 0 and 1, default 0. At 0 the feature is
  inert and no seat is diverted.
- Seats labelled `forge-full` are selected independently with that probability. Selection
  happens once per seat when the corpus is loaded, so a seat is diverted in every pairing it
  appears in or in none.
- A diverted seat's deck is built by Forge's own sealed deck builder from the seat's drafted
  pool, reconstructed from the record. The deck recorded in the corpus is not used.
- A diverted seat's label becomes `forge-native`. It is a distinct label everywhere: in the
  output rows, in mirror exclusion, and in the tally.
- Because the labels differ, a pod holding both a diverted and an undiverted `forge-full`
  seat can pair them against each other, comparing the two builders on the same pod.

# 6. Match execution

- Each pairing is played as one best-of-`--best-of` match, default 3. `--best-of` is a
  positive odd integer.
- `--workers` sets how many Forge worker processes play concurrently, default 12.
- A pairing whose match fails is skipped and produces no output rows. A crashed worker is
  restarted.
- Workers are recycled as the sealed `match-outcomes` supervisor recycles them. A pairing
  whose match is cut short by a recycle is skipped like any other failure.
- Rows are appended as each match completes, so an interrupted run retains every match
  already played and a further invocation adds to it.
- Progress reporting follows the sealed `match-outcomes` supervisor: a status line every 60
  seconds, in the form
  `[{elapsed}s] {n} matches completed | {rate} matches/min | {alive}/{workers} workers alive`,
  and the same worker-lifecycle lines on start, crash-restart and shutdown.
- A summary on exit, including on interrupt, reports matches played, elapsed time, and the
  output path. A pairing whose match fails is not counted: the worker that drew it simply
  draws another, so a failure costs time rather than data.

# 7. Output

`--output-path`, default `output/draft/draft-games.txt`. Append-only, in the sealed
match-outcome format, specified in
[`2026-03-28-sealed-deck-picker.md`](2026-03-28-sealed-deck-picker.md) § Phase 0 Step 4.
One row per match.

`method_A` and `method_B` carry the two seats' `agent` labels, taking the role the sealed
pipeline gives a `build-decks --label`. Every other field takes its documented meaning.

# 8. Reporting

`python scripts/analyze_winrates.py <output-path>` runs unchanged over § 7 and covers
per-agent win rate, the head-to-head matrix keyed by agent label, and win rate by deck
colour count, colour presence, creature count, and average nonland mana value. It also
simulates each match at shorter lengths than it was played, which carries information when
`--best-of` exceeds 1.

# 9. Operator surface

```
python -m draft play-draft-games
    [--drafts-path PATH] [--run-id ID]...
    [--output-path PATH]
    [--n-pairings N]
    [--best-of N]
    [--include-mirrors]
    [--forge-native-fraction F]
    [--workers N]
```

The output is append-only, so a further invocation adds to whatever a previous one left.

Startup validation runs before any game is played, and checks that:

- the corpus exists and parses;
- at least one pair survives the `--run-id` and mirror filters;
- `--best-of` is a positive odd integer;
- `--forge-native-fraction` is between 0 and 1.

Exit codes

| Code | Meaning |
|---|---|
| 0 | Requested matches recorded, or clean interrupt |
| 2 | Validation failure, missing file, or bad flag |
| 130 | Interrupt before the first match was played |
