# Quickstart: `play-draft-games`

**Feature**: `022-draft-game-evaluation`

## Prerequisites

- A sibling MTG Forge checkout at `../forge`, built with `mvn install -DskipTests`.
- The fat JAR built after this feature lands:
  `cd forge-connector && mvn package -DskipTests`.
- A draft corpus whose pods contain at least two distinct agent labels. The yardstick
  corpus at `output/draft/yardstick-drafts.jsonl` qualifies, its pods seating `gen4`
  alongside `gen1` and `forge-full`.

## Measure an agent against its field

```bash
python -m draft play-draft-games \
    --drafts-path output/draft/yardstick-drafts.jsonl \
    --output-path output/draft/draft-games.txt \
    --n-pairings 500 \
    --best-of 3
```

Expect a startup echo, then a status line each minute:

```text
[60s] 14 matches completed | 14.0 matches/min | 12/12 workers alive
```

Then tally:

```bash
python scripts/analyze_winrates.py output/draft/draft-games.txt
```

This reports per-agent win rate, the head-to-head matrix keyed by agent label, and win rate
by colour count, colour presence, creature count and average mana value.

## Run until you have enough

Omit `--n-pairings` and stop with Ctrl-C when the tally looks stable. The output is
append-only, so tally it whenever you like and start again later to add more.

```bash
python -m draft play-draft-games --drafts-path output/draft/yardstick-drafts.jsonl
```

## Evaluate one generation from a shared corpus

The default corpus accumulates records from every run that wrote to it. Scope with
`--run-id`, repeating the flag for more than one:

```bash
python -m draft play-draft-games --run-id 3f2a…c91 --n-pairings 300
```

The startup echo names the run ids in scope, or says the whole corpus is.

## Single games instead of matches

`--best-of 1` plays one game per pairing — more distinct pairings per hour, noisier per
pairing. The tally's shorter-length simulation columns carry no information at this
setting.

```bash
python -m draft play-draft-games --best-of 1 --n-pairings 2000
```

## Separate the drafting from the deck building

The `forge-full` seats drafted their cards, but their decks were assembled by this
project's builder. `--forge-native-fraction` diverts a share of them to Forge's own sealed
deck builder, working from the same drafted cards:

```bash
python -m draft play-draft-games --forge-native-fraction 0.5 --n-pairings 800
```

Diverted seats report as `forge-native`, so the tally shows both variants:

```text
method                instances     Bo3
---------------------------------------
gen4                        412   58.1%
forge-full                  201   49.3%
forge-native                198   44.8%
```

The gap between the two `forge-*` rows is the builder's contribution, holding the drafting
fixed. Both variants occupy the same pods, so they also meet head-to-head in the pairwise
matrix. The default of 0 leaves the feature inert.

## Reading the summary

```text
=== play-draft-games ===
matches played   500
elapsed          38m 12s
output           output/draft/draft-games.txt
```

Failed matches are not counted. A worker whose match fails simply draws another pairing, so
a failure costs time rather than data. What to watch instead is the status line: if the
match count stops climbing, or the live-worker count sits well below the configured one, the
workers are failing rather than playing — check a worker log before trusting the tally.

## Notes

- Mirror pairings, where both seats carry the same label, are excluded unless
  `--include-mirrors` is given. They cannot separate two agents.
- Matchup frequencies follow the corpus composition; they are not equalised. Labels with
  more seats appear in more matches.
- The same pairing can be drawn more than once. Sampling is with replacement.
- This command never writes to `output/sealed/`. Draft games do not enter scorer or encoder
  training data.
