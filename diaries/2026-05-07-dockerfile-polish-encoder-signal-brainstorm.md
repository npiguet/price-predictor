# May 7, 2026 — Dockerfile polish and encoder signal brainstorm

**TL;DR:** I cleaned up the Dockerfile so it actually works on TrueNAS
and extended the card-winnability spec to include a second regression head.
The second half of the day turned into an open-ended brainstorm about what
signals could tell the encoder "this card is good."

The Docker work started from a simple observation: the existing Dockerfile
built everything inside the image, but I didn't want that — I wanted to COPY
Forge and the forge-connector JAR in pre-built from the host. Claude walked
through the simplification and flagged immediately that the host layout
(Forge as a sibling of the project under `IdeaProjects/`) had to be mirrored
in the image, not folded under `/app`. The code was already resolving
`project_root().parent / "forge"`, so the fix was moving the COPY target to
`/forge` rather than `/app/forge`. A path bug that would only have surfaced
at runtime.

The question of how to get the `../forge` directory into the Docker build
context at all led to BuildKit named build contexts (`--build-context
forge=../forge`), which turned out to be the cleanest solution — no symlinks,
no junction points, no parent-dir-as-context. I hadn't known about that
feature before.

The rest of the Docker refinements were mostly about layer-cache ordering:
`resources/` (large MTGJSON dumps, rarely changed) moved to the top of the
volatile layers so that rebuilding the Python code or the JAR doesn't
re-transfer hundreds of megabytes. Once the container was actually running on
TrueNAS I hit the classic Python-buffering-under-docker-logs problem — nothing
appeared in `docker logs -f` because stdout was being fully buffered on a
non-TTY pipe. `ENV PYTHONUNBUFFERED=1` in the Dockerfile fixed it.

The second thread started when I noticed that `print_card_winrates.py` had no
way to surface cards the Forge AI never casts — Shackles was in 637 decks and
was never played once. I asked for an "unplayed count" column and an ascending
sort tiebreaker on it. That observation led directly to the question of whether
the encoder's regression target should explicitly account for never-played
cards.

The spec at that point used a single ratio, `wins_when_played /
wins_when_in_deck`, which mapped both "never played" and "genuinely bad
card" to the same label 0. I decided I wanted two heads instead: one for a
signed net-influence score, `(wins_played − losses_played) /
(wins_in_deck + losses_in_deck)`, and one for a played rate,
`(wins_played + losses_played) / (wins_in_deck + losses_in_deck)`. Claude
updated the spec accordingly, widening the `cards-win-rates.txt` format to
carry both heads' raw and shrunk values.

After the spec was updated, the conversation turned into a systematic brainstorm
of signals that could be extracted from the match data. Claude proposed five
ideas; I pushed back on several. Pool-conditional pick rate and card-card
co-occurrence both assume the Forge deck builder knows what it's doing — but
the 2-3k never-played cards are direct evidence it doesn't. Per-set shrinkage
toward a set-level prior sounded useful until I remembered that in sealed all
cards only ever play within their own set anyway, so the set-level mean
wouldn't be more informative than the global prior.

Two ideas held up well enough to log in `future-experiments.md`. The first is
a play/draw split: instead of one net-influence head, track separate counters
for games where the card's owner was on the play versus on the draw. The
labels become a (score_play, score_draw) pair instead of a single scalar,
which forces the encoder to learn a tempo-vs-catch-up axis the current label
collapses. The algebraic point that made me comfortable dropping the original
head entirely is that it's a linear combination of the two split heads — it
adds no gradient that isn't already in the pair.

The second is a cast-lift head: `P(win | card was cast) − P(win | card was in
deck but not cast)`. The example that made it concrete: a mediocre auto-include
might have head 1 = +0.10 because it lands in slightly-above-average decks
(because forge-best picks it when it shouldn't), but its lift is 0 — the deck
wins at the same rate whether the card hits the table or not. Head 1 can't see
that; cast-lift isolates the outcome change attributable to the act of resolving
the card. Claude also verified that "downforce" — the same formula over the
loss side — is just `-lift` algebraically, so there's no independent signal
in it.

The third idea that went in was MLM (masked language modeling): randomly mask
tokens in card text and train the encoder to reconstruct them as an auxiliary
loss. Claude explained the mechanic using an MTG-flavored example: given
`creature - human [MASK]   3/3   when ~ enters, you gain 2 life`, a model
that understands the game would guess `cleric` or `soldier` from the lifegain
trigger. The point that made it attractive to me is that it gives every card
dense gradient from the card text corpus itself, independent of how many
games that card appeared in — the long-tail coverage problem the regression
heads can't solve on their own.

By the end of the session the spec and the experiments file were committed.
The actual implementation — the new aggregation pass, the wider output format,
the two additional regression heads — was explicitly left for a later session.
