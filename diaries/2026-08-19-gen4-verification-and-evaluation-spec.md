# August 19, 2026 — Gen-4 verification and evaluation spec

**TL;DR:** Rewrote the gen-4 draft-agent writeup against a full set of
yardstick data and caught two of Claude's readings going the wrong way along
the way, then spent the rest of the day cutting the game-played evaluation
spec down through many rounds until only the essentials survived.

The day opened with a housekeeping squash-merge of a finished branch to
master, then moved into rewriting the gen-4 online-GRPO writeup now that all
four trained checkpoints had yardstick runs. I pointed out early on that the
win-rate data Claude had marked as gone actually still existed, just moved to
a NAS path. Pulling it back in reversed one of the writeup's headline claims:
gen-4 had looked like it eroded on card-power alignment relative to gen-3,
but on the real win-rate scale it had actually improved on its parent, and
the metric that best separated the four candidates from each other turned
out to be `cast_lift` rather than raw power.

A few other findings held up under scrutiny during the rewrite. The `T = 3`
exploration setting, an open question left over from gen-3, turned out to be
clearly the worst of the four settings tried, settling that question. Gen-1
and `forge-full` turned out to be statistically the same drafter once
checked side by side, which let me collapse them into one reference for
later comparisons. And a "margin decomposition" heuristic that earlier
writeups had used to read a training run's health turned out to rank the
four candidates in exactly the reverse of the yardstick's own ordering —
completely backwards, not just noisy.

I asked to have real games played to check that `deck_score` actually
predicts winning, and it did, but the calibration numbers needed more than
one pass. The first draft compared a through-the-origin fit here against a
with-intercept fit from the sealed pipeline and called them consistent by
coincidence; once caught, the honest range came out to 7-13 points of match
win rate per unit of score, bracketing the sealed pipeline's number either
way. I also pushed for the Bo7 match win rate, not the per-game rate, as the
headline figure, since a match doesn't need any modeling assumption about
what happens inside it and per-game rates disagreed with each other
depending on how they were computed. One side finding from that work: the
project's own deck-building method beats Forge's native sealed builder by
enough that the choice of builder is worth about as much win rate as an
entire generation of drafting improvement.

Editing the writeup surfaced a recurring problem I called out more than
once — sections that spent paragraphs rebutting objections nobody had
raised, especially right after Claude had just been corrected on something
and seemed to overcompensate on the next paragraph. That pattern showed up
at least three times in one document, so I had it write an explicit rule
against it into the shared style guide rather than just fixing it locally
each time.

The second half of the day went into the game-played evaluation spec, and
it was almost entirely a trimming exercise. I kept cutting scope that had
crept in: Swiss and bracket scheduling and sideboarding were declared fully
out of scope rather than deferred, the eligibility-checking rules got
dropped once Claude verified the real corpus never produces the pathological
cases they guarded against, bucketed sampling gave way to a plain random pod
draw with a retry on mirrors, and a failed match now just gets skipped
instead of triggering an abandon-after-N-attempts mechanism. Each
simplification got checked against the actual corpus numbers rather than
just asserted.

I also asked for a new feature mid-spec: letting a controllable fraction of
`forge-full` seats have their decks rebuilt by Forge's own deck builder
instead of the project's, so the two builders could be compared directly. I
pushed back on Claude's first design, which routed decisions through a
per-pairing "shard line" written by Python for Java to consume — I wanted
Java reading `drafts.jsonl` directly to minimize the interaction surface.
Claude found a real blocker (no JSON parser on the worker classpath) but the
design that came out of the back-and-forth was a flat seat table that
Python writes once and the Java workers sample from autonomously, which
ended up simpler than either original proposal.

Running the speckit pipeline on the finished spec turned up something I
hadn't expected: most of the Java-side work the plan called for already
existed. The Forge game player already returned everything a match-outcome
row needs — it was one worker class quietly throwing it all away into a
two-number result.
