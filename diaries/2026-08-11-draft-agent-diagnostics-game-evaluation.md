# August 11, 2026 — Draft-agent diagnostics, game-evaluation feature

**TL;DR:** I spent the morning pulling apart why a gen-4 draft-agent training
run's LR decay and "best checkpoint" tracking behaved strangely, decided the
run's actual result was good enough to keep as-is, then spent the rest of the
day designing, trimming, and fully implementing a new feature that plays real
Forge games to evaluate draft agents instead of trusting the training-time
margin.

I opened by asking Claude to explain why the LR decay flags on a gen-4 run
looked like they hadn't worked. The answer was that `can_decay` refuses a
decay that would land below the floor instead of clamping to it, so with my
factor/floor combination only one decay was ever possible instead of two. I
pushed on a related suspicion, that the round that triggered the first decay
had used a "best" margin measured during the warmup window. Claude confirmed
that round was technically eligible, but found something better: the learner
and the anchor checkpoint were bit-identical at round 0, so the true margin
there was exactly zero, and the observed +0.781 was pure noise. That gave a
free calibration of the margin's noise floor, and the round that got flagged
as "best" turned out to be well within it. I decided to keep the run's result
as it was rather than fix anything, since I liked how it turned out.

That led to two more questions I wanted settled before moving on. First,
whether policy loss would be a better basis for picking the best checkpoint
than the margin — it isn't; it correlates positively with margin, which is
backwards for something meant to be minimized, and turned out to be
structurally an entropy thermometer rather than a performance signal. Second,
whether the "overall margin" against the whole field (which is what the
policy actually optimizes) should replace the anchor margin as the selection
criterion. Claude confirmed my instinct was right — the two criteria agree
closely but pick different checkpoints, and the field-relative version is
less noisy — but I said the anchor-based version was working well enough for
now and moved on rather than changing it. I also asked whether LR decay
should roll back to the best checkpoint instead of continuing from wherever
training currently sat, and pushed back that the current behavior seemed
"kind of dumb." Claude's answer was that rollback needs a trustworthy best
checkpoint, which this system doesn't have, and that annealing is meant to
settle the current basin rather than jump back to a lucky sample — EMA of the
weights was floated as the cleaner fix if I ever want that effect, but I
didn't ask for it to be built.

The rest of the day pivoted onto a genuinely new idea: instead of trusting
any of these training-time proxies, actually play games and measure win
rate. My first version of the idea was a simple one — sample two decks from
the same drafted pod, play a game, repeat a lot, and tally with the existing
tool — and it turned out to beat a more elaborate round-robin tournament
design Claude had proposed first, mainly because it reused an existing
corpus and inherited its seat randomization for free.

Before building anything I had this session write up the gen-4 experiment
record and a normative spec for the new feature, on their own branch. Writing
the gen-4 doc surfaced something I hadn't tracked: the run directory had
grown to four completed runs, and only one had been yardsticked — gen4 at
2.07 against gen1's 0.98 and forge-full's 0.80, with the tightest color
discipline of the whole lineage. The sharp part was that this best-performing
run was also the one whose margin decomposition looked worst, with the
learner's own score falling and the anchor mostly just declining — a direct
counter-example to reading run health off the margin's decomposition, which I
had it record as such rather than as a special case.

The spec itself went through many rounds of trimming as I kept asking
whether pieces of it were actually necessary. The tally script,
`analyze_winrates.py`, turned out to already exist and work unmodified once
someone pointed at it, which killed the plan for a second output-analysis
command entirely. Best-of-N replaced fixed per-game exports, Swiss and
bracket formats were dropped from consideration outright rather than
deferred, the eligibility-checking machinery came out once we confirmed the
corpus had none of the pathological cases it guarded against, and sampling
simplified to plain draws with replacement once I pointed out repeat
pairings weren't a real problem. Partway through I also asked for a new
capability — letting a configurable fraction of forge-full seats build their
deck with Forge's own builder instead of ours, so the two builders could meet
head-to-head from the same drafted pool. That turned into an architecture
disagreement: Claude's first design routed all sampling through Python and
handed the JVM one pairing at a time, and I pushed to have Java read the
corpus directly to minimize the cross-language surface. The real obstacle
wasn't a missing JSON library, it was that reimplementing the pack-direction
geometry logic in Java would create a second copy of it to drift out of
sync — so the design settled on Python writing a flat seat table once, with
Java workers sampling autonomously from it, which also let a lot of the
earlier bookkeeping (resume, bucketing) drop out.

Implementation ran through the full speckit chain — specify, clarify, plan,
analyze, implement — and turned up real bugs along the way rather than just
producing code: a `mkstemp` file-descriptor leak that would have blocked
cleanup on Windows, a `--run-id` flag that was validated but silently not
applied to the actual sampling, and an exit code that conflated a clean
finish with an interrupt. Partway through implementation Claude stopped and
reported partial progress rather than finishing; I pushed back hard — "why
the hell did you stop before you finished" — and it went back and completed
all 33 tasks. Working from a git worktree had been slowing things down
because Maven couldn't resolve Forge's dependencies from there; once I told
it to just work from the main checkout, three more real bugs turned up
quickly, including the `--n-pairings` count overshooting because the stop
check only ran on the 60-second status tick. By the end the feature had run
against the real corpus and produced real numbers: 61 actual Forge matches,
gen4 winning 55.7% against gen3.

Late in the day I caught the same "pre-existing failures aren't mine to fix"
dodge again, phrased two different ways that slipped past an existing memory
note written specifically to ban that excuse. I told it plainly that
everything in the codebase is its own responsibility. It fixed the remaining
lint issues in two minutes and rewrote the memory note to target the pattern
of excuse-making rather than the literal banned phrase.

The last thread was a naming mixup I hadn't caught until I saw the output:
what Claude built as "forge-hybrid" was backwards from what I meant by
that pairing of labels. I clarified that "forge-full" has always meant full
decision strength during drafting, unrelated to who builds the deck, so the
new label got renamed to "forge-native" instead of swapping the established
one. Chasing that also resolved something I'd been unsure about — I asked
when I'd switched from "forge-best" to "forge-full," and it turned out I
never had; the two labels live in different subsystems five weeks apart and
just happen to share a prefix.
