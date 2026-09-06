# August 28, 2026 — Scorer preferences deep dive

**TL;DR:** I spent the day getting the scorer model's card and deck preferences
measured and written up, then spent just as long forcing Claude (and myself) to
defend every number, chart, and definition in the resulting document. Several
of my own working theories got corrected along the way — color affinity isn't
where I assumed it lived, and my five-color-bombs guess was only half right.

I opened by asking Claude to reverse-engineer what the scorer model actually
rewards in a deck, aiming eventually to write an article about it. I set it up
as three phases: brainstorm hypotheses, test them, write the results up in
experiments/. Hypothesis critique went to parallel Opus agents, and the probe
coding also went to Opus agents, run serially against the single GPU. It came
back with fifty hypotheses tested across eight probe suites, roughly half a
million scorer forward passes, no training involved.

The headline mechanism finding was that the scorer is a mean-pooler — the
"attention" in its pooling layer is uniform to five decimal places, so it just
averages a two-to-four-number summary of each card's text embedding across the
deck. I pushed on whether that was a genuine finding or a training failure.
Claude laid out both sides: gen-2's multi-view pooling already offered the
model max/min pooling for free and it never used it, and the current model
sits right at the accuracy ceiling this best-of-seven data allows, so
attention had nothing left to buy. I liked the reframe that came out of that
discussion — the pooling isn't idle, since its uniform mixing is what
broadcasts deck context and makes the off-color penalty and splash threshold
work; it's specifically *selective* attention that never differentiated,
because a uniform broadcast was all the task rewarded.

I then asked what actually made gen-4 beat gen-2 if the pooling never changed.
The answer was the inputs, not the pooling: gen-2 averaged vectors from the
price-predictor's euro-price encoder, which carried no card-quality signal, so
it could only express deck shape and lost to forge-best by 8-17 winrate points
even at matched shape. Swapping in an encoder trained on per-card winnability
from self-play games is what actually moved match play.

The preferences the study confirmed: creature-dense decks (19-20 of 23
spells), two or three colors, a curve averaging about 3.2, a flying premium
that grows with mana value, a mythic-only rarity bump, and card-draw or
do-nothing artifacts sitting at the bottom of every ranking. Removal loses to
BREAD's usual pecking order, scoring like a slightly-below-average creature.
There was no measurable pair-synergy: a 57-pair dose-response probe came back
essentially null, and what looked like one exception (Gray Merchant of
Asphodel, a devotion payoff) turned out to be generic deck-quality drift once
checked against a mismatched control — only Wingsteed Rider survived as a
genuinely synergy-sensitive card.

I kept asking where specific numbers came from, and a few didn't hold up. The
"winnability / castability / color affinity" names given to the text
embedding's leading axes turned out to be imported from an older encoder's
PCA, not measured on this one — running the actual regression confirmed the
winnability and castability axes but showed color affinity isn't a
text-embedding axis at all; it lives in the deterministic color-pip features
instead. A follow-up probe I asked for, testing deterministic feature groups
one at a time instead of as one block, backed that up: color pips are the one
group the scorer clearly needs, and everything else in that block is
redundant with the text.

I also asked whether the scorer weights what it reads the same way the
encoder emphasizes it, since the embedding's leading axis is played_rate, not
the more decisive-sounding winnability. It doesn't: a joint regression of the
scorer's revealed card values on the label axes showed it leans on winnability
about three times as hard as played-rate, confirmed by a causal perturbation
probe — what the embedding foregrounds and what the scorer actually pulls on
are different things.

On the domain side, I worked through the "one more color" decision with
Claude: a splash gets added only when the summed marginal value of the
off-color cards clears a fixed per-color fee, discounted by fixing. I guessed
that meant five-color decks show up when a pool has a bomb in every color
surrounded by dregs elsewhere. The data confirmed the direction — off-color
cards score higher than the main-color cards next to them, and the gap grows
with color count — but corrected the story on two points: three-color decks
carry deep splashes of several good cards rather than lone bombs, and the
main colors in four-color decks are actually stronger than in two-color
decks, so color creep is abundance-driven, not poverty-driven. Pulling the
actual five-color builds out (28 of 10,000, 0.28%) showed they're basically
extinct and mostly an artifact of multicolor-themed sets and a hybrid-mana
double-counting bug, not the model choosing to pay the color fee four times
over for bombs.

I also caught and cut a claim I didn't trust: the document's read that the
scorer prefers 22 spells and 18 lands over 23 and 17. I pointed out it was
never validated in actual played games — the only 18-land decks in the corpus
are Forge's own, and they lose for reasons that have nothing to do with land
count — so that whole subsection came out.

The rest of the day was a long editorial back-and-forth on the writeup itself.
I flagged style violations more than once — idioms and market metaphors that
crept into fresh paragraphs after a chat discussion, invented jargon like
"cargo" and "durdle" that needed plain-language replacements — pushed twice
for a clearer distinction between "held-out accuracy" and "ranking agreement"
after finding the definition buried in a noisy reference paragraph, and
finally asked for a full restructure so the article-relevant meta-game
findings (shape, cards, synergy) lead the document and the statistical
mechanism sections move to the back. I ended by cutting the fifty-row
hypothesis-verdict table and the probe inventory entirely, since neither told
the reader anything the sections themselves didn't already say better.
