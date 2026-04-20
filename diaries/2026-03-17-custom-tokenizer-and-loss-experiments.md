# March 17, 2026 — Custom Tokenizer, Then Loss Deep-Dive

**TL;DR:** I shipped the custom MTG word-level tokenizer (feature 010),
replacing BERT entirely, then spent the rest of the day chasing better
accuracy for expensive cards through a long series of loss-function and
architecture experiments that produced mixed results.

The day started around midnight with planning artifacts for feature 010:
research notes, a data model, CLI contracts, and a full implementation
plan, all committed before I went to sleep. In the morning I ran
`/speckit.tasks` to generate the 32-task breakdown, then `/speckit.analyze`
to verify cross-artifact consistency, and committed both. By 10 AM I was
into `/speckit.implement` and the tokenizer landed in one large commit:
a new `domain/tokenizer.py` (MtgTokenizer), `build_vocabulary.py`, and
`tokenizer_store.py`, with BertTokenizer removed from all five transformer
pipeline touch-points. Vocab size came in at 5,064 tokens with 98.4%
coverage — within spec.

The new tokenizer immediately underperformed the old BERT baseline on
validation metrics, which forced a quick diagnosis loop. The first problem
was that printing-data enrichment tokens (rarity values, format names)
were always mapping to UNK because the vocab was built from raw card texts
while training inputs include appended metadata fields. I fixed this by
seeding 20 hardcoded printing-data terms plus set-code fragments extracted
from AllPrintings.json — the vocabulary builder now needs a `--printings-
path` to do this properly. The second problem was that card names (proper
nouns like "Jace, the Mind Sculptor") were inflating the frequency counts
with ~3,200 tokens that could never appear at inference time, since the
tokenizer replaces them with a `cardname` placeholder. Excluding `name:`
lines from both frequency counting and tokenization collapsed the vocab
from 5,668 to 2,451 while pushing coverage to 99.5%.

Even after those fixes, accuracy was still not quite back to the BERT
baseline, which led me to look at the accuracy breakdown per price bucket.
Training on `log(price + offset)` concentrates gradient signal on the
mass of cheap cards, and the model was noticeably weak on cards above 2€.
I asked to make `log_offset` a configurable training parameter stored
inside the saved model file, so that runs with different offsets could be
compared without code changes. This became a standalone commit at 13:03.

From there the afternoon became a sustained exploration of ways to give
expensive cards more training signal. I first tried bucket-based loss
weights, where the weight for each sample is chosen so that the total
gradient contribution per price bucket is equalized. The weights turned
out to be numerically extreme (the rarest bucket received roughly 20×
weight), and I noticed that this should require more epochs to converge
for the same reason that reducing a learning rate does — we asked Claude
to explain this and the conversation confirmed the intuition. Results were
inconclusive at cap values of 5 and 20.

Around 15:30 I asked whether there was a fundamentally different output
representation for prices, and the conversation moved to ordinal/probit
regression: instead of a single log-price neuron, have one neuron per
price threshold that outputs a probability that the card exceeds that
threshold. I pushed on a variant where each neuron outputs a soft label
derived from the normal CDF centered on its threshold (so a 10€ card puts
most of its label mass on the 4.5€ and 13.5€ threshold neurons), which is
essentially the same as standard proportional-odds regression but with a
Gaussian kernel. Thresholds were set at `0.5 * 3^n` (0.5, 1.5, 4.5, ...
up to 1093.5€), all of these parameters made CLI flags and stored in the
model artifact.

There was also a domain detour around 15:48: I pointed out that an `abu`
flag (true if a card was only ever printed in Alpha, Beta, or Unlimited)
would be a very strong price signal, alongside filtering out non-legal
printings (Collector's Edition, online promos, etc.) when computing prices.
A commit at 16:03 captured this work after I confirmed that Claude had
found the ABU data and handled the filtering correctly.

The evening session tried CLS-token pooling as a replacement for mean
pooling, and a distance-weighted loss variant where neurons closer to the
true price bucket receive a larger gradient weight. Neither gave a clear
win. By 23:00 I stepped back and decided to simplify: roll back to a plain
`transformer → mean pooling → output` architecture and instead vary the
oversampling exponent (`WeightedRandomSampler`) to see if skewing the
training distribution toward expensive cards would help. A quick comparison
at 23:22 showed that sampling alpha 0 (no oversampling) still beat alpha
0.5, suggesting the model needs the full distribution of cheap cards to
learn the baseline signal correctly. The session ended without a committed
resolution to the loss-function question — the experiments were running
and the results were still coming in.

*Note: reconstructed from prompt history + git log; full session
transcripts were auto-deleted after 30 days.*
