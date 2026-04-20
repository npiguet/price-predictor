# March 18, 2026 — Pooling experiments after ordinal rollback

**TL;DR:** I rolled back an earlier ordinal regression experiment to a clean
baseline and spent the morning documenting what had been learned. The evening
became a live ablation study of pooling strategies — max, mean, and their
concatenation — ending with a small but meaningful improvement on the high-price
bucket.

The day started with a cleanup decision. I had been experimenting with ordinal
regression approaches — soft ordinal loss, proportional odds, distance-weighted
BCE, bucket-proportional sampling — and none of them had improved things. I
asked Claude to roll back to commit bd9a7b13 while preserving two specific
changes I considered keepers: the ABU flag and the no-mana-cost distinction. I
also wanted per-bucket evaluation metrics to survive the rollback, since that
diagnostic lens had proven useful. The first commit of the day reflects that
reset: the per-bucket breakdown was re-applied on top of the clean baseline, and
everything that had been tried was written up in an experiment document so the
reasoning and outcomes were not lost.

The morning pause gave way to two exploratory conversations. I asked Claude to
explain quantile loss in plain terms and whether it could help with the skew in
our price distribution. I also floated the idea of training four separate
subnets — one per price bucket — and routing gradient only through the subnet
whose bucket matched the card's actual price, with oversampling to balance the
buckets. Neither idea made it into code that day; I was mostly taking stock
before committing to a direction.

The evening session became a hands-on experiment with pooling. The existing
model used mean pooling over the transformer's token outputs, which averages
every token's representation — including generic filler tokens that dilute the
signal from price-relevant words like mana cost or keywords. I wondered whether
max pooling, which preserves the strongest activation per dimension, might do
better. Claude implemented it with -inf masking on padding positions so pad
tokens could not influence the max. I ran both configurations and pasted the
results back. The numbers were close enough that I asked to try concatenating
both vectors into a single 2×d_model representation fed to the output head, on
the theory that the head might learn to draw from whichever signal was more
useful per dimension.

The concatenated result showed the best performance on the >€50 bucket observed
so far, with a signed log error of −2.621. I also experimented briefly with
adding an attention pooling vector as a third component, but reverted it when it
did not help. A suggestion came in from outside the session that lowering dropout
from the default to 0.05 or 0 might help given the small dataset size; I tried
both and shared the results. Additional training epochs did not shift things
further, which suggested the model was not simply undertrained.

The session closed with me asking Claude to write up the pooling method
comparison and dropout ablation in the experiments document alongside the earlier
ordinal regression notes, and then to commit the current state. The final commit
captured the concatenated pooling architecture, the new --dropout CLI flag, and
the expanded experiment documentation. The day ended with a plan forming for the
next step: moving set metadata and a release-year feature out of the card text
and into independent input neurons that bypass the transformer.

*Note: reconstructed from prompt history + git log; full session transcripts
were auto-deleted after 30 days.*
