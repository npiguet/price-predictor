# March 15, 2026 — Model harmonization and printing data spec

**TL;DR:** A long session that started by fixing a metric and ended by
fully speccing feature 009 (printing metadata in training data). Along the
way, feature 008 harmonized the CLI, evaluate output, and REST API across
both ML models into a consistent interface, and was fully implemented.

The day opened at midnight, when I was looking at the 007 evaluation metric.
The spec said to measure median percentage error on the predicted price, but
when I thought about it, a percentage error in the raw EUR domain is a bad
yardstick for a log-price model — a handful of expensive cards skew it.
After asking Claude to walk through the numbers, I realized the right unit
was the mean absolute error in the shifted-log space, since that is what
the model actually optimizes and it has a clean geometric interpretation.
We updated the spec to use shifted-log MAE with a note explaining how a
0.25 unit error translates back to roughly a 28% multiplicative error on
the original price. That landed in two commits just after midnight.

Then I came back in the morning with a cleaner session to tackle feature
008. I had been planning to define a new feature for the `predict` CLI
accepting converted card text, but I quickly decided I wanted to go further
and harmonize everything: training inputs, predict inputs, evaluate outputs,
and the REST API endpoint, across both sklearn and transformer. I used
speckit to create, clarify, and plan the feature. One clarification round
established why a parser is still needed even though the text is fed as-is
to the transformer — sklearn requires structured fields for feature
engineering, so the parser bridges the two models without exposing different
input contracts to the user. I noted that down in the spec for future
reference.

Two other decisions got locked in during speckit.clarify: the REST endpoint
was renamed from `/api/v1/evaluate` to `/api/v1/predict` for consistency
with the CLI, and the artifact versioning scheme was unified to follow the
sklearn convention for both model types. Implementation ran across most of
the afternoon and touched 38 files — train, predict, evaluate, the server,
and extensive fixture and test updates. By the end, 256 unit tests were
green and dead CLI handler code (about 500 lines) was removed in a cleanup
pass.

After a clear, I discovered the transformer evaluate command was parsing
card text through the sklearn feature pipeline rather than reading the
converted text files directly. I spotted this by comparing the two models'
outputs and noticing the transformer was performing worse than expected. The
fix was straightforward once the root cause was identified. Separately, I
changed the `max_seq_len` selection algorithm from the p95 heuristic to the
true maximum of the corpus, eliminating truncation entirely — the max was
only 296 tokens, well within reason. I also standardized the evaluate output
format across both models so they return the same set of fields plus a
`model_name` discriminator, and removed the automatic evaluate run that
happened at the end of transformer training.

The last major thread of the evening was feature 009: printing metadata in
training data. I came to it after thinking about why the transformer was
underperforming and, rather than tweaking architecture, decided the bigger
leverage was in the input data. I raised the reserve list as an obvious
factor (any MTG player knows it drives prices) and Claude confirmed it was
not in the training data at all. After working through which MTGJSON fields
were actually relevant — restricting to the cheapest printing only — I
settled on: reserve list flag, rarity, number of total printings, set code
of the cheapest printing, and the list of constructed formats the card is
legal in. Online-only formats were excluded as irrelevant; unknown cards
(spoilers, custom cards) would get `UKN` set code and be assumed legal in
all constructed formats by default. A key decision was that these fields
belong in the card text body itself, not as separate API parameters, making
them proper training data rather than side-channel metadata. By 22:36 the
spec, plan, 37-task list, and analysis fixes were all committed and
`/speckit.implement` was running.

*Note: reconstructed from prompt history + git log; full session transcripts were auto-deleted after 30 days.*
