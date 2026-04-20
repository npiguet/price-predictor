# March 20, 2026 — Exploring transformer price-prediction mitigations

**TL;DR:** I shared transformer training results with full
hyperparameters and raised two mitigation ideas for the persistent
prediction-quality problem. No code was committed; the session was
exploratory.

I pasted the output and training log from a full transformer run at
`--d-model 256 --n-layers 6 --ff-dim 1024`. With that baseline in hand,
I asked Claude to add `--epochs` and `--patience` parameters to the
architecture search script so I could re-run at 100 epochs.

The rest of the session was a short discussion of mitigation strategies.
I floated the idea of training two outputs — a raw price and a correction
term, summed at inference — and asked whether that could help. I also
made clear that I wasn't under any illusion about finding the root cause:
previous experiments had ruled out the obvious culprits, and I was
explicitly looking for viable workarounds rather than a cure. That led me
to ask what the inputs to an sklearn ensemble model would look like in
that context.

No commits landed on this date.

*Note: reconstructed from prompt history + git log; full session
transcripts were auto-deleted after 30 days.*
