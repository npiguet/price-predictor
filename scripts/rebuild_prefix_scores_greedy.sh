#!/usr/bin/env bash
# Rebuild every prefix-deck-value corpus with the greedy builder the corpora used.
#
# The first pass used GenerateDraftDataConfig's default build_method ("picker"),
# which is not what the online trainer used ("greedy"), so its levels did not
# match the corpora's own deck_scores -- they agreed on 11.8 % of seats. The
# picker files are kept under picker-built/ for the picker-vs-greedy comparison;
# these are the ones the write-up should cite.
#
# The builder is GreedyDeckBuilder on its defaults, which is what
# generate_draft_data.py:486 constructs and therefore what every corpus used:
# temperature 0 (cold hill-climbing, not annealing), one random restart, 200 max
# iterations. Do not pass knobs here -- matching the corpora is the whole point.
#
# Resumable: analyze_prefix_deck_value.py skips (draft, seat, prefix) triples
# already present in its output and refuses to extend a file built by the other
# builder, so re-running after a kill picks up where it stopped. Roughly 43 h
# single-process; the yardstick corpora come first because the charts use them.
#
# Watch it with:  tail -f models/draft/agent/gen4/rebuild-greedy.log
set -u
cd "$(dirname "$0")/.."
G=models/draft/agent/gen4
LOG=${LOG:-$G/rebuild-greedy.log}

runs="lr1e-5_t2all_decay0.3 lr1e-5_t2all_nodecay lr1e-5_t3all_decay0.3 lr1e-5_t3learner_t2field_decay0.3"
corpora=""
for suffix in yardstick-v-forge yardstick-v-gen3; do
  for run in $runs; do
    corpora="$corpora $run-$suffix"
  done
done

total=$(echo $corpora | wc -w)
started=$(date +%s)
i=0
{
  echo "=== rebuild started $(date -Is): $total corpora, greedy builder ==="
  for name in $corpora; do
    i=$((i + 1))
    drafts=$G/$name-drafts.jsonl
    raw=$G/$name-greedy-prefix-deck-scores.jsonl
    if [ ! -f "$drafts" ]; then
      echo "[$i/$total] MISSING $drafts, skipping"
      continue
    fi
    elapsed=$(( $(date +%s) - started ))
    echo "[$i/$total] $name  (started $(date -Is), ${elapsed}s into the run)"
    if ! python scripts/analyze_prefix_deck_value.py \
        --build-method greedy --drafts "$drafts" --raw "$raw" --title "$name"; then
      echo "[$i/$total] FAILED on $name -- rerun this script to resume"
      exit 1
    fi
  done
  echo "=== rebuild finished $(date -Is) after $(( ($(date +%s) - started) / 60 )) min ==="
} >>"$LOG" 2>&1
