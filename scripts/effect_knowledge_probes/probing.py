"""One probing run over a checkpoint: every selected probe, as a scorecard.

Nothing here does work at import time.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from effect_knowledge_probes import common, halves, labels, record_walk
from effect_knowledge_probes.line_labelling import (
    ObservedOutcomes,
    ProbeGames,
    RecordObservers,
    interaction_pairs,
    label_lines,
)
from effect_knowledge_probes.line_ladders import LineSources, line_ladder
from effect_knowledge_probes.pair_ladders import pair_ladder
from effect_knowledge_probes.probe_stages import (
    HALF_CHOICES,
    AblationInputs,
    board_ladder,
    board_subsets,
    read_halves,
    run_ablations,
    run_sweeps,
    selected_families,
    selected_targets,
    validation_samples,
)
from effect_knowledge_probes.progress import progress
from effect_knowledge_probes.scorecard import Scorecard


class _HalfScoring:
    """Scores the rule tests, the grids and the gen-3 probes of each half read."""

    def __init__(self, args, context: common.ProbeContext, selection, probe_model) -> None:
        self.args, self.selection, self.probe_model = args, selection, probe_model
        self.tests = {}
        for half in HALF_CHOICES[args.half]:
            progress(f"rule tests, {half} half")
            self.tests[half] = halves.score_rule_tests(context, selection, probe_model, half)

    def score(self, read: halves.HalfRead, body: dict) -> None:
        body["games"] = read.games
        progress(f"{read.name}: {len(read.grids)} grids")
        body["grids"] = halves.score_grids(self.probe_model, self.selection, read)
        body["rule_tests"] = self.tests[read.half].get(read.stratum, halves.NO_RULE_TESTS)
        if not self.args.no_gen3_probes:
            body.update(halves.score_gen3_probes(self.probe_model, read))


class _Probing:
    """The stages of one run, in order, filling one scorecard."""

    def __init__(self, args, context: common.ProbeContext, selection) -> None:
        self.args, self.context, self.selection = args, context, selection
        self.families = selected_families(args)
        self.targets = selected_targets(args)
        sidecars = context.sidecars()
        acting_text = common.acting_text_function(
            sidecars, common.manifest_surface(context.manifest))
        self.games = ProbeGames()
        self.outcomes = ObservedOutcomes(acting_text)

    def _load_model(self, reads: list[halves.HalfRead], subsets: dict):
        every = [r for read in reads for r in read.records.values()]
        every += [r for by_subset in subsets.values() for rows in by_subset.values()
                  for r in rows]
        progress(f"loading {self.args.checkpoint}")
        return self.context.model(every, vocab_path=self.args.vocab_path,
                                  keyword_path=self.args.keyword_definitions)

    def _new_scorecard(self, probe_model) -> Scorecard:
        return Scorecard(
            checkpoint=str(self.args.checkpoint), selection_set_digest=self.selection.digest,
            corpus_digest=self.selection.corpus_digest,
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            e_dim=probe_model.e_dim, surface=probe_model.surface,
            join_rate=None,
            families={family: {"targets": {}} for family in sorted(self.families)},
        )

    def _score_halves(self, scorecard: Scorecard, probe_model,
                      reads: list[halves.HalfRead]) -> None:
        scoring = _HalfScoring(self.args, self.context, self.selection, probe_model)
        for read in reads:
            scoring.score(read, scorecard.half(read.stratum, read.half))

    def _observe_training(self) -> None:
        """The first ``--profile-shards`` training shards, for the observed-outcome labels."""
        for record in record_walk.records([self.context.training_dir],
                                          shards=self.args.profile_shards):
            self.outcomes.observe(record)

    def _line_level(self, scorecard: Scorecard, probe_model, act_hidden: dict) -> list:
        """The line ladders and the pair ladder; returns the ability cache's line items."""
        progress("line-level probes over the ability cache")
        self._observe_training()
        sidecars = self.context.sidecars()
        items = common.line_items(self.context.cards_folders, probe_model.surface,
                                  self.args.abilities_root)
        cache = (items, common.acting_text_function(sidecars, probe_model.surface))
        held_out_cards = set(self.context.checkpoint.provenance.held_out_cards)
        line_labels = label_lines(items, self.outcomes, sidecars)
        sources = LineSources(cache, act_hidden, held_out_cards)
        for name, entry in line_ladder(line_labels, probe_model, self.targets,
                                       sources).items():
            scorecard.put(labels.TARGETS_BY_NAME[name].family, name, entry)
        pairs, joined = interaction_pairs(self.games, items)
        scorecard.join_rate = joined.join_rate
        if "interactions" in self.families:
            scorecard.put(labels.TARGETS_BY_NAME["fires_pair"].family, "fires_pair",
                          pair_ladder(pairs, probe_model, cache))
        return items

    def run(self) -> Scorecard:
        import torch

        args, context = self.args, self.context
        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        reads = read_halves(args, context, self.selection,
                            RecordObservers(self.games, self.outcomes))
        subsets = board_subsets(args, context)
        probe_model = self._load_model(reads, subsets)
        scorecard = self._new_scorecard(probe_model)
        act_hidden = board_ladder(args, scorecard, probe_model, reads)
        self._score_halves(scorecard, probe_model, reads)
        items = self._line_level(scorecard, probe_model, act_hidden)
        if not args.no_sweeps:
            progress("board sweeps")
            scorecard.sweeps = run_sweeps(reads, probe_model)
        run_ablations(args, scorecard, probe_model, AblationInputs(
            validation_samples(context, args.ablation_records), items, subsets,
            context.corpus))
        scorecard.runtime = {
            "wall_seconds": time.time() - started,
            "peak_gpu_bytes": common.peak_gpu_bytes(),
        }
        return scorecard


def probe(args, context: common.ProbeContext, selection) -> Scorecard:
    """Run every selected probe over the selection set's halves, the line
    ladders, the sweeps and the ablations; return the scorecard."""
    return _Probing(args, context, selection).run()
