"""Probe one checkpoint's game knowledge, and how much of it lives in ``e``.

    # once per curated corpus, before any training
    python scripts/effect_knowledge_probes/run.py --freeze-probe-set --corpus DIR
    # then per checkpoint trained on that corpus
    python scripts/effect_knowledge_probes/run.py --checkpoint PATH

``--freeze-probe-set`` freezes the corpus's selection set
(``effects.application.freeze_selection_set``) from the corpus alone, with no
checkpoint: per validation stratum and half, the rung-3 probe items
(``labels.record_items``), the damage, combat and evasion grids' records
(``selection``), the rule tests and the deviance sample, written to
``selection-set-<corpus digest>.json`` under ``--probe-set-dir`` (default
``--reports-dir``). It exits 1, naming stratum, half and component, when the
corpus cannot meet the precision targets.

A probing run refuses without the selection set frozen for its checkpoint's
corpus, reads it, and writes one scorecard recording its digest. Every
board-dependent probe reads the records of the set's halves (``halves``), each
record once, and is reported per stratum and half (FR-070, FR-074):

* the read-out ladder over the frozen probe items (``ladder``), and each probe
  type's rung-3 score on the control score's scale;
* the frozen grids — damage, combat with one and with several blockers,
  evasion with its protection rows — scored as the control score scores them;
* the frozen rule tests (``rule_tests``);
* the turn and step conditions, anthem and target-legality probes;
* the five board sweeps over the reporting half's records (``sweeps``).

Beside them, reported only and never on a stratum's scale (research E5): the
line and pair ladders, labelled from the scripts and from what the probe
games' and the first ``--profile-shards`` training shards' records were seen
to do, each record once though the training stratum repeats its copies; the
ablation of ``e`` (``ablation``); the board-abilities ablation (``board``);
and optionally every trunk layer (``--per-layer``) and a shallow trunk trained
on frozen ``e`` (``--method-c``).

A probing run reads gen-3 records only, and loads only a ``gen-3`` checkpoint.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import md_table  # noqa: E402

from effect_knowledge_probes import (  # noqa: E402
    ablation,
    common,
    halves,
    labels,
    ladder,
    record_walk,
    rule_tests,
    sweeps,
)
from effect_knowledge_probes import board as board_probe  # noqa: E402
from effect_knowledge_probes.scorecard import Scorecard  # noqa: E402


def progress(message: str) -> None:
    """One timestamped line, flushed: a probing run takes the better part of an
    hour, and without these nothing says whether it is alive or where it is."""
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


#: Training shards read for the observed-outcome line labels.
PROFILE_SHARDS = 4
#: Validation-sample records per stratum the ablation reads.
ABLATION_RECORDS = 1024
METHOD_C_STEPS = 2000
#: Items a ladder needs in one stratum and half.
MIN_LADDER_ITEMS = 20
HALF_CHOICES = {"both": halves.HALVES, "selection": (halves.SELECTION,),
                "reporting": (halves.REPORTING,)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        help="the checkpoint to probe; refused with --freeze-probe-set")
    parser.add_argument("--corpus", type=Path,
                        help="curated corpus; default the checkpoint's recorded one, "
                             "required with --freeze-probe-set")
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="sidecar tree (repeatable); default the trees the "
                             "checkpoint recorded")
    parser.add_argument("--abilities-root", type=Path, default=common.DEFAULT_ABILITIES)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--keyword-definitions", type=Path)
    parser.add_argument("--freeze-probe-set", action="store_true")
    parser.add_argument("--families", type=lambda s: [f.strip() for f in s.split(",") if f],
                        help=f"comma list from: {', '.join(labels.FAMILIES)}")
    parser.add_argument("--half", choices=tuple(HALF_CHOICES), default="both",
                        help="the selection set's halves read and reported")
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--method-c", action="store_true")
    parser.add_argument("--method-c-layers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--method-c-steps", type=int, default=METHOD_C_STEPS)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-board", action="store_true",
                        help="skip the board-abilities ablation (board.py)")
    parser.add_argument("--no-gen3-probes", action="store_true",
                        help="skip the turn and step conditions, anthem and "
                             "target-legality probes")
    parser.add_argument("--reports-dir", type=Path, default=common.REPORTS,
                        help="where the report directory goes, and the selection "
                             "set unless --probe-set-dir is given")
    parser.add_argument("--probe-set-dir", type=Path, default=None,
                        help="where the selection set is written and read "
                             "(default: --reports-dir); every arm of a sweep "
                             "shares one set while its reports go beside its run")
    parser.add_argument("--probe-records", type=int, default=halves.PROBE_RECORDS,
                        help="qualifying records each gen-3 probe reads per half")
    parser.add_argument("--probe-per-game", type=int, default=halves.PROBE_PER_GAME)
    parser.add_argument("--profile-shards", type=int, default=PROFILE_SHARDS)
    parser.add_argument("--ablation-records", type=int, default=ABLATION_RECORDS)
    parser.add_argument("--board-records", type=int, default=None,
                        help="records per board subset per stratum (default board.py's)")
    parser.add_argument("--board-shards", type=int, default=None,
                        help="validation shards read per stratum for the board subsets")
    args = parser.parse_args(argv)
    unknown = set(args.families or ()) - set(labels.FAMILIES)
    if unknown:
        parser.error(f"unknown families: {sorted(unknown)}")
    if args.freeze_probe_set:
        if args.checkpoint is not None:
            parser.error("--freeze-probe-set freezes from the corpus alone, before any "
                         "model exists: drop --checkpoint")
        if args.corpus is None:
            parser.error("--freeze-probe-set needs --corpus DIR")
    elif args.checkpoint is None:
        parser.error("--checkpoint is required (or --freeze-probe-set --corpus DIR)")
    return args


# ── what the line labels observe ────────────────────────────────────────


class ObservedOutcomes:
    """What each text was seen to do: its effect profiles and the mana it made.

    Observed one record at a time and kept as per-text lists of small arrays,
    so the probe games and the training shards feed one set of labels without
    either being held. A record is observed once, however many rows of it a
    stratum holds: the training stratum repeats each copy curation wrote, and
    a probe of training records drops the repeats (FR-070).
    """

    def __init__(self, acting_text) -> None:
        self._acting_text = acting_text
        self._seen: set[str] = set()
        self.profiles: dict[str, list[np.ndarray]] = defaultdict(list)
        self.mana: dict[str, list[float]] = defaultdict(list)

    def observe(self, record) -> None:
        from effects.domain.records import Moment, RecordKind

        if record.record_id in self._seen:
            return
        self._seen.add(record.record_id)
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return
        text = self._acting_text(record.ability[0]) if record.ability else None
        if not text:
            return
        profile = labels.effect_profile(record)
        if profile is not None:
            self.profiles[text].append(profile)
        produced = labels.mana_observed(record)
        if produced is not None:
            self.mana[text].append(produced)


@dataclass
class ProbeGames:
    """The trigger join and the triggers of every distinct probe-game record,
    for the interaction pairs."""

    join: labels.TriggerJoin = field(default_factory=labels.TriggerJoin)
    #: ``trigger record id -> (trigger key, fired)``.
    triggers: dict[str, tuple] = field(default_factory=dict)

    def observe(self, record) -> None:
        from effects.domain.records import TriggerPayload

        self.join.observe(record)
        if isinstance(record.payload, TriggerPayload) and record.ability:
            self.triggers[record.record_id] = (record.ability[0], record.payload.fired)


def _observe_training(args, context: common.ProbeContext, outcomes: ObservedOutcomes) -> None:
    """The first ``--profile-shards`` training shards, for the observed-outcome labels."""
    for record in record_walk.records([context.training_dir], shards=args.profile_shards):
        outcomes.observe(record)


@dataclass(frozen=True, slots=True)
class LineLabels:
    """One ability line's labels, keyed by the provenance of its first carrier.

    ``weights`` holds a per-label weight where a label is an estimate over few
    records (the state-dependence spread).
    """

    key: object
    labels: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)


def label_lines(line_items, outcomes: ObservedOutcomes, sidecars) -> list[LineLabels]:
    """Every encoded line's script labels, plus what its records were seen to do."""
    from effects.infrastructure.sidecar_io import converted_text_path

    spread = labels.state_dependence(outcomes.profiles)
    out = []
    for item in line_items:
        try:
            path = sidecars.path_for(item.script_file)
        except KeyError:
            path = None
        types = (common.card_facts(converted_text_path(path))["type_line"].lower()
                 if path else "")
        found = labels.line_labels(item.script_text, item.line_kind, item.api, types)
        weights = {}
        if item.text in spread:
            found["spread"], n = spread[item.text]
            weights["spread"] = labels.state_weight(n)
        if outcomes.mana.get(item.text):
            found["observed_mana"] = float(np.mean(outcomes.mana[item.text]))
        if found:
            out.append(LineLabels(item.key, found, weights))
    return out


def interaction_pairs(games: ProbeGames, line_items) -> tuple[list[dict], labels.JoinResult]:
    """Trigger lines with their causes: joined from the games, then mined from scripts.

    Mined pairs are positives by construction and far outnumber the joined
    ones, so they are capped at the joined count: uncapped, a probe could score
    by telling a mined pair from a joined one rather than reading the pair.
    """
    joined = games.join.result()
    pair_labels: dict[tuple, float] = {}
    for record_id, (trigger_key, fired) in games.triggers.items():
        cause = joined.cause.get(record_id)
        if cause is None:
            continue
        pair = (json.dumps(trigger_key.as_dict(), sort_keys=True),
                json.dumps(cause.as_dict(), sort_keys=True))
        pair_labels[pair] = max(pair_labels.get(pair, 0.0), float(fired))
    pairs = [{"a": json.loads(a), "b": json.loads(b), "label": label, "source": "join"}
             for (a, b), label in sorted(pair_labels.items())]
    mined = []
    for a, b in labels.mined_pairs(line_items):
        signature = (json.dumps(a.as_dict(), sort_keys=True),
                     json.dumps(b.as_dict(), sort_keys=True))
        if signature not in pair_labels:
            mined.append((common.stable_hash("|".join(signature)), a, b))
    for _, a, b in sorted(mined, key=lambda row: row[0])[: len(pairs)]:
        pairs.append({"a": a.as_dict(), "b": b.as_dict(), "label": 1.0,
                      "source": "mined"})
    return pairs, joined


# ── probing: the line and pair ladders ──────────────────────────────────


def _union_groups(pairs: list[tuple[str, str]]) -> list[str]:
    """Fold groups for text pairs: connected components, so no text straddles."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    return [find(a) for a, _ in pairs]


def line_split(carriers: set[str], held_out_cards: set[str]) -> str:
    """``held-out`` when every card printing the text is held out, else ``trained``.

    Every game naming a held-out card is kept out of training, so a text printed
    on one held-out card and one trained card was still trained on, through the
    trained card. Only a text no trained card prints was never seen.
    """
    return "held-out" if carriers and carriers <= held_out_cards else "trained"


def line_ladder(line_labels: list[LineLabels], probe_model, cache, targets, act_hidden,
                held_out_cards) -> dict:
    """The short ladder for every line target: ``e``, its width control, ``[ACT]``.

    ``cache`` is ``(items, text_by_key)`` from ``common.line_items`` over the
    checkpoint's cache: every line it encodes, read once per run.
    """
    items, text_by_key = cache
    by_text = {item.text: item for item in items}
    e_dim = probe_model.e_dim
    act_mean = {text: np.mean(rows, axis=0) for text, rows in act_hidden.items()}
    results: dict[str, dict] = {}
    for spec in targets:
        if spec.scope != "line":
            continue
        rows = []
        for entry in line_labels:
            label = entry.labels.get(spec.name)
            if label is None:
                continue
            text = text_by_key.get(entry.key)
            item = by_text.get(text)
            if item is None or item.e is None:
                continue
            weight = entry.weights.get(spec.name, 1.0)
            split = line_split(item.carriers, held_out_cards)
            rows.append((text, item.e, label, weight, split))
        strata = {}
        for split in ("held-out", "trained"):
            chosen = [r for r in rows if r[4] == split]
            if len(chosen) < MIN_LADDER_ITEMS:
                continue
            texts = [r[0] for r in chosen]
            data = ladder.LadderInput(
                kind=spec.kind, y=np.array([r[2] for r in chosen], dtype=np.float64),
                groups=np.array(texts, dtype=object),
                features={"1": np.stack([r[1] for r in chosen]),
                          "1w": np.stack([ladder.width_vector(t, e_dim) for t in texts])},
                weight=np.array([r[3] for r in chosen], dtype=np.float64),
            )
            result = ladder.run_ladder(data, rungs=("1", "1w"))
            acted = [i for i, t in enumerate(texts) if t in act_mean]
            if len(acted) >= MIN_LADDER_ITEMS:
                subset = ladder.LadderInput(
                    kind=spec.kind, y=data.y[acted], groups=data.groups[acted],
                    features={"1": data.features["1"][acted],
                              "2": np.stack([act_mean[texts[i]] for i in acted])},
                    weight=data.weight[acted],
                )
                sub = ladder.run_ladder(subset, rungs=("1", "2"))
                result["rungs"]["2"] = sub["rungs"]["2"]
                result["rungs"]["1-acting"] = sub["rungs"]["1"]
                result["n_acting"] = sub["n"]
            strata[split] = result
        results[spec.name] = {"scope": "line", "kind": spec.kind, "strata": strata}
    return results


def pair_ladder(pairs: list[dict], probe_model, cache) -> dict:
    """Interaction pairs: a trigger line with its cause, read from both ``e``."""
    from effects.domain.provenance import ProvenanceKey

    items, text_by_key = cache
    by_text = {item.text: item for item in items}
    rows = []
    for pair in pairs:
        a = by_text.get(text_by_key.get(ProvenanceKey.from_dict(pair["a"])))
        b = by_text.get(text_by_key.get(ProvenanceKey.from_dict(pair["b"])))
        if a is None or b is None or a.e is None or b.e is None:
            continue
        rows.append((a, b, pair["label"]))
    if len(rows) < MIN_LADDER_ITEMS:
        return {"scope": "pair", "kind": "binary", "strata": {}}
    groups = _union_groups([(a.text, b.text) for a, b, _ in rows])
    grouping = "component"
    if max(groups.count(g) for g in set(groups)) > len(rows) / 2:
        groups, grouping = [a.text for a, _, _ in rows], "trigger-text"
    e_dim = probe_model.e_dim
    data = ladder.LadderInput(
        kind="binary", y=np.array([r[2] for r in rows], dtype=np.float64),
        groups=np.array(groups, dtype=object),
        features={
            "1": np.stack([np.concatenate([a.e, b.e]) for a, b, _ in rows]),
            "1w": np.stack([np.concatenate([ladder.width_vector(a.text, e_dim),
                                            ladder.width_vector(b.text, e_dim)])
                            for a, b, _ in rows]),
        },
    )
    result = ladder.run_ladder(data, rungs=("1", "1w"))
    result["grouping"] = grouping
    return {"scope": "pair", "kind": "binary", "strata": {"all": result}}


# ── probing: method C ───────────────────────────────────────────────────


class _NoTrunk:
    """Method C's zero-layer trunk: the slots reach the heads as embedded."""

    def __new__(cls):
        from torch import nn

        class NoTrunk(nn.Module):
            def forward(self, src, src_key_padding_mask=None, **_):  # noqa: ARG002
                return src

        return NoTrunk()


def method_c(args, probe_model, corpus: Path, eval_records: list) -> dict:
    """Train a shallow trunk on frozen ``e`` and score it beside the full model.

    It trains on the training stratum as curated, copies included: method C
    is a training run, not a probe item.
    """
    import torch

    from effects.domain.effect_losses import per_entity_loss
    from effects.domain.effect_model import EffectModel
    from effects.domain.surface_collation import per_entity_targets

    layers = args.method_c_layers
    config = replace(probe_model.checkpoint.model_config, n_layers=max(layers, 1))
    shallow = EffectModel(config).to(probe_model.device)
    if layers == 0:
        shallow.trunk = _NoTrunk()
    optimizer = torch.optim.AdamW(shallow.parameters(), lr=3e-4, weight_decay=1e-4)
    batcher, encoder, fields = probe_model.batcher, probe_model.encoder, probe_model.fields

    def stream():
        while True:
            yield from record_walk.records([Path(corpus) / "training"], distinct=False)

    records, steps, started = stream(), 0, time.time()
    shallow.train()
    while steps < args.method_c_steps:
        chunk = [next(records) for _ in range(16)]
        with torch.no_grad():
            batch, surfaces, _encoded = batcher.build(chunk, encoder)
        targets = per_entity_targets(chunk, surfaces, fields)
        hidden = shallow(batch)
        device = hidden.device
        gathered = shallow.per_entity_at(hidden, targets.index.to(device))
        loss, _ = per_entity_loss(
            gathered, targets.fields_on(device), fields=fields,
            log_dispersion=shallow.per_entity_dispersion(),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1
    shallow.eval()
    return {
        "layers": layers, "steps": steps, "seconds": time.time() - started,
        "shallow": ablation.field_losses(shallow, probe_model, eval_records),
        "full": ablation.field_losses(probe_model.model, probe_model, eval_records),
    }


# ── probing: the stages of one run ──────────────────────────────────────


def _board_subsets(args, context: common.ProbeContext) -> dict[str, dict[str, list]]:
    """Each validation stratum's board-abilities subsets (``board.py``), unless skipped."""
    if args.no_board:
        return {}
    chosen: dict[str, dict[str, list]] = {}
    for stratum in common.STRATA:
        directory = context.stratum_dir(stratum)
        if not directory.exists():
            continue
        progress(f"board subsets: scanning {stratum}")
        chosen[stratum] = board_probe.collect(
            record_walk.records(
                [directory], shards=args.board_shards or board_probe.SHARDS_PER_STRATUM),
            context.sidecars(),
            per_subset=args.board_records or board_probe.RECORDS_PER_SUBSET)
        progress("  " + ", ".join(
            f"{name} {len(rows)}" for name, rows in chosen[stratum].items()))
    return chosen


def _probe_scores(inputs: dict) -> dict[str, dict]:
    """Each frozen probe type's rung-3 score in one half, on the control
    score's scale too: ``2·AUC − 1`` for a yes/no type, R² as is."""
    from effects.domain.probe_readouts import item_score
    from effects.domain.score_scales import rescaled_auc

    out = {}
    for name, data in sorted(inputs.items()):
        if data.rung3 is None:
            out[name] = {"kind": data.kind, "n": int(len(data.y)), "score": None,
                         "unavailable": data.rung3_note}
            continue
        value = float(item_score(data.kind, data.y, data.rung3))
        out[name] = {"kind": data.kind, "n": int(len(data.y)), "score": value,
                     "rescaled": rescaled_auc(value) if data.kind == labels.BINARY else value}
    return out


def _board_ladder(args, scorecard: Scorecard, probe_model, reads: list[halves.HalfRead],
                  kinds: dict[str, str]) -> dict[str, list]:
    """Every frozen probe type's ladder, per stratum and half; returns
    ``[ACT]`` by text.

    The trunk's ``[ACT]`` output is gathered from the same forward pass, by
    acting text, for the line ladder's rung 2.
    """
    act_hidden: dict[str, list] = {}
    for read in reads:
        chosen = read.probe_records()
        progress(f"{read.name}: extracting rung features over {len(chosen)} records")
        extracted = ladder.extract_features(
            probe_model, chosen, read.probe_items, per_layer=args.per_layer,
            act_hidden=act_hidden)
        inputs = ladder.ladder_inputs(extracted, kinds)
        scorecard.half(read.stratum, read.half)["probes"] = _probe_scores(inputs)
        for position, (name, data) in enumerate(sorted(inputs.items()), start=1):
            if len(data.y) < MIN_LADDER_ITEMS:
                continue
            progress(f"{read.name}: ladder {position}/{len(inputs)} {name} (n={len(data.y)})")
            spec = labels.TARGETS_BY_NAME[name]
            entry = scorecard.target(spec.family, name, scope=spec.scope, kind=spec.kind)
            entry["strata"][read.name] = ladder.run_ladder(data)
    return act_hidden


def _run_sweeps(reads: list[halves.HalfRead], probe_model, checkpoint) -> dict:
    """Every board sweep over the reporting half's records: per-record curves
    and their mean, both strata pooled."""
    out: dict = {}
    verdict_trained = ladder.verdict_head_trained(checkpoint)
    curves: dict[str, list] = defaultdict(list)
    names: set[str] = set()
    for read in reads:
        for name, rows in read.sweep_rows.items():
            names.add(name)
            if name == "affordability" and not verdict_trained:
                continue
            for row in rows:
                record = read.records.get(row["record_id"])
                if record is None:
                    continue
                label = f"damage_{row['host']}" if name == "damage" else name
                curves[label].append(sweeps.run_sweep(name, probe_model, record, row))
    if "affordability" in names and not verdict_trained:
        # It reads the verdict head, which this checkpoint never trained: a
        # curve from random weights says nothing.
        out["affordability"] = {"unavailable": ladder.UNTRAINED_VERDICT_NOTE}
    for label, curve in curves.items():
        out[label] = {"mean": sweeps.summarize(curve), "records": curve}
    return out


def _validation_samples(context: common.ProbeContext, limit: int) -> list:
    """The first ``limit`` distinct records of each stratum's fixed validation sample."""
    from effects.domain.shard_generation import ACCEPTED_GENERATIONS
    from effects.infrastructure.shard_io import read_shard, require_generation

    samples = []
    for stratum in common.STRATA:
        path = Path(context.corpus) / "validation" / "samples" / f"{stratum}.jsonl.gz"
        if path.exists():
            require_generation([path], *ACCEPTED_GENERATIONS)
            for count, record in enumerate(halves.distinct(read_shard(path))):
                if count >= limit:
                    break
                samples.append(record)
    return samples


def _run_ablations(args, context: common.ProbeContext, scorecard: Scorecard, probe_model,
                   line_cache, board_subsets) -> None:
    """Method B over the validation samples and the board subsets, and method C.

    Both ablations draw their replacements for ``e`` from the ability cache,
    so a checkpoint whose cache was never encoded runs neither.
    """
    samples = _validation_samples(context, args.ablation_records)
    if not line_cache and (board_subsets or not args.no_ablation):
        progress(f"no ability cache under {args.abilities_root}: both ablations skipped")
    elif not args.no_ablation or board_subsets:
        stats = ablation.cache_statistics(line_cache)
        if not args.no_ablation and samples:
            progress(f"ablation over {len(samples)} records")
            scorecard.ablation = ablation.run_ablation(probe_model, samples, stats)
        if board_subsets:
            progress("board-abilities ablation")
            scorecard.board_abilities = board_probe.run(probe_model, board_subsets, stats)
    if args.method_c and samples:
        progress("method C: a shallow trunk on frozen e")
        scorecard.method_c = method_c(args, probe_model, context.corpus, samples)


def _read_halves(args, context: common.ProbeContext, selection, board_targets: set[str],
                 games: ProbeGames, outcomes: ObservedOutcomes) -> list[halves.HalfRead]:
    """Each stratum's halves, read once each (``halves.read_half``)."""
    from effects.domain.damage_step_keywords import KeywordResolver

    sidecars = context.sidecars()
    surface = common.manifest_surface(context.manifest)
    acting_text = common.acting_text_function(sidecars, surface)
    keywords_of = KeywordResolver(sidecars).keywords_of

    def observe(record) -> None:
        games.observe(record)
        outcomes.observe(record)

    reads = []
    for stratum in sorted(selection.halves):
        for half in HALF_CHOICES[args.half]:
            settings = halves.ReadSettings(
                probe_types=frozenset(board_targets), gen3_probes=not args.no_gen3_probes,
                sweeps=not args.no_sweeps and half == halves.REPORTING,
                limit=args.probe_records, per_game=args.probe_per_game)
            progress(f"reading {stratum}/{half}: "
                     f"{len(selection.halves[stratum].get(half, ()))} games")
            read = halves.read_half(context, selection, stratum, half, settings,
                                    acting_text=acting_text, keywords_of=keywords_of,
                                    observe=observe)
            progress(f"  {len(read.records)} records kept, {len(read.probe_items)} with "
                     f"probe items, {len(read.grids)} grids, {len(read.rule_tests)} rules")
            reads.append(read)
    if not any(read.records for read in reads):
        raise SystemExit("no record the selection set names was found in the corpus's "
                         "validation strata")
    return reads


def probe(args, context: common.ProbeContext, selection) -> Scorecard:
    """Run every selected probe over the selection set's halves, the line
    ladders, the sweeps and the ablations; return the scorecard."""
    import torch

    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    families = set(args.families or labels.FAMILIES)
    targets = [spec for spec in labels.TARGETS if spec.family in families]
    board_targets = {spec.name for spec in targets if spec.scope in ("act", "entity")}
    kinds = {spec.name: spec.kind for spec in labels.TARGETS}

    games = ProbeGames()
    sidecars = context.sidecars()
    surface = common.manifest_surface(context.manifest)
    outcomes = ObservedOutcomes(common.acting_text_function(sidecars, surface))
    reads = _read_halves(args, context, selection, board_targets, games, outcomes)
    board_subsets = _board_subsets(args, context)
    every = [r for read in reads for r in read.records.values()]
    every += [r for subsets in board_subsets.values() for rows in subsets.values() for r in rows]
    progress(f"loading {args.checkpoint}")
    probe_model = context.model(
        every, vocab_path=args.vocab_path, keyword_path=args.keyword_definitions)

    scorecard = Scorecard(
        checkpoint=str(args.checkpoint), selection_set_digest=selection.digest,
        corpus_digest=selection.corpus_digest,
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        e_dim=probe_model.e_dim, surface=probe_model.surface,
        join_rate=None,
        families={family: {"targets": {}} for family in sorted(families)},
    )
    act_hidden = _board_ladder(args, scorecard, probe_model, reads, kinds)
    rule_outputs = {rule: tuple(entry.outputs)
                    for rule, entry in context.manifest.rule_map.items()}
    for read in reads:
        body = scorecard.half(read.stratum, read.half)
        body["games"] = read.games
        progress(f"{read.name}: {len(read.grids)} grids, {len(read.rule_tests)} rule tests")
        body["grids"] = halves.score_grids(probe_model, selection, read)
        body["rule_tests"] = rule_tests.score(probe_model, read.rule_tests, rule_outputs)
        if not args.no_gen3_probes:
            body.update(halves.score_gen3_probes(probe_model, read))

    progress("line-level probes over the ability cache")
    _observe_training(args, context, outcomes)
    text_by_key: dict = {}
    cache = (common.line_items(context.cards_folders, probe_model.surface, args.abilities_root,
                               text_by_key=text_by_key), text_by_key)
    held_out_cards = set(context.checkpoint.provenance.held_out_cards)
    line_labels = label_lines(cache[0], outcomes, sidecars)
    for name, entry in line_ladder(line_labels, probe_model, cache, targets, act_hidden,
                                   held_out_cards).items():
        scorecard.put(labels.TARGETS_BY_NAME[name].family, name, entry)
    pairs, joined = interaction_pairs(games, cache[0])
    scorecard.join_rate = joined.join_rate
    if "interactions" in families:
        scorecard.put(labels.TARGETS_BY_NAME["fires_pair"].family, "fires_pair",
                      pair_ladder(pairs, probe_model, cache))

    if not args.no_sweeps:
        progress("board sweeps")
        scorecard.sweeps = _run_sweeps(reads, probe_model, context.checkpoint)
    _run_ablations(args, context, scorecard, probe_model, cache[0], board_subsets)

    scorecard.runtime = {
        "wall_seconds": time.time() - started,
        "peak_gpu_bytes": common.peak_gpu_bytes(),
    }
    return scorecard


# ── tables ──────────────────────────────────────────────────────────────


def _half_tables(scorecard: Scorecard) -> list[str]:
    """Per stratum and half: the probe types, grids, rule tests and gen-3 probes."""
    fmt = common.fmt
    lines = ["# Per stratum and half", ""]
    probes, grids, rules, gen3 = [], [], [], []
    for stratum, by_half in sorted(scorecard.halves.items()):
        for half, body in sorted(by_half.items()):
            where = f"{stratum}/{half}"
            probes += [[where, name, item["kind"], item["n"], fmt(item.get("score")),
                        fmt(item.get("rescaled"))]
                       for name, item in sorted(body.get("probes", {}).items())]
            grids += [[where, name, item["records"], fmt(item["mae"]), fmt(item["baseline"]),
                       fmt(item["score"])]
                      for name, item in sorted(body.get("grids", {}).items())]
            tests = body.get("rule_tests") or {}
            rules.append([where, len(tests.get("rules", {})), len(tests.get("left_out", {})),
                          fmt(tests.get("component"))])
            conditions = body.get("conditions") or {}
            gen3 += [[where, f"conditions: {subset}", cell["records"], fmt(cell["auc"])]
                     for subset, cell in conditions.items() if cell["items"]]
            if body.get("anthem"):
                gen3.append([where, "anthem", body["anthem"]["records"],
                             fmt(body["anthem"]["auc"])])
            legality = body.get("target_legality")
            if legality:
                gen3 += [[where, f"target legality: {name}", legality["records"], fmt(value)]
                         for name, value in legality["auc"].items()]
    lines += ["## Probe types, rung 3", "",
              md_table(["half", "probe type", "kind", "n", "AUC or R²", "rescaled"], probes,
                       "lllrrr"), "",
              "## Grids", "",
              md_table(["half", "grid", "records", "MAE", "MAE of the grid mean", "score"],
                       grids, "llrrrr"), "",
              "## Rule tests", "",
              md_table(["half", "rules scored", "rules left out", "mean 2·AUC − 1"], rules,
                       "lrrr"), ""]
    if gen3:
        lines += ["## Gen-3 probes", "",
                  md_table(["half", "probe", "records", "AUC"], gen3, "llrr"), ""]
    return lines


def write_tables(out_dir: Path, scorecard: Scorecard) -> None:
    """One markdown table per family, the halves, the sweeps and the ablations."""
    fmt = common.fmt
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, body in scorecard.families.items():
        rungs, shares = [], []
        for target, entry in sorted(body["targets"].items()):
            for stratum, result in sorted(entry["strata"].items()):
                rungs += [[target, stratum, result["n"], rung, fmt(scores.get("linear")),
                           fmt(scores.get("mlp")), fmt(scores.get("model"))]
                          for rung, scores in result["rungs"].items()]
                for probe_type, value in (result.get("share") or {}).items():
                    if value:
                        ci = value.get("ci") or [None, None]
                        shares.append([target, stratum, probe_type, fmt(value["value"]),
                                       fmt(ci[0]), fmt(ci[1])])
        lines = [f"# {family}", "", "## Rungs", "",
                 md_table(["target", "stratum", "n", "rung", "linear", "mlp", "model"], rungs,
                          "llrlrrr")]
        if shares:
            lines += ["", "## Share of the gap closed by e", "",
                      md_table(["target", "stratum", "probe", "share", "95% low", "95% high"],
                               shares, "lllrrr")]
        (out_dir / f"{family}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.halves:
        (out_dir / "halves.md").write_text("\n".join(_half_tables(scorecard)) + "\n",
                                           encoding="utf-8")
    if scorecard.sweeps:
        lines = ["# Board sweeps", ""]
        for name, body in scorecard.sweeps.items():
            if "unavailable" in body:
                lines += [f"## {name}", "", f"Not run: {body['unavailable']}.", ""]
                continue
            columns = ["wording", "value", "mean prediction", "n"]
            rows = [[p.get("wording", ""), p["value"], fmt(p["prediction"]), p["n"]]
                    for p in body["mean"]]
            lines += [f"## {name}", "", md_table(columns, rows, "lrrr"), ""]
        (out_dir / "sweeps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.ablation:
        header = [f"{r} {s}" for r in ablation.REPLACEMENTS for s in ablation.SCOPES]
        rows = [[name, fmt(body["base_loss"]),
                 *(fmt(body[r][s]) for r in ablation.REPLACEMENTS for s in ablation.SCOPES)]
                for name, body in sorted(scorecard.ablation.items())]
        lines = ["# Ablation: loss increase per field", "",
                 md_table(["field", "base", *header], rows, "lr" + "r" * len(header))]
        (out_dir / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.board_abilities:
        (out_dir / "board-abilities.md").write_text(
            board_probe.table(scorecard.board_abilities), encoding="utf-8")


def report_dir(reports_dir: Path, checkpoint: Path, date: str) -> Path:
    label = common.checkpoint_label(checkpoint)
    return Path(reports_dir) / f"knowledge-probes-{label}-{date.replace('-', '')}"


def freeze_selection(args: argparse.Namespace) -> int:
    """Freeze the corpus's selection set with no checkpoint (research E1): 0
    once written, 1 when a half or a component misses its target."""
    from effect_knowledge_probes import selection
    from effects.application import freeze_selection_set
    from effects.application.selection_deviance import TargetDevianceLosses, scoring_targets
    from effects.domain.damage_step_keywords import KeywordResolver
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.shard_io import read_shard

    corpus = Path(args.corpus)
    manifest = CorpusStore(corpus).load()
    cards_folders = tuple(args.cards_folder or common.DEFAULT_CARDS_FOLDERS)
    variant_scripts = common.rooted(manifest.variant_scripts) if manifest.variant_scripts else None
    sidecars = common.sidecar_cache(cards_folders, variant_scripts)
    surface = common.manifest_surface(manifest)
    keywords_of = KeywordResolver(sidecars).keywords_of

    def deviance(manifest):
        sample = []
        for stratum in common.STRATA:
            path = corpus / "validation" / "samples" / f"{stratum}.jsonl.gz"
            if path.exists():
                sample += [r for _, r in zip(range(256), read_shard(path))]
        prepare = scoring_targets(corpus, manifest, cards_folders=cards_folders,
                                  sample=sample)
        return TargetDevianceLosses(prepare, {
            rule: tuple(entry.outputs) for rule, entry in manifest.rule_map.items()})

    progress(f"freezing the selection set of {corpus}")
    return freeze_selection_set.run(
        corpus, reports_dir=args.probe_set_dir or args.reports_dir, lines=sidecars,
        probes=selection.ProbeItems(sidecars, common.acting_text_function(sidecars, surface)),
        grids=selection.grid_sources(sidecars, keywords_of), deviance=deviance,
        echo=progress,
    )


def main(argv: list[str] | None = None) -> int:
    from effects.domain.shard_generation import (
        MixedGenerationsError,
        UnacceptedGenerationError,
    )

    args = parse_args(argv)
    if args.freeze_probe_set:
        return freeze_selection(args)
    context = common.probe_context(
        args.checkpoint, corpus=args.corpus, cards_folders=args.cards_folder)
    selection = halves.load_selection(context, args.probe_set_dir or args.reports_dir)
    try:
        scorecard = probe(args, context, selection)
    except (MixedGenerationsError, UnacceptedGenerationError) as error:
        raise SystemExit(str(error)) from None
    out_dir = report_dir(args.reports_dir, args.checkpoint, scorecard.date)
    common.write_json(out_dir / "scorecard.json", scorecard.to_json())
    write_tables(out_dir, scorecard)
    peak = scorecard.runtime["peak_gpu_bytes"] / 2**30
    print(f"scorecard -> {out_dir / 'scorecard.json'} "
          f"({scorecard.runtime['wall_seconds']:.0f} s, peak GPU {peak:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
