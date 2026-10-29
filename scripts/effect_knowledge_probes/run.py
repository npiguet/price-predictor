"""Probe one checkpoint's game knowledge, and how much of it lives in ``e``.

    # once per curated corpus
    python scripts/effect_knowledge_probes/run.py --checkpoint PATH --freeze-probe-set
    # then per checkpoint trained on that corpus
    python scripts/effect_knowledge_probes/run.py --checkpoint PATH

``--freeze-probe-set`` enumerates the probe items keyed by provenance, their
labels, the probe games of both validation strata, the board-sweep records and
the interaction join rate, and writes them with a sha256 digest (FR-076,
FR-078). A probing run refuses without a frozen set for its corpus, reads the
frozen set, and writes one scorecard recording its digest (FR-088):

* the read-out ladder over the ten families (``ladder``), per stratum for the
  board-dependent targets and per held-out/trained split for line properties;
* the four board sweeps (``sweeps``);
* the ablation of ``e`` (``ablation``);
* the same ablation on records where a permanent's unlisted static, trigger or
  replacement decides the answer, beside controls (``board``);
* optionally every trunk layer (``--per-layer``) and a shallow trunk trained on
  frozen ``e`` (``--method-c``).

Probe records come from the corpus's own validation strata, which hold whole
games, rather than from the raw corpus: ``--records-dir`` overrides that.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_knowledge_probes import (  # noqa: E402
    ablation,
    common,
    labels,
    ladder,
    sweeps,
)
from effect_knowledge_probes import board as board_probe  # noqa: E402
from effect_knowledge_probes.probe_set import (  # noqa: E402
    FrozenLineItem,
    FrozenRecordItem,
    ProbeSet,
    load_probe_set,
    probe_set_path,
)
from effect_knowledge_probes.scorecard import Scorecard  # noqa: E402
from effects.domain.ability_tokenizer import Surface  # noqa: E402


def progress(message: str) -> None:
    """One timestamped line, flushed: a probing run takes the better part of an
    hour, and without these nothing says whether it is alive or where it is."""
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


#: Probe games per validation stratum, by smallest ``crc32`` of the game id.
GAMES_PER_STRATUM = 300
#: Board-dependent items per target per stratum, by smallest ``crc32``.
ITEMS_PER_TARGET = 3000
#: Training shards read for the observed-outcome line labels.
PROFILE_SHARDS = 4
#: Validation-sample records per stratum the ablation reads.
ABLATION_RECORDS = 1024
METHOD_C_STEPS = 2000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path,
                        help="curated corpus; default the checkpoint's recorded one")
    parser.add_argument("--records-dir", type=Path,
                        help="where the probe games' shards are; default the "
                             "corpus's validation strata")
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="sidecar tree (repeatable); default the trees the "
                             "checkpoint recorded")
    parser.add_argument("--abilities-root", type=Path, default=common.DEFAULT_ABILITIES)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--keyword-definitions", type=Path)
    parser.add_argument("--freeze-probe-set", action="store_true")
    parser.add_argument("--families", type=lambda s: [f.strip() for f in s.split(",") if f],
                        help=f"comma list from: {', '.join(labels.FAMILIES)}")
    parser.add_argument("--per-layer", action="store_true")
    parser.add_argument("--method-c", action="store_true")
    parser.add_argument("--method-c-layers", type=int, choices=(0, 1), default=1)
    parser.add_argument("--method-c-steps", type=int, default=METHOD_C_STEPS)
    parser.add_argument("--no-sweeps", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-board", action="store_true",
                        help="skip the board-abilities ablation (board.py)")
    parser.add_argument("--reports-dir", type=Path, default=common.DEFAULT_REPORTS,
                        help="where the report directory goes, and the probe set "
                             "unless --probe-set-dir is given")
    parser.add_argument("--probe-set-dir", type=Path, default=None,
                        help="where the frozen probe set is written and read "
                             "(default: --reports-dir); every arm of a sweep "
                             "shares one set while its reports go beside its run")
    parser.add_argument("--games-per-stratum", type=int, default=GAMES_PER_STRATUM)
    parser.add_argument("--items-per-target", type=int, default=ITEMS_PER_TARGET)
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
    return args


# ── freezing the probe set ──────────────────────────────────────────────


def _acting_text_function(sidecars, surface: Surface):
    from effects.domain.ability_encoder import encoding_text

    memo: dict = {}

    def acting_text(key):
        if key not in memo:
            line = sidecars.line_or_none(key)
            memo[key] = (None if line is None
                         else encoding_text(line, sidecars.prose_for(key), surface))
        return memo[key]

    return acting_text


def _game_shards(args, context: common.ProbeContext, stratum: str) -> list[Path]:
    if args.records_dir is not None:
        return common.shards_of(args.records_dir)
    return common.shards_of(context.stratum_dir(stratum))


class ObservedOutcomes:
    """What each text was seen to do: its effect profiles and the mana it made.

    Observed one record at a time and kept as per-text lists of small arrays,
    so the probe games and the training shards feed one set of labels without
    either being held.
    """

    def __init__(self, acting_text) -> None:
        self._acting_text = acting_text
        self.profiles: dict[str, list[np.ndarray]] = defaultdict(list)
        self.mana: dict[str, list[float]] = defaultdict(list)

    def observe(self, record) -> None:
        from effects.domain.records import Moment, RecordKind

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
class ProbeGameScan:
    """What one pass over the probe games of both strata yields."""

    games: dict[str, list[str]]
    sources: dict[str, list[str]]
    record_items: list[FrozenRecordItem]
    joined: labels.JoinResult
    sweeps: dict
    #: ``trigger record id -> (trigger key, fired)``, for the interaction pairs.
    triggers: dict[str, tuple]


def _scan_probe_games(args, context: common.ProbeContext, acting_text,
                      outcomes: ObservedOutcomes) -> ProbeGameScan:
    """Choose each stratum's probe games and label their record items, streamed.

    One pass per stratum, and nothing keeps a record past its loop body: the
    trigger join, the sweep selector and ``outcomes`` take what they need as
    each record goes by. A trigger item's partner is filled in once every game
    has been read, since its cause can be any resolution of the game.
    """
    from effects.domain.records import TriggerPayload

    corpus, sidecars = context.corpus, context.sidecars()
    strata_games = {
        "card-disjoint": list(context.manifest.card_disjoint_games),
        "game-disjoint": list(context.manifest.game_disjoint_games),
    }
    games, sources = {}, {}
    record_items: list[FrozenRecordItem] = []
    join = labels.TriggerJoin()
    selector = sweeps.SweepSelector(sidecars)
    triggers: dict[str, tuple] = {}
    empty_join = labels.JoinResult(cause={}, fired=0, fired_joined=0)
    for stratum, ids in strata_games.items():
        chosen = sorted(ids, key=common.stable_hash)[: args.games_per_stratum]
        games[stratum] = sorted(chosen)
        used: set[str] = set()
        per_target: dict[str, list[tuple[int, FrozenRecordItem]]] = defaultdict(list)
        for shard, record in common.read_games(
                _game_shards(args, context, stratum), set(chosen)):
            used.add(shard.relative_to(corpus).as_posix()
                     if shard.is_relative_to(corpus) else str(shard))
            outcomes.observe(record)
            join.observe(record)
            selector.observe(record)
            if isinstance(record.payload, TriggerPayload) and record.ability:
                triggers[record.record_id] = (record.ability[0], record.payload.fired)
            for item in labels.record_items(record, sidecars, empty_join, acting_text):
                order = common.stable_hash(f"{item.record_id}|{item.entity or ''}")
                per_target[item.target].append((order, FrozenRecordItem(
                    record_id=item.record_id, stratum=stratum,
                    family=labels.TARGETS_BY_NAME[item.target].family,
                    target=item.target, label=item.label, entity=item.entity,
                    partner=None, partner_entity=item.partner_entity, group=item.group,
                )))
        sources[stratum] = sorted(used)
        for rows in per_target.values():
            rows.sort(key=lambda pair: pair[0])
            record_items.extend(row for _, row in rows[: args.items_per_target])

    joined = join.result()
    record_items = [
        replace(item, partner=_key_json(joined.cause.get(item.record_id)))
        if item.target == "fires" else item
        for item in record_items
    ]
    return ProbeGameScan(games, sources, record_items, joined, selector.result(), triggers)


def _key_json(key) -> dict | None:
    return None if key is None else key.as_dict()


def _observe_training(args, context: common.ProbeContext, outcomes: ObservedOutcomes) -> None:
    """The first ``--profile-shards`` training shards, for the observed-outcome labels."""
    from effects.infrastructure.record_io import iter_shards, read_shard

    for shard in sorted(iter_shards(context.training_dir))[: args.profile_shards]:
        for record in read_shard(shard):
            outcomes.observe(record)


def _label_lines(context: common.ProbeContext, line_items, surface: Surface,
                 outcomes: ObservedOutcomes) -> list[FrozenLineItem]:
    """Every encoded line's script labels, plus what its records were seen to do."""
    from effects.infrastructure.sidecar_io import converted_text_path

    sidecars = context.sidecars()
    spread = labels.state_dependence(outcomes.profiles)
    frozen = []
    for item in line_items:
        try:
            path = sidecars.path_for(item.script_file)
        except KeyError:
            path = None
        types = common.card_types(converted_text_path(path)) if path else ""
        found = labels.line_labels(item.script_text, item.line_kind, item.api, types)
        weights = {}
        if item.text in spread:
            found["spread"], n = spread[item.text]
            weights["spread"] = labels.state_weight(n)
        if outcomes.mana.get(item.text):
            found["observed_mana"] = float(np.mean(outcomes.mana[item.text]))
        if found:
            frozen.append(FrozenLineItem(item.key.as_dict(), found, weights))
    return frozen


def _interaction_pairs(scan: ProbeGameScan, line_items) -> list[dict]:
    """Trigger lines with their causes: joined from the games, then mined from scripts.

    Mined pairs are positives by construction and far outnumber the joined
    ones, so they are capped at the joined count: uncapped, a probe could score
    by telling a mined pair from a joined one rather than reading the pair.
    """
    pair_labels: dict[tuple, float] = {}
    for record_id, (trigger_key, fired) in scan.triggers.items():
        cause = scan.joined.cause.get(record_id)
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
    return pairs


def freeze(args, context: common.ProbeContext) -> ProbeSet:
    """Enumerate and label every probe item once for this corpus."""
    surface = getattr(context.manifest, "surface", None) or Surface.SCRIPT
    acting_text = _acting_text_function(context.sidecars(), surface)
    outcomes = ObservedOutcomes(acting_text)
    scan = _scan_probe_games(args, context, acting_text, outcomes)
    _observe_training(args, context, outcomes)
    line_items = common.line_items(context.cards_folders, surface)
    return ProbeSet(
        corpus=str(context.corpus), corpus_digest=context.corpus_digest,
        surface=surface, games=scan.games, sources=scan.sources,
        items=_label_lines(context, line_items, surface, outcomes),
        pairs=_interaction_pairs(scan, line_items),
        record_items=scan.record_items, sweeps=scan.sweeps,
        join_rate=scan.joined.join_rate,
        join={"fired": scan.joined.fired, "fired_joined": scan.joined.fired_joined},
        settings={
            "games_per_stratum": args.games_per_stratum,
            "items_per_target": args.items_per_target,
            "profile_shards": args.profile_shards,
            "cards_folders": [str(f) for f in context.cards_folders],
        },
    ).sealed()


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


def line_ladder(probe_set: ProbeSet, probe_model, cache, targets, act_hidden,
                held_out_cards, mlp_options, bootstrap) -> dict:
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
        for entry in probe_set.items:
            label = entry.labels.get(spec.name)
            if label is None:
                continue
            text = text_by_key.get(entry.provenance_key())
            item = by_text.get(text)
            if item is None or item.e is None:
                continue
            weight = entry.weights.get(spec.name, 1.0)
            split = line_split(item.carriers, held_out_cards)
            rows.append((text, item.e, label, weight, split))
        strata = {}
        for split in ("held-out", "trained"):
            chosen = [r for r in rows if r[4] == split]
            if len(chosen) < 20:
                continue
            texts = [r[0] for r in chosen]
            data = ladder.LadderInput(
                kind=spec.kind, y=np.array([r[2] for r in chosen], dtype=np.float64),
                groups=np.array(texts, dtype=object),
                features={"1": np.stack([r[1] for r in chosen]),
                          "1w": np.stack([ladder.width_vector(t, e_dim) for t in texts])},
                weight=np.array([r[3] for r in chosen], dtype=np.float64),
            )
            result = ladder.run_ladder(data, rungs=("1", "1w"), mlp_options=mlp_options,
                                       bootstrap=bootstrap)
            acted = [i for i, t in enumerate(texts) if t in act_mean]
            if len(acted) >= 20:
                subset = ladder.LadderInput(
                    kind=spec.kind, y=data.y[acted], groups=data.groups[acted],
                    features={"1": data.features["1"][acted],
                              "2": np.stack([act_mean[texts[i]] for i in acted])},
                    weight=data.weight[acted],
                )
                sub = ladder.run_ladder(subset, rungs=("1", "2"), mlp_options=mlp_options,
                                        bootstrap=bootstrap)
                result["rungs"]["2"] = sub["rungs"]["2"]
                result["rungs"]["1-acting"] = sub["rungs"]["1"]
                result["n_acting"] = sub["n"]
            strata[split] = result
        results[spec.name] = {"scope": "line", "kind": spec.kind, "strata": strata}
    return results


def pair_ladder(probe_set: ProbeSet, probe_model, cache, mlp_options, bootstrap) -> dict:
    """Interaction pairs: a trigger line with its cause, read from both ``e``."""
    from effects.domain.provenance import ProvenanceKey

    items, text_by_key = cache
    by_text = {item.text: item for item in items}
    rows = []
    for pair in probe_set.pairs:
        a = by_text.get(text_by_key.get(ProvenanceKey.from_dict(pair["a"])))
        b = by_text.get(text_by_key.get(ProvenanceKey.from_dict(pair["b"])))
        if a is None or b is None or a.e is None or b.e is None:
            continue
        rows.append((a, b, pair["label"]))
    if len(rows) < 20:
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
    result = ladder.run_ladder(data, rungs=("1", "1w"), mlp_options=mlp_options,
                               bootstrap=bootstrap)
    result["grouping"] = grouping
    return {"scope": "pair", "kind": "binary", "strata": {"all": result}}


# ── probing: method C ───────────────────────────────────────────────────


class _NoTrunk:
    """Method C's zero-layer trunk: the slots reach the heads as embedded."""

    def __new__(cls):
        from torch import nn

        class NoTrunk(nn.Module):
            def forward(self, src, src_key_padding_mask=None):  # noqa: ARG002
                return src

        return NoTrunk()


def method_c(args, probe_model, corpus: Path, eval_records: list) -> dict:
    """Train a shallow trunk on frozen ``e`` and score it beside the full model."""
    import torch

    from effects.domain.effect_losses import per_entity_loss
    from effects.domain.effect_model import EffectModel
    from effects.domain.surface_collation import per_entity_targets
    from effects.infrastructure.record_io import iter_shards, read_shard

    layers = args.method_c_layers
    config = replace(probe_model.checkpoint.model_config, n_layers=max(layers, 1))
    shallow = EffectModel(config).to(probe_model.device)
    if layers == 0:
        shallow.trunk = _NoTrunk()
    optimizer = torch.optim.AdamW(shallow.parameters(), lr=3e-4, weight_decay=1e-4)
    batcher, encoder, fields = probe_model.batcher, probe_model.encoder, probe_model.fields

    def stream():
        while True:
            for shard in sorted(iter_shards(Path(corpus) / "training")):
                yield from read_shard(shard)

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
        loss, _ = per_entity_loss(gathered, targets.fields_on(device), fields=fields)
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


@dataclass
class ProbeRecords:
    """The records a probing run reads, by stratum, and its board items by record."""

    by_stratum: dict[str, list]
    items: dict[str, dict[str, list]]
    #: ``stratum -> {subset: records}`` for the board-abilities ablation.
    board_subsets: dict[str, dict[str, list]]

    def every(self) -> list:
        """Every record a forward pass will see, so the batcher measures them all."""
        found = [r for rows in self.by_stratum.values() for r in rows]
        for subsets in self.board_subsets.values():
            found += [r for rows in subsets.values() for r in rows]
        return found


def _read_probe_records(args, context: common.ProbeContext, probe_set: ProbeSet,
                        board_targets: set[str]) -> ProbeRecords:
    """The frozen items' records and the sweeps', then the board subsets."""
    wanted: dict[str, set[str]] = defaultdict(set)
    items: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for frozen in probe_set.record_items:
        if frozen.target not in board_targets:
            continue
        wanted[frozen.stratum].add(frozen.record_id)
        items[frozen.stratum][frozen.record_id].append(frozen.record_item())
    sweep_ids = set()
    if not args.no_sweeps:
        sweep_ids = {r["record_id"] for rows in probe_set.sweeps.values() for r in rows}

    by_stratum: dict[str, list] = {}
    for stratum, game_ids in probe_set.games.items():
        progress(f"reading {len(game_ids)} {stratum} probe games")
        by_stratum[stratum] = [
            record for _, record in common.read_games(
                context.source_shards(probe_set.sources.get(stratum, [])),
                set(game_ids), wanted[stratum] | sweep_ids)
        ]
        progress(f"  {len(by_stratum[stratum])} records")
    if not any(by_stratum.values()):
        raise SystemExit("no probe records were found in the probe set's sources")
    return ProbeRecords(by_stratum, items, _board_subsets(args, context))


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
            board_probe.stratum_records(
                directory, shards=args.board_shards or board_probe.SHARDS_PER_STRATUM),
            context.sidecars(),
            per_subset=args.board_records or board_probe.RECORDS_PER_SUBSET)
        progress("  " + ", ".join(
            f"{name} {len(rows)}" for name, rows in chosen[stratum].items()))
    return chosen


def _board_ladder(args, scorecard: Scorecard, probe_model, records: ProbeRecords,
                  kinds: dict[str, str], mlp_options, bootstrap) -> dict[str, list]:
    """Every board-dependent target's ladder, per stratum; returns ``[ACT]`` by text.

    The trunk's ``[ACT]`` output is gathered from the same forward pass, by
    acting text, for the line ladder's rung 2.
    """
    act_hidden: dict[str, list] = {}
    for stratum, stratum_records in records.by_stratum.items():
        items = records.items[stratum]
        chosen = [r for r in stratum_records if r.record_id in items]
        progress(f"{stratum}: extracting rung features over {len(chosen)} records")
        extracted = ladder.extract_features(
            probe_model, chosen, items, per_layer=args.per_layer, act_hidden=act_hidden)
        inputs = ladder.ladder_inputs(extracted, kinds)
        for position, (name, data) in enumerate(inputs.items(), start=1):
            if len(data.y) < 20:
                continue
            progress(f"{stratum}: ladder {position}/{len(inputs)} {name} (n={len(data.y)})")
            spec = labels.TARGETS_BY_NAME[name]
            entry = scorecard.target(spec.family, name, scope=spec.scope, kind=spec.kind)
            entry["strata"][stratum] = ladder.run_ladder(
                data, mlp_options=mlp_options, bootstrap=bootstrap)
            if data.layers:
                entry["strata"][stratum]["per_layer"] = entry["strata"][stratum].pop(
                    "per_layer", [])
    return act_hidden


def _run_sweeps(probe_set: ProbeSet, probe_model, records: list, checkpoint) -> dict:
    """Every board sweep over its frozen records: per-record curves and their mean."""
    by_id = {r.record_id: r for r in records}
    out: dict = {}
    verdict_trained = ladder.verdict_head_trained(checkpoint)
    for name, rows in probe_set.sweeps.items():
        if name == "affordability" and not verdict_trained:
            # It reads the verdict head, which this checkpoint never trained: a
            # curve from random weights says nothing.
            out[name] = {"unavailable": ladder.UNTRAINED_VERDICT_NOTE}
            continue
        curves: dict[str, list] = defaultdict(list)
        for row in rows:
            record = by_id.get(row["record_id"])
            if record is None:
                continue
            label = f"damage_{row['host']}" if name == "damage" else name
            curves[label].append(sweeps.run_sweep(name, probe_model, record, row))
        for label, curve in curves.items():
            out[label] = {"mean": sweeps.summarize(curve), "records": curve}
    return out


def _validation_samples(context: common.ProbeContext, limit: int) -> list:
    """The first ``limit`` records of each stratum's fixed validation sample."""
    from effects.infrastructure.record_io import read_shard

    samples = []
    for stratum in common.STRATA:
        path = Path(context.corpus) / "validation" / "samples" / f"{stratum}.jsonl.gz"
        if path.exists():
            for count, record in enumerate(read_shard(path)):
                if count >= limit:
                    break
                samples.append(record)
    return samples


def _run_ablations(args, context: common.ProbeContext, scorecard: Scorecard, probe_model,
                   line_cache, board_subsets) -> None:
    """Method B over the validation samples and the board subsets, and method C."""
    samples = _validation_samples(context, args.ablation_records)
    stats = None
    if not args.no_ablation and samples:
        progress(f"ablation over {len(samples)} records")
        stats = ablation.cache_statistics(line_cache)
        scorecard.ablation = ablation.run_ablation(probe_model, samples, stats)
    if board_subsets:
        progress("board-abilities ablation")
        stats = stats or ablation.cache_statistics(line_cache)
        scorecard.board_abilities = board_probe.run(probe_model, board_subsets, stats)
    if args.method_c and samples:
        progress("method C: a shallow trunk on frozen e")
        scorecard.method_c = method_c(args, probe_model, context.corpus, samples)


def probe(args, context: common.ProbeContext, probe_set: ProbeSet) -> Scorecard:
    """Run every selected family, the sweeps and the ablation; return the scorecard."""
    import torch

    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    families = set(args.families or labels.FAMILIES)
    targets = [spec for spec in labels.TARGETS if spec.family in families]
    board_targets = {spec.name for spec in targets if spec.scope in ("act", "entity")}
    kinds = {spec.name: spec.kind for spec in labels.TARGETS}

    records = _read_probe_records(args, context, probe_set, board_targets)
    every = records.every()
    progress(f"loading {args.checkpoint}")
    probe_model = context.model(
        every, vocab_path=args.vocab_path, keyword_path=args.keyword_definitions)
    mlp_options = {}
    bootstrap = ladder.BOOTSTRAP_RESAMPLES

    scorecard = Scorecard(
        checkpoint=str(args.checkpoint), probe_set_digest=probe_set.digest,
        corpus_digest=probe_set.corpus_digest,
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        e_dim=probe_model.e_dim, surface=probe_model.surface,
        join_rate=probe_set.join_rate,
        families={family: {"targets": {}} for family in sorted(families)},
    )
    act_hidden = _board_ladder(args, scorecard, probe_model, records, kinds,
                               mlp_options, bootstrap)

    progress("line-level probes over the ability cache")
    text_by_key: dict = {}
    cache = (common.line_items(context.cards_folders, probe_model.surface, args.abilities_root,
                               text_by_key=text_by_key), text_by_key)
    held_out_cards = set(context.checkpoint.provenance.held_out_cards)
    for name, entry in line_ladder(probe_set, probe_model, cache, targets, act_hidden,
                                   held_out_cards, mlp_options, bootstrap).items():
        scorecard.put(labels.TARGETS_BY_NAME[name].family, name, entry)
    if "interactions" in families:
        scorecard.put(labels.TARGETS_BY_NAME["fires_pair"].family, "fires_pair",
                      pair_ladder(probe_set, probe_model, cache, mlp_options, bootstrap))

    if not args.no_sweeps:
        progress("board sweeps")
        scorecard.sweeps = _run_sweeps(probe_set, probe_model, every, context.checkpoint)
    _run_ablations(args, context, scorecard, probe_model, cache[0], records.board_subsets)

    scorecard.runtime = {
        "wall_seconds": time.time() - started,
        "peak_gpu_bytes": common.peak_gpu_bytes(),
    }
    return scorecard


# ── tables ──────────────────────────────────────────────────────────────


def write_tables(out_dir: Path, scorecard: Scorecard) -> None:
    """One markdown table per family, plus the sweeps and the ablation."""
    fmt = common.fmt
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, body in scorecard.families.items():
        lines = [f"# {family}", "", "## Rungs", "",
                 "| target | stratum | n | rung | linear | mlp | model |",
                 "|---|---|---:|---|---:|---:|---:|"]
        shares = []
        for target, entry in sorted(body["targets"].items()):
            for stratum, result in sorted(entry["strata"].items()):
                for rung, scores in result["rungs"].items():
                    lines.append(
                        f"| {target} | {stratum} | {result['n']} | {rung} | "
                        f"{fmt(scores.get('linear'))} | {fmt(scores.get('mlp'))} | "
                        f"{fmt(scores.get('model'))} |")
                for probe_type, value in (result.get("share") or {}).items():
                    if value:
                        ci = value.get("ci") or [None, None]
                        shares.append(f"| {target} | {stratum} | {probe_type} | "
                                      f"{fmt(value['value'])} | {fmt(ci[0])} | "
                                      f"{fmt(ci[1])} |")
        if shares:
            lines += ["", "## Share of the gap closed by e", "",
                      "| target | stratum | probe | share | 95% low | 95% high |",
                      "|---|---|---|---:|---:|---:|", *shares]
        (out_dir / f"{family}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.sweeps:
        lines = ["# Board sweeps", ""]
        for name, body in scorecard.sweeps.items():
            if "unavailable" in body:
                lines += [f"## {name}", "", f"Not run: {body['unavailable']}.", ""]
                continue
            lines += [f"## {name}", "", "| value | mean prediction | n |", "|---:|---:|---:|"]
            lines += [f"| {p['value']} | {fmt(p['prediction'])} | {p['n']} |"
                      for p in body["mean"]]
            lines.append("")
        (out_dir / "sweeps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.ablation:
        header = [f"{r} {s}" for r in ablation.REPLACEMENTS for s in ablation.SCOPES]
        lines = ["# Ablation: loss increase per field", "",
                 "| field | base | " + " | ".join(header) + " |",
                 "|---|---:|" + "---:|" * len(header)]
        for name, body in sorted(scorecard.ablation.items()):
            cells = [fmt(body[r][s]) for r in ablation.REPLACEMENTS for s in ablation.SCOPES]
            lines.append(f"| {name} | {fmt(body['base_loss'])} | " + " | ".join(cells) + " |")
        (out_dir / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.board_abilities:
        (out_dir / "board-abilities.md").write_text(
            board_probe.table(scorecard.board_abilities), encoding="utf-8")


def probe_set_location(args: argparse.Namespace, corpus_digest: str) -> Path:
    """The shared probe set's path: ``--probe-set-dir``, else ``--reports-dir``."""
    return probe_set_path(args.probe_set_dir or args.reports_dir, corpus_digest)


def report_dir(reports_dir: Path, checkpoint: Path, date: str) -> Path:
    stem = common.checkpoint_stem(checkpoint)
    return Path(reports_dir) / f"knowledge-probes-{stem}-{date.replace('-', '')}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = common.probe_context(
        args.checkpoint, corpus=args.corpus, cards_folders=args.cards_folder)
    set_path = probe_set_location(args, context.corpus_digest)
    if args.freeze_probe_set:
        frozen = freeze(args, context)
        common.write_json(set_path, frozen.to_json())
        print(f"froze {len(frozen.items)} line items, "
              f"{len(frozen.record_items)} record items, "
              f"{len(frozen.pairs)} pairs; join rate "
              f"{frozen.join_rate:.3f}; digest {frozen.digest[:12]} -> {set_path}")
        return 0
    probe_set = load_probe_set(set_path, context.corpus_digest)
    scorecard = probe(args, context, probe_set)
    out_dir = report_dir(args.reports_dir, args.checkpoint, scorecard.date)
    common.write_json(out_dir / "scorecard.json", scorecard.to_json())
    write_tables(out_dir, scorecard)
    peak = scorecard.runtime["peak_gpu_bytes"] / 2**30
    print(f"scorecard -> {out_dir / 'scorecard.json'} "
          f"({scorecard.runtime['wall_seconds']:.0f} s, peak GPU {peak:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
