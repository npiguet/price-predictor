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
* optionally every trunk layer (``--per-layer``) and a shallow trunk trained on
  frozen ``e`` (``--method-c``).

Probe records come from the corpus's own validation strata, which hold whole
games, rather than from the raw corpus: ``--records-dir`` overrides that.

Nothing here does work at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def progress(message: str) -> None:
    """One timestamped line, flushed: a probing run takes the better part of an
    hour, and without these nothing says whether it is alive or where it is."""
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def sibling(name: str):
    """A module of this suite, loaded by path under a name of its own.

    By path because ``scripts/`` is not a package, and under
    ``effect_knowledge_probes.<name>`` because the embedding probes beside it
    have a ``common`` module of their own.
    """
    qualified = f"effect_knowledge_probes.{name}"
    if qualified not in sys.modules:
        spec = importlib.util.spec_from_file_location(qualified, _HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules[qualified]


#: Probe games per validation stratum, by smallest ``crc32`` of the game id.
GAMES_PER_STRATUM = 300
#: Board-dependent items per target per stratum, by smallest ``crc32``.
ITEMS_PER_TARGET = 3000
#: Training shards read for the observed-outcome line labels.
PROFILE_SHARDS = 4
#: Validation-sample records per stratum the ablation reads.
ABLATION_RECORDS = 1024
METHOD_C_STEPS = 2000


def canonical_digest(payload: dict) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace), ``digest`` excluded."""
    body = {k: v for k, v in payload.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def probe_set_path(reports_dir: Path, corpus_digest: str) -> Path:
    return Path(reports_dir) / f"knowledge-probes-set-{corpus_digest[:12]}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    common = sibling("common")
    labels = sibling("labels")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path,
                        help="curated corpus; default the checkpoint's recorded one")
    parser.add_argument("--records-dir", type=Path,
                        help="where the probe games' shards are; default the "
                             "corpus's validation strata")
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="sidecar tree (repeatable); default output/cardsfolder "
                             "and output/tokenscripts")
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
    parser.add_argument("--reports-dir", type=Path, default=common.DEFAULT_REPORTS,
                        help="where the probe set and the report directory go")
    parser.add_argument("--games-per-stratum", type=int, default=GAMES_PER_STRATUM)
    parser.add_argument("--items-per-target", type=int, default=ITEMS_PER_TARGET)
    parser.add_argument("--profile-shards", type=int, default=PROFILE_SHARDS)
    parser.add_argument("--ablation-records", type=int, default=ABLATION_RECORDS)
    args = parser.parse_args(argv)
    args.cards_folder = args.cards_folder or list(common.DEFAULT_CARDS_FOLDERS)
    unknown = set(args.families or ()) - set(labels.FAMILIES)
    if unknown:
        parser.error(f"unknown families: {sorted(unknown)}")
    return args


# ── freezing the probe set ──────────────────────────────────────────────


def _key_json(key) -> dict | None:
    return None if key is None else key.as_dict()


def _key_from(data: dict | None):
    from effects.domain.provenance import ProvenanceKey

    return None if data is None else ProvenanceKey.from_dict(data)


def _acting_text_function(sidecars, surface: str):
    from effects.domain.ability_encoder import encoding_text

    memo: dict = {}

    def acting_text(key):
        if key not in memo:
            try:
                line = sidecars.line_for(key)
            except KeyError:
                line = None
            memo[key] = (None if line is None
                         else encoding_text(line, sidecars.prose_for(key), surface))
        return memo[key]

    return acting_text


def _game_shards(args, corpus: Path, stratum: str) -> list[Path]:
    common = sibling("common")
    if args.records_dir is not None:
        return common.shards_of(args.records_dir)
    return common.shards_of(common.stratum_dir(corpus, stratum))


def freeze(args, checkpoint, corpus: Path, manifest, corpus_digest: str) -> dict:
    """Enumerate and label every probe item once for this corpus."""
    from effects.domain.records import Moment, RecordKind, TriggerPayload
    from effects.infrastructure.record_io import iter_shards, read_shard
    from effects.infrastructure.sidecar_io import converted_text_path

    common, labels, sweeps = sibling("common"), sibling("labels"), sibling("sweeps")
    surface = getattr(manifest, "surface", None) or "script"
    sidecars = common.sidecar_cache(args.cards_folder)
    acting_text = _acting_text_function(sidecars, surface)

    strata_games = {
        "card-disjoint": list(manifest.card_disjoint_games),
        "game-disjoint": list(manifest.game_disjoint_games),
    }
    games, sources = {}, {}
    record_items: list[dict] = []
    join = labels.TriggerJoin()
    selector = sweeps.SweepSelector(sidecars)
    profiles: dict[str, list[np.ndarray]] = defaultdict(list)
    mana: dict[str, list[float]] = defaultdict(list)
    #: trigger record id -> (trigger key, fired), for the interaction pairs.
    triggers: dict[str, tuple] = {}

    def observe(record) -> None:
        if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
            text = acting_text(record.ability[0]) if record.ability else None
            if text:
                profile = labels.effect_profile(record)
                if profile is not None:
                    profiles[text].append(profile)
                produced = labels.mana_observed(record)
                if produced is not None:
                    mana[text].append(produced)

    for stratum, ids in strata_games.items():
        chosen = sorted(ids, key=common.stable_hash)[: args.games_per_stratum]
        games[stratum] = sorted(chosen)
        used: set[str] = set()
        per_target: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        # One streaming pass: nothing keeps a record past this loop body. The
        # join's partners are filled in once every game has been read.
        empty_join = labels.JoinResult(cause={}, fired=0, fired_joined=0)
        for shard, record in common.read_games(
                _game_shards(args, corpus, stratum), set(chosen)):
            used.add(shard.relative_to(corpus).as_posix()
                     if shard.is_relative_to(corpus) else str(shard))
            observe(record)
            join.observe(record)
            selector.observe(record)
            if isinstance(record.payload, TriggerPayload) and record.ability:
                triggers[record.record_id] = (record.ability[0], record.payload.fired)
            for item in labels.record_items(record, sidecars, empty_join, acting_text):
                order = common.stable_hash(f"{item.record_id}|{item.entity or ''}")
                per_target[item.target].append((order, {
                    "record_id": item.record_id, "stratum": stratum,
                    "family": labels.TARGETS_BY_NAME[item.target].family,
                    "target": item.target, "label": item.label,
                    "entity": item.entity, "partner": None,
                    "partner_entity": item.partner_entity, "group": item.group,
                }))
        sources[stratum] = sorted(used)
        for rows in per_target.values():
            rows.sort(key=lambda pair: pair[0])
            record_items.extend(row for _, row in rows[: args.items_per_target])

    joined = join.result()
    for row in record_items:
        if row["target"] == "fires":
            row["partner"] = _key_json(joined.cause.get(row["record_id"]))
    sweep_set = selector.result()

    training = sorted(iter_shards(Path(corpus) / "training"))[: args.profile_shards]
    for shard in training:
        for record in read_shard(shard):
            observe(record)
    spread = labels.state_dependence(profiles)

    items_out = []
    line_items = common.line_items(args.cards_folder, surface)
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
        if mana.get(item.text):
            found["observed_mana"] = float(np.mean(mana[item.text]))
        if found:
            entry = {"key": item.key.as_dict(), "labels": found}
            if weights:
                entry["weights"] = weights
            items_out.append(entry)

    pair_labels: dict[tuple, float] = {}
    for record_id, (trigger_key, fired) in triggers.items():
        cause = joined.cause.get(record_id)
        if cause is None:
            continue
        pair = (json.dumps(trigger_key.as_dict(), sort_keys=True),
                json.dumps(cause.as_dict(), sort_keys=True))
        pair_labels[pair] = max(pair_labels.get(pair, 0.0), float(fired))
    pairs = [{"a": json.loads(a), "b": json.loads(b), "label": label, "source": "join"}
             for (a, b), label in sorted(pair_labels.items())]
    # Mined pairs are positives by construction and far outnumber the joined
    # ones, so they are capped at the joined count: uncapped, a probe could score
    # by telling a mined pair from a joined one rather than reading the pair.
    mined = []
    for a, b in labels.mined_pairs(line_items):
        signature = (json.dumps(a.as_dict(), sort_keys=True),
                     json.dumps(b.as_dict(), sort_keys=True))
        if signature not in pair_labels:
            mined.append((common.stable_hash("|".join(signature)), a, b))
    for _, a, b in sorted(mined, key=lambda row: row[0])[: len(pairs)]:
        pairs.append({"a": a.as_dict(), "b": b.as_dict(), "label": 1.0,
                      "source": "mined"})

    payload = {
        "corpus": str(corpus),
        "corpus_digest": corpus_digest,
        "surface": surface,
        "games": games,
        "sources": sources,
        "items": items_out,
        "pairs": pairs,
        "record_items": record_items,
        "sweeps": sweep_set,
        "join_rate": joined.join_rate,
        "join": {"fired": joined.fired, "fired_joined": joined.fired_joined},
        "settings": {
            "games_per_stratum": args.games_per_stratum,
            "items_per_target": args.items_per_target,
            "profile_shards": args.profile_shards,
            "cards_folders": [str(f) for f in args.cards_folder],
        },
    }
    payload["digest"] = canonical_digest(payload)
    return payload


def load_probe_set(path: Path, corpus_digest: str) -> dict:
    """The frozen set for this corpus; refuses a missing or mismatched one."""
    if not path.exists():
        raise SystemExit(
            f"no frozen probe set at {path}: run with --freeze-probe-set first "
            "(once per curated corpus)")
    probe_set = json.loads(path.read_text(encoding="utf-8"))
    if probe_set.get("corpus_digest") != corpus_digest:
        raise SystemExit(
            f"{path} was frozen against corpus digest "
            f"{probe_set.get('corpus_digest')}, not this corpus's {corpus_digest}")
    if probe_set.get("digest") != canonical_digest(probe_set):
        raise SystemExit(f"{path} does not match its own digest; it was edited")
    return probe_set


# ── probing ─────────────────────────────────────────────────────────────


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


def line_ladder(probe_set, probe_model, cache, targets, act_hidden, held_out_cards,
                mlp_options, bootstrap) -> dict:
    """The short ladder for every line target: ``e``, its width control, ``[ACT]``.

    ``cache`` is ``(items, text_by_key)`` from ``common.line_items`` over the
    checkpoint's cache: every line it encodes, read once per run.
    """
    from effects.domain.provenance import ProvenanceKey

    ladder = sibling("ladder")
    items, text_by_key = cache
    by_text = {item.text: item for item in items}
    e_dim = probe_model.e_dim
    act_mean = {text: np.mean(rows, axis=0) for text, rows in act_hidden.items()}
    results: dict[str, dict] = {}
    for spec in targets:
        if spec.scope != "line":
            continue
        rows = []
        for entry in probe_set["items"]:
            label = entry["labels"].get(spec.name)
            if label is None:
                continue
            text = text_by_key.get(ProvenanceKey.from_dict(entry["key"]))
            item = by_text.get(text)
            if item is None or item.e is None:
                continue
            weight = entry.get("weights", {}).get(spec.name, 1.0)
            split = "held-out" if item.carriers & held_out_cards else "trained"
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


def pair_ladder(probe_set, probe_model, cache, mlp_options, bootstrap) -> dict:
    """Interaction pairs: a trigger line with its cause, read from both ``e``."""
    from effects.domain.provenance import ProvenanceKey

    ladder = sibling("ladder")
    items, text_by_key = cache
    by_text = {item.text: item for item in items}
    rows = []
    for pair in probe_set["pairs"]:
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

    from effects.domain.effect_model import (
        EffectModel,
        entity_target_tensors,
        per_entity_loss,
    )
    from effects.domain.effect_targets import derive_targets
    from effects.infrastructure.record_io import iter_shards, read_shard

    ablation = sibling("ablation")
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
            batch, surfaces = batcher.build(chunk, encoder)
        outputs = shallow.per_entity(shallow(**batch))
        gate, targets, mask, index = entity_target_tensors(
            surfaces, [derive_targets(r) for r in chunk], fields)
        device = outputs.device
        gathered = outputs.gather(
            1, index.to(device).unsqueeze(-1).expand(-1, -1, outputs.shape[-1]))
        loss, _ = per_entity_loss(gathered, gate.to(device),
                                  {k: v.to(device) for k, v in targets.items()},
                                  mask.to(device), fields=fields)
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


def probe(args, checkpoint, corpus: Path, probe_set: dict) -> dict:
    """Run every selected family, the sweeps and the ablation; return the scorecard."""
    import torch

    from effects.infrastructure.record_io import read_shard

    common, labels, ladder = sibling("common"), sibling("labels"), sibling("ladder")
    sweeps, ablation = sibling("sweeps"), sibling("ablation")
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    families = set(args.families or labels.FAMILIES)
    targets = [spec for spec in labels.TARGETS if spec.family in families]
    board = {spec.name for spec in targets if spec.scope in ("act", "entity")}
    kinds = {spec.name: spec.kind for spec in labels.TARGETS}

    wanted: dict[str, set[str]] = defaultdict(set)
    items_by_stratum: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in probe_set["record_items"]:
        if row["target"] not in board:
            continue
        wanted[row["stratum"]].add(row["record_id"])
        items_by_stratum[row["stratum"]][row["record_id"]].append(labels.RecordItem(
            record_id=row["record_id"], target=row["target"], label=row["label"],
            entity=row["entity"], partner=_key_from(row["partner"]),
            partner_entity=row["partner_entity"], group=row["group"],
        ))
    sweep_ids = set()
    if not args.no_sweeps:
        sweep_ids = {r["record_id"] for rows in probe_set["sweeps"].values() for r in rows}

    def shards_for(stratum):
        names = probe_set["sources"].get(stratum, [])
        return [Path(n) if Path(n).is_absolute() else Path(corpus) / n for n in names]

    records_by_stratum: dict[str, list] = {}
    for stratum, game_ids in probe_set["games"].items():
        progress(f"reading {len(game_ids)} {stratum} probe games")
        records_by_stratum[stratum] = [
            record for _, record in common.read_games(
                shards_for(stratum), set(game_ids), wanted[stratum] | sweep_ids)
        ]
        progress(f"  {len(records_by_stratum[stratum])} records")
    every = [r for rows in records_by_stratum.values() for r in rows]
    if not every:
        raise SystemExit("no probe records were found in the probe set's sources")
    progress(f"loading {args.checkpoint}")
    probe_model = common.load_probe_model(
        args.checkpoint, args.cards_folder, every, checkpoint=checkpoint,
        vocab_path=args.vocab_path, keyword_path=args.keyword_definitions)
    mlp_options = {}
    bootstrap = ladder.BOOTSTRAP_RESAMPLES

    scorecard: dict = {
        "checkpoint": str(args.checkpoint),
        "probe_set_digest": probe_set["digest"],
        "corpus_digest": probe_set["corpus_digest"],
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "e_dim": probe_model.e_dim,
        "surface": probe_model.surface,
        "join_rate": probe_set.get("join_rate"),
        "families": {family: {"targets": {}} for family in sorted(families)},
    }

    def put(name: str, entry: dict) -> None:
        spec = labels.TARGETS_BY_NAME[name]
        scorecard["families"][spec.family]["targets"][name] = entry

    act_hidden: dict[str, list] = {}
    for stratum, records in records_by_stratum.items():
        items = items_by_stratum[stratum]
        chosen = [r for r in records if r.record_id in items]
        progress(f"{stratum}: extracting rung features over {len(chosen)} records")
        extracted = ladder.extract_features(
            probe_model, chosen, items, per_layer=args.per_layer, act_hidden=act_hidden)
        inputs = ladder.ladder_inputs(extracted, kinds)
        for position, (name, data) in enumerate(inputs.items(), start=1):
            if len(data.y) < 20:
                continue
            progress(f"{stratum}: ladder {position}/{len(inputs)} {name} (n={len(data.y)})")
            spec = labels.TARGETS_BY_NAME[name]
            entry = scorecard["families"][spec.family]["targets"].setdefault(
                name, {"scope": spec.scope, "kind": spec.kind, "strata": {}})
            entry["strata"][stratum] = ladder.run_ladder(
                data, mlp_options=mlp_options, bootstrap=bootstrap)
            if data.layers:
                entry["strata"][stratum]["per_layer"] = entry["strata"][stratum].pop(
                    "per_layer", [])

    progress("line-level probes over the ability cache")
    text_by_key: dict = {}
    cache = (common.line_items(args.cards_folder, probe_model.surface, args.abilities_root,
                               text_by_key=text_by_key), text_by_key)
    held_out_cards = set(checkpoint.provenance.held_out_cards)
    for name, entry in line_ladder(probe_set, probe_model, cache, targets, act_hidden,
                                   held_out_cards, mlp_options, bootstrap).items():
        put(name, entry)
    if "interactions" in families:
        put("fires_pair", pair_ladder(probe_set, probe_model, cache, mlp_options, bootstrap))

    if not args.no_sweeps:
        progress("board sweeps")
        by_id = {r.record_id: r for r in every}
        scorecard["sweeps"] = {}
        for name, rows in probe_set["sweeps"].items():
            curves: dict[str, list] = defaultdict(list)
            for row in rows:
                record = by_id.get(row["record_id"])
                if record is None:
                    continue
                label = f"damage_{row['host']}" if name == "damage" else name
                curves[label].append(sweeps.run_sweep(name, probe_model, record, row))
            for label, curve in curves.items():
                scorecard["sweeps"][label] = {
                    "mean": sweeps.summarize(curve), "records": curve,
                }

    samples = []
    for stratum in common.STRATA:
        path = Path(corpus) / "validation" / "samples" / f"{stratum}.jsonl.gz"
        if path.exists():
            for count, record in enumerate(read_shard(path)):
                if count >= args.ablation_records:
                    break
                samples.append(record)
    if not args.no_ablation and samples:
        progress(f"ablation over {len(samples)} records")
        stats = ablation.cache_statistics(cache[0])
        scorecard["ablation"] = ablation.run_ablation(probe_model, samples, stats)
    if args.method_c and samples:
        progress("method C: a shallow trunk on frozen e")
        scorecard["method_c"] = method_c(args, probe_model, corpus, samples)

    scorecard["runtime"] = {
        "wall_seconds": time.time() - started,
        "peak_gpu_bytes": common.peak_gpu_bytes(),
    }
    return scorecard


# ── tables ──────────────────────────────────────────────────────────────


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def write_tables(out_dir: Path, scorecard: dict) -> None:
    """One markdown table per family, plus the sweeps and the ablation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for family, body in scorecard["families"].items():
        lines = [f"# {family}", "", "## Rungs", "",
                 "| target | stratum | n | rung | linear | mlp | model |",
                 "|---|---|---:|---|---:|---:|---:|"]
        shares = []
        for target, entry in sorted(body["targets"].items()):
            for stratum, result in sorted(entry["strata"].items()):
                for rung, scores in result["rungs"].items():
                    lines.append(
                        f"| {target} | {stratum} | {result['n']} | {rung} | "
                        f"{_fmt(scores.get('linear'))} | {_fmt(scores.get('mlp'))} | "
                        f"{_fmt(scores.get('model'))} |")
                for probe_type, value in (result.get("share") or {}).items():
                    if value:
                        ci = value.get("ci") or [None, None]
                        shares.append(f"| {target} | {stratum} | {probe_type} | "
                                      f"{_fmt(value['value'])} | {_fmt(ci[0])} | "
                                      f"{_fmt(ci[1])} |")
        if shares:
            lines += ["", "## Share of the gap closed by e", "",
                      "| target | stratum | probe | share | 95% low | 95% high |",
                      "|---|---|---|---:|---:|---:|", *shares]
        (out_dir / f"{family}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.get("sweeps"):
        lines = ["# Board sweeps", ""]
        for name, body in scorecard["sweeps"].items():
            lines += [f"## {name}", "", "| value | mean prediction | n |", "|---:|---:|---:|"]
            lines += [f"| {p['value']} | {_fmt(p['prediction'])} | {p['n']} |"
                      for p in body["mean"]]
            lines.append("")
        (out_dir / "sweeps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if scorecard.get("ablation"):
        abl = sibling("ablation")
        header = [f"{r} {s}" for r in abl.REPLACEMENTS for s in abl.SCOPES]
        lines = ["# Ablation: loss increase per field", "",
                 "| field | base | " + " | ".join(header) + " |",
                 "|---|---:|" + "---:|" * len(header)]
        for name, body in sorted(scorecard["ablation"].items()):
            cells = [_fmt(body[r][s]) for r in abl.REPLACEMENTS for s in abl.SCOPES]
            lines.append(f"| {name} | {_fmt(body['base_loss'])} | " + " | ".join(cells) + " |")
        (out_dir / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_dir(reports_dir: Path, checkpoint: Path, date: str) -> Path:
    stem = sibling("common").checkpoint_stem(checkpoint)
    return Path(reports_dir) / f"knowledge-probes-{stem}-{date.replace('-', '')}"


def main(argv: list[str] | None = None) -> int:
    from effects.infrastructure.corpus_store import CorpusStore

    args = parse_args(argv)
    common = sibling("common")
    checkpoint = common.load_checkpoint(args.checkpoint)
    corpus = args.corpus or (Path(checkpoint.provenance.corpus_path)
                             if checkpoint.provenance.corpus_path else None)
    if corpus is None or not (Path(corpus) / "manifest.json").exists():
        raise SystemExit("no curated corpus: pass --corpus, or probe a checkpoint "
                         "trained with --corpus")
    manifest = CorpusStore(Path(corpus)).load()
    corpus_digest = manifest.digest()
    set_path = probe_set_path(args.reports_dir, corpus_digest)
    if args.freeze_probe_set:
        payload = freeze(args, checkpoint, Path(corpus), manifest, corpus_digest)
        common.write_json(set_path, payload)
        print(f"froze {len(payload['items'])} line items, "
              f"{len(payload['record_items'])} record items, "
              f"{len(payload['pairs'])} pairs; join rate "
              f"{payload['join_rate']:.3f}; digest {payload['digest'][:12]} -> {set_path}")
        return 0
    probe_set = load_probe_set(set_path, corpus_digest)
    scorecard = probe(args, checkpoint, Path(corpus), probe_set)
    out_dir = report_dir(args.reports_dir, args.checkpoint, scorecard["date"])
    common.write_json(out_dir / "scorecard.json", scorecard)
    write_tables(out_dir, scorecard)
    runtime = scorecard["runtime"]
    peak = runtime["peak_gpu_bytes"] / 2**30
    print(f"scorecard -> {out_dir / 'scorecard.json'} "
          f"({runtime['wall_seconds']:.0f} s, peak GPU {peak:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
