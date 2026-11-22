"""Reading the catalogue: the rules, the per-unit sidecars, Forge's card names and scripts."""

import json
from collections import Counter, defaultdict

from .locations import CARDSFOLDER, RULES
from .ordering import unit_sort_key


def load_rules():
    with open(RULES / "rules.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def units_with_sizes(rules):
    sizes, counts = Counter(), Counter()
    for r in rules:
        sizes[r["unit"]] += len(r["text"]) + sum(len(e) for e in r["examples"])
        counts[r["unit"]] += 1
    return sizes, counts


NAMES_CACHE = RULES / ".forge-card-names.txt"  # gitignored; delete it after updating Forge


def card_names():
    if NAMES_CACHE.exists():
        return set(NAMES_CACHE.read_text(encoding="utf-8").splitlines())
    names = set()
    for p in CARDSFOLDER.rglob("*.txt"):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Name:"):
                    names.add(line[5:].strip())
    NAMES_CACHE.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8", newline="\n")
    return names


SCRIPTS_CACHE = RULES / ".forge-card-scripts.tsv"  # gitignored; delete it after updating Forge


def card_scripts():
    """``card name -> the lines of its Forge script``, None for a name no script declares.

    Every ``Name:`` line of a script names it, so a face of a multi-faced card
    finds the card's file. The name-to-file index is cached beside the names.
    """
    if SCRIPTS_CACHE.exists():
        index = dict(line.split("\t", 1)
                     for line in SCRIPTS_CACHE.read_text(encoding="utf-8").splitlines() if line)
    else:
        index = {}
        for p in sorted(CARDSFOLDER.rglob("*.txt")):
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("Name:"):
                        index.setdefault(line[5:].strip(), p.relative_to(CARDSFOLDER).as_posix())
        SCRIPTS_CACHE.write_text("".join(f"{n}\t{f}\n" for n, f in sorted(index.items())),
                                 encoding="utf-8", newline="\n")

    def script(name):
        relative = index.get(name)
        if relative is None:
            return None
        return (CARDSFOLDER / relative).read_text(encoding="utf-8", errors="replace").splitlines()

    return script


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path.name}:{n}: {e}") from e
    return out


#: Each per-unit file a unit may have, by its directory under rules/.
UNIT_FILES = {"triage": ".jsonl", "scenarios": ".json", "heads": ".jsonl"}


def load_unit(kind, unit):
    """A unit's triage, scenarios or heads file, parsed; None while it is not written."""
    p = RULES / kind / f"{unit}{UNIT_FILES[kind]}"
    if not p.exists():
        return None
    return read_jsonl(p) if p.suffix == ".jsonl" else json.loads(p.read_text(encoding="utf-8"))


def load_scenarios_by_rule(rules_dir):
    """``rule id -> [scenario, …]`` over every scenarios file under ``rules_dir``."""
    by_rule = defaultdict(list)
    paths = sorted((rules_dir / "scenarios").glob("*.json"), key=lambda p: unit_sort_key(p.stem))
    for path in paths:
        for scenario in json.loads(path.read_text(encoding="utf-8")).get("scenarios", []):
            for rule in scenario.get("rules", []):
                by_rule[rule].append(scenario)
    return by_rule
