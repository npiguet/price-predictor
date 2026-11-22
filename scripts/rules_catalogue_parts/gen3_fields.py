"""``gen3-fields``: each heads line's gen-3 ids, the learnable list and GEN3-FIELDS.md."""

import json
import re
import sys

from .gen3_lines import gen3_line, learnable_today, turn_action_verdict_errors
from .gen3_registry import GEN3_FIELDS_NAME, GEN3_KEYS, LEARNABLE, gen3_registry
from .gen3_summary import write_gen3_summary
from .ordering import rule_sort_key, unit_sort_key
from .reading import load_scenarios_by_rule, read_jsonl


def write_learnable(path, learnable):
    """Set the registry's ``learnable`` member to ``learnable``, leaving every
    other byte of the hand-kept file as it is."""
    text = path.read_text(encoding="utf-8")
    value = json.dumps(learnable)
    member = re.compile(rf'^(  "{LEARNABLE}": )\[[^\]]*\]', re.M)
    if member.search(text):
        text = member.sub(lambda m: m.group(1) + value, text)
    else:
        body = text.rstrip()
        if not body.endswith("}"):
            raise ValueError(f"{path} does not end with its closing brace")
        text = f'{body[:-1].rstrip()},\n  "{LEARNABLE}": {value}\n}}\n'
    if json.loads(text)[LEARNABLE] != learnable:
        raise ValueError(f"{path}: could not write its {LEARNABLE} member")
    path.write_text(text, encoding="utf-8", newline="\n")


def print_learnable_change(previous, current):
    """The rules the change made learnable, and any it made unlearnable, against
    the list the previous run recorded."""
    if previous is None:
        print(f"no previous {LEARNABLE} list: recording {len(current)} rules as the baseline")
        return
    gained = sorted(set(current) - set(previous), key=rule_sort_key)
    lost = sorted(set(previous) - set(current), key=rule_sort_key)
    for rule in gained:
        print(f"NEWLY LEARNABLE {rule}")
    for rule in lost:
        print(f"NO LONGER LEARNABLE {rule}")
    print(f"{len(gained)} rules newly learnable, {len(lost)} no longer, "
          f"against the previous {GEN3_FIELDS_NAME}")


def _heads_files(rules_dir):
    return [(path, read_jsonl(path)) for path in
            sorted((rules_dir / "heads").glob("*.jsonl"), key=lambda p: unit_sort_key(p.stem))]


def _refuse_unrecorded(heads, rules_dir):
    """Stop, writing nothing, while a turn_action-only rule's verdict is not ``recorded``."""
    errors = turn_action_verdict_errors([row for _, rows in heads for row in rows],
                                        load_scenarios_by_rule(rules_dir))
    for error in errors:
        print(f"VERDICT {error}")
    if errors:
        print(f"{len(errors)} turn_action-only rules not marked recorded; nothing written")
        sys.exit(1)


def _update_row(row, registry):
    """Replace the row's gen-3 keys; the count of names nothing maps (each printed)."""
    fields, unmapped = gen3_line(row, *registry)
    for name in unmapped:
        print(f"UNMAPPED {row['id']}: {name}")
    for k in GEN3_KEYS:
        row.pop(k, None)
    row.update(fields)
    return len(unmapped)


def _write_heads(heads, registry):
    """Rewrite each heads file with its lines' gen-3 fields: ``(all rows, unmapped count)``."""
    all_rows, n_unmapped = [], 0
    for path, rows in heads:
        for row in rows:
            n_unmapped += _update_row(row, registry)
            all_rows.append(row)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
    return all_rows, n_unmapped


def cmd_gen3_fields(args):
    rules_dir = args.rules_dir
    registry_path = rules_dir / GEN3_FIELDS_NAME
    reg, *registry = gen3_registry(registry_path)
    heads = _heads_files(rules_dir)
    _refuse_unrecorded(heads, rules_dir)
    all_rows, n_unmapped = _write_heads(heads, registry)
    if n_unmapped:
        print(f"{n_unmapped} unmapped names")
        sys.exit(1)
    write_gen3_summary(reg, registry[0], all_rows, rules_dir)
    learnable = sorted((r["id"] for r in all_rows if learnable_today(r)), key=rule_sort_key)
    print_learnable_change(reg.get(LEARNABLE), learnable)
    write_learnable(registry_path, learnable)
