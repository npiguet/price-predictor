"""Rule ids and units: which unit a rule belongs to and the order both sort in."""

import re

SPLIT_SECTIONS = {"701", "702"}


def unit_of(section, number):
    return f"{section}.{number}" if section in SPLIT_SECTIONS else section


def unit_sort_key(unit):
    head, _, tail = unit.partition(".")
    return (int(head), int(tail) if tail else -1)


def rule_sort_key(rule_id):
    m = re.match(r"(\d{3})\.(\d+)([a-z]*)", rule_id)
    letters = m.group(3)
    return (int(m.group(1)), int(m.group(2)), len(letters), letters)


def rule_unit(rule_id):
    section, _, rest = rule_id.partition(".")
    return unit_of(section, re.match(r"\d+", rest).group(0))
