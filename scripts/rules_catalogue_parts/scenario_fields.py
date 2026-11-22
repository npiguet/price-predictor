"""``validate``: the gen-3 fields of one scenario, its detector, operations and departure."""

from .vocabulary import (
    DEPARTURE_RESOLUTION,
    DEPARTURE_STATUS,
    FORK_OPERATIONS,
)


def _names_a_card(value):
    """Whether a detector parameter's value names one card (a ``Card.namedX`` selector)."""
    return isinstance(value, str) and "named" in value.lower()


def _blank(value):
    return not str(value).strip()


def _same_as_names(detector):
    same_as = detector["same_as"]
    return same_as if isinstance(same_as, list) else [same_as]


def _same_as_errors(sid, detector):
    if set(detector) != {"same_as", "reason"} or _blank(detector["reason"]):
        return [f"{sid}: a same_as detector names its scenario and the fact the two "
                "share (reason), and nothing else"]
    if sid in _same_as_names(detector):
        return [f"{sid}: a scenario is no duplicate of itself"]
    return []


def _none_errors(sid, detector):
    if len(detector) != 1 or _blank(detector["none"]):
        return [f"{sid}: a none detector carries its reason and nothing else"]
    return []


def _family_errors(sid, detector, families):
    family = detector.get("family")
    if family not in families:
        return [f"{sid}: unknown detector family {family!r}"]
    undeclared = set(detector.get("params") or {}) - families[family]
    if undeclared:
        return [f"{sid}: {family} declares no parameter {sorted(undeclared)}"]
    return []


def _declared_family_errors(s, detector, families):
    sid = s.get("id")
    errs = _family_errors(sid, detector, families)
    if detector.get("record_kind") != s.get("record_kind"):
        errs.append(f"{sid}: detector record_kind {detector.get('record_kind')!r} is not "
                    f"the scenario's {s.get('record_kind')!r}")
    errs += [f"{sid}: detector parameter {name} names a card ({value!r}); "
             "describe the rule's condition, not the scenario's board"
             for name, value in (detector.get("params") or {}).items() if _names_a_card(value)]
    errs += [f"{sid}: detector has no {key}"
             for key in ("outcome", "near_miss") if _blank(detector.get(key) or "")]
    return errs


def _detector_errors(s, families):
    sid, detector = s.get("id"), s.get("detector")
    if not isinstance(detector, dict) or not detector:
        return [f"{sid}: no detector (declare one, or {{\"none\": reason}})"]
    if "none" in detector:
        return _none_errors(sid, detector)
    if "same_as" in detector:
        return _same_as_errors(sid, detector)
    return _declared_family_errors(s, detector, families)


def _base_errors(sid, variants):
    if not variants:
        return []
    base = variants[0]
    errs = []
    if base.get("delta"):
        errs.append(f"{sid}: the first variant is the base and its delta is not empty")
    if "operation" in base:
        errs.append(f"{sid}: the base variant carries an operation")
    return errs


def _operation_errors(label, op):
    if op is None:
        return [f"{label}: no operation (one of {sorted(map(str, FORK_OPERATIONS))}, "
                "or {\"unclassifiable\": reason})"]
    if isinstance(op, dict):
        if set(op) != {"unclassifiable"} or _blank(op["unclassifiable"]):
            return [f"{label}: operation {op!r}"]
        return []
    if op not in FORK_OPERATIONS:
        return [f"{label}: unknown operation {op!r}"]
    return []


def _variant_errors(sid, variants):
    errs = _base_errors(sid, variants)
    for v in variants[1:]:
        errs += _operation_errors(f"{sid}/{v.get('label')}", v.get("operation"))
    return errs


def _departure_errors(sid, departure):
    if departure is None:
        return []
    errs = []
    status, resolution = departure.get("status"), departure.get("resolution")
    if status not in DEPARTURE_STATUS:
        errs.append(f"{sid}: departure status {status!r}")
    if resolution is not None and resolution not in DEPARTURE_RESOLUTION:
        errs.append(f"{sid}: departure resolution {resolution!r}")
    if resolution is not None and status != "confirmed":
        errs.append(f"{sid}: a departure has a resolution only once confirmed")
    return errs


def validate_gen3_fields(s, families):
    """The detector, operation and departure fields of one scenario (FR-002, FR-003, FR-005).

    ``families`` is the detector registry, ``family id -> declared params``.
    """
    sid = s.get("id")
    return [*_detector_errors(s, families),
            *_variant_errors(sid, s.get("variants", [])),
            *_departure_errors(sid, s.get("departure"))]
