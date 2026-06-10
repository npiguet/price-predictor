"""Live-pick message round-trip + validation (data-model §2.1–2.3, T005)."""

from __future__ import annotations

import json

import pytest

from draft.application.generate_draft_data import (
    ABANDONED_SENTINEL,
    PICK_REQUEST_SENTINEL,
    PICK_RESPONSE_SENTINEL,
    AbandonedNotice,
    PickRequest,
    PickResponse,
    parse_abandoned,
    parse_pick_request,
)


def _request_line(**overrides) -> str:
    payload = {
        "draft_id": "d1",
        "seat": 3,
        "agent": "draft-agent",
        "pod_size": 8,
        "pack_number": 1,
        "pick_number": 5,
        "set_code": "BLB",
        "pack": ["Card A", "Card B", "Card C"],
    }
    payload.update(overrides)
    return PICK_REQUEST_SENTINEL + json.dumps(payload)


def test_well_formed_request_parses() -> None:
    request = parse_pick_request(_request_line())
    assert request == PickRequest(
        draft_id="d1", seat=3, agent="draft-agent", pod_size=8,
        pack_number=1, pick_number=5, set_code="BLB",
        pack=["Card A", "Card B", "Card C"],
    )


def test_non_request_line_is_none() -> None:
    assert parse_pick_request("Forge noise") is None
    assert parse_pick_request('{"draft_id":"d1"}') is None  # no sentinel
    # Sentinel with non-dict / unparseable JSON suffix → None (not a request).
    assert parse_pick_request(PICK_REQUEST_SENTINEL + "[1,2,3]") is None
    assert parse_pick_request(PICK_REQUEST_SENTINEL + "{bad json") is None


def test_request_field_validation() -> None:
    with pytest.raises(ValueError):
        parse_pick_request(_request_line(pod_size=4))     # must be 8
    with pytest.raises(ValueError):
        parse_pick_request(_request_line(pack_number=0))  # 1..PACKS
    with pytest.raises(ValueError):
        parse_pick_request(_request_line(pack_number=4))  # > PACKS (3)
    with pytest.raises(ValueError):
        parse_pick_request(_request_line(pick_number=0))  # >= 1
    with pytest.raises(ValueError):
        parse_pick_request(_request_line(pack=[]))        # non-empty


def test_response_enforces_exactly_one_of_pick_or_abort() -> None:
    pick = PickResponse("d1", 3, 1, 5, pick="Card B")
    abort = PickResponse("d1", 3, 1, 5, abort=True)
    assert json.loads(pick.serialize()[len(PICK_RESPONSE_SENTINEL):]) == {
        "draft_id": "d1", "seat": 3, "pack_number": 1, "pick_number": 5,
        "pick": "Card B",
    }
    assert json.loads(abort.serialize()[len(PICK_RESPONSE_SENTINEL):]) == {
        "draft_id": "d1", "seat": 3, "pack_number": 1, "pick_number": 5,
        "abort": True,
    }
    with pytest.raises(ValueError):
        PickResponse("d1", 3, 1, 5)                      # neither
    with pytest.raises(ValueError):
        PickResponse("d1", 3, 1, 5, pick="X", abort=True)  # both


def test_abandoned_notice_parses() -> None:
    line = ABANDONED_SENTINEL + json.dumps(
        {"draft_id": "d1", "reason": "response mismatch at seat 3 pack 1 pick 5"}
    )
    notice = parse_abandoned(line)
    assert notice == AbandonedNotice(
        draft_id="d1", reason="response mismatch at seat 3 pack 1 pick 5"
    )
    assert parse_abandoned("not abandoned") is None
